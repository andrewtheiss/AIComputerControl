from __future__ import annotations

import re
import time
from typing import Any, Dict

import requests

from ui_vision_common.mock_backends import mock_grounding_result, mock_ocr_result


class ProviderError(RuntimeError):
    pass


class BaseProvider:
    def __init__(self, model_id: str, backend: str):
        self.model_id = model_id
        self.backend = backend

    def run_ocr(self, image_bgr, min_score: float) -> Dict[str, Any]:
        raise ProviderError(f"OCR is not supported by {self.model_id}")

    def run_grounding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise ProviderError(f"Grounding is not supported by {self.model_id}")


class MockProvider(BaseProvider):
    def __init__(self, model_id: str, role: str):
        super().__init__(model_id=model_id, backend="mock")
        self.role = role

    def run_ocr(self, image_bgr, min_score: float) -> Dict[str, Any]:
        return mock_ocr_result(self.model_id, image_bgr, min_score=min_score)

    def run_grounding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return mock_grounding_result(
            self.model_id,
            instruction=str(payload.get("instruction", "") or ""),
            candidates=list(payload.get("candidates") or []),
            history=list(payload.get("history") or []),
            top_k=int(payload.get("top_k") or 5),
        )


class HTTPProxyProvider(BaseProvider):
    def __init__(self, model_id: str, role: str, remote_url: str):
        if not remote_url:
            raise ProviderError("MODEL_BACKEND=http_proxy requires REMOTE_MODEL_URL")
        super().__init__(model_id=model_id, backend="http_proxy")
        self.role = role
        self.remote_url = remote_url.rstrip("/")
        self.session = requests.Session()

    def run_ocr(self, image_bgr, min_score: float) -> Dict[str, Any]:
        from ui_vision_common.image_tools import encode_png_b64

        t0 = time.perf_counter()
        payload = {
            "image_b64": encode_png_b64(image_bgr),
            "return_level": "both",
            "min_score": min_score,
        }
        resp = self.session.post(f"{self.remote_url}/ocr", json=payload, timeout=(3.0, 30.0))
        resp.raise_for_status()
        data = resp.json()
        data["latency_ms"] = data.get("latency_ms") or int((time.perf_counter() - t0) * 1000)
        return data

    def run_grounding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        resp = self.session.post(f"{self.remote_url}/infer", json=payload, timeout=(3.0, 30.0))
        resp.raise_for_status()
        data = resp.json()
        data["latency_ms"] = data.get("latency_ms") or int((time.perf_counter() - t0) * 1000)
        return data


def _runtime_url(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    want = suffix.rstrip("/")
    if base.endswith(want):
        return base
    if want.startswith("/v1/") and base.endswith("/v1"):
        return base + want[len("/v1") :]
    return f"{base}{suffix}"


def _normalized_bbox_to_box(bbox: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1 = max(0, min(width - 1, int(round(float(bbox[0]) * width))))
        y1 = max(0, min(height - 1, int(round(float(bbox[1]) * height))))
        x2 = max(x1 + 1, min(width, int(round(float(bbox[2]) * width))))
        y2 = max(y1 + 1, min(height, int(round(float(bbox[3]) * height))))
    except (TypeError, ValueError):
        return None
    return [x1, y1, x2, y2]


def _split_words(content: str, box: list[int]) -> list[dict[str, Any]]:
    from ui_vision_common.image_tools import box_to_poly

    tokens = re.findall(r"\S+", content)
    if not tokens:
        return []

    x1, y1, x2, y2 = box
    total_chars = sum(max(1, len(token)) for token in tokens) or len(tokens)
    cursor = x1
    width = max(1, x2 - x1)
    words: list[dict[str, Any]] = []
    for idx, token in enumerate(tokens):
        share = max(1, int(round(width * (max(1, len(token)) / total_chars))))
        if idx == len(tokens) - 1:
            next_x = x2
        else:
            next_x = min(x2 - (len(tokens) - idx - 1), cursor + share)
        word_box = [cursor, y1, max(cursor + 1, next_x), y2]
        words.append({"poly": box_to_poly(word_box), "text": token})
        cursor = next_x
    return words


def _loc_value_to_px(value: Any, size: int) -> int:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(size, int(round((raw / 1000.0) * size))))


def _box_from_poly(poly: list[list[float]]) -> list[int]:
    xs = [int(round(point[0])) for point in poly]
    ys = [int(round(point[1])) for point in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


_LOC_TOKEN_RE = re.compile(r"<\|LOC_(\d+)\|>")


def _parse_paddle_spotting_output(content: str, width: int, height: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in str(content or "").splitlines():
        loc_values = [int(match) for match in _LOC_TOKEN_RE.findall(raw_line)]
        if len(loc_values) != 8:
            continue
        text = _LOC_TOKEN_RE.sub("", raw_line).strip()
        if not text:
            continue
        poly = [
            [_loc_value_to_px(loc_values[0], width), _loc_value_to_px(loc_values[1], height)],
            [_loc_value_to_px(loc_values[2], width), _loc_value_to_px(loc_values[3], height)],
            [_loc_value_to_px(loc_values[4], width), _loc_value_to_px(loc_values[5], height)],
            [_loc_value_to_px(loc_values[6], width), _loc_value_to_px(loc_values[7], height)],
        ]
        rows.append({"text": text, "poly": poly, "box": _box_from_poly(poly)})
    return rows


class OmniParserRuntimeProvider(BaseProvider):
    def __init__(self, model_id: str, remote_url: str):
        if not remote_url:
            raise ProviderError("MODEL_BACKEND=omniparser_runtime requires REMOTE_MODEL_URL")
        super().__init__(model_id=model_id, backend="omniparser_runtime")
        self.remote_url = remote_url.rstrip("/")
        self.session = requests.Session()

    def run_ocr(self, image_bgr, min_score: float) -> Dict[str, Any]:
        from ui_vision_common.image_tools import box_to_poly, encode_png_b64

        t0 = time.perf_counter()
        height, width = image_bgr.shape[:2]
        payload = {"base64_image": encode_png_b64(image_bgr)}
        resp = self.session.post(_runtime_url(self.remote_url, "/parse/"), json=payload, timeout=(3.0, 120.0))
        resp.raise_for_status()
        data = resp.json()

        words: list[dict[str, Any]] = []
        lines: list[dict[str, Any]] = []
        for item in list(data.get("parsed_content_list") or []):
            content = str(item.get("content", "") or "").strip()
            if not content:
                continue
            box = _normalized_bbox_to_box(item.get("bbox"), width=width, height=height)
            if not box:
                continue
            item_type = str(item.get("type", "") or "")
            source = str(item.get("source", "") or "")
            score = 0.92 if source == "box_ocr_content_ocr" or item_type == "text" else 0.72
            if score < min_score:
                continue
            lines.append(
                {
                    "poly": box_to_poly(box),
                    "text": content,
                    "score": round(score, 4),
                    "source_model": self.model_id,
                }
            )
            for word in _split_words(content, box):
                word["score"] = round(score, 4)
                word["source_model"] = self.model_id
                words.append(word)

        return {
            "width": width,
            "height": height,
            "words": words,
            "lines": lines,
            "latency_ms": data.get("latency_ms") or int((time.perf_counter() - t0) * 1000),
            "meta": {
                "runtime_endpoint": _runtime_url(self.remote_url, "/parse/"),
                "parsed_content_count": len(list(data.get("parsed_content_list") or [])),
            },
        }


class PaddleOCRVLRuntimeProvider(BaseProvider):
    def __init__(self, model_id: str, remote_url: str):
        if not remote_url:
            raise ProviderError("MODEL_BACKEND=paddleocr_vl_runtime requires REMOTE_MODEL_URL")
        super().__init__(model_id=model_id, backend="paddleocr_vl_runtime")
        self.remote_url = remote_url.rstrip("/")
        self.session = requests.Session()
        self._served_model_name: str | None = None

    def _model_name(self) -> str:
        if self._served_model_name:
            return self._served_model_name
        try:
            resp = self.session.get(_runtime_url(self.remote_url, "/v1/models"), timeout=(3.0, 30.0))
            resp.raise_for_status()
            data = resp.json()
            items = list(data.get("data") or [])
            if items:
                self._served_model_name = str(items[0].get("id") or "PaddleOCR-VL-1.5-0.9B")
            else:
                self._served_model_name = "PaddleOCR-VL-1.5-0.9B"
        except Exception:
            self._served_model_name = "PaddleOCR-VL-1.5-0.9B"
        return self._served_model_name

    def run_ocr(self, image_bgr, min_score: float) -> Dict[str, Any]:
        from ui_vision_common.image_tools import encode_png_b64

        t0 = time.perf_counter()
        height, width = image_bgr.shape[:2]
        image_b64 = encode_png_b64(image_bgr)
        payload = {
            "model": self._model_name(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        {"type": "text", "text": "Spotting:"},
                    ],
                }
            ],
            "max_tokens": 2048,
        }
        resp = self.session.post(
            _runtime_url(self.remote_url, "/v1/chat/completions"),
            json=payload,
            timeout=(3.0, 120.0),
        )
        resp.raise_for_status()
        data = resp.json()
        message = (((data.get("choices") or [{}])[0]).get("message") or {})
        content = str(message.get("content", "") or "")

        score = 0.89
        lines: list[dict[str, Any]] = []
        words: list[dict[str, Any]] = []
        for row in _parse_paddle_spotting_output(content, width=width, height=height):
            if score < min_score:
                continue
            lines.append(
                {
                    "poly": row["poly"],
                    "text": row["text"],
                    "score": round(score, 4),
                    "source_model": self.model_id,
                }
            )
            for word in _split_words(row["text"], row["box"]):
                word["score"] = round(score, 4)
                word["source_model"] = self.model_id
                words.append(word)

        return {
            "width": width,
            "height": height,
            "words": words,
            "lines": lines,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "meta": {
                "runtime_endpoint": _runtime_url(self.remote_url, "/v1/chat/completions"),
                "served_model_name": self._model_name(),
                "spotting_row_count": len(lines),
            },
        }


class SuryaRuntimeProvider(BaseProvider):
    def __init__(self, model_id: str, remote_url: str):
        if not remote_url:
            raise ProviderError("MODEL_BACKEND=surya_runtime requires REMOTE_MODEL_URL")
        super().__init__(model_id=model_id, backend="surya_runtime")
        self.remote_url = remote_url.rstrip("/")
        self.session = requests.Session()

    def run_ocr(self, image_bgr, min_score: float) -> Dict[str, Any]:
        import cv2

        t0 = time.perf_counter()
        height, width = image_bgr.shape[:2]
        ok, buf = cv2.imencode(".png", image_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if not ok:
            raise ProviderError("Failed to encode image for Surya runtime")

        resp = self.session.post(
            _runtime_url(self.remote_url, "/ocr"),
            files={"images": ("frame.png", buf.tobytes(), "image/png")},
            timeout=(3.0, 120.0),
        )
        resp.raise_for_status()
        data = resp.json()

        lines: list[dict[str, Any]] = []
        words: list[dict[str, Any]] = []
        pages = list(data.get("data") or [])
        for page in pages:
            for item in list(page.get("text_lines") or []):
                text = str(item.get("text", "") or "").strip()
                if not text:
                    continue
                confidence = float(item.get("confidence") or 0.0)
                if confidence < min_score:
                    continue
                poly = [[float(x), float(y)] for x, y in list(item.get("polygon") or [])[:4]]
                if len(poly) != 4:
                    box = list(item.get("bbox") or [])
                    if len(box) != 4:
                        continue
                    poly = [
                        [float(box[0]), float(box[1])],
                        [float(box[2]), float(box[1])],
                        [float(box[2]), float(box[3])],
                        [float(box[0]), float(box[3])],
                    ]
                box = _box_from_poly(poly)
                lines.append(
                    {
                        "poly": poly,
                        "text": text,
                        "score": round(confidence, 4),
                        "source_model": self.model_id,
                    }
                )
                for word in _split_words(text, box):
                    word["score"] = round(confidence, 4)
                    word["source_model"] = self.model_id
                    words.append(word)

        return {
            "width": width,
            "height": height,
            "words": words,
            "lines": lines,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "meta": {
                "runtime_endpoint": _runtime_url(self.remote_url, "/ocr"),
                "page_count": len(pages),
            },
        }


def build_provider(model_id: str, role: str, backend: str, remote_url: str) -> BaseProvider:
    if backend == "mock":
        return MockProvider(model_id=model_id, role=role)
    if backend == "http_proxy":
        return HTTPProxyProvider(model_id=model_id, role=role, remote_url=remote_url)
    if backend == "omniparser_runtime":
        return OmniParserRuntimeProvider(model_id=model_id, remote_url=remote_url)
    if backend == "paddleocr_vl_runtime":
        return PaddleOCRVLRuntimeProvider(model_id=model_id, remote_url=remote_url)
    if backend == "surya_runtime":
        return SuryaRuntimeProvider(model_id=model_id, remote_url=remote_url)
    raise ProviderError(f"Unsupported MODEL_BACKEND={backend}")
