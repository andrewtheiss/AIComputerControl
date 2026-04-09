from __future__ import annotations

import ast
import io
import json
import math
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
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_POINT_TAG_RE = re.compile(r"<point>\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*</point>", re.IGNORECASE)


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


def _message_content(data: Dict[str, Any]) -> str:
    message = (((data.get("choices") or [{}])[0]).get("message") or {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "") or ""))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def _extract_json_object(content: str) -> Dict[str, Any] | None:
    match = _TOOL_CALL_RE.search(content or "")
    snippet = match.group(1) if match else ""
    if not snippet:
        start = str(content or "").find("{")
        end = str(content or "").rfind("}")
        if start >= 0 and end > start:
            snippet = str(content or "")[start : end + 1]
    if not snippet:
        return None
    try:
        value = json.loads(snippet)
    except Exception:
        try:
            value = ast.literal_eval(snippet)
        except Exception:
            return None
    return value if isinstance(value, dict) else None


def _extract_number_list(content: str, expected: int) -> list[float] | None:
    text = str(content or "")
    for token in ("```python", "```json", "```", "<|im_end|>", "<|end|>"):
        text = text.replace(token, "")
    try:
        value = ast.literal_eval(text.strip())
    except Exception:
        value = None
    if isinstance(value, (list, tuple)) and len(value) >= expected:
        out: list[float] = []
        for item in list(value)[:expected]:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                return None
        return out
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(numbers) < expected:
        return None
    try:
        return [float(number) for number in numbers[:expected]]
    except ValueError:
        return None


def _scale_relative_value(raw_value: float, size: int) -> int:
    value = float(raw_value)
    if abs(value) <= 1.0:
        scaled = value * size
    else:
        scaled = (value / 1000.0) * size
    return max(0, min(size, int(round(scaled))))


def _maybe_absolute_point(values: list[float], width: int, height: int) -> list[int] | None:
    if len(values) < 2:
        return None
    if 0.0 <= float(values[0]) <= float(width) and 0.0 <= float(values[1]) <= float(height):
        return [int(round(float(values[0]))), int(round(float(values[1])))]
    return None


def _extract_point(content: str, width: int, height: int) -> list[int] | None:
    match = _POINT_TAG_RE.search(str(content or ""))
    if match:
        values = [float(match.group(1)), float(match.group(2))]
    else:
        values = _extract_number_list(content, expected=2)
        if not values:
            return None
    return _maybe_absolute_point(values, width=width, height=height) or _relative_point_to_px(values, width=width, height=height)


def _relative_point_to_px(values: list[float], width: int, height: int) -> list[int]:
    return [
        _scale_relative_value(values[0], width),
        _scale_relative_value(values[1], height),
    ]


def _candidate_clickable(candidate: Dict[str, Any]) -> bool:
    allowed = list(candidate.get("allowed_actions") or [])
    return not allowed or "click" in allowed


def _candidate_box(candidate: Dict[str, Any]) -> list[int] | None:
    box = list(candidate.get("box") or [])
    if len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _point_in_box(point: list[int], box: list[int]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _distance_point_to_box(point: list[int], box: list[int]) -> float:
    px, py = point
    dx = max(box[0] - px, 0, px - box[2])
    dy = max(box[1] - py, 0, py - box[3])
    return math.hypot(dx, dy)


def _box_center(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _box_iou(a: list[int], b: list[int]) -> float:
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter_area / float(area_a + area_b - inter_area)


def _score_candidate_for_point(point: list[int], box: list[int], width: int, height: int) -> float:
    diag = max(1.0, math.hypot(width, height))
    center_x, center_y = _box_center(box)
    center_distance = math.hypot(point[0] - center_x, point[1] - center_y) / diag
    if _point_in_box(point, box):
        area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / float(max(1, width * height))
        tightness = max(0.0, 1.0 - min(1.0, area_ratio))
        return max(0.05, min(0.99, 0.78 + (0.16 * (1.0 - min(1.0, center_distance * 2.0))) + (0.05 * tightness)))
    edge_distance = _distance_point_to_box(point, box) / diag
    return max(0.01, min(0.45, 0.35 * (1.0 - min(1.0, edge_distance * 3.0))))


def _score_candidate_for_box(raw_box: list[int], box: list[int], width: int, height: int) -> float:
    iou = _box_iou(raw_box, box)
    center_a = _box_center(raw_box)
    center_b = _box_center(box)
    diag = max(1.0, math.hypot(width, height))
    center_distance = math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]) / diag
    if iou > 0.0:
        return max(0.05, min(0.99, 0.65 + (0.30 * iou) + (0.04 * (1.0 - min(1.0, center_distance * 2.0)))))
    return max(0.01, min(0.40, 0.28 * (1.0 - min(1.0, center_distance * 2.5))))


def _rank_candidates_from_point(
    candidates: list[Dict[str, Any]],
    point: list[int],
    width: int,
    height: int,
    top_k: int,
    rationale: str | None = None,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("id", "") or "")
        box = _candidate_box(candidate)
        if not candidate_id or not box or not _candidate_clickable(candidate):
            continue
        ranked.append(
            {
                "candidate_id": candidate_id,
                "action": "click",
                "p": round(_score_candidate_for_point(point, box, width=width, height=height), 4),
                "raw_point": point,
                "rationale": rationale,
            }
        )
    ranked.sort(key=lambda item: item["p"], reverse=True)
    return ranked[: max(1, top_k)]


def _rank_candidates_from_box(
    candidates: list[Dict[str, Any]],
    raw_box: list[int],
    width: int,
    height: int,
    top_k: int,
    rationale: str | None = None,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("id", "") or "")
        box = _candidate_box(candidate)
        if not candidate_id or not box or not _candidate_clickable(candidate):
            continue
        ranked.append(
            {
                "candidate_id": candidate_id,
                "action": "click",
                "p": round(_score_candidate_for_box(raw_box, box, width=width, height=height), 4),
                "raw_box": raw_box,
                "rationale": rationale,
            }
        )
    ranked.sort(key=lambda item: item["p"], reverse=True)
    return ranked[: max(1, top_k)]


class _OpenAIRuntimeProvider(BaseProvider):
    def __init__(self, model_id: str, backend: str, remote_url: str):
        if not remote_url:
            raise ProviderError(f"MODEL_BACKEND={backend} requires REMOTE_MODEL_URL")
        super().__init__(model_id=model_id, backend=backend)
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
            self._served_model_name = str((items[0] if items else {}).get("id") or self.model_id)
        except Exception:
            self._served_model_name = self.model_id
        return self._served_model_name

    def _chat_completion(self, messages: list[dict[str, Any]], max_tokens: int = 128) -> tuple[dict[str, Any], str]:
        payload = {
            "model": self._model_name(),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stop": ["<|im_end|>", "<|end|>"],
        }
        resp = self.session.post(
            _runtime_url(self.remote_url, "/v1/chat/completions"),
            json=payload,
            timeout=(3.0, 180.0),
        )
        resp.raise_for_status()
        data = resp.json()
        return data, _message_content(data)


class GroundNextRuntimeProvider(_OpenAIRuntimeProvider):
    def __init__(self, model_id: str, remote_url: str):
        super().__init__(model_id=model_id, backend="groundnext_runtime", remote_url=remote_url)

    def run_grounding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from ui_vision_common.image_tools import decode_image_b64, encode_png_b64

        t0 = time.perf_counter()
        screenshot_b64 = str(payload.get("screenshot_b64", "") or "")
        image_bgr = decode_image_b64(screenshot_b64)
        height, width = image_bgr.shape[:2]
        instruction = str(payload.get("instruction", "") or "").strip()
        candidates = list(payload.get("candidates") or [])
        prompt = (
            f"Screen size: {width}x{height}.\n"
            "Identify the GUI element described by the instruction and return only one tool call in this exact format:\n"
            '<tool_call>{"name":"computer_use","arguments":{"action":"left_click","coordinate":[x,y]}}</tool_call>\n'
            f"Instruction: {instruction}"
        )
        _, content = self._chat_completion(
            [
                {
                    "role": "system",
                    "content": "You are a GUI grounding assistant that returns only a computer_use tool call.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_png_b64(image_bgr)}"}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            max_tokens=96,
        )
        tool_call = _extract_json_object(content)
        arguments = dict((tool_call or {}).get("arguments") or {})
        coordinates = list(arguments.get("coordinate") or [])
        if len(coordinates) != 2:
            raise ProviderError(f"GroundNext runtime returned unparseable tool call: {content[:300]}")
        raw_point = [int(round(float(coordinates[0]))), int(round(float(coordinates[1])))]
        predictions = _rank_candidates_from_point(
            candidates,
            point=raw_point,
            width=width,
            height=height,
            top_k=int(payload.get("top_k") or 5),
            rationale=content[:240],
        )
        return {
            "predictions": predictions,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "meta": {
                "runtime_endpoint": _runtime_url(self.remote_url, "/v1/chat/completions"),
                "served_model_name": self._model_name(),
                "raw_action": str(arguments.get("action") or "left_click"),
                "response_excerpt": content[:240],
            },
        }


class AriaUIRuntimeProvider(_OpenAIRuntimeProvider):
    def __init__(self, model_id: str, remote_url: str):
        super().__init__(model_id=model_id, backend="aria_ui_runtime", remote_url=remote_url)

    def run_grounding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from ui_vision_common.image_tools import decode_image_b64, encode_png_b64

        t0 = time.perf_counter()
        screenshot_b64 = str(payload.get("screenshot_b64", "") or "")
        image_bgr = decode_image_b64(screenshot_b64)
        height, width = image_bgr.shape[:2]
        instruction = str(payload.get("instruction", "") or "").strip()
        candidates = list(payload.get("candidates") or [])
        prompt = (
            "Given a GUI image, what are the relative (0-1000) pixel point coordinates "
            "for the element corresponding to the following instruction or description: "
            f"{instruction}\nReturn only a Python-style list [x, y]."
        )
        _, content = self._chat_completion(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_png_b64(image_bgr)}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=64,
        )
        relative_point = _extract_number_list(content, expected=2)
        if not relative_point:
            raise ProviderError(f"Aria runtime returned unparseable point: {content[:300]}")
        raw_point = _relative_point_to_px(relative_point, width=width, height=height)
        predictions = _rank_candidates_from_point(
            candidates,
            point=raw_point,
            width=width,
            height=height,
            top_k=int(payload.get("top_k") or 5),
            rationale=content[:240],
        )
        return {
            "predictions": predictions,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "meta": {
                "runtime_endpoint": _runtime_url(self.remote_url, "/v1/chat/completions"),
                "served_model_name": self._model_name(),
                "relative_point": [round(value, 4) for value in relative_point],
                "response_excerpt": content[:240],
            },
        }


class PhiGroundRuntimeProvider(_OpenAIRuntimeProvider):
    TARGET_WIDTH = 336 * 3
    TARGET_HEIGHT = 336 * 2

    def __init__(self, model_id: str, remote_url: str):
        super().__init__(model_id=model_id, backend="phi_ground_runtime", remote_url=remote_url)

    @classmethod
    def _prepare_image(cls, image_bgr) -> tuple[str, float]:
        import base64

        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image_ratio = image.width / float(max(1, image.height))
        target_ratio = cls.TARGET_WIDTH / float(cls.TARGET_HEIGHT)
        if image_ratio > target_ratio:
            new_width = cls.TARGET_WIDTH
            new_height = max(1, int(round(new_width / image_ratio)))
        else:
            new_height = cls.TARGET_HEIGHT
            new_width = max(1, int(round(new_height * image_ratio)))
        reshape_ratio = new_width / float(max(1, image.width))
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        padded = Image.new("RGB", (cls.TARGET_WIDTH, cls.TARGET_HEIGHT), (255, 255, 255))
        padded.paste(resized, (0, 0))
        buffer = io.BytesIO()
        padded.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii"), reshape_ratio

    @classmethod
    def _restore_box(cls, relative_box: list[float], reshape_ratio: float, width: int, height: int) -> list[int]:
        prepared = [
            _scale_relative_value(relative_box[0], cls.TARGET_WIDTH),
            _scale_relative_value(relative_box[1], cls.TARGET_HEIGHT),
            _scale_relative_value(relative_box[2], cls.TARGET_WIDTH),
            _scale_relative_value(relative_box[3], cls.TARGET_HEIGHT),
        ]
        restored = [
            int(round(prepared[0] / max(reshape_ratio, 1e-6))),
            int(round(prepared[1] / max(reshape_ratio, 1e-6))),
            int(round(prepared[2] / max(reshape_ratio, 1e-6))),
            int(round(prepared[3] / max(reshape_ratio, 1e-6))),
        ]
        x1 = max(0, min(width - 1, restored[0]))
        y1 = max(0, min(height - 1, restored[1]))
        x2 = max(x1 + 1, min(width, restored[2]))
        y2 = max(y1 + 1, min(height, restored[3]))
        return [x1, y1, x2, y2]

    def run_grounding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from ui_vision_common.image_tools import decode_image_b64

        t0 = time.perf_counter()
        screenshot_b64 = str(payload.get("screenshot_b64", "") or "")
        image_bgr = decode_image_b64(screenshot_b64)
        height, width = image_bgr.shape[:2]
        instruction = str(payload.get("instruction", "") or "").strip()
        candidates = list(payload.get("candidates") or [])
        prepared_b64, reshape_ratio = self._prepare_image(image_bgr)
        prompt = (
            "The description of the element:\n"
            f"{instruction}\n\n"
            "Locate the above described element in the image. "
            "The output should be bounding box using relative coordinates multiplying 1000.\n"
            "Return only a Python-style list [x1, y1, x2, y2]."
        )
        _, content = self._chat_completion(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prepared_b64}"}},
                    ],
                }
            ],
            max_tokens=96,
        )
        relative_box = _extract_number_list(content, expected=4)
        if relative_box:
            raw_box = self._restore_box(relative_box, reshape_ratio=reshape_ratio, width=width, height=height)
            predictions = _rank_candidates_from_box(
                candidates,
                raw_box=raw_box,
                width=width,
                height=height,
                top_k=int(payload.get("top_k") or 5),
                rationale=content[:240],
            )
            return {
                "predictions": predictions,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "meta": {
                    "runtime_endpoint": _runtime_url(self.remote_url, "/v1/chat/completions"),
                    "served_model_name": self._model_name(),
                    "relative_box": [round(value, 4) for value in relative_box],
                    "response_excerpt": content[:240],
                },
            }

        raw_point = _extract_point(content, width=width, height=height)
        if not raw_point:
            raise ProviderError(f"Phi-Ground runtime returned unparseable localization: {content[:300]}")
        predictions = _rank_candidates_from_point(
            candidates,
            point=raw_point,
            width=width,
            height=height,
            top_k=int(payload.get("top_k") or 5),
            rationale=content[:240],
        )
        return {
            "predictions": predictions,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "meta": {
                "runtime_endpoint": _runtime_url(self.remote_url, "/v1/chat/completions"),
                "served_model_name": self._model_name(),
                "raw_point": raw_point,
                "response_excerpt": content[:240],
            },
        }


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
        content = _message_content(data)

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
    if backend == "groundnext_runtime":
        return GroundNextRuntimeProvider(model_id=model_id, remote_url=remote_url)
    if backend == "aria_ui_runtime":
        return AriaUIRuntimeProvider(model_id=model_id, remote_url=remote_url)
    if backend == "phi_ground_runtime":
        return PhiGroundRuntimeProvider(model_id=model_id, remote_url=remote_url)
    raise ProviderError(f"Unsupported MODEL_BACKEND={backend}")
