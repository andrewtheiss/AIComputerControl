from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests


def _poly_to_box(poly: Any) -> Optional[List[int]]:
    if not isinstance(poly, list) or len(poly) < 4:
        return None
    try:
        xs = [float(pt[0]) for pt in poly]
        ys = [float(pt[1]) for pt in poly]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
    except Exception:
        return None


def _group_lines(words: List[Dict[str, Any]], y_tol: int = 12) -> List[Dict[str, Any]]:
    if not words:
        return []

    rows: List[List[Dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (item["box"][1], item["box"][0])):
        x1, y1, x2, y2 = word["box"]
        placed = False
        for row in rows:
            ry1 = min(v["box"][1] for v in row)
            ry2 = max(v["box"][3] for v in row)
            if not (y2 < ry1 - y_tol or y1 > ry2 + y_tol):
                row.append(word)
                placed = True
                break
        if not placed:
            rows.append([word])

    merged: List[Dict[str, Any]] = []
    for row in rows:
        row = sorted(row, key=lambda item: item["box"][0])
        xs1 = [item["box"][0] for item in row]
        ys1 = [item["box"][1] for item in row]
        xs2 = [item["box"][2] for item in row]
        ys2 = [item["box"][3] for item in row]
        confs = [float(item.get("conf", 0)) for item in row]
        merged.append(
            {
                "text": " ".join(str(item.get("text", "") or "").strip() for item in row).strip(),
                "box": [min(xs1), min(ys1), max(xs2), max(ys2)],
                "conf": int(sum(confs) / max(1, len(confs))),
                "level": "line",
            }
        )
    return [item for item in merged if item.get("text")]


class OCRClient:
    def __init__(self, url: Optional[str] = None, min_score: float = 0.45):
        self.url = (url or "").strip()
        self.min_score = float(min_score)

    def ocr(self, bgr_img: np.ndarray) -> List[Dict[str, Any]]:
        return self.ocr_levels(bgr_img).get("words", [])

    def ocr_levels(self, bgr_img: np.ndarray) -> Dict[str, Any]:
        if not self.url:
            return self._tesseract_fallback_levels(bgr_img)

        # OCR targeting is sensitive to compression artifacts, so prefer PNG.
        ok, buf = cv2.imencode(".png", bgr_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if not ok:
            return self._tesseract_fallback_levels(bgr_img)

        try:
            r = requests.post(
                self.url,
                files={"file": ("frame.png", buf.tobytes(), "image/png")},
                timeout=(3.0, 15.0),
            )
            r.raise_for_status()
            j = r.json()
        except Exception:
            # Fallback to local Tesseract if HTTP fails
            return self._tesseract_fallback_levels(bgr_img)

        words = self._normalize_api_items(j.get("words", []) or [], level="word")
        lines = self._normalize_api_items(j.get("lines", []) or [], level="line")
        if words and not lines:
            lines = _group_lines(words)
        return {"words": words, "lines": lines, "source": "http"}

    def _normalize_api_items(self, items: List[Dict[str, Any]], level: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in items:
            try:
                score = float(item.get("score", 0.0))
                if score < self.min_score:
                    continue
                box = None
                if isinstance(item.get("box"), list) and len(item.get("box")) == 4:
                    box = [int(v) for v in item.get("box")]
                if box is None:
                    box = _poly_to_box(item.get("poly") or [])
                if box is None:
                    continue
                out.append(
                    {
                        "text": str(item.get("text", "") or "").strip(),
                        "conf": int(score * 100) if score <= 1.0 else int(score),
                        "box": box,
                        "level": level,
                    }
                )
            except Exception:
                continue
        return [item for item in out if item.get("text")]

    def _tesseract_fallback(self, bgr_img: np.ndarray) -> List[Dict[str, Any]]:
        return self._tesseract_fallback_levels(bgr_img).get("words", [])

    def _tesseract_fallback_levels(self, bgr_img: np.ndarray) -> Dict[str, Any]:
        # Import here to avoid hard dependency if HTTP is used
        import pytesseract

        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        try:
            data = pytesseract.image_to_data(
                rgb,
                output_type=pytesseract.Output.DICT,
                config="--oem 1 --psm 6",
            )
        except Exception:
            return {"words": [], "lines": [], "source": "tesseract"}

        words: List[Dict[str, Any]] = []
        n = len(data.get("text", []))
        conf_cutoff = int(self.min_score * 100)
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            conf_str = str(data["conf"][i])
            if conf_str == "-1":
                continue
            try:
                conf = int(float(conf_str))
            except Exception:
                conf = 0
            if conf < conf_cutoff:
                continue
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            words.append({"text": txt, "box": [x, y, x + w, y + h], "conf": conf, "level": "word"})
        return {"words": words, "lines": _group_lines(words), "source": "tesseract"}


