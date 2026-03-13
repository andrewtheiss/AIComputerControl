from __future__ import annotations

import base64
import io
from typing import Iterable, List, Optional

import cv2
import numpy as np
from fastapi import HTTPException


def decode_image_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(status_code=400, detail="Empty image payload")
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image

        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as exc:  # pragma: no cover - pillow fallback
        raise HTTPException(status_code=415, detail=f"Unsupported image payload: {exc}") from exc


def decode_image_b64(image_b64: str) -> np.ndarray:
    try:
        payload = image_b64 or ""
        if payload.startswith("data:"):
            comma = payload.find(",")
            payload = payload[comma + 1 :] if comma != -1 else ""
        return decode_image_bytes(base64.b64decode(payload, validate=False))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image payload: {exc}") from exc


def encode_png_b64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise ValueError("Could not encode image as PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def poly_to_box(poly: Iterable[Iterable[float]]) -> Optional[List[int]]:
    pts = list(poly or [])
    if len(pts) < 4:
        return None
    try:
        xs = [float(pt[0]) for pt in pts]
        ys = [float(pt[1]) for pt in pts]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
    except Exception:
        return None


def box_to_poly(box: List[int]) -> List[List[float]]:
    x1, y1, x2, y2 = [int(v) for v in box]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
