import base64
import io
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests


class OCRClient:
    def __init__(self, url: Optional[str] = None, min_score: float = 0.45):
        self.url = (url or "").strip()
        self.min_score = float(min_score)

    def ocr(self, bgr_img: np.ndarray) -> List[Dict[str, Any]]:
        if not self.url:
            return self._tesseract_fallback(bgr_img)

        # Encode to JPEG and call HTTP API (multipart /ocr)
        ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return self._tesseract_fallback(bgr_img)

        try:
            r = requests.post(
                self.url,
                files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                timeout=(3.0, 15.0),
            )
            r.raise_for_status()
            j = r.json()
        except Exception:
            # Fallback to local Tesseract if HTTP fails
            return self._tesseract_fallback(bgr_img)

        out: List[Dict[str, Any]] = []
        words = j.get("words", []) or []
        for w in words:
            try:
                score = float(w.get("score", 0.0))
                if score < self.min_score:
                    continue
                poly = w.get("poly") or []
                x1, y1 = poly[0]
                x3, y3 = poly[2]
                out.append(
                    {
                        "text": w.get("text", ""),
                        "conf": int(score * 100),
                        "box": [int(x1), int(y1), int(x3), int(y3)],
                    }
                )
            except Exception:
                continue
        return out

    def _tesseract_fallback(self, bgr_img: np.ndarray) -> List[Dict[str, Any]]:
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
            return []

        out: List[Dict[str, Any]] = []
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
            out.append({"text": txt, "box": [x, y, x + w, y + h], "conf": conf})
        return out


