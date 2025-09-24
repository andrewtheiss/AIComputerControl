# ocr/postproc.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math

def _poly_to_xyxy(poly) -> List[int]:
    # poly is 4x2 or more; convert to tight bbox
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    return [x1, y1, x2, y2]

def to_polys_words(raw: Any, min_score: float = 0.45) -> List[Dict[str, Any]]:
    """
    Normalize PaddleOCR results (v2 or v3 pipeline) into word-level items:
      [{"text": str, "box": [x1,y1,x2,y2], "conf": float}, ...]
    """
    words: List[Dict[str, Any]] = []

    if raw is None:
        return words

    # ---- v3 pipeline (list[dict]) ----
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        for page in raw:
            polys   = page.get("dt_polys")   or page.get("boxes") or []
            texts   = page.get("rec_texts")  or []
            scores  = page.get("rec_scores") or []
            n = min(len(polys), len(texts), len(scores))
            for i in range(n):
                s = float(scores[i])
                if s < float(min_score):
                    continue
                box = _poly_to_xyxy(polys[i])
                words.append({"text": str(texts[i]), "box": box, "conf": s})
        return words

    # ---- v2 style ([[[poly, (text, score)], ...]]) ----
    # raw: [ [ [poly, (text, score)], ... ] ]
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        results = raw[0]
        for det in results:
            if not isinstance(det, list) or len(det) < 2:
                continue
            poly, rec = det[0], det[1]
            if not isinstance(rec, (list, tuple)) or len(rec) < 2:
                continue
            text, s = rec[0], float(rec[1])
            if s < float(min_score):
                continue
            box = _poly_to_xyxy(poly)
            words.append({"text": str(text), "box": box, "conf": s})
        return words

    # Unknown structure -> no words
    return words

def group_lines(words: List[Dict[str, Any]], y_tol: int = 12) -> List[Dict[str, Any]]:
    """
    Greedy y-banding: group words into lines by vertical overlap / proximity.
    Returns items with same schema (text, box, conf) where:
      - text is the concatenated words left->right
      - box is the union bbox
      - conf is the average confidence
    """
    if not words:
        return []

    # Sort by top (y1), then left (x1)
    ws = sorted(words, key=lambda w: (w["box"][1], w["box"][0]))
    lines: List[List[Dict[str, Any]]] = []

    for w in ws:
        x1, y1, x2, y2 = w["box"]
        placed = False
        for ln in lines:
            # check vertical overlap with the line's median band
            ly1 = min(v["box"][1] for v in ln)
            ly2 = max(v["box"][3] for v in ln)
            # overlap or within tolerance
            if not (y2 < ly1 - y_tol or y1 > ly2 + y_tol):
                ln.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    # merge each line left->right
    merged: List[Dict[str, Any]] = []
    for ln in lines:
        ln = sorted(ln, key=lambda w: w["box"][0])
        text = " ".join(w["text"] for w in ln)
        xs1 = [w["box"][0] for w in ln]; ys1 = [w["box"][1] for w in ln]
        xs2 = [w["box"][2] for w in ln]; ys2 = [w["box"][3] for w in ln]
        box = [min(xs1), min(ys1), max(xs2), max(ys2)]
        conf = sum(w["conf"] for w in ln) / max(1, len(ln))
        merged.append({"text": text, "box": box, "conf": conf})
    return merged
