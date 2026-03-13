from __future__ import annotations

import json
import os
import time
import uuid
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def new_artifact_dir(base_dir: str, service_name: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    return ensure_dir(os.path.join(base_dir, service_name, f"{stamp}-{run_id}"))


def write_json(path: str, payload: Dict) -> str:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def write_image(path: str, image_bgr: np.ndarray) -> str:
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, image_bgr)
    return path


def draw_boxes(
    image_bgr: np.ndarray,
    entries: Iterable[Tuple[List[int], str, Tuple[int, int, int], int]],
) -> np.ndarray:
    canvas = image_bgr.copy()
    for box, label, color, thickness in entries:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(
                canvas,
                label[:80],
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas


def ocr_overlay(image_bgr: np.ndarray, boxes: Iterable[Dict], color: Tuple[int, int, int]) -> np.ndarray:
    entries = []
    for item in boxes:
        entries.append((item["box"], f'{item.get("text", "")} {float(item.get("score", 0.0)):.2f}', color, 1))
    return draw_boxes(image_bgr, entries)


def grounding_overlay(
    image_bgr: np.ndarray,
    candidates: Iterable[Dict],
    candidate_scores: Optional[Dict[str, Dict[str, float]]] = None,
    final_candidate_id: str = "",
) -> np.ndarray:
    entries = []
    for cand in candidates:
        cid = str(cand.get("id", ""))
        box = [int(v) for v in cand.get("box", [0, 0, 0, 0])]
        scores = candidate_scores.get(cid, {}) if candidate_scores else {}
        score_bits = " ".join(f"{name}:{value:.2f}" for name, value in sorted(scores.items()))
        label = f"{cid} {cand.get('text', '')}".strip()
        if score_bits:
            label = f"{label} {score_bits}".strip()
        color = (255, 255, 255) if cid == final_candidate_id else (120, 120, 120)
        thickness = 3 if cid == final_candidate_id else 1
        entries.append((box, label, color, thickness))
    return draw_boxes(image_bgr, entries)
