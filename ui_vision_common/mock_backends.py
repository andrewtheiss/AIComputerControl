from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .image_tools import box_to_poly


OCR_MODEL_OFFSETS: Dict[str, Tuple[int, int]] = {
    "ppocr": (0, 0),
    "omniparser": (4, -2),
    "paddleocr-vl": (-3, 2),
    "surya": (2, 4),
}


def _clamp_box(box: List[int], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = box
    return [
        max(0, min(width - 1, int(x1))),
        max(0, min(height - 1, int(y1))),
        max(1, min(width, int(x2))),
        max(1, min(height, int(y2))),
    ]


def _default_word_specs(width: int, height: int) -> List[Tuple[str, List[int], float]]:
    return [
        ("Browser", [int(width * 0.04), int(height * 0.05), int(width * 0.18), int(height * 0.10)], 0.92),
        ("Sign", [int(width * 0.68), int(height * 0.16), int(width * 0.76), int(height * 0.21)], 0.96),
        ("in", [int(width * 0.77), int(height * 0.16), int(width * 0.81), int(height * 0.21)], 0.95),
        ("Continue", [int(width * 0.58), int(height * 0.62), int(width * 0.76), int(height * 0.68)], 0.88),
    ]


def mock_ocr_result(model_id: str, image_bgr: np.ndarray, min_score: float = 0.45) -> Dict[str, Any]:
    t0 = time.perf_counter()
    height, width = image_bgr.shape[:2]
    dx, dy = OCR_MODEL_OFFSETS.get(model_id, (0, 0))
    words: List[Dict[str, Any]] = []
    for text, box, score in _default_word_specs(width, height):
        shifted = _clamp_box([box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy], width, height)
        if score < min_score:
            continue
        words.append(
            {
                "poly": box_to_poly(shifted),
                "text": text,
                "score": round(score, 4),
                "source_model": model_id,
            }
        )

    lines = []
    if len(words) >= 3:
        x1 = min(int(words[1]["poly"][0][0]), int(words[2]["poly"][0][0]))
        y1 = min(int(words[1]["poly"][0][1]), int(words[2]["poly"][0][1]))
        x2 = max(int(words[1]["poly"][2][0]), int(words[2]["poly"][2][0]))
        y2 = max(int(words[1]["poly"][2][1]), int(words[2]["poly"][2][1]))
        lines.append(
            {
                "poly": box_to_poly([x1, y1, x2, y2]),
                "text": "Sign in",
                "score": round((words[1]["score"] + words[2]["score"]) / 2.0, 4),
                "source_model": model_id,
            }
        )
    if len(words) >= 4:
        lines.append(
            {
                "poly": words[3]["poly"],
                "text": "Continue",
                "score": words[3]["score"],
                "source_model": model_id,
            }
        )
    return {
        "width": width,
        "height": height,
        "words": words,
        "lines": lines,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }


MODEL_TEXT_WEIGHTS: Dict[str, float] = {
    "groundnext": 1.10,
    "aria-ui": 1.00,
    "phi-ground": 0.95,
}


def _tokenize(text: str) -> List[str]:
    return [chunk for chunk in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if chunk]


def _positional_bonus(instruction: str, box: List[int]) -> float:
    text = instruction.lower()
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bonus = 0.0
    if "top" in text:
        bonus += max(0.0, 1.0 - cy / 900.0) * 0.08
    if "bottom" in text:
        bonus += min(1.0, cy / 900.0) * 0.08
    if "left" in text:
        bonus += max(0.0, 1.0 - cx / 1440.0) * 0.08
    if "right" in text:
        bonus += min(1.0, cx / 1440.0) * 0.08
    return bonus


def _softmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    hi = max(scores)
    exps = [math.exp(score - hi) for score in scores]
    denom = sum(exps) or 1.0
    return [value / denom for value in exps]


def mock_grounding_result(
    model_id: str,
    instruction: str,
    candidates: Iterable[Dict[str, Any]],
    history: Iterable[str] | None = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    history_tokens = _tokenize(" ".join(history or []))
    inst_tokens = _tokenize(instruction)
    model_text_weight = MODEL_TEXT_WEIGHTS.get(model_id, 1.0)
    rows: List[Tuple[Dict[str, Any], float]] = []
    for candidate in candidates:
        cand_text = str(candidate.get("text", "") or "")
        cand_tokens = _tokenize(cand_text)
        overlap = len(set(inst_tokens) & set(cand_tokens))
        history_overlap = len(set(history_tokens) & set(cand_tokens))
        base = 0.10
        base += overlap * 0.35 * model_text_weight
        base += history_overlap * 0.05
        base += float(candidate.get("score", 0.0)) * 0.15
        if candidate.get("role") in ("button", "link", "textbox"):
            base += 0.05
        allowed_actions = candidate.get("allowed_actions") or []
        if "click" in allowed_actions:
            base += 0.04
        base += _positional_bonus(instruction, candidate.get("box", [0, 0, 0, 0]))
        rows.append((candidate, base))

    rows.sort(key=lambda item: item[1], reverse=True)
    probs = _softmax([score for _, score in rows])
    predictions = []
    for (candidate, raw_score), prob in list(zip(rows, probs))[: max(1, top_k)]:
        box = [int(v) for v in candidate.get("box", [0, 0, 0, 0])]
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        predictions.append(
            {
                "candidate_id": str(candidate.get("id", "")),
                "action": "click" if "click" in (candidate.get("allowed_actions") or ["click"]) else "focus",
                "p": round(float(prob), 6),
                "raw_box": box,
                "raw_point": [cx, cy],
                "rationale": f"text_overlap={raw_score:.3f}",
            }
        )
    return {
        "predictions": predictions,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }
