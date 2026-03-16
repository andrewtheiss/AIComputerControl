from __future__ import annotations

import re
from typing import Any, Dict, List


INTERACTABLE_ROLES = {
    "button",
    "checkbox",
    "combobox",
    "entry",
    "input",
    "link",
    "menuitem",
    "radio",
    "tab",
    "textarea",
    "textbox",
}

LIVE_EXECUTION_ACTIONS = {"click_text", "click_any_text", "click_near_text", "click_box"}


def build_target_ensemble_endpoint(base_url: str, debug: bool) -> str:
    base = (base_url or "").rstrip("/")
    if base.endswith("/infer/debug"):
        base = base[: -len("/infer/debug")]
    elif base.endswith("/infer"):
        base = base[: -len("/infer")]
    return base + ("/infer/debug" if debug else "/infer")


def compute_interactable_score(score: float, role: str, source: str, allowed_actions: List[str]) -> float:
    role = str(role or "").lower()
    source = str(source or "").lower()
    value = float(score)
    is_interactable = role in INTERACTABLE_ROLES or source in {"ax", "det"} or "omniparser" in source
    if is_interactable:
        value += 0.18
    if role in {"button", "link", "textbox", "input", "textarea"}:
        value += 0.08
    if "click" in (allowed_actions or []):
        value += 0.04
    return min(1.0, max(0.0, value))


def _box_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = [int(v) for v in a]
    bx1, by1, bx2, by2 = [int(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(max(1, a_area + b_area - inter))


def _center_distance(a: List[int], b: List[int]) -> float:
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def should_merge_shadow_node(
    norm_box: List[int],
    text_key: str,
    interactable_score: float,
    group: Dict[str, Any],
) -> bool:
    group_box = [int(v) for v in (group.get("box") or [0, 0, 0, 0])]
    group_text_key = str(group.get("text_key", "") or "")
    iou = _box_iou(norm_box, group_box)
    if iou >= 0.65:
        return True
    same_text = bool(text_key) and text_key == group_text_key
    if same_text and iou >= 0.18:
        return True
    if same_text and _center_distance(norm_box, group_box) <= 18.0:
        return True
    if (not text_key or not group_text_key) and iou >= 0.45:
        return True
    group_interactable = max(
        (float((node or {}).get("interactable_score", 0.0)) for node in group.get("nodes", [])),
        default=0.0,
    )
    return interactable_score >= 0.65 and group_interactable >= 0.65 and iou >= 0.35


def _pattern_matches_text(pattern: str, text: str) -> bool:
    raw_pattern = str(pattern or "").strip()
    raw_text = str(text or "")
    if not raw_pattern:
        return False
    try:
        return bool(re.search(raw_pattern, raw_text, re.I))
    except re.error:
        pat = raw_pattern.lower()
        txt = raw_text.lower()
        return pat == txt or pat in txt or txt in pat


def _normalize_box(box: Any) -> List[int] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        norm_box = [int(v) for v in box]
    except Exception:
        return None
    if norm_box[2] <= norm_box[0] or norm_box[3] <= norm_box[1]:
        return None
    return norm_box


def _box_center(box: List[int]) -> List[int]:
    return [int(round((box[0] + box[2]) / 2.0)), int(round((box[1] + box[3]) / 2.0))]


def _point_in_box(point: List[int], box: List[int]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _distance_point_to_box(point: List[int], box: List[int]) -> float:
    px, py = [int(v) for v in point]
    x = max(box[0], min(px, box[2]))
    y = max(box[1], min(py, box[3]))
    return ((px - x) ** 2 + (py - y) ** 2) ** 0.5


def _matches_requested_text(action: str, params: Dict[str, Any], text: str) -> bool:
    if action == "click_text":
        fuzzy_text = str((params or {}).get("fuzzy_text", "") or "").strip()
        regex = str((params or {}).get("regex", "") or "").strip()
        if fuzzy_text:
            return fuzzy_text.lower() in str(text or "").lower()
        if regex:
            return _pattern_matches_text(regex, text)
        return True
    if action == "click_any_text":
        patterns = (params or {}).get("patterns") or []
        if not isinstance(patterns, list) or not patterns:
            return True
        return any(_pattern_matches_text(str(pattern), text) for pattern in patterns if str(pattern or "").strip())
    return False


def _matches_requested_click_box(params: Dict[str, Any], final_box: List[int]) -> bool:
    requested_box = _normalize_box((params or {}).get("box"))
    if requested_box is None:
        return False
    final_center = _box_center(final_box)
    requested_center = _box_center(requested_box)
    if _point_in_box(final_center, requested_box) or _point_in_box(requested_center, final_box):
        return True
    return _box_iou(requested_box, final_box) >= 0.20


def _matches_requested_near_text(params: Dict[str, Any], resolved_target: Dict[str, Any] | None, final_box: List[int]) -> bool:
    if not isinstance(resolved_target, dict):
        return False
    anchor_box = _normalize_box(resolved_target.get("box"))
    if anchor_box is None:
        return False
    dx = int((params or {}).get("dx", 0) or 0)
    dy = int((params or {}).get("dy", 0) or 0)
    anchor_center = _box_center(anchor_box)
    intended_point = [anchor_center[0] + dx, anchor_center[1] + dy]
    return _point_in_box(intended_point, final_box) or _distance_point_to_box(intended_point, final_box) <= 24.0


def resolve_execution_override(
    action: str,
    params: Dict[str, Any],
    shadow_targeting: Dict[str, Any] | None,
    enabled: bool,
    resolved_target: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    if not enabled or action not in LIVE_EXECUTION_ACTIONS or not isinstance(shadow_targeting, dict):
        return None
    if not bool(shadow_targeting.get("auto_execute")):
        return None
    final_prediction = dict(shadow_targeting.get("final_prediction") or {})
    norm_box = _normalize_box(final_prediction.get("box"))
    if norm_box is None:
        return None
    final_text = str(final_prediction.get("text", "") or "")
    if action in {"click_text", "click_any_text"}:
        if not _matches_requested_text(action, params, final_text):
            return None
    elif action == "click_box":
        if not _matches_requested_click_box(params, norm_box):
            return None
    elif action == "click_near_text":
        if not _matches_requested_near_text(params, resolved_target, norm_box):
            return None
    return {
        "action": "click_box",
        "params": {"box": norm_box},
        "meta": {
            "candidate_id": str(final_prediction.get("candidate_id", "") or ""),
            "text": final_text,
            "score": float(final_prediction.get("score", 0.0) or 0.0),
            "resolution_mode": str(shadow_targeting.get("resolution_mode", "") or ""),
            "source_action": action,
            "applied": True,
        },
    }
