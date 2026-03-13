from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def infer_allowed_actions(element: Dict[str, Any]) -> List[str]:
    role = str(element.get("role", "") or "").lower()
    text = str(element.get("text", "") or "").strip().lower()
    source = str(element.get("source", "") or "").lower()
    if role in {"textbox", "entry", "textarea", "input"}:
        return ["click", "type", "focus"]
    if role in {"button", "link", "menuitem", "tab", "checkbox", "radio"}:
        return ["click", "hover"]
    if source == "ax" and role:
        return ["click", "focus"]
    if text:
        return ["click"]
    return ["click"]


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _normalize_box(box: Any) -> Optional[List[int]]:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in box]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _box_area(box: List[int]) -> int:
    return max(1, (box[2] - box[0]) * (box[3] - box[1]))


def _center(box: List[int]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _center_distance(a: List[int], b: List[int]) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    return inter / float(max(1, union))


def _box_union(a: List[int], b: List[int]) -> List[int]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _source_kind(source: str) -> str:
    src = (source or "").lower()
    if src in {"ocr", "ocr_word", "ocr_line", "ppocr", "paddleocr-vl", "surya"}:
        return "ocr"
    if "omniparser" in src:
        return "parser"
    if src == "ax":
        return "ax"
    if src == "det":
        return "det"
    return "other"


def _is_interactable(role: str, source: str, allowed_actions: List[str]) -> bool:
    if role in INTERACTABLE_ROLES:
        return True
    if source in {"ax", "det"}:
        return True
    if "omniparser" in source:
        return True
    return "click" in allowed_actions and not source.startswith("ocr_")


def _interactable_score(score: float, role: str, source: str, allowed_actions: List[str]) -> float:
    value = float(score)
    if _is_interactable(role=role, source=source, allowed_actions=allowed_actions):
        value += 0.18
    if role in {"button", "link", "textbox", "input", "textarea"}:
        value += 0.08
    if "click" in allowed_actions:
        value += 0.04
    return min(1.0, max(0.0, value))


def _make_node(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    box = _normalize_box(raw.get("box"))
    if box is None:
        return None
    text = str(raw.get("text", "") or "")
    role = str(raw.get("role", "") or "").lower()
    source = str(raw.get("source", "") or "")
    score = float(raw.get("score", 0.0) or 0.0)
    allowed_actions = list(raw.get("allowed_actions") or infer_allowed_actions(raw))
    return {
        "box": box,
        "text": text,
        "text_key": _normalize_text(text),
        "score": score,
        "role": role or None,
        "source": source,
        "source_kind": _source_kind(source),
        "allowed_actions": allowed_actions,
        "interactable_score": _interactable_score(score=score, role=role, source=source.lower(), allowed_actions=allowed_actions),
    }


def _should_merge(node: Dict[str, Any], group: Dict[str, Any]) -> bool:
    iou = _iou(node["box"], group["box"])
    if iou >= 0.65:
        return True
    same_text = bool(node["text_key"]) and node["text_key"] == group["text_key"]
    if same_text and iou >= 0.18:
        return True
    if same_text and _center_distance(node["box"], group["box"]) <= 18.0:
        return True
    one_empty = not node["text_key"] or not group["text_key"]
    if one_empty and iou >= 0.45:
        return True
    group_interactable = max((item["interactable_score"] for item in group.get("nodes", [])), default=0.0)
    both_interactable = node["interactable_score"] >= 0.65 and group_interactable >= 0.65
    if both_interactable and iou >= 0.35:
        return True
    return False


def _best_text(nodes: List[Dict[str, Any]]) -> str:
    best = ""
    best_rank = (-1.0, -1)
    for node in nodes:
        text = node["text"]
        if not text.strip():
            continue
        rank = (float(node["score"]), len(text.strip()))
        if rank > best_rank:
            best = text
            best_rank = rank
    return best


def _best_role(nodes: List[Dict[str, Any]]) -> Optional[str]:
    for node in sorted(nodes, key=lambda item: (item["interactable_score"], item["score"]), reverse=True):
        if node["role"]:
            return node["role"]
    return None


def _best_source(nodes: List[Dict[str, Any]]) -> str:
    def rank(node: Dict[str, Any]) -> Tuple[int, float]:
        source_kind = node["source_kind"]
        kind_rank = {"ax": 4, "parser": 3, "ocr": 2, "det": 1, "other": 0}.get(source_kind, 0)
        return (kind_rank, float(node["score"]))

    return max(nodes, key=rank)["source"] if nodes else ""


def _ocr_consensus(text_sources: Dict[str, Dict[str, Any]]) -> float:
    normalized = [_normalize_text((payload or {}).get("text", "")) for payload in text_sources.values()]
    normalized = [text for text in normalized if text]
    if not normalized:
        return 0.0
    distinct = set(normalized)
    if len(distinct) == 1:
        return 1.0 if len(normalized) > 1 else 0.75
    return max(0.3, 1.0 - 0.25 * (len(distinct) - 1))


def _resolve_viewport(groups: List[Dict[str, Any]], viewport: Optional[Dict[str, int] | Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(viewport, dict):
        width = int(viewport.get("width", 0) or 0)
        height = int(viewport.get("height", 0) or 0)
        if width > 0 and height > 0:
            return width, height
    if isinstance(viewport, tuple) and len(viewport) == 2:
        width = int(viewport[0] or 0)
        height = int(viewport[1] or 0)
        if width > 0 and height > 0:
            return width, height
    max_x = max((group["box"][2] for group in groups), default=1000)
    max_y = max((group["box"][3] for group in groups), default=1000)
    return max(1, max_x), max(1, max_y)


def _rel_1000(box: List[int], width: int, height: int) -> List[int]:
    return [
        int(round(box[0] * 1000.0 / max(1, width))),
        int(round(box[1] * 1000.0 / max(1, height))),
        int(round(box[2] * 1000.0 / max(1, width))),
        int(round(box[3] * 1000.0 / max(1, height))),
    ]


def build_candidate_graph(
    ui_elements: Iterable[Dict[str, Any]],
    limit: int = 80,
    viewport: Optional[Dict[str, int] | Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for raw in list(ui_elements)[:limit]:
        node = _make_node(raw)
        if node is None:
            continue
        matched = None
        for group in groups:
            if _should_merge(node, group):
                matched = group
                break
        if matched is None:
            groups.append({"box": node["box"], "text_key": node["text_key"], "nodes": [node]})
            continue
        matched["box"] = _box_union(matched["box"], node["box"])
        if not matched.get("text_key") and node["text_key"]:
            matched["text_key"] = node["text_key"]
        matched["nodes"].append(node)

    width, height = _resolve_viewport(groups, viewport)
    graph: List[Dict[str, Any]] = []
    for idx, group in enumerate(
        sorted(
            groups,
            key=lambda item: (
                max(node["interactable_score"] for node in item["nodes"]),
                max(node["score"] for node in item["nodes"]),
                -_box_area(item["box"]),
            ),
            reverse=True,
        ),
        start=1,
    ):
        nodes = group["nodes"]
        box = [int(v) for v in group["box"]]
        text = _best_text(nodes)
        role_hint = _best_role(nodes)
        allowed_actions = sorted({action for node in nodes for action in node["allowed_actions"]})
        text_sources = {
            node["source"]: {"text": node["text"], "conf": round(float(node["score"]), 4)}
            for node in nodes
            if node["text"].strip()
        }
        interactable_score = max(node["interactable_score"] for node in nodes)
        action_compatibility = 1.0 if "click" in allowed_actions else 0.5 if allowed_actions else 0.2
        source_mask = sorted({node["source"] for node in nodes if node["source"]})
        primary_source = _best_source(nodes)
        center_x, center_y = _center(box)
        graph.append(
            {
                "id": f"C{idx:03d}",
                "bbox_abs": box,
                "bbox_rel_1000": _rel_1000(box, width=width, height=height),
                "center_abs": [int(round(center_x)), int(round(center_y))],
                "text": text,
                "text_sources": text_sources,
                "interactable_score": round(interactable_score, 6),
                "ocr_consensus": round(_ocr_consensus(text_sources), 6),
                "action_compatibility": round(action_compatibility, 6),
                "role_hint": role_hint,
                "allowed_actions": allowed_actions,
                "source_mask": source_mask,
                "primary_source": primary_source,
                "candidate_type": "interactable" if interactable_score >= 0.65 else "text_only",
                "source_count": len(source_mask),
            }
        )
    return graph


def candidate_graph_to_candidates(candidate_graph: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for entry in candidate_graph:
        candidates.append(
            {
                "id": str(entry["id"]),
                "box": [int(v) for v in entry["bbox_abs"]],
                "text": str(entry.get("text", "") or ""),
                "score": float(entry.get("interactable_score", 0.0) or 0.0),
                "role": entry.get("role_hint"),
                "source": entry.get("primary_source"),
                "allowed_actions": list(entry.get("allowed_actions") or []),
                "extras": {
                    "bbox_rel_1000": list(entry.get("bbox_rel_1000") or []),
                    "center_abs": list(entry.get("center_abs") or []),
                    "text_sources": dict(entry.get("text_sources") or {}),
                    "interactable_score": float(entry.get("interactable_score", 0.0) or 0.0),
                    "ocr_consensus": float(entry.get("ocr_consensus", 0.0) or 0.0),
                    "action_compatibility": float(entry.get("action_compatibility", 0.0) or 0.0),
                    "source_mask": list(entry.get("source_mask") or []),
                    "candidate_type": str(entry.get("candidate_type", "") or ""),
                    "source_count": int(entry.get("source_count", 0) or 0),
                    "graph_version": "candidate_graph_v1",
                },
            }
        )
    return candidates


def build_candidates(
    ui_elements: Iterable[Dict[str, Any]],
    limit: int = 80,
    viewport: Optional[Dict[str, int] | Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    return candidate_graph_to_candidates(build_candidate_graph(ui_elements=ui_elements, limit=limit, viewport=viewport))
