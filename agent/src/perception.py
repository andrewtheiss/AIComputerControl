# agent/src/perception.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from ui_core import UIElement, Observation
import requests
import os

A11Y_URL = os.environ.get("A11Y_BRIDGE_URL", "").strip()  # optional sidecar

def normalize_score(s: float) -> float:
    # clamp to 0..1
    if s is None: return 0.0
    return max(0.0, min(1.0, float(s)))

def ocr_to_elements(ocr_items: List[Dict[str,Any]]) -> List[UIElement]:
    out = []
    for w in ocr_items:
        out.append(UIElement(
            source="ocr",
            text=w.get("text",""),
            box=[int(v) for v in w.get("box",[0,0,0,0])],
            score=normalize_score(w.get("conf", 0.0)),
            role=None
        ))
    return out

def det_to_elements(dets: List[Dict[str,Any]], label_map: Optional[Dict[int,str]] = None) -> List[UIElement]:
    out = []
    for d in dets or []:
        label = int(d.get("label", -1))
        name = label_map.get(label, f"label:{label}") if label_map else f"label:{label}"
        out.append(UIElement(
            source="det",
            text=name,
            box=[int(v) for v in d.get("box",[0,0,0,0])],
            score=normalize_score(d.get("score", 0.0)),
            role=None
        ))
    return out

# If you don’t have the A11y bridge yet, A11Y_BRIDGE_URL stays empty and that path is a no‑op
def a11y_snapshot() -> List[UIElement]:
    """Optional: query CDP/BiDi bridge for the Accessibility tree."""
    if not A11Y_URL:
        return []
    try:
        # Example contract: GET /ax/snapshot -> [{name, role, box:[x1,y1,x2,y2], score}]
        r = requests.get(f"{A11Y_URL}/ax/snapshot", timeout=0.6)
        r.raise_for_status()
        nodes = r.json()
    except Exception:
        return []
    out = []
    for n in nodes:
        out.append(UIElement(
            source="ax",
            text=n.get("name",""),
            box=[int(v) for v in n.get("box",[0,0,0,0])],
            score=normalize_score(n.get("score", 0.95)),  # AX is usually high-confidence
            role=n.get("role")
        ))
    return out

def fuse_observation(img_w: int, img_h: int,
                     ocr_items: List[Dict[str,Any]],
                     dets: List[Dict[str,Any]],
                     label_map: Optional[Dict[int,str]] = None) -> Observation:
    elems = []
    elems += ocr_to_elements(ocr_items)
    elems += det_to_elements(dets, label_map=label_map)
    elems += a11y_snapshot()
    return Observation(img_w=img_w, img_h=img_h, elements=elems)
