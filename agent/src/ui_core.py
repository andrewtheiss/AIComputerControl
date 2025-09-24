# agent/src/ui_core.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

@dataclass
class UIElement:
    source: Literal["ocr","det","ax"]
    text: str
    box: List[int]                    # [x1,y1,x2,y2]
    score: float                      # 0..1 (normalize upstream)
    role: Optional[str] = None        # e.g., "button","textbox" (from A11y)
    extras: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Observation:
    img_w: int
    img_h: int
    elements: List[UIElement]
