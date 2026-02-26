# agent/src/interrupts.py
from __future__ import annotations
from typing import List, Dict, Any
import re
from ui_core import Observation

# simple patterns; extend freely
MODAL_TERMS = [
    r"\b(allow|block|accept|agree|consent|got it|ok|okay|close|dismiss|not now|continue|skip)\b",
    r"\b(ad|advert|sponsored)\b",
    r"\b(update available|privacy|cookies|gdpr)\b",
]
COUNTDOWN = re.compile(r"\b(\d{1,2})\s*s(ec|econds)?\b", re.I)

def detect_interrupt(obs: Observation) -> Dict[str, Any]:
    """
    Returns {"type": "modal"|"ad"|"cookies"|..., "why": "..."} or {} if none.
    Uses OCR text & (optionally) AX roles.
    """
    txts = []
    roles = []
    for e in obs.elements:
        if e.source in ("ocr","ax"):
            name = (e.text or "").lower()
            if name: txts.append(name)
        if e.source == "ax" and e.role:
            roles.append(e.role.lower())

    txt = " ".join(txts)
    for rx in MODAL_TERMS:
        if re.search(rx, txt, re.I):
            return {"type": "modal", "why": f"matched '{rx}'"}
    if COUNTDOWN.search(txt):
        return {"type":"countdown", "why":"found seconds countdown"}
    if any(r in roles for r in ("dialog","alertdialog")):
        return {"type":"modal", "why":"AX role=dialog"}
    return {}
