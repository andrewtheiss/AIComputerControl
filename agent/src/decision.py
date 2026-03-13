from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Candidate:
    """
    A concrete action proposal the executor can attempt.
    - action: "click_box" | "keys"
    - params: { box: [x1,y1,x2,y2] } or { keys: ["ctrl+Return", ...] }
    - conf: confidence score 0..1 used for arbitration
    - why: short rationale for logging/traceability
    """
    action: str
    params: Dict[str, Any]
    conf: float
    why: str


def arbitrate(candidates: List[Candidate], k: int = 6) -> List[Candidate]:
    """Rank candidates by confidence and simple heuristics, return top-k."""
    # Prefer higher conf, then prefer click_box over keys as tie-breaker
    def _rank_key(c: Candidate):
        return (
            -float(c.conf or 0.0),
            0 if c.action == "click_box" else 1,
        )

    uniq: List[Candidate] = []
    seen = set()
    for c in candidates:
        sig = (c.action, tuple(c.params.get("box", [])) if c.action == "click_box" else tuple(c.params.get("keys", [])))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(c)

    uniq.sort(key=_rank_key)
    return uniq[:k]


class BaseProposer:
    def propose(self, intent: str, obs: Any) -> List[Candidate]:
        raise NotImplementedError

    @staticmethod
    def _add_clicks_for_texts(
        obs: Any,
        text_patterns: List[re.Pattern],
        base_conf: float = 0.6,
        role_bonus: float = 0.15,
        source_bonus: Dict[str, float] = {"ax": 0.15, "ocr": 0.0, "det": -0.05},
        why_label: str = ""
    ) -> List[Candidate]:
        cands: List[Candidate] = []
        for el in getattr(obs, "elements", []) or []:
            text = (el.text or "").strip()
            if not text:
                continue
            if not any(rx.search(text) for rx in text_patterns):
                continue
            conf = base_conf + float(getattr(el, "score", 0.0)) * 0.25
            # Prefer A11y role/button a bit more
            if getattr(el, "role", None) in ("button", "link"):
                conf += role_bonus
            conf += source_bonus.get(getattr(el, "source", "ocr"), 0.0)
            cands.append(Candidate(
                action="click_box",
                params={"box": [int(v) for v in el.box]},
                conf=max(0.0, min(1.0, conf)),
                why=f"{why_label} '{text}' via {getattr(el,'source','ocr')}"
            ))
        return cands


class ComposeProposer(BaseProposer):
    def propose(self, intent: str, obs: Any) -> List[Candidate]:
        if intent != "compose":
            return []

        # Common UI labels
        pats = [
            re.compile(r"^compose$", re.I),
            re.compile(r"^new( message)?$", re.I),
            re.compile(r"^new mail$", re.I),
            re.compile(r"^write( message)?$", re.I),
            re.compile(r"^start new message$", re.I),
        ]

        cands: List[Candidate] = []
        cands += self._add_clicks_for_texts(obs, pats, base_conf=0.62, why_label="compose")

        # Keyboard fallbacks (lower confidence, generic)
        # ctrl+n is common across many clients; Gmail uses 'c' but we keep low conf.
        cands.append(Candidate(action="keys", params={"keys": ["ctrl+n"]}, conf=0.50, why="compose via ctrl+n"))
        cands.append(Candidate(action="keys", params={"keys": ["c"]}, conf=0.35, why="compose via 'c' (Gmail)"))
        return cands


class SendProposer(BaseProposer):
    def propose(self, intent: str, obs: Any) -> List[Candidate]:
        if intent != "send":
            return []
        pats = [
            re.compile(r"^send$", re.I),
            re.compile(r"^send & archive$", re.I),
            re.compile(r"^send now$", re.I),
        ]
        cands: List[Candidate] = []
        cands += self._add_clicks_for_texts(obs, pats, base_conf=0.65, why_label="send")
        # Keyboard shortcuts (Gmail/clients): ctrl+Enter; include ctrl+Shift+Enter for variants
        cands.append(Candidate(action="keys", params={"keys": ["ctrl+Return"]}, conf=0.80, why="send via ctrl+Enter"))
        cands.append(Candidate(action="keys", params={"keys": ["ctrl+Shift+Return"]}, conf=0.60, why="send via ctrl+Shift+Enter"))
        return cands


class DismissModalProposer(BaseProposer):
    def propose(self, intent: str, obs: Any) -> List[Candidate]:
        # Offer as a general safety net for any intent
        pats = [
            re.compile(r"^ok$", re.I),
            re.compile(r"^close$", re.I),
            re.compile(r"^cancel$", re.I),
            re.compile(r"^dismiss$", re.I),
            re.compile(r"^got it$", re.I),
            re.compile(r"^no thanks$", re.I),
            re.compile(r"^not now$", re.I),
            re.compile(r"^allow$", re.I),
            re.compile(r"^yes$", re.I),
            re.compile(r"^continue$", re.I),
        ]
        cands: List[Candidate] = []
        cands += self._add_clicks_for_texts(obs, pats, base_conf=0.45, why_label="dismiss")
        return cands
