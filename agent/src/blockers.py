import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _union_box(items: List[Dict[str, Any]]) -> Optional[List[int]]:
    if not items:
        return None
    xs1 = [int(item["box"][0]) for item in items if isinstance(item.get("box"), list) and len(item.get("box")) == 4]
    ys1 = [int(item["box"][1]) for item in items if isinstance(item.get("box"), list) and len(item.get("box")) == 4]
    xs2 = [int(item["box"][2]) for item in items if isinstance(item.get("box"), list) and len(item.get("box")) == 4]
    ys2 = [int(item["box"][3]) for item in items if isinstance(item.get("box"), list) and len(item.get("box")) == 4]
    if not xs1 or not ys1 or not xs2 or not ys2:
        return None
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def _unique_targets(targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for target in targets:
        key = (
            _norm(str(target.get("label", "") or target.get("text", "") or "")),
            tuple(int(v) for v in (target.get("box") or [0, 0, 0, 0])),
            str(target.get("kind", "") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(target)
    return out


def _match_groups(
    text_items: List[Dict[str, Any]],
    groups: List[List[str]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    matched_items: List[Dict[str, Any]] = []
    evidence: List[str] = []
    for group in groups:
        group_hit = None
        for item in text_items:
            txt = str(item.get("text", "") or "")
            if any(re.search(pattern, txt, re.I) for pattern in group):
                group_hit = item
                break
        if group_hit is None:
            return [], []
        matched_items.append(group_hit)
        evidence.append(str(group_hit.get("text", "") or ""))
    return matched_items, evidence


def _find_targets(
    text_items: List[Dict[str, Any]],
    specs: List[Tuple[str, List[str], bool]],
) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for kind, patterns, dual_purpose in specs:
        for item in text_items:
            txt = str(item.get("text", "") or "")
            if not txt:
                continue
            if any(re.search(pattern, txt, re.I) for pattern in patterns):
                targets.append(
                    {
                        "kind": kind,
                        "text": txt,
                        "label": txt,
                        "box": [int(v) for v in item.get("box", [0, 0, 0, 0])],
                        "level": str(item.get("level", "word") or "word"),
                        "area": max(1, int(item.get("box", [0, 0, 1, 1])[2]) - int(item.get("box", [0, 0, 1, 1])[0])) * max(1, int(item.get("box", [0, 0, 1, 1])[3]) - int(item.get("box", [0, 0, 1, 1])[1])),
                        "dual_purpose": bool(dual_purpose),
                    }
                )
    return _unique_targets(targets)


def _descriptor(
    *,
    blocker_class: str,
    scope: str,
    priority: int,
    confidence: float,
    evidence: List[str],
    matched_items: List[Dict[str, Any]],
    resolve_targets: Optional[List[Dict[str, Any]]] = None,
    allow_page_click_through: bool = False,
    suggested_strategies: Optional[List[str]] = None,
    legacy_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    bbox = _union_box(matched_items)
    signature_payload = {
        "class": blocker_class,
        "scope": scope,
        "bbox": [int(v / 20) * 20 for v in bbox] if bbox else None,
        "evidence": [_norm(x) for x in evidence[:3]],
    }
    signature = hashlib.sha1(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"{blocker_class}:{signature}",
        "signature": signature,
        "class": blocker_class,
        "scope": scope,
        "priority": priority,
        "confidence": round(float(confidence), 2),
        "source": "ocr",
        "bbox": bbox,
        "evidence": evidence[:4],
        "resolve_targets": _unique_targets(resolve_targets or []),
        "allow_page_click_through": bool(allow_page_click_through),
        "suggested_strategies": list(suggested_strategies or []),
        "legacy_tags": list(legacy_tags or []),
        "summary": f"{blocker_class} in {scope}",
    }


def classify_blockers(text_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = [item for item in (text_items or []) if str(item.get("text", "") or "").strip()]
    blockers: List[Dict[str, Any]] = []

    session_matches, session_evidence = _match_groups(
        items,
        [
            [r"\bsession restore\b", r"\babout:sessionrestore\b", r"\brestore previous session\b"],
        ],
    )
    if session_matches:
        blockers.append(
            _descriptor(
                blocker_class="browser_session_restore",
                scope="browser_page",
                priority=100,
                confidence=0.98,
                evidence=session_evidence,
                matched_items=session_matches,
                resolve_targets=_find_targets(
                    items,
                    [
                        ("click_visible_resolve_target", [r"\bstart new session\b", r"\bstart new\b", r"\bnew session\b"], True),
                        ("click_visible_resolve_target", [r"\brestore session\b", r"\brestore previous session\b"], True),
                        ("click_visible_resolve_target", [r"\bclose tab\b", r"\bclose\b"], False),
                    ],
                ),
                allow_page_click_through=False,
                suggested_strategies=[
                    "click_visible_resolve_target",
                    "open_url",
                    "escape",
                ],
            )
        )

    permission_matches, permission_evidence = _match_groups(
        items,
        [
            [r"\ballow\b", r"\bblock\b", r"\bdon'?t allow\b", r"\bnot now\b"],
            [r"notification", r"location", r"camera", r"microphone", r"clipboard"],
        ],
    )
    if permission_matches:
        blockers.append(
            _descriptor(
                blocker_class="browser_permission_prompt",
                scope="browser_chrome",
                priority=95,
                confidence=0.94,
                evidence=permission_evidence,
                matched_items=permission_matches,
                resolve_targets=_find_targets(
                    items,
                    [
                        ("click_visible_resolve_target", [r"\ballow\b"], False),
                        ("click_visible_resolve_target", [r"\bblock\b", r"\bdon'?t allow\b"], False),
                        ("click_visible_resolve_target", [r"\bnot now\b", r"\bcancel\b"], False),
                    ],
                ),
                allow_page_click_through=False,
                suggested_strategies=[
                    "click_visible_resolve_target",
                    "escape",
                ],
            )
        )

    cookie_matches, cookie_evidence = _match_groups(
        items,
        [
            [r"\bcookies?\b", r"\bcookie policy\b", r"\bprivacy settings\b"],
        ],
    )
    if cookie_matches:
        blockers.append(
            _descriptor(
                blocker_class="cookie_banner",
                scope="page_overlay",
                priority=80,
                confidence=0.88,
                evidence=cookie_evidence,
                matched_items=cookie_matches,
                resolve_targets=_find_targets(
                    items,
                    [
                        ("click_visible_resolve_target", [r"\baccept\b", r"\bagree\b", r"\bok\b"], False),
                        ("click_visible_resolve_target", [r"\breject\b", r"\bdecline\b"], False),
                        ("click_visible_resolve_target", [r"\bclose\b", r"\bnot now\b"], False),
                    ],
                ),
                allow_page_click_through=False,
                suggested_strategies=[
                    "click_visible_resolve_target",
                    "click_away_page",
                    "escape",
                ],
                legacy_tags=["blocker:modal_like"],
            )
        )

    modal_matches, modal_evidence = _match_groups(
        items,
        [
            [r"\bdialog\b", r"\bmodal\b", r"\bpopup\b", r"\bpop-up\b"],
        ],
    )
    if modal_matches:
        blockers.append(
            _descriptor(
                blocker_class="modal_dialog",
                scope="modal",
                priority=78,
                confidence=0.8,
                evidence=modal_evidence,
                matched_items=modal_matches,
                resolve_targets=_find_targets(
                    items,
                    [
                        ("click_visible_resolve_target", [r"\bok\b", r"\bclose\b", r"\bcancel\b", r"\bnot now\b"], False),
                    ],
                ),
                allow_page_click_through=False,
                suggested_strategies=[
                    "click_visible_resolve_target",
                    "escape",
                ],
                legacy_tags=["blocker:modal_like"],
            )
        )

    interstitial_matches, interstitial_evidence = _match_groups(
        items,
        [
            [r"\bsomething went wrong\b", r"\bproblem loading page\b", r"\btry again\b", r"\btrouble restoring\b", r"\bwarning\b", r"\bsecure connection failed\b"],
        ],
    )
    if interstitial_matches:
        blockers.append(
            _descriptor(
                blocker_class="browser_interstitial_error",
                scope="browser_page",
                priority=88,
                confidence=0.86,
                evidence=interstitial_evidence,
                matched_items=interstitial_matches,
                resolve_targets=_find_targets(
                    items,
                    [
                        ("click_visible_resolve_target", [r"\btry again\b", r"\breload\b", r"\bgo back\b"], False),
                        ("click_visible_resolve_target", [r"\baccept the risk\b", r"\badvanced\b"], False),
                    ],
                ),
                allow_page_click_through=False,
                suggested_strategies=[
                    "click_visible_resolve_target",
                    "open_url",
                ],
            )
        )

    suggestion_matches, suggestion_evidence = _match_groups(
        items,
        [
            [r"\bsearch with google\b", r"\bswitch to tab\b", r"\bfirefox suggest\b", r"\bgoogle search\b", r"\bsearch bookmarks\b", r"\bvisit\b"],
        ],
    )
    if suggestion_matches:
        blockers.append(
            _descriptor(
                blocker_class="browser_url_suggestion_dropdown",
                scope="browser_chrome",
                priority=70,
                confidence=0.92,
                evidence=suggestion_evidence,
                matched_items=suggestion_matches,
                resolve_targets=[],
                allow_page_click_through=True,
                suggested_strategies=[
                    "escape",
                    "click_away_page",
                    "click_visible_page_target",
                    "open_url",
                    "refocus_urlbar",
                ],
                legacy_tags=["blocker:browser_suggestion_overlay"],
            )
        )

    blockers.sort(key=lambda blocker: (-int(blocker.get("priority", 0)), -float(blocker.get("confidence", 0.0))))
    return blockers
