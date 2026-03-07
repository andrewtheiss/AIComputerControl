from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _extract_system_prompt_from_taskplanner(repo_root: Path) -> str:
    """
    Best-effort extraction of SYSTEM_PROMPT triple-quoted string from taskPlanner/main.py.
    """
    src = _read_text(repo_root / "taskPlanner" / "main.py")
    i = src.find("SYSTEM_PROMPT")
    if i == -1:
        raise SystemExit("Could not find SYSTEM_PROMPT in taskPlanner/main.py")
    # find first triple quote after SYSTEM_PROMPT
    q1 = src.find('"""', i)
    if q1 == -1:
        raise SystemExit("Could not find opening triple quotes for SYSTEM_PROMPT")
    q2 = src.find('"""', q1 + 3)
    if q2 == -1:
        raise SystemExit("Could not find closing triple quotes for SYSTEM_PROMPT")
    return src[q1 + 3 : q2]


def _data_url_for_image(img_path: Path, mime: str) -> str:
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _guess_mime(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suf == ".png":
        return "image/png"
    return "application/octet-stream"


def _jpeg_size(p: Path) -> Tuple[int, int]:
    data = p.read_bytes()
    if data[0:2] != b"\xFF\xD8":
        raise ValueError("not jpeg")
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        # SOF0/SOF2
        if marker in (0xC0, 0xC2):
            h = int.from_bytes(data[i + 5 : i + 7], "big")
            w = int.from_bytes(data[i + 7 : i + 9], "big")
            return w, h
        # skip segment
        seglen = int.from_bytes(data[i + 2 : i + 4], "big")
        i += 2 + seglen
    raise ValueError("could not parse jpeg size")


def _png_size(p: Path) -> Tuple[int, int]:
    data = p.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not png")
    w = int.from_bytes(data[16:20], "big")
    h = int.from_bytes(data[20:24], "big")
    return w, h


def image_size(p: Path) -> Tuple[int, int]:
    suf = p.suffix.lower()
    if suf in (".jpg", ".jpeg"):
        return _jpeg_size(p)
    if suf == ".png":
        return _png_size(p)
    raise ValueError(f"unknown image type: {p.suffix}")


def make_user_prompt(dump: Dict[str, Any]) -> str:
    goal = dump.get("goal", "")
    actions = ", ".join(dump.get("available_actions") or [])
    hist = dump.get("task_history") or []
    hist_lines = []
    for h in hist[-5:]:
        hist_lines.append(f"- {h.get('action')}({h.get('parameters')}) => {((h.get('result') or {}).get('status'))}")
    ocr = dump.get("ocr_results") or []
    ocr_lines = [f"- {r.get('text')} @ {r.get('box')} (conf={int(r.get('conf', 0))})" for r in ocr[:300]]
    elems = dump.get("ui_elements") or []
    elem_lines = [
        f"- [{e.get('source')}{'/' + e.get('role') if e.get('role') else ''}] {e.get('text')} @ {e.get('box')} (score={float(e.get('score', 0.0)):.2f})"
        for e in elems[:200]
    ]
    return (
        f"GOAL:\n{goal}\n\n"
        f"AVAILABLE_ACTIONS: {actions}\n\n"
        f"RECENT_HISTORY:\n{chr(10).join(hist_lines) if hist_lines else 'No prior actions.'}\n\n"
        f"CURRENT_OCR:\n{chr(10).join(ocr_lines) if ocr_lines else 'No OCR text.'}\n\n"
        f"UI_ELEMENTS (image-relative boxes):\n{chr(10).join(elem_lines) if elem_lines else 'No UI elements.'}\n\n"
        f"Return strict JSON only."
    )


def _post_json(url: str, body: Dict[str, Any], timeout_s: float = 180.0) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _extract_text_from_openai_like(resp: Dict[str, Any]) -> str:
    # chat.completions
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        pass
    # responses api
    try:
        out = resp.get("output", [])
        if out and out[0].get("content"):
            # pick first text segment
            for c in out[0]["content"]:
                if c.get("type") in ("output_text", "text"):
                    return c.get("text", "")
    except Exception:
        pass
    return json.dumps(resp)[:2000]


def _parse_model_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        # fallback: extract {...}
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s : e + 1])
        raise


def _match_best_ocr(
    ocr: List[Dict[str, Any]],
    pattern: str,
    prefer_bold: bool,
    nth: int,
) -> Optional[Dict[str, Any]]:
    rx = re.compile(pattern, re.I)
    matches = []
    for w in ocr:
        t = str(w.get("text", ""))
        if not rx.search(t):
            continue
        box = w.get("box") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [int(v) for v in box]
        height = max(0, y2 - y1)
        conf = float(w.get("conf", 0.0))
        score = 100.0 + (height * 0.05 if prefer_bold else 0.0) + (conf * 0.001)
        matches.append((score, w))
    if not matches:
        return None
    matches.sort(key=lambda x: -x[0])
    idx = min(int(nth or 0), len(matches) - 1)
    return matches[idx][1]


def decision_to_overlay(
    decision: Dict[str, Any],
    dump: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return overlay primitives for HTML SVG rendering.
    """
    action = decision.get("action")
    params = decision.get("parameters") or {}
    ocr = dump.get("ocr_results") or []
    out: Dict[str, Any] = {"action": action, "prims": []}

    def rect(box, color="lime", width=3, label=""):
        x1, y1, x2, y2 = [int(v) for v in box]
        out["prims"].append({"type": "rect", "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "color": color, "width": width, "label": label})

    def cross(x, y, color="red"):
        out["prims"].append({"type": "cross", "x": int(x), "y": int(y), "color": color})

    if action == "click_box":
        box = params.get("box") or params.get("bbox")
        if box:
            rect(box, color="lime", label="click_box")
            x1, y1, x2, y2 = [int(v) for v in box]
            cross((x1 + x2) // 2, (y1 + y2) // 2)
        return out

    if action == "click_text":
        pat = params.get("regex") or params.get("text") or ""
        w = _match_best_ocr(ocr, pat, bool(params.get("prefer_bold", False)), int(params.get("nth", 0)))
        if w:
            box = w.get("box") or [0, 0, 0, 0]
            rect(box, color="cyan", label=f"click_text: {pat}")
            x1, y1, x2, y2 = [int(v) for v in box]
            cross((x1 + x2) // 2, (y1 + y2) // 2)
        return out

    if action == "click_any_text":
        patterns = params.get("patterns") or params.get("texts") or []
        prefer_bold = bool(params.get("prefer_bold", True))
        nth = int(params.get("nth", 0))
        for pat in patterns:
            w = _match_best_ocr(ocr, str(pat), prefer_bold, nth)
            if w:
                box = w.get("box") or [0, 0, 0, 0]
                rect(box, color="cyan", label=f"click_any_text: {pat}")
                x1, y1, x2, y2 = [int(v) for v in box]
                cross((x1 + x2) // 2, (y1 + y2) // 2)
                break
        return out

    if action == "click_near_text":
        anchor = params.get("anchor_regex") or params.get("anchor") or ""
        dx = int(params.get("dx", 0))
        dy = int(params.get("dy", 0))
        w = _match_best_ocr(ocr, str(anchor), prefer_bold=False, nth=0)
        if w:
            box = w.get("box") or [0, 0, 0, 0]
            rect(box, color="yellow", label=f"anchor: {anchor}")
            x1, y1, x2, y2 = [int(v) for v in box]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cross(cx + dx, cy + dy, color="red")
        return out

    return out


def render_compare_html(
    *,
    out_path: Path,
    screenshot_rel: str,
    w: int,
    h: int,
    rows: List[Dict[str, Any]],
    models: List[str],
) -> None:
    # rows: list of {model, decision_text, decision_obj, overlay_prims}
    cols = []
    for m in models:
        r = next((x for x in rows if x["model"] == m), None)
        if not r:
            cols.append(f"<div class='cell'><div class='m'>{m}</div><pre>(no result)</pre></div>")
            continue
        prims = r["overlay"]["prims"]
        svg_parts = [f"<svg viewBox='0 0 {w} {h}' class='ov'>"]
        for p in prims:
            if p["type"] == "rect":
                svg_parts.append(
                    f"<rect x='{p['x']}' y='{p['y']}' width='{p['w']}' height='{p['h']}' "
                    f"fill='none' stroke='{p['color']}' stroke-width='{p['width']}'/>"
                )
            elif p["type"] == "cross":
                x, y = p["x"], p["y"]
                svg_parts.append(f"<line x1='{x-10}' y1='{y}' x2='{x+10}' y2='{y}' stroke='{p['color']}' stroke-width='3'/>")
                svg_parts.append(f"<line x1='{x}' y1='{y-10}' x2='{x}' y2='{y+10}' stroke='{p['color']}' stroke-width='3'/>")
        svg_parts.append("</svg>")
        svg = "".join(svg_parts)

        decision_obj = r.get("decision_obj") or {}
        pretty = json.dumps(decision_obj, indent=2, ensure_ascii=False)[:4000]
        cols.append(
            f"""
<div class="cell">
  <div class="m">{m}</div>
  <div class="imgwrap">
    <img src="{screenshot_rel}" alt="screenshot"/>
    {svg}
  </div>
  <pre>{pretty}</pre>
</div>
"""
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>Model compare</title>
<style>
body {{ font-family: system-ui, Arial, sans-serif; margin: 16px; }}
.grid {{ display: grid; grid-template-columns: repeat({len(models)}, 1fr); gap: 12px; }}
.cell {{ border: 1px solid #ddd; border-radius: 10px; padding: 10px; background: #fff; }}
.m {{ font-weight: 700; margin-bottom: 6px; }}
.imgwrap {{ position: relative; width: 100%; }}
.imgwrap img {{ width: 100%; height: auto; border-radius: 8px; display: block; }}
.ov {{ position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; }}
pre {{ font-size: 12px; overflow: auto; max-height: 240px; background: #fafafa; border: 1px solid #eee; padding: 8px; border-radius: 8px; }}
</style></head>
<body>
<h2>Compare models (same snapshot)</h2>
<div class="grid">
{''.join(cols)}
</div>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare multiple LM Studio models on the same planner dump and render HTML overlays.")
    ap.add_argument("--dump", required=True, help="Path to planner-dumps/*.json (sanitized dump with screenshot_path)")
    ap.add_argument("--out", default="compare.html", help="Output HTML path")
    ap.add_argument("--base-url", default="http://127.0.0.1:1234/v1/chat/completions", help="LM Studio chat completions URL")
    ap.add_argument("--models", required=True, help="Comma-separated model names")
    args = ap.parse_args()

    dump_path = Path(args.dump)
    dump_dir = dump_path.parent
    dump = json.loads(_read_text(dump_path))
    screenshot_path = dump.get("screenshot_path")
    if not screenshot_path:
        raise SystemExit("dump has no screenshot_path (enable PLANNER_DUMP_REQUESTS_DIR)")
    img_path = (dump_dir / screenshot_path).resolve()
    if not img_path.exists():
        raise SystemExit(f"screenshot not found: {img_path}")

    w, h = image_size(img_path)
    mime = (dump.get("screenshot_mime") or _guess_mime(img_path)).strip() or _guess_mime(img_path)
    data_url = _data_url_for_image(img_path, mime)

    repo_root = Path.cwd()
    system_prompt = _extract_system_prompt_from_taskplanner(repo_root)
    user_prompt = make_user_prompt(dump)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []
    for m in models:
        body = {
            "model": m,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }
        resp = _post_json(args.base_url, body)
        txt = _extract_text_from_openai_like(resp)
        decision_obj = _parse_model_json(txt)
        overlay = decision_to_overlay(decision_obj, dump)
        rows.append({"model": m, "decision_obj": decision_obj, "overlay": overlay})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # place html next to dump so relative img path works
    screenshot_rel = os.path.relpath(str(img_path), start=str(out_path.parent))
    render_compare_html(out_path=out_path, screenshot_rel=screenshot_rel, w=w, h=h, rows=rows, models=models)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

