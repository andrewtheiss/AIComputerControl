import argparse
import base64
import json
import mimetypes
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _guess_mime(p: Path) -> str:
    mt, _ = mimetypes.guess_type(str(p))
    return mt or "image/jpeg"


def _post_json(url: str, body: dict, api_key: str | None, timeout_s: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            try:
                return json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                return {"_non_json_response": raw.decode("utf-8", errors="replace")[:4000]}
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code}: {msg[:4000]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a TaskPlanner /v1/actions/next request.")
    ap.add_argument("--url", default="http://localhost:28000/v1/actions/next", help="Planner endpoint URL")
    ap.add_argument("--api-key", default="", help="Planner bearer token (PLANNER_API_KEY)")
    ap.add_argument("--payload", required=True, help="Path to JSON payload (PlannerRequest)")
    ap.add_argument("--screenshot", default="", help="Optional screenshot file to embed (jpg/png)")
    ap.add_argument("--as-data-url", action="store_true", help="Embed screenshot as data:<mime>;base64,... (default: raw b64)")
    ap.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")
    args = ap.parse_args()

    payload_path = Path(args.payload)
    if not payload_path.exists():
        raise SystemExit(f"payload not found: {payload_path}")
    body = json.loads(payload_path.read_text(encoding="utf-8"))

    if args.screenshot:
        sp = Path(args.screenshot)
        if not sp.exists():
            raise SystemExit(f"screenshot not found: {sp}")
        mime = _guess_mime(sp)
        b64 = base64.b64encode(sp.read_bytes()).decode("ascii")
        body["screenshot_mime"] = mime
        body["screenshot_b64"] = f"data:{mime};base64,{b64}" if args.as_data_url else b64

    # Helpful summary before sending
    shot = body.get("screenshot_b64") or ""
    print(
        json.dumps(
            {
                "url": args.url,
                "goal_len": len(body.get("goal") or ""),
                "task_history_n": len(body.get("task_history") or []),
                "ocr_results_n": len(body.get("ocr_results") or []),
                "ui_elements_n": len(body.get("ui_elements") or []),
                "available_actions_n": len(body.get("available_actions") or []),
                "has_screenshot": bool(str(shot).strip()),
                "screenshot_len": len(shot) if isinstance(shot, str) else None,
            },
            indent=2,
        )
    )

    out = _post_json(args.url, body, api_key=(args.api_key or "").strip() or None, timeout_s=args.timeout)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

