import argparse
import base64
import json
import mimetypes
from pathlib import Path


def _latest_dump_json(dumps_dir: Path) -> Path:
    dumps = sorted(dumps_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not dumps:
        raise SystemExit(f"No dump .json files found in {dumps_dir}")
    return dumps[-1]


def _guess_mime(p: Path, fallback: str = "image/jpeg") -> str:
    mt, _ = mimetypes.guess_type(str(p))
    return mt or fallback


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert a TaskPlanner dump into a replayable payload.json.")
    ap.add_argument("--dumps-dir", default="planner-dumps", help="Directory containing dumps from PLANNER_DUMP_REQUESTS_DIR")
    ap.add_argument("--dump-json", default="", help="Specific dump json file (defaults to newest)")
    ap.add_argument("--out", default="planner_payload.json", help="Output payload JSON path")
    ap.add_argument("--embed-screenshot", action="store_true", help="Embed screenshot bytes into screenshot_b64")
    ap.add_argument("--as-data-url", action="store_true", help="When embedding, store as data:<mime>;base64,...")
    args = ap.parse_args()

    dumps_dir = Path(args.dumps_dir)
    if not dumps_dir.exists():
        raise SystemExit(f"dumps-dir not found: {dumps_dir}")

    dump_json = Path(args.dump_json) if args.dump_json else _latest_dump_json(dumps_dir)
    if not dump_json.exists():
        raise SystemExit(f"dump-json not found: {dump_json}")

    obj = json.loads(dump_json.read_text(encoding="utf-8"))
    screenshot_path = obj.get("screenshot_path")

    if args.embed_screenshot:
        if not screenshot_path:
            raise SystemExit("Dump JSON has no screenshot_path to embed.")
        sp = dumps_dir / str(screenshot_path)
        if not sp.exists():
            raise SystemExit(f"screenshot file not found: {sp}")
        mime = (obj.get("screenshot_mime") or _guess_mime(sp)).strip() or _guess_mime(sp)
        b64 = base64.b64encode(sp.read_bytes()).decode("ascii")
        obj["screenshot_mime"] = mime
        obj["screenshot_b64"] = f"data:{mime};base64,{b64}" if args.as_data_url else b64

    # Remove helper field used only for dumps
    obj.pop("screenshot_path", None)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} from {dump_json}")
    if screenshot_path:
        print(f"Screenshot file: {dumps_dir / str(screenshot_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

