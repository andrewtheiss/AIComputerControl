from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ui_vision_common.candidate_graph import (
    build_candidate_graph,
    build_candidates as shared_build_candidates,
    infer_allowed_actions as shared_infer_allowed_actions,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_planner_dumps(root: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for path in sorted(root.glob("*.json")):
        try:
            yield path, load_json(path)
        except Exception:
            continue


def load_screenshot_b64(dump_path: Path, payload: Dict[str, Any]) -> str:
    shot = str(payload.get("screenshot_b64", "") or "").strip()
    if shot and shot.lower() != "null":
        return shot
    screenshot_path = str(payload.get("screenshot_path", "") or "").strip()
    if not screenshot_path:
        return ""
    shot_file = dump_path.parent / screenshot_path
    if not shot_file.exists():
        return ""
    return base64.b64encode(shot_file.read_bytes()).decode("ascii")


def infer_allowed_actions(element: Dict[str, Any]) -> List[str]:
    return shared_infer_allowed_actions(element)


def build_candidates(ui_elements: Iterable[Dict[str, Any]], limit: int = 80) -> List[Dict[str, Any]]:
    return shared_build_candidates(ui_elements, limit=limit)


def build_candidate_graph_for_payload(payload: Dict[str, Any], candidate_limit: int = 80) -> List[Dict[str, Any]]:
    viewport = None
    screenshot_path = str(payload.get("screenshot_path", "") or "").strip()
    if screenshot_path:
        _ = screenshot_path
    return build_candidate_graph(payload.get("ui_elements") or [], limit=candidate_limit, viewport=viewport)


def derive_instruction(payload: Dict[str, Any]) -> str:
    history = payload.get("task_history") or []
    if history:
        last = history[-1]
        action = str(last.get("action", "") or "")
        params = last.get("parameters") or {}
        if action in {"click_text", "click_any_text"}:
            target = params.get("regex") or params.get("fuzzy_text") or params.get("patterns") or params.get("texts")
            if target:
                if isinstance(target, list):
                    target = target[0]
                return f"click the UI element matching {target}"
        if action == "click_box":
            return "click the intended target near the previously selected box"
    resolve_targets = (payload.get("current_state") or {}).get("visible_resolve_targets") or []
    if resolve_targets:
        return f"click {resolve_targets[0]}"
    goal = str(payload.get("goal", "") or "").strip()
    if goal:
        return goal
    return "click the most relevant actionable UI target"


def post_json(url: str, payload: Dict[str, Any], timeout_s: float = 30.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    successes = [row for row in rows if row.get("status") == "ok"]
    latencies = [int(row.get("latency_ms") or 0) for row in successes if row.get("latency_ms") is not None]
    summary = {
        "samples": len(rows),
        "successes": len(successes),
        "failures": len(rows) - len(successes),
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2) if latencies else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)


def replay_ocr(dumps_dir: Path, endpoint: str, out_dir: Path, limit: int) -> int:
    rows: List[Dict[str, Any]] = []
    for path, payload in iter_planner_dumps(dumps_dir):
        if len(rows) >= limit:
            break
        shot = load_screenshot_b64(path, payload)
        if not shot:
            continue
        t0 = time.perf_counter()
        try:
            resp = post_json(endpoint, {"image_b64": shot, "return_level": "both"})
            rows.append(
                {
                    "dump_file": path.name,
                    "status": "ok",
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                    "words": len(resp.get("words") or []),
                    "lines": len(resp.get("lines") or []),
                }
            )
        except Exception as exc:
            rows.append({"dump_file": path.name, "status": "error", "error": str(exc)})
    write_jsonl(out_dir / "ocr_replay.jsonl", rows)
    write_summary(out_dir / "ocr_summary.json", rows)
    return 0


def replay_targets(dumps_dir: Path, endpoint: str, out_dir: Path, limit: int, candidate_limit: int) -> int:
    rows: List[Dict[str, Any]] = []
    for path, payload in iter_planner_dumps(dumps_dir):
        if len(rows) >= limit:
            break
        shot = load_screenshot_b64(path, payload)
        candidate_graph = build_candidate_graph(payload.get("ui_elements") or [], limit=candidate_limit)
        candidates = shared_build_candidates(payload.get("ui_elements") or [], limit=candidate_limit)
        if not shot or not candidates:
            continue
        request_payload = {
            "instruction": derive_instruction(payload),
            "history": [str(item.get("action", "")) for item in (payload.get("task_history") or [])[-4:]],
            "screenshot_b64": shot,
            "candidates": candidates,
            "top_k": 5,
        }
        t0 = time.perf_counter()
        try:
            resp = post_json(endpoint, request_payload)
            final_pred = resp.get("final_prediction") or {}
            rows.append(
                {
                    "dump_file": path.name,
                    "status": "ok",
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                    "instruction": request_payload["instruction"],
                    "candidate_count": len(candidates),
                    "candidate_graph_count": len(candidate_graph),
                    "final_candidate_id": final_pred.get("candidate_id"),
                    "final_score": final_pred.get("score"),
                }
            )
        except Exception as exc:
            rows.append({"dump_file": path.name, "status": "error", "error": str(exc)})
    write_jsonl(out_dir / "target_replay.jsonl", rows)
    write_summary(out_dir / "target_summary.json", rows)
    return 0


def export_corpus(dumps_dir: Path, out_dir: Path, limit: int, candidate_limit: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    screenshot_dir = out_dir / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in iter_planner_dumps(dumps_dir):
        if len(rows) >= limit:
            break
        shot = load_screenshot_b64(path, payload)
        if not shot:
            continue
        screenshot_path = screenshot_dir / f"{path.stem}.jpg"
        try:
            screenshot_path.write_bytes(base64.b64decode(shot, validate=False))
        except Exception:
            continue
        rows.append(
            {
                "dump_file": path.name,
                "goal": payload.get("goal", ""),
                "instruction": derive_instruction(payload),
                "screenshot_path": str(screenshot_path),
                "candidate_count": len(shared_build_candidates(payload.get("ui_elements") or [], limit=candidate_limit)),
            }
        )
    write_jsonl(out_dir / "corpus.jsonl", rows)
    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Replay planner dumps against OCR or target ensemble services.")
    parser.add_argument("command", choices=["export-corpus", "replay-ocr", "replay-targets"])
    parser.add_argument("--dumps-dir", default="planner-dumps")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--out-dir", default="artifacts/ui-vision-harness")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--candidate-limit", type=int, default=80)
    args = parser.parse_args(argv)

    dumps_dir = Path(args.dumps_dir)
    out_dir = Path(args.out_dir)

    if args.command == "export-corpus":
        return export_corpus(dumps_dir, out_dir, limit=args.limit, candidate_limit=args.candidate_limit)
    if args.command == "replay-ocr":
        if not args.endpoint:
            raise SystemExit("--endpoint is required for replay-ocr")
        return replay_ocr(dumps_dir, args.endpoint, out_dir, limit=args.limit)
    if args.command == "replay-targets":
        if not args.endpoint:
            raise SystemExit("--endpoint is required for replay-targets")
        return replay_targets(dumps_dir, args.endpoint, out_dir, limit=args.limit, candidate_limit=args.candidate_limit)
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
