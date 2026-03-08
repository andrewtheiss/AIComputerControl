#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


RUN_DIR_RE = re.compile(r"^(?P<agent>.+)-(?P<stamp>\d{8}-\d{6})-(?P<suffix>[0-9a-fA-F]+)$")


@dataclass(frozen=True)
class TraceEntry:
    run_dir: Path
    trace_html: Path
    mtime: float
    batch_stamp: str


def _parse_batch_stamp(run_dir_name: str) -> str:
    match = RUN_DIR_RE.match(run_dir_name)
    return match.group("stamp") if match else ""


def _discover_traces(debug_root: Path) -> List[TraceEntry]:
    entries: List[TraceEntry] = []
    if not debug_root.exists():
        raise SystemExit(f"debug directory not found: {debug_root}")
    if not debug_root.is_dir():
        raise SystemExit(f"debug path is not a directory: {debug_root}")

    for run_dir in sorted(debug_root.iterdir()):
        if not run_dir.is_dir():
            continue
        trace_html = run_dir / "trace.html"
        if not trace_html.is_file():
            continue
        stat = trace_html.stat()
        entries.append(
            TraceEntry(
                run_dir=run_dir,
                trace_html=trace_html,
                mtime=stat.st_mtime,
                batch_stamp=_parse_batch_stamp(run_dir.name),
            )
        )

    return sorted(entries, key=lambda entry: entry.mtime, reverse=True)


def _select_active_traces(entries: List[TraceEntry], recent_seconds: float) -> List[TraceEntry]:
    if not entries:
        return []

    newest = entries[0]
    selected: List[TraceEntry] = []

    # Default behavior: open only traces from the newest batch stamp. This maps to
    # the latest set of sessions that were started together.
    if newest.batch_stamp:
        selected = [entry for entry in entries if entry.batch_stamp == newest.batch_stamp]
    else:
        selected = [newest]

    # Optional widening for manual use. Disabled by default because it can pull in
    # older completed runs that just happen to be close in time.
    if recent_seconds > 0:
        seen = {str(entry.trace_html) for entry in selected}
        for entry in entries:
            if newest.mtime - entry.mtime > recent_seconds:
                continue
            trace_key = str(entry.trace_html)
            if trace_key in seen:
                continue
            selected.append(entry)
            seen.add(trace_key)

    return sorted(selected, key=lambda entry: entry.mtime, reverse=True)


def _wsl_to_windows_path(path: Path) -> str:
    proc = subprocess.run(
        ["wslpath", "-w", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    win_path = proc.stdout.strip()
    if not win_path:
        raise RuntimeError(f"wslpath returned an empty path for {path}")
    return win_path


def _open_in_explorer(paths: Iterable[Path]) -> int:
    opened = 0
    for path in paths:
        win_path = _wsl_to_windows_path(path)
        subprocess.Popen(
            ["explorer.exe", win_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        opened += 1
    return opened


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description="Open the newest agent trace.html files in Explorer."
    )
    ap.add_argument(
        "--root",
        default=str(repo_root / "agent-debug"),
        help="Directory containing per-run debug folders (default: ./agent-debug).",
    )
    ap.add_argument(
        "--recent-seconds",
        type=float,
        default=0.0,
        help="Optionally include additional traces modified within this many seconds of the newest trace (default: disabled).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the traces that would be opened without launching Explorer.",
    )
    args = ap.parse_args()

    debug_root = Path(args.root).expanduser().resolve()
    entries = _discover_traces(debug_root)
    if not entries:
        raise SystemExit(f"no trace.html files found under: {debug_root}")

    selected = _select_active_traces(entries, recent_seconds=max(0.0, args.recent_seconds))
    if not selected:
        raise SystemExit("no active traces matched the selection rules")

    print(f"Debug root: {debug_root}")
    print(f"Newest trace: {selected[0].trace_html}")
    print("Opening traces:")
    for entry in selected:
        print(f"- {entry.trace_html}")

    if args.dry_run:
        return 0

    opened = _open_in_explorer(entry.trace_html for entry in selected)
    print(f"Opened {opened} trace file(s) in Explorer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
