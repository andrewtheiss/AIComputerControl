#!/usr/bin/env python3
"""OCR stack smoke test.

Probes every OCR ingredient in the running docker-compose stack against a
reference image and prints a single JSON report. Exits non-zero if any
service fails its probe.

Usage:
    python3 tools/ocr_smoke.py                # default image texty_image.png
    python3 tools/ocr_smoke.py --image path.png
    python3 tools/ocr_smoke.py --grounding    # also probe target-ensemble
    python3 tools/ocr_smoke.py --host 127.0.0.1

Requires the stack to be running with:
    docker compose --profile ui-vision up -d

It hits the host-mapped ports defined in docker-compose.yml (28020 for
ocr-api, 28101-28103 for modelService OCR sidecars, 28120 for
ocr-ensemble-api, 28130 for target-ensemble-api). No docker exec needed.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "texty_image.png"


def _post_json(url: str, payload: Dict[str, Any], timeout_s: float) -> Tuple[Dict[str, Any], int]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body, int(resp.status)


def _get_json(url: str, timeout_s: float = 5.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _first_words(words: List[Dict[str, Any]], n: int = 5) -> List[str]:
    out: List[str] = []
    for w in words[:n]:
        text = str(w.get("text", "") or "").strip()
        if text:
            out.append(text)
    return out


def probe_ocr_service(name: str, base_url: str, image_b64: str, timeout_s: float = 30.0) -> Dict[str, Any]:
    """Hit /health and /ocr on a modelService-style or ensemble-style OCR service."""
    result: Dict[str, Any] = {
        "service": name,
        "base_url": base_url,
        "status": "unknown",
        "latency_ms": None,
        "word_count": 0,
        "line_count": 0,
        "first_words": [],
        "per_model_status": None,
        "error": None,
    }
    # Health check
    try:
        _get_json(base_url + "/health", timeout_s=5.0)
    except Exception as exc:
        result["status"] = "health_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    # OCR probe
    t0 = time.perf_counter()
    try:
        body, _ = _post_json(
            base_url + "/ocr",
            {
                "image_b64": image_b64,
                "return_level": "both",
                "min_score": 0.45,
                "debug": False,
            },
            timeout_s=timeout_s,
        )
    except Exception as exc:
        result["status"] = "ocr_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        return result

    result["status"] = "ok"
    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    result["word_count"] = len(body.get("words") or [])
    result["line_count"] = len(body.get("lines") or [])
    result["first_words"] = _first_words(body.get("words") or [])
    meta = body.get("meta") or {}
    per_model_status = meta.get("per_model_status")
    if per_model_status is not None:
        result["per_model_status"] = per_model_status
        # Ensemble rollup. "ok" and "mock_skipped" are both positive signals —
        # mock_skipped is the intended state when OCR_ENSEMBLE_DROP_MOCK_BACKEND
        # is on and a sidecar is running with MODEL_BACKEND=mock. Only "error"
        # is a real failure.
        errored = [m for m, s in per_model_status.items() if s not in ("ok", "mock_skipped")]
        mock_skipped = [m for m, s in per_model_status.items() if s == "mock_skipped"]
        active = meta.get("active_models") or [m for m, s in per_model_status.items() if s == "ok"]
        if errored:
            result["status"] = "degraded"
            result["error"] = f"upstreams errored: {errored}"
        elif mock_skipped and not active:
            # Every upstream was mock — merged output is empty by design but
            # the ensemble has nothing real to report. Surface as "mock_only".
            result["status"] = "mock_only"
            result["error"] = f"all upstreams mock-backed: {mock_skipped}"
        result["active_models"] = active
        result["mock_skipped_models"] = mock_skipped
    return result


def probe_ocr_api_legacy(host: str, image_bytes: bytes, timeout_s: float = 30.0) -> Dict[str, Any]:
    """The legacy ocr-api takes multipart file upload, not JSON body. Separate helper."""
    base_url = f"http://{host}:28020"
    result: Dict[str, Any] = {
        "service": "ocr-api",
        "base_url": base_url,
        "status": "unknown",
        "latency_ms": None,
        "word_count": 0,
        "line_count": 0,
        "first_words": [],
        "per_model_status": None,
        "error": None,
    }
    # Health
    try:
        _get_json(base_url + "/health", timeout_s=5.0)
    except Exception as exc:
        result["status"] = "health_failed"
        result["error"] = (
            f"{type(exc).__name__}: {exc}. Is ocr-api exposed on port 28020? "
            "The audit added this mapping — if missing, re-pull docker-compose.yml."
        )
        return result

    # Multipart POST. Hand-built so we don't pull in the requests dep.
    boundary = "----ocrsmokeboundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="frame.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode("ascii") + image_bytes + f"\r\n--{boundary}--\r\n".encode("ascii")
    req = urllib.request.Request(
        base_url + "/ocr",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        result["status"] = "ocr_failed"
        result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["status"] = "ok"
    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    result["word_count"] = len(data.get("words") or [])
    result["line_count"] = len(data.get("lines") or [])
    result["first_words"] = _first_words(data.get("words") or [])
    return result


def probe_candidate_graph_selftest(host: str) -> Dict[str, Any]:
    base_url = f"http://{host}:28125"
    result = {"service": "candidate-graph-api", "base_url": base_url, "status": "unknown", "error": None}
    try:
        _get_json(base_url + "/health", timeout_s=5.0)
        body, _ = _post_json(base_url + "/admin/selftest", {}, timeout_s=10.0)
        checks = body.get("checks") or {}
        if int(checks.get("graph_count", 0)) <= 0:
            result["status"] = "empty"
            result["error"] = "selftest returned no candidates"
        else:
            result["status"] = "ok"
            result["checks"] = checks
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def probe_target_ensemble_selftest(host: str) -> Dict[str, Any]:
    base_url = f"http://{host}:28130"
    result = {"service": "target-ensemble-api", "base_url": base_url, "status": "unknown", "error": None}
    try:
        _get_json(base_url + "/health", timeout_s=5.0)
        body, _ = _post_json(base_url + "/admin/selftest", {}, timeout_s=15.0)
        final = body.get("final_prediction") or {}
        if not final.get("candidate_id"):
            result["status"] = "empty"
            result["error"] = "selftest returned no final_prediction"
        else:
            result["status"] = "ok"
            result["final_candidate_id"] = final.get("candidate_id")
            result["final_p"] = final.get("p")
            result["per_model"] = body.get("per_model")
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test the OCR stack via host-mapped ports.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Image to probe with (default: texty_image.png)")
    parser.add_argument("--host", default="127.0.0.1", help="Host where the compose stack is reachable")
    parser.add_argument("--grounding", action="store_true", help="Also selftest candidate-graph and target-ensemble")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = parser.parse_args(argv)

    image_path: Path = args.image
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 2
    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    host = args.host
    services: List[Tuple[str, str]] = [
        ("omniparser-api", f"http://{host}:28101"),
        ("paddleocr-vl-api", f"http://{host}:28102"),
        ("surya-api", f"http://{host}:28103"),
        ("ocr-ensemble-api", f"http://{host}:28120"),
    ]

    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(services) + 1) as pool:
        futures = {
            pool.submit(probe_ocr_service, name, base, image_b64): name
            for name, base in services
        }
        futures[pool.submit(probe_ocr_api_legacy, host, image_bytes)] = "ocr-api"
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    if args.grounding:
        results.append(probe_candidate_graph_selftest(host))
        results.append(probe_target_ensemble_selftest(host))

    # Sort by service name for stable output
    results.sort(key=lambda r: str(r.get("service", "")))

    report = {
        "image": str(image_path),
        "host": host,
        "probed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
    }

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"OCR smoke report (image={image_path.name}, host={host})")
        print("=" * 72)
        for r in results:
            svc = r.get("service", "?")
            status = r.get("status", "?")
            latency = r.get("latency_ms")
            latency_str = f"{latency}ms" if latency is not None else "-"
            words = r.get("word_count", 0)
            lines = r.get("line_count", 0)
            first = ", ".join(r.get("first_words") or [])
            print(f"  {svc:<22} {status:<14} {latency_str:>8}  words={words:<4} lines={lines:<4}  {first}")
            if r.get("per_model_status"):
                print(f"    per_model_status: {r['per_model_status']}")
            if r.get("active_models"):
                print(f"    active_models:    {r['active_models']}")
            if r.get("mock_skipped_models"):
                print(f"    mock_skipped:     {r['mock_skipped_models']}")
            if r.get("error"):
                print(f"    error: {r['error']}")
        print("=" * 72)

    # "ok" means this ingredient is healthy and contributing. "mock_only" and
    # "mock_skipped" mean the ingredient is alive but intentionally not in the
    # merged output — neither is a failure. Only "health_failed", "ocr_failed",
    # "failed", "empty", and "degraded" should fail the exit code.
    POSITIVE_STATUSES = {"ok", "mock_only", "mock_skipped"}
    any_fail = any(r.get("status") not in POSITIVE_STATUSES for r in results)
    if any_fail:
        print("One or more OCR ingredients are not healthy.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
