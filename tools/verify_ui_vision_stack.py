from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE = REPO_ROOT / "texty_image.png"
SAFE_TESTS = [
    "tests.test_model_service",
    "tests.test_ocr_ensemble",
    "tests.test_target_ensemble",
    "tests.test_ui_vision_harness",
]


def run(cmd: List[str], timeout_s: float = 300.0, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        check=check,
    )


def print_step(title: str) -> None:
    print(f"\n== {title} ==")


def get_json(url: str, timeout_s: float = 10.0) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict, timeout_s: float = 30.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_url(url: str, timeout_s: float = 45.0) -> dict:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            return get_json(url, timeout_s=2.0)
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def stop_services(services: Iterable[str]) -> None:
    services = list(services)
    if not services:
        return
    run(["docker", "compose", "stop", *services], timeout_s=120.0, check=False)


def start_services(services: Iterable[str], profile: str | None = None, timeout_s: float = 120.0) -> None:
    cmd = ["docker", "compose"]
    if profile:
        cmd.extend(["--profile", profile])
    cmd.extend(["up", "-d", *list(services)])
    result = run(cmd, timeout_s=timeout_s)
    sys.stdout.write(result.stdout)


def verify_safe_tests() -> None:
    print_step("Safe Test Suite")
    python_bin = REPO_ROOT / ".venv-ui-vision" / "bin" / "python"
    if not python_bin.exists():
        raise RuntimeError(f"Missing test interpreter: {python_bin}")
    result = run([str(python_bin), "-m", "unittest", *SAFE_TESTS], timeout_s=300.0)
    sys.stdout.write(result.stdout)
    if "OK" not in result.stdout:
        raise RuntimeError("Safe test suite did not end in OK")


def verify_mock_grounding() -> None:
    print_step("Mock Grounding Stack")
    services = ["groundnext-api", "aria-ui-api", "phi-ground-api", "target-ensemble-api"]
    start_services(services, profile="ui-vision")
    try:
        for url in [
            "http://127.0.0.1:28111/health",
            "http://127.0.0.1:28112/health",
            "http://127.0.0.1:28113/health",
            "http://127.0.0.1:28130/health",
        ]:
            print(json.dumps(wait_for_url(url), indent=2))
        body = post_json("http://127.0.0.1:28130/admin/selftest", {})
        if (body.get("final_prediction") or {}).get("candidate_id") != "C1":
            raise RuntimeError("Target ensemble selftest did not prefer C1")
        print(json.dumps({"final_prediction": body.get("final_prediction"), "weights": body.get("weights")}, indent=2))
    finally:
        stop_services(services)


def verify_mock_ocr() -> None:
    print_step("Mock OCR Stack")
    services = ["omniparser-api", "paddleocr-vl-api", "surya-api"]
    start_services(services, profile="ui-vision")
    try:
        for url in [
            "http://127.0.0.1:28101/health",
            "http://127.0.0.1:28102/health",
            "http://127.0.0.1:28103/health",
        ]:
            print(json.dumps(wait_for_url(url), indent=2))
        body = post_json("http://127.0.0.1:28101/admin/selftest", {})
        if int((body.get("checks") or {}).get("words", 0)) <= 0:
            raise RuntimeError("OmniParser selftest returned no words")
        print(json.dumps(body, indent=2))
    finally:
        stop_services(services)


def wait_for_ocr_api_internal(timeout_s: float = 120.0) -> dict:
    deadline = time.time() + timeout_s
    last_output = ""
    while time.time() < deadline:
        result = run(
            [
                "docker",
                "exec",
                "aicomputercontrol-ocr-api-1",
                "python",
                "-c",
                "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8020/health').read().decode())",
            ],
            timeout_s=10.0,
            check=False,
        )
        if result.returncode == 0:
            return json.loads(result.stdout.strip())
        last_output = result.stdout.strip()
        time.sleep(2.0)
    raise RuntimeError(f"ocr-api did not become healthy in time: {last_output}")


def verify_real_ocr_api() -> None:
    print_step("Real OCR API")
    start_services(["ocr-api"])
    health = wait_for_ocr_api_internal()
    if health.get("status") != "ok":
        raise RuntimeError(f"Unexpected ocr-api health: {health}")
    print(json.dumps(health, indent=2))


def verify_ocr_ensemble() -> None:
    print_step("OCR Ensemble With Real PPOCR")
    services = ["ocr-api", "omniparser-api", "paddleocr-vl-api", "surya-api", "ocr-ensemble-api"]
    start_services(services, profile="ui-vision")
    try:
        wait_for_ocr_api_internal()
        print(json.dumps(wait_for_url("http://127.0.0.1:28120/health"), indent=2))
        body = post_json("http://127.0.0.1:28120/admin/selftest", {})
        print(json.dumps(body.get("meta", {}).get("per_model_status", {}), indent=2))
        if TEST_IMAGE.exists():
            payload = {
                "image_b64": base64.b64encode(TEST_IMAGE.read_bytes()).decode("ascii"),
                "return_level": "both",
                "debug": True,
            }
            real_body = post_json("http://127.0.0.1:28120/ocr", payload, timeout_s=60.0)
            ppocr = ((real_body.get("meta") or {}).get("per_model") or {}).get("ppocr") or {}
            if ppocr.get("status") != "ok":
                raise RuntimeError(f"PPOCR probe failed: {ppocr}")
            print(
                json.dumps(
                    {
                        "per_model_status": (real_body.get("meta") or {}).get("per_model_status", {}),
                        "ppocr_word_count": len(ppocr.get("words") or []),
                        "ppocr_line_count": len(ppocr.get("lines") or []),
                    },
                    indent=2,
                )
            )
    finally:
        stop_services(services)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Verify the UI vision stack end to end.")
    parser.add_argument("--skip-real-ocr", action="store_true", help="Skip the real OCR API and OCR ensemble checks.")
    args = parser.parse_args(argv)

    verify_safe_tests()
    verify_mock_grounding()
    verify_mock_ocr()
    if not args.skip_real_ocr:
        verify_real_ocr_api()
        verify_ocr_ensemble()
    print("\nUI vision verification completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
