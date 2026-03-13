from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_png_bytes(width: int = 640, height: int = 480, rgb: tuple[int, int, int] = (255, 255, 255)) -> bytes:
    width = max(1, int(width))
    height = max(1, int(height))
    r, g, b = [max(0, min(255, int(v))) for v in rgb]
    row = b"\x00" + bytes([r, g, b]) * width
    raw = row * height

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", ihdr),
            chunk(b"IDAT", zlib.compress(raw, level=9)),
            chunk(b"IEND", b""),
        ]
    )


def make_png_b64(width: int = 640, height: int = 480, rgb: tuple[int, int, int] = (255, 255, 255)) -> str:
    return base64.b64encode(make_png_bytes(width=width, height=height, rgb=rgb)).decode("ascii")


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def http_get_json(url: str, timeout_s: float = 10.0) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return int(resp.status), json.loads(resp.read().decode("utf-8"))


def http_post_json(url: str, payload: dict, timeout_s: float = 10.0) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return int(resp.status), json.loads(resp.read().decode("utf-8"))


async def _asgi_json_request(app, method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.request(method=method, url=path, json=payload)
        return int(resp.status_code), resp.json()


def asgi_post_json(app, path: str, payload: dict) -> tuple[int, dict]:
    return asyncio.run(_asgi_json_request(app=app, method="POST", path=path, payload=payload))


def run_command(args: list[str], timeout_s: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(REPO_ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )


@contextlib.contextmanager
def running_compose_services(
    services: list[str],
    profile: str | None = None,
    no_deps: bool = False,
    timeout_s: float = 120.0,
):
    up_cmd = ["docker", "compose"]
    if profile:
        up_cmd.extend(["--profile", profile])
    up_cmd.extend(["up", "-d"])
    if no_deps:
        up_cmd.append("--no-deps")
    up_cmd.extend(services)
    run_command(up_cmd, timeout_s=timeout_s)
    try:
        yield
    finally:
        stop_cmd = ["docker", "compose", "stop", *services]
        subprocess.run(
            stop_cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )


def wait_for_json(url: str, timeout_s: float = 20.0) -> dict:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            status, body = http_get_json(url, timeout_s=2.0)
            if status == 200:
                return body
            last_error = f"status={status}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


@contextlib.contextmanager
def run_uvicorn_app(
    module_name: str,
    factory_name: str = "create_app",
    env: dict[str, str] | None = None,
    timeout_s: float = 15.0,
):
    port = _pick_free_port()
    child_env = os.environ.copy()
    if env:
        child_env.update({str(key): str(value) for key, value in env.items()})
    child_env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not child_env.get("PYTHONPATH")
        else f"{REPO_ROOT}{os.pathsep}{child_env['PYTHONPATH']}"
    )

    log_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            f"{module_name}:{factory_name}",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--loop",
            "asyncio",
            "--http",
            "h11",
            "--log-level",
            "warning",
        ],
        cwd=str(REPO_ROOT),
        env=child_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base_url = f"http://127.0.0.1:{port}"
    health_url = f"{base_url}/health"
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                log_file.seek(0)
                output = log_file.read()
                raise RuntimeError(f"{module_name} exited early with code {proc.returncode}: {output}")
            try:
                status, _ = http_get_json(health_url, timeout_s=1.0)
                if status == 200:
                    yield base_url
                    return
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                time.sleep(0.1)
        log_file.seek(0)
        output = log_file.read()
        raise RuntimeError(f"{module_name} did not become healthy within {timeout_s}s: {output}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        log_file.close()
