from __future__ import annotations

import time
from typing import Any, Dict

import requests

from ui_vision_common.mock_backends import mock_grounding_result, mock_ocr_result


class ProviderError(RuntimeError):
    pass


class BaseProvider:
    def __init__(self, model_id: str, backend: str):
        self.model_id = model_id
        self.backend = backend

    def run_ocr(self, image_bgr, min_score: float) -> Dict[str, Any]:
        raise ProviderError(f"OCR is not supported by {self.model_id}")

    def run_grounding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise ProviderError(f"Grounding is not supported by {self.model_id}")


class MockProvider(BaseProvider):
    def __init__(self, model_id: str, role: str):
        super().__init__(model_id=model_id, backend="mock")
        self.role = role

    def run_ocr(self, image_bgr, min_score: float) -> Dict[str, Any]:
        return mock_ocr_result(self.model_id, image_bgr, min_score=min_score)

    def run_grounding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return mock_grounding_result(
            self.model_id,
            instruction=str(payload.get("instruction", "") or ""),
            candidates=list(payload.get("candidates") or []),
            history=list(payload.get("history") or []),
            top_k=int(payload.get("top_k") or 5),
        )


class HTTPProxyProvider(BaseProvider):
    def __init__(self, model_id: str, role: str, remote_url: str):
        if not remote_url:
            raise ProviderError("MODEL_BACKEND=http_proxy requires REMOTE_MODEL_URL")
        super().__init__(model_id=model_id, backend="http_proxy")
        self.role = role
        self.remote_url = remote_url.rstrip("/")
        self.session = requests.Session()

    def run_ocr(self, image_bgr, min_score: float) -> Dict[str, Any]:
        from ui_vision_common.image_tools import encode_png_b64

        t0 = time.perf_counter()
        payload = {
            "image_b64": encode_png_b64(image_bgr),
            "return_level": "both",
            "min_score": min_score,
        }
        resp = self.session.post(f"{self.remote_url}/ocr", json=payload, timeout=(3.0, 30.0))
        resp.raise_for_status()
        data = resp.json()
        data["latency_ms"] = data.get("latency_ms") or int((time.perf_counter() - t0) * 1000)
        return data

    def run_grounding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        resp = self.session.post(f"{self.remote_url}/infer", json=payload, timeout=(3.0, 30.0))
        resp.raise_for_status()
        data = resp.json()
        data["latency_ms"] = data.get("latency_ms") or int((time.perf_counter() - t0) * 1000)
        return data


def build_provider(model_id: str, role: str, backend: str, remote_url: str) -> BaseProvider:
    if backend == "mock":
        return MockProvider(model_id=model_id, role=role)
    if backend == "http_proxy":
        return HTTPProxyProvider(model_id=model_id, role=role, remote_url=remote_url)
    raise ProviderError(f"Unsupported MODEL_BACKEND={backend}")
