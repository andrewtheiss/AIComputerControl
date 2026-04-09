import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ocrEnsemble.app.main import execute_ocr_fanout
from tests.support import (
    http_post_json,
    run_uvicorn_app,
    running_compose_services,
    wait_for_json,
)
from ui_vision_common.schemas import OCRRequest


MOCK_ONLY_URLS = {
    "ppocr": "mock://ppocr",
    "omniparser": "mock://omniparser",
    "paddleocr-vl": "mock://paddleocr-vl",
    "surya": "mock://surya",
}


def _blank_frame(width: int = 1024, height: int = 768) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


class OCREnsembleDropMockTest(unittest.TestCase):
    """Unit tests for the mock-drop logic.

    ``execute_ocr_fanout`` is the pure function entry point the FastAPI
    route wraps. Calling it directly bypasses HTTP, subprocess, and ASGI
    transport entirely — the exact same code path runs in docker, just
    without the serialization hops.
    """

    def test_drops_mock_upstreams_by_default(self):
        response = execute_ocr_fanout(
            image_bgr=_blank_frame(),
            req=OCRRequest(debug=True),
            model_urls=MOCK_ONLY_URLS,
            drop_mock_backend=True,
        )
        meta = response["meta"]
        self.assertTrue(meta["drop_mock_backend"])
        # Every configured upstream is mock, so the merged output is empty.
        self.assertEqual(response["words"], [])
        self.assertEqual(response["lines"], [])
        self.assertEqual(meta["active_models"], [])
        self.assertEqual(
            sorted(meta["dropped_mock_models"]),
            ["omniparser", "paddleocr-vl", "ppocr", "surya"],
        )
        per_model_status = meta["per_model_status"]
        for model_id in MOCK_ONLY_URLS:
            self.assertEqual(per_model_status[model_id], "mock_skipped")
        # per_model is attached when debug=True, and still carries the raw
        # mock payload so forensics work even though nothing was merged.
        per_model = meta["per_model"]
        omni_words = [str(w.get("text", "")) for w in per_model["omniparser"]["words"]]
        self.assertIn("Sign", omni_words)
        self.assertIn("in", omni_words)

    def test_keeps_mock_upstreams_when_disabled(self):
        response = execute_ocr_fanout(
            image_bgr=_blank_frame(),
            req=OCRRequest(debug=True),
            model_urls=MOCK_ONLY_URLS,
            drop_mock_backend=False,
        )
        meta = response["meta"]
        self.assertFalse(meta["drop_mock_backend"])
        self.assertEqual(meta["dropped_mock_models"], [])
        self.assertEqual(
            sorted(meta["active_models"]),
            ["omniparser", "paddleocr-vl", "ppocr", "surya"],
        )
        # Mock data should land in the merged output now.
        line_texts = {item["text"] for item in response["lines"]}
        self.assertIn("Sign in", line_texts)
        per_model_status = meta["per_model_status"]
        for model_id in MOCK_ONLY_URLS:
            self.assertEqual(per_model_status[model_id], "ok")

    def test_mixed_real_and_mock_only_merges_real(self):
        """Simulate one real upstream + three mock upstreams by monkey-patching
        ``_call_model`` for the real entry. The merged output should contain
        only the real text; the mock entries stay in per_model.
        """
        real_result = {
            "width": 1024,
            "height": 768,
            "backend": "ppocr",
            "words": [
                {
                    "poly": [[100, 100], [260, 100], [260, 140], [100, 140]],
                    "text": "Inbox",
                    "score": 0.97,
                }
            ],
            "lines": [
                {
                    "poly": [[100, 100], [260, 100], [260, 140], [100, 140]],
                    "text": "Inbox",
                    "score": 0.97,
                }
            ],
            "latency_ms": 12,
        }

        from ocrEnsemble.app import main as ocr_main

        original_call = ocr_main._call_model

        def fake_call(session, *, model_id, model_url, image_b64, min_score):
            if model_id == "ppocr":
                return dict(real_result)
            return original_call(
                session,
                model_id=model_id,
                model_url=model_url,
                image_b64=image_b64,
                min_score=min_score,
            )

        ocr_main._call_model = fake_call
        try:
            response = execute_ocr_fanout(
                image_bgr=_blank_frame(),
                req=OCRRequest(debug=True),
                model_urls={
                    "ppocr": "http://fake-ppocr:8020/ocr",
                    "omniparser": "mock://omniparser",
                    "paddleocr-vl": "mock://paddleocr-vl",
                    "surya": "mock://surya",
                },
                drop_mock_backend=True,
            )
        finally:
            ocr_main._call_model = original_call

        meta = response["meta"]
        # Only ppocr contributed; the three mock sidecars were dropped.
        self.assertEqual(meta["active_models"], ["ppocr"])
        self.assertEqual(
            sorted(meta["dropped_mock_models"]),
            ["omniparser", "paddleocr-vl", "surya"],
        )
        self.assertEqual(meta["per_model_status"]["ppocr"], "ok")
        for model_id in ("omniparser", "paddleocr-vl", "surya"):
            self.assertEqual(meta["per_model_status"][model_id], "mock_skipped")
        merged_texts = {item["text"] for item in response["words"]}
        self.assertIn("Inbox", merged_texts)
        # Mock placeholders must not leak into the merged output.
        self.assertNotIn("Sign", merged_texts)
        self.assertNotIn("Browser", merged_texts)
        self.assertNotIn("Continue", merged_texts)


class OCREnsembleUvicornTest(unittest.TestCase):
    """Integration-flavor test that brings the ensemble up under uvicorn and
    speaks real HTTP to it. Catches regressions in the FastAPI route wiring,
    JSON serialization, and subprocess startup — things the pure
    ``execute_ocr_fanout`` tests can't see.
    """

    def test_selftest_via_uvicorn_subprocess(self):
        env = {
            "OCR_ENSEMBLE_MODEL_URLS": json.dumps(
                {
                    "ppocr": "mock://ppocr",
                    "omniparser": "mock://omniparser",
                    "paddleocr-vl": "mock://paddleocr-vl",
                    "surya": "mock://surya",
                }
            ),
            "OCR_ENSEMBLE_DROP_MOCK_BACKEND": "1",
        }
        with run_uvicorn_app(module_name="ocrEnsemble.app.main", env=env) as base_url:
            status, body = http_post_json(base_url + "/admin/selftest", {})
            self.assertEqual(status, 200)
            meta = body["meta"]
            self.assertTrue(meta["drop_mock_backend"])
            self.assertEqual(
                sorted(meta["dropped_mock_models"]),
                ["omniparser", "paddleocr-vl", "ppocr", "surya"],
            )
            self.assertEqual(body["words"], [])
            self.assertEqual(body["lines"], [])
            # debug_artifacts come from the FastAPI wrapper's _artifact_bundle
            # call — this is the only path that exercises it under real HTTP.
            self.assertIn("artifact_dir", body["debug_artifacts"])


class OCREnsembleComposeTest(unittest.TestCase):
    """Docker-compose smoke test. Kept to catch build/start regressions.

    With the mock-drop default on, the merged output is empty because every
    OCR sidecar in this test is mock-backed (no ``ocr-api`` via ``no_deps``).
    We assert the smoke signal: the service builds, comes up, selftest
    succeeds, mock-backed upstreams are tagged ``mock_skipped``, and
    per_model still carries the raw mock payload for forensics.
    """

    def test_mock_ensemble_compose_smoke(self):
        services = ["omniparser-api", "paddleocr-vl-api", "surya-api", "ocr-ensemble-api"]
        with running_compose_services(services, profile="ui-vision", no_deps=True):
            wait_for_json("http://127.0.0.1:28101/health")
            wait_for_json("http://127.0.0.1:28102/health")
            wait_for_json("http://127.0.0.1:28103/health")
            wait_for_json("http://127.0.0.1:28120/health")
            status, body = http_post_json("http://127.0.0.1:28120/admin/selftest", {})
            self.assertEqual(status, 200)
            self.assertIn("artifact_dir", body["debug_artifacts"])
            meta = body["meta"]
            self.assertTrue(meta.get("drop_mock_backend"))
            for model_id in ("omniparser", "paddleocr-vl", "surya"):
                self.assertEqual(meta["per_model_status"][model_id], "mock_skipped")
            self.assertEqual(body["words"], [])
            self.assertEqual(body["lines"], [])
            per_model = meta["per_model"]
            omni_words = [str(w.get("text", "")) for w in per_model["omniparser"]["words"]]
            self.assertIn("Sign", omni_words)


if __name__ == "__main__":
    unittest.main()
