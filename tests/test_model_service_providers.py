import json
import socketserver
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modelService.app.providers import HTTPProxyProvider, ProviderError, build_provider


class _ProxyHandler(BaseHTTPRequestHandler):
    routes = {}

    def do_GET(self):
        response = dict(self.routes.get(self.path) or {})
        raw = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(length) if length else b"{}"
        content_type = str(self.headers.get("Content-Type", "") or "").lower()
        if "application/json" in content_type:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        else:
            payload = {"raw_body_len": len(raw_body)}
        response = dict(self.routes.get(self.path) or {})
        response["echo"] = payload
        raw = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format, *args):  # noqa: A003
        return


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


class HTTPProxyProviderTest(unittest.TestCase):
    def _run_server(self, routes):
        handler = type("DynamicProxyHandler", (_ProxyHandler,), {"routes": routes})
        server = _ThreadedHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def test_http_proxy_provider_for_ocr(self):
        server, thread = self._run_server(
            {
                "/ocr": {
                    "width": 640,
                    "height": 480,
                    "words": [{"poly": [[0, 0], [10, 0], [10, 10], [0, 10]], "text": "Sign", "score": 0.95}],
                    "lines": [],
                    "latency_ms": 7,
                }
            }
        )
        try:
            provider = HTTPProxyProvider("omniparser", "ocr", f"http://127.0.0.1:{server.server_port}")
            result = provider.run_ocr(image_bgr=np.full((16, 16, 3), 255, dtype=np.uint8), min_score=0.45)
        finally:
            server.shutdown()
            thread.join(timeout=2.0)
            server.server_close()
        self.assertEqual(result["width"], 640)
        self.assertEqual(result["echo"]["return_level"], "both")
        self.assertEqual(result["echo"]["min_score"], 0.45)

    def test_http_proxy_provider_for_grounding(self):
        server, thread = self._run_server(
            {
                "/infer": {
                    "predictions": [{"candidate_id": "C1", "action": "click", "p": 0.91}],
                    "latency_ms": 9,
                }
            }
        )
        try:
            provider = build_provider("groundnext", "grounding", "http_proxy", f"http://127.0.0.1:{server.server_port}")
            result = provider.run_grounding({"instruction": "click sign in", "candidates": [{"id": "C1"}]})
        finally:
            server.shutdown()
            thread.join(timeout=2.0)
            server.server_close()
        self.assertEqual(result["predictions"][0]["candidate_id"], "C1")
        self.assertEqual(result["echo"]["instruction"], "click sign in")

    def test_http_proxy_provider_requires_remote_url(self):
        with self.assertRaises(ProviderError):
            build_provider("omniparser", "ocr", "http_proxy", "")

    def test_omniparser_runtime_provider_maps_parse_output(self):
        server, thread = self._run_server(
            {
                "/parse/": {
                    "parsed_content_list": [
                        {
                            "type": "text",
                            "bbox": [0.10, 0.20, 0.30, 0.30],
                            "content": "Sign in",
                            "source": "box_ocr_content_ocr",
                        },
                        {
                            "type": "icon",
                            "bbox": [0.40, 0.50, 0.55, 0.62],
                            "content": "Settings",
                            "source": "box_caption",
                        },
                    ]
                }
            }
        )
        try:
            provider = build_provider("omniparser", "ocr", "omniparser_runtime", f"http://127.0.0.1:{server.server_port}")
            result = provider.run_ocr(image_bgr=np.full((100, 200, 3), 255, dtype=np.uint8), min_score=0.45)
        finally:
            server.shutdown()
            thread.join(timeout=2.0)
            server.server_close()

        self.assertEqual(result["width"], 200)
        self.assertEqual(result["height"], 100)
        self.assertEqual(len(result["lines"]), 2)
        self.assertGreaterEqual(len(result["words"]), 3)
        self.assertEqual(result["lines"][0]["text"], "Sign in")
        self.assertEqual(result["words"][0]["source_model"], "omniparser")
        self.assertEqual(result["meta"]["parsed_content_count"], 2)

    def test_paddleocr_vl_runtime_provider_maps_spotting_output(self):
        server, thread = self._run_server(
            {
                "/v1/models": {
                    "data": [
                        {"id": "PaddleOCR-VL-1.5-0.9B"},
                    ]
                },
                "/v1/chat/completions": {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Compose<|LOC_100|><|LOC_300|><|LOC_200|><|LOC_300|><|LOC_200|><|LOC_360|><|LOC_100|><|LOC_360|>\n"
                                    "Sign in<|LOC_600|><|LOC_120|><|LOC_720|><|LOC_120|><|LOC_720|><|LOC_170|><|LOC_600|><|LOC_170|>"
                                )
                            }
                        }
                    ]
                },
            }
        )
        try:
            provider = build_provider("paddleocr-vl", "ocr", "paddleocr_vl_runtime", f"http://127.0.0.1:{server.server_port}")
            result = provider.run_ocr(image_bgr=np.full((1000, 1000, 3), 255, dtype=np.uint8), min_score=0.45)
        finally:
            server.shutdown()
            thread.join(timeout=2.0)
            server.server_close()

        self.assertEqual(result["width"], 1000)
        self.assertEqual(result["height"], 1000)
        self.assertEqual(len(result["lines"]), 2)
        self.assertGreaterEqual(len(result["words"]), 3)
        self.assertEqual(result["lines"][0]["text"], "Compose")
        self.assertEqual(result["lines"][1]["poly"][0], [600, 120])
        self.assertEqual(result["meta"]["served_model_name"], "PaddleOCR-VL-1.5-0.9B")
        self.assertEqual(result["meta"]["spotting_row_count"], 2)

    def test_surya_runtime_provider_maps_text_lines(self):
        server, thread = self._run_server(
            {
                "/ocr": {
                    "data": [
                        {
                            "text_lines": [
                                {
                                    "polygon": [[10.0, 20.0], [90.0, 20.0], [90.0, 40.0], [10.0, 40.0]],
                                    "confidence": 0.98,
                                    "text": "Compose",
                                    "bbox": [10.0, 20.0, 90.0, 40.0],
                                },
                                {
                                    "polygon": [[100.0, 50.0], [220.0, 50.0], [220.0, 80.0], [100.0, 80.0]],
                                    "confidence": 0.95,
                                    "text": "Sign in",
                                    "bbox": [100.0, 50.0, 220.0, 80.0],
                                },
                            ]
                        }
                    ]
                }
            }
        )
        try:
            provider = build_provider("surya", "ocr", "surya_runtime", f"http://127.0.0.1:{server.server_port}")
            result = provider.run_ocr(image_bgr=np.full((200, 300, 3), 255, dtype=np.uint8), min_score=0.45)
        finally:
            server.shutdown()
            thread.join(timeout=2.0)
            server.server_close()

        self.assertEqual(result["width"], 300)
        self.assertEqual(result["height"], 200)
        self.assertEqual(len(result["lines"]), 2)
        self.assertGreaterEqual(len(result["words"]), 3)
        self.assertEqual(result["lines"][0]["text"], "Compose")
        self.assertEqual(result["lines"][1]["poly"][0], [100.0, 50.0])
        self.assertEqual(result["meta"]["page_count"], 1)


if __name__ == "__main__":
    unittest.main()
