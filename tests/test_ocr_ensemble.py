import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.support import http_post_json, running_compose_services, wait_for_json


class OCREnsembleTest(unittest.TestCase):
    def test_mock_ensemble_merges_outputs(self):
        services = ["omniparser-api", "paddleocr-vl-api", "surya-api", "ocr-ensemble-api"]
        with running_compose_services(services, profile="ui-vision", no_deps=True):
            wait_for_json("http://127.0.0.1:28101/health")
            wait_for_json("http://127.0.0.1:28102/health")
            wait_for_json("http://127.0.0.1:28103/health")
            wait_for_json("http://127.0.0.1:28120/health")
            status, body = http_post_json("http://127.0.0.1:28120/admin/selftest", {})
            self.assertEqual(status, 200)
            texts = {item["text"] for item in body["lines"]}
            self.assertIn("Sign in", texts)
            self.assertIn("artifact_dir", body["debug_artifacts"])
            self.assertEqual(body["meta"]["per_model_status"]["omniparser"], "ok")


if __name__ == "__main__":
    unittest.main()
