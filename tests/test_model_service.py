import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.support import http_post_json, running_compose_services, wait_for_json


class ModelServiceTest(unittest.TestCase):
    def test_mock_ocr_service(self):
        with running_compose_services(["omniparser-api"], profile="ui-vision"):
            wait_for_json("http://127.0.0.1:28101/health")
            status, body = http_post_json("http://127.0.0.1:28101/admin/selftest", {})
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "ok")
            self.assertEqual(body["model_id"], "omniparser")
            self.assertGreaterEqual(int(body["checks"]["words"]), 3)

    def test_mock_grounding_service(self):
        with running_compose_services(["groundnext-api"], profile="ui-vision"):
            wait_for_json("http://127.0.0.1:28111/health")
            status, body = http_post_json("http://127.0.0.1:28111/admin/selftest", {})
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "ok")
            self.assertEqual(body["model_id"], "groundnext")
            self.assertGreaterEqual(int(body["checks"]["predictions"]), 1)


if __name__ == "__main__":
    unittest.main()
