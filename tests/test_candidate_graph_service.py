import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.support import http_post_json, make_png_b64, running_compose_services, wait_for_json


class CandidateGraphServiceTest(unittest.TestCase):
    def test_candidate_graph_selftest_reports_merged_output(self):
        services = ["candidate-graph-api"]
        with running_compose_services(services, profile="ui-vision"):
            wait_for_json("http://127.0.0.1:28125/health")
            status, body = http_post_json("http://127.0.0.1:28125/admin/selftest", {})
            self.assertEqual(status, 200)
            self.assertEqual(body["checks"]["graph_count"], 2)
            self.assertEqual(body["checks"]["candidate_count"], 2)
            self.assertEqual(body["checks"]["top_text"], "Sign in")

    def test_candidate_graph_infer_returns_graph_and_candidates(self):
        services = ["candidate-graph-api"]
        with running_compose_services(services, profile="ui-vision"):
            wait_for_json("http://127.0.0.1:28125/health")
            status, body = http_post_json(
                "http://127.0.0.1:28125/infer/debug",
                {
                    "viewport": {"width": 1000, "height": 800},
                    "screenshot_b64": make_png_b64(1000, 800),
                    "ui_elements": [
                        {"source": "ocr_word", "text": "Sign in", "box": [100, 40, 180, 70], "score": 0.93},
                        {"source": "ax", "text": "Sign in", "box": [96, 36, 186, 76], "score": 0.99, "role": "button"},
                    ],
                },
            )
            self.assertEqual(status, 200)
            self.assertEqual(len(body["candidate_graph"]), 1)
            self.assertEqual(len(body["candidates"]), 1)
            self.assertEqual(body["candidate_graph"][0]["role_hint"], "button")
            self.assertEqual(body["candidates"][0]["text"], "Sign in")
            self.assertIn("artifact_dir", body["debug_artifacts"])
            self.assertEqual(body["meta"]["merged_count"], 1)


if __name__ == "__main__":
    unittest.main()
