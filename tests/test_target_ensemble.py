import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.support import http_post_json, make_png_b64, running_compose_services, wait_for_json


class TargetEnsembleTest(unittest.TestCase):
    def test_mock_target_ensemble_prefers_sign_in(self):
        services = ["groundnext-api", "aria-ui-api", "phi-ground-api", "target-ensemble-api"]
        with running_compose_services(services, profile="ui-vision"):
            wait_for_json("http://127.0.0.1:28111/health")
            wait_for_json("http://127.0.0.1:28112/health")
            wait_for_json("http://127.0.0.1:28113/health")
            wait_for_json("http://127.0.0.1:28130/health")
            status, body = http_post_json("http://127.0.0.1:28130/admin/selftest", {})
            self.assertEqual(status, 200)
            self.assertEqual(body["final_prediction"]["candidate_id"], "C1")
            self.assertIn("groundnext", body["candidate_scores"]["C1"])
            self.assertIn("artifact_dir", body["debug_artifacts"])
            self.assertTrue(body["auto_execute"])
            self.assertEqual(body["resolution_mode"], "auto_execute")
            self.assertEqual(body["gating"]["agreeing_model_count"], 3)
            self.assertEqual(body["validator"]["status"], "passed")
            self.assertTrue(body["validator"]["passed"])

    def test_target_ensemble_enters_repair_mode_when_ambiguous(self):
        services = ["groundnext-api", "aria-ui-api", "phi-ground-api", "target-ensemble-api"]
        with running_compose_services(services, profile="ui-vision"):
            wait_for_json("http://127.0.0.1:28111/health")
            wait_for_json("http://127.0.0.1:28112/health")
            wait_for_json("http://127.0.0.1:28113/health")
            wait_for_json("http://127.0.0.1:28130/health")
            status, body = http_post_json(
                "http://127.0.0.1:28130/infer",
                {
                    "instruction": "click the button",
                    "screenshot_b64": make_png_b64(640, 480),
                    "debug": True,
                    "candidates": [
                        {"id": "C1", "box": [430, 80, 520, 120], "text": "Continue", "score": 0.58, "role": "button", "allowed_actions": ["click"]},
                        {"id": "C2", "box": [530, 80, 620, 120], "text": "Continue", "score": 0.57, "role": "button", "allowed_actions": ["click"]},
                    ],
                },
            )
            self.assertEqual(status, 200)
            self.assertFalse(body["auto_execute"])
            self.assertEqual(body["resolution_mode"], "repair")
            self.assertIn("reason_codes", body["gating"])
            self.assertTrue(body["repair_plan"])
            self.assertGreaterEqual(len(body["repair_candidates"]), 1)
            self.assertEqual(body["repair_rerun"]["status"], "performed")
            self.assertFalse(body["repair_rerun"]["promoted"])
            self.assertEqual(body["validator"]["status"], "skipped")

    def test_target_ensemble_defaults_click_when_actions_missing(self):
        services = ["groundnext-api", "aria-ui-api", "phi-ground-api", "target-ensemble-api"]
        with running_compose_services(services, profile="ui-vision"):
            wait_for_json("http://127.0.0.1:28111/health")
            wait_for_json("http://127.0.0.1:28112/health")
            wait_for_json("http://127.0.0.1:28113/health")
            wait_for_json("http://127.0.0.1:28130/health")
            status, body = http_post_json(
                "http://127.0.0.1:28130/infer",
                {
                    "instruction": "click the Sign in button in the top right",
                    "screenshot_b64": make_png_b64(1024, 768),
                    "candidates": [
                        {"id": "C1", "box": [700, 120, 820, 170], "text": "Sign in", "score": 0.96, "role": "button"},
                    ],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(body["auto_execute"])
            self.assertTrue(body["gating"]["action_allowed"])
            self.assertEqual(body["final_prediction"]["candidate_id"], "C1")
            self.assertEqual(body["validator"]["status"], "passed")

    def test_target_ensemble_promotes_decisive_repair_rerun(self):
        services = ["groundnext-api", "aria-ui-api", "phi-ground-api", "target-ensemble-api"]
        with running_compose_services(services, profile="ui-vision"):
            wait_for_json("http://127.0.0.1:28111/health")
            wait_for_json("http://127.0.0.1:28112/health")
            wait_for_json("http://127.0.0.1:28113/health")
            wait_for_json("http://127.0.0.1:28130/health")
            status, body = http_post_json(
                "http://127.0.0.1:28130/infer",
                {
                    "instruction": "click the Sign in button in the top right",
                    "screenshot_b64": make_png_b64(1024, 768),
                    "candidates": [
                        {"id": "C1", "box": [700, 120, 820, 170], "text": "Sign in", "score": 0.55, "role": "button", "allowed_actions": ["click"]},
                        {"id": "C2", "box": [420, 360, 600, 410], "text": "Cancel", "score": 0.45, "role": "button", "allowed_actions": ["click"]},
                        {"id": "C3", "box": [80, 640, 180, 690], "text": "Help", "score": 0.40, "role": "link", "allowed_actions": ["click"]},
                    ],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(body["auto_execute"])
            self.assertEqual(body["resolution_mode"], "repair_rerun_auto_execute")
            self.assertEqual(body["final_prediction"]["candidate_id"], "C1")
            self.assertEqual(body["repair_rerun"]["status"], "performed")
            self.assertTrue(body["repair_rerun"]["promoted"])
            self.assertEqual(body["validator"]["status"], "passed")
            self.assertEqual(body["validator"]["stage"], "repair_rerun")

    def test_target_ensemble_validator_blocks_low_prior_auto_execute(self):
        services = ["groundnext-api", "aria-ui-api", "phi-ground-api", "target-ensemble-api"]
        with running_compose_services(services, profile="ui-vision"):
            wait_for_json("http://127.0.0.1:28111/health")
            wait_for_json("http://127.0.0.1:28112/health")
            wait_for_json("http://127.0.0.1:28113/health")
            wait_for_json("http://127.0.0.1:28130/health")
            status, body = http_post_json(
                "http://127.0.0.1:28130/infer",
                {
                    "instruction": "click the Sign in button in the top right",
                    "screenshot_b64": make_png_b64(1024, 768),
                    "candidates": [
                        {"id": "C1", "box": [700, 120, 820, 170], "text": "Sign in", "score": 0.55},
                    ],
                },
            )
            self.assertEqual(status, 200)
            self.assertFalse(body["auto_execute"])
            self.assertEqual(body["resolution_mode"], "repair_validation_failed")
            self.assertEqual(body["final_prediction"]["candidate_id"], "C1")
            self.assertEqual(body["validator"]["status"], "failed")
            self.assertIn("validator_candidate_prior_below_threshold", body["validator"]["reason_codes"])
            self.assertEqual(body["repair_plan"]["status"], "validation_failed")


if __name__ == "__main__":
    unittest.main()
