import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "agent" / "src"))

from target_ensemble_shadow_utils import (  # noqa: E402
    build_target_ensemble_endpoint,
    compute_interactable_score,
    resolve_execution_override,
    should_merge_shadow_node,
)


class TargetEnsembleShadowUtilsTest(unittest.TestCase):
    def test_build_target_ensemble_endpoint_normalizes_infer_suffixes(self):
        self.assertEqual(
            build_target_ensemble_endpoint("http://target-ensemble-api:8000/infer", True),
            "http://target-ensemble-api:8000/infer/debug",
        )
        self.assertEqual(
            build_target_ensemble_endpoint("http://target-ensemble-api:8000/infer/debug", False),
            "http://target-ensemble-api:8000/infer",
        )
        self.assertEqual(
            build_target_ensemble_endpoint("http://target-ensemble-api:8000", False),
            "http://target-ensemble-api:8000/infer",
        )

    def test_compute_interactable_score_matches_candidate_graph_rules(self):
        self.assertAlmostEqual(
            compute_interactable_score(0.50, "button", "ocr_word", ["click"]),
            0.80,
        )
        self.assertAlmostEqual(
            compute_interactable_score(0.50, "", "omniparser-v2", []),
            0.68,
        )

    def test_should_merge_shadow_node_accepts_same_text_nearby(self):
        group = {
            "box": [100, 100, 150, 130],
            "text_key": "continue",
            "nodes": [{"interactable_score": 0.40}],
        }
        self.assertTrue(
            should_merge_shadow_node(
                norm_box=[112, 104, 162, 134],
                text_key="continue",
                interactable_score=0.35,
                group=group,
            )
        )

    def test_should_merge_shadow_node_accepts_interactable_overlap(self):
        group = {
            "box": [100, 100, 180, 160],
            "text_key": "cancel",
            "nodes": [{"interactable_score": 0.80}],
        }
        self.assertTrue(
            should_merge_shadow_node(
                norm_box=[118, 110, 198, 170],
                text_key="continue",
                interactable_score=0.82,
                group=group,
            )
        )

    def test_resolve_execution_override_accepts_decisive_click_text_match(self):
        override = resolve_execution_override(
            action="click_text",
            params={"regex": "^Sign in$"},
            shadow_targeting={
                "auto_execute": True,
                "resolution_mode": "repair_rerun_auto_execute",
                "final_prediction": {
                    "candidate_id": "C1",
                    "text": "Sign in",
                    "score": 0.84,
                    "box": [700, 120, 820, 170],
                },
            },
            enabled=True,
        )
        self.assertEqual(override["action"], "click_box")
        self.assertEqual(override["params"]["box"], [700, 120, 820, 170])
        self.assertEqual(override["meta"]["candidate_id"], "C1")

    def test_resolve_execution_override_rejects_unsupported_action_or_text_mismatch(self):
        self.assertIsNone(
            resolve_execution_override(
                action="click_text",
                params={"regex": "^Sign in$"},
                shadow_targeting={
                    "auto_execute": True,
                    "final_prediction": {"text": "Cancel", "box": [430, 360, 640, 410]},
                },
                enabled=True,
            )
        )
        self.assertIsNone(
            resolve_execution_override(
                action="click_near_text",
                params={"anchor_regex": "^Sign in$"},
                shadow_targeting={
                    "auto_execute": True,
                    "final_prediction": {"text": "Sign in", "box": [700, 120, 820, 170]},
                },
                enabled=True,
            )
        )

    def test_resolve_execution_override_accepts_click_box_refinement_inside_requested_region(self):
        override = resolve_execution_override(
            action="click_box",
            params={"box": [640, 80, 900, 220]},
            shadow_targeting={
                "auto_execute": True,
                "final_prediction": {
                    "candidate_id": "C1",
                    "text": "Sign in",
                    "score": 0.81,
                    "box": [700, 120, 820, 170],
                },
            },
            enabled=True,
        )
        self.assertEqual(override["action"], "click_box")
        self.assertEqual(override["params"]["box"], [700, 120, 820, 170])

    def test_resolve_execution_override_rejects_click_box_when_far_from_requested_region(self):
        self.assertIsNone(
            resolve_execution_override(
                action="click_box",
                params={"box": [50, 50, 150, 120]},
                shadow_targeting={
                    "auto_execute": True,
                    "final_prediction": {
                        "candidate_id": "C9",
                        "text": "Sign in",
                        "score": 0.81,
                        "box": [700, 120, 820, 170],
                    },
                },
                enabled=True,
            )
        )

    def test_resolve_execution_override_accepts_click_near_text_when_predicted_box_contains_offset_point(self):
        override = resolve_execution_override(
            action="click_near_text",
            params={"anchor_regex": "^Email$", "dx": 120, "dy": 0},
            resolved_target={"box": [100, 100, 180, 140], "text": "Email"},
            shadow_targeting={
                "auto_execute": True,
                "final_prediction": {
                    "candidate_id": "C3",
                    "text": "Continue",
                    "score": 0.78,
                    "box": [240, 100, 320, 150],
                },
            },
            enabled=True,
        )
        self.assertEqual(override["params"]["box"], [240, 100, 320, 150])

    def test_resolve_execution_override_rejects_click_near_text_when_predicted_box_misses_offset_point(self):
        self.assertIsNone(
            resolve_execution_override(
                action="click_near_text",
                params={"anchor_regex": "^Email$", "dx": 120, "dy": 0},
                resolved_target={"box": [100, 100, 180, 140], "text": "Email"},
                shadow_targeting={
                    "auto_execute": True,
                    "final_prediction": {
                        "candidate_id": "C4",
                        "text": "Continue",
                        "score": 0.78,
                        "box": [360, 100, 420, 150],
                    },
                },
                enabled=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
