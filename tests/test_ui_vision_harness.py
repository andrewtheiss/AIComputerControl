import unittest

from tools.ui_vision_harness import build_candidates, derive_instruction, infer_allowed_actions
from ui_vision_common.candidate_graph import build_candidate_graph


class UIVisionHarnessTest(unittest.TestCase):
    def test_infer_allowed_actions_prefers_textbox_actions(self):
        actions = infer_allowed_actions({"role": "textbox", "text": "Email"})
        self.assertIn("type", actions)
        self.assertIn("focus", actions)

    def test_build_candidates_preserves_boxes(self):
        candidates = build_candidates(
            [
                {"source": "ocr_word", "text": "Sign in", "box": [10, 20, 90, 40], "score": 0.95, "role": "button"},
                {"source": "ax", "text": "Email", "box": [110, 120, 250, 150], "score": 0.99, "role": "textbox"},
            ]
        )
        self.assertEqual(len(candidates), 2)
        boxes = {tuple(candidate["box"]) for candidate in candidates}
        self.assertIn((10, 20, 90, 40), boxes)
        self.assertIn((110, 120, 250, 150), boxes)
        self.assertTrue(any("type" in candidate["allowed_actions"] for candidate in candidates))
        self.assertTrue(all("bbox_rel_1000" in candidate["extras"] for candidate in candidates))
        self.assertTrue(all("interactable_score" in candidate["extras"] for candidate in candidates))

    def test_derive_instruction_uses_last_click_text(self):
        instruction = derive_instruction(
            {
                "goal": "Complete sign in",
                "task_history": [
                    {"action": "click_text", "parameters": {"regex": "^Sign in$"}},
                ],
            }
        )
        self.assertIn("Sign in", instruction)

    def test_candidate_graph_merges_overlapping_sources(self):
        graph = build_candidate_graph(
            [
                {"source": "ocr_word", "text": "Sign in", "box": [100, 40, 180, 70], "score": 0.93, "role": None},
                {"source": "ax", "text": "Sign in", "box": [96, 36, 186, 76], "score": 0.99, "role": "button"},
            ],
            viewport={"width": 1000, "height": 800},
        )
        self.assertEqual(len(graph), 1)
        self.assertEqual(graph[0]["text"], "Sign in")
        self.assertEqual(graph[0]["role_hint"], "button")
        self.assertIn("ax", graph[0]["source_mask"])
        self.assertIn("ocr_word", graph[0]["source_mask"])


if __name__ == "__main__":
    unittest.main()
