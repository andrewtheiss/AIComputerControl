import unittest

from tools.ui_vision_harness import build_candidates, derive_instruction, infer_allowed_actions


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
        self.assertEqual(candidates[0]["box"], [10, 20, 90, 40])
        self.assertIn("type", candidates[1]["allowed_actions"])

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


if __name__ == "__main__":
    unittest.main()
