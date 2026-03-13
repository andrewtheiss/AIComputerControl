import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.support import http_post_json, running_compose_services, wait_for_json


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


if __name__ == "__main__":
    unittest.main()
