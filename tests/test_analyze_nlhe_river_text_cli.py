import subprocess
import unittest


class AnalyzeNlheRiverTextCLITest(unittest.TestCase):
    def test_nlhe_river_text_output_runs_without_key_error(self):
        output = subprocess.check_output(
            ["./scripts/e2e_nlhe_river_demo.sh"],
            text=True,
        )
        self.assertIn("game: nlhe_river", output)
        self.assertIn("recommended_action: call", output)
        self.assertIn("equity:", output)
        self.assertIn("ev_call:", output)


if __name__ == "__main__":
    unittest.main()
