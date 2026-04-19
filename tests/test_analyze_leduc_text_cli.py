import subprocess
import unittest


class AnalyzeLeducTextCLITest(unittest.TestCase):
    def test_leduc_text_output_runs_without_key_error(self):
        output = subprocess.check_output(
            ["./scripts/e2e_leduc_demo.sh"],
            text=True,
        )
        self.assertIn("game: leduc", output)
        self.assertIn("recommended_action:", output)
        self.assertIn("exploitability:", output)


if __name__ == "__main__":
    unittest.main()
