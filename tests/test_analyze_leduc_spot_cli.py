import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class AnalyzeLeducSpotCLITest(unittest.TestCase):
    def test_generic_cli_accepts_leduc_root_input(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "leduc_root.json"
            input_path.write_text(
                json.dumps(
                    {
                        "game": "leduc",
                        "hero_card": "K",
                        "public_card": None,
                        "round_histories": ["", ""],
                        "iterations": 50,
                    }
                ),
                encoding="utf-8",
            )
            output = subprocess.check_output(
                ["python3", "-m", "python.analyze_spot", "--input-file", str(input_path), "--format", "json"],
                text=True,
            )

        result = json.loads(output)
        self.assertEqual(result["game"], "leduc")
        self.assertEqual(result["hero_card"], "K")
        self.assertAlmostEqual(sum(result["strategy"].values()), 1.0, delta=1e-6)
        self.assertIn(result["recommended_action"], {"check", "bet"})
        self.assertIn("exploitability", result)
        self.assertIn("best_response_player_0", result)
        self.assertIn("best_response_player_1", result)


if __name__ == "__main__":
    unittest.main()
