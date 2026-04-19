import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from python.analyze_nlhe_river import analyze_nlhe_river


class NlheRiverAnalyzerTest(unittest.TestCase):
    def test_analyzer_calls_when_hero_has_100_percent_equity(self):
        result = analyze_nlhe_river(
            board=["Ah", "Kd", "7s", "2c", "2d"],
            hero_hand=["As", "Ac"],
            pot=100,
            to_call=50,
            villain_range=[["Qh", "Qs"], ["Jc", "Jh"]],
        )

        self.assertEqual(result["recommended_action"], "call")
        self.assertAlmostEqual(result["equity"], 1.0, delta=1e-9)
        self.assertGreater(result["ev_call"], 0.0)

    def test_analyzer_folds_when_hero_has_zero_equity(self):
        result = analyze_nlhe_river(
            board=["Ah", "Kd", "7s", "2c", "2d"],
            hero_hand=["Qh", "Qs"],
            pot=100,
            to_call=50,
            villain_range=[["As", "Ac"]],
        )

        self.assertEqual(result["recommended_action"], "fold")
        self.assertAlmostEqual(result["equity"], 0.0, delta=1e-9)
        self.assertLess(result["ev_call"], 0.0)

    def test_generic_cli_accepts_nlhe_river_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "river_spot.json"
            input_path.write_text(
                json.dumps(
                    {
                        "game": "nlhe_river",
                        "board": ["Ah", "Kd", "7s", "2c", "2d"],
                        "hero_hand": ["As", "Ac"],
                        "pot": 100,
                        "to_call": 50,
                        "villain_range": [["Qh", "Qs"], ["Jc", "Jh"]],
                    }
                ),
                encoding="utf-8",
            )
            output = subprocess.check_output(
                ["python3", "-m", "python.analyze_spot", "--input-file", str(input_path), "--format", "json"],
                text=True,
            )

        result = json.loads(output)
        self.assertEqual(result["game"], "nlhe_river")
        self.assertEqual(result["recommended_action"], "call")
        self.assertAlmostEqual(result["equity"], 1.0, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
