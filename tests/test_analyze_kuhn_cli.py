import json
import subprocess
import unittest


class AnalyzeKuhnCLITest(unittest.TestCase):
    def test_cli_returns_action_mix_for_root_spot(self):
        output = subprocess.check_output(
            [
                "python3",
                "-m",
                "python.analyze_kuhn",
                "--hero-card",
                "K",
                "--history",
                "",
                "--iterations",
                "20000",
                "--format",
                "json",
            ],
            text=True,
        )
        result = json.loads(output)

        self.assertEqual(result["game"], "kuhn")
        self.assertEqual(result["hero_card"], "K")
        self.assertEqual(result["history"], "")
        self.assertEqual(result["recommended_action"], "bet")
        self.assertAlmostEqual(sum(result["strategy"].values()), 1.0, delta=1e-6)
        self.assertGreater(result["strategy"]["bet"], 0.5)


if __name__ == "__main__":
    unittest.main()
