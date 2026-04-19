import json
import subprocess
import unittest

from python.leduc_cfr import train_leduc_cfr


class LeducCrossCheckTest(unittest.TestCase):
    def test_rust_and_python_leduc_summaries_match_closely(self):
        python_summary = train_leduc_cfr(iterations=100)
        cargo_output = subprocess.check_output(
            ["cargo", "run", "--quiet", "--bin", "leduc_summary", "--", "100"],
            text=True,
        )
        rust_summary = json.loads(cargo_output)

        self.assertAlmostEqual(rust_summary["player_0_value"], python_summary["player_0_value"], delta=0.01)
        self.assertAlmostEqual(rust_summary["exploitability"], python_summary["exploitability"], delta=0.02)
        for card in ("J", "Q", "K"):
            self.assertAlmostEqual(
                rust_summary["root_strategy"][card]["bet"],
                python_summary["root_strategy"][card]["bet"],
                delta=0.03,
            )


if __name__ == "__main__":
    unittest.main()
