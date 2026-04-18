import json
import subprocess
import unittest

from python.kuhn_cfr import train_kuhn_cfr


class KuhnCrossCheckTest(unittest.TestCase):
    def test_rust_and_python_kuhn_summaries_match_closely(self):
        python_summary = train_kuhn_cfr(iterations=20000)
        cargo_output = subprocess.check_output(
            ["cargo", "run", "--quiet", "--bin", "kuhn_summary", "--", "20000"],
            text=True,
        )
        rust_summary = json.loads(cargo_output)

        self.assertAlmostEqual(rust_summary["player_0_value"], python_summary["player_0_value"], delta=0.01)
        for card in ("J", "Q", "K"):
            self.assertAlmostEqual(
                rust_summary["root_strategy"][card]["bet"],
                python_summary["root_strategy"][card]["bet"],
                delta=0.03,
            )


if __name__ == "__main__":
    unittest.main()
