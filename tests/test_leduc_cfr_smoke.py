import math
import unittest

from python.leduc_cfr import train_leduc_cfr


class LeducCFRSmokeTest(unittest.TestCase):
    def test_training_returns_finite_value_and_normalized_root_strategies(self):
        result = train_leduc_cfr(iterations=50)

        self.assertTrue(math.isfinite(result["player_0_value"]))
        for card in ("J", "Q", "K"):
            self.assertIn(card, result["root_strategy"])
            mix = result["root_strategy"][card]
            self.assertAlmostEqual(sum(mix.values()), 1.0, delta=1e-6)
            self.assertGreaterEqual(mix["check"], 0.0)
            self.assertGreaterEqual(mix["bet"], 0.0)


if __name__ == "__main__":
    unittest.main()
