import math
import unittest

from python.leduc_cfr import train_leduc_cfr


class LeducCFRSmokeTest(unittest.TestCase):
    def test_training_returns_reference_shaped_root_strategy(self):
        result = train_leduc_cfr(iterations=100)

        self.assertTrue(math.isfinite(result["player_0_value"]))
        self.assertAlmostEqual(result["player_0_value"], -0.08553, delta=0.03)
        for card in ("J", "Q", "K"):
            self.assertIn(card, result["root_strategy"])
            mix = result["root_strategy"][card]
            self.assertAlmostEqual(sum(mix.values()), 1.0, delta=1e-6)
            self.assertGreaterEqual(mix["check"], 0.0)
            self.assertGreaterEqual(mix["bet"], 0.0)

        self.assertLess(result["root_strategy"]["J"]["bet"], 0.2)
        self.assertGreater(result["root_strategy"]["Q"]["bet"], 0.6)
        self.assertGreater(result["root_strategy"]["K"]["bet"], 0.6)


if __name__ == "__main__":
    unittest.main()
