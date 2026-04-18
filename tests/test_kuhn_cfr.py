import math
import unittest

from python.kuhn_cfr import train_kuhn_cfr


class KuhnCFRTest(unittest.TestCase):
    def test_average_strategy_converges_to_known_game_value_and_equilibrium_family(self):
        result = train_kuhn_cfr(iterations=20000)

        jack_bet = result["root_strategy"]["J"]["bet"]
        queen_bet = result["root_strategy"]["Q"]["bet"]
        king_bet = result["root_strategy"]["K"]["bet"]

        self.assertTrue(math.isclose(result["player_0_value"], -1 / 18, abs_tol=0.02))
        self.assertGreaterEqual(jack_bet, 0.0)
        self.assertLessEqual(jack_bet, 1 / 3 + 0.03)
        self.assertAlmostEqual(king_bet, min(1.0, 3 * jack_bet), delta=0.08)
        self.assertAlmostEqual(queen_bet, 0.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
