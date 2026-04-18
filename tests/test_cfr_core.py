import unittest

from python.cfr_core import InfoSet


class CFRCoreInfoSetTest(unittest.TestCase):
    def test_positive_regrets_normalize_to_current_strategy(self):
        info_set = InfoSet(actions=("check", "bet"), regret_sum=[3.0, 1.0], strategy_sum=[0.0, 0.0])

        self.assertEqual(info_set.current_strategy(), [0.75, 0.25])

    def test_average_strategy_uses_uniform_when_untrained(self):
        info_set = InfoSet(actions=("fold", "call"), regret_sum=[0.0, 0.0], strategy_sum=[0.0, 0.0])

        self.assertEqual(info_set.average_strategy(), {"fold": 0.5, "call": 0.5})


if __name__ == "__main__":
    unittest.main()
