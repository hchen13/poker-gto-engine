import unittest

from python.analyze_kuhn import analyze_kuhn


class AnalyzeKuhnDecisionPointTest(unittest.TestCase):
    def test_q_facing_bet_returns_fold_call_labels(self):
        result = analyze_kuhn(hero_card="Q", history="b", iterations=20000)

        self.assertEqual(result["recommended_action"], "fold")
        self.assertIn("fold", result["strategy"])
        self.assertIn("call", result["strategy"])
        self.assertGreater(result["strategy"]["fold"], 0.6)


if __name__ == "__main__":
    unittest.main()
