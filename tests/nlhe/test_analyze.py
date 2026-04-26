import unittest

from python.nlhe.analyze import analyze_river_spot


class AnalyzeRiverSpotTest(unittest.TestCase):
    def test_nut_vs_air_recommendation(self):
        result = analyze_river_spot(
            board=["Ad", "Kh", "7s", "3c", "2d"],
            pot=100,
            stacks=(200, 200),
            hero_range="AhAc, AsAc",  # two nut combos
            villain_range="2h2s",
            first_to_act=0,
            hero_hand=["As", "Ac"],
            iterations=200,
            max_raises=2,
        )
        self.assertEqual(result["game"], "nlhe_river_solve")
        self.assertEqual(result["acting_player"], "hero")
        self.assertGreater(result["hero_ev"], 70.0)
        self.assertIn("AsAc", result["strategy"])
        rec = result["recommendation"]
        self.assertEqual(rec["hand"], "AsAc")
        # All betting actions should outweigh "check"
        check_prob = next(
            (a["probability"] for a in rec["actions"] if a["action"] == "check"), 0.0
        )
        bet_prob = 1.0 - check_prob
        self.assertGreater(bet_prob, 0.5)

    def test_hero_hand_outside_range_emits_note(self):
        result = analyze_river_spot(
            board=["Ad", "Kh", "7s", "3c", "2d"],
            pot=100,
            stacks=(100, 100),
            hero_range="AA",
            villain_range="KQ",
            first_to_act=0,
            hero_hand=["Qs", "Js"],  # not in AA
            iterations=100,
        )
        rec = result["recommendation"]
        self.assertIn("note", rec)
        self.assertIn("Qs", rec["hand"])
        self.assertIn("Js", rec["hand"])

    def test_strategy_probabilities_sum_to_one(self):
        result = analyze_river_spot(
            board=["Ad", "Kh", "7s", "3c", "2d"],
            pot=100,
            stacks=(100, 100),
            hero_range="AhAc, QsJs",
            villain_range="JcJd, TcTd",
            first_to_act=0,
            iterations=100,
        )
        for hand, probs in result["strategy"].items():
            total = sum(probs.values())
            self.assertAlmostEqual(total, 1.0, places=5, msg=f"{hand}: {probs}")

    def test_bad_range_string_raises(self):
        with self.assertRaises(Exception):
            analyze_river_spot(
                board=["Ad", "Kh", "7s", "3c", "2d"],
                pot=100,
                stacks=(100, 100),
                hero_range="not-a-range",
                villain_range="AA",
                first_to_act=0,
                iterations=50,
            )

    def test_ip_hero_emits_acting_player_note(self):
        result = analyze_river_spot(
            board=["Ad", "Kh", "7s", "3c", "2d"],
            pot=100,
            stacks=(100, 100),
            hero_range="AhAc",
            villain_range="KsKc, QsQc",
            first_to_act=1,           # villain acts first
            hero_hand=["Ah", "Ac"],
            iterations=100,
        )
        self.assertEqual(result["acting_player"], "villain")
        rec = result["recommendation"]
        self.assertIn("note", rec)


if __name__ == "__main__":
    unittest.main()
