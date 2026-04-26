import unittest

from python.nlhe.hand_eval import (
    CATEGORY_FLUSH,
    CATEGORY_FULL_HOUSE,
    CATEGORY_HIGH_CARD,
    CATEGORY_PAIR,
    CATEGORY_QUADS,
    CATEGORY_STRAIGHT,
    CATEGORY_STRAIGHT_FLUSH,
    CATEGORY_TRIPS,
    CATEGORY_TWO_PAIR,
    evaluate_any,
    evaluate_seven,
)


class FiveCardCategoryTest(unittest.TestCase):
    """Every category must be detected correctly at 5-card level."""

    def test_high_card(self):
        self.assertEqual(evaluate_any(["2h", "5d", "7c", "9s", "Kh"])[0], CATEGORY_HIGH_CARD)

    def test_pair(self):
        self.assertEqual(evaluate_any(["Ah", "Ad", "7c", "9s", "Kh"])[0], CATEGORY_PAIR)

    def test_two_pair(self):
        self.assertEqual(evaluate_any(["Ah", "Ad", "7c", "7s", "Kh"])[0], CATEGORY_TWO_PAIR)

    def test_trips(self):
        self.assertEqual(evaluate_any(["Ah", "Ad", "Ac", "7s", "Kh"])[0], CATEGORY_TRIPS)

    def test_wheel_straight(self):
        cat, kickers = evaluate_any(["Ah", "2d", "3c", "4s", "5h"])
        self.assertEqual(cat, CATEGORY_STRAIGHT)
        self.assertEqual(kickers, (5,))

    def test_broadway_straight(self):
        cat, kickers = evaluate_any(["Th", "Jd", "Qc", "Ks", "Ah"])
        self.assertEqual(cat, CATEGORY_STRAIGHT)
        self.assertEqual(kickers, (14,))

    def test_mid_straight(self):
        cat, kickers = evaluate_any(["5h", "6d", "7c", "8s", "9h"])
        self.assertEqual(cat, CATEGORY_STRAIGHT)
        self.assertEqual(kickers, (9,))

    def test_flush(self):
        self.assertEqual(evaluate_any(["2h", "5h", "7h", "9h", "Kh"])[0], CATEGORY_FLUSH)

    def test_full_house(self):
        self.assertEqual(evaluate_any(["Ah", "Ad", "Ac", "7s", "7h"])[0], CATEGORY_FULL_HOUSE)

    def test_quads(self):
        self.assertEqual(evaluate_any(["Ah", "Ad", "Ac", "As", "Kh"])[0], CATEGORY_QUADS)

    def test_straight_flush(self):
        cat, kickers = evaluate_any(["5h", "6h", "7h", "8h", "9h"])
        self.assertEqual(cat, CATEGORY_STRAIGHT_FLUSH)
        self.assertEqual(kickers, (9,))

    def test_steel_wheel(self):
        cat, kickers = evaluate_any(["Ah", "2h", "3h", "4h", "5h"])
        self.assertEqual(cat, CATEGORY_STRAIGHT_FLUSH)
        self.assertEqual(kickers, (5,))

    def test_royal_flush(self):
        cat, kickers = evaluate_any(["Th", "Jh", "Qh", "Kh", "Ah"])
        self.assertEqual(cat, CATEGORY_STRAIGHT_FLUSH)
        self.assertEqual(kickers, (14,))


class TieBreakTest(unittest.TestCase):
    """Within a category, stronger hand must beat weaker."""

    def test_higher_pair_wins(self):
        aa = evaluate_any(["Ah", "Ad", "7c", "9s", "Kh"])
        kk = evaluate_any(["Kh", "Kd", "7c", "9s", "Jh"])
        self.assertGreater(aa, kk)

    def test_pair_kicker_matters(self):
        aa_k = evaluate_any(["Ah", "Ad", "Kc", "9s", "2h"])
        aa_q = evaluate_any(["As", "Ac", "Qc", "9d", "2d"])
        self.assertGreater(aa_k, aa_q)

    def test_quads_over_full_house(self):
        quads = evaluate_any(["Ah", "Ad", "Ac", "As", "2h"])
        boat = evaluate_any(["Kh", "Kd", "Kc", "Qs", "Qh"])
        self.assertGreater(quads, boat)

    def test_flush_over_straight(self):
        flush = evaluate_any(["2h", "5h", "7h", "9h", "Kh"])
        straight = evaluate_any(["5s", "6d", "7c", "8h", "9s"])
        self.assertGreater(flush, straight)

    def test_higher_flush_wins(self):
        ahi = evaluate_any(["Ah", "5h", "7h", "9h", "Kh"])
        khi = evaluate_any(["Kd", "5d", "7d", "9d", "Qd"])
        self.assertGreater(ahi, khi)

    def test_wheel_loses_to_six_high_straight(self):
        wheel = evaluate_any(["Ah", "2d", "3c", "4s", "5h"])
        six_high = evaluate_any(["2h", "3d", "4c", "5s", "6h"])
        self.assertGreater(six_high, wheel)

    def test_identical_hands_tie(self):
        a = evaluate_any(["Ah", "Kd", "Qc", "Js", "Th"])
        b = evaluate_any(["As", "Kc", "Qd", "Jh", "Td"])
        self.assertEqual(a, b)


class SevenCardTest(unittest.TestCase):
    def test_seven_cards_pick_best_five(self):
        # Board Ah Kd 7s 2c 2d + hero As Ac → full house AAA22
        rank = evaluate_seven(["Ah", "Kd", "7s", "2c", "2d", "As", "Ac"])
        self.assertEqual(rank[0], CATEGORY_FULL_HOUSE)
        self.assertEqual(rank[1], (14, 2))

    def test_seven_cards_quads(self):
        # Four twos available from 7 cards
        rank = evaluate_seven(["2h", "2d", "2s", "2c", "Ah", "Kh", "Qh"])
        self.assertEqual(rank[0], CATEGORY_QUADS)
        self.assertEqual(rank[1], (2, 14))

    def test_seven_cards_flush_beats_straight(self):
        # Possible straight (5-9) plus five hearts → flush wins
        rank = evaluate_seven(["2h", "5h", "7h", "9h", "Kh", "6s", "8s"])
        self.assertEqual(rank[0], CATEGORY_FLUSH)

    def test_seven_cards_wheel_from_board_and_hand(self):
        # Board Ah 2d 3c plus hero 4h 5s + rag = wheel
        rank = evaluate_seven(["Ah", "2d", "3c", "Kh", "Qh", "4h", "5s"])
        self.assertEqual(rank[0], CATEGORY_STRAIGHT)
        self.assertEqual(rank[1], (5,))

    def test_seven_card_rejects_wrong_count(self):
        with self.assertRaises(ValueError):
            evaluate_seven(["Ah", "Kd", "7s", "2c", "2d", "As"])


if __name__ == "__main__":
    unittest.main()
