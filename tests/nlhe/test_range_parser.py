import unittest

from python.nlhe.cards import NUM_COMBOS, combo_index
from python.nlhe.range_parser import (
    RangeParseError,
    parse_range,
    range_to_combos,
    range_weight_total,
)


def count_nonzero(vec):
    return sum(1 for w in vec if w > 0)


class ClassExpansionTest(unittest.TestCase):
    def test_vector_has_1326_slots(self):
        vec = parse_range("AA")
        self.assertEqual(len(vec), NUM_COMBOS)

    def test_pocket_pair_is_six_combos(self):
        vec = parse_range("AA")
        self.assertEqual(count_nonzero(vec), 6)

    def test_suited_is_four_combos(self):
        vec = parse_range("AKs")
        self.assertEqual(count_nonzero(vec), 4)

    def test_offsuit_is_twelve_combos(self):
        vec = parse_range("AKo")
        self.assertEqual(count_nonzero(vec), 12)

    def test_unqualified_two_cards_is_sixteen_combos(self):
        vec = parse_range("AK")
        self.assertEqual(count_nonzero(vec), 16)

    def test_suited_and_offsuit_disjoint_cover_all(self):
        s = parse_range("AKs")
        o = parse_range("AKo")
        full = parse_range("AK")
        for i in range(NUM_COMBOS):
            self.assertAlmostEqual(full[i], max(s[i], o[i]))

    def test_case_insensitive_ranks(self):
        self.assertEqual(parse_range("aks"), parse_range("AKs"))

    def test_rank_order_normalizes(self):
        self.assertEqual(parse_range("KAs"), parse_range("AKs"))


class PlusRangeTest(unittest.TestCase):
    def test_pair_plus_includes_all_higher_pairs(self):
        vec = parse_range("TT+")
        self.assertEqual(count_nonzero(vec), 5 * 6)

    def test_suited_plus_same_high_card(self):
        vec = parse_range("A2s+")
        self.assertEqual(count_nonzero(vec), 12 * 4)

    def test_offsuit_plus_same_high_card(self):
        vec = parse_range("A5o+")
        self.assertEqual(count_nonzero(vec), 9 * 12)

    def test_AKs_plus_is_just_AKs(self):
        # AKs is the strongest non-pair suited; AKs+ is accepted as AKs itself
        self.assertEqual(parse_range("AKs+"), parse_range("AKs"))


class DashRangeTest(unittest.TestCase):
    def test_pair_dash(self):
        vec = parse_range("55-77")
        self.assertEqual(count_nonzero(vec), 3 * 6)

    def test_suited_connector_dash(self):
        vec = parse_range("T9s-76s")
        self.assertEqual(count_nonzero(vec), 4 * 4)

    def test_offsuit_connector_dash(self):
        vec = parse_range("T9o-76o")
        self.assertEqual(count_nonzero(vec), 4 * 12)

    def test_dash_requires_same_gap(self):
        with self.assertRaises(RangeParseError):
            parse_range("T9s-63s")


class SpecificComboTest(unittest.TestCase):
    def test_specific_combo_is_one_slot(self):
        vec = parse_range("AhKs")
        self.assertEqual(count_nonzero(vec), 1)
        idx = combo_index("Ah", "Ks")
        self.assertEqual(vec[idx], 1.0)

    def test_specific_combo_rejects_duplicate(self):
        with self.assertRaises((RangeParseError, ValueError)):
            parse_range("AhAh")


class WeightTest(unittest.TestCase):
    def test_default_weight_is_one(self):
        vec = parse_range("AA")
        self.assertEqual(range_weight_total(vec), 6.0)

    def test_explicit_weight(self):
        vec = parse_range("AA:0.5")
        self.assertAlmostEqual(range_weight_total(vec), 3.0)

    def test_later_tokens_override_weight(self):
        # Start with AA at full weight, then reduce to 0.5
        vec = parse_range("AA, AA:0.5")
        self.assertAlmostEqual(range_weight_total(vec), 3.0)

    def test_zero_weight_effectively_removes(self):
        vec = parse_range("QQ+, QQ:0")
        self.assertEqual(count_nonzero(vec), 2 * 6)

    def test_weight_out_of_range_rejected(self):
        with self.assertRaises(RangeParseError):
            parse_range("AA:1.5")


class MultiTokenTest(unittest.TestCase):
    def test_comma_separated(self):
        vec = parse_range("AA, KK, QQ")
        self.assertEqual(count_nonzero(vec), 3 * 6)

    def test_whitespace_separated(self):
        vec = parse_range("AA KK QQ")
        self.assertEqual(count_nonzero(vec), 3 * 6)

    def test_mixed_tokens(self):
        vec = parse_range("TT+, AQs+, AKo")
        expected = 5 * 6 + 2 * 4 + 12
        self.assertEqual(count_nonzero(vec), expected)


class ErrorTest(unittest.TestCase):
    def test_invalid_rank_rejected(self):
        with self.assertRaises(RangeParseError):
            parse_range("XX")

    def test_invalid_suit_marker_rejected(self):
        with self.assertRaises(RangeParseError):
            parse_range("AKx")

    def test_unknown_form_rejected(self):
        with self.assertRaises(RangeParseError):
            parse_range("AKsoo")


class ConvertBackTest(unittest.TestCase):
    def test_combo_iteration_matches_count(self):
        vec = parse_range("AA")
        combos = range_to_combos(vec)
        self.assertEqual(len(combos), 6)
        for (c1, c2), w in combos:
            self.assertEqual(w, 1.0)
            self.assertEqual(c1[0], "A")
            self.assertEqual(c2[0], "A")
            self.assertNotEqual(c1, c2)


if __name__ == "__main__":
    unittest.main()
