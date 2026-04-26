import unittest

from python.nlhe.abstraction import bucket_by_ehs, compute_river_ehs
from python.nlhe.cards import NUM_COMBOS, combo_index
from python.nlhe.cfr import solve_river
from python.nlhe.range_parser import parse_range
from python.nlhe.tree import build_river_tree


class EHSSanityTest(unittest.TestCase):
    def test_ehs_strong_vs_weak(self):
        # Dry board: Ad Kh 7s 3c 2d
        board = ["Ad", "Kh", "7s", "3c", "2d"]
        ehs = compute_river_ehs(board)

        # AsAc = top set = strong
        nut_ehs = ehs[combo_index("As", "Ac")]
        # 8h9c = air (no pair, no straight possible — wheel needs 4+5, not 8+9)
        air_ehs = ehs[combo_index("8h", "9c")]
        # A-high two pair-ish: AhKd
        ak_ehs = ehs[combo_index("Ah", "Kd")]

        self.assertGreater(nut_ehs, 0.95, f"nut set EHS should be near 1, got {nut_ehs}")
        self.assertLess(air_ehs, 0.5, f"air EHS should be < 0.5, got {air_ehs}")
        self.assertGreater(nut_ehs, ak_ehs)
        self.assertGreater(ak_ehs, air_ehs)

    def test_ehs_excludes_board_conflict(self):
        board = ["Ad", "Kh", "7s", "3c", "2d"]
        ehs = compute_river_ehs(board)
        # AdAc includes Ad which is on board → should not appear
        self.assertNotIn(combo_index("Ad", "Ac"), ehs)
        # AsAc has no conflict → should appear
        self.assertIn(combo_index("As", "Ac"), ehs)


class BucketingTest(unittest.TestCase):
    def test_equal_weight_buckets_sum_count(self):
        board = ["Ad", "Kh", "7s", "3c", "2d"]
        ehs = compute_river_ehs(board)
        combos = list(ehs.keys())
        weights = [1.0] * len(combos)
        bucket_of = bucket_by_ehs(combos, weights, ehs, n_buckets=5)
        self.assertEqual(len(bucket_of), len(combos))
        for b in bucket_of:
            self.assertTrue(0 <= b < 5)
        # With uniform weights, buckets should be roughly balanced
        from collections import Counter
        counts = Counter(bucket_of)
        avg = len(combos) / 5
        for b, c in counts.items():
            self.assertLess(abs(c - avg) / avg, 0.2, f"bucket {b} has {c}, avg {avg}")

    def test_ordering_monotone_in_ehs(self):
        """Higher E[HS] combos should map to higher bucket ids."""
        board = ["Ad", "Kh", "7s", "3c", "2d"]
        ehs = compute_river_ehs(board)
        combos = list(ehs.keys())
        weights = [1.0] * len(combos)
        bucket_of = bucket_by_ehs(combos, weights, ehs, n_buckets=5)
        # Max E[HS] bucket should be >= min E[HS] bucket
        max_c = max(combos, key=lambda c: ehs[c])
        min_c = min(combos, key=lambda c: ehs[c])
        b_max = bucket_of[combos.index(max_c)]
        b_min = bucket_of[combos.index(min_c)]
        self.assertGreater(b_max, b_min)


class BucketedSolverEquivalenceTest(unittest.TestCase):
    """With n_buckets >= n_combos, bucketed solver should match unabstracted."""

    def test_small_range_bucketed_matches_exact(self):
        board = ["Ad", "Kh", "7s", "3c", "2d"]
        hero_range = parse_range("AhAc, QsJs")
        villain_range = parse_range("JcJd, TcTd")

        root = build_river_tree(pot=100, stacks=(100, 100), first_to_act=0)

        exact = solve_river(
            root=root,
            hero_range=hero_range,
            villain_range=villain_range,
            board=board,
            iterations=200,
        )

        # Per-combo buckets: each global combo in its own bucket (identity).
        hero_buckets = list(range(NUM_COMBOS))
        villain_buckets = list(range(NUM_COMBOS))
        bucketed = solve_river(
            root=root,
            hero_range=hero_range,
            villain_range=villain_range,
            board=board,
            iterations=200,
            hero_buckets=hero_buckets,
            villain_buckets=villain_buckets,
        )

        # Should be essentially identical (same trajectory)
        self.assertAlmostEqual(exact.hero_value, bucketed.hero_value, places=4)


class BucketedSolverScalesTest(unittest.TestCase):
    """Bucketed solver should produce sensible output on a multi-combo range."""

    def test_polarized_range_with_bucketing(self):
        board = ["Ad", "Kh", "7s", "3c", "2d"]
        # Hero: top pair+ range
        hero_range_str = "AA, KK, AKs, AKo, AQs"
        villain_range_str = "JJ, TT, 99, KQs, KJs"
        hero_range = parse_range(hero_range_str)
        villain_range = parse_range(villain_range_str)

        ehs = compute_river_ehs(board)

        # Local combo lists as seen by showdown filter
        from python.nlhe.showdown import compute_showdown_table

        table = compute_showdown_table(board, hero_range, villain_range)
        self.assertGreater(len(table.hero_combos), 10)

        # Build global-combo→bucket maps (4 buckets each side).
        hero_bucket_of_local = bucket_by_ehs(
            table.hero_combos, table.hero_weights, ehs, n_buckets=4
        )
        villain_bucket_of_local = bucket_by_ehs(
            table.villain_combos, table.villain_weights, ehs, n_buckets=4
        )
        hero_buckets = [0] * NUM_COMBOS
        for local_i, gi in enumerate(table.hero_combos):
            hero_buckets[gi] = hero_bucket_of_local[local_i]
        villain_buckets = [0] * NUM_COMBOS
        for local_j, gj in enumerate(table.villain_combos):
            villain_buckets[gj] = villain_bucket_of_local[local_j]

        root = build_river_tree(pot=100, stacks=(100, 100), first_to_act=0)
        result = solve_river(
            root=root,
            hero_range=hero_range,
            villain_range=villain_range,
            board=board,
            iterations=150,
            hero_buckets=hero_buckets,
            villain_buckets=villain_buckets,
        )

        # Hero should have positive EV (nutted range on A-K-high board)
        self.assertGreater(result.hero_value, 0)

        # Combos in same bucket must have identical strategies.
        by_bucket = {}
        for combo_idx, probs in result.root_strategy.items():
            b = hero_buckets[combo_idx]
            key = tuple(sorted(probs.items()))
            by_bucket.setdefault(b, []).append(key)
        for b, rows in by_bucket.items():
            self.assertEqual(
                len(set(rows)), 1, f"bucket {b} has differing strategies: {rows}"
            )


if __name__ == "__main__":
    unittest.main()
