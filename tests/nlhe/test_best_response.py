import unittest

from python.nlhe.best_response import (
    best_response_value,
    compute_exploitability,
    exploitability,
)
from python.nlhe.cfr import build_and_train
from python.nlhe.range_parser import parse_range
from python.nlhe.tree import build_river_tree


class ExploitabilityOnSolvedSpotsTest(unittest.TestCase):
    def test_exploitability_is_small_on_small_spot(self):
        board = ["Ad", "Kh", "7s", "3c", "2d"]
        hero_range = parse_range("AhAc, QsJs")
        villain_range = parse_range("JcJd, TcTd")
        root = build_river_tree(pot=100, stacks=(100, 100), first_to_act=0, max_raises=2)

        hero_value, br_h, br_v, exp = compute_exploitability(
            root=root,
            hero_range=hero_range,
            villain_range=villain_range,
            board=board,
            iterations=500,
        )

        # Both BR values should be achievable (>= avg-strategy values)
        self.assertGreaterEqual(br_h + 0.5, hero_value)
        # Exploitability non-negative up to small numerical slack
        self.assertGreater(exp, -1.0, f"exploitability {exp} too negative — BR bug?")
        # Below 10% of pot is a reasonable bar for 500 iters on this tiny spot
        self.assertLess(exp, 10.0, f"exploitability {exp} too large (pot=100)")

    def test_exploitability_decreases_with_iterations(self):
        """Classic convergence witness: more iterations should drive exploitability down."""
        board = ["Ad", "Kh", "7s", "3c", "2d"]
        hero_range = parse_range("AhAc, QsJs")
        villain_range = parse_range("JcJd, TcTd")
        root_factory = lambda: build_river_tree(
            pot=100, stacks=(100, 100), first_to_act=0, max_raises=2
        )

        _, _, _, exp_low = compute_exploitability(
            root=root_factory(),
            hero_range=hero_range,
            villain_range=villain_range,
            board=board,
            iterations=50,
        )
        _, _, _, exp_high = compute_exploitability(
            root=root_factory(),
            hero_range=hero_range,
            villain_range=villain_range,
            board=board,
            iterations=800,
        )

        self.assertLess(
            exp_high, exp_low,
            f"exploitability should drop with more iterations: 50→{exp_low:.3f}, 800→{exp_high:.3f}",
        )

    def test_exploitability_non_negative(self):
        """BR_hero + BR_villain >= initial_pot (exploitability is always >= 0).
        At Nash, equality holds; away from Nash, sum strictly exceeds the pot."""
        board = ["Ad", "Kh", "7s", "3c", "2d"]
        hero_range = parse_range("AhAc, KsKc, QsJs")
        villain_range = parse_range("JcJd, TcTd, AcQc")
        root = build_river_tree(pot=100, stacks=(150, 150), first_to_act=0, max_raises=2)

        solver = build_and_train(
            root, hero_range, villain_range, board, iterations=300,
        )
        br_h = best_response_value(solver, br_player=0)
        br_v = best_response_value(solver, br_player=1)

        # Sum must be >= pot (up to small numerical slack)
        self.assertGreaterEqual(br_h + br_v + 0.5, solver.root.pot)


class ExploitabilityWithAbstractionTest(unittest.TestCase):
    """Bucketing makes strategies worse (can't perfectly distinguish hands) —
    exploitability should be higher than unabstracted, but still non-pathological."""

    def test_bucketed_still_bounded(self):
        from python.nlhe.abstraction import bucket_by_ehs, compute_river_ehs
        from python.nlhe.cards import NUM_COMBOS
        from python.nlhe.showdown import compute_showdown_table

        board = ["Ad", "Kh", "7s", "3c", "2d"]
        hero_range = parse_range("AA, KK, AKs, AKo, AQs")
        villain_range = parse_range("JJ, TT, 99, KQs")

        ehs = compute_river_ehs(board)
        table = compute_showdown_table(board, hero_range, villain_range)
        hero_local = bucket_by_ehs(table.hero_combos, table.hero_weights, ehs, n_buckets=3)
        villain_local = bucket_by_ehs(table.villain_combos, table.villain_weights, ehs, n_buckets=3)
        hero_buckets = [0] * NUM_COMBOS
        for li, gi in enumerate(table.hero_combos):
            hero_buckets[gi] = hero_local[li]
        villain_buckets = [0] * NUM_COMBOS
        for lj, gj in enumerate(table.villain_combos):
            villain_buckets[gj] = villain_local[lj]

        root = build_river_tree(pot=100, stacks=(100, 100), first_to_act=0, max_raises=2)
        _, _, _, exp = compute_exploitability(
            root=root,
            hero_range=hero_range,
            villain_range=villain_range,
            board=board,
            iterations=200,
            hero_buckets=hero_buckets,
            villain_buckets=villain_buckets,
        )
        # Bucketed exploitability should still be bounded (< 30 chips on pot 100).
        self.assertLess(exp, 30.0, f"bucketed exploitability {exp} too large")
        self.assertGreater(exp, -1.0)


if __name__ == "__main__":
    unittest.main()
