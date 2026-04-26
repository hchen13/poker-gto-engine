import unittest

from python.nlhe.flop_cfr import solve_flop
from python.nlhe.flop_showdown import compute_flop_showdown
from python.nlhe.range_parser import parse_range
from python.nlhe.tree import build_flop_tree, count_chance_nodes, walk_nodes


class FlopTreeStructureTest(unittest.TestCase):
    def test_flop_tree_has_two_chance_levels(self):
        tree = build_flop_tree(
            board_3=["Ad", "Kh", "7s"], pot=50, stacks=(80, 80),
            first_to_act=0, max_raises=1, turn_max_raises=1, river_max_raises=1,
        )
        # Outer chance has 49 children (turn cards), each turn subtree contains
        # inner chance(48) at its showdowns.
        # Just count any chance-with-49 and any chance-with-48 in the tree.
        outer = sum(1 for n in walk_nodes(tree) if n.is_chance and len(n.children) == 49)
        inner = sum(1 for n in walk_nodes(tree) if n.is_chance and len(n.children) == 48)
        self.assertGreater(outer, 0)
        self.assertGreater(inner, 0)

    def test_deep_showdowns_have_tuple_keys(self):
        tree = build_flop_tree(
            board_3=["Ad", "Kh", "7s"], pot=50, stacks=(80, 80),
            first_to_act=0, max_raises=1, turn_max_raises=1, river_max_raises=1,
        )
        showdowns = [
            n for n in walk_nodes(tree)
            if n.is_terminal and n.terminal_winner is None
        ]
        self.assertGreater(len(showdowns), 0)
        for n in showdowns:
            self.assertIsInstance(n.showdown_board_key, tuple)
            self.assertEqual(len(n.showdown_board_key), 2)


class FlopShowdownTest(unittest.TestCase):
    def test_outcomes_dict_has_expected_size(self):
        hero_range = parse_range("AhAc")
        villain_range = parse_range("KsKc")
        table = compute_flop_showdown(
            board_3=["Ad", "Kh", "7s"], hero_range=hero_range, villain_range=villain_range,
        )
        # Expected runouts: 49 turn cards × 48 rivers (excluding self) = 49*48 = 2352
        # But only kept distinct (turn, river) pairs where river ≠ turn.
        self.assertEqual(len(table.outcomes_by_runout), 49 * 48)


class FlopSolverSanityTest(unittest.TestCase):
    def test_nut_vs_air_positive_ev(self):
        """Hero has top set on dry flop vs villain's air. Hero EV should be high."""
        board = ["Ad", "Kh", "7s"]
        hero_range = parse_range("AhAc")
        villain_range = parse_range("3c2c")  # total air, no draw, low cards
        tree = build_flop_tree(
            board_3=board, pot=20, stacks=(40, 40), first_to_act=0,
            max_raises=1, turn_max_raises=1, river_max_raises=1,
        )
        result = solve_flop(
            root=tree, hero_range=hero_range, villain_range=villain_range,
            board_3=board, iterations=15,
        )
        self.assertGreater(result.hero_value, 5.0)


if __name__ == "__main__":
    unittest.main()
