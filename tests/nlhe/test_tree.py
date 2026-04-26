import unittest

from python.nlhe.tree import (
    Action,
    build_river_tree,
    count_decision_nodes,
    count_nodes,
    count_terminals,
    walk_nodes,
)


class RootShapeTest(unittest.TestCase):
    def test_check_is_always_first_action_at_root(self):
        root = build_river_tree(pot=100, stacks=(500, 500))
        self.assertEqual(root.actions[0].kind, "check")

    def test_root_offers_all_bet_fractions_plus_allin(self):
        root = build_river_tree(pot=100, stacks=(500, 500))
        kinds = [a.kind for a in root.actions]
        # check + 5 pot-fraction bets + allin
        self.assertEqual(kinds[0], "check")
        self.assertEqual(kinds.count("bet"), 5)
        self.assertEqual(kinds.count("allin"), 1)

    def test_root_with_shallow_stack_drops_oversized_bets(self):
        # pot 100, stack 20 → only bet sizes < 20 are possible; everything else
        # collapses to all-in
        root = build_river_tree(pot=100, stacks=(20, 20))
        kinds = [a.kind for a in root.actions]
        # 33% of 100 = 33 > 20 so all bets collapse, just check + all-in
        self.assertEqual(kinds, ["check", "allin"])

    def test_terminal_pot_respects_initial_pot_on_check_check(self):
        root = build_river_tree(pot=100, stacks=(500, 500))
        # Follow check -> check path
        check_child = root.children[0]
        self.assertFalse(check_child.is_terminal)
        # Second player's check is child[0]
        showdown = check_child.children[0]
        self.assertTrue(showdown.is_terminal)
        self.assertIsNone(showdown.terminal_winner)
        self.assertEqual(showdown.terminal_pot, 100)


class FoldAndCallTest(unittest.TestCase):
    def test_fold_after_bet_gives_pot_to_bettor(self):
        root = build_river_tree(pot=100, stacks=(500, 500))
        # Find the first 'bet' action and follow to opponent's decision
        bet_idx = next(i for i, a in enumerate(root.actions) if a.kind == "bet")
        opp_node = root.children[bet_idx]
        fold_idx = next(i for i, a in enumerate(opp_node.actions) if a.kind == "fold")
        folded = opp_node.children[fold_idx]
        self.assertTrue(folded.is_terminal)
        # Player 0 bet, player 1 folds → player 0 wins
        self.assertEqual(folded.terminal_winner, 0)
        # Pot at fold = pot before bet + bet amount committed by player 0
        self.assertAlmostEqual(folded.terminal_pot, 100 + root.actions[bet_idx].amount)

    def test_call_ends_in_showdown_with_correct_pot(self):
        root = build_river_tree(pot=100, stacks=(500, 500))
        # Bet pot (size 100)
        bet_idx = next(
            i for i, a in enumerate(root.actions) if a.kind == "bet" and abs(a.amount - 100) < 1e-6
        )
        after_bet = root.children[bet_idx]
        call_idx = next(i for i, a in enumerate(after_bet.actions) if a.kind == "call")
        called = after_bet.children[call_idx]
        self.assertTrue(called.is_terminal)
        self.assertIsNone(called.terminal_winner)
        # Pot after both sides committed 100 = 100 + 100 + 100 = 300
        self.assertAlmostEqual(called.terminal_pot, 300)


class StacksAfterActionTest(unittest.TestCase):
    def test_stacks_decrease_correctly_after_bet_and_call(self):
        root = build_river_tree(pot=100, stacks=(500, 500))
        bet_idx = next(
            i for i, a in enumerate(root.actions) if a.kind == "bet" and abs(a.amount - 100) < 1e-6
        )
        after_bet = root.children[bet_idx]
        self.assertEqual(after_bet.stacks, (400, 500))
        call_idx = next(i for i, a in enumerate(after_bet.actions) if a.kind == "call")
        called = after_bet.children[call_idx]
        self.assertEqual(called.stacks, (400, 400))


class RaiseBehaviorTest(unittest.TestCase):
    def test_facing_bet_offers_fold_call_raise_allin(self):
        root = build_river_tree(pot=100, stacks=(500, 500))
        bet_idx = next(i for i, a in enumerate(root.actions) if a.kind == "bet")
        opp = root.children[bet_idx]
        kinds = [a.kind for a in opp.actions]
        self.assertEqual(kinds[0], "fold")
        self.assertEqual(kinds[1], "call")
        self.assertIn("raise", kinds)
        self.assertIn("allin", kinds)

    def test_raise_budget_prevents_infinite_loop(self):
        # With raises_left cap, we should get a finite tree
        root = build_river_tree(pot=100, stacks=(10000, 10000), max_raises=3)
        n_nodes = count_nodes(root)
        self.assertLess(n_nodes, 10000)  # finite, reasonable


class TerminalityTest(unittest.TestCase):
    def test_all_paths_terminate(self):
        root = build_river_tree(pot=100, stacks=(200, 200))
        for node in walk_nodes(root):
            if not node.is_terminal:
                self.assertGreater(len(node.children), 0)
            else:
                self.assertEqual(len(node.children), 0)

    def test_terminal_node_count_matches_leaves(self):
        root = build_river_tree(pot=100, stacks=(200, 200))
        self.assertEqual(count_nodes(root), count_terminals(root) + count_decision_nodes(root))


class EnergyConservationTest(unittest.TestCase):
    """For every terminal node, chips in pot + both players' remaining stacks
    must equal the initial (pot + stack0 + stack1). No chips appear or vanish."""

    def test_chip_conservation_tight_stacks(self):
        initial_pot = 100
        initial_stacks = (200, 200)
        total = initial_pot + sum(initial_stacks)
        root = build_river_tree(pot=initial_pot, stacks=initial_stacks)
        for node in walk_nodes(root):
            if node.is_terminal:
                observed = node.terminal_pot + node.stacks[0] + node.stacks[1]
                self.assertAlmostEqual(observed, total, places=6)

    def test_chip_conservation_deep_stacks(self):
        initial_pot = 100
        initial_stacks = (1000, 1000)
        total = initial_pot + sum(initial_stacks)
        root = build_river_tree(pot=initial_pot, stacks=initial_stacks, max_raises=3)
        for node in walk_nodes(root):
            if node.is_terminal:
                observed = node.terminal_pot + node.stacks[0] + node.stacks[1]
                self.assertAlmostEqual(observed, total, places=6)


class SmokeTest(unittest.TestCase):
    def test_default_tree_is_reasonable_size(self):
        root = build_river_tree(pot=100, stacks=(200, 200))
        n = count_nodes(root)
        self.assertGreater(n, 10)
        self.assertLess(n, 5000)


if __name__ == "__main__":
    unittest.main()
