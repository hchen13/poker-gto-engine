import unittest

from python.leduc_rules import deal_public_card, initial_state, apply_action, legal_actions, terminal_utility


class LeducTransitionTest(unittest.TestCase):
    def test_dealing_public_card_restores_player_zero_turn(self):
        state = initial_state(("J1", "Q1"))
        state = apply_action(state, "check")
        state = apply_action(state, "check")
        state = deal_public_card(state, "K1")

        self.assertEqual(state.public_card, "K1")
        self.assertEqual(state.current_player, 0)
        self.assertEqual(legal_actions(state), ["check", "bet"])

    def test_bet_call_on_second_round_reaches_terminal(self):
        state = initial_state(("K1", "Q1"))
        state = apply_action(state, "check")
        state = apply_action(state, "check")
        state = deal_public_card(state, "J1")
        state = apply_action(state, "bet")
        state = apply_action(state, "call")

        self.assertEqual(terminal_utility(state), 5)


if __name__ == "__main__":
    unittest.main()
