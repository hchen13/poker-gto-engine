import unittest

from python.leduc_rules import LeducState, apply_action, initial_state, legal_actions, showdown_winner, terminal_utility


class LeducRulesTest(unittest.TestCase):
    def test_initial_state_has_check_and_bet(self):
        state = initial_state(("J1", "Q1"))
        self.assertEqual(legal_actions(state), ["check", "bet"])

    def test_after_bet_opponent_can_fold_call_or_raise(self):
        state = apply_action(initial_state(("J1", "Q1")), "bet")
        self.assertEqual(legal_actions(state), ["fold", "call", "raise"])

    def test_check_check_advances_to_second_round(self):
        state = initial_state(("J1", "Q1"))
        state = apply_action(state, "check")
        state = apply_action(state, "check")
        self.assertEqual(state.round_index, 1)
        self.assertIsNone(state.public_card)
        self.assertEqual(state.round_contributions, (0, 0))

    def test_round_two_without_public_card_is_chance_pending(self):
        state = initial_state(("J1", "Q1"))
        state = apply_action(state, "check")
        state = apply_action(state, "check")
        self.assertTrue(state.is_chance_pending())

    def test_fold_ends_hand_with_correct_payoff(self):
        state = initial_state(("J1", "Q1"))
        state = apply_action(state, "bet")
        state = apply_action(state, "fold")
        self.assertEqual(terminal_utility(state), 1)

    def test_pair_beats_high_card_at_showdown(self):
        state = LeducState(
            private_cards=("J1", "Q1"),
            public_card="J2",
            round_index=1,
            current_player=0,
            contributions=(3, 3),
            round_contributions=(0, 0),
            raises_in_round=0,
            round_histories=("bc", "xx"),
            folded_player=None,
        )
        self.assertEqual(showdown_winner(state), 0)
        self.assertEqual(terminal_utility(state), 3)

    def test_high_card_wins_when_no_pair_exists(self):
        state = LeducState(
            private_cards=("Q1", "K1"),
            public_card="J2",
            round_index=1,
            current_player=0,
            contributions=(5, 5),
            round_contributions=(0, 0),
            raises_in_round=0,
            round_histories=("brc", "xx"),
            folded_player=None,
        )
        self.assertEqual(showdown_winner(state), 1)
        self.assertEqual(terminal_utility(state), -5)

    def test_same_rank_without_pair_splits_the_pot(self):
        state = LeducState(
            private_cards=("J1", "J2"),
            public_card="Q1",
            round_index=1,
            current_player=0,
            contributions=(5, 5),
            round_contributions=(0, 0),
            raises_in_round=0,
            round_histories=("bc", "xx"),
            folded_player=None,
        )
        self.assertIsNone(showdown_winner(state))
        self.assertEqual(terminal_utility(state), 0)


if __name__ == "__main__":
    unittest.main()
