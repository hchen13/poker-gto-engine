from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

INITIAL_ANTE = 1
BET_SIZES = (2, 4)
MAX_RAISES_PER_ROUND = 2
RANK_ORDER = {"J": 0, "Q": 1, "K": 2}


@dataclass(frozen=True)
class LeducState:
    private_cards: Tuple[str, str]
    public_card: Optional[str]
    round_index: int
    current_player: int
    contributions: Tuple[int, int]
    round_contributions: Tuple[int, int]
    raises_in_round: int
    round_histories: Tuple[str, str]
    folded_player: Optional[int] = None

    def is_chance_pending(self) -> bool:
        return self.round_index == 1 and self.public_card is None and self.folded_player is None


def initial_state(private_cards: Tuple[str, str]) -> LeducState:
    return LeducState(
        private_cards=private_cards,
        public_card=None,
        round_index=0,
        current_player=0,
        contributions=(INITIAL_ANTE, INITIAL_ANTE),
        round_contributions=(0, 0),
        raises_in_round=0,
        round_histories=("", ""),
        folded_player=None,
    )


def legal_actions(state: LeducState) -> list[str]:
    opponent = 1 - state.current_player
    outstanding = state.round_contributions[state.current_player] < state.round_contributions[opponent]
    if outstanding:
        actions = ["fold", "call"]
        if state.raises_in_round < MAX_RAISES_PER_ROUND:
            actions.append("raise")
        return actions
    return ["check", "bet"]


def apply_action(state: LeducState, action: str) -> LeducState:
    if state.is_chance_pending():
        raise ValueError("cannot apply player action while public card is pending")

    player = state.current_player
    opponent = 1 - player
    contributions = list(state.contributions)
    round_contributions = list(state.round_contributions)
    round_histories = list(state.round_histories)
    round_histories[state.round_index] += history_char(action)
    bet_size = BET_SIZES[state.round_index]

    if action == "fold":
        return replace(state, folded_player=player, round_histories=tuple(round_histories))

    if action == "check":
        if state.round_histories[state.round_index] == "x":
            return advance_round_or_terminal(state, tuple(contributions), tuple(round_contributions), tuple(round_histories))
        return replace(state, current_player=opponent, round_histories=tuple(round_histories))

    if action == "bet":
        contributions[player] += bet_size
        round_contributions[player] += bet_size
        return replace(
            state,
            current_player=opponent,
            contributions=tuple(contributions),
            round_contributions=tuple(round_contributions),
            raises_in_round=1,
            round_histories=tuple(round_histories),
        )

    if action == "call":
        call_amount = round_contributions[opponent] - round_contributions[player]
        contributions[player] += call_amount
        round_contributions[player] += call_amount
        return advance_round_or_terminal(state, tuple(contributions), tuple(round_contributions), tuple(round_histories))

    if action == "raise":
        raise_amount = (round_contributions[opponent] - round_contributions[player]) + bet_size
        contributions[player] += raise_amount
        round_contributions[player] += raise_amount
        return replace(
            state,
            current_player=opponent,
            contributions=tuple(contributions),
            round_contributions=tuple(round_contributions),
            raises_in_round=state.raises_in_round + 1,
            round_histories=tuple(round_histories),
        )

    raise ValueError(f"unknown action: {action}")


def deal_public_card(state: LeducState, public_card: str) -> LeducState:
    if not state.is_chance_pending():
        raise ValueError("public card can only be dealt at the round transition")
    return replace(state, public_card=public_card, current_player=0)


def advance_round_or_terminal(
    state: LeducState,
    contributions: Tuple[int, int],
    round_contributions: Tuple[int, int],
    round_histories: Tuple[str, str],
) -> LeducState:
    if state.round_index == 0:
        return replace(
            state,
            round_index=1,
            current_player=0,
            contributions=contributions,
            round_contributions=(0, 0),
            raises_in_round=0,
            round_histories=round_histories,
        )
    return replace(
        state,
        contributions=contributions,
        round_contributions=round_contributions,
        round_histories=round_histories,
    )


def showdown_winner(state: LeducState) -> int:
    player_0_pair = card_rank(state.private_cards[0]) == card_rank(state.public_card)
    player_1_pair = card_rank(state.private_cards[1]) == card_rank(state.public_card)
    if player_0_pair and not player_1_pair:
        return 0
    if player_1_pair and not player_0_pair:
        return 1
    return 0 if card_rank(state.private_cards[0]) > card_rank(state.private_cards[1]) else 1


def terminal_utility(state: LeducState) -> Optional[int]:
    if state.folded_player is not None:
        winner = 1 - state.folded_player
        return state.contributions[1] if winner == 0 else -state.contributions[0]

    if state.public_card is None:
        return None

    round_history = state.round_histories[1]
    if round_history.endswith("xx") or round_history.endswith("bc") or round_history.endswith("rc"):
        winner = showdown_winner(state)
        return state.contributions[1] if winner == 0 else -state.contributions[0]
    return None


def history_char(action: str) -> str:
    return {
        "check": "x",
        "bet": "b",
        "call": "c",
        "raise": "r",
        "fold": "f",
    }[action]


def card_rank(card: Optional[str]) -> int:
    if card is None:
        raise ValueError("card missing")
    return RANK_ORDER[card[0]]
