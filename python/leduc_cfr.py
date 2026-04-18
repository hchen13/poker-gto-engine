from __future__ import annotations

import itertools
from typing import Dict, List

from .cfr_core import InfoSet
from .leduc_rules import deal_public_card, initial_state, legal_actions, apply_action, terminal_utility, LeducState

RANKS = ("J", "Q", "K")
DECK = ("J1", "J2", "Q1", "Q2", "K1", "K2")


class LeducCFRTrainer:
    def __init__(self) -> None:
        self.info_sets: Dict[str, InfoSet] = {}

    def train(self, iterations: int) -> Dict[str, object]:
        utility_sum = 0.0
        private_deals = list(itertools.permutations(DECK, 2))

        for _ in range(iterations):
            for private_cards in private_deals:
                utility_sum += self._cfr(initial_state(private_cards), 1.0, 1.0)

        average_game_value = utility_sum / (iterations * len(private_deals))
        root_strategy = {}
        for rank in RANKS:
            key = self._infoset_key(rank, None, ("", ""))
            if key in self.info_sets:
                root_strategy[rank] = self.info_sets[key].average_strategy()

        return {
            "player_0_value": average_game_value,
            "root_strategy": root_strategy,
            "infoset_count": len(self.info_sets),
        }

    def _cfr(self, state: LeducState, reach_0: float, reach_1: float) -> float:
        terminal = terminal_utility(state)
        if terminal is not None:
            return float(terminal)

        if state.is_chance_pending():
            remaining_cards = [card for card in DECK if card not in state.private_cards]
            total = 0.0
            chance_weight = 1.0 / len(remaining_cards)
            for public_card in remaining_cards:
                next_state = deal_public_card(state, public_card)
                total += chance_weight * self._cfr(next_state, reach_0, reach_1)
            return total

        actions = legal_actions(state)
        player = state.current_player
        infoset_key = self._state_infoset_key(state)
        info_set = self.info_sets.setdefault(infoset_key, InfoSet(actions=tuple(actions)))
        strategy = info_set.current_strategy()

        if player == 0:
            for index, probability in enumerate(strategy):
                info_set.strategy_sum[index] += reach_0 * probability
        else:
            for index, probability in enumerate(strategy):
                info_set.strategy_sum[index] += reach_1 * probability

        action_utilities: List[float] = [0.0 for _ in actions]
        node_utility = 0.0

        for index, action in enumerate(actions):
            next_state = apply_action(state, action)
            if player == 0:
                action_utilities[index] = self._cfr(next_state, reach_0 * strategy[index], reach_1)
            else:
                action_utilities[index] = self._cfr(next_state, reach_0, reach_1 * strategy[index])
            node_utility += strategy[index] * action_utilities[index]

        for index in range(len(actions)):
            if player == 0:
                regret = action_utilities[index] - node_utility
                info_set.regret_sum[index] += reach_1 * regret
            else:
                regret = node_utility - action_utilities[index]
                info_set.regret_sum[index] += reach_0 * regret

        return node_utility

    def _state_infoset_key(self, state: LeducState) -> str:
        private_rank = state.private_cards[state.current_player][0]
        public_rank = state.public_card[0] if state.public_card else None
        return self._infoset_key(private_rank, public_rank, state.round_histories)

    def _infoset_key(self, private_rank: str, public_rank: str | None, round_histories: tuple[str, str]) -> str:
        public_part = public_rank if public_rank is not None else "-"
        return f"{private_rank}|{public_part}|{round_histories[0]}|{round_histories[1]}"


def train_leduc_cfr(iterations: int) -> Dict[str, object]:
    trainer = LeducCFRTrainer()
    return trainer.train(iterations)
