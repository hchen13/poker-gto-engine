from __future__ import annotations

from typing import Dict, List
import itertools

from .cfr_core import InfoSet

ACTIONS = ("check", "bet")
PASS_ACTION = "p"
BET_ACTION = "b"
CARD_RANK = {"J": 0, "Q": 1, "K": 2}


class KuhnCFRTrainer:
    def __init__(self) -> None:
        self.info_sets: Dict[str, InfoSet] = {}

    def train(self, iterations: int) -> Dict[str, object]:
        cards = ("J", "Q", "K")
        utility_sum = 0.0

        for _ in range(iterations):
            for permutation in itertools.permutations(cards):
                utility_sum += self._cfr(list(permutation), "", 1.0, 1.0)

        average_game_value = utility_sum / (iterations * 6)
        root_strategy = {
            card: self.info_sets[card].average_strategy()
            for card in cards
            if card in self.info_sets
        }
        infoset_strategy = {
            key: info_set.average_strategy()
            for key, info_set in self.info_sets.items()
        }
        return {
            "player_0_value": average_game_value,
            "root_strategy": root_strategy,
            "infoset_strategy": infoset_strategy,
        }

    def _cfr(self, cards: List[str], history: str, reach_0: float, reach_1: float) -> float:
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        terminal_utility = self._terminal_utility(cards, history)
        if terminal_utility is not None:
            return terminal_utility

        info_key = cards[player] + history
        info_set = self.info_sets.setdefault(info_key, InfoSet(actions=ACTIONS))
        strategy = info_set.current_strategy()

        if player == 0:
            for index, probability in enumerate(strategy):
                info_set.strategy_sum[index] += reach_0 * probability
        else:
            for index, probability in enumerate(strategy):
                info_set.strategy_sum[index] += reach_1 * probability

        action_utilities = [0.0, 0.0]
        node_utility = 0.0

        for index, action in enumerate((PASS_ACTION, BET_ACTION)):
            next_history = history + action
            if player == 0:
                action_utilities[index] = -self._cfr(cards, next_history, reach_0 * strategy[index], reach_1)
            else:
                action_utilities[index] = -self._cfr(cards, next_history, reach_0, reach_1 * strategy[index])
            node_utility += strategy[index] * action_utilities[index]

        for index in range(2):
            regret = action_utilities[index] - node_utility
            if player == 0:
                info_set.regret_sum[index] += reach_1 * regret
            else:
                info_set.regret_sum[index] += reach_0 * regret

        return node_utility

    def _terminal_utility(self, cards: List[str], history: str) -> float | None:
        if len(history) < 2:
            return None

        plays = len(history)
        player = plays % 2
        opponent = 1 - player
        terminal_pass = history[-1] == PASS_ACTION
        double_bet = history[-2:] == BET_ACTION * 2

        if terminal_pass:
            if history == PASS_ACTION * 2:
                return 1.0 if CARD_RANK[cards[player]] > CARD_RANK[cards[opponent]] else -1.0
            return 1.0

        if double_bet:
            return 2.0 if CARD_RANK[cards[player]] > CARD_RANK[cards[opponent]] else -2.0

        return None


def train_kuhn_cfr(iterations: int) -> Dict[str, object]:
    trainer = KuhnCFRTrainer()
    return trainer.train(iterations)
