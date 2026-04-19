from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .cfr_core import InfoSet
from .leduc_rules import LeducState, apply_action, deal_public_card, initial_state, legal_actions, terminal_utility

RANKS = ("J", "Q", "K")
DECK = ("J1", "J2", "Q1", "Q2", "K1", "K2")


@dataclass
class BestResponseInfoSetStats:
    actions: Sequence[str]
    value_sum: List[float]
    weight_sum: float = 0.0

    @classmethod
    def create(cls, actions: Sequence[str]) -> "BestResponseInfoSetStats":
        return cls(actions=tuple(actions), value_sum=[0.0 for _ in actions], weight_sum=0.0)

    def accumulate(self, opponent_reach: float, action_values: Sequence[float]) -> None:
        self.weight_sum += opponent_reach
        for index, value in enumerate(action_values):
            self.value_sum[index] += opponent_reach * value

    def best_action_index(self) -> int:
        if self.weight_sum == 0.0:
            return 0
        best_index = 0
        best_value = float("-inf")
        for index, total in enumerate(self.value_sum):
            average_value = total / self.weight_sum
            if average_value > best_value:
                best_value = average_value
                best_index = index
        return best_index


class LeducCFRTrainer:
    def __init__(self) -> None:
        self.info_sets: Dict[str, InfoSet] = {}

    def train(self, iterations: int) -> Dict[str, object]:
        utility_sum = 0.0
        private_deals = list(itertools.permutations(DECK, 2))

        for _ in range(iterations):
            for private_cards in private_deals:
                utility_sum += self._cfr(initial_state(private_cards), 1.0, 1.0)

        training_game_value = utility_sum / (iterations * len(private_deals))
        infoset_strategy = {
            key: info_set.average_strategy()
            for key, info_set in self.info_sets.items()
        }
        root_strategy = {}
        for rank in RANKS:
            key = self._infoset_key(rank, None, ("", ""))
            if key in infoset_strategy:
                root_strategy[rank] = infoset_strategy[key]

        average_strategy_value = self.evaluate_average_strategy(infoset_strategy)
        best_response_player_0 = self.best_response_value(infoset_strategy, 0)
        best_response_player_1 = self.best_response_value(infoset_strategy, 1)
        exploitability = (
            (best_response_player_0 - average_strategy_value)
            + (best_response_player_1 - (-average_strategy_value))
        ) / 2.0

        return {
            "player_0_value": average_strategy_value,
            "training_game_value": training_game_value,
            "root_strategy": root_strategy,
            "infoset_count": len(self.info_sets),
            "infoset_strategy": infoset_strategy,
            "best_response_player_0": best_response_player_0,
            "best_response_player_1": best_response_player_1,
            "exploitability": exploitability,
        }

    def evaluate_average_strategy(self, infoset_strategy: Dict[str, Dict[str, float]]) -> float:
        private_deals = list(itertools.permutations(DECK, 2))
        total = 0.0
        for private_cards in private_deals:
            total += self._evaluate_strategy_profile(initial_state(private_cards), infoset_strategy)
        return total / len(private_deals)

    def best_response_value(self, infoset_strategy: Dict[str, Dict[str, float]], br_player: int) -> float:
        private_deals = list(itertools.permutations(DECK, 2))
        policy: Dict[str, int] = {}

        for _ in range(10):
            stats: Dict[str, BestResponseInfoSetStats] = {}
            for private_cards in private_deals:
                self._evaluate_with_policy(
                    initial_state(private_cards),
                    infoset_strategy,
                    br_player,
                    1.0,
                    policy,
                    stats,
                )

            changed = False
            for key, info_stats in stats.items():
                best_index = info_stats.best_action_index()
                if policy.get(key) != best_index:
                    policy[key] = best_index
                    changed = True
            if not changed:
                break

        total = 0.0
        for private_cards in private_deals:
            total += self._evaluate_final_policy(initial_state(private_cards), infoset_strategy, br_player, policy)
        return total / len(private_deals)

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

    def _evaluate_strategy_profile(self, state: LeducState, infoset_strategy: Dict[str, Dict[str, float]]) -> float:
        terminal = terminal_utility(state)
        if terminal is not None:
            return float(terminal)

        if state.is_chance_pending():
            remaining_cards = [card for card in DECK if card not in state.private_cards]
            chance_weight = 1.0 / len(remaining_cards)
            return sum(
                chance_weight * self._evaluate_strategy_profile(deal_public_card(state, public_card), infoset_strategy)
                for public_card in remaining_cards
            )

        actions = legal_actions(state)
        strategy = self._strategy_for_state(state, actions, infoset_strategy)
        expected_value = 0.0
        for index, action in enumerate(actions):
            if strategy[index] == 0.0:
                continue
            expected_value += strategy[index] * self._evaluate_strategy_profile(apply_action(state, action), infoset_strategy)
        return expected_value

    def _evaluate_with_policy(
        self,
        state: LeducState,
        infoset_strategy: Dict[str, Dict[str, float]],
        br_player: int,
        opponent_reach: float,
        policy: Dict[str, int],
        stats: Dict[str, BestResponseInfoSetStats],
    ) -> float:
        terminal = terminal_utility(state)
        if terminal is not None:
            return float(terminal if br_player == 0 else -terminal)

        if state.is_chance_pending():
            remaining_cards = [card for card in DECK if card not in state.private_cards]
            chance_weight = 1.0 / len(remaining_cards)
            return sum(
                self._evaluate_with_policy(
                    deal_public_card(state, public_card),
                    infoset_strategy,
                    br_player,
                    opponent_reach * chance_weight,
                    policy,
                    stats,
                )
                * chance_weight
                for public_card in remaining_cards
            )

        actions = legal_actions(state)
        player = state.current_player
        if player == br_player:
            action_values = [
                self._evaluate_with_policy(apply_action(state, action), infoset_strategy, br_player, opponent_reach, policy, stats)
                for action in actions
            ]
            key = self._state_infoset_key(state)
            info_stats = stats.setdefault(key, BestResponseInfoSetStats.create(actions))
            info_stats.accumulate(opponent_reach, action_values)
            policy_index = policy.setdefault(key, 0)
            return action_values[policy_index]

        strategy = self._strategy_for_state(state, actions, infoset_strategy)
        expected_value = 0.0
        for index, action in enumerate(actions):
            if strategy[index] == 0.0:
                continue
            child_value = self._evaluate_with_policy(
                apply_action(state, action),
                infoset_strategy,
                br_player,
                opponent_reach * strategy[index],
                policy,
                stats,
            )
            expected_value += strategy[index] * child_value
        return expected_value

    def _evaluate_final_policy(
        self,
        state: LeducState,
        infoset_strategy: Dict[str, Dict[str, float]],
        br_player: int,
        policy: Dict[str, int],
    ) -> float:
        terminal = terminal_utility(state)
        if terminal is not None:
            return float(terminal if br_player == 0 else -terminal)

        if state.is_chance_pending():
            remaining_cards = [card for card in DECK if card not in state.private_cards]
            chance_weight = 1.0 / len(remaining_cards)
            return sum(
                chance_weight * self._evaluate_final_policy(
                    deal_public_card(state, public_card),
                    infoset_strategy,
                    br_player,
                    policy,
                )
                for public_card in remaining_cards
            )

        actions = legal_actions(state)
        player = state.current_player
        if player == br_player:
            key = self._state_infoset_key(state)
            action_index = policy[key]
            return self._evaluate_final_policy(apply_action(state, actions[action_index]), infoset_strategy, br_player, policy)

        strategy = self._strategy_for_state(state, actions, infoset_strategy)
        expected_value = 0.0
        for index, action in enumerate(actions):
            if strategy[index] == 0.0:
                continue
            expected_value += strategy[index] * self._evaluate_final_policy(
                apply_action(state, action),
                infoset_strategy,
                br_player,
                policy,
            )
        return expected_value

    def _strategy_for_state(
        self,
        state: LeducState,
        actions: Sequence[str],
        infoset_strategy: Dict[str, Dict[str, float]],
    ) -> List[float]:
        infoset_key = self._state_infoset_key(state)
        strategy_map = infoset_strategy.get(infoset_key)
        if strategy_map is None:
            return [1.0 / len(actions) for _ in actions]
        return [strategy_map.get(action, 0.0) for action in actions]

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
