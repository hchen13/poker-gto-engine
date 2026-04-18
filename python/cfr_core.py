from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass
class InfoSet:
    actions: Sequence[str]
    regret_sum: List[float] = field(default_factory=list)
    strategy_sum: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.regret_sum:
            self.regret_sum = [0.0 for _ in self.actions]
        if not self.strategy_sum:
            self.strategy_sum = [0.0 for _ in self.actions]

    def current_strategy(self) -> List[float]:
        positive_regrets = [max(regret, 0.0) for regret in self.regret_sum]
        normalizer = sum(positive_regrets)
        if normalizer > 0:
            return [regret / normalizer for regret in positive_regrets]
        return [1.0 / len(self.actions) for _ in self.actions]

    def average_strategy(self) -> Dict[str, float]:
        normalizer = sum(self.strategy_sum)
        if normalizer > 0:
            return {
                action: self.strategy_sum[index] / normalizer
                for index, action in enumerate(self.actions)
            }
        uniform = 1.0 / len(self.actions)
        return {action: uniform for action in self.actions}
