from __future__ import annotations

from typing import Dict, List, Sequence

from .cards import validate_cards
from .hand_eval import evaluate_seven


def river_equity_vs_range(
    board: Sequence[str],
    hero_hand: Sequence[str],
    pot: float,
    to_call: float,
    villain_range: Sequence[Sequence[str]],
) -> Dict[str, object]:
    validate_cards(board, expected_count=5, label="board")
    validate_cards(hero_hand, expected_count=2, label="hero_hand")
    if pot < 0 or to_call < 0:
        raise ValueError("pot and to_call must be non-negative")
    if not villain_range:
        raise ValueError("villain_range must contain at least one combo")

    used = set(board) | set(hero_hand)
    wins = ties = losses = 0
    hero_rank = evaluate_seven(list(board) + list(hero_hand))
    counted: List[List[str]] = []

    for combo in villain_range:
        validate_cards(combo, expected_count=2, label="villain_range combo")
        if set(combo) & used:
            raise ValueError(f"villain combo overlaps with known cards: {combo}")
        counted.append(list(combo))
        v_rank = evaluate_seven(list(board) + list(combo))
        if hero_rank > v_rank:
            wins += 1
        elif hero_rank == v_rank:
            ties += 1
        else:
            losses += 1

    total = len(counted)
    equity = (wins + 0.5 * ties) / total
    required_equity = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
    ev_call = equity * (pot + to_call) - to_call
    recommended_action = "call" if ev_call >= 0 else "fold"

    return {
        "game": "nlhe_river",
        "board": list(board),
        "hero_hand": list(hero_hand),
        "pot": pot,
        "to_call": to_call,
        "villain_combo_count": total,
        "equity": equity,
        "required_equity": required_equity,
        "ev_call": ev_call,
        "ev_fold": 0.0,
        "player_0_value": ev_call,
        "recommended_action": recommended_action,
        "win_rate": wins / total,
        "tie_rate": ties / total,
        "loss_rate": losses / total,
    }
