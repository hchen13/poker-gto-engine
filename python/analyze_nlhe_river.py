from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

RANK_TO_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}
VALID_SUITS = {"s", "h", "d", "c"}


def analyze_nlhe_river(
    board: Sequence[str],
    hero_hand: Sequence[str],
    pot: float,
    to_call: float,
    villain_range: Sequence[Sequence[str]],
) -> Dict[str, object]:
    validate_cards(board, expected_count=5, label="board")
    validate_cards(hero_hand, expected_count=2, label="hero_hand")
    if pot < 0 or to_call < 0:
        raise SystemExit("pot and to_call must be non-negative")
    if not villain_range:
        raise SystemExit("villain_range must contain at least one combo")

    used_cards = set(board) | set(hero_hand)
    wins = 0
    ties = 0
    losses = 0
    valid_villain_combos: List[List[str]] = []
    hero_rank = best_seven_card_hand(list(board) + list(hero_hand))

    for combo in villain_range:
        validate_cards(combo, expected_count=2, label="villain_range combo")
        if set(combo) & used_cards:
            raise SystemExit(f"villain combo overlaps with known cards: {combo}")
        if combo[0] == combo[1]:
            raise SystemExit(f"villain combo duplicates a card: {combo}")
        valid_villain_combos.append(list(combo))
        villain_rank = best_seven_card_hand(list(board) + list(combo))
        if hero_rank > villain_rank:
            wins += 1
        elif hero_rank == villain_rank:
            ties += 1
        else:
            losses += 1

    total = len(valid_villain_combos)
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


def validate_cards(cards: Sequence[str], expected_count: int, label: str) -> None:
    if len(cards) != expected_count:
        raise SystemExit(f"{label} must contain {expected_count} cards")
    if len(set(cards)) != len(cards):
        raise SystemExit(f"{label} contains duplicate cards")
    for card in cards:
        if len(card) != 2 or card[0] not in RANK_TO_VALUE or card[1] not in VALID_SUITS:
            raise SystemExit(f"invalid card: {card}")


def best_seven_card_hand(cards: Sequence[str]) -> Tuple[int, Tuple[int, ...]]:
    return max(rank_five_card_hand(combo) for combo in combinations(cards, 5))


def rank_five_card_hand(cards: Iterable[str]) -> Tuple[int, Tuple[int, ...]]:
    ranks = sorted((RANK_TO_VALUE[card[0]] for card in cards), reverse=True)
    suits = [card[1] for card in cards]
    rank_counts = Counter(ranks)
    count_groups = sorted(((count, rank) for rank, count in rank_counts.items()), reverse=True)
    is_flush = len(set(suits)) == 1
    straight_high = straight_high_card(ranks)

    if is_flush and straight_high is not None:
        return 8, (straight_high,)
    if count_groups[0][0] == 4:
        quad_rank = count_groups[0][1]
        kicker = max(rank for rank in ranks if rank != quad_rank)
        return 7, (quad_rank, kicker)
    if count_groups[0][0] == 3 and count_groups[1][0] == 2:
        return 6, (count_groups[0][1], count_groups[1][1])
    if is_flush:
        return 5, tuple(ranks)
    if straight_high is not None:
        return 4, (straight_high,)
    if count_groups[0][0] == 3:
        trips_rank = count_groups[0][1]
        kickers = tuple(rank for rank in ranks if rank != trips_rank)
        return 3, (trips_rank,) + kickers
    if count_groups[0][0] == 2 and count_groups[1][0] == 2:
        pair_ranks = sorted((rank for count, rank in count_groups if count == 2), reverse=True)
        kicker = max(rank for rank in ranks if rank not in pair_ranks)
        return 2, (pair_ranks[0], pair_ranks[1], kicker)
    if count_groups[0][0] == 2:
        pair_rank = count_groups[0][1]
        kickers = tuple(rank for rank in ranks if rank != pair_rank)
        return 1, (pair_rank,) + kickers
    return 0, tuple(ranks)


def straight_high_card(ranks: Sequence[int]) -> int | None:
    unique = sorted(set(ranks), reverse=True)
    if unique == [14, 5, 4, 3, 2]:
        return 5
    if len(unique) != 5:
        return None
    if unique[0] - unique[4] == 4:
        return unique[0]
    return None
