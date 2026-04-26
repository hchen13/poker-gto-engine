from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, Sequence, Tuple

from .cards import RANK_TO_VALUE

HandRank = Tuple[int, Tuple[int, ...]]

CATEGORY_HIGH_CARD = 0
CATEGORY_PAIR = 1
CATEGORY_TWO_PAIR = 2
CATEGORY_TRIPS = 3
CATEGORY_STRAIGHT = 4
CATEGORY_FLUSH = 5
CATEGORY_FULL_HOUSE = 6
CATEGORY_QUADS = 7
CATEGORY_STRAIGHT_FLUSH = 8

CATEGORY_NAMES = {
    CATEGORY_HIGH_CARD: "high_card",
    CATEGORY_PAIR: "pair",
    CATEGORY_TWO_PAIR: "two_pair",
    CATEGORY_TRIPS: "trips",
    CATEGORY_STRAIGHT: "straight",
    CATEGORY_FLUSH: "flush",
    CATEGORY_FULL_HOUSE: "full_house",
    CATEGORY_QUADS: "quads",
    CATEGORY_STRAIGHT_FLUSH: "straight_flush",
}


def evaluate_seven(cards: Sequence[str]) -> HandRank:
    if len(cards) != 7:
        raise ValueError(f"evaluate_seven expects 7 cards, got {len(cards)}")
    return max(_rank_five(combo) for combo in combinations(cards, 5))


def evaluate_any(cards: Sequence[str]) -> HandRank:
    if len(cards) == 5:
        return _rank_five(cards)
    if len(cards) < 5 or len(cards) > 7:
        raise ValueError(f"evaluate_any expects 5–7 cards, got {len(cards)}")
    return max(_rank_five(combo) for combo in combinations(cards, 5))


def _rank_five(cards: Iterable[str]) -> HandRank:
    ranks = sorted((RANK_TO_VALUE[c[0]] for c in cards), reverse=True)
    suits = [c[1] for c in cards]
    counts = Counter(ranks)
    groups = sorted(((cnt, rank) for rank, cnt in counts.items()), reverse=True)
    is_flush = len(set(suits)) == 1
    straight_high = _straight_high(ranks)

    if is_flush and straight_high is not None:
        return CATEGORY_STRAIGHT_FLUSH, (straight_high,)
    if groups[0][0] == 4:
        quad = groups[0][1]
        kicker = max(r for r in ranks if r != quad)
        return CATEGORY_QUADS, (quad, kicker)
    if groups[0][0] == 3 and groups[1][0] == 2:
        return CATEGORY_FULL_HOUSE, (groups[0][1], groups[1][1])
    if is_flush:
        return CATEGORY_FLUSH, tuple(ranks)
    if straight_high is not None:
        return CATEGORY_STRAIGHT, (straight_high,)
    if groups[0][0] == 3:
        trips = groups[0][1]
        kickers = tuple(r for r in ranks if r != trips)
        return CATEGORY_TRIPS, (trips,) + kickers
    if groups[0][0] == 2 and groups[1][0] == 2:
        pairs = sorted((r for cnt, r in groups if cnt == 2), reverse=True)
        kicker = max(r for r in ranks if r not in pairs)
        return CATEGORY_TWO_PAIR, (pairs[0], pairs[1], kicker)
    if groups[0][0] == 2:
        pair = groups[0][1]
        kickers = tuple(r for r in ranks if r != pair)
        return CATEGORY_PAIR, (pair,) + kickers
    return CATEGORY_HIGH_CARD, tuple(ranks)


def _straight_high(ranks: Sequence[int]) -> int | None:
    unique = sorted(set(ranks), reverse=True)
    if unique == [14, 5, 4, 3, 2]:
        return 5
    if len(unique) != 5:
        return None
    if unique[0] - unique[4] == 4:
        return unique[0]
    return None
