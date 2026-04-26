from __future__ import annotations

from typing import Sequence

RANKS = "23456789TJQKA"
SUITS = "shdc"

RANK_TO_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
VALUE_TO_RANK = {v: r for r, v in RANK_TO_VALUE.items()}
VALID_SUITS = set(SUITS)

CARD_TO_INDEX = {f"{r}{s}": i for i, (r, s) in enumerate((r, s) for r in RANKS for s in SUITS)}
INDEX_TO_CARD = {i: c for c, i in CARD_TO_INDEX.items()}

NUM_CARDS = 52
NUM_COMBOS = 1326

_COMBOS: list[tuple[int, int]] = []
for _a in range(NUM_CARDS):
    for _b in range(_a + 1, NUM_CARDS):
        _COMBOS.append((_a, _b))

COMBO_TO_INDEX: dict[tuple[int, int], int] = {combo: i for i, combo in enumerate(_COMBOS)}
INDEX_TO_COMBO: list[tuple[int, int]] = list(_COMBOS)
assert len(INDEX_TO_COMBO) == NUM_COMBOS


def combo_index(card_a: str, card_b: str) -> int:
    ia, ib = CARD_TO_INDEX[card_a], CARD_TO_INDEX[card_b]
    if ia == ib:
        raise ValueError(f"duplicate cards in combo: {card_a}, {card_b}")
    lo, hi = (ia, ib) if ia < ib else (ib, ia)
    return COMBO_TO_INDEX[(lo, hi)]


def combo_cards(idx: int) -> tuple[str, str]:
    a, b = INDEX_TO_COMBO[idx]
    return INDEX_TO_CARD[a], INDEX_TO_CARD[b]


def validate_cards(cards: Sequence[str], expected_count: int | None, label: str) -> None:
    if expected_count is not None and len(cards) != expected_count:
        raise ValueError(f"{label} must contain {expected_count} cards, got {len(cards)}")
    if len(set(cards)) != len(cards):
        raise ValueError(f"{label} contains duplicate cards: {cards}")
    for card in cards:
        if len(card) != 2 or card[0] not in RANK_TO_VALUE or card[1] not in VALID_SUITS:
            raise ValueError(f"invalid card: {card!r}")


def card_to_index(card: str) -> int:
    if card not in CARD_TO_INDEX:
        raise ValueError(f"invalid card: {card!r}")
    return CARD_TO_INDEX[card]


def index_to_card(index: int) -> str:
    if not 0 <= index < NUM_CARDS:
        raise ValueError(f"card index out of range: {index}")
    return INDEX_TO_CARD[index]


def rank_of(card: str) -> int:
    return RANK_TO_VALUE[card[0]]


def suit_of(card: str) -> str:
    return card[1]
