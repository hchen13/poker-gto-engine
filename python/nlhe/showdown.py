"""Precompute pairwise showdown outcomes for a river spot.

For a fixed board and two ranges (hero / villain), this module enumerates every
compatible (hero_combo, villain_combo) pair and records who wins. The resulting
lookup lets the CFR solver compute terminal value in constant time per pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .cards import INDEX_TO_CARD, INDEX_TO_COMBO, card_to_index
from .hand_eval import evaluate_seven


WIN_HERO = 1
WIN_VILLAIN = -1
WIN_TIE = 0


@dataclass
class ShowdownTable:
    board: Tuple[str, ...]
    hero_combos: List[int]            # global combo indices, filtered (no board conflict)
    hero_weights: List[float]
    villain_combos: List[int]
    villain_weights: List[float]
    # outcome[i][j]: +1 hero wins, -1 villain wins, 0 tie. -2 sentinel: combos conflict.
    outcome: List[List[int]]


CONFLICT_SENTINEL = -2


def compute_showdown_table(
    board: Sequence[str],
    hero_range: Sequence[float],
    villain_range: Sequence[float],
) -> ShowdownTable:
    board_tuple = tuple(board)
    board_indices = {card_to_index(c) for c in board_tuple}

    hero_combos, hero_weights = _filter_range(hero_range, board_indices)
    villain_combos, villain_weights = _filter_range(villain_range, board_indices)

    outcome: List[List[int]] = [[CONFLICT_SENTINEL] * len(villain_combos) for _ in hero_combos]

    hero_card_pairs = [INDEX_TO_COMBO[c] for c in hero_combos]
    villain_card_pairs = [INDEX_TO_COMBO[c] for c in villain_combos]

    hero_ranks: List[tuple] = []
    for a, b in hero_card_pairs:
        seven = list(board_tuple) + [INDEX_TO_CARD[a], INDEX_TO_CARD[b]]
        hero_ranks.append(evaluate_seven(seven))

    villain_ranks: List[tuple] = []
    for a, b in villain_card_pairs:
        seven = list(board_tuple) + [INDEX_TO_CARD[a], INDEX_TO_CARD[b]]
        villain_ranks.append(evaluate_seven(seven))

    for i, (ha, hb) in enumerate(hero_card_pairs):
        hero_set = {ha, hb}
        hr = hero_ranks[i]
        row = outcome[i]
        for j, (va, vb) in enumerate(villain_card_pairs):
            if va in hero_set or vb in hero_set:
                continue  # CONFLICT_SENTINEL already set
            vr = villain_ranks[j]
            if hr > vr:
                row[j] = WIN_HERO
            elif hr < vr:
                row[j] = WIN_VILLAIN
            else:
                row[j] = WIN_TIE

    return ShowdownTable(
        board=board_tuple,
        hero_combos=hero_combos,
        hero_weights=hero_weights,
        villain_combos=villain_combos,
        villain_weights=villain_weights,
        outcome=outcome,
    )


def _filter_range(
    range_vec: Sequence[float], board_indices: set[int]
) -> Tuple[List[int], List[float]]:
    combos: List[int] = []
    weights: List[float] = []
    for idx, w in enumerate(range_vec):
        if w <= 0:
            continue
        a, b = INDEX_TO_COMBO[idx]
        if a in board_indices or b in board_indices:
            continue
        combos.append(idx)
        weights.append(w)
    return combos, weights
