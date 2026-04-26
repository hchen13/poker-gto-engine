"""Precompute per-river showdown outcome matrices for a turn subgame.

On the turn the board is only 4 cards. When action reaches a showdown at a
chance node, the river card is dealt uniformly from the remaining 48 cards,
and each resulting 5-card board has its own outcome matrix.

This module builds a set of per-river outcome matrices using TURN-LEVEL local
combo indexing — i.e. the same ``hero_combos`` / ``villain_combos`` lists (filtered
only by the 4-card board) are reused across every river, and combos that conflict
with a specific river card are marked with CONFLICT_SENTINEL in that river's
matrix. Keeping a single combo-index space across all rivers lets the solver
propagate reach vectors through chance nodes without any index translation.

Precompute cost is O(n_combos × 48) hand evaluations to build the rank table,
plus O(n_h × n_v × 48) cheap rank comparisons for the outcome matrices. The
heavy work is the rank table; once built, outcome matrices are fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .cards import INDEX_TO_CARD, INDEX_TO_COMBO, card_to_index
from .hand_eval import evaluate_seven
from .showdown import CONFLICT_SENTINEL, WIN_HERO, WIN_TIE, WIN_VILLAIN


@dataclass
class TurnShowdown:
    """Turn-level showdown data.

    - ``board_4``: the 4 known community cards
    - ``river_cards``: card indices of remaining cards (48 entries)
    - ``hero_combos`` / ``hero_weights``: global combo indices in hero range,
      board-4 filtered (same length across all rivers)
    - ``villain_combos`` / ``villain_weights``: same for villain
    - ``outcomes_by_river``: dict keyed by river card index → 2D outcome matrix
      (len hero_combos × len villain_combos), entries in {WIN_HERO, WIN_VILLAIN,
      WIN_TIE, CONFLICT_SENTINEL}. CONFLICT covers: hero/villain combos sharing
      a card with each other, with the river card, or (rare) with the board.
    """

    board_4: Tuple[str, ...]
    river_cards: List[int]
    hero_combos: List[int]
    hero_weights: List[float]
    villain_combos: List[int]
    villain_weights: List[float]
    outcomes_by_river: Dict[int, List[List[int]]]


def compute_turn_showdown(
    board_4: Sequence[str],
    hero_range: Sequence[float],
    villain_range: Sequence[float],
) -> TurnShowdown:
    board_tuple = tuple(board_4)
    board_indices = {card_to_index(c) for c in board_tuple}
    if len(board_indices) != 4:
        raise ValueError("board_4 must be 4 distinct cards")

    river_cards = [c for c in range(52) if c not in board_indices]

    hero_combos, hero_weights = _filter_range(hero_range, board_indices)
    villain_combos, villain_weights = _filter_range(villain_range, board_indices)

    if not hero_combos or not villain_combos:
        raise ValueError("one side has no combos compatible with the 4-card board")

    hero_card_pairs = [INDEX_TO_COMBO[c] for c in hero_combos]
    villain_card_pairs = [INDEX_TO_COMBO[c] for c in villain_combos]

    # Precompute hand ranks: rank[(combo_local_idx, river_idx)] → rank tuple
    # or None if combo conflicts with river.
    hero_ranks: Dict[Tuple[int, int], Tuple] = {}
    for i, (a, b) in enumerate(hero_card_pairs):
        for r in river_cards:
            if r == a or r == b:
                continue
            seven = list(board_tuple) + [INDEX_TO_CARD[r], INDEX_TO_CARD[a], INDEX_TO_CARD[b]]
            hero_ranks[(i, r)] = evaluate_seven(seven)

    villain_ranks: Dict[Tuple[int, int], Tuple] = {}
    for j, (a, b) in enumerate(villain_card_pairs):
        for r in river_cards:
            if r == a or r == b:
                continue
            seven = list(board_tuple) + [INDEX_TO_CARD[r], INDEX_TO_CARD[a], INDEX_TO_CARD[b]]
            villain_ranks[(j, r)] = evaluate_seven(seven)

    outcomes_by_river: Dict[int, List[List[int]]] = {}
    n_h = len(hero_combos)
    n_v = len(villain_combos)
    for r in river_cards:
        matrix = [[CONFLICT_SENTINEL] * n_v for _ in range(n_h)]
        for i, (ha, hb) in enumerate(hero_card_pairs):
            if ha == r or hb == r:
                continue  # entire row is CONFLICT
            hrank = hero_ranks[(i, r)]
            hset = {ha, hb}
            for j, (va, vb) in enumerate(villain_card_pairs):
                if va == r or vb == r:
                    continue
                if va in hset or vb in hset:
                    continue
                vrank = villain_ranks[(j, r)]
                if hrank > vrank:
                    matrix[i][j] = WIN_HERO
                elif hrank < vrank:
                    matrix[i][j] = WIN_VILLAIN
                else:
                    matrix[i][j] = WIN_TIE
        outcomes_by_river[r] = matrix

    return TurnShowdown(
        board_4=board_tuple,
        river_cards=river_cards,
        hero_combos=hero_combos,
        hero_weights=hero_weights,
        villain_combos=villain_combos,
        villain_weights=villain_weights,
        outcomes_by_river=outcomes_by_river,
    )


def _filter_range(
    range_vec: Sequence[float], board_indices: set
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
