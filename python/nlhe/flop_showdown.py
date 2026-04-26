"""Per-(turn,river) showdown matrices for the flop solver.

Built on top of the same idea as ``turn_showdown``: filter combos by the
3-card flop board, then for each (turn_card, river_card) pair, compute the
outcome matrix in flop-local combo indexing. Combos containing the turn or
river card get marked CONFLICT_SENTINEL in that pair's matrix, so a single
combo space spans all run-outs.

Memory cost: 49 × 48 = 2352 (turn, river) board pairs × n_h × n_v × 1 int
each. For small ranges (≤10 combos/side) this is sub-MB. For real-size
ranges we need card abstraction (deferred — turn/river run-out clustering
will be the public-card abstraction layer in Inc 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .cards import INDEX_TO_CARD, INDEX_TO_COMBO, card_to_index
from .hand_eval import evaluate_seven
from .showdown import CONFLICT_SENTINEL, WIN_HERO, WIN_TIE, WIN_VILLAIN


@dataclass
class FlopShowdown:
    board_3: Tuple[str, ...]
    turn_cards: List[int]                 # 49 indices
    hero_combos: List[int]
    hero_weights: List[float]
    villain_combos: List[int]
    villain_weights: List[float]
    # outcomes_by_runout[(turn_idx, river_idx)] = 2D matrix (len hero × len villain)
    outcomes_by_runout: Dict[Tuple[int, int], List[List[int]]]


def compute_flop_showdown(
    board_3: Sequence[str],
    hero_range: Sequence[float],
    villain_range: Sequence[float],
) -> FlopShowdown:
    board_tuple = tuple(board_3)
    board_indices = {card_to_index(c) for c in board_tuple}
    if len(board_indices) != 3:
        raise ValueError("board_3 must be 3 distinct cards")

    turn_cards = [c for c in range(52) if c not in board_indices]

    hero_combos, hero_weights = _filter_range(hero_range, board_indices)
    villain_combos, villain_weights = _filter_range(villain_range, board_indices)
    if not hero_combos or not villain_combos:
        raise ValueError("one side has no combos compatible with the 3-card board")

    hero_pairs = [INDEX_TO_COMBO[c] for c in hero_combos]
    villain_pairs = [INDEX_TO_COMBO[c] for c in villain_combos]
    n_h = len(hero_combos)
    n_v = len(villain_combos)

    outcomes_by_runout: Dict[Tuple[int, int], List[List[int]]] = {}

    for ti in range(len(turn_cards)):
        tc = turn_cards[ti]
        # rivers are turn_cards minus tc
        for ri in range(len(turn_cards)):
            rc = turn_cards[ri]
            if rc == tc:
                continue
            matrix = [[CONFLICT_SENTINEL] * n_v for _ in range(n_h)]
            for i, (ha, hb) in enumerate(hero_pairs):
                if ha == tc or hb == tc or ha == rc or hb == rc:
                    continue
                seven_h = list(board_tuple) + [
                    INDEX_TO_CARD[tc], INDEX_TO_CARD[rc],
                    INDEX_TO_CARD[ha], INDEX_TO_CARD[hb],
                ]
                hrank = evaluate_seven(seven_h)
                hset = {ha, hb}
                for j, (va, vb) in enumerate(villain_pairs):
                    if va == tc or vb == tc or va == rc or vb == rc:
                        continue
                    if va in hset or vb in hset:
                        continue
                    seven_v = list(board_tuple) + [
                        INDEX_TO_CARD[tc], INDEX_TO_CARD[rc],
                        INDEX_TO_CARD[va], INDEX_TO_CARD[vb],
                    ]
                    vrank = evaluate_seven(seven_v)
                    if hrank > vrank:
                        matrix[i][j] = WIN_HERO
                    elif hrank < vrank:
                        matrix[i][j] = WIN_VILLAIN
                    else:
                        matrix[i][j] = WIN_TIE
            outcomes_by_runout[(tc, rc)] = matrix

    return FlopShowdown(
        board_3=board_tuple,
        turn_cards=turn_cards,
        hero_combos=hero_combos,
        hero_weights=hero_weights,
        villain_combos=villain_combos,
        villain_weights=villain_weights,
        outcomes_by_runout=outcomes_by_runout,
    )


def _filter_range(range_vec, board_indices):
    combos = []
    weights = []
    for idx, w in enumerate(range_vec):
        if w <= 0:
            continue
        a, b = INDEX_TO_COMBO[idx]
        if a in board_indices or b in board_indices:
            continue
        combos.append(idx)
        weights.append(w)
    return combos, weights
