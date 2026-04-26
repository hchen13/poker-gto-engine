"""End-to-end analyzer for NLHE river heads-up subgames.

Glue layer that turns a user-facing spot description (board + range strings +
stacks + pot) into a solved GTO strategy.

Pipeline:
  1. Parse range strings into 1326-dim weight vectors.
  2. Build the HU river decision tree.
  3. Run CFR+ on the tree with card-aware infosets.
  4. Format the root strategy per-combo and (optionally) pick out the
     recommendation for a specific hero hand.

Convention: hero = player 0. ``first_to_act=0`` means hero is OOP (acts first
on the river); ``first_to_act=1`` means hero is IP.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .abstraction import bucket_by_ehs, compute_river_ehs
from .best_response import best_response_value
from .cards import NUM_COMBOS, combo_index, validate_cards
from .cfr import build_and_train, solve_river
from .range_parser import parse_range
from .showdown import compute_showdown_table
from .tree import build_river_tree


def analyze_river_spot(
    board: Sequence[str],
    pot: float,
    stacks: Tuple[float, float],
    hero_range: str,
    villain_range: str,
    first_to_act: int = 0,
    hero_hand: Optional[Sequence[str]] = None,
    iterations: int = 500,
    max_raises: int = 2,
    n_buckets: Optional[int] = None,
    compute_exploitability: bool = False,
) -> Dict[str, Any]:
    validate_cards(board, expected_count=5, label="board")
    if first_to_act not in (0, 1):
        raise ValueError("first_to_act must be 0 or 1")
    if pot <= 0:
        raise ValueError("pot must be positive")
    if stacks[0] < 0 or stacks[1] < 0:
        raise ValueError("stacks must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    hero_vec = parse_range(hero_range)
    villain_vec = parse_range(villain_range)

    if sum(hero_vec) <= 0:
        raise ValueError(f"hero_range parsed to zero weight: {hero_range!r}")
    if sum(villain_vec) <= 0:
        raise ValueError(f"villain_range parsed to zero weight: {villain_range!r}")

    root = build_river_tree(
        pot=float(pot),
        stacks=(float(stacks[0]), float(stacks[1])),
        first_to_act=first_to_act,
        max_raises=max_raises,
    )

    hero_buckets_arg = None
    villain_buckets_arg = None
    bucketing_info: Optional[Dict[str, Any]] = None
    if n_buckets is not None and n_buckets > 0:
        ehs = compute_river_ehs(board)
        table = compute_showdown_table(
            board=board, hero_range=hero_vec, villain_range=villain_vec
        )
        hero_local = bucket_by_ehs(
            table.hero_combos, table.hero_weights, ehs, n_buckets=n_buckets
        )
        villain_local = bucket_by_ehs(
            table.villain_combos, table.villain_weights, ehs, n_buckets=n_buckets
        )
        hero_buckets_arg = [0] * NUM_COMBOS
        for li, gi in enumerate(table.hero_combos):
            hero_buckets_arg[gi] = hero_local[li]
        villain_buckets_arg = [0] * NUM_COMBOS
        for lj, gj in enumerate(table.villain_combos):
            villain_buckets_arg[gj] = villain_local[lj]
        bucketing_info = {
            "n_buckets_requested": n_buckets,
            "hero_buckets_used": len(set(hero_local)),
            "villain_buckets_used": len(set(villain_local)),
        }

    if compute_exploitability:
        solver = build_and_train(
            root, hero_vec, villain_vec, board, iterations,
            hero_buckets=hero_buckets_arg,
            villain_buckets=villain_buckets_arg,
        )
        from .cfr import SolveResult as _SolveResult
        result = _SolveResult(
            iterations=iterations,
            root_strategy=solver.extract_root_strategy(),
            hero_value=solver.last_root_value(),
            last_iter_values=list(solver._iter_values),
        )
        exp_br_h = best_response_value(solver, 0)
        exp_br_v = best_response_value(solver, 1)
        exploitability_info = {
            "br_hero": exp_br_h,
            "br_villain": exp_br_v,
            "initial_pot": solver.root.pot,
            "exploitability": exp_br_h + exp_br_v - solver.root.pot,
        }
    else:
        result = solve_river(
            root=root,
            hero_range=hero_vec,
            villain_range=villain_vec,
            board=board,
            iterations=iterations,
            hero_buckets=hero_buckets_arg,
            villain_buckets=villain_buckets_arg,
        )
        exploitability_info = None

    acting_player_label = "hero" if first_to_act == 0 else "villain"
    acting_range_str = hero_range if first_to_act == 0 else villain_range

    strategy_by_combo = {
        _combo_label(idx): dict(probs)
        for idx, probs in result.root_strategy.items()
    }

    out: Dict[str, Any] = {
        "game": "nlhe_river_solve",
        "board": list(board),
        "pot": float(pot),
        "stacks": [float(stacks[0]), float(stacks[1])],
        "first_to_act": first_to_act,
        "acting_player": acting_player_label,
        "acting_range": acting_range_str,
        "iterations": iterations,
        "max_raises": max_raises,
        "hero_ev": result.hero_value,
        "hero_range_combos": _nonzero_count(hero_vec),
        "villain_range_combos": _nonzero_count(villain_vec),
        "strategy": strategy_by_combo,
        "ev_trace_last20": result.last_iter_values[-20:],
    }
    if bucketing_info is not None:
        out["bucketing"] = bucketing_info
    if exploitability_info is not None:
        out["exploitability"] = exploitability_info

    if hero_hand is not None:
        validate_cards(hero_hand, expected_count=2, label="hero_hand")
        hand_idx = combo_index(hero_hand[0], hero_hand[1])
        hand_label = _combo_label(hand_idx)
        out["hero_hand"] = hand_label

        if first_to_act != 0:
            out["recommendation"] = {
                "note": (
                    "hero is IP; first_to_act=1 means villain acts first. "
                    "Strategy shown is villain's — hero's response depends on what villain picks."
                ),
                "acting_player": acting_player_label,
            }
        elif hand_idx not in result.root_strategy:
            out["recommendation"] = {
                "note": f"hand {hand_label} is not in hero_range",
                "hand": hand_label,
            }
        else:
            probs = result.root_strategy[hand_idx]
            sorted_actions = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            out["recommendation"] = {
                "hand": hand_label,
                "top_action": sorted_actions[0][0],
                "top_probability": sorted_actions[0][1],
                "actions": [
                    {"action": a, "probability": p} for a, p in sorted_actions
                ],
            }

    return out


def _combo_label(combo_idx: int) -> str:
    from .cards import INDEX_TO_CARD, INDEX_TO_COMBO

    a, b = INDEX_TO_COMBO[combo_idx]
    return f"{INDEX_TO_CARD[a]}{INDEX_TO_CARD[b]}"


def _nonzero_count(vec: Sequence[float]) -> int:
    return sum(1 for w in vec if w > 0)
