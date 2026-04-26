"""Best-response and exploitability for the flop solver.

Same structure as turn_best_response, with two chance node levels handled
identically (uniform average over children, no maximization at chance).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .flop_cfr import _FlopSolver, build_and_train_flop
from .showdown import CONFLICT_SENTINEL, WIN_HERO, WIN_TIE, WIN_VILLAIN
from .tree import Node


def best_response_value_flop(solver: _FlopSolver, br_player: int) -> float:
    table = solver.table
    if br_player == 0:
        br_range = list(table.hero_weights)
        opp_range = list(table.villain_weights)
    else:
        br_range = list(table.villain_weights)
        opp_range = list(table.hero_weights)
    values = _traverse(solver, solver.root, opp_range, br_player)
    denom = solver.pair_weight_total_per_runout
    if denom <= 0:
        return 0.0
    return sum(br_range[i] * values[i] for i in range(len(br_range))) / denom


def exploitability_flop(solver: _FlopSolver) -> float:
    return best_response_value_flop(solver, 0) + best_response_value_flop(solver, 1) - solver.root.pot


def compute_exploitability_flop(
    root: Node, hero_range: Sequence[float], villain_range: Sequence[float],
    board_3: Sequence[str], iterations: int = 60,
    hero_buckets=None, villain_buckets=None,
) -> Tuple[float, float, float, float]:
    solver = build_and_train_flop(
        root, hero_range, villain_range, board_3, iterations, hero_buckets, villain_buckets,
    )
    hv = solver.last_root_value()
    br_h = best_response_value_flop(solver, 0)
    br_v = best_response_value_flop(solver, 1)
    return hv, br_h, br_v, br_h + br_v - solver.root.pot


def _traverse(solver, node, reach_opp, br_player):
    if node.is_terminal:
        return _terminal_utility(solver, node, reach_opp, br_player)
    if node.is_chance:
        nc_br = solver.n_h if br_player == 0 else solver.n_v
        total = [0.0] * nc_br
        for child in node.children:
            child_util = _traverse(solver, child, reach_opp, br_player)
            for i in range(nc_br):
                total[i] += child_util[i]
        denom = len(node.children)
        return [x / denom for x in total]
    player = node.player_to_act
    node_id = getattr(node, "_node_id")
    na = len(node.actions)
    nc_br = solver.n_h if br_player == 0 else solver.n_v
    if player != br_player:
        opp_strat = solver.avg_strategy_at_node(node_id)
        opp_bucket_of = solver.villain_bucket_of if br_player == 0 else solver.hero_bucket_of
        child_values = []
        for a in range(na):
            new_reach_opp = [
                reach_opp[j] * opp_strat[opp_bucket_of[j]][a]
                for j in range(len(reach_opp))
            ]
            child_values.append(_traverse(solver, node.children[a], new_reach_opp, br_player))
        out = [0.0] * nc_br
        for combo in range(nc_br):
            for a in range(na):
                out[combo] += child_values[a][combo]
        return out
    child_values = []
    for a in range(na):
        child_values.append(_traverse(solver, node.children[a], reach_opp, br_player))
    out = [0.0] * nc_br
    for combo in range(nc_br):
        best = child_values[0][combo]
        for a in range(1, na):
            v = child_values[a][combo]
            if v > best:
                best = v
        out[combo] = best
    return out


def _terminal_utility(solver, node, reach_opp, br_player):
    from .cards import INDEX_TO_COMBO
    hero_c = solver.initial_stacks[0] - node.stacks[0]
    villain_c = solver.initial_stacks[1] - node.stacks[1]
    pot = node.terminal_pot
    table = solver.table

    if node.terminal_winner == 0:
        hero_profit = pot - hero_c; villain_profit = -villain_c
    elif node.terminal_winner == 1:
        hero_profit = -hero_c; villain_profit = pot - villain_c
    else:
        hero_profit = villain_profit = None

    if node.terminal_winner is not None:
        hero_pairs = [INDEX_TO_COMBO[c] for c in table.hero_combos]
        villain_pairs = [INDEX_TO_COMBO[c] for c in table.villain_combos]
        if br_player == 0:
            out = [0.0] * solver.n_h
            for i, (ha, hb) in enumerate(hero_pairs):
                hset = {ha, hb}
                total_v = 0.0
                for j, (va, vb) in enumerate(villain_pairs):
                    if va in hset or vb in hset: continue
                    total_v += reach_opp[j]
                out[i] = total_v * hero_profit
            return out
        out = [0.0] * solver.n_v
        for j, (va, vb) in enumerate(villain_pairs):
            vset = {va, vb}
            total_h = 0.0
            for i, (ha, hb) in enumerate(hero_pairs):
                if ha in vset or hb in vset: continue
                total_h += reach_opp[i]
            out[j] = total_h * villain_profit
        return out

    key = node.showdown_board_key
    outcome = table.outcomes_by_runout[key]
    if br_player == 0:
        out = [0.0] * solver.n_h
        for i in range(solver.n_h):
            acc = 0.0
            row = outcome[i]
            for j in range(solver.n_v):
                o = row[j]
                if o == CONFLICT_SENTINEL: continue
                if o == WIN_HERO: share = pot
                elif o == WIN_TIE: share = pot * 0.5
                else: share = 0.0
                acc += reach_opp[j] * (share - hero_c)
            out[i] = acc
        return out
    out = [0.0] * solver.n_v
    for j in range(solver.n_v):
        acc = 0.0
        for i in range(solver.n_h):
            o = outcome[i][j]
            if o == CONFLICT_SENTINEL: continue
            if o == WIN_VILLAIN: share = pot
            elif o == WIN_TIE: share = pot * 0.5
            else: share = 0.0
            acc += reach_opp[i] * (share - villain_c)
        out[j] = acc
    return out
