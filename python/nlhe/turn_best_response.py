"""Best-response and exploitability for the turn HU solver.

Structure mirrors ``best_response.py`` (for river). The extra wrinkle is the
chance node: at a chance node the BR traversal does the same thing as CFR's
chance handling — sum child values across 48 rivers, divide by 48. No
maximization happens at chance nodes because they're not decision points.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .showdown import CONFLICT_SENTINEL, WIN_HERO, WIN_TIE, WIN_VILLAIN
from .tree import Node
from .turn_cfr import _TurnSolver, build_and_train_turn


def best_response_value_turn(solver: _TurnSolver, br_player: int) -> float:
    if br_player not in (0, 1):
        raise ValueError("br_player must be 0 or 1")

    table = solver.table
    if br_player == 0:
        br_range = list(table.hero_weights)
        opp_range = list(table.villain_weights)
    else:
        br_range = list(table.villain_weights)
        opp_range = list(table.hero_weights)

    values = _traverse(solver, solver.root, opp_range, br_player)
    denom = solver.pair_weight_total_per_river
    if denom <= 0:
        return 0.0
    return sum(br_range[i] * values[i] for i in range(len(br_range))) / denom


def exploitability_turn(solver: _TurnSolver) -> float:
    br0 = best_response_value_turn(solver, br_player=0)
    br1 = best_response_value_turn(solver, br_player=1)
    return br0 + br1 - solver.root.pot


def compute_exploitability_turn(
    root: Node,
    hero_range: Sequence[float],
    villain_range: Sequence[float],
    board_4: Sequence[str],
    iterations: int = 200,
    hero_buckets=None,
    villain_buckets=None,
) -> Tuple[float, float, float, float]:
    solver = build_and_train_turn(
        root, hero_range, villain_range, board_4,
        iterations, hero_buckets, villain_buckets,
    )
    hero_value = solver.last_root_value()
    br_h = best_response_value_turn(solver, 0)
    br_v = best_response_value_turn(solver, 1)
    return hero_value, br_h, br_v, br_h + br_v - solver.root.pot


def _traverse(
    solver: _TurnSolver,
    node: Node,
    reach_opp: List[float],
    br_player: int,
) -> List[float]:
    if node.is_terminal:
        return _terminal_utility(solver, node, reach_opp, br_player)
    if node.is_chance:
        nc_br = solver.n_h if br_player == 0 else solver.n_v
        total = [0.0] * nc_br
        for child in node.children:
            child_util = _traverse(solver, child, reach_opp, br_player)
            for i in range(nc_br):
                total[i] += child_util[i]
        denom = solver._chance_denom
        return [x / denom for x in total]

    player = node.player_to_act
    node_id = getattr(node, "_node_id")
    na = len(node.actions)
    nc_br = solver.n_h if br_player == 0 else solver.n_v

    if player != br_player:
        opp_strat = solver.avg_strategy_at_node(node_id)
        opp_bucket_of = (
            solver.villain_bucket_of if br_player == 0 else solver.hero_bucket_of
        )
        child_values: List[List[float]] = []
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


def _terminal_utility(
    solver: _TurnSolver, node: Node, reach_opp: List[float], br_player: int
) -> List[float]:
    # Reuse solver's terminal logic but from BR's perspective, with reach_opp
    # swapped into the right slot.
    from .cards import INDEX_TO_COMBO
    hero_c = solver.initial_stacks[0] - node.stacks[0]
    villain_c = solver.initial_stacks[1] - node.stacks[1]
    pot = node.terminal_pot
    table = solver.table

    if node.terminal_winner == 0:
        hero_profit = pot - hero_c
        villain_profit = -villain_c
    elif node.terminal_winner == 1:
        hero_profit = -hero_c
        villain_profit = pot - villain_c
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
                    if va in hset or vb in hset:
                        continue
                    total_v += reach_opp[j]
                out[i] = total_v * hero_profit
            return out
        out = [0.0] * solver.n_v
        for j, (va, vb) in enumerate(villain_pairs):
            vset = {va, vb}
            total_h = 0.0
            for i, (ha, hb) in enumerate(hero_pairs):
                if ha in vset or hb in vset:
                    continue
                total_h += reach_opp[i]
            out[j] = total_h * villain_profit
        return out

    river_idx = node.showdown_board_key
    if river_idx is None:
        raise RuntimeError("showdown terminal missing showdown_board_key (turn BR)")
    outcome = table.outcomes_by_river[river_idx]

    if br_player == 0:
        out = [0.0] * solver.n_h
        for i in range(solver.n_h):
            acc = 0.0
            row = outcome[i]
            for j in range(solver.n_v):
                o = row[j]
                if o == CONFLICT_SENTINEL:
                    continue
                if o == WIN_HERO:
                    share = pot
                elif o == WIN_TIE:
                    share = pot * 0.5
                else:
                    share = 0.0
                acc += reach_opp[j] * (share - hero_c)
            out[i] = acc
        return out

    out = [0.0] * solver.n_v
    for j in range(solver.n_v):
        acc = 0.0
        for i in range(solver.n_h):
            o = outcome[i][j]
            if o == CONFLICT_SENTINEL:
                continue
            if o == WIN_VILLAIN:
                share = pot
            elif o == WIN_TIE:
                share = pot * 0.5
            else:
                share = 0.0
            acc += reach_opp[i] * (share - villain_c)
        out[j] = acc
    return out
