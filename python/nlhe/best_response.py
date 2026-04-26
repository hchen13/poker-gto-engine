"""Best-response and exploitability for solved river subgames.

Given a solver that has trained on a river HU subgame, this module computes
the best-response (BR) value for each player against the other's average
strategy. In a constant-sum game the exploitability — how far the solved
strategy pair is from Nash — is:

    exploitability = BR_hero + BR_villain - initial_pot

At Nash equilibrium each side's BR equals their equilibrium value, so
exploitability converges to 0. A non-zero value directly bounds how much
chips a perfect opponent could steal from the solved strategy.

The BR traversal mirrors CFR's own traversal structure:
- At BR player's decision nodes, pick argmax action per BR combo.
- At opponent nodes, integrate over opponent's avg strategy (per bucket).
- Reach propagation is only applied to the opponent's side — the BR player
  picks independently per combo, so their reach stays constant.
- Terminal utilities use the existing showdown table (per-combo, opp-reach-integrated).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .cfr import _Solver, build_and_train
from .showdown import CONFLICT_SENTINEL, WIN_HERO, WIN_TIE, WIN_VILLAIN
from .tree import Node


def best_response_value(solver: _Solver, br_player: int) -> float:
    """Return BR value to ``br_player`` averaged over their range, assuming
    the opponent plays the solver's current average strategy."""
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
    denom = solver.pair_weight_total
    if denom <= 0:
        return 0.0
    return sum(br_range[i] * values[i] for i in range(len(br_range))) / denom


def exploitability(solver: _Solver) -> float:
    """BR_hero + BR_villain - initial_pot. Approaches 0 at Nash."""
    br0 = best_response_value(solver, br_player=0)
    br1 = best_response_value(solver, br_player=1)
    return br0 + br1 - solver.root.pot


def compute_exploitability(
    root: Node,
    hero_range: Sequence[float],
    villain_range: Sequence[float],
    board: Sequence[str],
    iterations: int = 500,
    hero_buckets=None,
    villain_buckets=None,
) -> Tuple[float, float, float]:
    """Convenience: solve the spot and return (hero_value, br_hero, br_villain, exploitability)."""
    solver = build_and_train(
        root, hero_range, villain_range, board,
        iterations, hero_buckets, villain_buckets,
    )
    hero_value = solver.last_root_value()
    br_h = best_response_value(solver, br_player=0)
    br_v = best_response_value(solver, br_player=1)
    return hero_value, br_h, br_v, br_h + br_v - solver.root.pot


def _traverse(
    solver: _Solver,
    node: Node,
    reach_opp: List[float],
    br_player: int,
) -> List[float]:
    """Return BR-player-combo → counterfactual value (opp-reach baked in)."""
    if node.is_terminal:
        return _terminal_utility(solver, node, reach_opp, br_player)

    player = node.player_to_act
    node_id = getattr(node, "_node_id")
    na = len(node.actions)
    nc_br = solver.n_h if br_player == 0 else solver.n_v

    if player != br_player:
        # Opponent node: propagate opp reach by their avg strategy per bucket.
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

    # BR player's node: pick argmax per BR combo.
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
    solver: _Solver, node: Node, reach_opp: List[float], br_player: int
) -> List[float]:
    """Copy of CFR's terminal utility, but only returning the BR player's side."""
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

    if br_player == 0:
        out = [0.0] * solver.n_h
        if node.terminal_winner is not None:
            for i in range(solver.n_h):
                total_v = 0.0
                for j in range(solver.n_v):
                    if table.outcome[i][j] == CONFLICT_SENTINEL:
                        continue
                    total_v += reach_opp[j]
                out[i] = total_v * hero_profit
            return out
        for i in range(solver.n_h):
            acc = 0.0
            for j in range(solver.n_v):
                o = table.outcome[i][j]
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
    if node.terminal_winner is not None:
        for j in range(solver.n_v):
            total_h = 0.0
            for i in range(solver.n_h):
                if table.outcome[i][j] == CONFLICT_SENTINEL:
                    continue
                total_h += reach_opp[i]
            out[j] = total_h * villain_profit
        return out
    for j in range(solver.n_v):
        acc = 0.0
        for i in range(solver.n_h):
            o = table.outcome[i][j]
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
