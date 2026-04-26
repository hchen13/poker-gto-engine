"""CFR+ solver for NLHE flop heads-up subgames.

Two-level chance handling: the flop tree (built by ``build_flop_tree``)
contains an outer chance node (turn card, 49 children) at every showdown
endpoint of the flop action tree. Each of those children is a turn subtree
that itself contains chance nodes (river card, 48 children) at its showdown
endpoints. Both chance levels just average uniformly over their children;
the deepest showdown terminals carry a ``(turn_card, river_card)`` key into
the precomputed flop showdown table.

Combo indexing is flop-local (filtered by 3-card board). Combos that conflict
with a specific turn or river card are marked CONFLICT_SENTINEL in that
run-out's outcome matrix, so a single combo space spans all run-outs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .flop_showdown import FlopShowdown, compute_flop_showdown
from .showdown import CONFLICT_SENTINEL, WIN_HERO, WIN_TIE, WIN_VILLAIN
from .tree import Node, walk_nodes


@dataclass
class FlopSolveResult:
    iterations: int
    root_strategy: Dict[int, Dict[str, float]]
    hero_value: float
    last_iter_values: List[float]


def solve_flop(
    root: Node,
    hero_range: Sequence[float],
    villain_range: Sequence[float],
    board_3: Sequence[str],
    iterations: int = 60,
    hero_buckets: Optional[Sequence[int]] = None,
    villain_buckets: Optional[Sequence[int]] = None,
) -> FlopSolveResult:
    if sum(hero_range) <= 0 or sum(villain_range) <= 0:
        raise ValueError("both ranges must have positive total weight")
    table = compute_flop_showdown(board_3, hero_range, villain_range)
    hero_local = _resolve_bucket_map(hero_buckets, table.hero_combos)
    villain_local = _resolve_bucket_map(villain_buckets, table.villain_combos)
    solver = _FlopSolver(root, table, root.stacks, hero_local, villain_local)
    solver.train(iterations)
    return FlopSolveResult(
        iterations=iterations,
        root_strategy=solver.extract_root_strategy(),
        hero_value=solver.last_root_value(),
        last_iter_values=list(solver._iter_values),
    )


def build_and_train_flop(
    root, hero_range, villain_range, board_3, iterations,
    hero_buckets=None, villain_buckets=None,
):
    if sum(hero_range) <= 0 or sum(villain_range) <= 0:
        raise ValueError("both ranges must have positive total weight")
    table = compute_flop_showdown(board_3, hero_range, villain_range)
    hero_local = _resolve_bucket_map(hero_buckets, table.hero_combos)
    villain_local = _resolve_bucket_map(villain_buckets, table.villain_combos)
    solver = _FlopSolver(root, table, root.stacks, hero_local, villain_local)
    solver.train(iterations)
    return solver


def _resolve_bucket_map(buckets, combo_indices):
    if buckets is None:
        return list(range(len(combo_indices)))
    out = []
    for gi in combo_indices:
        if gi < 0 or gi >= len(buckets):
            raise ValueError(f"bucket map too short: need {gi}, have {len(buckets)}")
        out.append(int(buckets[gi]))
    return out


class _FlopSolver:
    def __init__(self, root, table: FlopShowdown, initial_stacks, hero_buckets, villain_buckets):
        self.root = root
        self.table = table
        self.initial_stacks = initial_stacks
        self.n_h = len(table.hero_combos)
        self.n_v = len(table.villain_combos)

        self.hero_bucket_of, self.n_buckets_h = _normalize_buckets(hero_buckets)
        self.villain_bucket_of, self.n_buckets_v = _normalize_buckets(villain_buckets)

        self.hero_combos_in_bucket = [[] for _ in range(self.n_buckets_h)]
        for i, b in enumerate(self.hero_bucket_of):
            self.hero_combos_in_bucket[b].append(i)
        self.villain_combos_in_bucket = [[] for _ in range(self.n_buckets_v)]
        for j, b in enumerate(self.villain_bucket_of):
            self.villain_combos_in_bucket[b].append(j)

        # All decision nodes across the entire flop+turn+river tree.
        self.decision_nodes = [
            n for n in walk_nodes(root) if not n.is_terminal and not n.is_chance
        ]
        for i, n in enumerate(self.decision_nodes):
            setattr(n, "_node_id", i)

        self.regrets = []
        self.strategy_sum = []
        for n in self.decision_nodes:
            na = len(n.actions)
            nb = self.n_buckets_h if n.player_to_act == 0 else self.n_buckets_v
            self.regrets.append([[0.0] * na for _ in range(nb)])
            self.strategy_sum.append([[0.0] * na for _ in range(nb)])

        # Avg per-runout pair weight (proper EV normalizer).
        n_runouts = len(table.outcomes_by_runout)
        total = 0.0
        for (tc, rc), matrix in table.outcomes_by_runout.items():
            for i in range(self.n_h):
                hw = table.hero_weights[i]
                row = matrix[i]
                for j in range(self.n_v):
                    if row[j] == CONFLICT_SENTINEL:
                        continue
                    total += hw * table.villain_weights[j]
        self.pair_weight_total_per_runout = total / n_runouts if n_runouts else 0.0

        self._iter_values: List[float] = []

    def train(self, iterations: int) -> None:
        hero_w = list(self.table.hero_weights)
        villain_w = list(self.table.villain_weights)
        for t in range(1, iterations + 1):
            v0 = self._cfr(self.root, hero_w, villain_w, updating=0, t=t)
            self._cfr(self.root, hero_w, villain_w, updating=1, t=t)
            if self.pair_weight_total_per_runout > 0:
                ev = sum(hero_w[i] * v0[i] for i in range(self.n_h)) / self.pair_weight_total_per_runout
                self._iter_values.append(ev)

    def _cfr(self, node, reach_h, reach_v, updating, t):
        if node.is_terminal:
            return self._terminal_utility(node, reach_h, reach_v, updating)
        if node.is_chance:
            return self._chance_value(node, reach_h, reach_v, updating, t)

        player = node.player_to_act
        node_id = getattr(node, "_node_id")
        na = len(node.actions)
        nc_upd = self.n_h if updating == 0 else self.n_v

        if player == 0:
            bucket_of = self.hero_bucket_of
            nb_own = self.n_buckets_h
        else:
            bucket_of = self.villain_bucket_of
            nb_own = self.n_buckets_v

        strategy_b = self._regret_matching(node_id, nb_own, na)

        action_util = []
        for a in range(na):
            if player == 0:
                new_reach_h = [reach_h[i] * strategy_b[bucket_of[i]][a] for i in range(self.n_h)]
                new_reach_v = reach_v
            else:
                new_reach_h = reach_h
                new_reach_v = [reach_v[j] * strategy_b[bucket_of[j]][a] for j in range(self.n_v)]
            action_util.append(self._cfr(node.children[a], new_reach_h, new_reach_v, updating, t))

        node_util = [0.0] * nc_upd
        if player == updating:
            for i in range(nc_upd):
                s_row = strategy_b[bucket_of[i]]
                for a in range(na):
                    node_util[i] += s_row[a] * action_util[a][i]
        else:
            for i in range(nc_upd):
                for a in range(na):
                    node_util[i] += action_util[a][i]

        if player == updating:
            own_reach = reach_h if updating == 0 else reach_v
            combos_in_bucket = self.hero_combos_in_bucket if updating == 0 else self.villain_combos_in_bucket
            for b in range(nb_own):
                combos = combos_in_bucket[b]
                if not combos:
                    continue
                bucket_reach = 0.0
                s_row = strategy_b[b]
                cur_ev = [0.0] * len(combos)
                for ci, i in enumerate(combos):
                    s = 0.0
                    for ap in range(na):
                        s += s_row[ap] * action_util[ap][i]
                    cur_ev[ci] = s
                    bucket_reach += own_reach[i]
                regrets_bucket = self.regrets[node_id][b]
                strat_sum_bucket = self.strategy_sum[node_id][b]
                for a in range(na):
                    agg = 0.0
                    for ci, i in enumerate(combos):
                        agg += own_reach[i] * (action_util[a][i] - cur_ev[ci])
                    new_r = regrets_bucket[a] + agg
                    regrets_bucket[a] = new_r if new_r > 0.0 else 0.0
                    strat_sum_bucket[a] += t * bucket_reach * s_row[a]

        return node_util

    def _chance_value(self, node, reach_h, reach_v, updating, t):
        nc_upd = self.n_h if updating == 0 else self.n_v
        total = [0.0] * nc_upd
        for child in node.children:
            child_util = self._cfr(child, reach_h, reach_v, updating, t)
            for i in range(nc_upd):
                total[i] += child_util[i]
        denom = len(node.children)
        return [x / denom for x in total]

    def _terminal_utility(self, node, reach_h, reach_v, updating):
        from .cards import INDEX_TO_COMBO
        hero_c = self.initial_stacks[0] - node.stacks[0]
        villain_c = self.initial_stacks[1] - node.stacks[1]
        pot = node.terminal_pot
        if node.terminal_winner == 0:
            hero_profit = pot - hero_c; villain_profit = -villain_c
        elif node.terminal_winner == 1:
            hero_profit = -hero_c; villain_profit = pot - villain_c
        else:
            hero_profit = villain_profit = None

        if node.terminal_winner is not None:
            hero_pairs = [INDEX_TO_COMBO[c] for c in self.table.hero_combos]
            villain_pairs = [INDEX_TO_COMBO[c] for c in self.table.villain_combos]
            if updating == 0:
                out = [0.0] * self.n_h
                for i, (ha, hb) in enumerate(hero_pairs):
                    hset = {ha, hb}
                    total_v = 0.0
                    for j, (va, vb) in enumerate(villain_pairs):
                        if va in hset or vb in hset:
                            continue
                        total_v += reach_v[j]
                    out[i] = total_v * hero_profit
                return out
            out = [0.0] * self.n_v
            for j, (va, vb) in enumerate(villain_pairs):
                vset = {va, vb}
                total_h = 0.0
                for i, (ha, hb) in enumerate(hero_pairs):
                    if ha in vset or hb in vset:
                        continue
                    total_h += reach_h[i]
                out[j] = total_h * villain_profit
            return out

        key = node.showdown_board_key
        if not isinstance(key, tuple) or len(key) != 2:
            raise RuntimeError(f"flop showdown terminal needs (turn,river) tuple key, got {key!r}")
        outcome = self.table.outcomes_by_runout[key]
        if updating == 0:
            out = [0.0] * self.n_h
            for i in range(self.n_h):
                acc = 0.0
                row = outcome[i]
                for j in range(self.n_v):
                    o = row[j]
                    if o == CONFLICT_SENTINEL: continue
                    if o == WIN_HERO: share = pot
                    elif o == WIN_TIE: share = pot * 0.5
                    else: share = 0.0
                    acc += reach_v[j] * (share - hero_c)
                out[i] = acc
            return out
        out = [0.0] * self.n_v
        for j in range(self.n_v):
            acc = 0.0
            for i in range(self.n_h):
                o = outcome[i][j]
                if o == CONFLICT_SENTINEL: continue
                if o == WIN_VILLAIN: share = pot
                elif o == WIN_TIE: share = pot * 0.5
                else: share = 0.0
                acc += reach_h[i] * (share - villain_c)
            out[j] = acc
        return out

    def _regret_matching(self, node_id, n_rows, n_actions):
        strategy = [[0.0] * n_actions for _ in range(n_rows)]
        regrets = self.regrets[node_id]
        uniform = 1.0 / n_actions
        for i in range(n_rows):
            s = 0.0
            for a in range(n_actions):
                if regrets[i][a] > 0.0:
                    s += regrets[i][a]
            if s > 0.0:
                for a in range(n_actions):
                    r = regrets[i][a]
                    strategy[i][a] = (r / s) if r > 0.0 else 0.0
            else:
                for a in range(n_actions):
                    strategy[i][a] = uniform
        return strategy

    def extract_root_strategy(self):
        root_id = getattr(self.root, "_node_id")
        player = self.root.player_to_act
        if player == 0:
            combos = self.table.hero_combos
            bucket_of = self.hero_bucket_of
        else:
            combos = self.table.villain_combos
            bucket_of = self.villain_bucket_of
        na = len(self.root.actions)
        strat = self.strategy_sum[root_id]
        bucket_probs = {}
        out = {}
        for i, combo_idx in enumerate(combos):
            b = bucket_of[i]
            if b not in bucket_probs:
                row = strat[b]
                total = sum(row)
                if total > 0:
                    bucket_probs[b] = [v / total for v in row]
                else:
                    bucket_probs[b] = [1.0 / na] * na
            probs = bucket_probs[b]
            out[combo_idx] = {_action_label(self.root.actions[a]): probs[a] for a in range(na)}
        return out

    def last_root_value(self):
        return self._iter_values[-1] if self._iter_values else 0.0

    def avg_strategy_at_node(self, node_id):
        node = self.decision_nodes[node_id]
        na = len(node.actions)
        nb = self.n_buckets_h if node.player_to_act == 0 else self.n_buckets_v
        out = []
        for b in range(nb):
            row = self.strategy_sum[node_id][b]
            total = sum(row)
            if total > 0:
                out.append([v / total for v in row])
            else:
                out.append([1.0 / na] * na)
        return out


def _action_label(action):
    if action.kind in ("check", "fold", "call"):
        return action.kind
    return f"{action.kind}_{action.amount:.2f}"


def _normalize_buckets(raw):
    remap = {}
    out = []
    for b in raw:
        if b not in remap:
            remap[b] = len(remap)
        out.append(remap[b])
    return out, len(remap)
