"""CFR+ solver for NLHE river heads-up subgames.

Card-aware CFR: each decision node has an independent infoset per possible
combo held by the player to act. Regrets and average strategy are tracked
per (node_id, local_combo_idx, action_idx).

Payoff (zero-sum, from the *updating* player's POV):

  fold         → if I folded, -my_contribution; if opponent folded, pot - my_contribution
  showdown     → pot × equity_of_my_hand_vs_opp_hand - my_contribution

  my_contribution = initial_stacks[me] - node.stacks[me]

The initial pot already sitting in the middle is baked into node.terminal_pot
and awarded at terminals.

Algorithm: one CFR traversal per updating player per iteration (two passes total).
CFR+ regret clamping (regrets ≥ 0) + linear-CFR strategy averaging (weight t).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .showdown import (
    CONFLICT_SENTINEL,
    ShowdownTable,
    WIN_HERO,
    WIN_TIE,
    WIN_VILLAIN,
    compute_showdown_table,
)
from .tree import Node, walk_nodes


@dataclass
class SolveResult:
    iterations: int
    root_strategy: Dict[int, Dict[str, float]]  # {global_combo_idx: {action_label: prob}}
    hero_value: float                            # EV to player 0 averaged over hero range
    last_iter_values: List[float]


def solve_river(
    root: Node,
    hero_range: Sequence[float],
    villain_range: Sequence[float],
    board: Sequence[str],
    iterations: int = 500,
    hero_buckets: Optional[Sequence[int]] = None,
    villain_buckets: Optional[Sequence[int]] = None,
) -> SolveResult:
    """Solve a river HU subgame.

    ``hero_buckets`` / ``villain_buckets`` (optional): abstraction map from
    each side's global combo index to a bucket id. When provided, combos
    sharing a bucket share regret/strategy tables. Card conflicts and terminal
    payoffs remain per-combo, so this is a lossy but safe abstraction.
    When omitted, every combo is its own bucket (exact / unabstracted).
    """
    solver = build_and_train(
        root, hero_range, villain_range, board,
        iterations, hero_buckets, villain_buckets,
    )
    return SolveResult(
        iterations=iterations,
        root_strategy=solver.extract_root_strategy(),
        hero_value=solver.last_root_value(),
        last_iter_values=list(solver._iter_values),
    )


def build_and_train(
    root: Node,
    hero_range: Sequence[float],
    villain_range: Sequence[float],
    board: Sequence[str],
    iterations: int,
    hero_buckets: Optional[Sequence[int]] = None,
    villain_buckets: Optional[Sequence[int]] = None,
) -> "_Solver":
    """Construct the solver, run training, and return the populated solver state.

    Exposed so post-hoc analyses (e.g. exploitability / best-response) can
    traverse the solved strategy without re-solving.
    """
    if sum(hero_range) <= 0 or sum(villain_range) <= 0:
        raise ValueError("both ranges must have positive total weight")

    table = compute_showdown_table(board=board, hero_range=hero_range, villain_range=villain_range)
    if not table.hero_combos or not table.villain_combos:
        raise ValueError("one side has no combos compatible with the board")

    hero_local_buckets = _resolve_bucket_map(hero_buckets, table.hero_combos)
    villain_local_buckets = _resolve_bucket_map(villain_buckets, table.villain_combos)

    solver = _Solver(
        root, table, root.stacks, hero_local_buckets, villain_local_buckets
    )
    solver.train(iterations)
    return solver


def _resolve_bucket_map(
    buckets: Optional[Sequence[int]], combo_indices: Sequence[int]
) -> List[int]:
    """Translate a global-combo→bucket mapping into a local-combo→bucket list.

    ``buckets`` may be indexed by global combo idx (length 1326) OR may be
    omitted entirely (each local combo becomes its own bucket).
    """
    if buckets is None:
        return list(range(len(combo_indices)))
    out: List[int] = []
    for gi in combo_indices:
        if gi < 0 or gi >= len(buckets):
            raise ValueError(
                f"bucket map too short: need index {gi}, have length {len(buckets)}"
            )
        out.append(int(buckets[gi]))
    return out


class _Solver:
    def __init__(
        self,
        root: Node,
        table: ShowdownTable,
        initial_stacks: Tuple[float, float],
        hero_buckets: List[int],
        villain_buckets: List[int],
    ) -> None:
        self.root = root
        self.table = table
        self.initial_stacks = initial_stacks

        self.n_h = len(table.hero_combos)
        self.n_v = len(table.villain_combos)

        # Normalize bucket ids to be contiguous 0..K-1 per side.
        self.hero_bucket_of, self.n_buckets_h = _normalize_buckets(hero_buckets)
        self.villain_bucket_of, self.n_buckets_v = _normalize_buckets(villain_buckets)

        # Non-conflict pair weight total: the proper normalizer for expected
        # utilities. Using sum(hw) * sum(vw) over-counts because card-conflict
        # pairs (e.g. hero and villain sharing a hole card) contribute zero but
        # still appear in the product.
        self.pair_weight_total = 0.0
        for i in range(self.n_h):
            hw_i = table.hero_weights[i]
            row = table.outcome[i]
            for j in range(self.n_v):
                if row[j] == CONFLICT_SENTINEL:
                    continue
                self.pair_weight_total += hw_i * table.villain_weights[j]

        # Precompute combo-index lists per bucket per side (for regret aggregation).
        self.hero_combos_in_bucket: List[List[int]] = [
            [] for _ in range(self.n_buckets_h)
        ]
        for i, b in enumerate(self.hero_bucket_of):
            self.hero_combos_in_bucket[b].append(i)
        self.villain_combos_in_bucket: List[List[int]] = [
            [] for _ in range(self.n_buckets_v)
        ]
        for j, b in enumerate(self.villain_bucket_of):
            self.villain_combos_in_bucket[b].append(j)

        self.decision_nodes: List[Node] = [n for n in walk_nodes(root) if not n.is_terminal]
        for i, n in enumerate(self.decision_nodes):
            setattr(n, "_node_id", i)

        # regrets / strategy_sum: per node_id, per bucket (of acting side), per action
        self.regrets: List[List[List[float]]] = []
        self.strategy_sum: List[List[List[float]]] = []
        for n in self.decision_nodes:
            na = len(n.actions)
            nb = self.n_buckets_h if n.player_to_act == 0 else self.n_buckets_v
            self.regrets.append([[0.0] * na for _ in range(nb)])
            self.strategy_sum.append([[0.0] * na for _ in range(nb)])

        self._iter_values: List[float] = []

    # ------------------------------------------------------------------ train

    def train(self, iterations: int) -> None:
        hero_w = list(self.table.hero_weights)
        villain_w = list(self.table.villain_weights)

        for t in range(1, iterations + 1):
            v0 = self._cfr(self.root, hero_w, villain_w, updating=0, t=t)
            self._cfr(self.root, hero_w, villain_w, updating=1, t=t)
            # v0[i] = hero EV when hero holds combo i, integrated over villain reach.
            # Overall EV to player 0 = sum_i P(hero holds i) × v0[i] / sum_i P(hero=i)
            # v0[i] = counterfactual value for hero combo i (has villain-range weight baked in).
            # Overall EV to hero = (sum_i hero_w[i] × v0[i]) / (sum(hero_w) × sum(villain_w)).
            if self.pair_weight_total > 0:
                ev = sum(hero_w[i] * v0[i] for i in range(self.n_h)) / self.pair_weight_total
                self._iter_values.append(ev)

    # ------------------------------------------------------------------ cfr

    def _cfr(
        self,
        node: Node,
        reach_h: List[float],
        reach_v: List[float],
        updating: int,
        t: int,
    ) -> List[float]:
        """Return vector of utilities to ``updating`` player, indexed by that
        player's local combo index, integrated over opponent's reach."""
        if node.is_terminal:
            return self._terminal_utility(node, reach_h, reach_v, updating)

        player = node.player_to_act
        node_id = getattr(node, "_node_id")
        na = len(node.actions)

        nc_upd = self.n_h if updating == 0 else self.n_v

        # Bucket-level strategy for the acting player; broadcast to combos below.
        if player == 0:
            bucket_of = self.hero_bucket_of
            nb_own = self.n_buckets_h
            nc_own = self.n_h
        else:
            bucket_of = self.villain_bucket_of
            nb_own = self.n_buckets_v
            nc_own = self.n_v
        strategy_b = self._regret_matching(node_id, nb_own, na)

        # Recurse on each action child (reach is propagated per-combo using the
        # combo's bucket-strategy).
        action_util: List[List[float]] = []  # indexed [action][updating-player combo]
        for a in range(na):
            if player == 0:
                new_reach_h = [
                    reach_h[i] * strategy_b[bucket_of[i]][a] for i in range(self.n_h)
                ]
                new_reach_v = reach_v
            else:
                new_reach_h = reach_h
                new_reach_v = [
                    reach_v[j] * strategy_b[bucket_of[j]][a] for j in range(self.n_v)
                ]
            child_util = self._cfr(node.children[a], new_reach_h, new_reach_v, updating, t)
            action_util.append(child_util)

        # node_util[i] for updating player.
        node_util: List[float] = [0.0] * nc_upd
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
            # Aggregate per-combo regrets into per-bucket regrets, reach-weighted.
            # bucket_regret[b][a] = sum_{i in b} own_reach[i] * (action_util[a][i] - cur_ev[i])
            own_reach = reach_h if updating == 0 else reach_v
            combos_in_bucket = (
                self.hero_combos_in_bucket if updating == 0 else self.villain_combos_in_bucket
            )
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

    # ----------------------------------------------------------- terminals

    def _terminal_utility(
        self, node: Node, reach_h: List[float], reach_v: List[float], updating: int
    ) -> List[float]:
        hero_c = self.initial_stacks[0] - node.stacks[0]
        villain_c = self.initial_stacks[1] - node.stacks[1]
        pot = node.terminal_pot

        if node.terminal_winner == 0:  # villain folded; hero wins pot
            hero_profit = pot - hero_c
            villain_profit = -villain_c
        elif node.terminal_winner == 1:  # hero folded; villain wins pot
            hero_profit = -hero_c
            villain_profit = pot - villain_c
        else:
            hero_profit = villain_profit = None  # showdown — handled below

        if updating == 0:
            out = [0.0] * self.n_h
            if node.terminal_winner is not None:
                for i in range(self.n_h):
                    total_v = 0.0
                    for j in range(self.n_v):
                        if self.table.outcome[i][j] == CONFLICT_SENTINEL:
                            continue
                        total_v += reach_v[j]
                    out[i] = total_v * hero_profit
                return out
            for i in range(self.n_h):
                acc = 0.0
                for j in range(self.n_v):
                    o = self.table.outcome[i][j]
                    if o == CONFLICT_SENTINEL:
                        continue
                    if o == WIN_HERO:
                        share = pot
                    elif o == WIN_TIE:
                        share = pot * 0.5
                    else:
                        share = 0.0
                    acc += reach_v[j] * (share - hero_c)
                out[i] = acc
            return out

        # updating == 1 (villain)
        out = [0.0] * self.n_v
        if node.terminal_winner is not None:
            for j in range(self.n_v):
                total_h = 0.0
                for i in range(self.n_h):
                    if self.table.outcome[i][j] == CONFLICT_SENTINEL:
                        continue
                    total_h += reach_h[i]
                out[j] = total_h * villain_profit
            return out
        for j in range(self.n_v):
            acc = 0.0
            for i in range(self.n_h):
                o = self.table.outcome[i][j]
                if o == CONFLICT_SENTINEL:
                    continue
                if o == WIN_VILLAIN:
                    share = pot
                elif o == WIN_TIE:
                    share = pot * 0.5
                else:
                    share = 0.0
                acc += reach_h[i] * (share - villain_c)
            out[j] = acc
        return out

    # ---------------------------------------------------------- regret match

    def _regret_matching(self, node_id: int, n_rows: int, n_actions: int) -> List[List[float]]:
        """Return a [n_rows][n_actions] strategy table. ``n_rows`` is the
        number of buckets (or combos, when running unabstracted)."""
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

    # -------------------------------------------------------- output helpers

    def extract_root_strategy(self) -> Dict[int, Dict[str, float]]:
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
        # Normalize once per bucket, then broadcast to each combo in that bucket.
        bucket_probs: Dict[int, List[float]] = {}
        out: Dict[int, Dict[str, float]] = {}
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
            out[combo_idx] = {
                _action_label(self.root.actions[a]): probs[a] for a in range(na)
            }
        return out

    def last_root_value(self) -> float:
        return self._iter_values[-1] if self._iter_values else 0.0

    def avg_strategy_at_node(self, node_id: int) -> List[List[float]]:
        """Return normalized average strategy for ``node_id`` indexed
        [bucket][action]. Falls back to uniform for buckets with no mass yet."""
        node = self.decision_nodes[node_id]
        na = len(node.actions)
        nb = self.n_buckets_h if node.player_to_act == 0 else self.n_buckets_v
        out: List[List[float]] = []
        for b in range(nb):
            row = self.strategy_sum[node_id][b]
            total = sum(row)
            if total > 0:
                out.append([v / total for v in row])
            else:
                out.append([1.0 / na] * na)
        return out


def _action_label(action) -> str:
    if action.kind in ("check", "fold", "call"):
        return action.kind
    return f"{action.kind}_{action.amount:.2f}"


def _normalize_buckets(raw: Sequence[int]) -> Tuple[List[int], int]:
    """Compact bucket ids to contiguous 0..K-1, preserving the per-combo mapping."""
    remap: Dict[int, int] = {}
    out: List[int] = []
    for b in raw:
        if b not in remap:
            remap[b] = len(remap)
        out.append(remap[b])
    return out, len(remap)
