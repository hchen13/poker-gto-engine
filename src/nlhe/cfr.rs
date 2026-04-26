//! CFR+ solver (river HU subgame) — Rust port of `python/nlhe/cfr.py`.
//!
//! Same payoff convention: zero-sum-shifted-by-pot, hero = player 0.
//! Same averaging: linear-CFR strategy averaging with `t` weight.
//! Same regret update: per-bucket aggregation, reach-weighted, CFR+ clamp ≥ 0.
//! Bucketing optional; default = identity (each combo its own bucket).

use std::collections::HashMap;

use super::showdown::{ShowdownTable, CONFLICT, WIN_HERO, WIN_TIE, WIN_VILLAIN};
use super::tree::Node;

#[derive(Debug, Clone)]
pub struct SolveResult {
    pub iterations: u32,
    /// {global_combo_idx: [(action_label, probability), ...]}
    pub root_strategy: HashMap<u16, Vec<(String, f32)>>,
    pub hero_value: f32,
    pub last_iter_values: Vec<f32>,
}

/// Trains CFR on a river HU subgame.
///
/// `hero_buckets` / `villain_buckets`: optional global-combo → bucket id maps
/// (length NUM_COMBOS each). When omitted, each combo is its own bucket.
pub fn solve_river(
    root: &mut Node,
    table: &ShowdownTable,
    initial_stacks: (f32, f32),
    iterations: u32,
    hero_buckets: Option<&[i32]>,
    villain_buckets: Option<&[i32]>,
) -> SolveResult {
    let solver = Solver::new(root, table, initial_stacks, hero_buckets, villain_buckets);
    let mut state = solver.state;
    state.train(root, iterations);
    let root_strategy = state.extract_root_strategy(root);
    SolveResult {
        iterations,
        root_strategy,
        hero_value: state.last_root_value(),
        last_iter_values: state.iter_values.clone(),
    }
}

pub struct Solver<'a> {
    pub state: SolverState<'a>,
}

impl<'a> Solver<'a> {
    pub fn new(
        root: &mut Node,
        table: &'a ShowdownTable,
        initial_stacks: (f32, f32),
        hero_buckets: Option<&[i32]>,
        villain_buckets: Option<&[i32]>,
    ) -> Self {
        let n_h = table.n_hero;
        let n_v = table.n_villain;

        let hero_local = resolve_bucket_map(hero_buckets, &table.hero_combos);
        let villain_local = resolve_bucket_map(villain_buckets, &table.villain_combos);

        let (hero_bucket_of, n_buckets_h) = normalize_buckets(&hero_local);
        let (villain_bucket_of, n_buckets_v) = normalize_buckets(&villain_local);

        let mut hero_combos_in_bucket = vec![Vec::new(); n_buckets_h];
        for (i, &b) in hero_bucket_of.iter().enumerate() {
            hero_combos_in_bucket[b].push(i);
        }
        let mut villain_combos_in_bucket = vec![Vec::new(); n_buckets_v];
        for (j, &b) in villain_bucket_of.iter().enumerate() {
            villain_combos_in_bucket[b].push(j);
        }

        // Assign node ids to decision nodes; allocate regret/strategy_sum per-bucket.
        let mut decision_nodes: Vec<NodeMeta> = Vec::new();
        let mut regrets: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut strategy_sum: Vec<Vec<Vec<f32>>> = Vec::new();
        assign_node_ids(root, &mut decision_nodes, &mut regrets, &mut strategy_sum, n_buckets_h, n_buckets_v);

        // Pair weight total (non-conflict pairs)
        let mut pair_weight_total = 0.0;
        for i in 0..n_h {
            let hw = table.hero_weights[i];
            for j in 0..n_v {
                if table.outcome_at(i, j) == CONFLICT {
                    continue;
                }
                pair_weight_total += hw * table.villain_weights[j];
            }
        }

        let state = SolverState {
            table,
            initial_stacks,
            n_h,
            n_v,
            hero_bucket_of,
            villain_bucket_of,
            n_buckets_h,
            n_buckets_v,
            hero_combos_in_bucket,
            villain_combos_in_bucket,
            decision_nodes,
            regrets,
            strategy_sum,
            pair_weight_total,
            iter_values: Vec::new(),
        };
        Solver { state }
    }
}

#[derive(Debug)]
struct NodeMeta {
    n_actions: usize,
    player: i8,
}

pub struct SolverState<'a> {
    pub table: &'a ShowdownTable,
    pub initial_stacks: (f32, f32),
    pub n_h: usize,
    pub n_v: usize,
    pub hero_bucket_of: Vec<usize>,
    pub villain_bucket_of: Vec<usize>,
    pub n_buckets_h: usize,
    pub n_buckets_v: usize,
    pub hero_combos_in_bucket: Vec<Vec<usize>>,
    pub villain_combos_in_bucket: Vec<Vec<usize>>,
    decision_nodes: Vec<NodeMeta>,
    /// regrets[node_id][bucket][action]
    pub regrets: Vec<Vec<Vec<f32>>>,
    pub strategy_sum: Vec<Vec<Vec<f32>>>,
    pub pair_weight_total: f32,
    pub iter_values: Vec<f32>,
}

impl<'a> SolverState<'a> {
    pub fn train(&mut self, root: &Node, iterations: u32) {
        let hero_w: Vec<f32> = self.table.hero_weights.clone();
        let villain_w: Vec<f32> = self.table.villain_weights.clone();
        for t in 1..=iterations {
            let v0 = self.cfr(root, &hero_w, &villain_w, 0, t);
            let _ = self.cfr(root, &hero_w, &villain_w, 1, t);
            if self.pair_weight_total > 0.0 {
                let mut acc = 0.0;
                for i in 0..self.n_h {
                    acc += hero_w[i] * v0[i];
                }
                self.iter_values.push(acc / self.pair_weight_total);
            }
        }
    }

    fn cfr(
        &mut self,
        node: &Node,
        reach_h: &[f32],
        reach_v: &[f32],
        updating: i8,
        t: u32,
    ) -> Vec<f32> {
        if node.is_terminal {
            return self.terminal_utility(node, reach_h, reach_v, updating);
        }
        if node.is_chance {
            let nc_upd = if updating == 0 { self.n_h } else { self.n_v };
            let mut total = vec![0.0f32; nc_upd];
            for child in node.children.iter() {
                let child_util = self.cfr(child, reach_h, reach_v, updating, t);
                for i in 0..nc_upd {
                    total[i] += child_util[i];
                }
            }
            let denom = node.children.len() as f32;
            for v in total.iter_mut() {
                *v /= denom;
            }
            return total;
        }

        let player = node.player_to_act;
        let node_id = node.node_id.get() as usize;
        let na = node.actions.len();
        let nc_upd = if updating == 0 { self.n_h } else { self.n_v };

        let strategy_b = self.regret_matching(node_id, player, na);

        let mut action_util: Vec<Vec<f32>> = Vec::with_capacity(na);
        for a in 0..na {
            let new_reach_h: Vec<f32>;
            let new_reach_v: Vec<f32>;
            if player == 0 {
                new_reach_h = (0..self.n_h)
                    .map(|i| reach_h[i] * strategy_b[self.hero_bucket_of[i]][a])
                    .collect();
                new_reach_v = reach_v.to_vec();
                action_util.push(self.cfr(&node.children[a], &new_reach_h, &new_reach_v, updating, t));
            } else {
                new_reach_h = reach_h.to_vec();
                new_reach_v = (0..self.n_v)
                    .map(|j| reach_v[j] * strategy_b[self.villain_bucket_of[j]][a])
                    .collect();
                action_util.push(self.cfr(&node.children[a], &new_reach_h, &new_reach_v, updating, t));
            }
        }

        let mut node_util = vec![0.0f32; nc_upd];
        if player == updating {
            let bucket_of = if player == 0 { &self.hero_bucket_of } else { &self.villain_bucket_of };
            for i in 0..nc_upd {
                let s_row = &strategy_b[bucket_of[i]];
                for a in 0..na {
                    node_util[i] += s_row[a] * action_util[a][i];
                }
            }
        } else {
            for i in 0..nc_upd {
                for a in 0..na {
                    node_util[i] += action_util[a][i];
                }
            }
        }

        if player == updating {
            let own_reach: &[f32] = if updating == 0 { reach_h } else { reach_v };
            let combos_in_bucket = if updating == 0 {
                &self.hero_combos_in_bucket
            } else {
                &self.villain_combos_in_bucket
            };
            let nb_own = if player == 0 { self.n_buckets_h } else { self.n_buckets_v };

            for b in 0..nb_own {
                let combos = &combos_in_bucket[b];
                if combos.is_empty() {
                    continue;
                }
                let s_row = &strategy_b[b];
                let mut bucket_reach = 0.0;
                let mut cur_ev = vec![0.0f32; combos.len()];
                for (ci, &i) in combos.iter().enumerate() {
                    let mut s = 0.0;
                    for ap in 0..na {
                        s += s_row[ap] * action_util[ap][i];
                    }
                    cur_ev[ci] = s;
                    bucket_reach += own_reach[i];
                }
                for a in 0..na {
                    let mut agg = 0.0;
                    for (ci, &i) in combos.iter().enumerate() {
                        agg += own_reach[i] * (action_util[a][i] - cur_ev[ci]);
                    }
                    let new_r = self.regrets[node_id][b][a] + agg;
                    self.regrets[node_id][b][a] = if new_r > 0.0 { new_r } else { 0.0 };
                    self.strategy_sum[node_id][b][a] += (t as f32) * bucket_reach * s_row[a];
                }
            }
        }

        node_util
    }

    fn terminal_utility(
        &self,
        node: &Node,
        reach_h: &[f32],
        reach_v: &[f32],
        updating: i8,
    ) -> Vec<f32> {
        let hero_c = self.initial_stacks.0 - node.stacks.0;
        let villain_c = self.initial_stacks.1 - node.stacks.1;
        let pot = node.terminal_pot;

        let (hero_profit, villain_profit) = match node.terminal_winner {
            Some(0) => (pot - hero_c, -villain_c),
            Some(1) => (-hero_c, pot - villain_c),
            _ => (0.0, 0.0),
        };

        let key = node.showdown_key;
        let outcome_table: &ShowdownTable = self.table; // single-board river

        if updating == 0 {
            let mut out = vec![0.0f32; self.n_h];
            if node.terminal_winner.is_some() {
                for i in 0..self.n_h {
                    let mut total_v = 0.0;
                    for j in 0..self.n_v {
                        if outcome_table.outcome_at(i, j) == CONFLICT {
                            continue;
                        }
                        total_v += reach_v[j];
                    }
                    out[i] = total_v * hero_profit;
                }
                return out;
            }
            for i in 0..self.n_h {
                let mut acc = 0.0;
                for j in 0..self.n_v {
                    let o = outcome_table.outcome_at(i, j);
                    if o == CONFLICT {
                        continue;
                    }
                    let share = if o == WIN_HERO {
                        pot
                    } else if o == WIN_TIE {
                        pot * 0.5
                    } else {
                        0.0
                    };
                    acc += reach_v[j] * (share - hero_c);
                }
                out[i] = acc;
            }
            let _ = key;
            return out;
        }

        let mut out = vec![0.0f32; self.n_v];
        if node.terminal_winner.is_some() {
            for j in 0..self.n_v {
                let mut total_h = 0.0;
                for i in 0..self.n_h {
                    if outcome_table.outcome_at(i, j) == CONFLICT {
                        continue;
                    }
                    total_h += reach_h[i];
                }
                out[j] = total_h * villain_profit;
            }
            return out;
        }
        for j in 0..self.n_v {
            let mut acc = 0.0;
            for i in 0..self.n_h {
                let o = outcome_table.outcome_at(i, j);
                if o == CONFLICT {
                    continue;
                }
                let share = if o == WIN_VILLAIN {
                    pot
                } else if o == WIN_TIE {
                    pot * 0.5
                } else {
                    0.0
                };
                acc += reach_h[i] * (share - villain_c);
            }
            out[j] = acc;
        }
        out
    }

    fn regret_matching(&self, node_id: usize, player: i8, na: usize) -> Vec<Vec<f32>> {
        let nb = if player == 0 { self.n_buckets_h } else { self.n_buckets_v };
        let mut strategy = vec![vec![0.0f32; na]; nb];
        let regrets = &self.regrets[node_id];
        let uniform = 1.0 / (na as f32);
        for i in 0..nb {
            let mut s = 0.0;
            for a in 0..na {
                if regrets[i][a] > 0.0 {
                    s += regrets[i][a];
                }
            }
            if s > 0.0 {
                for a in 0..na {
                    let r = regrets[i][a];
                    strategy[i][a] = if r > 0.0 { r / s } else { 0.0 };
                }
            } else {
                for a in 0..na {
                    strategy[i][a] = uniform;
                }
            }
        }
        strategy
    }

    pub fn last_root_value(&self) -> f32 {
        *self.iter_values.last().unwrap_or(&0.0)
    }

    pub fn extract_root_strategy(&self, root: &Node) -> HashMap<u16, Vec<(String, f32)>> {
        let root_id = root.node_id.get() as usize;
        let player = root.player_to_act;
        let combos: &[u16] = if player == 0 { &self.table.hero_combos } else { &self.table.villain_combos };
        let bucket_of = if player == 0 { &self.hero_bucket_of } else { &self.villain_bucket_of };
        let na = root.actions.len();
        let strat = &self.strategy_sum[root_id];

        let mut bucket_probs: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut out: HashMap<u16, Vec<(String, f32)>> = HashMap::new();
        for (i, &combo_idx) in combos.iter().enumerate() {
            let b = bucket_of[i];
            let probs = bucket_probs.entry(b).or_insert_with(|| {
                let row = &strat[b];
                let total: f32 = row.iter().sum();
                if total > 0.0 {
                    row.iter().map(|v| v / total).collect()
                } else {
                    vec![1.0 / (na as f32); na]
                }
            }).clone();
            let labeled: Vec<(String, f32)> = root
                .actions
                .iter()
                .zip(probs.iter())
                .map(|(action, p)| (action.label(), *p))
                .collect();
            out.insert(combo_idx, labeled);
        }
        out
    }
}

fn assign_node_ids(
    root: &Node,
    metas: &mut Vec<NodeMeta>,
    regrets: &mut Vec<Vec<Vec<f32>>>,
    strategy_sum: &mut Vec<Vec<Vec<f32>>>,
    n_buckets_h: usize,
    n_buckets_v: usize,
) {
    fn walk(
        node: &Node,
        metas: &mut Vec<NodeMeta>,
        regrets: &mut Vec<Vec<Vec<f32>>>,
        strategy_sum: &mut Vec<Vec<Vec<f32>>>,
        n_buckets_h: usize,
        n_buckets_v: usize,
    ) {
        if !node.is_terminal && !node.is_chance {
            let id = metas.len() as i32;
            node.node_id.set(id);
            let na = node.actions.len();
            let nb = if node.player_to_act == 0 { n_buckets_h } else { n_buckets_v };
            metas.push(NodeMeta { n_actions: na, player: node.player_to_act });
            regrets.push(vec![vec![0.0; na]; nb]);
            strategy_sum.push(vec![vec![0.0; na]; nb]);
        }
        for child in node.children.iter() {
            walk(child, metas, regrets, strategy_sum, n_buckets_h, n_buckets_v);
        }
    }
    walk(root, metas, regrets, strategy_sum, n_buckets_h, n_buckets_v);
}

fn resolve_bucket_map(buckets: Option<&[i32]>, combo_indices: &[u16]) -> Vec<i32> {
    match buckets {
        None => (0..combo_indices.len() as i32).collect(),
        Some(b) => combo_indices.iter().map(|&gi| b[gi as usize]).collect(),
    }
}

fn normalize_buckets(raw: &[i32]) -> (Vec<usize>, usize) {
    let mut remap: HashMap<i32, usize> = HashMap::new();
    let mut out = Vec::with_capacity(raw.len());
    for &b in raw {
        let n = remap.len();
        let id = *remap.entry(b).or_insert(n);
        out.push(id);
    }
    let count = remap.len();
    (out, count)
}

#[cfg(test)]
mod tests {
    use super::super::cards::{card_from_str, combo_index, NUM_COMBOS};
    use super::super::range_parser::parse_range;
    use super::super::showdown::compute_showdown_table;
    use super::super::tree::build_river_tree;
    use super::*;

    fn board(s: &[&str]) -> [u8; 5] {
        let mut out = [0u8; 5];
        for (i, c) in s.iter().enumerate() {
            out[i] = card_from_str(c).unwrap();
        }
        out
    }

    #[test]
    fn nut_vs_air_hero_wins_pot() {
        let b = board(&["Ad", "Kh", "7s", "3c", "2d"]);
        let hero = parse_range("AsAc").unwrap();
        let villain = parse_range("2s2h").unwrap();
        let table = compute_showdown_table(b, &hero, &villain);
        let mut root = build_river_tree(100.0, (200.0, 200.0), 0, 2);
        let result = solve_river(&mut root, &table, (200.0, 200.0), 200, None, None);
        // Hero EV near +100 (villain folds)
        assert!(result.hero_value > 70.0, "hero_value = {}", result.hero_value);
    }

    #[test]
    fn tie_gives_half_pot() {
        // Board straight 2-6: both hands play board
        let b = board(&["2h", "3d", "4s", "5c", "6d"]);
        let hero = parse_range("AsAc").unwrap();
        let villain = parse_range("KhKd").unwrap();
        let table = compute_showdown_table(b, &hero, &villain);
        let mut root = build_river_tree(100.0, (200.0, 200.0), 0, 2);
        let result = solve_river(&mut root, &table, (200.0, 200.0), 200, None, None);
        // Tie → hero EV ≈ pot/2 = 50
        assert!(
            (result.hero_value - 50.0).abs() < 12.0,
            "hero_value = {}",
            result.hero_value,
        );
    }

    #[test]
    fn convergence_iter_values_stabilize() {
        let b = board(&["Ad", "Kh", "7s", "3c", "2d"]);
        let hero = parse_range("AsAc, KsQs").unwrap();
        let villain = parse_range("KdKc, JsTs").unwrap();
        let table = compute_showdown_table(b, &hero, &villain);
        let mut root = build_river_tree(100.0, (100.0, 100.0), 0, 2);
        let result = solve_river(&mut root, &table, (100.0, 100.0), 200, None, None);
        let tail = &result.last_iter_values[result.last_iter_values.len() - 20..];
        let mn = tail.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = tail.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(mx - mn < 5.0, "spread = {}", mx - mn);
    }
}
