//! Multi-board CFR+ for turn and flop subgames.
//!
//! Trait-based: any "outcome lookup" that exposes the right interface can be
//! plugged into the same CFR core. River solving uses `cfr::solve_river`
//! (single board); turn/flop solving uses this with `TurnShowdown` /
//! `FlopShowdown` adapters.

use std::collections::HashMap;

use super::showdown::{CONFLICT, WIN_HERO, WIN_TIE, WIN_VILLAIN};
use super::tree::{Node, ShowdownKey};
use super::turn_showdown::TurnShowdown;

pub trait MultiBoardLookup {
    fn n_hero(&self) -> usize;
    fn n_villain(&self) -> usize;
    fn hero_combos(&self) -> &[u16];
    fn hero_weights(&self) -> &[f32];
    fn villain_combos(&self) -> &[u16];
    fn villain_weights(&self) -> &[f32];
    /// Outcome for (hero_local, villain_local) at the showdown identified by key.
    fn outcome(&self, key: ShowdownKey, hero_local: usize, villain_local: usize) -> i8;
    /// Hero pair cards (a, b) for a local index — used for fold terminals where
    /// we need to filter mutually-conflicting (i, j) pairs without a per-board
    /// outcome matrix.
    fn hero_pair(&self, hero_local: usize) -> (u8, u8);
    fn villain_pair(&self, villain_local: usize) -> (u8, u8);
    /// Average per-runout total non-conflict pair weight (the EV denominator).
    fn pair_weight_avg(&self) -> f32;
}

impl MultiBoardLookup for TurnShowdown {
    fn n_hero(&self) -> usize { self.n_hero }
    fn n_villain(&self) -> usize { self.n_villain }
    fn hero_combos(&self) -> &[u16] { &self.hero_combos }
    fn hero_weights(&self) -> &[f32] { &self.hero_weights }
    fn villain_combos(&self) -> &[u16] { &self.villain_combos }
    fn villain_weights(&self) -> &[f32] { &self.villain_weights }
    fn outcome(&self, key: ShowdownKey, h: usize, v: usize) -> i8 {
        match key {
            ShowdownKey::River(r) => self.outcome_at(r, h, v),
            _ => panic!("turn solver expects ShowdownKey::River, got {:?}", key),
        }
    }
    fn hero_pair(&self, h: usize) -> (u8, u8) {
        super::cards::combo_cards(self.hero_combos[h] as usize)
    }
    fn villain_pair(&self, v: usize) -> (u8, u8) {
        super::cards::combo_cards(self.villain_combos[v] as usize)
    }
    fn pair_weight_avg(&self) -> f32 {
        let mut total = 0.0f32;
        for &r in &self.river_cards {
            for i in 0..self.n_hero {
                let hw = self.hero_weights[i];
                for j in 0..self.n_villain {
                    if self.outcome_at(r, i, j) == CONFLICT { continue; }
                    total += hw * self.villain_weights[j];
                }
            }
        }
        total / (self.river_cards.len() as f32)
    }
}

#[derive(Debug, Clone)]
pub struct MultiSolveResult {
    pub iterations: u32,
    pub root_strategy: HashMap<u16, Vec<(String, f32)>>,
    pub hero_value: f32,
    pub last_iter_values: Vec<f32>,
}

pub fn solve_multi<L: MultiBoardLookup>(
    root: &mut Node,
    lookup: &L,
    initial_stacks: (f32, f32),
    iterations: u32,
    hero_buckets: Option<&[i32]>,
    villain_buckets: Option<&[i32]>,
) -> MultiSolveResult {
    let mut state = MultiSolverState::new(root, lookup, initial_stacks, hero_buckets, villain_buckets);
    state.train(root, iterations);
    let root_strategy = state.extract_root_strategy(root);
    MultiSolveResult {
        iterations,
        root_strategy,
        hero_value: state.last_root_value(),
        last_iter_values: state.iter_values.clone(),
    }
}

pub struct MultiSolverState<'a, L: MultiBoardLookup> {
    pub lookup: &'a L,
    pub initial_stacks: (f32, f32),
    pub n_h: usize,
    pub n_v: usize,
    pub hero_bucket_of: Vec<usize>,
    pub villain_bucket_of: Vec<usize>,
    pub n_buckets_h: usize,
    pub n_buckets_v: usize,
    pub hero_combos_in_bucket: Vec<Vec<usize>>,
    pub villain_combos_in_bucket: Vec<Vec<usize>>,
    pub regrets: Vec<Vec<Vec<f32>>>,
    pub strategy_sum: Vec<Vec<Vec<f32>>>,
    pub pair_weight_avg: f32,
    pub iter_values: Vec<f32>,
    pub hero_pairs: Vec<(u8, u8)>,
    pub villain_pairs: Vec<(u8, u8)>,
}

impl<'a, L: MultiBoardLookup> MultiSolverState<'a, L> {
    pub fn new(
        root: &mut Node,
        lookup: &'a L,
        initial_stacks: (f32, f32),
        hero_buckets: Option<&[i32]>,
        villain_buckets: Option<&[i32]>,
    ) -> Self {
        let n_h = lookup.n_hero();
        let n_v = lookup.n_villain();

        let hero_local: Vec<i32> = match hero_buckets {
            None => (0..n_h as i32).collect(),
            Some(b) => lookup.hero_combos().iter().map(|&gi| b[gi as usize]).collect(),
        };
        let villain_local: Vec<i32> = match villain_buckets {
            None => (0..n_v as i32).collect(),
            Some(b) => lookup.villain_combos().iter().map(|&gi| b[gi as usize]).collect(),
        };
        let (hero_bucket_of, n_buckets_h) = normalize_buckets(&hero_local);
        let (villain_bucket_of, n_buckets_v) = normalize_buckets(&villain_local);

        let mut hero_combos_in_bucket = vec![Vec::new(); n_buckets_h];
        for (i, &b) in hero_bucket_of.iter().enumerate() { hero_combos_in_bucket[b].push(i); }
        let mut villain_combos_in_bucket = vec![Vec::new(); n_buckets_v];
        for (j, &b) in villain_bucket_of.iter().enumerate() { villain_combos_in_bucket[b].push(j); }

        let mut regrets: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut strategy_sum: Vec<Vec<Vec<f32>>> = Vec::new();
        assign_node_ids(root, &mut regrets, &mut strategy_sum, n_buckets_h, n_buckets_v);

        let hero_pairs: Vec<(u8, u8)> = (0..n_h).map(|i| lookup.hero_pair(i)).collect();
        let villain_pairs: Vec<(u8, u8)> = (0..n_v).map(|j| lookup.villain_pair(j)).collect();

        Self {
            lookup,
            initial_stacks,
            n_h, n_v,
            hero_bucket_of, villain_bucket_of,
            n_buckets_h, n_buckets_v,
            hero_combos_in_bucket, villain_combos_in_bucket,
            regrets, strategy_sum,
            pair_weight_avg: lookup.pair_weight_avg(),
            iter_values: Vec::new(),
            hero_pairs, villain_pairs,
        }
    }

    pub fn train(&mut self, root: &Node, iterations: u32) {
        let hero_w: Vec<f32> = self.lookup.hero_weights().to_vec();
        let villain_w: Vec<f32> = self.lookup.villain_weights().to_vec();
        for t in 1..=iterations {
            let v0 = self.cfr(root, &hero_w, &villain_w, 0, t);
            let _ = self.cfr(root, &hero_w, &villain_w, 1, t);
            if self.pair_weight_avg > 0.0 {
                let mut acc = 0.0f32;
                for i in 0..self.n_h { acc += hero_w[i] * v0[i]; }
                self.iter_values.push(acc / self.pair_weight_avg);
            }
        }
    }

    fn cfr(&mut self, node: &Node, reach_h: &[f32], reach_v: &[f32], updating: i8, t: u32) -> Vec<f32> {
        if node.is_terminal {
            return self.terminal_utility(node, reach_h, reach_v, updating);
        }
        if node.is_chance {
            let nc_upd = if updating == 0 { self.n_h } else { self.n_v };
            let mut total = vec![0.0f32; nc_upd];
            for child in node.children.iter() {
                let cu = self.cfr(child, reach_h, reach_v, updating, t);
                for i in 0..nc_upd { total[i] += cu[i]; }
            }
            let denom = node.children.len() as f32;
            for v in total.iter_mut() { *v /= denom; }
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
                new_reach_h = (0..self.n_h).map(|i| reach_h[i] * strategy_b[self.hero_bucket_of[i]][a]).collect();
                new_reach_v = reach_v.to_vec();
            } else {
                new_reach_h = reach_h.to_vec();
                new_reach_v = (0..self.n_v).map(|j| reach_v[j] * strategy_b[self.villain_bucket_of[j]][a]).collect();
            }
            action_util.push(self.cfr(&node.children[a], &new_reach_h, &new_reach_v, updating, t));
        }

        let mut node_util = vec![0.0f32; nc_upd];
        if player == updating {
            let bucket_of = if player == 0 { &self.hero_bucket_of } else { &self.villain_bucket_of };
            for i in 0..nc_upd {
                let s_row = &strategy_b[bucket_of[i]];
                for a in 0..na { node_util[i] += s_row[a] * action_util[a][i]; }
            }
        } else {
            for i in 0..nc_upd {
                for a in 0..na { node_util[i] += action_util[a][i]; }
            }
        }

        if player == updating {
            let own_reach: &[f32] = if updating == 0 { reach_h } else { reach_v };
            let combos_in_bucket = if updating == 0 { &self.hero_combos_in_bucket } else { &self.villain_combos_in_bucket };
            let nb_own = if player == 0 { self.n_buckets_h } else { self.n_buckets_v };
            for b in 0..nb_own {
                let combos = &combos_in_bucket[b];
                if combos.is_empty() { continue; }
                let s_row = &strategy_b[b];
                let mut bucket_reach = 0.0;
                let mut cur_ev = vec![0.0f32; combos.len()];
                for (ci, &i) in combos.iter().enumerate() {
                    let mut s = 0.0;
                    for ap in 0..na { s += s_row[ap] * action_util[ap][i]; }
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

    fn terminal_utility(&self, node: &Node, reach_h: &[f32], reach_v: &[f32], updating: i8) -> Vec<f32> {
        let hero_c = self.initial_stacks.0 - node.stacks.0;
        let villain_c = self.initial_stacks.1 - node.stacks.1;
        let pot = node.terminal_pot;

        let (hero_profit, villain_profit) = match node.terminal_winner {
            Some(0) => (pot - hero_c, -villain_c),
            Some(1) => (-hero_c, pot - villain_c),
            _ => (0.0f32, 0.0f32),
        };

        if node.terminal_winner.is_some() {
            // Fold terminal — filter (i, j) pairs that share cards.
            if updating == 0 {
                let mut out = vec![0.0f32; self.n_h];
                for i in 0..self.n_h {
                    let (ha, hb) = self.hero_pairs[i];
                    let mut total_v = 0.0;
                    for j in 0..self.n_v {
                        let (va, vb) = self.villain_pairs[j];
                        if va == ha || va == hb || vb == ha || vb == hb { continue; }
                        total_v += reach_v[j];
                    }
                    out[i] = total_v * hero_profit;
                }
                return out;
            }
            let mut out = vec![0.0f32; self.n_v];
            for j in 0..self.n_v {
                let (va, vb) = self.villain_pairs[j];
                let mut total_h = 0.0;
                for i in 0..self.n_h {
                    let (ha, hb) = self.hero_pairs[i];
                    if ha == va || ha == vb || hb == va || hb == vb { continue; }
                    total_h += reach_h[i];
                }
                out[j] = total_h * villain_profit;
            }
            return out;
        }

        let key = node.showdown_key;
        if updating == 0 {
            let mut out = vec![0.0f32; self.n_h];
            for i in 0..self.n_h {
                let mut acc = 0.0;
                for j in 0..self.n_v {
                    let o = self.lookup.outcome(key, i, j);
                    if o == CONFLICT { continue; }
                    let share = if o == WIN_HERO { pot } else if o == WIN_TIE { pot * 0.5 } else { 0.0 };
                    acc += reach_v[j] * (share - hero_c);
                }
                out[i] = acc;
            }
            return out;
        }
        let mut out = vec![0.0f32; self.n_v];
        for j in 0..self.n_v {
            let mut acc = 0.0;
            for i in 0..self.n_h {
                let o = self.lookup.outcome(key, i, j);
                if o == CONFLICT { continue; }
                let share = if o == WIN_VILLAIN { pot } else if o == WIN_TIE { pot * 0.5 } else { 0.0 };
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
                if regrets[i][a] > 0.0 { s += regrets[i][a]; }
            }
            if s > 0.0 {
                for a in 0..na {
                    let r = regrets[i][a];
                    strategy[i][a] = if r > 0.0 { r / s } else { 0.0 };
                }
            } else {
                for a in 0..na { strategy[i][a] = uniform; }
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
        let combos: &[u16] = if player == 0 { self.lookup.hero_combos() } else { self.lookup.villain_combos() };
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
                if total > 0.0 { row.iter().map(|v| v / total).collect() }
                else { vec![1.0 / (na as f32); na] }
            }).clone();
            let labeled: Vec<(String, f32)> = root.actions.iter().zip(probs.iter())
                .map(|(action, p)| (action.label(), *p)).collect();
            out.insert(combo_idx, labeled);
        }
        out
    }
}

fn assign_node_ids(
    root: &Node,
    regrets: &mut Vec<Vec<Vec<f32>>>,
    strategy_sum: &mut Vec<Vec<Vec<f32>>>,
    n_buckets_h: usize,
    n_buckets_v: usize,
) {
    fn walk(node: &Node, regrets: &mut Vec<Vec<Vec<f32>>>, strategy_sum: &mut Vec<Vec<Vec<f32>>>, n_buckets_h: usize, n_buckets_v: usize) {
        if !node.is_terminal && !node.is_chance {
            let id = regrets.len() as i32;
            node.node_id.set(id);
            let na = node.actions.len();
            let nb = if node.player_to_act == 0 { n_buckets_h } else { n_buckets_v };
            regrets.push(vec![vec![0.0; na]; nb]);
            strategy_sum.push(vec![vec![0.0; na]; nb]);
        }
        for child in node.children.iter() {
            walk(child, regrets, strategy_sum, n_buckets_h, n_buckets_v);
        }
    }
    walk(root, regrets, strategy_sum, n_buckets_h, n_buckets_v);
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
    use super::super::cards::card_from_str;
    use super::super::range_parser::parse_range;
    use super::super::tree::build_turn_tree;
    use super::super::turn_showdown::compute_turn_showdown;
    use super::*;

    fn b4(s: &[&str]) -> [u8; 4] {
        let mut out = [0u8; 4];
        for (i, c) in s.iter().enumerate() { out[i] = card_from_str(c).unwrap(); }
        out
    }

    #[test]
    fn turn_solver_runs() {
        let board = b4(&["Ad", "Kh", "7s", "3c"]);
        let hero = parse_range("AsAc").unwrap();
        let villain = parse_range("3h2h").unwrap();
        let table = compute_turn_showdown(board, &hero, &villain);
        let mut root = build_turn_tree(board, 100.0, (100.0, 100.0), 0, 1, 1);
        let result = solve_multi(&mut root, &table, (100.0, 100.0), 30, None, None);
        // Hero (AA) crushes villain (32 high) on this board → high EV
        assert!(result.hero_value > 30.0, "hero_value = {}", result.hero_value);
    }
}
