//! Bucketed CFR+ for multi-board subgames (turn, flop, anything with chance).
//!
//! All internal state is K-dimensional. Chance nodes are uniform-averaged.
//! Terminal showdowns look up per-board equity from a table keyed by
//! `ShowdownKey` (`River(c)` for turn solves, `TurnRiver(t,r)` for flop).
//!
//! Bucketing is FIXED at the root level — combos are bucketed once based on
//! the root board's E[HS], and the same bucketing is used throughout the
//! tree. This is a lossy abstraction (two combos in the same root-bucket may
//! have different equity on a specific runout) but it's the standard
//! approach in production solvers and gives the speedup we need.

use std::collections::HashMap;

use super::bucket_equity::BucketEquityTable;
use super::bucketing::Bucketing;
use super::tree::{Node, ShowdownKey};

/// A keyed collection of bucket equity tables, one per terminal board state.
pub struct BucketEquityStore {
    pub tables: HashMap<ShowdownKey, BucketEquityTable>,
}

impl BucketEquityStore {
    pub fn get(&self, key: ShowdownKey) -> Option<&BucketEquityTable> {
        self.tables.get(&key)
    }
}

#[derive(Debug, Clone)]
pub struct BucketSolveResult {
    pub iterations: u32,
    pub root_strategy: Vec<Vec<f32>>,
    pub action_labels: Vec<String>,
    pub hero_value: f32,
    pub last_iter_values: Vec<f32>,
}

pub fn solve_multi_bucketed(
    root: &mut Node,
    hero_bucketing: &Bucketing,
    villain_bucketing: &Bucketing,
    equity_store: &BucketEquityStore,
    initial_stacks: (f32, f32),
    iterations: u32,
) -> BucketSolveResult {
    let mut state = MultiBucketSolverState::new(
        root, hero_bucketing, villain_bucketing, equity_store, initial_stacks,
    );
    state.train(root, iterations);
    let (root_strategy, action_labels) = state.root_strategy(root);
    BucketSolveResult {
        iterations,
        root_strategy,
        action_labels,
        hero_value: state.last_root_value(),
        last_iter_values: state.iter_values.clone(),
    }
}

pub struct MultiBucketSolverState<'a> {
    pub hero_bucketing: &'a Bucketing,
    pub villain_bucketing: &'a Bucketing,
    pub equity_store: &'a BucketEquityStore,
    pub initial_stacks: (f32, f32),
    pub k_h: usize,
    pub k_v: usize,
    /// Flat storage: regrets[node_id] is Vec<f32> of size K×na, indexed as [bucket*na + action].
    pub regrets: Vec<Vec<f32>>,
    pub strategy_sum: Vec<Vec<f32>>,
    /// Per-node (na, k) metadata to decode the flat layout.
    pub node_meta: Vec<(u8, u8)>,  // (na, k)
    pub iter_values: Vec<f32>,
    /// Average per-runout total pair weight (the EV denominator).
    pub pair_weight_avg: f32,
    /// Pre-resolved equity tables (flat vec indexed by terminal's numeric key).
    /// Avoids HashMap lookups in the hot CFR loop.
    equity_tables: Vec<BucketEquityTable>,
    /// Default (fold) table pointer (for when terminal_winner.is_some()).
    fold_fallback_idx: usize,
}

impl<'a> MultiBucketSolverState<'a> {
    pub fn new(
        root: &mut Node,
        hero_bucketing: &'a Bucketing,
        villain_bucketing: &'a Bucketing,
        equity_store: &'a BucketEquityStore,
        initial_stacks: (f32, f32),
    ) -> Self {
        let k_h = hero_bucketing.k;
        let k_v = villain_bucketing.k;

        let mut regrets: Vec<Vec<f32>> = Vec::new();
        let mut strategy_sum: Vec<Vec<f32>> = Vec::new();
        let mut node_meta: Vec<(u8, u8)> = Vec::new();
        walk_decision(root, &mut |node| {
            let id = regrets.len() as i32;
            node.node_id.set(id);
            let na = node.actions.len();
            let k = if node.player_to_act == 0 { k_h } else { k_v };
            regrets.push(vec![0.0; k * na]);
            strategy_sum.push(vec![0.0; k * na]);
            node_meta.push((na as u8, k as u8));
        });

        // pair_weight_avg = average over keyed equity tables of total pair_weight
        let mut total = 0.0f32;
        let n = equity_store.tables.len().max(1);
        for table in equity_store.tables.values() {
            total += table.pair_weight.iter().sum::<f32>();
        }
        let pair_weight_avg = total / n as f32;

        // Flatten equity tables + resolve terminals to direct indices.
        // Node visits hot loop: O(1) array index instead of HashMap lookup.
        let equity_tables: Vec<BucketEquityTable> =
            equity_store.tables.values().cloned().collect();
        let mut key_to_idx: HashMap<ShowdownKey, usize> = HashMap::new();
        for (idx, k) in equity_store.tables.keys().enumerate() {
            key_to_idx.insert(*k, idx);
        }
        // Tag each terminal (showdown or fold) with its equity-table index.
        resolve_terminals(root, &key_to_idx);

        Self {
            hero_bucketing, villain_bucketing, equity_store, initial_stacks,
            k_h, k_v, regrets, strategy_sum, node_meta, iter_values: Vec::new(),
            pair_weight_avg,
            equity_tables,
            fold_fallback_idx: 0,
        }
    }

    pub fn train(&mut self, root: &Node, iterations: u32) {
        let reach_h = self.hero_bucketing.bucket_weight.clone();
        let reach_v = self.villain_bucketing.bucket_weight.clone();
        let mut v0 = vec![0.0f32; self.k_h];
        let mut v1 = vec![0.0f32; self.k_v];
        for t in 1..=iterations {
            for x in v0.iter_mut() { *x = 0.0; }
            for x in v1.iter_mut() { *x = 0.0; }
            self.cfr(root, &reach_h, &reach_v, 0, t, &mut v0);
            self.cfr(root, &reach_h, &reach_v, 1, t, &mut v1);
            if self.pair_weight_avg > 0.0 {
                let mut acc = 0.0f32;
                for h in 0..self.k_h { acc += reach_h[h] * v0[h]; }
                self.iter_values.push(acc / self.pair_weight_avg);
            }
        }
    }

    /// Caller-buffer version of CFR. Writes the per-bucket counterfactual
    /// value for the `updating` player into `out` (caller must size it to k_upd
    /// and zero it beforehand — this fn writes only, does NOT accumulate).
    fn cfr(&mut self, node: &Node, reach_h: &[f32], reach_v: &[f32], updating: i8, t: u32, out: &mut [f32]) {
        if node.is_terminal {
            self.terminal_utility_into(node, reach_h, reach_v, updating, out);
            return;
        }
        if node.is_chance {
            let k_upd = if updating == 0 { self.k_h } else { self.k_v };
            for x in out.iter_mut() { *x = 0.0; }
            let denom = node.children.len() as f32;
            let inv_denom = 1.0 / denom;
            let mut child_buf = vec![0.0f32; k_upd];
            for child in node.children.iter() {
                for x in child_buf.iter_mut() { *x = 0.0; }
                self.cfr(child, reach_h, reach_v, updating, t, &mut child_buf);
                for i in 0..k_upd { out[i] += child_buf[i] * inv_denom; }
            }
            return;
        }

        let player = node.player_to_act;
        let nid = node.node_id.get() as usize;
        let na = node.actions.len();
        let k_own = if player == 0 { self.k_h } else { self.k_v };
        let k_upd = if updating == 0 { self.k_h } else { self.k_v };

        let strategy = self.regret_matching(nid, k_own, na);

        let k_for_scratch = if player == 0 { self.k_h } else { self.k_v };
        let mut scratch_reach = vec![0.0f32; k_for_scratch];

        // Flat action_util: Vec<f32> of size na * k_upd, indexed [a*k_upd + b].
        let mut action_util = vec![0.0f32; na * k_upd];
        for a in 0..na {
            let slot = &mut action_util[a * k_upd..(a + 1) * k_upd];
            if player == 0 {
                for h in 0..self.k_h { scratch_reach[h] = reach_h[h] * strategy[h * na + a]; }
                self.cfr(&node.children[a], &scratch_reach, reach_v, updating, t, slot);
            } else {
                for v in 0..self.k_v { scratch_reach[v] = reach_v[v] * strategy[v * na + a]; }
                self.cfr(&node.children[a], reach_h, &scratch_reach, updating, t, slot);
            }
        }

        // Compute node_util (output)
        for x in out.iter_mut() { *x = 0.0; }
        if player == updating {
            for b in 0..k_upd {
                let sbase = b * na;
                let mut acc = 0.0f32;
                for a in 0..na { acc += strategy[sbase + a] * action_util[a * k_upd + b]; }
                out[b] = acc;
            }
        } else {
            for b in 0..k_upd {
                let mut acc = 0.0f32;
                for a in 0..na { acc += action_util[a * k_upd + b]; }
                out[b] = acc;
            }
        }

        // Regret + strategy_sum update
        if player == updating {
            let own_reach: &[f32] = if updating == 0 { reach_h } else { reach_v };
            let t_f = t as f32;
            let regrets = &mut self.regrets[nid];
            let strat_sum = &mut self.strategy_sum[nid];
            for b in 0..k_own {
                let rbase = b * na;
                let own_r = own_reach[b];
                let mut cur_ev = 0.0f32;
                for a in 0..na { cur_ev += strategy[rbase + a] * action_util[a * k_upd + b]; }
                for a in 0..na {
                    let regret = action_util[a * k_upd + b] - cur_ev;
                    let new_r = regrets[rbase + a] + own_r * regret;
                    regrets[rbase + a] = if new_r > 0.0 { new_r } else { 0.0 };
                    strat_sum[rbase + a] += t_f * own_r * strategy[rbase + a];
                }
            }
        }
    }

    /// In-place variant: writes result into `out` slice. Still keeps the
    /// original return-Vec helper below for Best-Response etc that want a Vec.
    fn terminal_utility_into(&self, node: &Node, reach_h: &[f32], reach_v: &[f32], updating: i8, out: &mut [f32]) {
        let v = self.terminal_utility(node, reach_h, reach_v, updating);
        out.copy_from_slice(&v);
    }

    fn terminal_utility(&self, node: &Node, reach_h: &[f32], reach_v: &[f32], updating: i8) -> Vec<f32> {
        let hero_c = self.initial_stacks.0 - node.stacks.0;
        let villain_c = self.initial_stacks.1 - node.stacks.1;
        let pot = node.terminal_pot;

        let (hero_profit, villain_profit) = match node.terminal_winner {
            Some(0) => (pot - hero_c, -villain_c),
            Some(1) => (-hero_c, pot - villain_c),
            _ => (0.0, 0.0),
        };

        // Look up the equity table via the pre-resolved index on the node.
        // Avoids HashMap hashing on the hot path.
        let eidx = node.equity_idx.get();
        let equity_table: &BucketEquityTable = if eidx >= 0 && (eidx as usize) < self.equity_tables.len() {
            &self.equity_tables[eidx as usize]
        } else if node.terminal_winner.is_some() && !self.equity_tables.is_empty() {
            // Fold: any table's pair_weight matrix is fine (non-conflict mask is shared
            // across boards since it only depends on hole-card overlap).
            &self.equity_tables[self.fold_fallback_idx]
        } else {
            return vec![0.0; if updating == 0 { self.k_h } else { self.k_v }];
        };

        if updating == 0 {
            let mut out = vec![0.0f32; self.k_h];
            for h in 0..self.k_h {
                let bw_h = self.hero_bucketing.bucket_weight[h];
                if bw_h <= 0.0 { continue; }
                let mut acc = 0.0f32;
                for v in 0..self.k_v {
                    let pw = equity_table.pair_weight_at(h, v);
                    let bw_v = self.villain_bucketing.bucket_weight[v];
                    if bw_v <= 0.0 || pw <= 0.0 { continue; }
                    let payoff = if node.terminal_winner.is_some() {
                        hero_profit
                    } else {
                        let eq = equity_table.equity_sum_at(h, v) / pw;
                        eq * pot - hero_c
                    };
                    acc += reach_v[v] * (pw / bw_v) * payoff;
                }
                out[h] = acc / bw_h;
            }
            return out;
        }

        let mut out = vec![0.0f32; self.k_v];
        for v in 0..self.k_v {
            let bw_v = self.villain_bucketing.bucket_weight[v];
            if bw_v <= 0.0 { continue; }
            let mut acc = 0.0f32;
            for h in 0..self.k_h {
                let pw = equity_table.pair_weight_at(h, v);
                let bw_h = self.hero_bucketing.bucket_weight[h];
                if bw_h <= 0.0 || pw <= 0.0 { continue; }
                let payoff = if node.terminal_winner.is_some() {
                    villain_profit
                } else {
                    let eq_hero = equity_table.equity_sum_at(h, v) / pw;
                    let eq_villain = 1.0 - eq_hero;
                    eq_villain * pot - villain_c
                };
                acc += reach_h[h] * (pw / bw_h) * payoff;
            }
            out[v] = acc / bw_v;
        }
        out
    }

    /// Returns a flat Vec<f32> of size k*na indexed by [bucket*na + action].
    fn regret_matching(&self, nid: usize, k: usize, na: usize) -> Vec<f32> {
        let mut strategy = vec![0.0f32; k * na];
        let regrets = &self.regrets[nid];
        let uniform = 1.0 / (na as f32);
        for b in 0..k {
            let base = b * na;
            let mut s = 0.0f32;
            for a in 0..na {
                let r = regrets[base + a];
                if r > 0.0 { s += r; }
            }
            if s > 0.0 {
                for a in 0..na {
                    let r = regrets[base + a];
                    strategy[base + a] = if r > 0.0 { r / s } else { 0.0 };
                }
            } else {
                for a in 0..na { strategy[base + a] = uniform; }
            }
        }
        strategy
    }

    pub fn last_root_value(&self) -> f32 {
        *self.iter_values.last().unwrap_or(&0.0)
    }

    pub fn root_strategy(&self, root: &Node) -> (Vec<Vec<f32>>, Vec<String>) {
        let nid = root.node_id.get() as usize;
        let na = root.actions.len();
        let (_, k) = self.node_meta[nid];
        let k = k as usize;
        let strat = &self.strategy_sum[nid];
        let probs: Vec<Vec<f32>> = (0..k).map(|b| {
            let base = b * na;
            let total: f32 = (0..na).map(|a| strat[base + a]).sum();
            if total > 0.0 { (0..na).map(|a| strat[base + a] / total).collect() }
            else { vec![1.0 / (na as f32); na] }
        }).collect();
        let labels: Vec<String> = root.actions.iter().map(|a| a.label()).collect();
        (probs, labels)
    }
}

fn walk_decision(root: &mut Node, f: &mut impl FnMut(&mut Node)) {
    if !root.is_terminal && !root.is_chance {
        f(root);
    }
    for child in root.children.iter_mut() {
        walk_decision(child, f);
    }
}

/// Tag each terminal's `equity_idx` with the index into the solver's flat
/// `equity_tables` array. Showdown terminals use their showdown_key; fold
/// terminals get -1 (unused).
fn resolve_terminals(root: &mut Node, key_to_idx: &HashMap<ShowdownKey, usize>) {
    if root.is_terminal {
        let idx = if root.terminal_winner.is_none() {
            key_to_idx.get(&root.showdown_key).copied().unwrap_or(usize::MAX)
        } else {
            usize::MAX
        };
        root.equity_idx.set(idx as i32);
        return;
    }
    for child in root.children.iter_mut() {
        resolve_terminals(child, key_to_idx);
    }
}

// ===== Helpers to build BucketEquityStore for turn / flop =====

use super::bucket_equity::compute_bucket_equity_river;

/// For each remaining river card, compute a bucket equity table on the
/// 5-card board (board_4 + river). Caller supplies the FIXED (turn-level)
/// bucketings to use across all 48 boards.
pub fn build_turn_equity_store(
    board_4: [u8; 4],
    hero_bucketing: &Bucketing,
    villain_bucketing: &Bucketing,
    river_subset: Option<&[u8]>,
) -> BucketEquityStore {
    use super::cards::NUM_CARDS;
    let board_mask: u64 = board_4.iter().fold(0u64, |a, &c| a | (1u64 << c));
    let rivers: Vec<u8> = match river_subset {
        None => (0..NUM_CARDS as u8).filter(|c| board_mask & (1u64 << c) == 0).collect(),
        Some(s) => s.to_vec(),
    };
    let mut tables = HashMap::new();
    for r in rivers {
        let mut board_5 = [0u8; 5];
        board_5[..4].copy_from_slice(&board_4);
        board_5[4] = r;
        let table = compute_bucket_equity_river(&board_5, hero_bucketing, villain_bucketing);
        tables.insert(ShowdownKey::River(r), table);
    }
    BucketEquityStore { tables }
}

/// For each (turn, river) pair, compute a bucket equity table.
pub fn build_flop_equity_store(
    board_3: [u8; 3],
    hero_bucketing: &Bucketing,
    villain_bucketing: &Bucketing,
    turn_subset: Option<&[u8]>,
    river_subset: Option<&[u8]>,
) -> BucketEquityStore {
    use super::cards::NUM_CARDS;
    let board_mask: u64 = board_3.iter().fold(0u64, |a, &c| a | (1u64 << c));
    let all_remaining: Vec<u8> = (0..NUM_CARDS as u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
    let turns: Vec<u8> = match turn_subset {
        None => all_remaining.clone(),
        Some(s) => s.to_vec(),
    };
    let rivers_base: Vec<u8> = match river_subset {
        None => all_remaining.clone(),
        Some(s) => s.to_vec(),
    };
    let mut tables = HashMap::new();
    for &t in &turns {
        for &r in &rivers_base {
            if r == t { continue; }
            let mut board_5 = [0u8; 5];
            board_5[..3].copy_from_slice(&board_3);
            board_5[3] = t;
            board_5[4] = r;
            let table = compute_bucket_equity_river(&board_5, hero_bucketing, villain_bucketing);
            tables.insert(ShowdownKey::TurnRiver(t, r), table);
        }
    }
    BucketEquityStore { tables }
}

#[cfg(test)]
mod tests {
    use super::super::bucketing::{bucket_by_ehs, river_ehs};
    use super::super::cards::{card_from_str, NUM_COMBOS};
    use super::super::tree::build_turn_tree;
    use super::*;

    #[test]
    fn bucketed_turn_runs() {
        let board_4: [u8; 4] = ["Ad", "Kh", "7s", "3c"].map(|s| card_from_str(s).unwrap());
        // Use river-EHS approximation for the bucketing (faster than turn_ehs in tests)
        let mut board_5 = [0u8; 5];
        board_5[..4].copy_from_slice(&board_4);
        board_5[4] = card_from_str("2d").unwrap();
        let ehs = river_ehs(&board_5);
        let any_two = vec![1.0f32; NUM_COMBOS];
        let bucketing = bucket_by_ehs(&ehs, &any_two, &board_5, 8);

        let store = build_turn_equity_store(board_4, &bucketing, &bucketing, None);
        assert_eq!(store.tables.len(), 48);

        let mut root = build_turn_tree(board_4, 100.0, (200.0, 200.0), 0, 1, 1);
        let result = solve_multi_bucketed(
            &mut root, &bucketing, &bucketing, &store, (200.0, 200.0), 50,
        );
        // Symmetric → ~pot/2 = 50
        assert!(result.hero_value > 20.0 && result.hero_value < 80.0,
                "turn hero_value = {}", result.hero_value);
    }
}
