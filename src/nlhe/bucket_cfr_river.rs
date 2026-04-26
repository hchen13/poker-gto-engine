//! Bucketed CFR+ for river HU subgames.
//!
//! Everything is K-dimensional: reach, regrets, strategy, per-bucket values.
//! Terminal evaluation uses a precomputed `BucketEquityTable` (O(K²) per terminal,
//! vs O(n_combos²) in the unbucketed implementation).
//!
//! Card conflicts are handled at BucketEquityTable construction time (the
//! `pair_weight` field excludes conflicting combo pairs), so the CFR inner
//! loop doesn't need per-combo conflict checks.

use std::collections::HashMap;

use super::bucket_equity::BucketEquityTable;
use super::bucketing::Bucketing;
use super::tree::Node;

#[derive(Debug, Clone)]
pub struct BucketSolveResult {
    pub iterations: u32,
    /// [bucket][action] = prob. Only for the hero side (player 0 at root).
    pub root_strategy: Vec<Vec<f32>>,
    pub action_labels: Vec<String>,
    pub hero_value: f32,
    pub last_iter_values: Vec<f32>,
    /// Strategies for every decision node: (path, player, action_labels, strategy[bucket][action]).
    pub all_nodes: Vec<(String, i8, Vec<String>, Vec<Vec<f32>>)>,
}

pub fn solve_river_bucketed(
    root: &mut Node,
    hero_bucketing: &Bucketing,
    villain_bucketing: &Bucketing,
    equity_table: &BucketEquityTable,
    initial_stacks: (f32, f32),
    iterations: u32,
) -> BucketSolveResult {
    let mut state = BucketSolverState::new(
        root, hero_bucketing, villain_bucketing, equity_table, initial_stacks,
    );
    state.train(root, iterations);
    let (root_strategy, action_labels) = state.root_strategy(root);
    let all_nodes = state.all_node_strategies(root);
    BucketSolveResult {
        iterations,
        root_strategy,
        action_labels,
        hero_value: state.last_root_value(),
        last_iter_values: state.iter_values.clone(),
        all_nodes,
    }
}

pub struct BucketSolverState<'a> {
    pub hero_bucketing: &'a Bucketing,
    pub villain_bucketing: &'a Bucketing,
    pub equity: &'a BucketEquityTable,
    pub initial_stacks: (f32, f32),
    pub k_h: usize,
    pub k_v: usize,
    pub regrets: Vec<Vec<Vec<f32>>>,       // [node][bucket][action]
    pub strategy_sum: Vec<Vec<Vec<f32>>>,
    pub iter_values: Vec<f32>,
    pub total_pair_weight: f32,             // sum over all (h,v) of pair_weight
}

impl<'a> BucketSolverState<'a> {
    pub fn new(
        root: &mut Node,
        hero_bucketing: &'a Bucketing,
        villain_bucketing: &'a Bucketing,
        equity: &'a BucketEquityTable,
        initial_stacks: (f32, f32),
    ) -> Self {
        let k_h = hero_bucketing.k;
        let k_v = villain_bucketing.k;

        let mut regrets: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut strategy_sum: Vec<Vec<Vec<f32>>> = Vec::new();
        walk_decision_nodes(root, &mut |node| {
            let na = node.actions.len();
            let k = if node.player_to_act == 0 { k_h } else { k_v };
            let id = regrets.len() as i32;
            node.node_id.set(id);
            regrets.push(vec![vec![0.0; na]; k]);
            strategy_sum.push(vec![vec![0.0; na]; k]);
        });

        let total_pair_weight: f32 = equity.pair_weight.iter().sum();

        Self {
            hero_bucketing, villain_bucketing, equity, initial_stacks,
            k_h, k_v, regrets, strategy_sum, iter_values: Vec::new(),
            total_pair_weight,
        }
    }

    pub fn train(&mut self, root: &Node, iterations: u32) {
        let reach_h: Vec<f32> = self.hero_bucketing.bucket_weight.clone();
        let reach_v: Vec<f32> = self.villain_bucketing.bucket_weight.clone();
        for t in 1..=iterations {
            let v0 = self.cfr(root, &reach_h, &reach_v, 0, t);
            let _ = self.cfr(root, &reach_h, &reach_v, 1, t);
            if self.total_pair_weight > 0.0 {
                let mut acc = 0.0f32;
                for h in 0..self.k_h { acc += reach_h[h] * v0[h]; }
                self.iter_values.push(acc / self.total_pair_weight);
            }
        }
    }

    fn cfr(&mut self, node: &Node, reach_h: &[f32], reach_v: &[f32], updating: i8, t: u32) -> Vec<f32> {
        if node.is_terminal {
            return self.terminal_utility(node, reach_h, reach_v, updating);
        }
        // River bucket solver: no chance nodes.
        let player = node.player_to_act;
        let node_id = node.node_id.get() as usize;
        let na = node.actions.len();
        let k_own = if player == 0 { self.k_h } else { self.k_v };
        let k_upd = if updating == 0 { self.k_h } else { self.k_v };

        let strategy = self.regret_matching(node_id, k_own, na);

        let mut action_util: Vec<Vec<f32>> = Vec::with_capacity(na);
        for a in 0..na {
            if player == 0 {
                let new_reach: Vec<f32> = (0..self.k_h).map(|h| reach_h[h] * strategy[h][a]).collect();
                action_util.push(self.cfr(&node.children[a], &new_reach, reach_v, updating, t));
            } else {
                let new_reach: Vec<f32> = (0..self.k_v).map(|v| reach_v[v] * strategy[v][a]).collect();
                action_util.push(self.cfr(&node.children[a], reach_h, &new_reach, updating, t));
            }
        }

        let mut node_util = vec![0.0f32; k_upd];
        if player == updating {
            for b in 0..k_upd {
                for a in 0..na {
                    node_util[b] += strategy[b][a] * action_util[a][b];
                }
            }
        } else {
            for b in 0..k_upd {
                for a in 0..na {
                    node_util[b] += action_util[a][b];
                }
            }
        }

        if player == updating {
            let own_reach: &[f32] = if updating == 0 { reach_h } else { reach_v };
            for b in 0..k_own {
                let mut cur_ev = 0.0f32;
                for a in 0..na { cur_ev += strategy[b][a] * action_util[a][b]; }
                for a in 0..na {
                    let regret = action_util[a][b] - cur_ev;
                    let new_r = self.regrets[node_id][b][a] + own_reach[b] * regret;
                    self.regrets[node_id][b][a] = if new_r > 0.0 { new_r } else { 0.0 };
                    self.strategy_sum[node_id][b][a] += (t as f32) * own_reach[b] * strategy[b][a];
                }
            }
        }

        node_util
    }

    fn terminal_utility(&self, node: &Node, reach_h: &[f32], reach_v: &[f32], updating: i8) -> Vec<f32> {
        let hero_c = self.initial_stacks.0 - node.stacks.0;
        let villain_c = self.initial_stacks.1 - node.stacks.1;
        let pot = node.terminal_pot;

        // Fold payoffs
        let (hero_profit, villain_profit) = match node.terminal_winner {
            Some(0) => (pot - hero_c, -villain_c),
            Some(1) => (-hero_c, pot - villain_c),
            _ => (0.0, 0.0),
        };

        // Bucket-averaged per-combo CF value at terminal:
        //   v0_bucket[h] = (1/bw_h) × Σ_v reach_v[v] × (pair_weight[h][v] / bw_v) × (hero_payoff_hv)
        // where hero_payoff_hv = equity(h,v) × pot − hero_c for showdown, or the fold constant.
        // The 1/bw_h factor is the bucket-uniformity average over combos in bucket h.
        if updating == 0 {
            let mut out = vec![0.0f32; self.k_h];
            for h in 0..self.k_h {
                let bw_h = self.hero_bucketing.bucket_weight[h];
                if bw_h <= 0.0 { continue; }
                let mut acc = 0.0f32;
                for v in 0..self.k_v {
                    let pw = self.equity.pair_weight_at(h, v);
                    let bw_v = self.villain_bucketing.bucket_weight[v];
                    if bw_v <= 0.0 || pw <= 0.0 { continue; }
                    let payoff = if node.terminal_winner.is_some() {
                        hero_profit
                    } else {
                        let eq = self.equity.equity_sum_at(h, v) / pw;
                        eq * pot - hero_c
                    };
                    acc += reach_v[v] * (pw / bw_v) * payoff;
                }
                out[h] = acc / bw_h;
            }
            return out;
        }

        // Villain side
        let mut out = vec![0.0f32; self.k_v];
        for v in 0..self.k_v {
            let bw_v = self.villain_bucketing.bucket_weight[v];
            if bw_v <= 0.0 { continue; }
            let mut acc = 0.0f32;
            for h in 0..self.k_h {
                let pw = self.equity.pair_weight_at(h, v);
                let bw_h = self.hero_bucketing.bucket_weight[h];
                if bw_h <= 0.0 || pw <= 0.0 { continue; }
                let payoff = if node.terminal_winner.is_some() {
                    villain_profit
                } else {
                    let eq_hero = self.equity.equity_sum_at(h, v) / pw;
                    let eq_villain = 1.0 - eq_hero;
                    eq_villain * pot - villain_c
                };
                acc += reach_h[h] * (pw / bw_h) * payoff;
            }
            out[v] = acc / bw_v;
        }
        out
    }

    fn regret_matching(&self, node_id: usize, k: usize, na: usize) -> Vec<Vec<f32>> {
        let mut strategy = vec![vec![0.0f32; na]; k];
        let regrets = &self.regrets[node_id];
        let uniform = 1.0 / (na as f32);
        for b in 0..k {
            let mut s = 0.0f32;
            for a in 0..na { if regrets[b][a] > 0.0 { s += regrets[b][a]; } }
            if s > 0.0 {
                for a in 0..na {
                    let r = regrets[b][a];
                    strategy[b][a] = if r > 0.0 { r / s } else { 0.0 };
                }
            } else {
                for a in 0..na { strategy[b][a] = uniform; }
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
        let strat = &self.strategy_sum[nid];
        let probs: Vec<Vec<f32>> = strat.iter().map(|row| {
            let total: f32 = row.iter().sum();
            if total > 0.0 { row.iter().map(|v| v / total).collect() }
            else { vec![1.0 / (na as f32); na] }
        }).collect();
        let labels: Vec<String> = root.actions.iter().map(|a| a.label()).collect();
        (probs, labels)
    }

    /// Return strategies for every decision node in the tree.
    /// Each entry: (path, player_to_act, action_labels, strategy_[bucket][action]).
    pub fn all_node_strategies(&self, root: &Node) -> Vec<(String, i8, Vec<String>, Vec<Vec<f32>>)> {
        let mut out = Vec::new();
        self.walk_for_strategies(root, "".to_string(), &mut out);
        out
    }

    fn walk_for_strategies(
        &self,
        node: &Node,
        path: String,
        out: &mut Vec<(String, i8, Vec<String>, Vec<Vec<f32>>)>,
    ) {
        if node.is_terminal || node.is_chance {
            return;
        }
        let nid = node.node_id.get() as usize;
        let na = node.actions.len();
        let strat = &self.strategy_sum[nid];
        let probs: Vec<Vec<f32>> = strat.iter().map(|row| {
            let total: f32 = row.iter().sum();
            if total > 0.0 { row.iter().map(|v| v / total).collect() }
            else { vec![1.0 / (na as f32); na] }
        }).collect();
        let labels: Vec<String> = node.actions.iter().map(|a| a.label()).collect();
        out.push((path.clone(), node.player_to_act, labels.clone(), probs));

        for (a, child) in node.actions.iter().zip(node.children.iter()) {
            let next_path = if path.is_empty() {
                a.label()
            } else {
                format!("{}/{}", path, a.label())
            };
            self.walk_for_strategies(child, next_path, out);
        }
    }
}

fn walk_decision_nodes(root: &mut Node, f: &mut impl FnMut(&mut Node)) {
    if !root.is_terminal && !root.is_chance {
        f(root);
    }
    for child in root.children.iter_mut() {
        walk_decision_nodes(child, f);
    }
}

#[cfg(test)]
mod tests {
    use super::super::bucket_equity::compute_bucket_equity_river;
    use super::super::bucketing::{bucket_by_ehs, river_ehs};
    use super::super::cards::{card_from_str, NUM_COMBOS};
    use super::super::cfr::Solver;
    use super::super::range_parser::parse_range;
    use super::super::showdown::compute_showdown_table;
    use super::super::tree::build_river_tree;
    use super::*;

    fn board(cards: &[&str]) -> [u8; 5] {
        let mut out = [0u8; 5];
        for (i, c) in cards.iter().enumerate() { out[i] = card_from_str(c).unwrap(); }
        out
    }

    #[test]
    fn bucketed_river_converges() {
        let b = board(&["Ad", "Kh", "7s", "3c", "2d"]);
        let range = vec![1.0f32; NUM_COMBOS];
        let ehs = river_ehs(&b);
        let bucketing = bucket_by_ehs(&ehs, &range, &b, 16);
        let equity = compute_bucket_equity_river(&b, &bucketing, &bucketing);
        let mut root = build_river_tree(100.0, (200.0, 200.0), 0, 2);
        let result = solve_river_bucketed(
            &mut root, &bucketing, &bucketing, &equity, (200.0, 200.0), 500,
        );
        // Symmetric ranges and identical bucketing → hero_value should be ~pot/2 = 50
        // (since hero starts first-to-act and eventually check-down gives pot/2).
        // We're lenient because bucketing introduces approximation.
        assert!(
            result.hero_value > 20.0 && result.hero_value < 80.0,
            "hero_value = {} (expected ~50 for symmetric spot)",
            result.hero_value,
        );
    }

    #[test]
    fn bucketed_matches_unbucketed_within_tolerance() {
        // Small spot where both approaches should agree within a few %
        let b = board(&["Ad", "Kh", "7s", "3c", "2d"]);
        let hero = parse_range("AhAc, KsKc, QsJs").unwrap();
        let villain = parse_range("JcJd, TcTd, AcQc").unwrap();

        // Unbucketed (reference)
        let table = compute_showdown_table(b, &hero, &villain);
        let mut root1 = build_river_tree(100.0, (150.0, 150.0), 0, 2);
        let mut solver = Solver::new(&mut root1, &table, (150.0, 150.0), None, None);
        solver.state.train(&root1, 500);
        let ref_value = solver.state.last_root_value();

        // Bucketed
        let ehs = river_ehs(&b);
        let hero_bucketing = bucket_by_ehs(&ehs, &hero, &b, 8);
        let villain_bucketing = bucket_by_ehs(&ehs, &villain, &b, 8);
        let equity = compute_bucket_equity_river(&b, &hero_bucketing, &villain_bucketing);
        let mut root2 = build_river_tree(100.0, (150.0, 150.0), 0, 2);
        let bucketed = solve_river_bucketed(
            &mut root2, &hero_bucketing, &villain_bucketing, &equity, (150.0, 150.0), 500,
        );

        // With K=8 and only 3 combos per side, bucketing is essentially identity
        // and results should match closely.
        let diff = (bucketed.hero_value - ref_value).abs();
        assert!(
            diff < 5.0,
            "bucketed={:.3} vs ref={:.3}, diff={:.3}",
            bucketed.hero_value, ref_value, diff,
        );
    }
}
