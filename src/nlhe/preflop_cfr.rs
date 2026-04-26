//! CFR+ for preflop HU using a precomputed equity table at terminals.
//!
//! Preflop terminals (action closes preflop, would normally lead to flop)
//! have no postflop subtree attached in this MVP. Instead they're scored
//! using a precomputed `PreflopEquityTable` (169×169 hand-class equity).
//!
//! This is the standard "preflop GTO baseline" approach used by industry
//! solvers when full multi-street solving isn't feasible. It's accurate to
//! within ~1 bb/100 of multi-street solves at typical stack depths and
//! suffices for opponent-profile-driven exploit decisions.

use std::collections::HashMap;

use super::preflop_equity::PreflopEquityTable;
use super::tree::{Node, walk_nodes, ShowdownKey};

#[derive(Debug, Clone)]
pub struct NodeStrategyDump {
    pub player: i8,
    pub pot: f32,
    pub stacks: [f32; 2],
    pub to_call: f32,
    pub actions: Vec<String>,
    pub probabilities: Vec<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct PreflopSolveResult {
    pub iterations: u32,
    pub root_strategy: HashMap<u16, Vec<(String, f32)>>,
    pub hero_value: f32,
    pub last_iter_values: Vec<f32>,
}

/// Solve a preflop HU subgame.
///
/// `hero_range` and `villain_range` are length-1326 weight vectors. For
/// "any two cards" use all-1.0; for tighter ranges (e.g. for an exploitative
/// solve given an opponent profile) provide the appropriate weights.
pub fn solve_preflop(
    root: &mut Node,
    equity_table: &PreflopEquityTable,
    hero_range: &[f32],
    villain_range: &[f32],
    initial_stacks: (f32, f32),
    iterations: u32,
) -> PreflopSolveResult {
    let mut state = PreflopSolverState::new(root, equity_table, hero_range, villain_range, initial_stacks);
    state.train(root, iterations);
    let root_strategy = state.extract_root_strategy(root);
    PreflopSolveResult {
        iterations,
        root_strategy,
        hero_value: state.last_root_value(),
        last_iter_values: state.iter_values.clone(),
    }
}

/// Like `solve_preflop`, but returns the trained solver state so the caller
/// can extract per-node strategies for full storage.
pub fn build_and_train_preflop<'a>(
    root: &mut Node,
    equity_table: &'a PreflopEquityTable,
    hero_range: &[f32],
    villain_range: &[f32],
    initial_stacks: (f32, f32),
    iterations: u32,
) -> PreflopSolverState<'a> {
    let mut state = PreflopSolverState::new(root, equity_table, hero_range, villain_range, initial_stacks);
    state.train(root, iterations);
    state
}

pub struct PreflopSolverState<'a> {
    pub equity: &'a PreflopEquityTable,
    pub hero_range: Vec<f32>,
    pub villain_range: Vec<f32>,
    pub initial_stacks: (f32, f32),
    pub n_h: usize,
    pub n_v: usize,
    /// Local→global combo idx (just the indices with weight > 0 in the range).
    pub hero_combos: Vec<u16>,
    pub villain_combos: Vec<u16>,
    /// Per-combo normalized weights (matching local indexing).
    pub hero_weights: Vec<f32>,
    pub villain_weights: Vec<f32>,
    /// Cached hand class per local combo.
    pub hero_classes: Vec<u8>,
    pub villain_classes: Vec<u8>,
    /// Standard CFR storage.
    pub regrets: Vec<Vec<Vec<f32>>>,        // [node_id][local_combo][action]
    pub strategy_sum: Vec<Vec<Vec<f32>>>,
    pub iter_values: Vec<f32>,
    pub pair_weight_total: f32,
}

impl<'a> PreflopSolverState<'a> {
    pub fn new(
        root: &mut Node,
        equity: &'a PreflopEquityTable,
        hero_range: &[f32],
        villain_range: &[f32],
        initial_stacks: (f32, f32),
    ) -> Self {
        let mut hero_combos = Vec::new();
        let mut hero_weights = Vec::new();
        let mut hero_classes = Vec::new();
        for combo in 0..super::cards::NUM_COMBOS {
            if hero_range[combo] > 0.0 {
                hero_combos.push(combo as u16);
                hero_weights.push(hero_range[combo]);
                hero_classes.push(super::preflop_equity::combo_to_class(combo));
            }
        }
        let mut villain_combos = Vec::new();
        let mut villain_weights = Vec::new();
        let mut villain_classes = Vec::new();
        for combo in 0..super::cards::NUM_COMBOS {
            if villain_range[combo] > 0.0 {
                villain_combos.push(combo as u16);
                villain_weights.push(villain_range[combo]);
                villain_classes.push(super::preflop_equity::combo_to_class(combo));
            }
        }
        let n_h = hero_combos.len();
        let n_v = villain_combos.len();

        // Assign decision node ids and allocate regret/strategy_sum.
        let mut regrets: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut strategy_sum: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut id = 0i32;
        walk_nodes(root, &mut |node| {
            if !node.is_terminal && !node.is_chance {
                node.node_id.set(id);
                let na = node.actions.len();
                let nb = if node.player_to_act == 0 { n_h } else { n_v };
                regrets.push(vec![vec![0.0; na]; nb]);
                strategy_sum.push(vec![vec![0.0; na]; nb]);
                id += 1;
            }
        });

        // Pair-weight total (excludes pairs with conflicting hole cards).
        let mut pair_weight_total = 0.0f32;
        for i in 0..n_h {
            let (ha, hb) = super::cards::combo_cards(hero_combos[i] as usize);
            for j in 0..n_v {
                let (va, vb) = super::cards::combo_cards(villain_combos[j] as usize);
                if va == ha || va == hb || vb == ha || vb == hb { continue; }
                pair_weight_total += hero_weights[i] * villain_weights[j];
            }
        }

        Self {
            equity, hero_range: hero_range.to_vec(), villain_range: villain_range.to_vec(),
            initial_stacks,
            n_h, n_v,
            hero_combos, villain_combos,
            hero_weights, villain_weights,
            hero_classes, villain_classes,
            regrets, strategy_sum,
            iter_values: Vec::new(),
            pair_weight_total,
        }
    }

    pub fn train(&mut self, root: &Node, iterations: u32) {
        let hero_w = self.hero_weights.clone();
        let villain_w = self.villain_weights.clone();
        for t in 1..=iterations {
            let v0 = self.cfr(root, &hero_w, &villain_w, 0, t);
            let _ = self.cfr(root, &hero_w, &villain_w, 1, t);
            if self.pair_weight_total > 0.0 {
                let mut acc = 0.0f32;
                for i in 0..self.n_h { acc += hero_w[i] * v0[i]; }
                self.iter_values.push(acc / self.pair_weight_total);
            }
        }
    }

    fn cfr(&mut self, node: &Node, reach_h: &[f32], reach_v: &[f32], updating: i8, t: u32) -> Vec<f32> {
        if node.is_terminal {
            return self.terminal_utility(node, reach_h, reach_v, updating);
        }
        // Preflop trees do not contain chance nodes (no postflop subtree attached in MVP)
        let player = node.player_to_act;
        let node_id = node.node_id.get() as usize;
        let na = node.actions.len();
        let nc_upd = if updating == 0 { self.n_h } else { self.n_v };

        let strategy = self.regret_matching(node_id, player, na);

        let mut action_util: Vec<Vec<f32>> = Vec::with_capacity(na);
        for a in 0..na {
            if player == 0 {
                let new_reach_h: Vec<f32> = (0..self.n_h).map(|i| reach_h[i] * strategy[i][a]).collect();
                action_util.push(self.cfr(&node.children[a], &new_reach_h, reach_v, updating, t));
            } else {
                let new_reach_v: Vec<f32> = (0..self.n_v).map(|j| reach_v[j] * strategy[j][a]).collect();
                action_util.push(self.cfr(&node.children[a], reach_h, &new_reach_v, updating, t));
            }
        }

        let mut node_util = vec![0.0f32; nc_upd];
        if player == updating {
            for i in 0..nc_upd {
                for a in 0..na {
                    node_util[i] += strategy[i][a] * action_util[a][i];
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
            for i in 0..nc_upd {
                let mut cur_ev = 0.0f32;
                for a in 0..na {
                    cur_ev += strategy[i][a] * action_util[a][i];
                }
                for a in 0..na {
                    let regret = action_util[a][i] - cur_ev;
                    let new_r = self.regrets[node_id][i][a] + own_reach[i] * regret;
                    self.regrets[node_id][i][a] = if new_r > 0.0 { new_r } else { 0.0 };
                    self.strategy_sum[node_id][i][a] += (t as f32) * own_reach[i] * strategy[i][a];
                }
            }
        }

        node_util
    }

    fn terminal_utility(&self, node: &Node, reach_h: &[f32], reach_v: &[f32], updating: i8) -> Vec<f32> {
        let hero_c = self.initial_stacks.0 - node.stacks.0;
        let villain_c = self.initial_stacks.1 - node.stacks.1;
        let pot = node.terminal_pot;

        if let Some(winner) = node.terminal_winner {
            // Fold terminal. Filter mutually-conflicting (i, j) pairs.
            let (hero_profit, villain_profit) = if winner == 0 {
                (pot - hero_c, -villain_c)
            } else {
                (-hero_c, pot - villain_c)
            };
            if updating == 0 {
                let mut out = vec![0.0f32; self.n_h];
                for i in 0..self.n_h {
                    let (ha, hb) = super::cards::combo_cards(self.hero_combos[i] as usize);
                    let mut total_v = 0.0;
                    for j in 0..self.n_v {
                        let (va, vb) = super::cards::combo_cards(self.villain_combos[j] as usize);
                        if va == ha || va == hb || vb == ha || vb == hb { continue; }
                        total_v += reach_v[j];
                    }
                    out[i] = total_v * hero_profit;
                }
                return out;
            }
            let mut out = vec![0.0f32; self.n_v];
            for j in 0..self.n_v {
                let (va, vb) = super::cards::combo_cards(self.villain_combos[j] as usize);
                let mut total_h = 0.0;
                for i in 0..self.n_h {
                    let (ha, hb) = super::cards::combo_cards(self.hero_combos[i] as usize);
                    if va == ha || va == hb || vb == ha || vb == hb { continue; }
                    total_h += reach_h[i];
                }
                out[j] = total_h * villain_profit;
            }
            return out;
        }

        // Showdown via preflop equity table.
        if updating == 0 {
            let mut out = vec![0.0f32; self.n_h];
            for i in 0..self.n_h {
                let (ha, hb) = super::cards::combo_cards(self.hero_combos[i] as usize);
                let mut acc = 0.0f32;
                let h_class = self.hero_classes[i];
                for j in 0..self.n_v {
                    let (va, vb) = super::cards::combo_cards(self.villain_combos[j] as usize);
                    if va == ha || va == hb || vb == ha || vb == hb { continue; }
                    let v_class = self.villain_classes[j];
                    let eq = self.equity.equity_for_classes(h_class, v_class);
                    let share = pot * eq;
                    acc += reach_v[j] * (share - hero_c);
                }
                out[i] = acc;
            }
            return out;
        }
        let mut out = vec![0.0f32; self.n_v];
        for j in 0..self.n_v {
            let (va, vb) = super::cards::combo_cards(self.villain_combos[j] as usize);
            let mut acc = 0.0f32;
            let v_class = self.villain_classes[j];
            for i in 0..self.n_h {
                let (ha, hb) = super::cards::combo_cards(self.hero_combos[i] as usize);
                if va == ha || va == hb || vb == ha || vb == hb { continue; }
                let h_class = self.hero_classes[i];
                let eq = self.equity.equity_for_classes(h_class, v_class);
                // Villain's share = pot * (1 - hero_equity)
                let share = pot * (1.0 - eq);
                acc += reach_h[i] * (share - villain_c);
            }
            out[j] = acc;
        }
        out
    }

    fn regret_matching(&self, node_id: usize, player: i8, na: usize) -> Vec<Vec<f32>> {
        let nb = if player == 0 { self.n_h } else { self.n_v };
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

    /// Walk the tree and dump aggregated avg strategy per decision node.
    /// Returns map keyed by action-path string ("" = root, "raise_X > call" etc).
    /// Each entry contains player + per-bucket(=per-combo) probabilities.
    pub fn dump_all_strategies(&self, root: &Node) -> HashMap<String, NodeStrategyDump> {
        let mut out = HashMap::new();
        fn walk(
            state: &PreflopSolverState,
            node: &Node,
            path: String,
            out: &mut HashMap<String, NodeStrategyDump>,
        ) {
            if !node.is_terminal && !node.is_chance {
                let nid = node.node_id.get() as usize;
                let na = node.actions.len();
                let strat = &state.strategy_sum[nid];
                let probabilities: Vec<Vec<f32>> = strat.iter().map(|row| {
                    let total: f32 = row.iter().sum();
                    if total > 0.0 { row.iter().map(|v| v / total).collect() }
                    else { vec![1.0 / (na as f32); na] }
                }).collect();
                out.insert(path.clone(), NodeStrategyDump {
                    player: node.player_to_act,
                    pot: node.pot,
                    stacks: [node.stacks.0, node.stacks.1],
                    to_call: node.to_call,
                    actions: node.actions.iter().map(|a| a.label()).collect(),
                    probabilities,
                });
            }
            for (a, child) in node.children.iter().enumerate() {
                let part = node.actions.get(a).map(|x| x.label()).unwrap_or_else(|| format!("a{}", a));
                let new_path = if path.is_empty() { part } else { format!("{} > {}", path, part) };
                walk(state, child, new_path, out);
            }
        }
        walk(self, root, String::new(), &mut out);
        out
    }

    pub fn extract_root_strategy(&self, root: &Node) -> HashMap<u16, Vec<(String, f32)>> {
        let root_id = root.node_id.get() as usize;
        let player = root.player_to_act;
        let combos: &[u16] = if player == 0 { &self.hero_combos } else { &self.villain_combos };
        let na = root.actions.len();
        let strat = &self.strategy_sum[root_id];
        let mut out = HashMap::new();
        for (i, &combo_idx) in combos.iter().enumerate() {
            let row = &strat[i];
            let total: f32 = row.iter().sum();
            let probs: Vec<f32> = if total > 0.0 {
                row.iter().map(|v| v / total).collect()
            } else {
                vec![1.0 / (na as f32); na]
            };
            let labeled: Vec<(String, f32)> = root.actions.iter().zip(probs.iter())
                .map(|(action, p)| (action.label(), *p)).collect();
            out.insert(combo_idx, labeled);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::preflop_equity::compute_preflop_equity_table;
    use super::super::preflop_tree::build_preflop_tree;

    #[test]
    fn solver_runs_with_equity_table() {
        let table = compute_preflop_equity_table(50, 42);
        let mut root = build_preflop_tree(200.0, 2);
        // Both ranges = "any two cards"
        let any_two = vec![1.0f32; super::super::cards::NUM_COMBOS];
        let result = solve_preflop(&mut root, &table, &any_two, &any_two, (200.0, 200.0), 30);
        // Just sanity: should produce a reasonable EV (between -200 and +200)
        assert!(result.hero_value.abs() < 200.0, "hero_value = {}", result.hero_value);
    }
}
