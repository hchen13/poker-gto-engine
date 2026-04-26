//! Bucketed CFR+ over a flattened tree (pointer-chase-free).
//!
//! Same semantics as `bucket_cfr_multi::MultiBucketSolverState`, but the
//! tree is a contiguous Vec<FlatNode> with u32 child indices. Better cache
//! locality in the hot CFR loop.

use std::collections::HashMap;

use super::bucket_cfr_multi::BucketEquityStore;
use super::bucket_equity::BucketEquityTable;
use super::bucketing::Bucketing;
use super::flat_tree::{flatten, FlatTree};
use super::simd_util::{axpy, dot, fma_slice, mul_into, cfr_plus_update};
use super::tree::{Node, ShowdownKey};

/// Strategy at a single decision node in the tree.
/// Used to serialize the full flop+turn strategy tree.
#[derive(Debug, Clone)]
pub struct NodeStrategy {
    pub node_id: usize,
    /// 0 = OOP (hero/BB 3-bettor), 1 = IP (villain/SB caller)
    pub player: i8,
    /// "flop" or "turn" (river skipped — on-demand)
    pub street: String,
    /// Action path to reach this node, e.g. "" (root), "check", "bet_10.00/call"
    pub path: String,
    /// The turn card (card index 0..52) if street=="turn" or "river"
    pub turn_card: Option<u8>,
    /// Strategy matrix: strategy[b][a] = prob of action a for bucket b (normalized avg)
    pub strategy: Vec<Vec<f32>>,
    /// Action labels parallel to strategy columns
    pub action_labels: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BucketSolveResult {
    pub iterations: u32,
    pub root_strategy: Vec<Vec<f32>>,
    pub action_labels: Vec<String>,
    pub hero_value: f32,
    pub last_iter_values: Vec<f32>,
    /// Full flop + turn strategy for all decision nodes (both players)
    pub all_nodes: Vec<NodeStrategy>,
}

pub fn solve_multi_bucketed_flat(
    root: &mut Node,
    hero_bucketing: &Bucketing,
    villain_bucketing: &Bucketing,
    equity_store: &BucketEquityStore,
    initial_stacks: (f32, f32),
    iterations: u32,
) -> BucketSolveResult {
    assign_ids_and_equity(root, equity_store);
    let flat = flatten(root);
    let mut state = FlatSolverState::new(&flat, hero_bucketing, villain_bucketing, equity_store, initial_stacks);
    state.train(iterations);
    let (root_strategy, action_labels) = state.root_strategy(&flat);
    let all_nodes = state.extract_all_node_strategies(&flat);
    BucketSolveResult {
        iterations,
        root_strategy,
        action_labels,
        hero_value: state.last_root_value(),
        last_iter_values: state.iter_values.clone(),
        all_nodes,
    }
}

fn assign_ids_and_equity(root: &mut Node, equity_store: &BucketEquityStore) {
    // Build key → idx map
    let mut key_to_idx: HashMap<ShowdownKey, usize> = HashMap::new();
    for (idx, k) in equity_store.tables.keys().enumerate() {
        key_to_idx.insert(*k, idx);
    }
    // Assign node_ids to decision nodes in DFS pre-order; set equity_idx on terminals
    let mut next_id = 0i32;
    fn walk(node: &mut Node, next_id: &mut i32, key_to_idx: &HashMap<ShowdownKey, usize>) {
        if node.is_terminal {
            let idx = if node.terminal_winner.is_none() {
                key_to_idx.get(&node.showdown_key).copied().unwrap_or(usize::MAX)
            } else {
                usize::MAX
            };
            node.equity_idx.set(idx as i32);
        } else if !node.is_chance {
            node.node_id.set(*next_id);
            *next_id += 1;
        }
        for child in node.children.iter_mut() {
            walk(child, next_id, key_to_idx);
        }
    }
    walk(root, &mut next_id, &key_to_idx);
}

pub struct FlatSolverState<'a> {
    pub flat: &'a FlatTree,
    pub hero_bucketing: &'a Bucketing,
    pub villain_bucketing: &'a Bucketing,
    pub equity_tables: Vec<BucketEquityTable>,
    pub initial_stacks: (f32, f32),
    pub k_h: usize,
    pub k_v: usize,
    /// Flat: regrets[node_id] is Vec<f32> of size K×na indexed [bucket*na + action]
    pub regrets: Vec<Vec<f32>>,
    pub strategy_sum: Vec<Vec<f32>>,
    pub iter_values: Vec<f32>,
    pub pair_weight_avg: f32,
}

impl<'a> FlatSolverState<'a> {
    pub fn new(
        flat: &'a FlatTree,
        hero_bucketing: &'a Bucketing,
        villain_bucketing: &'a Bucketing,
        equity_store: &'a BucketEquityStore,
        initial_stacks: (f32, f32),
    ) -> Self {
        let k_h = hero_bucketing.k;
        let k_v = villain_bucketing.k;

        // Count decision nodes (in flat node_id order — already set by assign_ids)
        let mut max_node_id = -1i32;
        for n in flat.nodes.iter() {
            if n.node_id >= 0 && n.node_id > max_node_id { max_node_id = n.node_id; }
        }
        let n_decision = (max_node_id + 1) as usize;

        let mut regrets: Vec<Vec<f32>> = Vec::with_capacity(n_decision);
        let mut strategy_sum: Vec<Vec<f32>> = Vec::with_capacity(n_decision);
        regrets.resize(n_decision, Vec::new());
        strategy_sum.resize(n_decision, Vec::new());

        for n in flat.nodes.iter() {
            if n.is_terminal || n.is_chance { continue; }
            let id = n.node_id as usize;
            let na = n.action_count as usize;
            let k = if n.player_to_act == 0 { k_h } else { k_v };
            regrets[id] = vec![0.0f32; k * na];
            strategy_sum[id] = vec![0.0f32; k * na];
        }

        let mut total = 0.0f32;
        let n_tables = equity_store.tables.len().max(1);
        for t in equity_store.tables.values() {
            total += t.pair_weight.iter().sum::<f32>();
        }
        let pair_weight_avg = total / n_tables as f32;

        let equity_tables: Vec<BucketEquityTable> = equity_store.tables.values().cloned().collect();

        Self {
            flat, hero_bucketing, villain_bucketing, equity_tables, initial_stacks,
            k_h, k_v, regrets, strategy_sum, iter_values: Vec::new(),
            pair_weight_avg,
        }
    }

    pub fn train(&mut self, iterations: u32) {
        let reach_h = self.hero_bucketing.bucket_weight.clone();
        let reach_v = self.villain_bucketing.bucket_weight.clone();
        let mut v0 = vec![0.0f32; self.k_h];
        let mut v1 = vec![0.0f32; self.k_v];
        for t in 1..=iterations {
            for x in v0.iter_mut() { *x = 0.0; }
            for x in v1.iter_mut() { *x = 0.0; }
            self.cfr(0, &reach_h, &reach_v, 0, t, &mut v0);
            self.cfr(0, &reach_h, &reach_v, 1, t, &mut v1);
            if self.pair_weight_avg > 0.0 {
                let mut acc = 0.0f32;
                for h in 0..self.k_h { acc += reach_h[h] * v0[h]; }
                self.iter_values.push(acc / self.pair_weight_avg);
            }
        }
    }

    // Layout convention (action-major): all per-node buffers of size K*na are
    // indexed as `[a * k + b]` — each action's per-bucket vector is a contiguous
    // length-k slice. This is what f32x4 SIMD wants to chew through.
    fn cfr(&mut self, node_idx: u32, reach_h: &[f32], reach_v: &[f32], updating: i8, t: u32, out: &mut [f32]) {
        const K_MAX: usize = 64;
        const NA_MAX: usize = 12;
        const KNA_MAX: usize = K_MAX * NA_MAX;

        let n_clone = self.flat.nodes[node_idx as usize].clone();
        let n = &n_clone;
        if n.is_terminal {
            self.terminal(n, reach_h, reach_v, updating, out);
            return;
        }
        if n.is_chance {
            let k_upd = if updating == 0 { self.k_h } else { self.k_v };
            assert!(k_upd <= K_MAX);
            for x in out[..k_upd].iter_mut() { *x = 0.0; }
            let c_off = n.chance_offset as usize;
            let c_cnt = n.chance_count as usize;
            let inv_denom = 1.0 / (c_cnt as f32);
            let mut child_buf = [0.0f32; K_MAX];
            for i in 0..c_cnt {
                let c = self.flat.chance_children[c_off + i];
                for x in child_buf[..k_upd].iter_mut() { *x = 0.0; }
                self.cfr(c, reach_h, reach_v, updating, t, &mut child_buf[..k_upd]);
                axpy(&mut out[..k_upd], inv_denom, &child_buf[..k_upd]);
            }
            return;
        }

        let player = n.player_to_act;
        let nid = n.node_id as usize;
        let na = n.action_count as usize;
        let k_own = if player == 0 { self.k_h } else { self.k_v };
        let k_upd = if updating == 0 { self.k_h } else { self.k_v };
        assert!(na <= NA_MAX && k_own <= K_MAX && k_upd <= K_MAX);

        // Stack-alloc strategy buffer (action-major: strategy[a*k_own + b])
        let mut strategy = [0.0f32; KNA_MAX];
        self.regret_matching_into(nid, k_own, na, &mut strategy);

        // Stack-alloc scratch reach and action_util (action-major: action_util[a*k_upd + b])
        let mut scratch_reach = [0.0f32; K_MAX];
        let mut action_util = [0.0f32; NA_MAX * K_MAX];
        let a_off = n.action_offset as usize;
        for a in 0..na {
            let c = self.flat.action_children[a_off + a];
            let slot = &mut action_util[a * k_upd..(a + 1) * k_upd];
            let strat_row = &strategy[a * k_own..(a + 1) * k_own];
            if player == 0 {
                // scratch_reach[h] = reach_h[h] * strategy[a,h] — SIMD mul
                mul_into(&mut scratch_reach[..self.k_h], reach_h, strat_row);
                self.cfr(c, &scratch_reach[..self.k_h], reach_v, updating, t, slot);
            } else {
                mul_into(&mut scratch_reach[..self.k_v], reach_v, strat_row);
                self.cfr(c, reach_h, &scratch_reach[..self.k_v], updating, t, slot);
            }
        }

        // Node util: out[b] = sum_a strategy[a,b] * action_util[a,b] (when player==updating)
        // or sum_a action_util[a,b] (otherwise). `for a: fma/axpy` — contiguous length-k ops.
        for x in out[..k_upd].iter_mut() { *x = 0.0; }
        if player == updating {
            // k_own == k_upd when player == updating
            for a in 0..na {
                let util_row = &action_util[a * k_upd..(a + 1) * k_upd];
                let strat_row = &strategy[a * k_own..(a + 1) * k_own];
                fma_slice(&mut out[..k_upd], strat_row, util_row);
            }
        } else {
            for a in 0..na {
                let util_row = &action_util[a * k_upd..(a + 1) * k_upd];
                axpy(&mut out[..k_upd], 1.0, util_row);
            }
        }

        if player == updating {
            let own_reach: &[f32] = if updating == 0 { reach_h } else { reach_v };
            let t_f = t as f32;
            let regrets = &mut self.regrets[nid];
            let strat_sum = &mut self.strategy_sum[nid];
            for a in 0..na {
                let rbase = a * k_own;
                // k_own == k_upd; out holds cur_ev
                cfr_plus_update(
                    &mut regrets[rbase..rbase + k_own],
                    &mut strat_sum[rbase..rbase + k_own],
                    own_reach,
                    &action_util[a * k_own..(a + 1) * k_own],
                    &out[..k_own],
                    &strategy[rbase..rbase + k_own],
                    t_f,
                );
            }
        }
    }

    fn terminal(&self, n: &super::flat_tree::FlatNode, reach_h: &[f32], reach_v: &[f32], updating: i8, out: &mut [f32]) {
        let hero_c = self.initial_stacks.0 - n.stacks[0];
        let villain_c = self.initial_stacks.1 - n.stacks[1];
        let pot = n.terminal_pot;

        let terminal_winner = if n.terminal_winner < 0 { None } else { Some(n.terminal_winner as u8) };
        let (hero_profit, villain_profit) = match terminal_winner {
            Some(0) => (pot - hero_c, -villain_c),
            Some(1) => (-hero_c, pot - villain_c),
            _ => (0.0, 0.0),
        };

        let eidx = n.equity_idx;
        let equity_table: &BucketEquityTable = if eidx >= 0 && (eidx as usize) < self.equity_tables.len() {
            &self.equity_tables[eidx as usize]
        } else if terminal_winner.is_some() && !self.equity_tables.is_empty() {
            &self.equity_tables[0]
        } else {
            for x in out.iter_mut() { *x = 0.0; }
            return;
        };

        if updating == 0 {
            // out[h] = (1/bw_h) * sum_v reach_v[v] * pw_over_bwv[h,v] * payoff(h,v)
            let bw_h = &self.hero_bucketing.bucket_weight;
            for h in 0..self.k_h {
                if bw_h[h] <= 0.0 { out[h] = 0.0; continue; }
                let pw_row = &equity_table.pw_over_bwv[h * self.k_v..(h + 1) * self.k_v];
                let acc = if let Some(_w) = terminal_winner {
                    // fold/bet-win: payoff is scalar hero_profit, so sum_v reach_v * pw_over_bwv
                    hero_profit * dot(reach_v, pw_row)
                } else {
                    // showdown: payoff[v] = equity[h,v]*pot - hero_c = equity_row[v]*pot - hero_c
                    // sum_v reach_v[v] * pw_over_bwv[h,v] * (eq[h,v]*pot - hero_c)
                    // = pot * dot(reach_v * pw_row, equity_row) - hero_c * dot(reach_v, pw_row)
                    let eq_row = &equity_table.equity[h * self.k_v..(h + 1) * self.k_v];
                    // scratch: reach_v_pw[v] = reach_v[v] * pw_row[v]
                    const K_MAX: usize = 64;
                    let mut tmp = [0.0f32; K_MAX];
                    let kv = self.k_v;
                    mul_into(&mut tmp[..kv], reach_v, pw_row);
                    let pw_sum = tmp[..kv].iter().sum::<f32>();
                    pot * dot(&tmp[..kv], eq_row) - hero_c * pw_sum
                };
                out[h] = acc / bw_h[h];
            }
            return;
        }

        // updating == 1 (villain)
        // out[v] = (1/bw_v) * sum_h reach_h[h] * pw_over_bwh[v,h] * payoff(h,v)
        let bw_v = &self.villain_bucketing.bucket_weight;
        for v in 0..self.k_v {
            if bw_v[v] <= 0.0 { out[v] = 0.0; continue; }
            let pw_row = &equity_table.pw_over_bwh[v * self.k_h..(v + 1) * self.k_h];
            let acc = if let Some(_w) = terminal_winner {
                villain_profit * dot(reach_h, pw_row)
            } else {
                // payoff_villain[h] = (1-equity[h,v])*pot - villain_c
                // Need eq_villain row: equity is stored hero-major, so eq[h,v] = equity[h*k_v+v]
                // sum_h reach_h[h] * pw_over_bwh[v,h] * ((1-eq[h,v])*pot - villain_c)
                // = pot*(dot(pw_row,reach_h) - dot(pw_row * reach_h, eq_col)) - villain_c*dot(pw_row,reach_h)
                const K_MAX: usize = 64;
                let mut eq_col = [0.0f32; K_MAX];
                let kh = self.k_h;
                for h in 0..kh { eq_col[h] = equity_table.equity[h * self.k_v + v]; }
                let mut tmp = [0.0f32; K_MAX];
                mul_into(&mut tmp[..kh], reach_h, pw_row);
                let pw_sum = tmp[..kh].iter().sum::<f32>();
                let eq_dot = dot(&tmp[..kh], &eq_col[..kh]);
                (pot - villain_c) * pw_sum - pot * eq_dot
            };
            out[v] = acc / bw_v[v];
        }
    }

    // Action-major layout: regrets[a*k + b], strategy[a*k + b].
    // For each bucket b, sum positive regrets across actions (strided), normalize per-bucket.
    #[inline]
    fn regret_matching_into(&self, nid: usize, k: usize, na: usize, strategy: &mut [f32]) {
        let regrets = &self.regrets[nid];
        let uniform = 1.0 / (na as f32);
        for b in 0..k {
            let mut s = 0.0f32;
            for a in 0..na {
                let r = regrets[a * k + b];
                if r > 0.0 { s += r; }
            }
            if s > 0.0 {
                for a in 0..na {
                    let r = regrets[a * k + b];
                    strategy[a * k + b] = if r > 0.0 { r / s } else { 0.0 };
                }
            } else {
                for a in 0..na { strategy[a * k + b] = uniform; }
            }
        }
    }

    pub fn last_root_value(&self) -> f32 {
        *self.iter_values.last().unwrap_or(&0.0)
    }

    /// Extract normalized strategies for all flop + turn decision nodes (both players).
    /// River is skipped — solve on-demand when needed.
    /// `chance_depth` tracks how many chance nodes (turn/river) we've passed:
    ///   0 = flop decisions, 1 = turn decisions, 2+ = river (skipped).
    pub fn extract_all_node_strategies(&self, flat: &FlatTree) -> Vec<NodeStrategy> {
        let mut out = Vec::new();
        self.walk_node(flat, 0, 0, "", None, &mut out);
        out
    }

    fn walk_node(
        &self,
        flat: &FlatTree,
        node_idx: u32,
        chance_depth: u8,
        path: &str,
        turn_card: Option<u8>,
        out: &mut Vec<NodeStrategy>,
    ) {
        let n = &flat.nodes[node_idx as usize];
        if n.is_terminal { return; }

        if n.is_chance {
            if chance_depth >= 1 { return; } // entering river territory — skip
            let c_off = n.chance_offset as usize;
            let c_cnt = n.chance_count as usize;
            for i in 0..c_cnt {
                let card = flat.chance_cards[c_off + i];
                let child_idx = flat.chance_children[c_off + i];
                self.walk_node(flat, child_idx, chance_depth + 1, path, Some(card), out);
            }
            return;
        }

        // Decision node — extract normalized average strategy
        let nid = n.node_id as usize;
        let na = n.action_count as usize;
        let player = n.player_to_act;
        let k = if player == 0 { self.k_h } else { self.k_v };
        let street = if chance_depth == 0 { "flop" } else { "turn" };

        let strat_sum = &self.strategy_sum[nid];
        let strategy: Vec<Vec<f32>> = (0..k).map(|b| {
            let total: f32 = (0..na).map(|a| strat_sum[a * k + b]).sum();
            (0..na).map(|a| {
                if total > 0.0 { strat_sum[a * k + b] / total } else { 1.0 / na as f32 }
            }).collect()
        }).collect();

        let action_labels: Vec<String> = flat.actions_of(node_idx).iter().map(|a| a.label()).collect();

        out.push(NodeStrategy {
            node_id: nid,
            player,
            street: street.into(),
            path: path.into(),
            turn_card,
            strategy,
            action_labels: action_labels.clone(),
        });

        // Recurse into action children
        let a_off = n.action_offset as usize;
        for a in 0..na {
            let child_idx = flat.action_children[a_off + a];
            let child_path = if path.is_empty() {
                action_labels[a].clone()
            } else {
                format!("{}/{}", path, action_labels[a])
            };
            self.walk_node(flat, child_idx, chance_depth, &child_path, turn_card, out);
        }
    }

    pub fn root_strategy(&self, flat: &FlatTree) -> (Vec<Vec<f32>>, Vec<String>) {
        let root = &flat.nodes[0];
        let nid = root.node_id as usize;
        let na = root.action_count as usize;
        let k = if root.player_to_act == 0 { self.k_h } else { self.k_v };
        let strat = &self.strategy_sum[nid];
        // strategy_sum is action-major [a*k + b]; normalize per bucket
        let probs: Vec<Vec<f32>> = (0..k).map(|b| {
            let total: f32 = (0..na).map(|a| strat[a * k + b]).sum();
            if total > 0.0 { (0..na).map(|a| strat[a * k + b] / total).collect() }
            else { vec![1.0 / (na as f32); na] }
        }).collect();
        let labels: Vec<String> = flat.actions_of(0).iter().map(|a| a.label()).collect();
        (probs, labels)
    }
}
