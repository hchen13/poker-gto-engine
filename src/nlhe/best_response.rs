//! Best-response and exploitability for the Rust solvers.
//!
//! Mirrors `python/nlhe/best_response.py` and `turn_best_response.py`. Works
//! against any `MultiBoardLookup` plus a `SolverState<L>`-like accessor.
//! Also provides a river-specific implementation that uses the simpler
//! single-board `cfr::SolverState`.

use super::cfr::SolverState as RiverSolverState;
use super::multi_cfr::{MultiBoardLookup, MultiSolverState};
use super::showdown::{CONFLICT, WIN_HERO, WIN_TIE, WIN_VILLAIN};
use super::tree::Node;

/// Best-response value for a multi-board solver (turn or flop).
pub fn best_response_value_multi<L: MultiBoardLookup>(
    state: &MultiSolverState<L>,
    root: &Node,
    br_player: i8,
) -> f32 {
    let opp_range: Vec<f32> = if br_player == 0 {
        state.lookup.villain_weights().to_vec()
    } else {
        state.lookup.hero_weights().to_vec()
    };
    let br_range: Vec<f32> = if br_player == 0 {
        state.lookup.hero_weights().to_vec()
    } else {
        state.lookup.villain_weights().to_vec()
    };
    let values = traverse_multi(state, root, &opp_range, br_player);
    let denom = state.pair_weight_avg;
    if denom <= 0.0 { return 0.0; }
    let mut acc = 0.0;
    for i in 0..br_range.len() {
        acc += br_range[i] * values[i];
    }
    acc / denom
}

pub fn exploitability_multi<L: MultiBoardLookup>(
    state: &MultiSolverState<L>, root: &Node,
) -> f32 {
    best_response_value_multi(state, root, 0) + best_response_value_multi(state, root, 1) - root.pot
}

fn traverse_multi<L: MultiBoardLookup>(
    state: &MultiSolverState<L>, node: &Node, reach_opp: &[f32], br_player: i8,
) -> Vec<f32> {
    if node.is_terminal {
        return terminal_multi(state, node, reach_opp, br_player);
    }
    if node.is_chance {
        let nc_br = if br_player == 0 { state.n_h } else { state.n_v };
        let mut total = vec![0.0f32; nc_br];
        for child in node.children.iter() {
            let cu = traverse_multi(state, child, reach_opp, br_player);
            for i in 0..nc_br { total[i] += cu[i]; }
        }
        let denom = node.children.len() as f32;
        for v in total.iter_mut() { *v /= denom; }
        return total;
    }
    let player = node.player_to_act;
    let node_id = node.node_id.get() as usize;
    let na = node.actions.len();
    let nc_br = if br_player == 0 { state.n_h } else { state.n_v };

    if player != br_player {
        let opp_strat = avg_strategy_at_node_multi(state, node_id, player, na);
        let opp_bucket_of = if br_player == 0 { &state.villain_bucket_of } else { &state.hero_bucket_of };
        let mut child_values: Vec<Vec<f32>> = Vec::with_capacity(na);
        for a in 0..na {
            let new_reach_opp: Vec<f32> = (0..reach_opp.len())
                .map(|j| reach_opp[j] * opp_strat[opp_bucket_of[j]][a])
                .collect();
            child_values.push(traverse_multi(state, &node.children[a], &new_reach_opp, br_player));
        }
        let mut out = vec![0.0f32; nc_br];
        for combo in 0..nc_br {
            for a in 0..na { out[combo] += child_values[a][combo]; }
        }
        return out;
    }

    let mut child_values: Vec<Vec<f32>> = Vec::with_capacity(na);
    for a in 0..na {
        child_values.push(traverse_multi(state, &node.children[a], reach_opp, br_player));
    }
    let mut out = vec![0.0f32; nc_br];
    for combo in 0..nc_br {
        let mut best = child_values[0][combo];
        for a in 1..na {
            let v = child_values[a][combo];
            if v > best { best = v; }
        }
        out[combo] = best;
    }
    out
}

fn terminal_multi<L: MultiBoardLookup>(
    state: &MultiSolverState<L>, node: &Node, reach_opp: &[f32], br_player: i8,
) -> Vec<f32> {
    let hero_c = state.initial_stacks.0 - node.stacks.0;
    let villain_c = state.initial_stacks.1 - node.stacks.1;
    let pot = node.terminal_pot;
    let (hero_profit, villain_profit) = match node.terminal_winner {
        Some(0) => (pot - hero_c, -villain_c),
        Some(1) => (-hero_c, pot - villain_c),
        _ => (0.0f32, 0.0f32),
    };

    if node.terminal_winner.is_some() {
        if br_player == 0 {
            let mut out = vec![0.0f32; state.n_h];
            for i in 0..state.n_h {
                let (ha, hb) = state.hero_pairs[i];
                let mut total_v = 0.0;
                for j in 0..state.n_v {
                    let (va, vb) = state.villain_pairs[j];
                    if va == ha || va == hb || vb == ha || vb == hb { continue; }
                    total_v += reach_opp[j];
                }
                out[i] = total_v * hero_profit;
            }
            return out;
        }
        let mut out = vec![0.0f32; state.n_v];
        for j in 0..state.n_v {
            let (va, vb) = state.villain_pairs[j];
            let mut total_h = 0.0;
            for i in 0..state.n_h {
                let (ha, hb) = state.hero_pairs[i];
                if ha == va || ha == vb || hb == va || hb == vb { continue; }
                total_h += reach_opp[i];
            }
            out[j] = total_h * villain_profit;
        }
        return out;
    }

    let key = node.showdown_key;
    if br_player == 0 {
        let mut out = vec![0.0f32; state.n_h];
        for i in 0..state.n_h {
            let mut acc = 0.0;
            for j in 0..state.n_v {
                let o = state.lookup.outcome(key, i, j);
                if o == CONFLICT { continue; }
                let share = if o == WIN_HERO { pot } else if o == WIN_TIE { pot * 0.5 } else { 0.0 };
                acc += reach_opp[j] * (share - hero_c);
            }
            out[i] = acc;
        }
        return out;
    }
    let mut out = vec![0.0f32; state.n_v];
    for j in 0..state.n_v {
        let mut acc = 0.0;
        for i in 0..state.n_h {
            let o = state.lookup.outcome(key, i, j);
            if o == CONFLICT { continue; }
            let share = if o == WIN_VILLAIN { pot } else if o == WIN_TIE { pot * 0.5 } else { 0.0 };
            acc += reach_opp[i] * (share - villain_c);
        }
        out[j] = acc;
    }
    out
}

fn avg_strategy_at_node_multi<L: MultiBoardLookup>(
    state: &MultiSolverState<L>, node_id: usize, player: i8, na: usize,
) -> Vec<Vec<f32>> {
    let nb = if player == 0 { state.n_buckets_h } else { state.n_buckets_v };
    let mut out = Vec::with_capacity(nb);
    let strat = &state.strategy_sum[node_id];
    let uniform = 1.0 / (na as f32);
    for b in 0..nb {
        let row = &strat[b];
        let total: f32 = row.iter().sum();
        if total > 0.0 {
            out.push(row.iter().map(|v| v / total).collect());
        } else {
            out.push(vec![uniform; na]);
        }
    }
    out
}

/// River single-board BR.
pub fn best_response_value_river(
    state: &RiverSolverState, root: &Node, br_player: i8,
) -> f32 {
    let opp_range: Vec<f32> = if br_player == 0 {
        state.table.villain_weights.clone()
    } else {
        state.table.hero_weights.clone()
    };
    let br_range: Vec<f32> = if br_player == 0 {
        state.table.hero_weights.clone()
    } else {
        state.table.villain_weights.clone()
    };
    let values = traverse_river(state, root, &opp_range, br_player);
    let denom = state.pair_weight_total;
    if denom <= 0.0 { return 0.0; }
    let mut acc = 0.0;
    for i in 0..br_range.len() {
        acc += br_range[i] * values[i];
    }
    acc / denom
}

pub fn exploitability_river(state: &RiverSolverState, root: &Node) -> f32 {
    best_response_value_river(state, root, 0) + best_response_value_river(state, root, 1) - root.pot
}

fn traverse_river(
    state: &RiverSolverState, node: &Node, reach_opp: &[f32], br_player: i8,
) -> Vec<f32> {
    if node.is_terminal {
        return terminal_river(state, node, reach_opp, br_player);
    }
    let player = node.player_to_act;
    let node_id = node.node_id.get() as usize;
    let na = node.actions.len();
    let nc_br = if br_player == 0 { state.n_h } else { state.n_v };
    if player != br_player {
        let opp_strat = avg_strategy_at_node_river(state, node_id, player, na);
        let opp_bucket_of = if br_player == 0 { &state.villain_bucket_of } else { &state.hero_bucket_of };
        let mut child_values: Vec<Vec<f32>> = Vec::with_capacity(na);
        for a in 0..na {
            let new_reach_opp: Vec<f32> = (0..reach_opp.len())
                .map(|j| reach_opp[j] * opp_strat[opp_bucket_of[j]][a]).collect();
            child_values.push(traverse_river(state, &node.children[a], &new_reach_opp, br_player));
        }
        let mut out = vec![0.0f32; nc_br];
        for combo in 0..nc_br {
            for a in 0..na { out[combo] += child_values[a][combo]; }
        }
        return out;
    }
    let mut child_values: Vec<Vec<f32>> = Vec::with_capacity(na);
    for a in 0..na {
        child_values.push(traverse_river(state, &node.children[a], reach_opp, br_player));
    }
    let mut out = vec![0.0f32; nc_br];
    for combo in 0..nc_br {
        let mut best = child_values[0][combo];
        for a in 1..na {
            let v = child_values[a][combo];
            if v > best { best = v; }
        }
        out[combo] = best;
    }
    out
}

fn terminal_river(
    state: &RiverSolverState, node: &Node, reach_opp: &[f32], br_player: i8,
) -> Vec<f32> {
    let hero_c = state.initial_stacks.0 - node.stacks.0;
    let villain_c = state.initial_stacks.1 - node.stacks.1;
    let pot = node.terminal_pot;
    let (hero_profit, villain_profit) = match node.terminal_winner {
        Some(0) => (pot - hero_c, -villain_c),
        Some(1) => (-hero_c, pot - villain_c),
        _ => (0.0f32, 0.0f32),
    };

    if br_player == 0 {
        let mut out = vec![0.0f32; state.n_h];
        if node.terminal_winner.is_some() {
            for i in 0..state.n_h {
                let mut total_v = 0.0;
                for j in 0..state.n_v {
                    if state.table.outcome_at(i, j) == CONFLICT { continue; }
                    total_v += reach_opp[j];
                }
                out[i] = total_v * hero_profit;
            }
            return out;
        }
        for i in 0..state.n_h {
            let mut acc = 0.0;
            for j in 0..state.n_v {
                let o = state.table.outcome_at(i, j);
                if o == CONFLICT { continue; }
                let share = if o == WIN_HERO { pot } else if o == WIN_TIE { pot * 0.5 } else { 0.0 };
                acc += reach_opp[j] * (share - hero_c);
            }
            out[i] = acc;
        }
        return out;
    }

    let mut out = vec![0.0f32; state.n_v];
    if node.terminal_winner.is_some() {
        for j in 0..state.n_v {
            let mut total_h = 0.0;
            for i in 0..state.n_h {
                if state.table.outcome_at(i, j) == CONFLICT { continue; }
                total_h += reach_opp[i];
            }
            out[j] = total_h * villain_profit;
        }
        return out;
    }
    for j in 0..state.n_v {
        let mut acc = 0.0;
        for i in 0..state.n_h {
            let o = state.table.outcome_at(i, j);
            if o == CONFLICT { continue; }
            let share = if o == WIN_VILLAIN { pot } else if o == WIN_TIE { pot * 0.5 } else { 0.0 };
            acc += reach_opp[i] * (share - villain_c);
        }
        out[j] = acc;
    }
    out
}

fn avg_strategy_at_node_river(
    state: &RiverSolverState, node_id: usize, player: i8, na: usize,
) -> Vec<Vec<f32>> {
    let nb = if player == 0 { state.n_buckets_h } else { state.n_buckets_v };
    let mut out = Vec::with_capacity(nb);
    let strat = &state.strategy_sum[node_id];
    let uniform = 1.0 / (na as f32);
    for b in 0..nb {
        let row = &strat[b];
        let total: f32 = row.iter().sum();
        if total > 0.0 {
            out.push(row.iter().map(|v| v / total).collect());
        } else {
            out.push(vec![uniform; na]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::cards::card_from_str;
    use super::super::cfr::Solver;
    use super::super::range_parser::parse_range;
    use super::super::showdown::compute_showdown_table;
    use super::super::tree::build_river_tree;
    use super::*;

    fn board(s: &[&str]) -> [u8; 5] {
        let mut out = [0u8; 5];
        for (i, c) in s.iter().enumerate() { out[i] = card_from_str(c).unwrap(); }
        out
    }

    #[test]
    fn river_exploitability_non_negative() {
        let b = board(&["Ad", "Kh", "7s", "3c", "2d"]);
        let hero = parse_range("AhAc, KsKc, QsJs").unwrap();
        let villain = parse_range("JcJd, TcTd, AcQc").unwrap();
        let table = compute_showdown_table(b, &hero, &villain);
        let mut root = build_river_tree(100.0, (150.0, 150.0), 0, 2);
        let mut solver = Solver::new(&mut root, &table, (150.0, 150.0), None, None);
        solver.state.train(&root, 300);
        let exp = exploitability_river(&solver.state, &root);
        assert!(exp >= -0.5, "exploitability = {}", exp);
        // and on a tiny pot 100 should be small after 300 iters
        assert!(exp < 15.0, "exploitability = {}", exp);
    }
}
