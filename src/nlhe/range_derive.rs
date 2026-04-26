//! Derive post-action ranges from a solved preflop tree.
//!
//! Walk the preflop solution following a specific action path; at each
//! decision node, multiply hero/villain reach by the strategy probability
//! for the action taken. The reach vector at the end of the path is the
//! "range that arrived here" — exactly what postflop CFR needs as input.

use std::collections::HashMap;

use super::cards::NUM_COMBOS;
use super::preflop_cfr::PreflopSolverState;
use super::tree::{Action, Node};

#[derive(Debug, Clone)]
pub struct DerivedRanges {
    /// Length-NUM_COMBOS weight vector for hero (= player 0).
    pub hero: Vec<f32>,
    /// Length-NUM_COMBOS weight vector for villain (= player 1).
    pub villain: Vec<f32>,
    /// Pot at the end of the action line.
    pub pot: f32,
    /// Stacks remaining for each player.
    pub stacks: (f32, f32),
}

/// Walk the preflop tree following `action_indices` (indices into each node's
/// `actions` list, in order). Returns the resulting (range, pot, stacks).
///
/// `initial_hero_range` and `initial_villain_range` are the starting reach
/// vectors at the root (typically all-1.0 for "any two cards").
pub fn derive_ranges_along_path(
    root: &Node,
    state: &PreflopSolverState,
    action_indices: &[usize],
    initial_hero_range: &[f32],
    initial_villain_range: &[f32],
) -> Option<DerivedRanges> {
    assert_eq!(initial_hero_range.len(), NUM_COMBOS);
    assert_eq!(initial_villain_range.len(), NUM_COMBOS);

    let mut hero = initial_hero_range.to_vec();
    let mut villain = initial_villain_range.to_vec();
    let mut node = root;

    for &a in action_indices {
        if node.is_terminal || node.is_chance { return None; }
        if a >= node.actions.len() { return None; }

        let player = node.player_to_act;
        let nid = node.node_id.get();
        if nid < 0 { return None; }

        // Compute the player's avg strategy at this node, per local-combo
        // (preflop CFR uses per-combo strategy, no bucketing).
        let na = node.actions.len();
        let strat = &state.strategy_sum[nid as usize];

        let combos = if player == 0 { &state.hero_combos } else { &state.villain_combos };
        let weights = if player == 0 { &state.hero_weights } else { &state.villain_weights };
        let _ = weights;

        // For each combo in the player's range, multiply that combo's reach
        // by the avg probability of the chosen action.
        let target = if player == 0 { &mut hero } else { &mut villain };
        for (local_idx, &global_idx) in combos.iter().enumerate() {
            let row = &strat[local_idx];
            let total: f32 = row.iter().sum();
            let prob = if total > 0.0 {
                row[a] / total
            } else {
                1.0 / na as f32
            };
            target[global_idx as usize] *= prob;
        }

        // Move to child
        node = &node.children[a];
    }

    Some(DerivedRanges {
        hero,
        villain,
        pot: node.pot.max(node.terminal_pot),
        stacks: node.stacks,
    })
}

/// Find every preflop action line that ends with both players having seen the
/// flop (terminal showdown with no fold). Returns each path along with
/// the derived (range, pot, stacks) state at that terminal.
pub fn enumerate_postflop_entry_points(
    root: &Node,
    state: &PreflopSolverState,
    initial_hero_range: &[f32],
    initial_villain_range: &[f32],
) -> Vec<(Vec<usize>, DerivedRanges)> {
    let mut out = Vec::new();
    fn walk(
        node: &Node,
        path: Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if node.is_terminal && node.terminal_winner.is_none() {
            out.push(path);
            return;
        }
        if node.is_terminal { return; } // fold terminal — no flop seen
        for (a, child) in node.children.iter().enumerate() {
            let mut p = path.clone();
            p.push(a);
            walk(child, p, out);
        }
    }
    let mut paths = Vec::new();
    walk(root, Vec::new(), &mut paths);
    for path in paths {
        if let Some(d) = derive_ranges_along_path(root, state, &path, initial_hero_range, initial_villain_range) {
            out.push((path, d));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::preflop_cfr::build_and_train_preflop;
    use super::super::preflop_equity::compute_preflop_equity_table;
    use super::super::preflop_tree::build_preflop_tree;

    #[test]
    fn derive_for_call_call_path() {
        let table = compute_preflop_equity_table(50, 42);
        let mut root = build_preflop_tree(200.0, 2);
        let any_two = vec![1.0f32; NUM_COMBOS];
        let state = build_and_train_preflop(
            &mut root, &table, &any_two, &any_two, (200.0, 200.0), 30,
        );
        let entries = enumerate_postflop_entry_points(&root, &state, &any_two, &any_two);
        // Should have at least one entry (limp-check or some call line)
        assert!(!entries.is_empty(), "no postflop entries found");
        // Each derived range should sum to a positive number ≤ original total
        for (_path, ranges) in &entries {
            let h_sum: f32 = ranges.hero.iter().sum();
            let v_sum: f32 = ranges.villain.iter().sum();
            assert!(h_sum > 0.0 && h_sum <= NUM_COMBOS as f32);
            assert!(v_sum > 0.0 && v_sum <= NUM_COMBOS as f32);
        }
    }
}
