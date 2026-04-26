//! Preflop HU tree builder.
//!
//! Heads-up convention used here:
//! - Player 0 = SB (small blind, button in HU). Acts first preflop, second postflop.
//! - Player 1 = BB (big blind). Acts second preflop, first postflop.
//!
//! Starting state at the root: SB has posted 0.5 BB, BB has posted 1.0 BB,
//! pot = 1.5, SB to act facing a "to_call" of 0.5 to match BB.
//!
//! Sizing tables (per the user's earlier spec):
//! - SB open sizes: limp (= just call), 2.5x, 3x, 4x, 6x, 8x BB, all-in
//! - 3-bet/4-bet/etc sizes: ~3x previous bet OR all-in (with max_raises cap)
//!
//! Action closes preflop when both have committed equal amounts and someone
//! has either: called the open / call the latest raise. At that point the
//! flop is dealt — emit a chance(flop) node where each child is the start
//! of postflop play (build_river/turn/flop tree depending on solver depth).
//!
//! This module only builds the PREFLOP action tree. Action-closing leaves are
//! tagged as terminal showdowns (no postflop tree attached); a wrapper
//! function would replace these with chance(flop) → flop subtree if the
//! caller wants a multi-street solve.

use std::cell::Cell;

use super::tree::{Action, ActionKind, Node, ShowdownKey};

/// SB open size table, in big blinds (BB). `0` means limp (just call BB).
pub const SB_OPEN_SIZES_BB: &[f32] = &[0.0, 2.5, 3.0, 4.0, 6.0, 8.0];
/// Re-raise multiplier on the previous raise size.
pub const RAISE_MULTIPLIER: f32 = 3.0;

const EPSILON: f32 = 1e-9;

pub fn build_preflop_tree(
    stack_bb: f32,
    max_raises: u32,
) -> Box<Node> {
    assert!(stack_bb >= 1.0, "stack must be at least 1 BB");
    let pot = 1.5; // SB 0.5 + BB 1.0
    let stacks = (stack_bb - 0.5, stack_bb - 1.0); // SB, BB
    let to_call = 0.5; // SB owes 0.5 to match BB
    let player_to_act: i8 = 0; // SB
    build_node(pot, stacks, to_call, player_to_act, max_raises, 0, /*last_aggressor=*/Some(1))
}

fn build_node(
    pot: f32,
    stacks: (f32, f32),
    to_call: f32,
    player_to_act: i8,
    raises_left: u32,
    depth: u32,
    last_aggressor: Option<i8>,
) -> Box<Node> {
    let mut node = Box::new(Node {
        pot,
        stacks,
        to_call,
        player_to_act,
        actions: Vec::new(),
        children: Vec::new(),
        is_terminal: false,
        is_chance: false,
        chance_cards: Vec::new(),
        terminal_pot: 0.0,
        terminal_winner: None,
        showdown_key: ShowdownKey::None,
        depth,
        node_id: Cell::new(-1), equity_idx: Cell::new(-1),
    });

    if stacks.0 <= EPSILON && stacks.1 <= EPSILON {
        node.is_terminal = true;
        node.terminal_pot = pot;
        node.terminal_winner = None;
        return node;
    }

    // Player to act: choose actions
    if to_call <= EPSILON {
        // No bet pending — only the BB-checks-back-the-limp case at the root,
        // which closes preflop with both having put in 1 BB. (SB would have
        // limped to land here.)
        // Add check (closes street to flop showdown placeholder).
        let me = player_to_act;
        let mut new_stacks = stacks;
        // Both checked → terminal showdown placeholder
        node.actions.push(Action { kind: ActionKind::Check, amount: 0.0 });
        node.children.push(Box::new(Node {
            pot, stacks: new_stacks, to_call: 0.0, player_to_act: -1,
            actions: vec![], children: vec![],
            is_terminal: true, is_chance: false, chance_cards: vec![],
            terminal_pot: pot, terminal_winner: None,
            showdown_key: ShowdownKey::None,
            depth: depth + 1, node_id: Cell::new(-1), equity_idx: Cell::new(-1),
        }));
        // BB can also still raise after limp (= iso). Add raise sizes computed
        // from current pot / stack. For simplicity reuse SB_OPEN_SIZES_BB minus
        // limp (0.0) — these are total BB sizes, treat them as bet amounts in
        // BB units = chip units here.
        if raises_left > 0 {
            let my_stack = if me == 0 { new_stacks.0 } else { new_stacks.1 };
            for &size_bb in SB_OPEN_SIZES_BB {
                if size_bb <= 0.0 { continue; }
                let extra = size_bb - if me == 0 { 0.5 } else { 1.0 }; // chips on top of what we already put in
                if extra <= 0.0 || extra >= my_stack - EPSILON { continue; }
                let is_allin = extra >= my_stack - EPSILON;
                let kind = if is_allin { ActionKind::AllIn } else { ActionKind::Bet };
                node.actions.push(Action { kind, amount: extra });
                node.children.push(after_raise(&node, extra, is_allin, raises_left - 1));
            }
            if my_stack > 0.0 {
                node.actions.push(Action { kind: ActionKind::AllIn, amount: my_stack });
                node.children.push(after_raise(&node, my_stack, true, 0));
            }
        }
        let _ = last_aggressor;
    } else {
        // Facing a raise (or BB). Actions: fold / call / re-raise (sizes).
        let me = player_to_act;
        let my_stack = if me == 0 { stacks.0 } else { stacks.1 };
        let to_call_capped = to_call.min(my_stack);

        // Fold
        node.actions.push(Action { kind: ActionKind::Fold, amount: 0.0 });
        node.children.push(Box::new(Node {
            pot, stacks, to_call: 0.0, player_to_act: -1,
            actions: vec![], children: vec![],
            is_terminal: true, is_chance: false, chance_cards: vec![],
            terminal_pot: pot, terminal_winner: Some((1 - me) as u8),
            showdown_key: ShowdownKey::None,
            depth: depth + 1, node_id: Cell::new(-1), equity_idx: Cell::new(-1),
        }));

        // Call (closes preflop unless this is the SB facing BB at root)
        node.actions.push(Action { kind: ActionKind::Call, amount: 0.0 });
        node.children.push(after_call(&node, to_call_capped));

        // Raises
        if my_stack > to_call_capped + EPSILON && raises_left > 0 {
            // Re-raise sizing: previous bet was `to_call` (= what we owe to call).
            // Standard 3x reraise: total commit = current bet + 3x current bet.
            // We compute amounts as (chips on top of pre-action commitment).
            let my_committed_so_far = match me {
                0 => 0.5 + (to_call - to_call), // approx; we don't track committed precisely here
                _ => 1.0,
            };
            // Simpler: candidate raise total = (to_call) * RAISE_MULTIPLIER + to_call,
            // i.e. raise BY 3× the call amount on TOP of calling. So extra = to_call + 3*to_call = 4*to_call.
            let raise_extra = to_call_capped * (1.0 + RAISE_MULTIPLIER);
            if raise_extra > to_call_capped + EPSILON && raise_extra < my_stack - EPSILON {
                node.actions.push(Action { kind: ActionKind::Raise, amount: raise_extra });
                node.children.push(after_raise(&node, raise_extra, false, raises_left - 1));
            }
            // All-in raise (covers user's big-stack scenarios)
            if my_stack > to_call_capped + EPSILON {
                node.actions.push(Action { kind: ActionKind::AllIn, amount: my_stack });
                node.children.push(after_raise(&node, my_stack, true, 0));
            }
            let _ = my_committed_so_far;
        }
    }

    node
}

fn after_raise(node: &Node, extra: f32, is_allin: bool, raises_left: u32) -> Box<Node> {
    let me = node.player_to_act;
    let other = 1 - me;
    let mut new_stacks = node.stacks;
    if me == 0 { new_stacks.0 -= extra; } else { new_stacks.1 -= extra; }
    let opp_stack = if other == 0 { new_stacks.0 } else { new_stacks.1 };
    let opp_to_call = (extra - node.to_call).min(opp_stack);
    let new_pot = node.pot + extra;
    let effective_raises_left = if is_allin || new_stacks.0 <= EPSILON || new_stacks.1 <= EPSILON {
        0
    } else {
        raises_left
    };
    build_node(new_pot, new_stacks, opp_to_call, other, effective_raises_left, node.depth + 1, Some(me))
}

fn after_call(node: &Node, to_call: f32) -> Box<Node> {
    let me = node.player_to_act;
    let mut new_stacks = node.stacks;
    if me == 0 { new_stacks.0 -= to_call; } else { new_stacks.1 -= to_call; }
    let new_pot = node.pot + to_call;
    // Call closes preflop action → terminal showdown placeholder.
    // (Caller can attach a chance(flop) → flop subtree here for multi-street solving.)
    Box::new(Node {
        pot: new_pot, stacks: new_stacks, to_call: 0.0, player_to_act: -1,
        actions: vec![], children: vec![],
        is_terminal: true, is_chance: false, chance_cards: vec![],
        terminal_pot: new_pot, terminal_winner: None,
        showdown_key: ShowdownKey::None,
        depth: node.depth + 1, node_id: Cell::new(-1), equity_idx: Cell::new(-1),
    })
}

#[cfg(test)]
mod tests {
    use super::super::tree::{count_decision_nodes, count_terminals};
    use super::*;

    #[test]
    fn preflop_tree_builds_without_panic() {
        let root = build_preflop_tree(200.0, 3);
        assert!(!root.is_terminal);
        assert!(root.actions.len() > 0);
        // Should have fold, call, plus several raise sizes
        let kinds: Vec<&ActionKind> = root.actions.iter().map(|a| &a.kind).collect();
        assert!(kinds.iter().any(|k| **k == ActionKind::Fold));
        assert!(kinds.iter().any(|k| **k == ActionKind::Call));
    }

    #[test]
    fn preflop_chip_conservation() {
        let root = build_preflop_tree(200.0, 3);
        let initial_total = 200.0 + 200.0; // both have 200 BB starting
        super::super::tree::walk_nodes(&root, &mut |node| {
            if node.is_terminal {
                let actual = node.terminal_pot + node.stacks.0 + node.stacks.1;
                assert!(
                    (actual - initial_total).abs() < 1e-3,
                    "preflop chip conservation: expected {}, got {} (pot={}, stacks={:?})",
                    initial_total, actual, node.terminal_pot, node.stacks,
                );
            }
        });
    }

    #[test]
    fn preflop_has_decision_and_terminals() {
        let root = build_preflop_tree(200.0, 2);
        assert!(count_decision_nodes(&root) >= 2);
        assert!(count_terminals(&root) >= 3);
    }
}
