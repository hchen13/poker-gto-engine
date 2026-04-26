//! Game tree for HU postflop subgames. Mirrors `python/nlhe/tree.py`.
//!
//! Tree structure:
//!   - decision nodes have an `actions` list and `children` list (parallel)
//!   - terminal nodes have `terminal_winner` (0/1 for fold, None for showdown)
//!   - chance nodes have `is_chance = true`, with one child per remaining card
//!     and `chance_cards` listing those card indices
//!   - showdown terminals carry an optional `showdown_board_key` for multi-board
//!     subgames (turn / flop)

use std::cell::Cell;

const EPSILON: f32 = 1e-9;

pub const BET_FRACTIONS: &[f32] = &[0.33, 0.50, 0.67, 1.00, 1.50];
pub const RAISE_POT_FRACTIONS: &[f32] = &[1.00];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionKind {
    Check,
    Fold,
    Call,
    Bet,
    Raise,
    AllIn,
}

#[derive(Debug, Clone, Copy)]
pub struct Action {
    pub kind: ActionKind,
    pub amount: f32,
}

impl Action {
    pub fn label(&self) -> String {
        match self.kind {
            ActionKind::Check => "check".into(),
            ActionKind::Fold => "fold".into(),
            ActionKind::Call => "call".into(),
            ActionKind::Bet => format!("bet_{:.2}", self.amount),
            ActionKind::Raise => format!("raise_{:.2}", self.amount),
            ActionKind::AllIn => format!("allin_{:.2}", self.amount),
        }
    }
}

/// Showdown board key. For single-board subgames (river), `None`. For turn,
/// just the river card index. For flop, (turn_card, river_card).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShowdownKey {
    None,
    River(u8),
    TurnRiver(u8, u8),
}

#[derive(Debug)]
pub struct Node {
    pub pot: f32,
    pub stacks: (f32, f32),
    pub to_call: f32,
    pub player_to_act: i8, // 0, 1, or -1 for terminal/chance
    pub actions: Vec<Action>,
    pub children: Vec<Box<Node>>,
    pub is_terminal: bool,
    pub is_chance: bool,
    pub chance_cards: Vec<u8>,
    pub terminal_pot: f32,
    pub terminal_winner: Option<u8>, // None = showdown
    pub showdown_key: ShowdownKey,
    pub depth: u32,
    /// Solver-assigned id for decision nodes. Set during solver init.
    pub node_id: Cell<i32>,
    /// Solver-assigned equity-table index for terminal nodes (bucket multi-board CFR).
    /// -1 if not a showdown terminal (fold).
    pub equity_idx: Cell<i32>,
}

impl Node {
    fn new_terminal(pot: f32, stacks: (f32, f32), winner: Option<u8>, depth: u32) -> Self {
        Self {
            pot,
            stacks,
            to_call: 0.0,
            player_to_act: -1,
            actions: Vec::new(),
            children: Vec::new(),
            is_terminal: true,
            is_chance: false,
            chance_cards: Vec::new(),
            terminal_pot: pot,
            terminal_winner: winner,
            showdown_key: ShowdownKey::None,
            depth,
            node_id: Cell::new(-1), equity_idx: Cell::new(-1),
        }
    }
}

pub fn build_river_tree(
    pot: f32,
    stacks: (f32, f32),
    first_to_act: i8,
    max_raises: u32,
) -> Box<Node> {
    assert!(pot >= 0.0);
    assert!(stacks.0 >= 0.0 && stacks.1 >= 0.0);
    assert!(first_to_act == 0 || first_to_act == 1);
    build_node(pot, stacks, 0.0, first_to_act, max_raises, None, 0)
}

fn build_node(
    pot: f32,
    stacks: (f32, f32),
    to_call: f32,
    player_to_act: i8,
    raises_left: u32,
    last_aggressor: Option<i8>,
    depth: u32,
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

    // Both all-in pre-action: showdown immediately
    if stacks.0 <= EPSILON && stacks.1 <= EPSILON {
        node.is_terminal = true;
        node.terminal_pot = pot;
        node.terminal_winner = None;
        return node;
    }

    if to_call <= EPSILON {
        add_noncall_actions(&mut node, raises_left, last_aggressor);
    } else {
        add_facing_bet_actions(&mut node, raises_left);
    }
    node
}

fn add_noncall_actions(node: &mut Node, raises_left: u32, last_aggressor: Option<i8>) {
    let me = node.player_to_act;
    let other = 1 - me;
    let my_stack = if me == 0 { node.stacks.0 } else { node.stacks.1 };

    // Check
    let check_child = after_check(node, last_aggressor);
    node.actions.push(Action {
        kind: ActionKind::Check,
        amount: 0.0,
    });
    node.children.push(check_child);

    // Bets
    let bet_sizes = bet_candidates(node.pot, my_stack);
    for amount in bet_sizes {
        let is_allin = amount >= my_stack - EPSILON;
        let kind = if is_allin { ActionKind::AllIn } else { ActionKind::Bet };
        node.actions.push(Action { kind, amount });
        node.children.push(after_bet_or_raise(node, amount, is_allin, raises_left.saturating_sub(1)));
    }

    let _ = other;
}

fn add_facing_bet_actions(node: &mut Node, raises_left: u32) {
    let me = node.player_to_act;
    let _other = 1 - me;
    let my_stack = if me == 0 { node.stacks.0 } else { node.stacks.1 };
    let to_call = node.to_call.min(my_stack);

    // Fold
    node.actions.push(Action { kind: ActionKind::Fold, amount: 0.0 });
    node.children.push(after_fold(node));

    // Call
    node.actions.push(Action { kind: ActionKind::Call, amount: 0.0 });
    node.children.push(after_call(node, to_call));

    // Raise
    if my_stack > to_call + EPSILON && raises_left > 0 {
        let raise_sizes = raise_candidates(node.pot, to_call, my_stack);
        for total_extra in raise_sizes {
            let is_allin = total_extra >= my_stack - EPSILON;
            let kind = if is_allin { ActionKind::AllIn } else { ActionKind::Raise };
            node.actions.push(Action { kind, amount: total_extra });
            node.children.push(after_bet_or_raise(node, total_extra, is_allin, raises_left.saturating_sub(1)));
        }
    }
}

fn bet_candidates(pot: f32, stack: f32) -> Vec<f32> {
    let mut out = Vec::new();
    let mut seen = Vec::new();
    for &frac in BET_FRACTIONS {
        let amt = frac * pot;
        if amt <= 0.0 || amt >= stack - EPSILON {
            continue;
        }
        if approx_in(amt, &seen) {
            continue;
        }
        seen.push(amt);
        out.push(amt);
    }
    if stack > 0.0 {
        out.push(stack); // all-in
    }
    out
}

fn raise_candidates(pot_before_call: f32, to_call: f32, my_stack: f32) -> Vec<f32> {
    let pot_after_call = pot_before_call + 2.0 * to_call;
    let mut out = Vec::new();
    let mut seen = Vec::new();
    for &frac in RAISE_POT_FRACTIONS {
        let raise_portion = frac * pot_after_call;
        let total_extra = to_call + raise_portion;
        if total_extra <= to_call + EPSILON || total_extra >= my_stack - EPSILON {
            continue;
        }
        if approx_in(total_extra, &seen) {
            continue;
        }
        seen.push(total_extra);
        out.push(total_extra);
    }
    if my_stack > to_call + EPSILON {
        out.push(my_stack); // all-in raise
    }
    out
}

fn after_check(node: &Node, _last_aggressor: Option<i8>) -> Box<Node> {
    let me = node.player_to_act;
    let other = 1 - me;
    if node.depth == 0 {
        // Opener checks → other player can check or bet (raises_left reset by convention)
        return build_node(node.pot, node.stacks, 0.0, other, 0, None, node.depth + 1);
    }
    // Second check → showdown
    Box::new(Node::new_terminal(node.pot, node.stacks, None, node.depth + 1))
}

fn after_bet_or_raise(node: &Node, extra: f32, is_allin: bool, raises_left: u32) -> Box<Node> {
    let me = node.player_to_act;
    let other = 1 - me;
    let mut new_stacks = node.stacks;
    if me == 0 {
        new_stacks.0 -= extra;
    } else {
        new_stacks.1 -= extra;
    }
    let opp_stack = if other == 0 { new_stacks.0 } else { new_stacks.1 };
    let opp_to_call = (extra - node.to_call).min(opp_stack);
    let new_pot = node.pot + extra;

    let effective_raises_left = if is_allin || new_stacks.0 <= EPSILON || new_stacks.1 <= EPSILON {
        0
    } else {
        raises_left
    };
    build_node(new_pot, new_stacks, opp_to_call, other, effective_raises_left, Some(me), node.depth + 1)
}

fn after_fold(node: &Node) -> Box<Node> {
    let folder = node.player_to_act;
    let winner = (1 - folder) as u8;
    Box::new(Node::new_terminal(node.pot, node.stacks, Some(winner), node.depth + 1))
}

fn after_call(node: &Node, to_call: f32) -> Box<Node> {
    let me = node.player_to_act;
    let mut new_stacks = node.stacks;
    if me == 0 {
        new_stacks.0 -= to_call;
    } else {
        new_stacks.1 -= to_call;
    }
    let new_pot = node.pot + to_call;
    Box::new(Node::new_terminal(new_pot, new_stacks, None, node.depth + 1))
}

fn approx_in(x: f32, lst: &[f32]) -> bool {
    lst.iter().any(|&y| (x - y).abs() < EPSILON)
}

pub fn walk_nodes_mut(root: &mut Node, f: &mut impl FnMut(&mut Node)) {
    f(root);
    for child in root.children.iter_mut() {
        walk_nodes_mut(child, f);
    }
}

pub fn walk_nodes(root: &Node, f: &mut impl FnMut(&Node)) {
    f(root);
    for child in root.children.iter() {
        walk_nodes(child, f);
    }
}

pub fn count_nodes(root: &Node) -> usize {
    let mut n = 0;
    walk_nodes(root, &mut |_| n += 1);
    n
}

pub fn count_decision_nodes(root: &Node) -> usize {
    let mut n = 0;
    walk_nodes(root, &mut |node| {
        if !node.is_terminal && !node.is_chance {
            n += 1;
        }
    });
    n
}

pub fn count_terminals(root: &Node) -> usize {
    let mut n = 0;
    walk_nodes(root, &mut |node| {
        if node.is_terminal {
            n += 1;
        }
    });
    n
}

pub fn count_chance_nodes(root: &Node) -> usize {
    let mut n = 0;
    walk_nodes(root, &mut |node| {
        if node.is_chance { n += 1; }
    });
    n
}

/// Build a turn HU subgame: turn decisions → chance(river) → river subtree.
pub fn build_turn_tree(
    board_4: [u8; 4],
    pot: f32,
    stacks: (f32, f32),
    first_to_act: i8,
    max_raises: u32,
    river_max_raises: u32,
) -> Box<Node> {
    build_turn_tree_subset(board_4, pot, stacks, first_to_act, max_raises, river_max_raises, None)
}

/// Same as `build_turn_tree`, but if `river_subset` is provided the chance
/// node enumerates ONLY those river cards (must be a subset of remaining
/// cards). This is the public-card abstraction hook — caller-supplied subset
/// trades accuracy for tree size.
pub fn build_turn_tree_subset(
    board_4: [u8; 4],
    pot: f32,
    stacks: (f32, f32),
    first_to_act: i8,
    max_raises: u32,
    river_max_raises: u32,
    river_subset: Option<&[u8]>,
) -> Box<Node> {
    let board_mask: u64 = board_4.iter().fold(0u64, |a, &c| a | (1u64 << c));
    assert_eq!(board_mask.count_ones(), 4, "board_4 must have 4 distinct cards");
    let all_remaining: Vec<u8> = (0..super::cards::NUM_CARDS as u8)
        .filter(|c| board_mask & (1u64 << c) == 0)
        .collect();
    let remaining: Vec<u8> = match river_subset {
        None => all_remaining,
        Some(sub) => {
            for &c in sub {
                assert!(board_mask & (1u64 << c) == 0, "subset contains board card");
            }
            sub.to_vec()
        }
    };
    let mut tree = build_river_tree(pot, stacks, first_to_act, max_raises);
    expand_showdowns_to_chance(&mut tree, &remaining, river_max_raises);
    tree
}

fn expand_showdowns_to_chance(node: &mut Node, remaining: &[u8], river_max_raises: u32) {
    if node.is_terminal && node.terminal_winner.is_none() {
        // Replace with chance node
        let mut children: Vec<Box<Node>> = Vec::with_capacity(remaining.len());
        for &card in remaining {
            let mut river_root = build_river_tree(node.terminal_pot, node.stacks, 0, river_max_raises);
            tag_river_showdowns(&mut river_root, card);
            children.push(river_root);
        }
        node.is_terminal = false;
        node.is_chance = true;
        node.chance_cards = remaining.to_vec();
        node.children = children;
        return;
    }
    for child in node.children.iter_mut() {
        expand_showdowns_to_chance(child, remaining, river_max_raises);
    }
}

fn tag_river_showdowns(node: &mut Node, river_card: u8) {
    if node.is_terminal && node.terminal_winner.is_none() {
        node.showdown_key = ShowdownKey::River(river_card);
        return;
    }
    for child in node.children.iter_mut() {
        tag_river_showdowns(child, river_card);
    }
}

/// Build a flop HU subgame: flop decisions → chance(turn) → turn subtree
/// (which itself contains chance(river) → river subtree). Deepest showdowns
/// carry `ShowdownKey::TurnRiver(turn, river)`.
pub fn build_flop_tree(
    board_3: [u8; 3],
    pot: f32,
    stacks: (f32, f32),
    first_to_act: i8,
    max_raises: u32,
    turn_max_raises: u32,
    river_max_raises: u32,
) -> Box<Node> {
    build_flop_tree_subset(
        board_3, pot, stacks, first_to_act, max_raises,
        turn_max_raises, river_max_raises, None, None,
    )
}

/// Abstraction-aware flop tree builder. `turn_subset` and `river_subset`
/// (when provided) restrict each chance node's enumeration to those cards.
/// This is how flop precompute becomes tractable: subsetting 49 turn × 48
/// river = 2352 runouts down to e.g. 8 × 8 = 64 representative pairs.
pub fn build_flop_tree_subset(
    board_3: [u8; 3],
    pot: f32,
    stacks: (f32, f32),
    first_to_act: i8,
    max_raises: u32,
    turn_max_raises: u32,
    river_max_raises: u32,
    turn_subset: Option<&[u8]>,
    river_subset: Option<&[u8]>,
) -> Box<Node> {
    let board_mask: u64 = board_3.iter().fold(0u64, |a, &c| a | (1u64 << c));
    assert_eq!(board_mask.count_ones(), 3, "board_3 must have 3 distinct cards");
    let all_remaining: Vec<u8> = (0..super::cards::NUM_CARDS as u8)
        .filter(|c| board_mask & (1u64 << c) == 0)
        .collect();
    let turn_cards: Vec<u8> = match turn_subset {
        None => all_remaining.clone(),
        Some(sub) => {
            for &c in sub {
                assert!(board_mask & (1u64 << c) == 0, "subset has board card");
            }
            sub.to_vec()
        }
    };
    let mut tree = build_river_tree(pot, stacks, first_to_act, max_raises);
    expand_showdowns_to_turn_chance_subset(
        &mut tree, board_3, &turn_cards, turn_max_raises, river_max_raises, river_subset,
    );
    tree
}

fn expand_showdowns_to_turn_chance_subset(
    node: &mut Node,
    board_3: [u8; 3],
    turn_cards: &[u8],
    turn_max_raises: u32,
    river_max_raises: u32,
    river_subset: Option<&[u8]>,
) {
    if node.is_terminal && node.terminal_winner.is_none() {
        let mut children: Vec<Box<Node>> = Vec::with_capacity(turn_cards.len());
        for &turn_card in turn_cards {
            let mut new_board_4 = [0u8; 4];
            new_board_4[..3].copy_from_slice(&board_3);
            new_board_4[3] = turn_card;
            // For each turn child, restrict inner chance to the river_subset
            // (if given), filtering out the just-dealt turn card.
            let mut turn_subtree = match river_subset {
                None => build_turn_tree(
                    new_board_4, node.terminal_pot, node.stacks, 0,
                    turn_max_raises, river_max_raises,
                ),
                Some(s) => {
                    let filtered: Vec<u8> = s.iter().copied().filter(|&c| c != turn_card).collect();
                    build_turn_tree_subset(
                        new_board_4, node.terminal_pot, node.stacks, 0,
                        turn_max_raises, river_max_raises, Some(&filtered),
                    )
                }
            };
            retag_with_turn_card(&mut turn_subtree, turn_card);
            children.push(turn_subtree);
        }
        node.is_terminal = false;
        node.is_chance = true;
        node.chance_cards = turn_cards.to_vec();
        node.children = children;
        return;
    }
    for child in node.children.iter_mut() {
        expand_showdowns_to_turn_chance_subset(
            child, board_3, turn_cards, turn_max_raises, river_max_raises, river_subset,
        );
    }
}

fn expand_showdowns_to_turn_chance(
    node: &mut Node,
    board_3: [u8; 3],
    remaining: &[u8],
    turn_max_raises: u32,
    river_max_raises: u32,
) {
    if node.is_terminal && node.terminal_winner.is_none() {
        let mut children: Vec<Box<Node>> = Vec::with_capacity(remaining.len());
        for &turn_card in remaining {
            let mut new_board_4 = [0u8; 4];
            new_board_4[..3].copy_from_slice(&board_3);
            new_board_4[3] = turn_card;
            let mut turn_subtree = build_turn_tree(
                new_board_4, node.terminal_pot, node.stacks, 0,
                turn_max_raises, river_max_raises,
            );
            retag_with_turn_card(&mut turn_subtree, turn_card);
            children.push(turn_subtree);
        }
        node.is_terminal = false;
        node.is_chance = true;
        node.chance_cards = remaining.to_vec();
        node.children = children;
        return;
    }
    for child in node.children.iter_mut() {
        expand_showdowns_to_turn_chance(child, board_3, remaining, turn_max_raises, river_max_raises);
    }
}

fn retag_with_turn_card(node: &mut Node, turn_card: u8) {
    if node.is_terminal && node.terminal_winner.is_none() {
        if let ShowdownKey::River(river) = node.showdown_key {
            node.showdown_key = ShowdownKey::TurnRiver(turn_card, river);
        }
        return;
    }
    for child in node.children.iter_mut() {
        retag_with_turn_card(child, turn_card);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn river_tree_builds() {
        let root = build_river_tree(100.0, (200.0, 200.0), 0, 2);
        assert!(!root.is_terminal);
        assert!(root.actions.len() > 0);
    }

    #[test]
    fn chip_conservation_at_terminals() {
        let root = build_river_tree(100.0, (150.0, 150.0), 0, 2);
        let initial_total = 100.0 + 150.0 + 150.0;
        walk_nodes(&root, &mut |node| {
            if node.is_terminal {
                let actual = node.terminal_pot + node.stacks.0 + node.stacks.1;
                assert!(
                    (actual - initial_total).abs() < 1e-3,
                    "chip conservation violated: pot={} stacks={:?} expected total={}",
                    node.terminal_pot, node.stacks, initial_total,
                );
            }
        });
    }

    #[test]
    fn fold_terminals_have_winner() {
        let root = build_river_tree(100.0, (200.0, 200.0), 0, 2);
        let mut found_fold = false;
        walk_nodes(&root, &mut |node| {
            if node.is_terminal && node.terminal_winner.is_some() {
                found_fold = true;
            }
        });
        assert!(found_fold);
    }
}
