//! Flat, cache-friendly tree representation.
//!
//! The pointer-chasing `Vec<Box<Node>>` tree is correct but causes cache
//! misses in the CFR hot loop. This module converts it to a flat array with
//! u32 child indices, yielding better memory locality (all nodes packed in
//! one contiguous buffer).
//!
//! Semantics match `tree::Node` exactly — same fields, same conventions.

use super::tree::{Action, Node, ShowdownKey};

#[derive(Debug, Clone)]
pub struct FlatNode {
    pub pot: f32,
    pub stacks: [f32; 2],
    pub to_call: f32,
    pub player_to_act: i8,
    pub is_terminal: bool,
    pub is_chance: bool,
    pub terminal_winner: i8, // -1 for none / showdown
    pub equity_idx: i32,
    pub node_id: i32,
    pub showdown_key: ShowdownKey,
    pub terminal_pot: f32,
    /// Inclusive range into FlatTree.actions and FlatTree.children_indices.
    pub action_offset: u32,
    pub action_count: u32,
    pub chance_offset: u32,
    pub chance_count: u32,
}

pub struct FlatTree {
    pub nodes: Vec<FlatNode>,
    /// All actions flattened; indexed via FlatNode.action_offset..+action_count.
    pub actions: Vec<Action>,
    /// Per-action child indices (parallel to actions).
    pub action_children: Vec<u32>,
    /// Chance-card → child-index, for chance nodes only.
    pub chance_children: Vec<u32>,
    pub chance_cards: Vec<u8>,
}

pub fn flatten(root: &Node) -> FlatTree {
    let mut tree = FlatTree {
        nodes: Vec::new(),
        actions: Vec::new(),
        action_children: Vec::new(),
        chance_children: Vec::new(),
        chance_cards: Vec::new(),
    };
    visit(root, &mut tree);
    tree
}

fn visit(node: &Node, out: &mut FlatTree) -> u32 {
    // Pre-allocate slot (fill in child indices after recursing)
    let idx = out.nodes.len() as u32;
    let action_offset = out.actions.len() as u32;
    let chance_offset = out.chance_children.len() as u32;

    out.nodes.push(FlatNode {
        pot: node.pot,
        stacks: [node.stacks.0, node.stacks.1],
        to_call: node.to_call,
        player_to_act: node.player_to_act,
        is_terminal: node.is_terminal,
        is_chance: node.is_chance,
        terminal_winner: node.terminal_winner.map(|w| w as i8).unwrap_or(-1),
        equity_idx: node.equity_idx.get(),
        node_id: node.node_id.get(),
        showdown_key: node.showdown_key,
        terminal_pot: node.terminal_pot,
        action_offset,
        action_count: node.actions.len() as u32,
        chance_offset,
        chance_count: node.chance_cards.len() as u32,
    });

    // Reserve action slots
    for a in node.actions.iter() {
        out.actions.push(*a);
        out.action_children.push(0); // placeholder
    }
    // Reserve chance slots
    for &c in node.chance_cards.iter() {
        out.chance_cards.push(c);
        out.chance_children.push(0); // placeholder
    }

    // Recurse children, filling in indices
    if node.is_chance {
        for (i, child) in node.children.iter().enumerate() {
            let child_idx = visit(child, out);
            out.chance_children[(chance_offset as usize) + i] = child_idx;
        }
    } else {
        for (i, child) in node.children.iter().enumerate() {
            let child_idx = visit(child, out);
            out.action_children[(action_offset as usize) + i] = child_idx;
        }
    }

    idx
}

impl FlatTree {
    #[inline]
    pub fn action_children_of(&self, node_idx: u32) -> &[u32] {
        let n = &self.nodes[node_idx as usize];
        let start = n.action_offset as usize;
        let end = start + n.action_count as usize;
        &self.action_children[start..end]
    }
    #[inline]
    pub fn actions_of(&self, node_idx: u32) -> &[Action] {
        let n = &self.nodes[node_idx as usize];
        let start = n.action_offset as usize;
        let end = start + n.action_count as usize;
        &self.actions[start..end]
    }
    #[inline]
    pub fn chance_children_of(&self, node_idx: u32) -> &[u32] {
        let n = &self.nodes[node_idx as usize];
        let start = n.chance_offset as usize;
        let end = start + n.chance_count as usize;
        &self.chance_children[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::super::tree::build_river_tree;
    use super::*;

    #[test]
    fn flatten_preserves_counts() {
        let root = build_river_tree(100.0, (200.0, 200.0), 0, 2);
        let flat = flatten(&root);
        // root node should be index 0 with na children matching actions
        assert_eq!(flat.nodes[0].action_count as usize, root.actions.len());
        assert_eq!(flat.actions_of(0).len(), root.actions.len());
    }
}
