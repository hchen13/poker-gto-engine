//! Solve preflop, enumerate "see flop" action lines, print derived ranges.
//! Used to understand what postflop starting states we need to precompute for.

use poker_gto_engine::nlhe::cards::{combo_cards, NUM_COMBOS};
use poker_gto_engine::nlhe::preflop_cfr::build_and_train_preflop;
use poker_gto_engine::nlhe::preflop_equity::{compute_preflop_equity_table, combo_to_class};
use poker_gto_engine::nlhe::preflop_tree::build_preflop_tree;
use poker_gto_engine::nlhe::range_derive::enumerate_postflop_entry_points;
use poker_gto_engine::nlhe::tree::{Action, Node};

fn main() {
    let stack_bb: f32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200.0);
    let max_raises: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let iters: u32 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(500);

    println!("Preflop solve: {}BB, max_raises={}, iters={}", stack_bb, max_raises, iters);
    println!("Computing 169×169 equity table (500 samples/cell)...");
    let equity = compute_preflop_equity_table(500, 0xC0FFEE);

    let mut root = build_preflop_tree(stack_bb, max_raises);
    let any_two = vec![1.0f32; NUM_COMBOS];
    println!("Running preflop CFR...");
    let state = build_and_train_preflop(
        &mut root, &equity, &any_two, &any_two, (stack_bb, stack_bb), iters,
    );
    println!("Preflop hero EV: {:.3} BB", state.last_root_value());

    // Enumerate action lines ending at postflop entry
    let entries = enumerate_postflop_entry_points(&root, &state, &any_two, &any_two);
    println!("\nFound {} postflop entry points:\n", entries.len());

    for (i, (path, ranges)) in entries.iter().enumerate() {
        let labels = labels_for_path(&root, path);
        let h_total: f32 = ranges.hero.iter().sum();
        let v_total: f32 = ranges.villain.iter().sum();
        let h_nonzero = ranges.hero.iter().filter(|&&w| w > 0.0).count();
        let v_nonzero = ranges.villain.iter().filter(|&&w| w > 0.0).count();

        println!("[{}] action line: {}", i, labels.join(" → "));
        println!("     pot: {:.2} BB, stacks: {:?}", ranges.pot, ranges.stacks);
        println!("     hero range:    {} combos, total weight {:.1} (of 1326)", h_nonzero, h_total);
        println!("     villain range: {} combos, total weight {:.1} (of 1326)", v_nonzero, v_total);

        // Top-5 hand classes by weight
        print!("     hero top classes: ");
        print_top_classes(&ranges.hero, 5);
        print!("     vill top classes: ");
        print_top_classes(&ranges.villain, 5);
        println!();
    }
}

fn labels_for_path(root: &Node, path: &[usize]) -> Vec<String> {
    let mut out = Vec::new();
    let mut node = root;
    for &a in path {
        if a >= node.actions.len() { break; }
        out.push(node.actions[a].label());
        node = &node.children[a];
    }
    out
}

fn print_top_classes(range: &[f32], n: usize) {
    let mut by_class: std::collections::HashMap<u8, f32> = std::collections::HashMap::new();
    for (combo, &w) in range.iter().enumerate() {
        if w <= 0.0 { continue; }
        let c = combo_to_class(combo);
        *by_class.entry(c).or_insert(0.0) += w;
    }
    let mut pairs: Vec<(u8, f32)> = by_class.into_iter().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (i, (cls, w)) in pairs.iter().take(n).enumerate() {
        if i > 0 { print!(", "); }
        print!("{}={:.1}", class_label(*cls), w);
    }
    println!();
}

fn class_label(cls: u8) -> String {
    // Inverse of combo_to_class — approximate.
    if cls < 13 {
        let rank = cls + 2;
        let r = rank_char(rank);
        format!("{}{}", r, r)
    } else if cls < 13 + 78 {
        // Suited
        let idx = (cls - 13) as u32;
        let (hi, lo) = non_pair_hi_lo(idx);
        format!("{}{}s", rank_char(hi), rank_char(lo))
    } else {
        let idx = (cls - 13 - 78) as u32;
        let (hi, lo) = non_pair_hi_lo(idx);
        format!("{}{}o", rank_char(hi), rank_char(lo))
    }
}

fn rank_char(r: u8) -> char {
    match r {
        2..=9 => ('0' as u8 + r) as char,
        10 => 'T', 11 => 'J', 12 => 'Q', 13 => 'K', 14 => 'A',
        _ => '?',
    }
}

fn non_pair_hi_lo(idx: u32) -> (u8, u8) {
    let mut i = 0u32;
    for h in (3..=14).rev() {
        for l in (2..h).rev() {
            if i == idx { return (h as u8, l as u8); }
            i += 1;
        }
    }
    (0, 0)
}
