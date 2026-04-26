//! Bench: bucketed vs unbucketed turn CFR on the user's 99 hand.

use poker_gto_engine::nlhe::bucket_cfr_multi::{build_turn_equity_store, solve_multi_bucketed};
use poker_gto_engine::nlhe::bucketing::{bucket_by_ehs, river_ehs};
use poker_gto_engine::nlhe::cards::card_from_str;
use poker_gto_engine::nlhe::multi_cfr::MultiSolverState;
use poker_gto_engine::nlhe::range_parser::parse_range;
use poker_gto_engine::nlhe::tree::build_turn_tree;
use poker_gto_engine::nlhe::turn_showdown::compute_turn_showdown;
use std::time::Instant;

fn main() {
    let board_4: [u8; 4] = ["Kc", "Jh", "9s", "5d"].map(|s| card_from_str(s).unwrap());
    let hero = parse_range("99, JJ, KK, AKs, AKo").unwrap();
    let villain = parse_range("AA, KK, QQ, JJ, AKs, AKo").unwrap();
    let pot = 203.0;
    let stacks = (1920.0, 820.0);
    let iterations = 200;

    // ===== Unbucketed =====
    println!("=== Unbucketed turn CFR ===");
    let table = compute_turn_showdown(board_4, &hero, &villain);
    println!("  {} hero × {} villain combos", table.n_hero, table.n_villain);
    let mut root1 = build_turn_tree(board_4, pot, stacks, 1, 1, 1);
    let t0 = Instant::now();
    let mut state = MultiSolverState::new(&mut root1, &table, stacks, None, None);
    state.train(&root1, iterations);
    let t_unbucketed = t0.elapsed();
    println!("  {} iters in {:.2?}", iterations, t_unbucketed);
    println!("  hero_value: {:.3}", state.last_root_value());

    // ===== Bucketed (K=16) =====
    for k in [8, 16, 32] {
        println!("\n=== Bucketed turn CFR (K={}) ===", k);
        // Use simple river-EHS at a representative 5-card extension
        let mut board_5 = [0u8; 5];
        board_5[..4].copy_from_slice(&board_4);
        board_5[4] = card_from_str("2s").unwrap(); // dummy river for bucketing
        let ehs = river_ehs(&board_5);
        let t_prep = Instant::now();
        let hero_b = bucket_by_ehs(&ehs, &hero, &board_5, k);
        let villain_b = bucket_by_ehs(&ehs, &villain, &board_5, k);
        let store = build_turn_equity_store(board_4, &hero_b, &villain_b, None);
        let prep_time = t_prep.elapsed();
        println!("  prep ({} equity tables): {:.2?}", store.tables.len(), prep_time);

        let mut root2 = build_turn_tree(board_4, pot, stacks, 1, 1, 1);
        let t0 = Instant::now();
        let result = solve_multi_bucketed(
            &mut root2, &hero_b, &villain_b, &store, stacks, iterations,
        );
        let solve_time = t0.elapsed();
        println!("  {} iters in {:.2?}", iterations, solve_time);
        println!("  hero_value: {:.3}", result.hero_value);
        let speedup = t_unbucketed.as_secs_f64() / solve_time.as_secs_f64();
        println!("  SPEEDUP: {:.1}×", speedup);
    }
}
