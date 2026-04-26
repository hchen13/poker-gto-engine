//! Bench: bucketed vs unbucketed river CFR on "any two cards" range.
//!
//! This is the POC for Phase 1: confirms that range bucketing actually
//! gives the promised 100×+ speedup on realistic-scale problems.

use poker_gto_engine::nlhe::bucket_cfr_river::solve_river_bucketed;
use poker_gto_engine::nlhe::bucket_equity::compute_bucket_equity_river;
use poker_gto_engine::nlhe::bucketing::{bucket_by_ehs, river_ehs};
use poker_gto_engine::nlhe::cards::{card_from_str, NUM_COMBOS};
use poker_gto_engine::nlhe::cfr::Solver;
use poker_gto_engine::nlhe::showdown::compute_showdown_table;
use poker_gto_engine::nlhe::tree::build_river_tree;
use std::time::Instant;

fn main() {
    let board: [u8; 5] = ["Ad", "Kh", "7s", "3c", "2d"].map(|s| card_from_str(s).unwrap());
    let any_two = vec![1.0f32; NUM_COMBOS];
    let iterations = 200;

    // ===== Unbucketed =====
    println!("=== Unbucketed river CFR (any two cards) ===");
    let table = compute_showdown_table(board, &any_two, &any_two);
    println!("  {} hero combos × {} villain combos (1126² = {} pair eval per terminal)",
             table.n_hero, table.n_villain, table.n_hero * table.n_villain);
    let mut root1 = build_river_tree(100.0, (200.0, 200.0), 0, 2);
    let t0 = Instant::now();
    let mut solver = Solver::new(&mut root1, &table, (200.0, 200.0), None, None);
    solver.state.train(&root1, iterations);
    let unbucketed_time = t0.elapsed();
    println!("  {} iters in {:.2?} ({:.1}ms/iter)",
             iterations, unbucketed_time, unbucketed_time.as_millis() as f64 / iterations as f64);
    println!("  hero_value: {:.3}", solver.state.last_root_value());

    // ===== Bucketed (K=16) =====
    for k in [8, 16, 32] {
        println!("\n=== Bucketed river CFR (K={}) ===", k);
        let ehs = river_ehs(&board);
        let t_prep = Instant::now();
        let bucketing = bucket_by_ehs(&ehs, &any_two, &board, k);
        let equity = compute_bucket_equity_river(&board, &bucketing, &bucketing);
        let prep_time = t_prep.elapsed();
        println!("  prep (EHS + bucketing + equity table): {:.2?}", prep_time);

        let mut root2 = build_river_tree(100.0, (200.0, 200.0), 0, 2);
        let t0 = Instant::now();
        let result = solve_river_bucketed(
            &mut root2, &bucketing, &bucketing, &equity, (200.0, 200.0), iterations,
        );
        let solve_time = t0.elapsed();
        println!("  {} iters in {:.2?} ({:.2}ms/iter)",
                 iterations, solve_time, solve_time.as_millis() as f64 / iterations as f64);
        println!("  hero_value: {:.3}", result.hero_value);
        let speedup = unbucketed_time.as_secs_f64() / solve_time.as_secs_f64();
        println!("  SPEEDUP vs unbucketed: {:.1}×", speedup);
    }
}
