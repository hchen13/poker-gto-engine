//! Benchmark: Rust river CFR on a representative spot.
//! Compare against `python/nlhe/cfr.py` for speedup multiple.

use poker_gto_engine::nlhe::cards::card_from_str;
use poker_gto_engine::nlhe::cfr::solve_river;
use poker_gto_engine::nlhe::range_parser::parse_range;
use poker_gto_engine::nlhe::showdown::compute_showdown_table;
use poker_gto_engine::nlhe::tree::build_river_tree;
use std::time::Instant;

fn main() {
    let board = ["Ad", "Kh", "7s", "3c", "2d"]
        .map(|c| card_from_str(c).unwrap());

    let hero = parse_range("AhAc, KsKc, QsJs, AKs, AKo, AQs").unwrap();
    let villain = parse_range("JcJd, TcTd, AcQc, KQs, KJs, QJs").unwrap();

    let table = compute_showdown_table(board, &hero, &villain);
    println!(
        "spot: {} hero combos × {} villain combos",
        table.n_hero, table.n_villain,
    );

    let mut root = build_river_tree(100.0, (200.0, 200.0), 0, 2);

    let iterations = 500;
    let start = Instant::now();
    let result = solve_river(&mut root, &table, (200.0, 200.0), iterations, None, None);
    let elapsed = start.elapsed();

    println!(
        "{} iterations in {:.3}s ({:.1} ms/iter)",
        iterations,
        elapsed.as_secs_f64(),
        elapsed.as_millis() as f64 / iterations as f64,
    );
    println!("hero_value (last iter): {:.3}", result.hero_value);
    println!("strategies extracted for {} hero combos", result.root_strategy.len());
}
