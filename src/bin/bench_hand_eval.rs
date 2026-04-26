//! Microbench: how fast can we evaluate 7-card hands?
//!
//! Compare against Python `hand_eval.evaluate_seven` to estimate the speedup
//! achievable by the Rust port, which justifies (or doesn't) Inc 4's investment.

use poker_gto_engine::nlhe::cards::{combo_cards, NUM_COMBOS};
use poker_gto_engine::nlhe::hand_eval::evaluate_seven;
use std::time::Instant;

fn main() {
    // Build a deterministic stream of 7-card boards by enumerating combos as
    // hole cards on a fixed 5-card board.
    let board: [u8; 5] = [
        // As, Kh, 7s, 3c, 2d encoded:
        // As = (14-2)*4 + 0 = 48; Kh = (13-2)*4 + 1 = 45; 7s = (7-2)*4 + 0 = 20;
        // 3c = (3-2)*4 + 3 = 7; 2d = (2-2)*4 + 2 = 2.
        48, 45, 20, 7, 2,
    ];
    let board_set: u64 = board.iter().fold(0u64, |acc, &c| acc | (1u64 << c));

    // Build a list of valid 7-card hands (board + non-board hole pair)
    let mut hands: Vec<[u8; 7]> = Vec::with_capacity(NUM_COMBOS);
    for combo in 0..NUM_COMBOS {
        let (a, b) = combo_cards(combo);
        if (board_set >> a) & 1 == 1 || (board_set >> b) & 1 == 1 {
            continue;
        }
        let mut h = [0u8; 7];
        h[..5].copy_from_slice(&board);
        h[5] = a;
        h[6] = b;
        hands.push(h);
    }

    let n_hands = hands.len();
    println!("evaluating {} unique 7-card hands", n_hands);

    // Warm up
    let mut acc: u32 = 0;
    for h in hands.iter() {
        acc ^= evaluate_seven(*h);
    }

    let iters = 50;
    let start = Instant::now();
    for _ in 0..iters {
        for h in hands.iter() {
            acc ^= evaluate_seven(*h);
        }
    }
    let elapsed = start.elapsed();
    let total_evals = (n_hands * iters) as f64;
    let per_eval_ns = elapsed.as_nanos() as f64 / total_evals;
    println!(
        "{} evals in {:.3}s = {:.1} ns/eval ({:.2}M evals/sec)  [acc={}]",
        total_evals as u64,
        elapsed.as_secs_f64(),
        per_eval_ns,
        total_evals / elapsed.as_secs_f64() / 1e6,
        acc,
    );
}
