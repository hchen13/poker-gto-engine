//! On-demand river subgame solver.
//!
//! Reads JSON from stdin, runs bucketed river CFR+, writes JSON to stdout.
//!
//! Input JSON:
//!   {
//!     "board": [c0, c1, c2, c3, c4],   // 5 card indices (u8)
//!     "oop_weights": [f32 × 1326],      // reach weights for OOP (BB), 0.0 = not in range
//!     "ip_weights":  [f32 × 1326],      // reach weights for IP (SB)
//!     "pot": f32,                        // pot size in BB at river
//!     "oop_stack": f32,                  // effective stack for OOP
//!     "ip_stack":  f32,                  // effective stack for IP
//!     "first_to_act": 0 | 1,            // 0=OOP, 1=IP
//!     "iterations": u32,                 // default 500
//!     "k_buckets": usize                 // default 16
//!   }
//!
//! Output JSON:
//!   {
//!     "action_labels": ["fold","call","bet_X",...],
//!     "hero_strategy": [[f32 × n_actions] × k_buckets],  // OOP strategy (player 0 at root)
//!     "bucket_hands": ["AKo","AKs",...],     // OOP bucket representative hands (from EHS)
//!     "villain_bucket_hands": ["KQo",...],   // IP bucket representative hands
//!     "hero_ehs_range": [[lo, hi] × k],      // EHS range per OOP bucket
//!     "villain_ehs_range": [[lo, hi] × k],   // EHS range per IP bucket
//!     "hero_value": f32,                     // EV for OOP in chips
//!     "iterations": u32
//!   }
//!
//! Usage:
//!   echo '<json>' | cargo run --release --bin solve_river_subgame
//!   # or from Python: subprocess with stdin pipe

use std::io::{self, Read};

use poker_gto_engine::nlhe::bucket_cfr_river::solve_river_bucketed;
use poker_gto_engine::nlhe::bucket_equity::compute_bucket_equity_river;
use poker_gto_engine::nlhe::bucketing::{bucket_by_ehs, river_ehs};
use poker_gto_engine::nlhe::cards::NUM_COMBOS;
use poker_gto_engine::nlhe::tree::build_river_tree;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize)]
struct Input {
    board: [u8; 5],
    oop_weights: Vec<f32>,
    ip_weights: Vec<f32>,
    pot: f32,
    oop_stack: f32,
    ip_stack: f32,
    #[serde(default)]
    first_to_act: i8, // 0=OOP, 1=IP
    #[serde(default = "default_iterations")]
    iterations: u32,
    #[serde(default = "default_k")]
    k_buckets: usize,
}

fn default_iterations() -> u32 { 500 }
fn default_k() -> usize { 16 }

#[derive(Serialize)]
struct NodeOut {
    path: String,
    player: i8,                         // 0=OOP, 1=IP
    action_labels: Vec<String>,
    strategy: Vec<Vec<f32>>,            // [bucket][action]
}

#[derive(Serialize)]
struct Output {
    action_labels: Vec<String>,
    hero_strategy: Vec<Vec<f32>>,       // [bucket][action] — root OOP strategy (kept for back-compat)
    nodes: Vec<NodeOut>,                // all decision nodes (includes IP response nodes)
    bucket_hands: Vec<Value>,           // OOP bucket representative hands
    villain_bucket_hands: Vec<Value>,   // IP bucket representative hands
    hero_ehs_range: Vec<[f32; 2]>,
    villain_ehs_range: Vec<[f32; 2]>,
    hero_value: f32,
    iterations: u32,
    k_buckets: usize,
}

fn ehs_range_per_bucket(ehs: &[f32], weights: &[f32], k: usize, bucket_of_combo: &[i8]) -> Vec<[f32; 2]> {
    let mut ranges: Vec<[f32; 2]> = vec![[f32::MAX, f32::MIN]; k];
    for combo in 0..NUM_COMBOS {
        let b = bucket_of_combo[combo];
        if b < 0 { continue; }
        let e = ehs[combo];
        if e < 0.0 { continue; }
        let bi = b as usize;
        if e < ranges[bi][0] { ranges[bi][0] = e; }
        if e > ranges[bi][1] { ranges[bi][1] = e; }
    }
    // Replace unset buckets with [0,0]
    for r in &mut ranges {
        if r[0] == f32::MAX { *r = [0.0, 0.0]; }
    }
    // round to 3 decimal places
    for r in &mut ranges {
        r[0] = (r[0] * 1000.0).round() / 1000.0;
        r[1] = (r[1] * 1000.0).round() / 1000.0;
    }
    ranges
}

fn hand_types_for_bucket(bucket_of_combo: &[i8], k: usize) -> Vec<Value> {
    use poker_gto_engine::nlhe::cards::{combo_cards};

    // Map combo index to hand type string
    fn hand_type(a: u8, b: u8) -> String {
        let rank_chars = ['2','3','4','5','6','7','8','9','T','J','Q','K','A'];
        let ra = (a / 4) as usize;
        let rb = (b / 4) as usize;
        let sa = a % 4;
        let sb = b % 4;
        let (hi, lo, suited) = if ra >= rb {
            (ra, rb, sa == sb)
        } else {
            (rb, ra, sa == sb)
        };
        if hi == lo {
            format!("{}{}", rank_chars[hi], rank_chars[lo])
        } else if suited {
            format!("{}{}s", rank_chars[hi], rank_chars[lo])
        } else {
            format!("{}{}o", rank_chars[hi], rank_chars[lo])
        }
    }

    // Collect unique hand types per bucket
    let mut buckets: Vec<Vec<String>> = (0..k).map(|_| Vec::new()).collect();
    for combo in (0..NUM_COMBOS).rev() {
        let b = bucket_of_combo[combo];
        if b < 0 { continue; }
        let (a, c) = combo_cards(combo);
        let ht = hand_type(a, c);
        let bv = &mut buckets[b as usize];
        if !bv.contains(&ht) { bv.push(ht); }
    }
    buckets.into_iter().map(|v| {
        Value::Array(v.into_iter().map(Value::String).collect())
    }).collect()
}

fn main() {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf).expect("read stdin");

    let inp: Input = match serde_json::from_str(&buf) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{}", serde_json::json!({"error": format!("parse error: {}", e)}));
            std::process::exit(1);
        }
    };

    // Validate
    if inp.oop_weights.len() != NUM_COMBOS || inp.ip_weights.len() != NUM_COMBOS {
        eprintln!("{}", serde_json::json!({"error": "oop_weights and ip_weights must have length 1326"}));
        std::process::exit(1);
    }
    if inp.pot <= 0.0 || inp.oop_stack < 0.0 || inp.ip_stack < 0.0 {
        eprintln!("{}", serde_json::json!({"error": "invalid pot or stacks"}));
        std::process::exit(1);
    }

    let ehs = river_ehs(&inp.board);

    let oop_b = bucket_by_ehs(&ehs, &inp.oop_weights, &inp.board, inp.k_buckets);
    let ip_b  = bucket_by_ehs(&ehs, &inp.ip_weights,  &inp.board, inp.k_buckets);

    let equity = compute_bucket_equity_river(&inp.board, &oop_b, &ip_b);

    let stacks = (inp.oop_stack, inp.ip_stack);
    let mut root = build_river_tree(inp.pot, stacks, inp.first_to_act, 2);

    let result = solve_river_bucketed(
        &mut root, &oop_b, &ip_b, &equity, stacks, inp.iterations,
    );

    let hero_ehs_range  = ehs_range_per_bucket(&ehs, &inp.oop_weights, oop_b.k, &oop_b.bucket_of_combo);
    let villain_ehs_range = ehs_range_per_bucket(&ehs, &inp.ip_weights,  ip_b.k,  &ip_b.bucket_of_combo);

    let bucket_hands         = hand_types_for_bucket(&oop_b.bucket_of_combo, oop_b.k);
    let villain_bucket_hands = hand_types_for_bucket(&ip_b.bucket_of_combo,  ip_b.k);

    let nodes: Vec<NodeOut> = result.all_nodes.into_iter().map(|(path, player, labels, strat)| {
        NodeOut { path, player, action_labels: labels, strategy: strat }
    }).collect();

    let out = Output {
        action_labels: result.action_labels,
        hero_strategy: result.root_strategy,
        nodes,
        bucket_hands,
        villain_bucket_hands,
        hero_ehs_range,
        villain_ehs_range,
        hero_value: result.hero_value,
        iterations: result.iterations,
        k_buckets: inp.k_buckets,
    };

    println!("{}", serde_json::to_string(&out).unwrap());
}
