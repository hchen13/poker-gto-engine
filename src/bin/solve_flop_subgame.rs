//! On-demand flop subgame solver (bucketed CFR+).
//!
//! Reads JSON from stdin, runs bucketed flop CFR+ (flop → turn chance → river
//! chance → showdown), writes JSON to stdout. Same machinery as precompute.
//!
//! Input JSON:
//!   {
//!     "board": [c0, c1, c2],             // 3 card indices
//!     "oop_weights": [f32 × 1326],
//!     "ip_weights":  [f32 × 1326],
//!     "pot": f32, "oop_stack": f32, "ip_stack": f32,
//!     "first_to_act": 0 | 1,
//!     "iterations": u32,                   // default 200
//!     "k_buckets": usize,                  // default 16
//!     "max_raises": u32,                   // default 1 (flop)
//!     "turn_max_raises": u32,              // default 1
//!     "river_max_raises": u32,             // default 1
//!     "use_subset": bool                   // default true (subsample turn+river)
//!   }
//!
//! Output matches solve_turn_subgame shape with "nodes" carrying per-decision
//! strategies (flop + turn nodes; river is handled by showdown terminals).

use std::io::{self, Read};

use poker_gto_engine::nlhe::abstraction::rank_subsample;
use poker_gto_engine::nlhe::bucket_cfr_flat::solve_multi_bucketed_flat;
use poker_gto_engine::nlhe::bucket_cfr_multi::build_flop_equity_store;
use poker_gto_engine::nlhe::bucketing::{bucket_by_ehs, river_ehs};
use poker_gto_engine::nlhe::cards::{card_to_string, NUM_CARDS, NUM_COMBOS};
use poker_gto_engine::nlhe::tree::build_flop_tree_subset;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize)]
struct Input {
    board: [u8; 3],
    oop_weights: Vec<f32>,
    ip_weights: Vec<f32>,
    pot: f32,
    oop_stack: f32,
    ip_stack: f32,
    #[serde(default)]
    first_to_act: i8,
    #[serde(default = "d_iter")]
    iterations: u32,
    #[serde(default = "d_k")]
    k_buckets: usize,
    #[serde(default = "d_mr")]
    max_raises: u32,
    #[serde(default = "d_mr")]
    turn_max_raises: u32,
    #[serde(default = "d_mr")]
    river_max_raises: u32,
    #[serde(default = "d_subset")]
    use_subset: bool,
}

fn d_iter() -> u32 { 200 }
fn d_k() -> usize { 16 }
fn d_mr() -> u32 { 1 }
fn d_subset() -> bool { true }

#[derive(Serialize)]
struct NodeOut {
    path: String,
    player: i8,
    street: String,
    turn_card: Option<String>,
    action_labels: Vec<String>,
    strategy: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct Output {
    action_labels: Vec<String>,
    hero_strategy: Vec<Vec<f32>>,
    nodes: Vec<NodeOut>,
    bucket_hands: Vec<Value>,
    villain_bucket_hands: Vec<Value>,
    hero_ehs_range: Vec<[f32; 2]>,
    villain_ehs_range: Vec<[f32; 2]>,
    hero_value: f32,
    iterations: u32,
    k_buckets: usize,
}

fn ehs_range_per_bucket(ehs: &[f32], k: usize, boc: &[i8]) -> Vec<[f32; 2]> {
    let mut r: Vec<[f32; 2]> = vec![[f32::MAX, f32::MIN]; k];
    for c in 0..NUM_COMBOS {
        let b = boc[c]; if b < 0 { continue; }
        let e = ehs[c]; if e < 0.0 { continue; }
        let bi = b as usize;
        if e < r[bi][0] { r[bi][0] = e; }
        if e > r[bi][1] { r[bi][1] = e; }
    }
    for x in &mut r {
        if x[0] == f32::MAX { *x = [0.0, 0.0]; }
        x[0] = (x[0] * 1000.0).round() / 1000.0;
        x[1] = (x[1] * 1000.0).round() / 1000.0;
    }
    r
}

fn hand_types_for_bucket(boc: &[i8], k: usize) -> Vec<Value> {
    use poker_gto_engine::nlhe::cards::combo_cards;
    fn ht(a: u8, b: u8) -> String {
        let rc = ['2','3','4','5','6','7','8','9','T','J','Q','K','A'];
        let (ra, rb, sa, sb) = (a/4, b/4, a%4, b%4);
        let (hi, lo) = if ra >= rb { (ra, rb) } else { (rb, ra) };
        let suited = sa == sb;
        if hi == lo { format!("{}{}", rc[hi as usize], rc[lo as usize]) }
        else if suited { format!("{}{}s", rc[hi as usize], rc[lo as usize]) }
        else { format!("{}{}o", rc[hi as usize], rc[lo as usize]) }
    }
    let mut bs: Vec<Vec<String>> = (0..k).map(|_| Vec::new()).collect();
    for c in (0..NUM_COMBOS).rev() {
        let b = boc[c]; if b < 0 { continue; }
        let (a, cc) = combo_cards(c);
        let s = ht(a, cc);
        let bv = &mut bs[b as usize];
        if !bv.contains(&s) { bv.push(s); }
    }
    bs.into_iter().map(|v| Value::Array(v.into_iter().map(Value::String).collect())).collect()
}

fn main() {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf).expect("read stdin");
    let inp: Input = match serde_json::from_str(&buf) {
        Ok(v) => v,
        Err(e) => { eprintln!("{}", serde_json::json!({"error": format!("parse error: {}", e)})); std::process::exit(1); }
    };
    if inp.oop_weights.len() != NUM_COMBOS || inp.ip_weights.len() != NUM_COMBOS {
        eprintln!("{}", serde_json::json!({"error": "weights must be length 1326"})); std::process::exit(1);
    }
    if inp.pot <= 0.0 {
        eprintln!("{}", serde_json::json!({"error": "invalid pot"})); std::process::exit(1);
    }

    // Same rep5 construction as precompute: flop + mid-rank rep turn + next rep river
    let board_mask: u64 = inp.board.iter().fold(0u64, |a, &c| a | (1u64 << c));
    assert_eq!(board_mask.count_ones(), 3, "board must have 3 distinct cards");
    let remaining: Vec<u8> = (0..NUM_CARDS as u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
    let turn_rep = rank_subsample(&remaining);
    let river_rep = rank_subsample(&remaining);
    let mut rep5 = [0u8; 5];
    rep5[..3].copy_from_slice(&inp.board);
    rep5[3] = turn_rep[turn_rep.len() / 2];
    let mut ri = (river_rep.len() / 2 + 1) % river_rep.len();
    if river_rep[ri] == rep5[3] { ri = 0; }
    rep5[4] = river_rep[ri];

    let ehs_rep = river_ehs(&rep5);
    let oop_b = bucket_by_ehs(&ehs_rep, &inp.oop_weights, &rep5, inp.k_buckets);
    let ip_b  = bucket_by_ehs(&ehs_rep, &inp.ip_weights,  &rep5, inp.k_buckets);

    let (turn_subset_opt, river_subset_opt): (Option<Vec<u8>>, Option<Vec<u8>>) = if inp.use_subset {
        (Some(turn_rep.clone()), Some(river_rep.clone()))
    } else {
        (None, None)
    };
    let turn_subset_ref: Option<&[u8]> = turn_subset_opt.as_deref();
    let river_subset_ref: Option<&[u8]> = river_subset_opt.as_deref();

    let equity_store = build_flop_equity_store(inp.board, &oop_b, &ip_b, turn_subset_ref, river_subset_ref);
    let stacks = (inp.oop_stack, inp.ip_stack);
    let mut tree = build_flop_tree_subset(
        inp.board, inp.pot, stacks, inp.first_to_act,
        inp.max_raises, inp.turn_max_raises, inp.river_max_raises,
        turn_subset_ref, river_subset_ref,
    );

    let result = solve_multi_bucketed_flat(
        &mut tree, &oop_b, &ip_b, &equity_store, stacks, inp.iterations,
    );

    let hero_ehs_range    = ehs_range_per_bucket(&ehs_rep, oop_b.k, &oop_b.bucket_of_combo);
    let villain_ehs_range = ehs_range_per_bucket(&ehs_rep, ip_b.k,  &ip_b.bucket_of_combo);
    let bucket_hands         = hand_types_for_bucket(&oop_b.bucket_of_combo, oop_b.k);
    let villain_bucket_hands = hand_types_for_bucket(&ip_b.bucket_of_combo,  ip_b.k);

    let nodes: Vec<NodeOut> = result.all_nodes.into_iter().map(|n| NodeOut {
        path: n.path,
        player: n.player,
        street: n.street,
        turn_card: n.turn_card.map(card_to_string),
        action_labels: n.action_labels,
        strategy: n.strategy,
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
