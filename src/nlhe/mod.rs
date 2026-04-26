//! NLHE (No-Limit Texas Hold'em) solver — Rust port of `python/nlhe/`.
//!
//! The Python implementation is the reference; the Rust port exists to make
//! the inner CFR/showdown loops fast enough for offline precomputation
//! (preflop + flop GTO tables for HU at 200/500BB).
//!
//! Build out order:
//!   1. `cards`, `hand_eval` — pure functions, no state. Compare bit-for-bit
//!      with Python via cross-test.
//!   2. `showdown` — outcome matrices.
//!   3. `cfr` — solver inner loop.
//!   4. `tree` — game tree builders (river → turn → flop → preflop).
//!   5. `precompute` — orchestration: solve all standard preflop / flop spots
//!      and serialize results.

pub mod abstraction;
pub mod best_response;
pub mod bucket_cfr_flat;
pub mod bucket_cfr_multi;
pub mod bucket_cfr_river;
pub mod bucket_equity;
pub mod bucketing;
pub mod cache;
pub mod cards;
pub mod flat_tree;
pub mod cfr;
pub mod flop_showdown;
pub mod hand_eval;
pub mod multi_cfr;
pub mod preflop_cfr;
pub mod preflop_equity;
pub mod preflop_tree;
pub mod range_derive;
pub mod range_parser;
pub mod showdown;
pub mod simd_util;
pub mod storage;
pub mod tree;
pub mod turn_showdown;
