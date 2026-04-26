//! Offline precompute orchestrator.
//!
//! V2: actually runs preflop GTO using precomputed equity table, persists to disk.
//!
//! Pipeline:
//!   1. Compute (or load) preflop equity table (169×169)
//!   2. Build preflop HU tree at given stack depth
//!   3. Run CFR+ with "any two cards" ranges for both players (default GTO)
//!   4. Persist average strategy + EV trace to disk
//!
//! Future iterations: per-flop postflop solves (needs flop subgame attachment
//! at preflop terminals or as a separate orchestrated step).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use poker_gto_engine::nlhe::abstraction::rank_subsample;
use poker_gto_engine::nlhe::cards::{combo_cards, NUM_CARDS, NUM_COMBOS};
use poker_gto_engine::nlhe::flop_showdown::{compute_flop_showdown_subset};
use poker_gto_engine::nlhe::multi_cfr::MultiSolverState;
use poker_gto_engine::nlhe::preflop_cfr::{build_and_train_preflop, NodeStrategyDump};
use poker_gto_engine::nlhe::preflop_equity::compute_preflop_equity_table;
use poker_gto_engine::nlhe::preflop_tree::build_preflop_tree;
use poker_gto_engine::nlhe::storage::{save_to_file, BucketStrategy, StoredSolution};
use poker_gto_engine::nlhe::tree::{build_flop_tree_subset, count_decision_nodes, count_terminals, Node};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct PrecomputeConfig {
    stack_bb: f32,
    #[serde(default = "default_max_raises")]
    max_raises_preflop: u32,
    #[serde(default = "default_iters")]
    iterations_preflop: u32,
    #[serde(default = "default_equity_samples")]
    equity_table_samples: u32,
    output_dir: String,
    /// If > 0, after preflop solve, also run postflop solves for N sample flops
    /// per "see-flop" preflop action line. Each flop solve uses the
    /// abstraction-aware tree builder.
    #[serde(default)]
    n_flops_per_line: u32,
    #[serde(default = "default_postflop_iters")]
    iterations_postflop: u32,
    #[serde(default = "default_max_raises_postflop")]
    max_raises_postflop: u32,
}

fn default_max_raises() -> u32 { 3 }
fn default_iters() -> u32 { 200 }
fn default_equity_samples() -> u32 { 1000 }
fn default_postflop_iters() -> u32 { 100 }
fn default_max_raises_postflop() -> u32 { 1 }

fn main() {
    let path = std::env::args().nth(1).expect("usage: precompute <config.json>");
    let path = if path == "--input-file" {
        std::env::args().nth(2).expect("--input-file requires path")
    } else {
        path
    };
    let input = fs::read_to_string(&path).expect("read config");
    let config: PrecomputeConfig = serde_json::from_str(&input).expect("parse config");

    let outdir = PathBuf::from(&config.output_dir);
    fs::create_dir_all(&outdir).expect("create output dir");

    println!("=== precompute job ===");
    println!("stack: {} BB", config.stack_bb);
    println!("max_raises_preflop: {}", config.max_raises_preflop);
    println!("iterations: {}", config.iterations_preflop);
    println!("equity table samples: {}", config.equity_table_samples);
    println!("output: {}", outdir.display());

    // Step 1: Equity table
    println!("\n[1/3] Computing preflop equity table ({} samples per pair)...", config.equity_table_samples);
    let t0 = Instant::now();
    let equity = compute_preflop_equity_table(config.equity_table_samples, 0xC0FFEE);
    println!("    done in {:.1?}", t0.elapsed());

    // Step 2: Build tree
    let mut tree = build_preflop_tree(config.stack_bb, config.max_raises_preflop);
    let n_dec = count_decision_nodes(&tree);
    let n_term = count_terminals(&tree);
    println!("\n[2/3] Preflop tree: {} decision nodes, {} terminals", n_dec, n_term);

    // Step 3: Solve
    println!("\n[3/3] Running CFR+ ({} iterations)...", config.iterations_preflop);
    let any_two = vec![1.0f32; NUM_COMBOS];
    let t0 = Instant::now();
    let state = build_and_train_preflop(
        &mut tree, &equity, &any_two, &any_two,
        (config.stack_bb, config.stack_bb),
        config.iterations_preflop,
    );
    println!("    done in {:.1?}", t0.elapsed());
    println!("    hero EV (last iter): {:.3} BB", state.last_root_value());

    // Step 4: Persist (all decision nodes, full tree)
    let key = format!("preflop_{}bb_iters{}", config.stack_bb as i32, config.iterations_preflop);
    let all_dumps = state.dump_all_strategies(&tree);
    let strategy = dumps_to_storage(&all_dumps);
    let last_iter_values = state.iter_values.clone();
    let hero_value = state.last_root_value();
    let root_strategy = state.extract_root_strategy(&tree);

    let solution = StoredSolution {
        key: key.clone(),
        iterations: config.iterations_preflop,
        hero_value,
        exploitability: None,
        strategy,
        last_iter_values,
    };
    let outpath = outdir.join(format!("{}.json", key));
    save_to_file(&outpath, &solution).expect("write solution");
    println!("\nWrote solution to {}", outpath.display());

    // ===== Optional Phase: per-flop postflop solves =====
    if config.n_flops_per_line > 0 {
        println!("\n[4/4] Running per-flop postflop solves ({} flops/line)...", config.n_flops_per_line);
        run_postflop_solves(
            &config, &outdir, config.stack_bb,
        );
    }

    // Print top-line strategy (a few representative hand classes)
    println!("\n=== Sample strategies (SB open from BTN, HU) ===");
    let labels: Vec<String> = tree.actions.iter().map(|a| a.label()).collect();
    print_class_strategy(&root_strategy, "AsAh", &labels);
    print_class_strategy(&root_strategy, "KsKh", &labels);
    print_class_strategy(&root_strategy, "AsKs", &labels);
    print_class_strategy(&root_strategy, "AsKh", &labels);
    print_class_strategy(&root_strategy, "TsTh", &labels);
    print_class_strategy(&root_strategy, "7s2h", &labels);
}

/// Run a small set of postflop solves to seed the table cache.
///
/// MVP scope: pick a small number of representative flops uniformly from
/// 22100 possible 3-card boards (or use a fixed sampling seed). For each:
/// build flop tree (with rank-subset abstraction for turn/river chance),
/// solve, persist.
///
/// Ranges: for this MVP we use "any two cards" baseline ranges. A full
/// implementation would derive per-action-line ranges from the preflop
/// solution. That's a future enhancement.
fn run_postflop_solves(config: &PrecomputeConfig, outdir: &PathBuf, stack_bb: f32) {
    let n_flops = config.n_flops_per_line as usize;
    let flops = sample_representative_flops(n_flops, 0xDEC0DE);

    // Postflop pot/stacks: assume "called preflop open of 3x" → pot = 6 BB, stacks = stack_bb - 3 each
    let pot = 6.0;
    let stacks = (stack_bb - 3.0, stack_bb - 3.0);
    let any_two = vec![1.0f32; NUM_COMBOS];

    let postflop_dir = outdir.join("postflop");
    fs::create_dir_all(&postflop_dir).expect("create postflop dir");

    for (idx, flop) in flops.iter().enumerate() {
        let flop_label: String = flop.iter().map(|&c| poker_gto_engine::nlhe::cards::card_to_string(c)).collect::<Vec<_>>().join("");
        let key = format!("flop_{}_{}bb_iters{}", flop_label, stack_bb as i32, config.iterations_postflop);
        let outpath = postflop_dir.join(format!("{}.json", key));
        if outpath.exists() {
            println!("    [{}/{}] {} (cached, skipping)", idx + 1, n_flops, flop_label);
            continue;
        }

        let t0 = Instant::now();
        // Use rank-subset abstraction for both turn and river chance
        let board_mask: u64 = flop.iter().fold(0u64, |a, &c| a | (1u64 << c));
        let remaining: Vec<u8> = (0..NUM_CARDS as u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
        let turn_subset = rank_subsample(&remaining);
        let river_subset = rank_subsample(&remaining);

        // Compute showdown only for runouts the tree will actually traverse
        let table = compute_flop_showdown_subset(
            *flop, &any_two, &any_two,
            Some(&turn_subset), Some(&river_subset),
        );

        let mut tree = build_flop_tree_subset(
            *flop, pot, stacks, 0,
            config.max_raises_postflop,
            config.max_raises_postflop,
            config.max_raises_postflop,
            Some(&turn_subset),
            Some(&river_subset),
        );
        let mut state = MultiSolverState::new(&mut tree, &table, stacks, None, None);
        state.train(&tree, config.iterations_postflop);
        let elapsed = t0.elapsed();

        // Persist root strategy only for now (full per-node dump TODO)
        let root_strategy = state.extract_root_strategy(&tree);
        let mut probs_per_combo: Vec<Vec<f32>> = Vec::new();
        for (_, action_probs) in &root_strategy {
            probs_per_combo.push(action_probs.iter().map(|(_, p)| *p).collect());
        }
        let action_labels: Vec<String> = tree.actions.iter().map(|a| a.label()).collect();
        let mut strategy = HashMap::new();
        strategy.insert(0u32, BucketStrategy {
            player: tree.player_to_act,
            n_buckets: probs_per_combo.len() as u32,
            action_labels,
            probabilities: probs_per_combo,
        });

        let solution = StoredSolution {
            key: key.clone(),
            iterations: config.iterations_postflop,
            hero_value: state.last_root_value(),
            exploitability: None,
            strategy,
            last_iter_values: state.iter_values.clone(),
        };
        save_to_file(&outpath, &solution).expect("write postflop solution");
        println!("    [{}/{}] {} solved in {:.1?} (EV={:.2})",
                 idx + 1, n_flops, flop_label, elapsed, solution.hero_value);
    }
}

/// Sample N representative 3-card flops from the deck, deterministic given seed.
fn sample_representative_flops(n: usize, seed: u64) -> Vec<[u8; 3]> {
    let mut rng = seed | 1;
    let mut out: Vec<[u8; 3]> = Vec::with_capacity(n);
    let mut seen: std::collections::HashSet<[u8; 3]> = std::collections::HashSet::new();
    while out.len() < n {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let mut cards = [0u8; 3];
        for k in 0..3 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            cards[k] = (rng % NUM_CARDS as u64) as u8;
        }
        // dedup cards
        if cards[0] == cards[1] || cards[0] == cards[2] || cards[1] == cards[2] { continue; }
        // canonical: sort ascending
        cards.sort_unstable();
        if seen.insert(cards) {
            out.push(cards);
        }
    }
    out
}

fn dumps_to_storage(dumps: &HashMap<String, NodeStrategyDump>) -> HashMap<u32, BucketStrategy> {
    let mut out = HashMap::new();
    let mut sorted_paths: Vec<&String> = dumps.keys().collect();
    sorted_paths.sort();
    for (id, path) in sorted_paths.iter().enumerate() {
        let d = &dumps[path.as_str()];
        out.insert(id as u32, BucketStrategy {
            player: d.player,
            n_buckets: d.probabilities.len() as u32,
            action_labels: d.actions.clone(),
            probabilities: d.probabilities.clone(),
        });
    }
    out
}

fn print_class_strategy(
    strategy: &HashMap<u16, Vec<(String, f32)>>,
    hand_str: &str,
    _labels: &[String],
) {
    let cards: Vec<&str> = vec![&hand_str[..2], &hand_str[2..]];
    let a = poker_gto_engine::nlhe::cards::card_from_str(cards[0]);
    let b = poker_gto_engine::nlhe::cards::card_from_str(cards[1]);
    if a.is_none() || b.is_none() { return; }
    let combo = poker_gto_engine::nlhe::cards::combo_index(a.unwrap(), b.unwrap()) as u16;
    if let Some(probs) = strategy.get(&combo) {
        let mut entries: Vec<&(String, f32)> = probs.iter().filter(|(_, p)| *p > 0.02).collect();
        entries.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        let parts: Vec<String> = entries.iter().map(|(a, p)| format!("{}={:.0}%", a, p * 100.0)).collect();
        println!("  {}: {}", hand_str, parts.join("  "));
    }
}
