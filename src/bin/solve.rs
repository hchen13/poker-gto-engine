//! Standalone Rust solver CLI. Reads a JSON spec from stdin (or --input-file),
//! runs the requested solver, and writes the strategy + EV as JSON to stdout.
//!
//! Designed to be invoked from the Python `analyze_spot` wrapper or directly
//! from the `/poker` skill: input is a structured spot, output is the
//! `{action, frequency, EV, ...}` table the skill needs.
//!
//! Schema (input):
//! {
//!   "game": "river" | "turn" | "flop",
//!   "board": ["Ad","Kh","7s","3c","2d"],     // 5 / 4 / 3 cards
//!   "pot": 100.0,
//!   "stacks": [200.0, 200.0],
//!   "first_to_act": 0,                       // 0 = hero (player 0), 1 = villain
//!   "hero_range": "AhAc, KsKc, QsJs",
//!   "villain_range": "JcJd, TcTd, AcQc",
//!   "iterations": 500,
//!   "max_raises": 2,
//!   "turn_max_raises": 2,                    // for flop
//!   "river_max_raises": 2,                   // for turn / flop
//!   "compute_exploitability": true
//! }

use std::io::{self, Read, Write};

use poker_gto_engine::nlhe::best_response::{
    best_response_value_multi, best_response_value_river, exploitability_multi, exploitability_river,
};
use poker_gto_engine::nlhe::cache::{cache_key_for, PersistentCache};
use poker_gto_engine::nlhe::cards::{card_from_str, card_to_string, combo_cards};
use poker_gto_engine::nlhe::cfr::{Solver, SolverState};
use poker_gto_engine::nlhe::flop_showdown::compute_flop_showdown;
use poker_gto_engine::nlhe::multi_cfr::{MultiBoardLookup, MultiSolverState};
use poker_gto_engine::nlhe::range_parser::parse_range;
use poker_gto_engine::nlhe::showdown::{compute_showdown_table, ShowdownTable};
use poker_gto_engine::nlhe::tree::{build_flop_tree, build_river_tree, build_turn_tree, Node};
use poker_gto_engine::nlhe::turn_showdown::compute_turn_showdown;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize, Serialize)]
struct SolveSpec {
    game: String,
    board: Vec<String>,
    pot: f32,
    stacks: [f32; 2],
    #[serde(default)]
    first_to_act: i8,
    hero_range: String,
    villain_range: String,
    #[serde(default = "default_iters")]
    iterations: u32,
    #[serde(default = "default_max_raises")]
    max_raises: u32,
    #[serde(default = "default_max_raises")]
    turn_max_raises: u32,
    #[serde(default = "default_max_raises")]
    river_max_raises: u32,
    #[serde(default)]
    compute_exploitability: bool,
    #[serde(default)]
    all_strategies: bool,
}

fn default_iters() -> u32 { 200 }
fn default_max_raises() -> u32 { 1 }

#[derive(Debug, Serialize)]
struct SolveOutput {
    game: String,
    board: Vec<String>,
    pot: f32,
    stacks: [f32; 2],
    first_to_act: i8,
    iterations: u32,
    hero_value: f32,
    n_hero_combos: usize,
    n_villain_combos: usize,
    hero_combos: Vec<String>,
    villain_combos: Vec<String>,
    /// {hero_combo_label: {action_label: probability}}
    strategy: HashMap<String, HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exploitability: Option<ExploitabilityInfo>,
    /// All decision node strategies indexed by action path from root.
    /// Key format: "action1 > action2 > ..." (empty string = root).
    #[serde(skip_serializing_if = "Option::is_none")]
    all_strategies: Option<HashMap<String, NodeStrategy>>,
}

#[derive(Debug, Serialize)]
struct NodeStrategy {
    player: i8,
    pot: f32,
    stacks: [f32; 2],
    to_call: f32,
    /// Aggregated strategy across all combos at this player (averaged).
    /// For per-combo detail, the consumer can re-solve with combo-specific output.
    actions: Vec<String>,
    /// [bucket][action] = probability
    probabilities: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct ExploitabilityInfo {
    br_hero: f32,
    br_villain: f32,
    initial_pot: f32,
    exploitability: f32,
}

fn combo_label(idx: u16) -> String {
    let (a, b) = combo_cards(idx as usize);
    let a_str = poker_gto_engine::nlhe::cards::card_to_string(a);
    let b_str = poker_gto_engine::nlhe::cards::card_to_string(b);
    format!("{}{}", a_str, b_str)
}

fn main() {
    let mut input = String::new();
    let arg_input = std::env::args().nth(1);
    match arg_input.as_deref() {
        Some("--input-file") => {
            let path = std::env::args().nth(2).expect("--input-file requires path");
            input = std::fs::read_to_string(path).expect("read input");
        }
        Some(p) if p == "-" || p.is_empty() => {
            io::stdin().read_to_string(&mut input).expect("stdin");
        }
        Some(other) => {
            input = std::fs::read_to_string(other).expect("read input");
        }
        None => {
            io::stdin().read_to_string(&mut input).expect("stdin");
        }
    }
    let spec: SolveSpec = serde_json::from_str(&input).expect("parse JSON spec");

    // Cache layer (V1: simple key→JSON file). Skipped if all_strategies is requested
    // (per-spot dump, less useful to cache the bigger blob).
    let cache_dir = std::env::var("POKER_CACHE_DIR")
        .ok()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            // Default: project_root/precompute_out/by_spec
            let exe = std::env::current_exe().unwrap_or_default();
            // exe = .../target/release/solve  → up 3 = project root
            let mut p = exe;
            for _ in 0..3 { p.pop(); }
            p.join("precompute_out").join("by_spec")
        });
    let cache = PersistentCache::new(&cache_dir).ok();
    let canonical = serde_json::to_string(&spec).unwrap_or_default();
    let key = cache_key_for(&canonical);

    if let Some(c) = &cache {
        if let Some(cached_json) = c.get(&key) {
            // Cache hit — just emit cached JSON
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            handle.write_all(cached_json.as_bytes()).ok();
            if !cached_json.ends_with('\n') {
                handle.write_all(b"\n").ok();
            }
            return;
        }
    }

    let board_cards: Vec<u8> = spec.board.iter()
        .map(|s| card_from_str(s).expect("invalid card"))
        .collect();

    let hero_range = parse_range(&spec.hero_range).expect("hero_range parse");
    let villain_range = parse_range(&spec.villain_range).expect("villain_range parse");
    let stacks = (spec.stacks[0], spec.stacks[1]);

    let output = match spec.game.as_str() {
        "river" => {
            assert_eq!(board_cards.len(), 5, "river requires 5 board cards");
            let mut board5 = [0u8; 5];
            board5.copy_from_slice(&board_cards);
            let table = compute_showdown_table(board5, &hero_range, &villain_range);
            let mut root = build_river_tree(spec.pot, stacks, spec.first_to_act, spec.max_raises);
            let mut solver = Solver::new(&mut root, &table, stacks, None, None);
            solver.state.train(&root, spec.iterations);
            let strategy = solver.state.extract_root_strategy(&root);
            let exp = if spec.compute_exploitability {
                let br_h = best_response_value_river(&solver.state, &root, 0);
                let br_v = best_response_value_river(&solver.state, &root, 1);
                Some(ExploitabilityInfo {
                    br_hero: br_h, br_villain: br_v,
                    initial_pot: root.pot,
                    exploitability: br_h + br_v - root.pot,
                })
            } else { None };
            let all = if spec.all_strategies { Some(dump_all_strategies_river(&solver.state, &root)) } else { None };
            build_output(spec, table.n_hero, table.n_villain, solver.state.last_root_value(), strategy, exp, all, &table.hero_combos, &table.villain_combos)
        }
        "turn" => {
            assert_eq!(board_cards.len(), 4, "turn requires 4 board cards");
            let mut board4 = [0u8; 4];
            board4.copy_from_slice(&board_cards);
            let table = compute_turn_showdown(board4, &hero_range, &villain_range);
            let mut root = build_turn_tree(board4, spec.pot, stacks, spec.first_to_act, spec.max_raises, spec.river_max_raises);
            let mut state = MultiSolverState::new(&mut root, &table, stacks, None, None);
            state.train(&root, spec.iterations);
            let strategy = state.extract_root_strategy(&root);
            let exp = if spec.compute_exploitability {
                let br_h = best_response_value_multi(&state, &root, 0);
                let br_v = best_response_value_multi(&state, &root, 1);
                Some(ExploitabilityInfo {
                    br_hero: br_h, br_villain: br_v,
                    initial_pot: root.pot,
                    exploitability: br_h + br_v - root.pot,
                })
            } else { None };
            let all = if spec.all_strategies { Some(dump_all_strategies_multi(&state, &root)) } else { None };
            build_output(spec, table.n_hero, table.n_villain, state.last_root_value(), strategy, exp, all, &table.hero_combos, &table.villain_combos)
        }
        "flop" => {
            assert_eq!(board_cards.len(), 3, "flop requires 3 board cards");
            let mut board3 = [0u8; 3];
            board3.copy_from_slice(&board_cards);
            let table = compute_flop_showdown(board3, &hero_range, &villain_range);
            let mut root = build_flop_tree(board3, spec.pot, stacks, spec.first_to_act, spec.max_raises, spec.turn_max_raises, spec.river_max_raises);
            let mut state = MultiSolverState::new(&mut root, &table, stacks, None, None);
            state.train(&root, spec.iterations);
            let strategy = state.extract_root_strategy(&root);
            let exp = if spec.compute_exploitability {
                let br_h = best_response_value_multi(&state, &root, 0);
                let br_v = best_response_value_multi(&state, &root, 1);
                Some(ExploitabilityInfo {
                    br_hero: br_h, br_villain: br_v,
                    initial_pot: root.pot,
                    exploitability: br_h + br_v - root.pot,
                })
            } else { None };
            let all = if spec.all_strategies { Some(dump_all_strategies_multi(&state, &root)) } else { None };
            build_output(spec, table.n_hero, table.n_villain, state.last_root_value(), strategy, exp, all, &table.hero_combos, &table.villain_combos)
        }
        other => panic!("unknown game: {}", other),
    };

    let json = serde_json::to_string_pretty(&output).expect("serialize");
    // Write to cache (best-effort)
    if let Some(c) = &cache {
        let _ = c.put(&key, &json);
    }
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    writeln!(handle, "{}", json).unwrap();
}

fn build_output(
    spec: SolveSpec,
    n_hero: usize,
    n_villain: usize,
    hero_value: f32,
    strategy: HashMap<u16, Vec<(String, f32)>>,
    exp: Option<ExploitabilityInfo>,
    all: Option<HashMap<String, NodeStrategy>>,
    hero_combo_indices: &[u16],
    villain_combo_indices: &[u16],
) -> SolveOutput {
    let hero_combos: Vec<String> = hero_combo_indices.iter().map(|&i| combo_label(i)).collect();
    let villain_combos: Vec<String> = villain_combo_indices.iter().map(|&i| combo_label(i)).collect();
    let strategy_labeled: HashMap<String, HashMap<String, f32>> = strategy.into_iter()
        .map(|(combo_idx, action_probs)| {
            let label = combo_label(combo_idx);
            let probs: HashMap<String, f32> = action_probs.into_iter().collect();
            (label, probs)
        })
        .collect();
    SolveOutput {
        game: spec.game,
        board: spec.board,
        pot: spec.pot,
        stacks: spec.stacks,
        first_to_act: spec.first_to_act,
        iterations: spec.iterations,
        hero_value,
        n_hero_combos: n_hero,
        n_villain_combos: n_villain,
        hero_combos,
        villain_combos,
        strategy: strategy_labeled,
        exploitability: exp,
        all_strategies: all,
    }
}

/// Walk the tree, dump per-decision-node aggregated strategy keyed by action path.
fn dump_all_strategies_river(state: &SolverState, root: &Node) -> HashMap<String, NodeStrategy> {
    let mut out = HashMap::new();
    fn walk(state: &SolverState, node: &Node, path: String, out: &mut HashMap<String, NodeStrategy>) {
        if !node.is_terminal && !node.is_chance {
            let nid = node.node_id.get() as usize;
            let na = node.actions.len();
            let strat = &state.strategy_sum[nid];
            let probabilities: Vec<Vec<f32>> = strat.iter().map(|row| {
                let total: f32 = row.iter().sum();
                if total > 0.0 { row.iter().map(|v| v / total).collect() }
                else { vec![1.0 / (na as f32); na] }
            }).collect();
            let actions: Vec<String> = node.actions.iter().map(|a| a.label()).collect();
            out.insert(path.clone(), NodeStrategy {
                player: node.player_to_act,
                pot: node.pot,
                stacks: [node.stacks.0, node.stacks.1],
                to_call: node.to_call,
                actions,
                probabilities,
            });
        }
        for (a, child) in node.children.iter().enumerate() {
            let part = if node.is_chance {
                format!("chance({})", card_to_string(node.chance_cards[a]))
            } else {
                node.actions.get(a).map(|x| x.label()).unwrap_or_else(|| format!("a{}", a))
            };
            let new_path = if path.is_empty() { part } else { format!("{} > {}", path, part) };
            walk(state, child, new_path, out);
        }
    }
    walk(state, root, String::new(), &mut out);
    out
}

fn dump_all_strategies_multi<L: MultiBoardLookup>(state: &MultiSolverState<L>, root: &Node) -> HashMap<String, NodeStrategy> {
    let mut out = HashMap::new();
    fn walk<L: MultiBoardLookup>(state: &MultiSolverState<L>, node: &Node, path: String, out: &mut HashMap<String, NodeStrategy>) {
        if !node.is_terminal && !node.is_chance {
            let nid = node.node_id.get() as usize;
            let na = node.actions.len();
            let strat = &state.strategy_sum[nid];
            let probabilities: Vec<Vec<f32>> = strat.iter().map(|row| {
                let total: f32 = row.iter().sum();
                if total > 0.0 { row.iter().map(|v| v / total).collect() }
                else { vec![1.0 / (na as f32); na] }
            }).collect();
            let actions: Vec<String> = node.actions.iter().map(|a| a.label()).collect();
            out.insert(path.clone(), NodeStrategy {
                player: node.player_to_act,
                pot: node.pot,
                stacks: [node.stacks.0, node.stacks.1],
                to_call: node.to_call,
                actions,
                probabilities,
            });
        }
        for (a, child) in node.children.iter().enumerate() {
            let part = if node.is_chance {
                format!("chance({})", card_to_string(node.chance_cards[a]))
            } else {
                node.actions.get(a).map(|x| x.label()).unwrap_or_else(|| format!("a{}", a))
            };
            let new_path = if path.is_empty() { part } else { format!("{} > {}", path, part) };
            walk(state, child, new_path, out);
        }
    }
    walk(state, root, String::new(), &mut out);
    out
}
