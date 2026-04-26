//! Bucketed precompute orchestrator with live progress reporting.
//!
//! Writes a heartbeat file at `<output_dir>/progress.json` every solve
//! completion (and at periodic intervals during long phases). Use
//! `tail -f progress.log` or read `progress.json` to monitor.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use poker_gto_engine::nlhe::abstraction::rank_subsample;
use poker_gto_engine::nlhe::bucket_cfr_multi::{
    build_flop_equity_store, solve_multi_bucketed,
};
use poker_gto_engine::nlhe::bucketing::{bucket_by_ehs, river_ehs};
use poker_gto_engine::nlhe::cards::{card_to_string, NUM_CARDS, NUM_COMBOS};
use poker_gto_engine::nlhe::tree::build_flop_tree_subset;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct Config {
    stack_bb: f32,
    /// Hero range (range_parser syntax). Use "AA-22, A2s+, K2s+, ..." for "any two cards-lite".
    hero_range: String,
    villain_range: String,
    n_flops: usize,
    /// K for combo bucketing.
    k_buckets: usize,
    iterations: u32,
    max_raises: u32,
    output_dir: String,
    /// Optional: only solve flops whose 3-card-string starts with this prefix.
    /// Useful for parallel sharding across multiple processes.
    #[serde(default)]
    flop_filter_prefix: Option<String>,
}

#[derive(Debug, Serialize)]
struct Progress {
    state: String,
    completed: usize,
    total: usize,
    current: String,
    started_at_unix: u64,
    elapsed_sec: f64,
    estimated_remaining_sec: Option<f64>,
    last_flop_time_sec: Option<f64>,
    last_flop_hero_value: Option<f32>,
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs()).unwrap_or(0)
}

fn write_progress(path: &PathBuf, p: &Progress) {
    if let Ok(json) = serde_json::to_string_pretty(p) {
        let _ = fs::write(path, json);
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: precompute_bucketed <config.json>");
    let path = if path == "--input-file" {
        std::env::args().nth(2).expect("--input-file requires path")
    } else {
        path
    };
    let input = fs::read_to_string(&path).expect("read config");
    let config: Config = serde_json::from_str(&input).expect("parse config");

    let outdir = PathBuf::from(&config.output_dir);
    fs::create_dir_all(&outdir).expect("create output dir");
    let progress_path = outdir.join("progress.json");
    let log_path = outdir.join("progress.log");

    let mut log = fs::OpenOptions::new()
        .create(true).append(true)
        .open(&log_path).expect("open log");

    let log_line = |log: &mut fs::File, s: &str| {
        let line = format!("[{}] {}\n", chrono_now(), s);
        let _ = log.write_all(line.as_bytes());
        print!("{}", line);
    };

    log_line(&mut log, &format!("=== precompute_bucketed start ==="));
    log_line(&mut log, &format!("stack: {} BB", config.stack_bb));
    log_line(&mut log, &format!("ranges: hero={:?} villain={:?}", config.hero_range, config.villain_range));
    log_line(&mut log, &format!("K_buckets={}, n_flops={}, iters={}, max_raises={}",
        config.k_buckets, config.n_flops, config.iterations, config.max_raises));
    log_line(&mut log, &format!("output: {}", outdir.display()));

    let started = Instant::now();
    let started_unix = unix_now();

    // Sample N flops deterministically
    let flops = sample_flops(config.n_flops, 0xDEC0DE);
    let flops: Vec<[u8; 3]> = if let Some(prefix) = &config.flop_filter_prefix {
        flops.into_iter().filter(|f| flop_label(f).starts_with(prefix)).collect()
    } else { flops };
    let total = flops.len();
    log_line(&mut log, &format!("Sampled {} flops to solve", total));

    let hero_range_v = poker_gto_engine::nlhe::range_parser::parse_range(&config.hero_range)
        .expect("hero range parse");
    let villain_range_v = poker_gto_engine::nlhe::range_parser::parse_range(&config.villain_range)
        .expect("villain range parse");

    write_progress(&progress_path, &Progress {
        state: "starting".into(),
        completed: 0, total,
        current: "".into(),
        started_at_unix: started_unix,
        elapsed_sec: 0.0,
        estimated_remaining_sec: None,
        last_flop_time_sec: None,
        last_flop_hero_value: None,
    });

    let mut completed = 0usize;
    let mut times: Vec<f64> = Vec::new();
    let mut last_value = 0.0f32;

    for (idx, flop) in flops.iter().enumerate() {
        let label = flop_label(flop);
        write_progress(&progress_path, &Progress {
            state: "solving".into(),
            completed,
            total,
            current: label.clone(),
            started_at_unix: started_unix,
            elapsed_sec: started.elapsed().as_secs_f64(),
            estimated_remaining_sec: estimate_remaining(&times, total - completed),
            last_flop_time_sec: times.last().copied(),
            last_flop_hero_value: Some(last_value),
        });

        let outpath = outdir.join(format!("flop_{}.json", label));
        if outpath.exists() {
            log_line(&mut log, &format!("[{}/{}] {} (cached, skipping)", idx + 1, total, label));
            completed += 1;
            continue;
        }

        let t_flop = Instant::now();

        // Bucket the ranges using river-EHS at a representative 5-card extension
        // (for now: append a dummy river card to the flop+turn-rep).
        let board_mask: u64 = flop.iter().fold(0u64, |a, &c| a | (1u64 << c));
        let remaining: Vec<u8> = (0..NUM_CARDS as u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
        let turn_subset = rank_subsample(&remaining);
        let river_subset = rank_subsample(&remaining);

        // Use a representative 5-card board for bucketing (flop + median turn + median river)
        let mut rep5 = [0u8; 5];
        rep5[..3].copy_from_slice(flop);
        rep5[3] = turn_subset[turn_subset.len() / 2];
        rep5[4] = river_subset[(river_subset.len() / 2 + 1) % river_subset.len()];
        if rep5[4] == rep5[3] { rep5[4] = river_subset[0]; }

        let ehs = river_ehs(&rep5);
        let hero_b = bucket_by_ehs(&ehs, &hero_range_v, &rep5, config.k_buckets);
        let villain_b = bucket_by_ehs(&ehs, &villain_range_v, &rep5, config.k_buckets);

        let store = build_flop_equity_store(
            *flop, &hero_b, &villain_b, Some(&turn_subset), Some(&river_subset),
        );

        let mut tree = build_flop_tree_subset(
            *flop, 6.0, (config.stack_bb - 3.0, config.stack_bb - 3.0), 0,
            config.max_raises, config.max_raises, config.max_raises,
            Some(&turn_subset), Some(&river_subset),
        );

        let result = solve_multi_bucketed(
            &mut tree, &hero_b, &villain_b, &store,
            (config.stack_bb - 3.0, config.stack_bb - 3.0),
            config.iterations,
        );

        let elapsed = t_flop.elapsed();
        times.push(elapsed.as_secs_f64());
        last_value = result.hero_value;

        // Persist
        let solution = SolutionFile {
            flop_label: label.clone(),
            stack_bb: config.stack_bb,
            k_buckets: config.k_buckets,
            iterations: config.iterations,
            hero_value: result.hero_value,
            root_strategy: result.root_strategy,
            action_labels: result.action_labels,
            elapsed_sec: elapsed.as_secs_f64(),
        };
        let _ = fs::write(&outpath, serde_json::to_string_pretty(&solution).unwrap_or_default());
        completed += 1;

        log_line(&mut log, &format!("[{}/{}] {} solved in {:.1}s (EV={:.2}, ETA {})",
            idx + 1, total, label, elapsed.as_secs_f64(), result.hero_value,
            format_eta(estimate_remaining(&times, total - completed))));

        write_progress(&progress_path, &Progress {
            state: "solving".into(),
            completed,
            total,
            current: "".into(),
            started_at_unix: started_unix,
            elapsed_sec: started.elapsed().as_secs_f64(),
            estimated_remaining_sec: estimate_remaining(&times, total - completed),
            last_flop_time_sec: Some(elapsed.as_secs_f64()),
            last_flop_hero_value: Some(result.hero_value),
        });
    }

    let total_elapsed = started.elapsed();
    write_progress(&progress_path, &Progress {
        state: "done".into(),
        completed,
        total,
        current: "".into(),
        started_at_unix: started_unix,
        elapsed_sec: total_elapsed.as_secs_f64(),
        estimated_remaining_sec: Some(0.0),
        last_flop_time_sec: times.last().copied(),
        last_flop_hero_value: Some(last_value),
    });
    log_line(&mut log, &format!("=== done {} flops in {:.1}s ({:.1}s avg) ===",
        completed, total_elapsed.as_secs_f64(),
        if !times.is_empty() { times.iter().sum::<f64>() / times.len() as f64 } else { 0.0 }));
}

#[derive(Debug, Serialize)]
struct SolutionFile {
    flop_label: String,
    stack_bb: f32,
    k_buckets: usize,
    iterations: u32,
    hero_value: f32,
    root_strategy: Vec<Vec<f32>>,
    action_labels: Vec<String>,
    elapsed_sec: f64,
}

fn flop_label(flop: &[u8; 3]) -> String {
    flop.iter().map(|&c| card_to_string(c)).collect::<Vec<_>>().join("")
}

fn sample_flops(n: usize, seed: u64) -> Vec<[u8; 3]> {
    let mut rng = seed | 1;
    let mut out = Vec::with_capacity(n);
    let mut seen = std::collections::HashSet::new();
    while out.len() < n {
        let mut cards = [0u8; 3];
        for k in 0..3 {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            cards[k] = (rng % NUM_CARDS as u64) as u8;
        }
        if cards[0] == cards[1] || cards[0] == cards[2] || cards[1] == cards[2] { continue; }
        cards.sort_unstable();
        if seen.insert(cards) {
            out.push(cards);
        }
    }
    out
}

fn estimate_remaining(times: &[f64], remaining: usize) -> Option<f64> {
    if times.is_empty() { return None; }
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    Some(avg * remaining as f64)
}

fn format_eta(eta: Option<f64>) -> String {
    match eta {
        None => "?".into(),
        Some(s) if s < 60.0 => format!("{:.0}s", s),
        Some(s) if s < 3600.0 => format!("{:.1}m", s / 60.0),
        Some(s) => format!("{:.1}h", s / 3600.0),
    }
}

fn chrono_now() -> String {
    use std::time::SystemTime;
    let duration = SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or(Duration::ZERO);
    let secs = duration.as_secs();
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
}
