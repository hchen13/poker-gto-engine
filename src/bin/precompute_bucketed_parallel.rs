//! Parallel version of precompute_bucketed — uses std::thread to solve N
//! flops concurrently across `num_workers` cores. Per-flop solve is itself
//! single-threaded; the parallelism is across distinct flops.
//!
//! Shared state:
//! - Progress (Mutex-guarded): completed count, ETA calculation
//! - Output files: one per flop, so no contention
//! - Log file: Mutex-guarded

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use poker_gto_engine::nlhe::abstraction::rank_subsample;
use poker_gto_engine::nlhe::bucket_cfr_multi::build_flop_equity_store;
use poker_gto_engine::nlhe::bucket_cfr_flat::solve_multi_bucketed_flat;
use poker_gto_engine::nlhe::bucketing::{bucket_by_ehs, river_ehs};
use poker_gto_engine::nlhe::cards::{card_to_string, combo_label, NUM_CARDS, NUM_COMBOS};
use poker_gto_engine::nlhe::tree::build_flop_tree_subset;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct Config {
    stack_bb: f32,
    /// Option A: direct hero/villain range strings.
    #[serde(default)]
    hero_range: Option<String>,
    #[serde(default)]
    villain_range: Option<String>,
    #[serde(default)]
    pot_bb: Option<f32>,
    /// Option B: load ranges from a ranges file + action_line key.
    #[serde(default)]
    ranges_file: Option<String>,
    #[serde(default)]
    action_line: Option<String>,
    n_flops: usize,
    k_buckets: usize,
    iterations: u32,
    max_raises: u32,
    output_dir: String,
    #[serde(default = "default_workers")]
    num_workers: usize,
}

fn default_workers() -> usize { 8 }

#[derive(Debug, Deserialize)]
struct RangesFile {
    action_lines: std::collections::HashMap<String, ActionLine>,
}

#[derive(Debug, Deserialize)]
struct ActionLine {
    sb_range: String,
    bb_range: String,
    pot_bb: f32,
    #[serde(default)]
    description: String,
}

fn resolve_ranges(config: &Config) -> (String, String, f32, String) {
    // Returns (hero_range_str, villain_range_str, pot_bb, label)
    if let (Some(hr), Some(vr)) = (&config.hero_range, &config.villain_range) {
        let pot = config.pot_bb.unwrap_or(6.0);
        return (hr.clone(), vr.clone(), pot, "custom".into());
    }
    let file_path = config.ranges_file.as_ref().expect("either hero/villain_range or ranges_file required");
    let line_key = config.action_line.as_ref().expect("action_line required when using ranges_file");
    let raw = fs::read_to_string(file_path).expect("read ranges file");
    let ranges: RangesFile = serde_json::from_str(&raw).expect("parse ranges file");
    let line = ranges.action_lines.get(line_key).unwrap_or_else(|| {
        let keys: Vec<&String> = ranges.action_lines.keys().collect();
        panic!("action_line '{}' not found. Available: {:?}", line_key, keys);
    });
    // By convention: SB = hero (player 0, button, acts first pre-flop but IP post-flop)
    // For postflop solve: OOP acts first = BB = player 0 ... hmm this depends on convention.
    // Existing precompute uses first_to_act=0 at root = hero, pot=6BB suggesting SR-called postflop
    // with hero = OOP = BB. Keep that convention: hero = BB (OOP) = sb side solved for.
    // Wait — that's confusing. Let me just say: "hero" in precompute is whichever side acts first
    // postflop. In HU postflop, BB acts first (OOP). So hero=BB-range, villain=SB-range.
    (line.bb_range.clone(), line.sb_range.clone(), line.pot_bb, line_key.clone())
}

#[derive(Debug, Serialize, Clone)]
struct Progress {
    state: String,
    completed: usize,
    total: usize,
    num_workers: usize,
    currently_solving: Vec<String>,
    started_at_unix: u64,
    elapsed_sec: f64,
    estimated_remaining_sec: Option<f64>,
    avg_flop_time_sec: Option<f64>,
    last_flop_hero_value: Option<f32>,
}

struct ProgressTracker {
    inner: Mutex<Progress>,
    times: Mutex<Vec<f64>>,
    path: PathBuf,
}

impl ProgressTracker {
    fn new(total: usize, num_workers: usize, path: PathBuf) -> Self {
        let p = Progress {
            state: "starting".into(),
            completed: 0,
            total,
            num_workers,
            currently_solving: Vec::new(),
            started_at_unix: unix_now(),
            elapsed_sec: 0.0,
            estimated_remaining_sec: None,
            avg_flop_time_sec: None,
            last_flop_hero_value: None,
        };
        Self { inner: Mutex::new(p), times: Mutex::new(Vec::new()), path }
    }

    fn mark_start(&self, label: &str) {
        let mut p = self.inner.lock().unwrap();
        p.state = "solving".into();
        p.currently_solving.push(label.into());
        self.write(&p);
    }

    fn mark_done(&self, label: &str, started: Instant, elapsed_flop_sec: f64, hero_value: f32, total_elapsed: f64) {
        {
            let mut t = self.times.lock().unwrap();
            t.push(elapsed_flop_sec);
        }
        let avg = {
            let t = self.times.lock().unwrap();
            t.iter().sum::<f64>() / t.len() as f64
        };
        let mut p = self.inner.lock().unwrap();
        p.currently_solving.retain(|c| c != label);
        p.completed += 1;
        p.elapsed_sec = total_elapsed;
        p.last_flop_hero_value = Some(hero_value);
        p.avg_flop_time_sec = Some(avg);
        // ETA assumes remaining flops spread across workers at current avg rate
        let remaining = p.total.saturating_sub(p.completed);
        if remaining > 0 && p.num_workers > 0 {
            p.estimated_remaining_sec = Some(avg * remaining as f64 / p.num_workers as f64);
        } else {
            p.estimated_remaining_sec = Some(0.0);
        }
        self.write(&p);
        let _ = started;
    }

    fn mark_cached(&self, label: &str) {
        let mut p = self.inner.lock().unwrap();
        p.currently_solving.retain(|c| c != label);
        p.completed += 1;
        self.write(&p);
    }

    fn mark_done_all(&self, total_elapsed: f64) {
        let mut p = self.inner.lock().unwrap();
        p.state = "done".into();
        p.elapsed_sec = total_elapsed;
        p.estimated_remaining_sec = Some(0.0);
        self.write(&p);
    }

    fn write(&self, p: &Progress) {
        if let Ok(json) = serde_json::to_string_pretty(p) {
            let _ = fs::write(&self.path, json);
        }
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs()).unwrap_or(0)
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: precompute_bucketed_parallel <config.json>");
    let path = if path == "--input-file" {
        std::env::args().nth(2).expect("--input-file requires path")
    } else { path };
    let input = fs::read_to_string(&path).expect("read config");
    let config: Config = serde_json::from_str(&input).expect("parse config");

    let outdir = PathBuf::from(&config.output_dir);
    fs::create_dir_all(&outdir).expect("create output dir");
    let progress_path = outdir.join("progress.json");
    let log_path = outdir.join("progress.log");

    let log = Arc::new(Mutex::new(
        fs::OpenOptions::new().create(true).append(true).open(&log_path).expect("open log")
    ));
    let log_line = |s: String| {
        let line = format!("[{}] {}\n", chrono_now(), s);
        print!("{}", line);
        if let Ok(mut f) = log.lock() { let _ = f.write_all(line.as_bytes()); }
    };

    log_line(format!("=== precompute_bucketed_parallel start ==="));
    let (hero_range_str, villain_range_str, pot_bb, line_label) = resolve_ranges(&config);
    log_line(format!("stack={}BB, line={}, pot={}BB, K={}, n_flops={}, iters={}, workers={}",
        config.stack_bb, line_label, pot_bb, config.k_buckets,
        config.n_flops, config.iterations, config.num_workers));
    log_line(format!("output: {}", outdir.display()));

    let started = Instant::now();
    // Canonical iso-class enumeration: all 1755 unique flop classes, one
    // representative each. Covers every real flop exactly once under suit
    // isomorphism. Falls back to random sampling only if n_flops < 1755
    // (legacy/debug configs).
    let flops: Vec<[u8; 3]> = if config.n_flops >= 1755 {
        poker_gto_engine::nlhe::abstraction::enumerate_canonical_flops()
    } else {
        sample_flops(config.n_flops, 0xDEC0DE)
    };
    log_line(format!(
        "flop set: {} {} flops",
        if config.n_flops >= 1755 { "canonical" } else { "sampled" },
        flops.len()
    ));

    let hero_range = Arc::new(poker_gto_engine::nlhe::range_parser::parse_range(&hero_range_str)
        .expect("hero range parse"));
    let villain_range = Arc::new(poker_gto_engine::nlhe::range_parser::parse_range(&villain_range_str)
        .expect("villain range parse"));
    let hero_n: usize = hero_range.iter().filter(|&&w| w > 0.0).count();
    let villain_n: usize = villain_range.iter().filter(|&&w| w > 0.0).count();
    log_line(format!("ranges: hero={} combos, villain={} combos", hero_n, villain_n));

    let tracker = Arc::new(ProgressTracker::new(flops.len(), config.num_workers, progress_path));

    // Partition flops into chunks per worker (round-robin for load balance)
    let mut worker_flops: Vec<Vec<[u8; 3]>> = (0..config.num_workers).map(|_| Vec::new()).collect();
    for (i, flop) in flops.iter().enumerate() {
        worker_flops[i % config.num_workers].push(*flop);
    }

    let stack_bb = config.stack_bb;
    let pot_chips = pot_bb;
    let k_buckets = config.k_buckets;
    let iterations = config.iterations;
    let max_raises = config.max_raises;
    // Post-preflop remaining stack = stack_bb - (each player's preflop contribution)
    let per_side_invested = pot_bb / 2.0;
    let postflop_stack = stack_bb - per_side_invested;
    let outdir_arc = Arc::new(outdir.clone());
    let log_arc = log.clone();

    let handles: Vec<_> = worker_flops.into_iter().enumerate().map(|(wid, my_flops)| {
        let hero_range = hero_range.clone();
        let villain_range = villain_range.clone();
        let tracker = tracker.clone();
        let outdir = outdir_arc.clone();
        let log = log_arc.clone();
        let start_time = started;

        thread::spawn(move || {
            for flop in my_flops {
                let label = flop_label(&flop);
                let outpath = outdir.join(format!("flop_{}.json", label));
                if outpath.exists() {
                    tracker.mark_cached(&label);
                    continue;
                }

                tracker.mark_start(&label);
                let t0 = Instant::now();

                // Bucketing prep
                let board_mask: u64 = flop.iter().fold(0u64, |a, &c| a | (1u64 << c));
                let remaining: Vec<u8> = (0..NUM_CARDS as u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
                let turn_subset = rank_subsample(&remaining);
                let river_subset = rank_subsample(&remaining);
                let mut rep5 = [0u8; 5];
                rep5[..3].copy_from_slice(&flop);
                rep5[3] = turn_subset[turn_subset.len() / 2];
                rep5[4] = river_subset[(river_subset.len() / 2 + 1) % river_subset.len()];
                if rep5[4] == rep5[3] { rep5[4] = river_subset[0]; }

                let t_ehs = Instant::now();
                let ehs = river_ehs(&rep5);
                let t_ehs_ms = t_ehs.elapsed().as_millis();

                let t_bucket = Instant::now();
                let hero_b = bucket_by_ehs(&ehs, &*hero_range, &rep5, k_buckets);
                let villain_b = bucket_by_ehs(&ehs, &*villain_range, &rep5, k_buckets);
                let t_bucket_ms = t_bucket.elapsed().as_millis();

                let t_store = Instant::now();
                let store = build_flop_equity_store(
                    flop, &hero_b, &villain_b, Some(&turn_subset), Some(&river_subset),
                );
                let t_store_ms = t_store.elapsed().as_millis();

                let t_tree = Instant::now();
                let mut tree = build_flop_tree_subset(
                    flop, pot_chips, (postflop_stack, postflop_stack), 0,
                    max_raises, max_raises, max_raises,
                    Some(&turn_subset), Some(&river_subset),
                );
                let t_tree_ms = t_tree.elapsed().as_millis();

                let t_cfr = Instant::now();
                let result = solve_multi_bucketed_flat(
                    &mut tree, &hero_b, &villain_b, &store,
                    (postflop_stack, postflop_stack),
                    iterations,
                );
                let t_cfr_ms = t_cfr.elapsed().as_millis();

                let elapsed = t0.elapsed();
                let _prof_line = format!(
                    "  profile: ehs={}ms bucket={}ms store={}ms tree={}ms cfr={}ms total={}ms",
                    t_ehs_ms, t_bucket_ms, t_store_ms, t_tree_ms, t_cfr_ms, elapsed.as_millis()
                );
                // Build bucket_hands: for each bucket, collect hand-type labels
                // weighted by combo weight, dedup, sort by total weight desc.
                let bucket_hands: Vec<Vec<String>> = (0..k_buckets).map(|b| {
                    let mut weight_map: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
                    for &(combo_u16, w) in &hero_b.combos_in_bucket[b] {
                        let label = combo_label(combo_u16 as usize);
                        *weight_map.entry(label).or_default() += w;
                    }
                    let mut pairs: Vec<(String, f32)> = weight_map.into_iter().collect();
                    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    pairs.into_iter().map(|(lbl, _)| lbl).collect()
                }).collect();

                // Build bucket_ehs_range: min/max EHS per bucket from ehs array
                let bucket_ehs_range: Vec<[f32; 2]> = (0..k_buckets).map(|b| {
                    let mut mn = f32::INFINITY;
                    let mut mx = f32::NEG_INFINITY;
                    for &(combo_u16, _) in &hero_b.combos_in_bucket[b] {
                        let e = ehs[combo_u16 as usize];
                        if e < mn { mn = e; }
                        if e > mx { mx = e; }
                    }
                    if mn == f32::INFINITY { [0.0, 0.0] } else { [mn, mx] }
                }).collect();

                let nodes: Vec<NodeStrategyFile> = result.all_nodes.into_iter().map(|n| {
                    NodeStrategyFile {
                        node_id: n.node_id,
                        player: n.player,
                        street: n.street,
                        path: n.path,
                        turn_card: n.turn_card.map(card_to_string),
                        strategy: n.strategy,
                        action_labels: n.action_labels,
                    }
                }).collect();

                let solution = SolutionFile {
                    flop_label: label.clone(),
                    stack_bb,
                    k_buckets,
                    iterations,
                    hero_value: result.hero_value,
                    root_strategy: result.root_strategy,
                    action_labels: result.action_labels,
                    bucket_hands,
                    bucket_ehs_range,
                    nodes,
                    elapsed_sec: elapsed.as_secs_f64(),
                };
                let _ = fs::write(&outpath, serde_json::to_string_pretty(&solution).unwrap_or_default());

                let total_elapsed = start_time.elapsed().as_secs_f64();
                tracker.mark_done(&label, t0, elapsed.as_secs_f64(), result.hero_value, total_elapsed);

                let line = format!("[W{}] {} done in {:.1}s (EV={:.2}){}",
                    wid, label, elapsed.as_secs_f64(), result.hero_value, _prof_line);
                let l2 = format!("[{}] {}\n", chrono_now(), line);
                print!("{}", l2);
                if let Ok(mut f) = log.lock() { let _ = f.write_all(l2.as_bytes()); }
            }
        })
    }).collect();

    for h in handles { let _ = h.join(); }

    let total_elapsed = started.elapsed();
    tracker.mark_done_all(total_elapsed.as_secs_f64());
    log_line(format!("=== done {} flops in {:.1}s ===", flops.len(), total_elapsed.as_secs_f64()));

    // Print summary stats
    let times = tracker.times.lock().unwrap();
    if !times.is_empty() {
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_single: f64 = times.iter().sum();
        let actual_speedup = sum_single / total_elapsed.as_secs_f64();
        log_line(format!(
            "per-flop: avg={:.1}s min={:.1}s max={:.1}s | wall={:.1}s | effective speedup={:.2}x",
            avg, min, max, total_elapsed.as_secs_f64(), actual_speedup,
        ));
    }
}

#[derive(Debug, Serialize)]
struct NodeStrategyFile {
    node_id: usize,
    player: i8,
    street: String,
    path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    turn_card: Option<String>,  // human-readable card label, e.g. "Th"
    /// strategy[b] = [p_action0, p_action1, ...] for bucket b
    strategy: Vec<Vec<f32>>,
    action_labels: Vec<String>,
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
    /// Per bucket: deduplicated hand types sorted by weight desc
    bucket_hands: Vec<Vec<String>>,
    /// Per bucket: [min_ehs, max_ehs]
    bucket_ehs_range: Vec<[f32; 2]>,
    /// Full strategy for all flop+turn decision nodes (both players)
    nodes: Vec<NodeStrategyFile>,
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

fn chrono_now() -> String {
    let duration = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or(Duration::ZERO);
    let secs = duration.as_secs();
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
}
