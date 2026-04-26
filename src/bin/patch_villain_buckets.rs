//! One-shot patcher: add `villain_bucket_hands` and `villain_bucket_ehs_range`
//! to every existing precompute JSON file.
//!
//! Usage:
//!   cargo run --release --bin patch_villain_buckets
//!
//! Reads all dirs matching precompute_out/full_*_{200,500}bb/, derives the
//! action_line and stack from the dir name, loads the corresponding ranges
//! fixture, recomputes villain bucketing per-flop, and patches each JSON.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use poker_gto_engine::nlhe::abstraction::rank_subsample;
use poker_gto_engine::nlhe::bucketing::{bucket_by_ehs, river_ehs};
use poker_gto_engine::nlhe::cards::{card_from_str, combo_label, NUM_COMBOS};
use poker_gto_engine::nlhe::range_parser::parse_range;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct RangesFile {
    action_lines: HashMap<String, ActionLine>,
}

#[derive(Debug, Deserialize)]
struct ActionLine {
    sb_range: String,
    #[allow(dead_code)]
    bb_range: String,
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn ranges_file_for_stack(stack_bb: u32) -> PathBuf {
    project_root()
        .join("fixtures")
        .join("precompute")
        .join(format!("hu_ranges_{}bb.json", stack_bb))
}

/// Given dir name like "full_3bet_called_200bb", return (action_line, stack_bb).
fn parse_dir_name(name: &str) -> Option<(String, u32)> {
    let name = name.strip_prefix("full_")?;
    let (action, stack) = if let Some(pos) = name.rfind("_200bb") {
        (&name[..pos], 200u32)
    } else if let Some(pos) = name.rfind("_500bb") {
        (&name[..pos], 500u32)
    } else {
        return None;
    };
    Some((action.to_string(), stack))
}

/// Reconstruct rep5 from 3-card flop (same logic as precompute_bucketed_parallel).
fn make_rep5(flop: &[u8; 3]) -> [u8; 5] {
    let board_mask: u64 = flop.iter().fold(0u64, |a, &c| a | (1u64 << c));
    let remaining: Vec<u8> = (0..52u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
    let turn_subset = rank_subsample(&remaining);
    let river_subset = rank_subsample(&remaining);
    let mut rep5 = [0u8; 5];
    rep5[0] = flop[0];
    rep5[1] = flop[1];
    rep5[2] = flop[2];
    rep5[3] = turn_subset[turn_subset.len() / 2];
    let mut ri = (river_subset.len() / 2 + 1) % river_subset.len();
    if river_subset[ri] == rep5[3] {
        ri = 0;
    }
    rep5[4] = river_subset[ri];
    rep5
}

/// Parse "Kd2c6h" label into 3 card indices.
fn parse_flop_label(label: &str) -> Option<[u8; 3]> {
    if label.len() != 6 {
        return None;
    }
    let a = card_from_str(&label[0..2])?;
    let b = card_from_str(&label[2..4])?;
    let c = card_from_str(&label[4..6])?;
    Some([a, b, c])
}

fn build_villain_bucket_hands(
    ehs: &[f32],
    villain_weights: &[f32],
    rep5: &[u8; 5],
    k_buckets: usize,
) -> (Vec<Vec<String>>, Vec<[f32; 2]>) {
    let bucketing = bucket_by_ehs(ehs, villain_weights, rep5, k_buckets);

    let bucket_hands: Vec<Vec<String>> = (0..k_buckets).map(|b| {
        let mut weight_map: HashMap<String, f32> = HashMap::new();
        for &(combo_u16, w) in &bucketing.combos_in_bucket[b] {
            let label = combo_label(combo_u16 as usize);
            *weight_map.entry(label).or_default() += w;
        }
        let mut pairs: Vec<(String, f32)> = weight_map.into_iter().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.into_iter().map(|(lbl, _)| lbl).collect()
    }).collect();

    let bucket_ehs: Vec<[f32; 2]> = (0..k_buckets).map(|b| {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &(combo_u16, _) in &bucketing.combos_in_bucket[b] {
            let e = ehs[combo_u16 as usize];
            if e < mn { mn = e; }
            if e > mx { mx = e; }
        }
        if mn == f32::INFINITY { [0.0, 0.0] } else { [mn, mx] }
    }).collect();

    (bucket_hands, bucket_ehs)
}

fn patch_file(path: &Path, villain_weights: &[f32], k_buckets: usize) -> Result<bool, String> {
    let is_gz = path.extension().map(|e| e == "gz").unwrap_or(false);
    let raw = if is_gz {
        let file = fs::File::open(path).map_err(|e| e.to_string())?;
        let mut gz = flate2::read::GzDecoder::new(file);
        let mut s = String::new();
        std::io::Read::read_to_string(&mut gz, &mut s).map_err(|e| e.to_string())?;
        s
    } else {
        fs::read_to_string(path).map_err(|e| e.to_string())?
    };

    let mut data: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;

    // Skip if already patched
    if data.get("villain_bucket_hands").is_some() {
        return Ok(false);
    }

    let flop_label = data["flop_label"].as_str().ok_or("missing flop_label")?;
    let flop = parse_flop_label(flop_label).ok_or(format!("bad label: {}", flop_label))?;

    let rep5 = make_rep5(&flop);
    let ehs = river_ehs(&rep5);
    let (vb_hands, vb_ehs) = build_villain_bucket_hands(&ehs, villain_weights, &rep5, k_buckets);

    data["villain_bucket_hands"] = serde_json::to_value(&vb_hands).unwrap();
    data["villain_bucket_ehs_range"] = serde_json::to_value(&vb_ehs).unwrap();

    if is_gz {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::io::Write;
        let out = serde_json::to_string(&data).map_err(|e| e.to_string())?; // compact
        let file = fs::File::create(path).map_err(|e| e.to_string())?;
        let mut enc = GzEncoder::new(file, Compression::default());
        enc.write_all(out.as_bytes()).map_err(|e: std::io::Error| e.to_string())?;
        enc.finish().map_err(|e: std::io::Error| e.to_string())?;
    } else {
        let out = serde_json::to_string_pretty(&data).map_err(|e| e.to_string())?;
        fs::write(path, out).map_err(|e| e.to_string())?;
    }
    Ok(true)
}

fn main() {
    let precompute_root = project_root().join("precompute_out");
    if !precompute_root.exists() {
        eprintln!("precompute_out not found at {:?}", precompute_root);
        std::process::exit(1);
    }

    // Collect all (dir, action_line, stack_bb) tuples
    let mut jobs: Vec<(PathBuf, String, u32)> = Vec::new();
    for entry in fs::read_dir(&precompute_root).unwrap() {
        let entry = entry.unwrap();
        let dir = entry.path();
        if !dir.is_dir() { continue; }
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        if let Some((action_line, stack_bb)) = parse_dir_name(&name) {
            jobs.push((dir, action_line, stack_bb));
        }
    }

    if jobs.is_empty() {
        eprintln!("No valid precompute dirs found.");
        return;
    }

    println!("Patching {} directories...", jobs.len());

    // Load range weights per (action_line, stack_bb) — cache to avoid re-parsing
    let mut range_cache: HashMap<(String, u32), Vec<f32>> = HashMap::new();
    for (_, action_line, stack_bb) in &jobs {
        let key = (action_line.clone(), *stack_bb);
        if range_cache.contains_key(&key) { continue; }
        let rf_path = ranges_file_for_stack(*stack_bb);
        let rf_raw = match fs::read_to_string(&rf_path) {
            Ok(r) => r,
            Err(e) => { eprintln!("Cannot read {:?}: {}", rf_path, e); continue; }
        };
        let rf: RangesFile = match serde_json::from_str(&rf_raw) {
            Ok(r) => r,
            Err(e) => { eprintln!("Cannot parse {:?}: {}", rf_path, e); continue; }
        };
        let al = match rf.action_lines.get(action_line) {
            Some(a) => a,
            None => { eprintln!("action_line '{}' not in {:?}", action_line, rf_path); continue; }
        };
        let weights = match parse_range(&al.sb_range) {
            Ok(w) => w,
            Err(e) => { eprintln!("Cannot parse SB range for {}/{}: {:?}", action_line, stack_bb, e); continue; }
        };
        range_cache.insert(key, weights);
    }

    let total_patched = Arc::new(Mutex::new(0usize));
    let total_skipped = Arc::new(Mutex::new(0usize));
    let total_errors = Arc::new(Mutex::new(0usize));

    let mut handles = Vec::new();
    let jobs = Arc::new(jobs);
    let range_cache = Arc::new(range_cache);

    let num_threads = 8;
    let jobs_per_thread = (jobs.len() + num_threads - 1) / num_threads;

    for t in 0..num_threads {
        let jobs = Arc::clone(&jobs);
        let range_cache = Arc::clone(&range_cache);
        let total_patched = Arc::clone(&total_patched);
        let total_skipped = Arc::clone(&total_skipped);
        let total_errors = Arc::clone(&total_errors);

        handles.push(thread::spawn(move || {
            let start = t * jobs_per_thread;
            let end = ((t + 1) * jobs_per_thread).min(jobs.len());
            for i in start..end {
                let (dir, action_line, stack_bb) = &jobs[i];
                let key = (action_line.clone(), *stack_bb);
                let villain_weights = match range_cache.get(&key) {
                    Some(w) => w,
                    None => continue,
                };

                let flop_files: Vec<PathBuf> = match fs::read_dir(dir) {
                    Ok(rd) => rd
                        .filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| {
                            let n = p.file_name()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or_default();
                            n.starts_with("flop_") && (n.ends_with(".json.gz") || n.ends_with(".json"))
                        })
                        .collect(),
                    Err(e) => {
                        eprintln!("Cannot read dir {:?}: {}", dir, e);
                        continue;
                    }
                };

                let k_buckets = 16;
                let mut patched = 0usize;
                let mut skipped = 0usize;
                let mut errors = 0usize;

                for fpath in &flop_files {
                    match patch_file(fpath, villain_weights, k_buckets) {
                        Ok(true) => patched += 1,
                        Ok(false) => skipped += 1,
                        Err(e) => {
                            eprintln!("Error patching {:?}: {}", fpath, e);
                            errors += 1;
                        }
                    }
                }

                println!(
                    "[T{}] {} {}/{}bb: patched={} skipped={} errors={}",
                    t, action_line, stack_bb, stack_bb, patched, skipped, errors
                );

                *total_patched.lock().unwrap() += patched;
                *total_skipped.lock().unwrap() += skipped;
                *total_errors.lock().unwrap() += errors;
            }
        }));
    }

    for h in handles { let _ = h.join(); }

    println!(
        "\nDone. patched={} skipped={} errors={}",
        *total_patched.lock().unwrap(),
        *total_skipped.lock().unwrap(),
        *total_errors.lock().unwrap(),
    );
}
