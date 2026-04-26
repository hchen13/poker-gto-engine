//! Patch all precomputed JSON files to add oop_bucket_of_combo and ip_bucket_of_combo.
//!
//! These arrays (length 1326, values -1 or 0..k-1) map each global combo index to
//! its bucket for OOP (hero/BB) and IP (villain/SB). Required by the river subgame
//! solver to filter ranges along an action path.
//!
//! Usage:
//!   cargo run --release --bin patch_bucket_arrays

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

use poker_gto_engine::nlhe::abstraction::rank_subsample;
use poker_gto_engine::nlhe::bucketing::{bucket_by_ehs, river_ehs};
use poker_gto_engine::nlhe::cards::{card_from_str, NUM_CARDS, NUM_COMBOS};
use poker_gto_engine::nlhe::range_parser::parse_range;
use serde_json::Value;

#[derive(Debug, serde::Deserialize)]
struct RangesFile {
    action_lines: HashMap<String, ActionLine>,
}

#[derive(Debug, serde::Deserialize)]
struct ActionLine {
    sb_range: String,
    bb_range: String,
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn ranges_file_for_stack(stack_bb: u32) -> PathBuf {
    project_root()
        .join("fixtures").join("precompute")
        .join(format!("hu_ranges_{}bb.json", stack_bb))
}

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

fn parse_flop_label(label: &str) -> Option<[u8; 3]> {
    if label.len() != 6 { return None; }
    let a = card_from_str(&label[0..2])?;
    let b = card_from_str(&label[2..4])?;
    let c = card_from_str(&label[4..6])?;
    Some([a, b, c])
}

fn make_rep5(flop: &[u8; 3]) -> [u8; 5] {
    let board_mask: u64 = flop.iter().fold(0u64, |a, &c| a | (1u64 << c));
    let remaining: Vec<u8> = (0..NUM_CARDS as u8)
        .filter(|c| board_mask & (1u64 << c) == 0).collect();
    let turn_subset = rank_subsample(&remaining);
    let river_subset = rank_subsample(&remaining);
    let mut rep5 = [0u8; 5];
    rep5[..3].copy_from_slice(flop);
    rep5[3] = turn_subset[turn_subset.len() / 2];
    let mut ri = (river_subset.len() / 2 + 1) % river_subset.len();
    if river_subset[ri] == rep5[3] { ri = 0; }
    rep5[4] = river_subset[ri];
    rep5
}

fn patch_file(
    path: &std::path::Path,
    oop_weights: &[f32],
    ip_weights: &[f32],
    k_buckets: usize,
) -> Result<bool, String> {
    let raw = if path.extension().map(|e| e == "gz").unwrap_or(false) {
        let file = fs::File::open(path).map_err(|e| e.to_string())?;
        let mut gz = flate2::read::GzDecoder::new(file);
        let mut s = String::new();
        std::io::Read::read_to_string(&mut gz, &mut s).map_err(|e| e.to_string())?;
        s
    } else {
        fs::read_to_string(path).map_err(|e| e.to_string())?
    };

    let mut data: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;

    if data.get("oop_bucket_of_combo").is_some() {
        return Ok(false); // already patched
    }

    let flop_label = data["flop_label"].as_str().ok_or("missing flop_label")?;
    let flop = parse_flop_label(flop_label).ok_or(format!("bad label: {}", flop_label))?;
    let rep5 = make_rep5(&flop);
    let ehs = river_ehs(&rep5);

    let oop_b = bucket_by_ehs(&ehs, oop_weights, &rep5, k_buckets);
    let ip_b  = bucket_by_ehs(&ehs, ip_weights,  &rep5, k_buckets);

    // Store as arrays of i8 (-1 = not in range/conflict, 0..k-1 = bucket)
    let oop_arr: Vec<i8> = oop_b.bucket_of_combo.iter().map(|&x| x).collect();
    let ip_arr:  Vec<i8> = ip_b.bucket_of_combo.iter().map(|&x| x).collect();

    data["oop_bucket_of_combo"] = serde_json::to_value(&oop_arr).unwrap();
    data["ip_bucket_of_combo"]  = serde_json::to_value(&ip_arr).unwrap();

    let json_str = serde_json::to_string(&data).map_err(|e| e.to_string())?; // compact

    if path.extension().map(|e| e == "gz").unwrap_or(false) {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::io::Write;
        let file = fs::File::create(path).map_err(|e| e.to_string())?;
        let mut enc = GzEncoder::new(file, Compression::default());
        enc.write_all(json_str.as_bytes()).map_err(|e: std::io::Error| e.to_string())?;
        enc.finish().map_err(|e: std::io::Error| e.to_string())?;
    } else {
        fs::write(path, json_str).map_err(|e| e.to_string())?;
    }

    Ok(true)
}

fn main() {
    let precompute_root = project_root().join("precompute_out");

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
    println!("Patching {} directories...", jobs.len());

    // Load ranges
    let mut range_cache: HashMap<(String, u32), (Vec<f32>, Vec<f32>)> = HashMap::new();
    for (_, action_line, stack_bb) in &jobs {
        let key = (action_line.clone(), *stack_bb);
        if range_cache.contains_key(&key) { continue; }
        let rf_path = ranges_file_for_stack(*stack_bb);
        let rf_raw = match fs::read_to_string(&rf_path) {
            Ok(r) => r, Err(e) => { eprintln!("Cannot read {:?}: {}", rf_path, e); continue; }
        };
        let rf: RangesFile = match serde_json::from_str(&rf_raw) {
            Ok(r) => r, Err(e) => { eprintln!("Cannot parse {:?}: {}", rf_path, e); continue; }
        };
        let al = match rf.action_lines.get(action_line) {
            Some(a) => a, None => { eprintln!("action_line '{}' not found", action_line); continue; }
        };
        let oop_w = parse_range(&al.bb_range).unwrap_or_default(); // hero=BB=OOP
        let ip_w  = parse_range(&al.sb_range).unwrap_or_default(); // villain=SB=IP
        range_cache.insert(key, (oop_w, ip_w));
    }

    let patched_total = Arc::new(Mutex::new(0usize));
    let skipped_total = Arc::new(Mutex::new(0usize));
    let errors_total  = Arc::new(Mutex::new(0usize));

    let jobs = Arc::new(jobs);
    let range_cache = Arc::new(range_cache);
    let n = jobs.len();
    let per = (n + 7) / 8;

    let mut handles = Vec::new();
    for t in 0..8usize {
        let jobs = Arc::clone(&jobs);
        let range_cache = Arc::clone(&range_cache);
        let pat = Arc::clone(&patched_total);
        let ski = Arc::clone(&skipped_total);
        let err = Arc::clone(&errors_total);
        handles.push(thread::spawn(move || {
            let start = t * per;
            let end = ((t + 1) * per).min(n);
            for i in start..end {
                let (dir, action_line, stack_bb) = &jobs[i];
                let key = (action_line.clone(), *stack_bb);
                let (oop_w, ip_w) = match range_cache.get(&key) {
                    Some(w) => w, None => continue,
                };
                let flop_files: Vec<_> = match fs::read_dir(dir) {
                    Ok(rd) => rd.filter_map(|e| e.ok()).map(|e| e.path())
                        .filter(|p| {
                            let n = p.file_name().unwrap_or_default().to_string_lossy().to_string();
                            n.starts_with("flop_") && (n.ends_with(".json.gz") || n.ends_with(".json"))
                        }).collect(),
                    Err(e) => { eprintln!("read_dir {:?}: {}", dir, e); continue; }
                };
                let (mut p, mut s, mut e) = (0, 0, 0);
                for fp in &flop_files {
                    match patch_file(fp, oop_w, ip_w, 16) {
                        Ok(true) => p += 1,
                        Ok(false) => s += 1,
                        Err(msg) => { eprintln!("Error {:?}: {}", fp, msg); e += 1; }
                    }
                }
                println!("[T{}] {}/{}bb  patched={} skipped={} errors={}", t, action_line, stack_bb, p, s, e);
                *pat.lock().unwrap() += p;
                *ski.lock().unwrap() += s;
                *err.lock().unwrap() += e;
            }
        }));
    }
    for h in handles { let _ = h.join(); }
    println!("\nDone. patched={} skipped={} errors={}",
        *patched_total.lock().unwrap(),
        *skipped_total.lock().unwrap(),
        *errors_total.lock().unwrap());
}
