//! On-disk storage format for solved subgames.
//!
//! Each "solution" is the average strategy + EV trajectory of one CFR run.
//! For precomputed tables we want a compact, indexable, mmap-friendly format.
//!
//! V1 schema (per-subgame file):
//!
//!   header (magic, version, n_decision_nodes, n_combos)
//!   for each decision node:
//!     u32 node_id, u32 player, u32 n_buckets, u32 n_actions
//!     for each (bucket, action): f32 strategy_probability  (avg strategy, normalized)
//!   trailer (CRC, scalar EV)
//!
//! For now this lives as a "write to file, read back" pair without mmap or
//! cross-language access. Sufficient to checkpoint precompute progress and
//! reload solutions without re-solving.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Result as IoResult};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredSolution {
    /// Free-form key describing the spot (e.g. "preflop_200bb_sb_open_3x").
    pub key: String,
    /// CFR iterations run.
    pub iterations: u32,
    /// Final EV to player 0 averaged over both ranges.
    pub hero_value: f32,
    /// Optional exploitability info from BR validation.
    pub exploitability: Option<f32>,
    /// Per-decision-node strategy: node_id → bucket → action → probability.
    /// Decision-node ids are assigned by the solver in DFS pre-order, so they
    /// are stable across runs of the same tree.
    pub strategy: HashMap<u32, BucketStrategy>,
    /// Optional: scalar metrics (EV trace, time, etc.).
    pub last_iter_values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketStrategy {
    pub player: i8,
    pub n_buckets: u32,
    pub action_labels: Vec<String>,
    /// Row-major: [bucket][action] = probability
    pub probabilities: Vec<Vec<f32>>,
}

pub fn save_to_file(path: impl AsRef<Path>, solution: &StoredSolution) -> IoResult<()> {
    let f = File::create(path)?;
    let w = BufWriter::new(f);
    serde_json::to_writer(w, solution)?;
    Ok(())
}

pub fn load_from_file(path: impl AsRef<Path>) -> IoResult<StoredSolution> {
    let f = File::open(path)?;
    let r = BufReader::new(f);
    let s: StoredSolution = serde_json::from_reader(r)?;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn roundtrip_solution() {
        let mut strategy = HashMap::new();
        strategy.insert(0u32, BucketStrategy {
            player: 0,
            n_buckets: 2,
            action_labels: vec!["check".into(), "bet_50.00".into()],
            probabilities: vec![
                vec![0.7, 0.3],
                vec![0.1, 0.9],
            ],
        });
        let s = StoredSolution {
            key: "test".into(),
            iterations: 100,
            hero_value: 50.0,
            exploitability: Some(2.5),
            strategy,
            last_iter_values: vec![10.0, 20.0],
        };
        let path = temp_dir().join("test_solution.json");
        save_to_file(&path, &s).unwrap();
        let loaded = load_from_file(&path).unwrap();
        assert_eq!(loaded.key, "test");
        assert_eq!(loaded.iterations, 100);
        assert_eq!(loaded.hero_value, 50.0);
        assert_eq!(loaded.strategy[&0].probabilities[0][0], 0.7);
    }
}
