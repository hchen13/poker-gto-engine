//! On-disk persistent cache for solved subgames.
//!
//! V1: simple key→value JSON store, keyed by canonical-JSON fingerprint of
//! the solver spec. Used for two things:
//!
//! 1. **Skill query memoization** — when the user asks the same spot twice,
//!    second response is instant.
//! 2. **Precompute checkpointing** — re-runnable precompute that skips
//!    already-solved entries.
//!
//! NOT YET implemented: in-CFR leaf evaluator cache. That needs range
//! bucketing infrastructure to get meaningful hit rates and is a future
//! upgrade. See `docs/roadmap.md` "tower precompute" section.

use std::fs;
use std::path::{Path, PathBuf};

/// FNV-1a 64-bit hash. Stable across machines; no external deps.
pub fn stable_hash(s: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in s {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub struct PersistentCache {
    base_dir: PathBuf,
}

impl PersistentCache {
    pub fn new(base_dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let base_dir = base_dir.into();
        fs::create_dir_all(&base_dir)?;
        Ok(Self { base_dir })
    }

    pub fn get(&self, key: &str) -> Option<String> {
        let path = self.path_for(key);
        fs::read_to_string(&path).ok()
    }

    pub fn put(&self, key: &str, value: &str) -> std::io::Result<()> {
        let path = self.path_for(key);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, value)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.path_for(key).exists()
    }

    /// Path for a given key. Splits into 2-char shards for filesystem hygiene.
    pub fn path_for(&self, key: &str) -> PathBuf {
        if key.len() < 4 {
            return self.base_dir.join(format!("{}.json", key));
        }
        let shard = &key[..2];
        self.base_dir.join(shard).join(format!("{}.json", key))
    }
}

/// Compute the canonical cache key for a solver spec. Caller passes a
/// canonical JSON serialization (sorted keys, no extra whitespace).
pub fn cache_key_for(canonical_spec_json: &str) -> String {
    let h = stable_hash(canonical_spec_json.as_bytes());
    format!("{:016x}", h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn fnv_stable() {
        assert_eq!(stable_hash(b"foo"), stable_hash(b"foo"));
        assert_ne!(stable_hash(b"foo"), stable_hash(b"bar"));
    }

    #[test]
    fn cache_roundtrip() {
        let dir = temp_dir().join(format!("nlhe-cache-test-{}", std::process::id()));
        let cache = PersistentCache::new(&dir).unwrap();
        let key = cache_key_for(r#"{"game":"river","board":["Ad"]}"#);
        cache.put(&key, r#"{"hero_value": 100.0}"#).unwrap();
        assert!(cache.contains(&key));
        let got = cache.get(&key).unwrap();
        assert!(got.contains("100.0"));
        fs::remove_dir_all(&dir).ok();
    }
}
