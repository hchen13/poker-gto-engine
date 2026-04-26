//! Bucket-level equity table. For each (hero_bucket h, villain_bucket v):
//!   equity_sum[h][v] = Σ_{i ∈ h, j ∈ v, no card conflict} w_i · w_j · eq(i, j)
//!   pair_weight[h][v] = Σ_{i ∈ h, j ∈ v, no card conflict} w_i · w_j
//!
//! Where eq(i, j) ∈ {0, 0.5, 1} from the showdown outcome. These two tables
//! let the bucketed CFR compute terminal value at O(K²) per-terminal cost,
//! with card conflicts correctly handled at table-construction time (so the
//! solver can assume bucket-level uniformity during CFR iteration).

use super::bucketing::Bucketing;
use super::cards::{combo_cards, NUM_COMBOS};
use super::hand_eval::evaluate_seven;
use super::showdown::{WIN_HERO, WIN_TIE};

#[derive(Debug, Clone)]
pub struct BucketEquityTable {
    pub k_hero: usize,
    pub k_villain: usize,
    /// Row-major [k_hero][k_villain] — sum of w_i·w_j·eq(i,j) over non-conflict pairs.
    pub equity_sum: Vec<f32>,
    /// Row-major [k_hero][k_villain] — sum of w_i·w_j over non-conflict pairs.
    pub pair_weight: Vec<f32>,
    /// Precomputed equity[h*k_v + v] = equity_sum / pair_weight (0 if pw==0).
    pub equity: Vec<f32>,
    /// For hero update: pw_over_bwv[h*k_v + v] = pair_weight[h,v] / bucket_weight_villain[v].
    /// Outer loop is h, inner dot is over contiguous length-k_v slice. Villain bw baked in.
    pub pw_over_bwv: Vec<f32>,
    /// For villain update: pw_over_bwh[v*k_h + h] = pair_weight[h,v] / bucket_weight_hero[h].
    /// Stored villain-major so inner dot is over contiguous length-k_h slice.
    pub pw_over_bwh: Vec<f32>,
}

impl BucketEquityTable {
    pub fn equity(&self, h: usize, v: usize) -> f32 {
        self.equity[h * self.k_villain + v]
    }
    pub fn equity_sum_at(&self, h: usize, v: usize) -> f32 {
        self.equity_sum[h * self.k_villain + v]
    }
    pub fn pair_weight_at(&self, h: usize, v: usize) -> f32 {
        self.pair_weight[h * self.k_villain + v]
    }
    pub fn equity_at(&self, h: usize, v: usize) -> f32 {
        self.equity[h * self.k_villain + v]
    }
}

/// Build the bucket equity table for a river board (5 cards).
pub fn compute_bucket_equity_river(
    board: &[u8; 5],
    hero_bucketing: &Bucketing,
    villain_bucketing: &Bucketing,
) -> BucketEquityTable {
    let k_h = hero_bucketing.k;
    let k_v = villain_bucketing.k;
    let mut equity_sum = vec![0.0f32; k_h * k_v];
    let mut pair_weight = vec![0.0f32; k_h * k_v];

    let board_mask: u64 = board.iter().fold(0u64, |a, &c| a | (1u64 << c));

    // Precompute hand ranks per combo on this board.
    let mut ranks: Vec<Option<u32>> = vec![None; NUM_COMBOS];
    for combo in 0..NUM_COMBOS {
        let (a, b) = combo_cards(combo);
        if board_mask & ((1u64 << a) | (1u64 << b)) != 0 { continue; }
        let mut seven = [0u8; 7];
        seven[..5].copy_from_slice(board);
        seven[5] = a;
        seven[6] = b;
        ranks[combo] = Some(evaluate_seven(seven));
    }

    // For each pair (hero_combo in some bucket h, villain_combo in some bucket v)
    // where neither combo is filtered (-1 bucket) and they don't share cards.
    for h in 0..k_h {
        for (hero_combo_u16, w_h) in &hero_bucketing.combos_in_bucket[h] {
            let hero_combo = *hero_combo_u16 as usize;
            let hrank = match ranks[hero_combo] { Some(r) => r, None => continue };
            let (ha, hb) = combo_cards(hero_combo);
            let hero_mask = (1u64 << ha) | (1u64 << hb);
            for v in 0..k_v {
                for (villain_combo_u16, w_v) in &villain_bucketing.combos_in_bucket[v] {
                    let villain_combo = *villain_combo_u16 as usize;
                    let vrank = match ranks[villain_combo] { Some(r) => r, None => continue };
                    let (va, vb) = combo_cards(villain_combo);
                    if hero_mask & ((1u64 << va) | (1u64 << vb)) != 0 { continue; }
                    let eq: f32 = if hrank > vrank { 1.0 }
                                   else if hrank == vrank { 0.5 }
                                   else { 0.0 };
                    let pw = *w_h * *w_v;
                    let idx = h * k_v + v;
                    equity_sum[idx] += pw * eq;
                    pair_weight[idx] += pw;
                }
            }
        }
    }

    let _ = (WIN_HERO, WIN_TIE);
    // Precompute normalized equity per (h, v) — avoids division in CFR hot loop.
    let mut equity = vec![0.0f32; k_h * k_v];
    for i in 0..(k_h * k_v) {
        if pair_weight[i] > 0.0 {
            equity[i] = equity_sum[i] / pair_weight[i];
        }
    }

    // pw_over_bwv[h*k_v + v] = pair_weight[h,v] / bucket_weight_villain[v]  (hero-major, inner v)
    let mut pw_over_bwv = vec![0.0f32; k_h * k_v];
    for h in 0..k_h {
        for v in 0..k_v {
            let bw_v = villain_bucketing.bucket_weight[v];
            if bw_v > 0.0 {
                pw_over_bwv[h * k_v + v] = pair_weight[h * k_v + v] / bw_v;
            }
        }
    }

    // pw_over_bwh[v*k_h + h] = pair_weight[h,v] / bucket_weight_hero[h]  (villain-major, inner h)
    let mut pw_over_bwh = vec![0.0f32; k_v * k_h];
    for v in 0..k_v {
        for h in 0..k_h {
            let bw_h = hero_bucketing.bucket_weight[h];
            if bw_h > 0.0 {
                pw_over_bwh[v * k_h + h] = pair_weight[h * k_v + v] / bw_h;
            }
        }
    }

    BucketEquityTable { k_hero: k_h, k_villain: k_v, equity_sum, pair_weight, equity, pw_over_bwv, pw_over_bwh }
}

#[cfg(test)]
mod tests {
    use super::super::bucketing::{bucket_by_ehs, river_ehs};
    use super::super::cards::{card_from_str, NUM_COMBOS};
    use super::*;

    fn board(cards: &[&str]) -> [u8; 5] {
        let mut out = [0u8; 5];
        for (i, c) in cards.iter().enumerate() { out[i] = card_from_str(c).unwrap(); }
        out
    }

    #[test]
    fn top_bucket_crushes_bottom() {
        let b = board(&["Ad", "Kh", "7s", "3c", "2d"]);
        let ehs = river_ehs(&b);
        let range = vec![1.0f32; NUM_COMBOS];
        let bucketing = bucket_by_ehs(&ehs, &range, &b, 8);
        let eq_table = compute_bucket_equity_river(&b, &bucketing, &bucketing);
        let top = bucketing.k - 1;
        let bottom = 0;
        let top_vs_bottom = eq_table.equity(top, bottom);
        let bottom_vs_top = eq_table.equity(bottom, top);
        assert!(top_vs_bottom > 0.8, "top bucket should win vs bottom bucket, got {}", top_vs_bottom);
        assert!(bottom_vs_top < 0.2, "bottom bucket should lose vs top bucket, got {}", bottom_vs_top);
    }

    #[test]
    fn symmetric_matrix_sums_to_one_minus_ties() {
        // For same bucketing both sides, equity[h][v] + equity[v][h] should equal 1 (minus 2× tie / 2)
        let b = board(&["Ad", "Kh", "7s", "3c", "2d"]);
        let ehs = river_ehs(&b);
        let range = vec![1.0f32; NUM_COMBOS];
        let bucketing = bucket_by_ehs(&ehs, &range, &b, 4);
        let eq_table = compute_bucket_equity_river(&b, &bucketing, &bucketing);
        for h in 0..bucketing.k {
            for v in 0..bucketing.k {
                let sum = eq_table.equity(h, v) + eq_table.equity(v, h);
                // equity[h][v] + equity[v][h] = wins_h + ties/2 + wins_v + ties/2 = 1 (exactly)
                if eq_table.pair_weight_at(h, v) > 0.0 {
                    assert!((sum - 1.0).abs() < 1e-4, "({},{}): equity sum = {}", h, v, sum);
                }
            }
        }
    }
}
