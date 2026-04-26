//! Range bucketing — collapse 1326 combos to K buckets based on E[HS].
//!
//! This is the key infrastructure for the "bucketed CFR" speedup. Once combos
//! are bucketed, all of CFR (regret, strategy, reach, terminals) operates on
//! K-dimensional vectors instead of 1326-dim, giving O(K²) terminal cost
//! instead of O(n_combos²). At K=16, that's ~6000× per-terminal speedup.
//!
//! The metric E[HS] (expected hand strength) = P(hero wins | board, hero_combo)
//! vs a uniform random opponent combo drawn from non-conflicting cards.
//!
//! Bucketing is equal-weight quantile: sort combos in the range by E[HS],
//! cumulative-weight partition into K buckets.

use super::cards::{combo_cards, NUM_CARDS, NUM_COMBOS};
use super::hand_eval::evaluate_seven;

#[derive(Debug, Clone)]
pub struct Bucketing {
    /// Number of buckets actually populated (≤ k_max).
    pub k: usize,
    /// For each of the 1326 combos: bucket id (0..k), or -1 if the combo is
    /// not in the range or conflicts with the board.
    pub bucket_of_combo: Vec<i8>,
    /// For each bucket: sum of combo weights assigned to it.
    pub bucket_weight: Vec<f32>,
    /// For each bucket: list of (global_combo_idx, weight) in that bucket.
    /// Useful for conflict-aware terminal evaluation.
    pub combos_in_bucket: Vec<Vec<(u16, f32)>>,
}

impl Bucketing {
    pub fn bucket_for(&self, combo: usize) -> i8 {
        self.bucket_of_combo[combo]
    }
}

/// E[HS] at a 4-card board: for each combo, average win probability over the
/// 48 possible river cards (and uniform random opponent at each).
///
/// More expensive than `river_ehs` (~46× per combo), but better captures
/// hand-strength uncertainty before the river is dealt. Used to bucket combos
/// at turn-level subgame solves.
pub fn turn_ehs(board_4: &[u8; 4]) -> Vec<f32> {
    let mut out = vec![-1.0f32; NUM_COMBOS];
    let board_mask: u64 = board_4.iter().fold(0u64, |a, &c| a | (1u64 << c));
    assert_eq!(board_mask.count_ones(), 4);
    let remaining: Vec<u8> = (0..NUM_CARDS as u8).filter(|c| board_mask & (1u64 << c) == 0).collect();

    for combo in 0..NUM_COMBOS {
        let (a, b) = combo_cards(combo);
        if board_mask & ((1u64 << a) | (1u64 << b)) != 0 { continue; }

        let mut win_acc = 0.0f64;
        let mut count = 0u32;
        for &river in &remaining {
            if river == a || river == b { continue; }
            let mut board_5 = [0u8; 5];
            board_5[..4].copy_from_slice(board_4);
            board_5[4] = river;
            let mut seven = [0u8; 7];
            seven[..5].copy_from_slice(&board_5);
            seven[5] = a; seven[6] = b;
            let hrank = evaluate_seven(seven);

            let hero_mask = (1u64 << a) | (1u64 << b) | (1u64 << river);
            // Compute equity vs uniform random opponent on this 5-card board
            let mut win = 0u32;
            let mut tie = 0u32;
            let mut tot = 0u32;
            for opp_combo in 0..NUM_COMBOS {
                let (oa, ob) = combo_cards(opp_combo);
                if hero_mask & ((1u64 << oa) | (1u64 << ob)) != 0 { continue; }
                if board_mask & ((1u64 << oa) | (1u64 << ob)) != 0 { continue; }
                let mut osev = [0u8; 7];
                osev[..5].copy_from_slice(&board_5);
                osev[5] = oa; osev[6] = ob;
                let orank = evaluate_seven(osev);
                tot += 1;
                if hrank > orank { win += 1; }
                else if hrank == orank { tie += 1; }
            }
            if tot > 0 {
                win_acc += (win as f64 + 0.5 * tie as f64) / tot as f64;
                count += 1;
            }
        }
        if count > 0 {
            out[combo] = (win_acc / count as f64) as f32;
        }
    }
    out
}

/// E[HS] at a 3-card board (flop): averages over the 47×46 = 2162 (turn, river)
/// runouts. Very expensive; for production use you'd cache or downsample.
pub fn flop_ehs_sampled(board_3: &[u8; 3], n_samples: u32, seed: u64) -> Vec<f32> {
    let mut out = vec![-1.0f32; NUM_COMBOS];
    let board_mask: u64 = board_3.iter().fold(0u64, |a, &c| a | (1u64 << c));
    assert_eq!(board_mask.count_ones(), 3);

    let mut rng = seed | 1;
    for combo in 0..NUM_COMBOS {
        let (a, b) = combo_cards(combo);
        if board_mask & ((1u64 << a) | (1u64 << b)) != 0 { continue; }
        let hero_mask = (1u64 << a) | (1u64 << b);
        let used_mask = board_mask | hero_mask;
        let mut deck: Vec<u8> = (0..NUM_CARDS as u8).filter(|c| used_mask & (1u64 << c) == 0).collect();

        let mut win_acc = 0.0f64;
        for _ in 0..n_samples {
            // Sample (turn, river)
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let i1 = (rng as usize) % deck.len();
            deck.swap(0, i1);
            let turn = deck[0];
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let i2 = 1 + (rng as usize) % (deck.len() - 1);
            deck.swap(1, i2);
            let river = deck[1];

            let mut board_5 = [0u8; 5];
            board_5[..3].copy_from_slice(board_3);
            board_5[3] = turn;
            board_5[4] = river;
            let mut seven = [0u8; 7];
            seven[..5].copy_from_slice(&board_5);
            seven[5] = a; seven[6] = b;
            let hrank = evaluate_seven(seven);

            let runout_used = hero_mask | (1u64 << turn) | (1u64 << river) | board_mask;

            // Equity vs uniform random opponent on this 5-card board
            let mut win = 0u32;
            let mut tie = 0u32;
            let mut tot = 0u32;
            for opp_combo in 0..NUM_COMBOS {
                let (oa, ob) = combo_cards(opp_combo);
                if runout_used & ((1u64 << oa) | (1u64 << ob)) != 0 { continue; }
                let mut osev = [0u8; 7];
                osev[..5].copy_from_slice(&board_5);
                osev[5] = oa; osev[6] = ob;
                let orank = evaluate_seven(osev);
                tot += 1;
                if hrank > orank { win += 1; }
                else if hrank == orank { tie += 1; }
            }
            if tot > 0 {
                win_acc += (win as f64 + 0.5 * tie as f64) / tot as f64;
            }
        }
        out[combo] = (win_acc / n_samples as f64) as f32;
    }
    out
}

/// Compute E[HS] for each non-board-conflict combo given a 5-card board.
/// Returns a vector of length NUM_COMBOS with -1.0 for conflicting combos.
pub fn river_ehs(board: &[u8; 5]) -> Vec<f32> {
    let mut out = vec![-1.0f32; NUM_COMBOS];
    let board_mask: u64 = board.iter().fold(0u64, |a, &c| a | (1u64 << c));
    assert_eq!(board_mask.count_ones(), 5);

    // Precompute hand rank per non-conflict combo on this board
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

    // For each hero combo, count wins/ties across all non-conflicting villain combos
    for (hero, hrank_opt) in ranks.iter().enumerate() {
        let hrank = match hrank_opt { Some(r) => *r, None => continue };
        let (ha, hb) = combo_cards(hero);
        let hero_mask = (1u64 << ha) | (1u64 << hb);
        let mut win = 0u32;
        let mut tie = 0u32;
        let mut total = 0u32;
        for (villain, vrank_opt) in ranks.iter().enumerate() {
            let vrank = match vrank_opt { Some(r) => *r, None => continue };
            let (va, vb) = combo_cards(villain);
            if hero_mask & ((1u64 << va) | (1u64 << vb)) != 0 { continue; }
            total += 1;
            if hrank > vrank { win += 1; }
            else if hrank == vrank { tie += 1; }
        }
        if total > 0 {
            out[hero] = (win as f32 + 0.5 * tie as f32) / total as f32;
        }
    }
    out
}

/// Quantile-bucket combos by E[HS]. Combos with weight ≤ 0 are excluded.
pub fn bucket_by_ehs(
    ehs: &[f32],
    range_weights: &[f32],
    board: &[u8; 5],
    k_max: usize,
) -> Bucketing {
    assert_eq!(ehs.len(), NUM_COMBOS);
    assert_eq!(range_weights.len(), NUM_COMBOS);

    // Gather (combo, weight, ehs) for combos in range
    let board_mask: u64 = board.iter().fold(0u64, |a, &c| a | (1u64 << c));
    let mut entries: Vec<(usize, f32, f32)> = Vec::new();
    for combo in 0..NUM_COMBOS {
        if range_weights[combo] <= 0.0 { continue; }
        let (a, b) = combo_cards(combo);
        if board_mask & ((1u64 << a) | (1u64 << b)) != 0 { continue; }
        if ehs[combo] < 0.0 { continue; }
        entries.push((combo, range_weights[combo], ehs[combo]));
    }

    // Sort ascending by E[HS]
    entries.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let total_weight: f32 = entries.iter().map(|(_, w, _)| *w).sum();
    let k = k_max.min(entries.len()).max(1);
    let target_per_bucket = total_weight / k as f32;

    let mut bucket_of_combo = vec![-1i8; NUM_COMBOS];
    let mut bucket_weight = vec![0.0f32; k];
    let mut combos_in_bucket: Vec<Vec<(u16, f32)>> = vec![Vec::new(); k];

    let mut cum: f32 = 0.0;
    let mut current = 0usize;
    for (combo, weight, _) in entries.iter() {
        bucket_of_combo[*combo] = current as i8;
        bucket_weight[current] += *weight;
        combos_in_bucket[current].push((*combo as u16, *weight));
        cum += *weight;
        if cum >= target_per_bucket * (current as f32 + 1.0) && current + 1 < k {
            current += 1;
        }
    }

    Bucketing { k, bucket_of_combo, bucket_weight, combos_in_bucket }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cards::card_from_str;

    fn board(cards: &[&str]) -> [u8; 5] {
        let mut out = [0u8; 5];
        for (i, c) in cards.iter().enumerate() {
            out[i] = card_from_str(c).unwrap();
        }
        out
    }

    #[test]
    fn ehs_nut_vs_air() {
        let b = board(&["Ad", "Kh", "7s", "3c", "2d"]);
        let ehs = river_ehs(&b);
        // AsAc = top set, near 1.0
        let aa = super::super::cards::combo_index(
            card_from_str("As").unwrap(), card_from_str("Ac").unwrap());
        // 8h9c = air on this board
        let air = super::super::cards::combo_index(
            card_from_str("8h").unwrap(), card_from_str("9c").unwrap());
        assert!(ehs[aa] > 0.95, "AA ehs = {}", ehs[aa]);
        assert!(ehs[air] < 0.5, "8h9c ehs = {}", ehs[air]);
    }

    #[test]
    fn bucketing_partitions_evenly() {
        let b = board(&["Ad", "Kh", "7s", "3c", "2d"]);
        let ehs = river_ehs(&b);
        let range = vec![1.0f32; NUM_COMBOS];
        let bucketing = bucket_by_ehs(&ehs, &range, &b, 16);
        assert!(bucketing.k <= 16);
        // Weights roughly balanced
        let total: f32 = bucketing.bucket_weight.iter().sum();
        let avg = total / bucketing.k as f32;
        for &w in &bucketing.bucket_weight {
            assert!((w - avg).abs() / avg < 0.2, "bucket weight imbalanced: {} vs avg {}", w, avg);
        }
    }

    #[test]
    fn bucketing_monotone_in_ehs() {
        // Highest-EHS combo should end up in the highest bucket
        let b = board(&["Ad", "Kh", "7s", "3c", "2d"]);
        let ehs = river_ehs(&b);
        let range = vec![1.0f32; NUM_COMBOS];
        let bucketing = bucket_by_ehs(&ehs, &range, &b, 8);
        let aa = super::super::cards::combo_index(
            card_from_str("As").unwrap(), card_from_str("Ac").unwrap());
        let air = super::super::cards::combo_index(
            card_from_str("8h").unwrap(), card_from_str("9c").unwrap());
        assert!(bucketing.bucket_of_combo[aa] > bucketing.bucket_of_combo[air]);
    }
}
