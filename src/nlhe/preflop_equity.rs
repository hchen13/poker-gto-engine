//! Precomputed preflop equity table.
//!
//! Used as the terminal evaluator in preflop CFR: at a preflop showdown
//! terminal (e.g. flop will be dealt), look up E[hero wins | hero combo i,
//! villain combo j] averaged over all possible boards.
//!
//! The "169 hand classes" abstraction (AA, AKs, AKo, ..., 22) collapses
//! 1326×1326 to 169×169 = 28k entries (14k unique by symmetry). Card-removal
//! effects within a class are ignored — accurate enough for preflop GTO.
//!
//! Computation approach: for each (hero_class, villain_class), sample N
//! random boards and average hero's showdown equity. With N=2000 this is
//! ~30 sec on a single thread. The output table can be persisted and
//! reused across all preflop solves at any stack depth.

use std::collections::HashMap;

use super::cards::{combo_cards, NUM_CARDS, NUM_COMBOS};
use super::hand_eval::evaluate_seven;

/// Hand class identifier: 0..169.
pub type HandClass = u8;

/// 169-class equity table: row-major hero_class × villain_class.
/// Cell value = P(hero wins) + 0.5 × P(tie), or 0 for invalid (same-card)
/// cross-pairings.
pub struct PreflopEquityTable {
    pub equity: Vec<f32>,            // length 169 * 169
    pub n_classes: usize,
}

impl PreflopEquityTable {
    pub fn equity_for_classes(&self, hero_class: HandClass, villain_class: HandClass) -> f32 {
        self.equity[hero_class as usize * self.n_classes + villain_class as usize]
    }

    pub fn equity_for_combos(&self, hero_combo: usize, villain_combo: usize) -> f32 {
        let h = combo_to_class(hero_combo);
        let v = combo_to_class(villain_combo);
        self.equity_for_classes(h, v)
    }
}

/// Map a 1326-combo idx to its 169-class idx.
///
/// Layout (matches standard convention):
///   class 0..12  = pocket pairs 22..AA (rank-2)
///   class 13..90 = suited non-pair, in canonical order (highRank desc, lowRank desc)
///   class 91..168 = offsuit non-pair, same ordering
pub fn combo_to_class(combo: usize) -> HandClass {
    let (a, b) = combo_cards(combo);
    let ra = (a / 4) + 2;
    let rb = (b / 4) + 2;
    let sa = a % 4;
    let sb = b % 4;
    let suited = sa == sb;
    let (hi, lo) = if ra > rb { (ra, rb) } else { (rb, ra) };

    if hi == lo {
        return (hi - 2) as HandClass; // 0..12
    }
    // Index for non-pair: walk over pairs (hi, lo) with hi > lo.
    // 78 unique (hi, lo) pairs.
    let pair_idx = non_pair_index(hi, lo);
    if suited {
        13 + pair_idx as HandClass
    } else {
        13 + 78 + pair_idx as HandClass
    }
}

fn non_pair_index(hi: u8, lo: u8) -> u32 {
    // Order: AKx, AQx, AJx, ..., A2x (12), KQx, KJx, ..., K2x (11), ...
    // pair_idx = sum_{r=hi+1..=14} (r-2) ... Hmm let me think.
    // Easier: enumerate all 78 (hi, lo) with hi > lo, lexicographic descending hi then descending lo.
    // For (hi, lo): index = (14 - hi) * (something) + (hi - lo - 1)
    // Actually use direct enumeration:
    let mut idx = 0u32;
    for h in (3..=14).rev() {
        for l in (2..h).rev() {
            if h == hi && l == lo {
                return idx;
            }
            idx += 1;
        }
    }
    panic!("invalid hi={} lo={}", hi, lo);
}

pub const N_HAND_CLASSES: usize = 169;

/// Compute the preflop equity table by Monte Carlo sampling. `n_samples` is
/// the number of random 5-card boards drawn per (hero_class, villain_class)
/// pair (deck excludes both holes). For accuracy ≈ 1% standard error use
/// n_samples ≥ 10000 per pair; for a quick demo 2000 is enough.
pub fn compute_preflop_equity_table(n_samples: u32, seed: u64) -> PreflopEquityTable {
    let mut equity = vec![0.0f32; N_HAND_CLASSES * N_HAND_CLASSES];
    // For each (hero_class, villain_class), pick one canonical combo for hero
    // and one for villain that don't share cards. Compute equity by sampling
    // boards from the remaining 48 cards.
    for hc in 0..N_HAND_CLASSES {
        let hero_combo = canonical_combo_for_class(hc as HandClass);
        let (ha, hb) = combo_cards(hero_combo);
        for vc in 0..N_HAND_CLASSES {
            // Find a villain combo of class vc that doesn't share cards with hero
            let villain_combo = match canonical_combo_avoiding(vc as HandClass, ha, hb) {
                Some(c) => c,
                None => {
                    // No valid combo (e.g. hero AA & villain AA — only 6 AA combos exist
                    // and they all overlap). Fallback: equity = 0.5 (tie surrogate).
                    equity[hc * N_HAND_CLASSES + vc] = 0.5;
                    continue;
                }
            };
            let (va, vb) = combo_cards(villain_combo);
            let used = [ha, hb, va, vb];
            let eq = sample_equity(used, ha, hb, va, vb, n_samples, seed.wrapping_add((hc * N_HAND_CLASSES + vc) as u64));
            equity[hc * N_HAND_CLASSES + vc] = eq;
        }
    }
    PreflopEquityTable { equity, n_classes: N_HAND_CLASSES }
}

/// Pick the lexicographically smallest combo of the given hand class.
fn canonical_combo_for_class(class: HandClass) -> usize {
    for combo in 0..NUM_COMBOS {
        if combo_to_class(combo) == class {
            return combo;
        }
    }
    panic!("no combo found for class {}", class)
}

fn canonical_combo_avoiding(class: HandClass, used_a: u8, used_b: u8) -> Option<usize> {
    for combo in 0..NUM_COMBOS {
        if combo_to_class(combo) == class {
            let (a, b) = combo_cards(combo);
            if a != used_a && a != used_b && b != used_a && b != used_b {
                return Some(combo);
            }
        }
    }
    None
}

fn sample_equity(
    used_cards: [u8; 4],
    ha: u8, hb: u8, va: u8, vb: u8,
    n_samples: u32, seed: u64,
) -> f32 {
    let mut deck: Vec<u8> = (0..NUM_CARDS as u8).filter(|c| !used_cards.contains(c)).collect();
    let mut rng_state = seed | 1; // avoid zero
    let mut wins = 0.0f64;
    for _ in 0..n_samples {
        // Fisher–Yates partial shuffle to get 5 random distinct cards
        let mut board = [0u8; 5];
        let n = deck.len();
        for i in 0..5 {
            // xorshift64
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let pick = (rng_state as usize) % (n - i);
            let idx = i + pick;
            deck.swap(i, idx);
            board[i] = deck[i];
        }

        let mut hero_seven = [0u8; 7];
        hero_seven[..5].copy_from_slice(&board);
        hero_seven[5] = ha; hero_seven[6] = hb;
        let hero_rank = evaluate_seven(hero_seven);

        let mut villain_seven = [0u8; 7];
        villain_seven[..5].copy_from_slice(&board);
        villain_seven[5] = va; villain_seven[6] = vb;
        let villain_rank = evaluate_seven(villain_seven);

        wins += if hero_rank > villain_rank { 1.0 }
                else if hero_rank == villain_rank { 0.5 }
                else { 0.0 };
    }
    (wins / n_samples as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cards::card_from_str;

    #[test]
    fn pair_class_indices() {
        // 2c2s should be class 0; AcAs should be class 12.
        let two_idx = (card_from_str("2s").unwrap(), card_from_str("2h").unwrap());
        let a_idx = (card_from_str("As").unwrap(), card_from_str("Ah").unwrap());
        let two_combo = super::super::cards::combo_index(two_idx.0, two_idx.1);
        let a_combo = super::super::cards::combo_index(a_idx.0, a_idx.1);
        assert_eq!(combo_to_class(two_combo), 0);
        assert_eq!(combo_to_class(a_combo), 12);
    }

    #[test]
    fn suited_offsuit_distinct_classes() {
        let aks = combo_to_class(super::super::cards::combo_index(
            card_from_str("As").unwrap(), card_from_str("Ks").unwrap(),
        ));
        let ako = combo_to_class(super::super::cards::combo_index(
            card_from_str("As").unwrap(), card_from_str("Kh").unwrap(),
        ));
        assert_ne!(aks, ako);
        // AKs should be among 13..91, AKo among 91..169
        assert!(aks >= 13 && aks < 91);
        assert!(ako >= 91 && ako < 169);
    }

    #[test]
    fn small_table_sanity() {
        // AA vs 22: AA should win > 80%
        let table = compute_preflop_equity_table(500, 42);
        let aa = combo_to_class(super::super::cards::combo_index(
            card_from_str("As").unwrap(), card_from_str("Ah").unwrap(),
        ));
        let twos = combo_to_class(super::super::cards::combo_index(
            card_from_str("2s").unwrap(), card_from_str("2h").unwrap(),
        ));
        let eq = table.equity_for_classes(aa, twos);
        assert!(eq > 0.78 && eq < 0.92, "AA vs 22 equity = {}", eq);
        let eq_other = table.equity_for_classes(twos, aa);
        assert!(eq_other > 0.08 && eq_other < 0.22, "22 vs AA equity = {}", eq_other);
    }
}
