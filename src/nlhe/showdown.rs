//! Per-board showdown outcome matrix.
//!
//! For a given 5-card board and two ranges, enumerate every compatible
//! (hero_combo, villain_combo) pair and record who wins. The CFR solver looks
//! up these outcomes at every showdown terminal — this is the inner-loop
//! data structure.
//!
//! Matches `python/nlhe/showdown.py` semantics. Outcomes are encoded as i8:
//!
//!   +1  hero wins
//!   -1  villain wins
//!    0  tie
//!   -2  conflict (hero/villain combos share a card, or one contains a board card)

use super::cards::{combo_cards, NUM_COMBOS};
use super::hand_eval::{evaluate_seven, HandRank};

pub const WIN_HERO: i8 = 1;
pub const WIN_VILLAIN: i8 = -1;
pub const WIN_TIE: i8 = 0;
pub const CONFLICT: i8 = -2;

#[derive(Debug, Clone)]
pub struct ShowdownTable {
    pub board: [u8; 5],
    pub hero_combos: Vec<u16>,    // local-to-global combo idx mapping
    pub hero_weights: Vec<f32>,
    pub villain_combos: Vec<u16>,
    pub villain_weights: Vec<f32>,
    /// Row-major: outcome[hero_local][villain_local]
    pub outcome: Vec<i8>,
    pub n_hero: usize,
    pub n_villain: usize,
}

impl ShowdownTable {
    #[inline]
    pub fn outcome_at(&self, hero_local: usize, villain_local: usize) -> i8 {
        self.outcome[hero_local * self.n_villain + villain_local]
    }
}

/// Build the showdown table.
///
/// `hero_range` and `villain_range` are length-NUM_COMBOS weight vectors
/// (zero weight = combo not in range). The board cards are filtered out
/// automatically.
pub fn compute_showdown_table(
    board: [u8; 5],
    hero_range: &[f32],
    villain_range: &[f32],
) -> ShowdownTable {
    assert_eq!(hero_range.len(), NUM_COMBOS);
    assert_eq!(villain_range.len(), NUM_COMBOS);

    let board_mask: u64 = board.iter().fold(0u64, |a, &c| a | (1u64 << c));
    assert_eq!(board_mask.count_ones(), 5, "board must have 5 distinct cards");

    let (hero_combos, hero_weights) = filter_range(hero_range, board_mask);
    let (villain_combos, villain_weights) = filter_range(villain_range, board_mask);

    let n_hero = hero_combos.len();
    let n_villain = villain_combos.len();

    // Precompute hand ranks per side
    let hero_ranks: Vec<HandRank> = hero_combos
        .iter()
        .map(|&combo| {
            let (a, b) = combo_cards(combo as usize);
            let mut seven = [0u8; 7];
            seven[..5].copy_from_slice(&board);
            seven[5] = a;
            seven[6] = b;
            evaluate_seven(seven)
        })
        .collect();
    let villain_ranks: Vec<HandRank> = villain_combos
        .iter()
        .map(|&combo| {
            let (a, b) = combo_cards(combo as usize);
            let mut seven = [0u8; 7];
            seven[..5].copy_from_slice(&board);
            seven[5] = a;
            seven[6] = b;
            evaluate_seven(seven)
        })
        .collect();

    // Precompute per-combo card sets for fast conflict checking
    let hero_card_sets: Vec<u64> = hero_combos
        .iter()
        .map(|&combo| {
            let (a, b) = combo_cards(combo as usize);
            (1u64 << a) | (1u64 << b)
        })
        .collect();

    let mut outcome = vec![CONFLICT; n_hero * n_villain];
    for j in 0..n_villain {
        let (va, vb) = combo_cards(villain_combos[j] as usize);
        let villain_set: u64 = (1u64 << va) | (1u64 << vb);
        let vrank = villain_ranks[j];
        for i in 0..n_hero {
            if hero_card_sets[i] & villain_set != 0 {
                continue; // conflict
            }
            let hrank = hero_ranks[i];
            let result = if hrank > vrank {
                WIN_HERO
            } else if hrank < vrank {
                WIN_VILLAIN
            } else {
                WIN_TIE
            };
            outcome[i * n_villain + j] = result;
        }
    }

    ShowdownTable {
        board,
        hero_combos,
        hero_weights,
        villain_combos,
        villain_weights,
        outcome,
        n_hero,
        n_villain,
    }
}

fn filter_range(range: &[f32], board_mask: u64) -> (Vec<u16>, Vec<f32>) {
    let mut combos = Vec::new();
    let mut weights = Vec::new();
    for combo in 0..NUM_COMBOS {
        let w = range[combo];
        if w <= 0.0 {
            continue;
        }
        let (a, b) = combo_cards(combo);
        let pair_mask = (1u64 << a) | (1u64 << b);
        if pair_mask & board_mask != 0 {
            continue;
        }
        combos.push(combo as u16);
        weights.push(w);
    }
    (combos, weights)
}

#[cfg(test)]
mod tests {
    use super::super::cards::{card_from_str, combo_index};
    use super::*;

    fn board(s: &[&str]) -> [u8; 5] {
        let mut out = [0u8; 5];
        for (i, c) in s.iter().enumerate() {
            out[i] = card_from_str(c).unwrap();
        }
        out
    }

    fn range_with(combos: &[(u8, u8)]) -> Vec<f32> {
        let mut v = vec![0.0f32; NUM_COMBOS];
        for &(a, b) in combos {
            v[combo_index(a, b)] = 1.0;
        }
        v
    }

    #[test]
    fn nut_set_beats_underpair() {
        let b = board(&["Ad", "Kh", "7s", "3c", "2d"]);
        let hero = range_with(&[(card_from_str("As").unwrap(), card_from_str("Ac").unwrap())]);
        let villain = range_with(&[(card_from_str("2s").unwrap(), card_from_str("2h").unwrap())]);
        let table = compute_showdown_table(b, &hero, &villain);
        assert_eq!(table.n_hero, 1);
        assert_eq!(table.n_villain, 1);
        assert_eq!(table.outcome_at(0, 0), WIN_HERO);
    }

    #[test]
    fn board_conflict_combos_filtered() {
        let b = board(&["Ad", "Kh", "7s", "3c", "2d"]);
        // AdAh would conflict with Ad on board → filtered out
        let hero = range_with(&[
            (card_from_str("Ad").unwrap(), card_from_str("Ah").unwrap()),
            (card_from_str("As").unwrap(), card_from_str("Ac").unwrap()),
        ]);
        let villain = range_with(&[(card_from_str("2s").unwrap(), card_from_str("2h").unwrap())]);
        let table = compute_showdown_table(b, &hero, &villain);
        assert_eq!(table.n_hero, 1, "Ad-conflict combo should be filtered");
    }

    #[test]
    fn pair_card_conflict_marked() {
        let b = board(&["Ad", "Kh", "7s", "3c", "2d"]);
        // hero AsAc, villain AcQc → both contain Ac → conflict
        let hero = range_with(&[(card_from_str("As").unwrap(), card_from_str("Ac").unwrap())]);
        let villain = range_with(&[(card_from_str("Ac").unwrap(), card_from_str("Qc").unwrap())]);
        let table = compute_showdown_table(b, &hero, &villain);
        assert_eq!(table.outcome_at(0, 0), CONFLICT);
    }
}
