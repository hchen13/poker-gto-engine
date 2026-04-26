//! Per-river showdown matrices for the turn HU subgame.
//!
//! Mirrors `python/nlhe/turn_showdown.py`. One combo-index space (filtered by
//! the 4-card board) is reused across all 48 rivers; combos containing the
//! river card are marked CONFLICT in that river's matrix.

use std::collections::HashMap;

use super::cards::{combo_cards, NUM_CARDS, NUM_COMBOS};
use super::hand_eval::{evaluate_seven, HandRank};
use super::showdown::{CONFLICT, WIN_HERO, WIN_TIE, WIN_VILLAIN};

#[derive(Debug)]
pub struct TurnShowdown {
    pub board_4: [u8; 4],
    pub river_cards: Vec<u8>,
    pub hero_combos: Vec<u16>,
    pub hero_weights: Vec<f32>,
    pub villain_combos: Vec<u16>,
    pub villain_weights: Vec<f32>,
    /// outcomes_by_river[river_card] = row-major outcome matrix
    pub outcomes_by_river: HashMap<u8, Vec<i8>>,
    pub n_hero: usize,
    pub n_villain: usize,
}

impl TurnShowdown {
    #[inline]
    pub fn outcome_at(&self, river: u8, hero_local: usize, villain_local: usize) -> i8 {
        self.outcomes_by_river[&river][hero_local * self.n_villain + villain_local]
    }
}

pub fn compute_turn_showdown(
    board_4: [u8; 4],
    hero_range: &[f32],
    villain_range: &[f32],
) -> TurnShowdown {
    assert_eq!(hero_range.len(), NUM_COMBOS);
    assert_eq!(villain_range.len(), NUM_COMBOS);

    let board_mask: u64 = board_4.iter().fold(0u64, |a, &c| a | (1u64 << c));
    assert_eq!(board_mask.count_ones(), 4, "board must have 4 distinct cards");
    let river_cards: Vec<u8> = (0..NUM_CARDS as u8).filter(|c| board_mask & (1u64 << c) == 0).collect();

    let (hero_combos, hero_weights) = filter_range(hero_range, board_mask);
    let (villain_combos, villain_weights) = filter_range(villain_range, board_mask);
    assert!(!hero_combos.is_empty() && !villain_combos.is_empty(), "ranges have no compatible combos");

    let n_hero = hero_combos.len();
    let n_villain = villain_combos.len();

    // Precompute hand ranks per (combo, river)
    let hero_pairs: Vec<(u8, u8)> = hero_combos.iter().map(|&c| combo_cards(c as usize)).collect();
    let villain_pairs: Vec<(u8, u8)> = villain_combos.iter().map(|&c| combo_cards(c as usize)).collect();

    let mut hero_ranks: HashMap<(usize, u8), HandRank> = HashMap::new();
    for (i, &(a, b)) in hero_pairs.iter().enumerate() {
        for &r in &river_cards {
            if r == a || r == b { continue; }
            let mut seven = [0u8; 7];
            seven[..4].copy_from_slice(&board_4);
            seven[4] = r;
            seven[5] = a;
            seven[6] = b;
            hero_ranks.insert((i, r), evaluate_seven(seven));
        }
    }
    let mut villain_ranks: HashMap<(usize, u8), HandRank> = HashMap::new();
    for (j, &(a, b)) in villain_pairs.iter().enumerate() {
        for &r in &river_cards {
            if r == a || r == b { continue; }
            let mut seven = [0u8; 7];
            seven[..4].copy_from_slice(&board_4);
            seven[4] = r;
            seven[5] = a;
            seven[6] = b;
            villain_ranks.insert((j, r), evaluate_seven(seven));
        }
    }

    let mut outcomes_by_river: HashMap<u8, Vec<i8>> = HashMap::new();
    for &r in &river_cards {
        let mut matrix = vec![CONFLICT; n_hero * n_villain];
        for (i, &(ha, hb)) in hero_pairs.iter().enumerate() {
            if ha == r || hb == r { continue; }
            let hrank = hero_ranks[&(i, r)];
            let hset: u64 = (1u64 << ha) | (1u64 << hb);
            for (j, &(va, vb)) in villain_pairs.iter().enumerate() {
                if va == r || vb == r { continue; }
                if hset & ((1u64 << va) | (1u64 << vb)) != 0 { continue; }
                let vrank = villain_ranks[&(j, r)];
                matrix[i * n_villain + j] = if hrank > vrank { WIN_HERO } else if hrank < vrank { WIN_VILLAIN } else { WIN_TIE };
            }
        }
        outcomes_by_river.insert(r, matrix);
    }

    TurnShowdown {
        board_4,
        river_cards,
        hero_combos,
        hero_weights,
        villain_combos,
        villain_weights,
        outcomes_by_river,
        n_hero,
        n_villain,
    }
}

fn filter_range(range: &[f32], board_mask: u64) -> (Vec<u16>, Vec<f32>) {
    let mut combos = Vec::new();
    let mut weights = Vec::new();
    for combo in 0..NUM_COMBOS {
        let w = range[combo];
        if w <= 0.0 { continue; }
        let (a, b) = combo_cards(combo);
        let pair_mask = (1u64 << a) | (1u64 << b);
        if pair_mask & board_mask != 0 { continue; }
        combos.push(combo as u16);
        weights.push(w);
    }
    (combos, weights)
}

#[cfg(test)]
mod tests {
    use super::super::cards::card_from_str;
    use super::super::range_parser::parse_range;
    use super::*;

    fn b4(s: &[&str]) -> [u8; 4] {
        let mut out = [0u8; 4];
        for (i, c) in s.iter().enumerate() { out[i] = card_from_str(c).unwrap(); }
        out
    }

    #[test]
    fn turn_showdown_has_48_rivers() {
        let board = b4(&["Ad", "Kh", "7s", "3c"]);
        let hero = parse_range("AsAc").unwrap();
        let villain = parse_range("2s2h").unwrap();
        let table = compute_turn_showdown(board, &hero, &villain);
        assert_eq!(table.river_cards.len(), 48);
        assert_eq!(table.outcomes_by_river.len(), 48);
    }

    #[test]
    fn ace_river_makes_villain_set() {
        // Hero AsAc beats 2s2h on Ad-Kh-7s-3c-x for most x except: any 2 gives villain set.
        // Pick a 2 river and verify outcome flips.
        let board = b4(&["Ad", "Kh", "7s", "3c"]);
        let hero = parse_range("AsAc").unwrap();
        let villain = parse_range("2s2h").unwrap();
        let table = compute_turn_showdown(board, &hero, &villain);
        // 2d: with 2s2h hole and 2d on river, villain has 222 trips. AsAc has AAA trips. AAA > 222.
        // So hero still wins.
        let two_d = card_from_str("2d").unwrap();
        // hero AsAc vs villain 2s2h: with board Ad-Kh-7s-3c-2d, hero: AAA/3 kicker; villain: 222 trips with A kicker
        // Hand evals: trips of A vs trips of 2 → A trips win.
        assert_eq!(table.outcome_at(two_d, 0, 0), WIN_HERO);
    }
}
