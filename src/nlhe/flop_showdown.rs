//! Per-(turn, river) showdown matrices for the flop HU subgame.
//! Mirrors `python/nlhe/flop_showdown.py`.

use std::collections::HashMap;

use super::cards::{combo_cards, NUM_CARDS, NUM_COMBOS};
use super::hand_eval::evaluate_seven;
use super::multi_cfr::MultiBoardLookup;
use super::showdown::{CONFLICT, WIN_HERO, WIN_TIE, WIN_VILLAIN};
use super::tree::ShowdownKey;

#[derive(Debug)]
pub struct FlopShowdown {
    pub board_3: [u8; 3],
    pub turn_cards: Vec<u8>,
    pub hero_combos: Vec<u16>,
    pub hero_weights: Vec<f32>,
    pub villain_combos: Vec<u16>,
    pub villain_weights: Vec<f32>,
    /// Keyed by (turn_card, river_card)
    pub outcomes_by_runout: HashMap<(u8, u8), Vec<i8>>,
    pub n_hero: usize,
    pub n_villain: usize,
}

impl FlopShowdown {
    #[inline]
    pub fn outcome_at(&self, turn: u8, river: u8, h: usize, v: usize) -> i8 {
        self.outcomes_by_runout[&(turn, river)][h * self.n_villain + v]
    }
}

impl MultiBoardLookup for FlopShowdown {
    fn n_hero(&self) -> usize { self.n_hero }
    fn n_villain(&self) -> usize { self.n_villain }
    fn hero_combos(&self) -> &[u16] { &self.hero_combos }
    fn hero_weights(&self) -> &[f32] { &self.hero_weights }
    fn villain_combos(&self) -> &[u16] { &self.villain_combos }
    fn villain_weights(&self) -> &[f32] { &self.villain_weights }
    fn outcome(&self, key: ShowdownKey, h: usize, v: usize) -> i8 {
        match key {
            ShowdownKey::TurnRiver(t, r) => self.outcome_at(t, r, h, v),
            _ => panic!("flop solver expects ShowdownKey::TurnRiver, got {:?}", key),
        }
    }
    fn hero_pair(&self, h: usize) -> (u8, u8) {
        combo_cards(self.hero_combos[h] as usize)
    }
    fn villain_pair(&self, v: usize) -> (u8, u8) {
        combo_cards(self.villain_combos[v] as usize)
    }
    fn pair_weight_avg(&self) -> f32 {
        let n_runouts = self.outcomes_by_runout.len() as f32;
        if n_runouts == 0.0 { return 0.0; }
        let mut total = 0.0f32;
        for matrix in self.outcomes_by_runout.values() {
            for i in 0..self.n_hero {
                let hw = self.hero_weights[i];
                for j in 0..self.n_villain {
                    if matrix[i * self.n_villain + j] == CONFLICT { continue; }
                    total += hw * self.villain_weights[j];
                }
            }
        }
        total / n_runouts
    }
}

pub fn compute_flop_showdown(
    board_3: [u8; 3],
    hero_range: &[f32],
    villain_range: &[f32],
) -> FlopShowdown {
    compute_flop_showdown_subset(board_3, hero_range, villain_range, None, None)
}

/// Same as `compute_flop_showdown`, but if `turn_subset` / `river_subset`
/// are provided, only outcome matrices for the (turn, river) pairs in those
/// subsets are computed. This matches the abstraction used by
/// `build_flop_tree_subset` and avoids the 90% waste of computing 2352
/// matrices when CFR only uses K_t × K_r of them.
pub fn compute_flop_showdown_subset(
    board_3: [u8; 3],
    hero_range: &[f32],
    villain_range: &[f32],
    turn_subset: Option<&[u8]>,
    river_subset: Option<&[u8]>,
) -> FlopShowdown {
    assert_eq!(hero_range.len(), NUM_COMBOS);
    assert_eq!(villain_range.len(), NUM_COMBOS);
    let board_mask: u64 = board_3.iter().fold(0u64, |a, &c| a | (1u64 << c));
    assert_eq!(board_mask.count_ones(), 3);

    let all_remaining: Vec<u8> = (0..NUM_CARDS as u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
    // turn_cards stored on the table is the FULL remaining list (used by lookups),
    // but enumeration of (turn, river) pairs respects the subsets if given.
    let turn_cards = all_remaining.clone();
    let turn_iter: Vec<u8> = match turn_subset {
        None => all_remaining.clone(),
        Some(s) => s.to_vec(),
    };
    let river_iter_base: Vec<u8> = match river_subset {
        None => all_remaining.clone(),
        Some(s) => s.to_vec(),
    };

    let (hero_combos, hero_weights) = filter_range(hero_range, board_mask);
    let (villain_combos, villain_weights) = filter_range(villain_range, board_mask);
    assert!(!hero_combos.is_empty() && !villain_combos.is_empty());

    let n_hero = hero_combos.len();
    let n_villain = villain_combos.len();

    let hero_pairs: Vec<(u8, u8)> = hero_combos.iter().map(|&c| combo_cards(c as usize)).collect();
    let villain_pairs: Vec<(u8, u8)> = villain_combos.iter().map(|&c| combo_cards(c as usize)).collect();

    let mut outcomes_by_runout: HashMap<(u8, u8), Vec<i8>> = HashMap::new();
    for &tc in &turn_iter {
        for &rc in &river_iter_base {
            if rc == tc { continue; }
            let mut matrix = vec![CONFLICT; n_hero * n_villain];
            for (i, &(ha, hb)) in hero_pairs.iter().enumerate() {
                if ha == tc || hb == tc || ha == rc || hb == rc { continue; }
                let mut seven = [0u8; 7];
                seven[..3].copy_from_slice(&board_3);
                seven[3] = tc; seven[4] = rc; seven[5] = ha; seven[6] = hb;
                let hrank = evaluate_seven(seven);
                let hset: u64 = (1u64 << ha) | (1u64 << hb);
                for (j, &(va, vb)) in villain_pairs.iter().enumerate() {
                    if va == tc || vb == tc || va == rc || vb == rc { continue; }
                    if hset & ((1u64 << va) | (1u64 << vb)) != 0 { continue; }
                    let mut seven_v = [0u8; 7];
                    seven_v[..3].copy_from_slice(&board_3);
                    seven_v[3] = tc; seven_v[4] = rc; seven_v[5] = va; seven_v[6] = vb;
                    let vrank = evaluate_seven(seven_v);
                    matrix[i * n_villain + j] = if hrank > vrank { WIN_HERO } else if hrank < vrank { WIN_VILLAIN } else { WIN_TIE };
                }
            }
            outcomes_by_runout.insert((tc, rc), matrix);
        }
    }

    FlopShowdown {
        board_3, turn_cards, hero_combos, hero_weights, villain_combos, villain_weights,
        outcomes_by_runout, n_hero, n_villain,
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
    use super::super::multi_cfr::solve_multi;
    use super::super::range_parser::parse_range;
    use super::super::tree::build_flop_tree;
    use super::*;

    fn b3(s: &[&str]) -> [u8; 3] {
        let mut out = [0u8; 3];
        for (i, c) in s.iter().enumerate() { out[i] = card_from_str(c).unwrap(); }
        out
    }

    #[test]
    fn flop_showdown_runout_count() {
        let board = b3(&["Ad", "Kh", "7s"]);
        let hero = parse_range("AsAc").unwrap();
        let villain = parse_range("KsKc").unwrap();
        let table = compute_flop_showdown(board, &hero, &villain);
        // 49 turn cards, 48 rivers each = 2352 runouts
        assert_eq!(table.outcomes_by_runout.len(), 49 * 48);
    }

    #[test]
    fn flop_solver_runs() {
        let board = b3(&["Ad", "Kh", "7s"]);
        let hero = parse_range("AsAc").unwrap();
        let villain = parse_range("3c2c").unwrap();
        let table = compute_flop_showdown(board, &hero, &villain);
        let mut root = build_flop_tree(board, 20.0, (40.0, 40.0), 0, 1, 1, 1);
        let result = solve_multi(&mut root, &table, (40.0, 40.0), 5, None, None);
        assert!(result.hero_value > 5.0, "hero_value = {}", result.hero_value);
    }
}
