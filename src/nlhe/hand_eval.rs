//! 5/7-card hand evaluator. Output is a totally-ordered `HandRank` value such
//! that better hands compare greater. Matches `python/nlhe/hand_eval.py`
//! semantics exactly.
//!
//! The HandRank is encoded as a packed `u32`:
//!   bits 28..32: category (0..=8 per CATEGORY_*)
//!   bits 20..28, 16..20, 12..16, 8..12, 4..8, 0..4: tiebreaker rank values
//!
//! Tiebreakers occupy 4 bits each (rank 2..14 fits in 4 bits if we encode
//! 2 → 0, 3 → 1, ..., 14 → 12, but we keep raw rank values and use 4 bits
//! anyway since 14 > 15 no — 14 fits in 4 bits exactly). This gives a single
//! `u32` that sorts hands correctly.

use super::cards::{rank_of, suit_of};

pub const CATEGORY_HIGH_CARD: u8 = 0;
pub const CATEGORY_PAIR: u8 = 1;
pub const CATEGORY_TWO_PAIR: u8 = 2;
pub const CATEGORY_TRIPS: u8 = 3;
pub const CATEGORY_STRAIGHT: u8 = 4;
pub const CATEGORY_FLUSH: u8 = 5;
pub const CATEGORY_FULL_HOUSE: u8 = 6;
pub const CATEGORY_QUADS: u8 = 7;
pub const CATEGORY_STRAIGHT_FLUSH: u8 = 8;

pub type HandRank = u32;

#[inline]
fn pack(category: u8, t: &[u8]) -> HandRank {
    let mut r: u32 = (category as u32) << 28;
    let mut shift = 24;
    for &v in t.iter() {
        r |= (v as u32) << shift;
        if shift < 4 {
            break;
        }
        shift -= 4;
    }
    r
}

/// Rank a 5-card hand. Each card is a u8 in 0..52.
pub fn rank_five(cards: [u8; 5]) -> HandRank {
    let mut ranks = [0u8; 5];
    for i in 0..5 {
        ranks[i] = rank_of(cards[i]);
    }
    // sort descending
    ranks.sort_unstable_by(|a, b| b.cmp(a));

    // Suit equality test
    let s0 = suit_of(cards[0]);
    let is_flush = (1..5).all(|i| suit_of(cards[i]) == s0);

    // Straight detection
    let straight_high = straight_high(&ranks);

    // Group ranks (count duplicates). Sort groups by (count desc, rank desc).
    let mut counts: [u8; 15] = [0; 15]; // index = rank value (2..14)
    for &r in ranks.iter() {
        counts[r as usize] += 1;
    }
    // Build (count, rank) groups, sorted by count desc then rank desc
    let mut groups: Vec<(u8, u8)> = (2u8..=14u8)
        .filter(|&r| counts[r as usize] > 0)
        .map(|r| (counts[r as usize], r))
        .collect();
    groups.sort_by(|a, b| b.cmp(a));

    if is_flush && straight_high.is_some() {
        return pack(CATEGORY_STRAIGHT_FLUSH, &[straight_high.unwrap()]);
    }
    if groups[0].0 == 4 {
        let quad = groups[0].1;
        let kicker = ranks.iter().copied().find(|&r| r != quad).unwrap();
        return pack(CATEGORY_QUADS, &[quad, kicker]);
    }
    if groups[0].0 == 3 && groups.len() > 1 && groups[1].0 == 2 {
        return pack(CATEGORY_FULL_HOUSE, &[groups[0].1, groups[1].1]);
    }
    if is_flush {
        return pack(CATEGORY_FLUSH, &ranks);
    }
    if let Some(h) = straight_high {
        return pack(CATEGORY_STRAIGHT, &[h]);
    }
    if groups[0].0 == 3 {
        let trips = groups[0].1;
        let kickers: Vec<u8> = ranks.iter().copied().filter(|&r| r != trips).collect();
        let mut t = vec![trips];
        t.extend(kickers);
        return pack(CATEGORY_TRIPS, &t);
    }
    if groups[0].0 == 2 && groups.len() > 1 && groups[1].0 == 2 {
        let mut pairs = vec![groups[0].1, groups[1].1];
        pairs.sort_by(|a, b| b.cmp(a));
        let kicker = ranks.iter().copied().find(|&r| r != pairs[0] && r != pairs[1]).unwrap();
        return pack(CATEGORY_TWO_PAIR, &[pairs[0], pairs[1], kicker]);
    }
    if groups[0].0 == 2 {
        let pair = groups[0].1;
        let kickers: Vec<u8> = ranks.iter().copied().filter(|&r| r != pair).collect();
        let mut t = vec![pair];
        t.extend(kickers);
        return pack(CATEGORY_PAIR, &t);
    }
    pack(CATEGORY_HIGH_CARD, &ranks)
}

fn straight_high(ranks_desc: &[u8; 5]) -> Option<u8> {
    // Wheel: A-2-3-4-5
    if ranks_desc == &[14, 5, 4, 3, 2] {
        return Some(5);
    }
    // All distinct, span of 4
    let mut prev = ranks_desc[0];
    for i in 1..5 {
        if ranks_desc[i] >= prev || prev - ranks_desc[i] != 1 {
            return None;
        }
        prev = ranks_desc[i];
    }
    Some(ranks_desc[0])
}

/// Best 5-card rank from 7 cards. O(C(7,5)) = 21 evaluations.
pub fn evaluate_seven(cards: [u8; 7]) -> HandRank {
    let mut best: HandRank = 0;
    // 21 combinations of 5 from 7
    for i in 0..3 {
        for j in (i + 1)..4 {
            for k in (j + 1)..5 {
                for l in (k + 1)..6 {
                    for m in (l + 1)..7 {
                        let r = rank_five([cards[i], cards[j], cards[k], cards[l], cards[m]]);
                        if r > best {
                            best = r;
                        }
                    }
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::super::cards::card_from_str;
    use super::*;

    fn cards5(s: &[&str]) -> [u8; 5] {
        let mut out = [0u8; 5];
        for (i, c) in s.iter().enumerate() {
            out[i] = card_from_str(c).unwrap();
        }
        out
    }

    fn cards7(s: &[&str]) -> [u8; 7] {
        let mut out = [0u8; 7];
        for (i, c) in s.iter().enumerate() {
            out[i] = card_from_str(c).unwrap();
        }
        out
    }

    fn category_of(r: HandRank) -> u8 {
        ((r >> 28) & 0xF) as u8
    }

    #[test]
    fn high_card() {
        let r = rank_five(cards5(&["As", "9h", "7d", "4c", "2s"]));
        assert_eq!(category_of(r), CATEGORY_HIGH_CARD);
    }

    #[test]
    fn pair_beats_high_card() {
        let high = rank_five(cards5(&["As", "Kh", "7d", "4c", "2s"]));
        let pair = rank_five(cards5(&["2s", "2h", "Td", "8c", "5s"]));
        assert!(pair > high);
    }

    #[test]
    fn two_pair_beats_pair() {
        let pair = rank_five(cards5(&["As", "Ah", "Td", "8c", "5s"]));
        let two_pair = rank_five(cards5(&["2s", "2h", "5d", "5c", "Ks"]));
        assert!(two_pair > pair);
    }

    #[test]
    fn flush_beats_straight() {
        let straight = rank_five(cards5(&["9s", "8h", "7d", "6c", "5s"]));
        let flush = rank_five(cards5(&["As", "Ks", "9s", "5s", "2s"]));
        assert!(flush > straight);
    }

    #[test]
    fn full_house_beats_flush() {
        let flush = rank_five(cards5(&["As", "Ks", "9s", "5s", "2s"]));
        let boat = rank_five(cards5(&["3s", "3h", "3d", "9c", "9s"]));
        assert!(boat > flush);
    }

    #[test]
    fn quads_beats_full_house() {
        let boat = rank_five(cards5(&["3s", "3h", "3d", "9c", "9s"]));
        let quads = rank_five(cards5(&["7s", "7h", "7d", "7c", "2s"]));
        assert!(quads > boat);
    }

    #[test]
    fn straight_flush_beats_quads() {
        let quads = rank_five(cards5(&["7s", "7h", "7d", "7c", "Ks"]));
        let sf = rank_five(cards5(&["9s", "8s", "7s", "6s", "5s"]));
        assert!(sf > quads);
    }

    #[test]
    fn wheel_straight() {
        let wheel = rank_five(cards5(&["As", "5h", "4d", "3c", "2s"]));
        assert_eq!(category_of(wheel), CATEGORY_STRAIGHT);
        // wheel high is 5
        let regular = rank_five(cards5(&["6s", "5h", "4d", "3c", "2s"]));
        // Regular 6-high straight should beat wheel
        assert!(regular > wheel);
    }

    #[test]
    fn steel_wheel_is_straight_flush() {
        let sw = rank_five(cards5(&["As", "5s", "4s", "3s", "2s"]));
        assert_eq!(category_of(sw), CATEGORY_STRAIGHT_FLUSH);
    }

    #[test]
    fn evaluate_seven_picks_best() {
        // 7 cards: A-K-Q-J-T-9-2 hearts → straight flush A-K-Q-J-T (royal)
        let r = evaluate_seven(cards7(&["Ah", "Kh", "Qh", "Jh", "Th", "9h", "2c"]));
        assert_eq!(category_of(r), CATEGORY_STRAIGHT_FLUSH);
    }
}
