//! Card representation matching `python/nlhe/cards.py`.
//!
//! Cards are encoded as `u8` in the range 0..52, with ordering identical to
//! the Python module: `RANKS = "23456789TJQKA"` (rank values 2..14) crossed
//! with `SUITS = "shdc"`. Card index = rank_idx * 4 + suit_idx where
//! rank_idx = (value - 2). This must match Python so showdown table indices
//! and combo indices are bit-compatible.

pub const NUM_CARDS: usize = 52;
pub const NUM_RANKS: usize = 13;
pub const NUM_SUITS: usize = 4;
pub const NUM_COMBOS: usize = 1326;

pub const RANKS: &[u8; NUM_RANKS] = b"23456789TJQKA";
pub const SUITS: &[u8; NUM_SUITS] = b"shdc";

/// Returns (rank_value, suit_idx) for a card index. Rank value is 2..=14.
#[inline]
pub fn rank_of(card: u8) -> u8 {
    (card / 4) + 2
}

#[inline]
pub fn suit_of(card: u8) -> u8 {
    card % 4
}

/// Parse a 2-character card like "As", "Th", "2c". Returns the 0..52 index.
pub fn card_from_str(s: &str) -> Option<u8> {
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return None;
    }
    let rank_idx = RANKS.iter().position(|&b| b == bytes[0])? as u8;
    let suit_idx = SUITS.iter().position(|&b| b == bytes[1])? as u8;
    Some(rank_idx * 4 + suit_idx)
}

pub fn card_to_string(card: u8) -> String {
    let rank_idx = (card / 4) as usize;
    let suit_idx = (card % 4) as usize;
    let mut s = String::with_capacity(2);
    s.push(RANKS[rank_idx] as char);
    s.push(SUITS[suit_idx] as char);
    s
}

/// Combo index in the canonical 1326-combo enumeration: pairs (a, b) with a < b
/// listed in lex order over (a, b). Identical to Python's `combo_index`.
pub fn combo_index(card_a: u8, card_b: u8) -> usize {
    let (lo, hi) = if card_a < card_b {
        (card_a as usize, card_b as usize)
    } else {
        (card_b as usize, card_a as usize)
    };
    // Closed-form: combos are ordered (0,1),(0,2),...,(0,51),(1,2),...,(50,51).
    // Index of (lo, hi) = sum_{k=0..lo} (51 - k) + (hi - lo - 1)
    //                  = lo * 51 - lo*(lo-1)/2 + (hi - lo - 1)
    lo * 51 - lo * (lo.saturating_sub(1)) / 2 + (hi - lo - 1)
}

/// Human-readable hand label for a combo: "AKs", "QJo", "TT", etc.
/// Higher rank always printed first; "s" suffix if suited, "o" if offsuit, none if pair.
pub fn combo_label(combo: usize) -> String {
    let (a, b) = combo_cards(combo);
    let ri_a = (a / 4) as usize;
    let ri_b = (b / 4) as usize;
    let si_a = a % 4;
    let si_b = b % 4;
    let (r1, r2, s1, s2) = if ri_a >= ri_b { (ri_a, ri_b, si_a, si_b) } else { (ri_b, ri_a, si_b, si_a) };
    let r1c = RANKS[r1] as char;
    let r2c = RANKS[r2] as char;
    if r1 == r2 {
        format!("{}{}", r1c, r2c)
    } else if s1 == s2 {
        format!("{}{}s", r1c, r2c)
    } else {
        format!("{}{}o", r1c, r2c)
    }
}

/// Inverse of `combo_index`. Returns (card_a, card_b) with a < b.
pub fn combo_cards(combo: usize) -> (u8, u8) {
    // Search via cumulative formula. NUM_COMBOS is small enough that this is fine.
    let mut remaining = combo;
    for lo in 0..(NUM_CARDS - 1) {
        let row_size = NUM_CARDS - 1 - lo;
        if remaining < row_size {
            return (lo as u8, (lo + 1 + remaining) as u8);
        }
        remaining -= row_size;
    }
    panic!("combo index out of range: {}", combo);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_suit_roundtrip() {
        for c in 0..NUM_CARDS as u8 {
            let r = rank_of(c);
            let s = suit_of(c);
            assert!((2..=14).contains(&r));
            assert!(s < 4);
            // Confirm reconstruction
            assert_eq!((r - 2) * 4 + s, c);
        }
    }

    #[test]
    fn parse_and_format_cards() {
        // Known mappings derived from Python's RANKS = "23456789TJQKA", SUITS = "shdc"
        // 2s = 0, 2h = 1, 2d = 2, 2c = 3, 3s = 4, ..., As = 48, Ah = 49, Ad = 50, Ac = 51
        assert_eq!(card_from_str("2s"), Some(0));
        assert_eq!(card_from_str("Ac"), Some(51));
        assert_eq!(card_from_str("Th"), Some((8) * 4 + 1)); // T is rank index 8
        assert_eq!(card_to_string(0), "2s");
        assert_eq!(card_to_string(51), "Ac");
    }

    #[test]
    fn combo_index_roundtrip() {
        let mut seen = vec![false; NUM_COMBOS];
        for a in 0..NUM_CARDS as u8 {
            for b in (a + 1)..NUM_CARDS as u8 {
                let i = combo_index(a, b);
                assert!(i < NUM_COMBOS);
                assert!(!seen[i], "duplicate combo index for ({},{})", a, b);
                seen[i] = true;
                let (ra, rb) = combo_cards(i);
                assert_eq!((ra, rb), (a, b));
            }
        }
        assert!(seen.iter().all(|&x| x));
    }

    #[test]
    fn combo_index_matches_python() {
        // Spot-check against Python's INDEX_TO_COMBO entries:
        // First combo (0): (2s=0, 2h=1) → index 0
        assert_eq!(combo_index(0, 1), 0);
        // Last combo (1325): (Ad=50, Ac=51) → index 1325
        assert_eq!(combo_index(50, 51), 1325);
    }
}
