//! Range string parser, matching `python/nlhe/range_parser.py` syntax.
//!
//! Supported tokens (case-insensitive ranks, lowercase suits):
//!   AA              pocket pair (6 combos)
//!   AKs             suited (4 combos)
//!   AKo             offsuit (12 combos)
//!   AK              suited + offsuit (16 combos)
//!   TT+             pair plus: TT,JJ,QQ,KK,AA
//!   88-TT           pair range
//!   A2s+            suited connector plus: A2s..AKs
//!   A5o+            offsuit plus
//!   T9s-76s         descending suited run
//!   AhKs            specific combo (explicit cards)
//!   AA:0.5          weight override (default 1.0)
//!
//! Tokens separated by commas or whitespace. Later tokens for the same combo
//! override earlier ones.

use std::collections::HashMap;

use super::cards::{combo_index, NUM_COMBOS, RANKS, SUITS};

#[derive(Debug, Clone)]
pub struct RangeParseError(pub String);

impl std::fmt::Display for RangeParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "range parse error: {}", self.0)
    }
}
impl std::error::Error for RangeParseError {}

pub fn parse_range(s: &str) -> Result<Vec<f32>, RangeParseError> {
    let mut weights: HashMap<usize, f32> = HashMap::new();

    for raw in s.split(|c: char| c == ',' || c.is_whitespace()) {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let (token, weight) = split_weight(raw)?;
        let combos = expand_token(token)?;
        if combos.is_empty() {
            return Err(RangeParseError(format!("token matched no combos: {raw}")));
        }
        for c in combos {
            weights.insert(c, weight);
        }
    }

    let mut out = vec![0.0f32; NUM_COMBOS];
    for (i, w) in weights {
        out[i] = w;
    }
    Ok(out)
}

fn split_weight(raw: &str) -> Result<(&str, f32), RangeParseError> {
    if let Some(idx) = raw.find(':') {
        let (token, rest) = raw.split_at(idx);
        let weight: f32 = rest[1..]
            .trim()
            .parse()
            .map_err(|_| RangeParseError(format!("bad weight in {raw}")))?;
        Ok((token, weight))
    } else {
        Ok((raw, 1.0))
    }
}

fn rank_value(c: u8) -> Option<u8> {
    let upper = c.to_ascii_uppercase();
    RANKS.iter().position(|&b| b == upper).map(|i| i as u8 + 2)
}

fn suit_idx(c: u8) -> Option<u8> {
    let lower = c.to_ascii_lowercase();
    SUITS.iter().position(|&b| b == lower).map(|i| i as u8)
}

fn card_idx_from(rank_val: u8, suit_idx: u8) -> u8 {
    (rank_val - 2) * 4 + suit_idx
}

fn expand_token(token: &str) -> Result<Vec<usize>, RangeParseError> {
    let bytes = token.as_bytes();
    // Specific combo: 4 chars, two valid cards (e.g. "AhKs")
    if bytes.len() == 4 {
        if let (Some(r1), Some(s1), Some(r2), Some(s2)) = (
            rank_value(bytes[0]),
            suit_idx(bytes[1]),
            rank_value(bytes[2]),
            suit_idx(bytes[3]),
        ) {
            let c1 = card_idx_from(r1, s1);
            let c2 = card_idx_from(r2, s2);
            if c1 == c2 {
                return Err(RangeParseError(format!("duplicate cards in {token}")));
            }
            return Ok(vec![combo_index(c1, c2)]);
        }
    }

    // Range tokens like "88-TT", "T9s-76s"
    if let Some(idx) = token.find('-') {
        let (lo_tok, hi_tok) = token.split_at(idx);
        let hi_tok = &hi_tok[1..];
        return expand_range(lo_tok, hi_tok);
    }

    // Plus tokens: "TT+", "A2s+", "A5o+"
    if let Some(stripped) = token.strip_suffix('+') {
        return expand_plus(stripped);
    }

    // Class tokens: "AA", "AKs", "AKo", "AK"
    expand_class(token)
}

fn expand_class(token: &str) -> Result<Vec<usize>, RangeParseError> {
    let bytes = token.as_bytes();
    if bytes.len() == 2 {
        // Pair "AA" or both ranks unspecified suit "AK" → suited+offsuit
        let r1 = rank_value(bytes[0]);
        let r2 = rank_value(bytes[1]);
        match (r1, r2) {
            (Some(a), Some(b)) if a == b => Ok(pair_combos(a)),
            (Some(a), Some(b)) => {
                let mut out = suited_combos(a, b);
                out.extend(offsuit_combos(a, b));
                Ok(out)
            }
            _ => Err(RangeParseError(format!("bad token: {token}"))),
        }
    } else if bytes.len() == 3 {
        // "AKs" or "AKo"
        let r1 = rank_value(bytes[0]).ok_or_else(|| RangeParseError(format!("bad rank in {token}")))?;
        let r2 = rank_value(bytes[1]).ok_or_else(|| RangeParseError(format!("bad rank in {token}")))?;
        match bytes[2].to_ascii_lowercase() {
            b's' => Ok(suited_combos(r1, r2)),
            b'o' => Ok(offsuit_combos(r1, r2)),
            _ => Err(RangeParseError(format!("expected s or o suffix in {token}"))),
        }
    } else {
        Err(RangeParseError(format!("unrecognized token: {token}")))
    }
}

fn expand_plus(token: &str) -> Result<Vec<usize>, RangeParseError> {
    let bytes = token.as_bytes();
    if bytes.len() == 2 {
        // Pair plus: "TT+" → TT, JJ, ..., AA
        let r1 = rank_value(bytes[0]);
        let r2 = rank_value(bytes[1]);
        if r1 != r2 || r1.is_none() {
            return Err(RangeParseError(format!("expected pair in plus token: {token}")));
        }
        let r = r1.unwrap();
        let mut out = Vec::new();
        for v in r..=14u8 {
            out.extend(pair_combos(v));
        }
        return Ok(out);
    }
    if bytes.len() == 3 {
        // "AKs+" "A2s+" "A5o+"
        let r1 = rank_value(bytes[0]).ok_or_else(|| RangeParseError(format!("bad rank in {token}")))?;
        let r2 = rank_value(bytes[1]).ok_or_else(|| RangeParseError(format!("bad rank in {token}")))?;
        let suffix = bytes[2].to_ascii_lowercase();
        if r1 <= r2 {
            return Err(RangeParseError(format!("expected r1 > r2 in {token}")));
        }
        let mut out = Vec::new();
        for v in r2..r1 {
            match suffix {
                b's' => out.extend(suited_combos(r1, v)),
                b'o' => out.extend(offsuit_combos(r1, v)),
                _ => return Err(RangeParseError(format!("bad suffix in {token}"))),
            }
        }
        return Ok(out);
    }
    Err(RangeParseError(format!("unrecognized plus token: {token}")))
}

fn expand_range(lo_tok: &str, hi_tok: &str) -> Result<Vec<usize>, RangeParseError> {
    let lo = lo_tok.as_bytes();
    let hi = hi_tok.as_bytes();
    // Pair range: "88-TT"
    if lo.len() == 2 && hi.len() == 2 {
        let lo_r = rank_value(lo[0]);
        if rank_value(lo[1]) != lo_r {
            return Err(RangeParseError(format!("expected pair in {lo_tok}")));
        }
        let hi_r = rank_value(hi[0]);
        if rank_value(hi[1]) != hi_r {
            return Err(RangeParseError(format!("expected pair in {hi_tok}")));
        }
        let lo_r = lo_r.unwrap();
        let hi_r = hi_r.unwrap();
        let (lo_r, hi_r) = if lo_r > hi_r { (hi_r, lo_r) } else { (lo_r, hi_r) };
        let mut out = Vec::new();
        for v in lo_r..=hi_r {
            out.extend(pair_combos(v));
        }
        return Ok(out);
    }
    // Suited/offsuit run: "T9s-76s"
    if lo.len() == 3 && hi.len() == 3 && lo[2] == hi[2] {
        let suffix = lo[2].to_ascii_lowercase();
        let lo_a = rank_value(lo[0]).ok_or_else(|| RangeParseError(format!("bad rank")))?;
        let lo_b = rank_value(lo[1]).ok_or_else(|| RangeParseError(format!("bad rank")))?;
        let hi_a = rank_value(hi[0]).ok_or_else(|| RangeParseError(format!("bad rank")))?;
        let hi_b = rank_value(hi[1]).ok_or_else(|| RangeParseError(format!("bad rank")))?;
        // Same gap
        let gap_lo = lo_a as i8 - lo_b as i8;
        let gap_hi = hi_a as i8 - hi_b as i8;
        if gap_lo != gap_hi || gap_lo <= 0 {
            return Err(RangeParseError(format!(
                "gap mismatch or non-positive in {lo_tok}-{hi_tok}"
            )));
        }
        let (start, end) = if lo_b > hi_b { (hi_b, lo_b) } else { (lo_b, hi_b) };
        let mut out = Vec::new();
        for low in start..=end {
            let high = low as i8 + gap_lo;
            if high < 2 || high > 14 {
                continue;
            }
            match suffix {
                b's' => out.extend(suited_combos(high as u8, low)),
                b'o' => out.extend(offsuit_combos(high as u8, low)),
                _ => return Err(RangeParseError(format!("bad suffix in {lo_tok}-{hi_tok}"))),
            }
        }
        return Ok(out);
    }
    Err(RangeParseError(format!("unsupported range token: {lo_tok}-{hi_tok}")))
}

fn pair_combos(rank: u8) -> Vec<usize> {
    let mut out = Vec::new();
    for s1 in 0..4u8 {
        for s2 in (s1 + 1)..4u8 {
            let c1 = card_idx_from(rank, s1);
            let c2 = card_idx_from(rank, s2);
            out.push(combo_index(c1, c2));
        }
    }
    out
}

fn suited_combos(r1: u8, r2: u8) -> Vec<usize> {
    if r1 == r2 {
        return Vec::new();
    }
    let (hi, lo) = if r1 > r2 { (r1, r2) } else { (r2, r1) };
    (0..4u8)
        .map(|s| combo_index(card_idx_from(hi, s), card_idx_from(lo, s)))
        .collect()
}

fn offsuit_combos(r1: u8, r2: u8) -> Vec<usize> {
    if r1 == r2 {
        return Vec::new();
    }
    let (hi, lo) = if r1 > r2 { (r1, r2) } else { (r2, r1) };
    let mut out = Vec::new();
    for s1 in 0..4u8 {
        for s2 in 0..4u8 {
            if s1 == s2 {
                continue;
            }
            out.push(combo_index(card_idx_from(hi, s1), card_idx_from(lo, s2)));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_nonzero(v: &[f32]) -> usize {
        v.iter().filter(|&&x| x > 0.0).count()
    }

    #[test]
    fn pocket_pair_has_six_combos() {
        let v = parse_range("AA").unwrap();
        assert_eq!(count_nonzero(&v), 6);
    }

    #[test]
    fn suited_has_four_combos() {
        let v = parse_range("AKs").unwrap();
        assert_eq!(count_nonzero(&v), 4);
    }

    #[test]
    fn offsuit_has_twelve_combos() {
        let v = parse_range("AKo").unwrap();
        assert_eq!(count_nonzero(&v), 12);
    }

    #[test]
    fn unsuited_token_combines_both() {
        let v = parse_range("AK").unwrap();
        assert_eq!(count_nonzero(&v), 16);
    }

    #[test]
    fn pair_plus() {
        let v = parse_range("TT+").unwrap();
        // TT, JJ, QQ, KK, AA = 5 ranks × 6 combos = 30
        assert_eq!(count_nonzero(&v), 30);
    }

    #[test]
    fn pair_range() {
        let v = parse_range("88-TT").unwrap();
        // 88, 99, TT
        assert_eq!(count_nonzero(&v), 18);
    }

    #[test]
    fn suited_plus() {
        let v = parse_range("A2s+").unwrap();
        // A2s..AKs = 12 hands × 4 combos = 48
        assert_eq!(count_nonzero(&v), 48);
    }

    #[test]
    fn weighted_token() {
        let v = parse_range("AA:0.5").unwrap();
        let nz: Vec<f32> = v.iter().copied().filter(|&x| x > 0.0).collect();
        assert_eq!(nz.len(), 6);
        for w in nz {
            assert!((w - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn multi_token() {
        let v = parse_range("AA, KK, AKs").unwrap();
        assert_eq!(count_nonzero(&v), 6 + 6 + 4);
    }

    #[test]
    fn specific_combo() {
        let v = parse_range("AhKs").unwrap();
        assert_eq!(count_nonzero(&v), 1);
    }

    #[test]
    fn descending_suited_run() {
        let v = parse_range("T9s-76s").unwrap();
        // T9s, 98s, 87s, 76s = 4 hands × 4 combos = 16
        assert_eq!(count_nonzero(&v), 16);
    }
}
