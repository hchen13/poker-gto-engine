//! Public-card abstraction for chance nodes.
//!
//! Two strategies provided:
//!   1. **rank_buckets** — group remaining cards by rank (max 13 buckets).
//!      Loses suit info, but exact for non-flush boards. Fastest to compute.
//!   2. **equity_buckets** — group cards by hero's average equity vs villain
//!      on this card (E[HS]-like signature). Better for boards with flush
//!      potential. Slower (one full equity calc per candidate card).
//!
//! Both return a list of buckets and recommended representative cards.

use std::collections::BTreeMap;

/// Group `cards` by rank value (2..=14). Order of returned buckets is
/// ascending by rank. Each bucket holds all suits of that rank present
/// in the input.
pub fn rank_buckets(cards: &[u8]) -> Vec<Vec<u8>> {
    let mut by_rank: BTreeMap<u8, Vec<u8>> = BTreeMap::new();
    for &c in cards {
        let r = (c / 4) + 2;
        by_rank.entry(r).or_default().push(c);
    }
    by_rank.into_values().collect()
}

/// Pick one representative card per bucket (the lexicographically smallest).
pub fn pick_representatives(buckets: &[Vec<u8>]) -> Vec<u8> {
    buckets.iter().filter_map(|b| b.iter().min().copied()).collect()
}

/// Convenience: rank-bucket the remaining cards and return one rep per bucket.
pub fn rank_subsample(cards: &[u8]) -> Vec<u8> {
    pick_representatives(&rank_buckets(cards))
}

/// Enumerate all 1755 suit-isomorphism-canonical flop representatives.
///
/// Two flops are "suit-isomorphic" if you can permute (spade, heart, diamond,
/// club) labels to map one to the other. Under this equivalence there are
/// exactly 1755 distinct flop classes in No-Limit Hold'em:
///   - 3 distinct ranks: C(13,3) × 5 suit patterns = 1430
///     (1 rainbow + 3 two-tone variants by which 2 ranks share + 1 monotone)
///   - 2 ranks (pair + kicker): 13 × 12 × 2 patterns = 312
///     (kicker-matches-pair-suit vs kicker-different-suit)
///   - trip (all 3 same rank): 13 × 1 = 13
///
/// For each class we emit one canonical representative using suits in order
/// (s, h, d) with the lowest-indexed suit assigned to the "shared" group.
/// Returned cards are sorted ascending (rank*4 + suit encoding).
pub fn enumerate_canonical_flops() -> Vec<[u8; 3]> {
    // Suit indices: 0=s, 1=h, 2=d, 3=c
    let mut out: Vec<[u8; 3]> = Vec::with_capacity(1755);
    let card = |r: u8, s: u8| -> u8 { r * 4 + s };
    let push_sorted = |out: &mut Vec<[u8; 3]>, a: u8, b: u8, c: u8| {
        let mut v = [a, b, c];
        v.sort();
        out.push(v);
    };

    // --- 3 distinct ranks ---
    for r1 in 0u8..13 {
        for r2 in (r1 + 1)..13 {
            for r3 in (r2 + 1)..13 {
                // 1. rainbow: r1=s, r2=h, r3=d
                push_sorted(&mut out, card(r1, 0), card(r2, 1), card(r3, 2));
                // 2. r1 & r2 share (s), r3 = h
                push_sorted(&mut out, card(r1, 0), card(r2, 0), card(r3, 1));
                // 3. r1 & r3 share (s), r2 = h
                push_sorted(&mut out, card(r1, 0), card(r2, 1), card(r3, 0));
                // 4. r2 & r3 share (s), r1 = h
                push_sorted(&mut out, card(r1, 1), card(r2, 0), card(r3, 0));
                // 5. monotone
                push_sorted(&mut out, card(r1, 0), card(r2, 0), card(r3, 0));
            }
        }
    }

    // --- Pair + kicker ---
    // pair rank p, kicker rank k (p != k)
    for p in 0u8..13 {
        for k in 0u8..13 {
            if k == p { continue; }
            // Canonical pair uses two lowest suits: p_s, p_h
            // (a) kicker matches pair suit → kicker = k_s (flush-draw pattern)
            push_sorted(&mut out, card(p, 0), card(p, 1), card(k, 0));
            // (b) kicker different suit → kicker = k_d
            push_sorted(&mut out, card(p, 0), card(p, 1), card(k, 2));
        }
    }

    // --- Trip ---
    for t in 0u8..13 {
        // Three suits: s, h, d (any 3 of 4 are isomorphic, canonical uses 0,1,2)
        push_sorted(&mut out, card(t, 0), card(t, 1), card(t, 2));
    }

    debug_assert_eq!(out.len(), 1755, "canonical flop enumeration should yield 1755");
    out
}

/// Canonical iso-class key for any flop. Two flops produce the same key iff
/// they are suit-isomorphic. Used to look up a given flop in a precompute
/// indexed by canonical representatives.
///
/// Algorithm: enumerate all 24 suit permutations; for each apply to the 3 cards,
/// sort, and record the 6-byte signature. Return the lexicographically minimum
/// signature. This is correct under the full symmetric group S_4 on suits.
pub fn flop_iso_key(cards: [u8; 3]) -> [u8; 6] {
    // All 4! = 24 permutations of [0, 1, 2, 3]
    const PERMS: [[u8; 4]; 24] = [
        [0,1,2,3],[0,1,3,2],[0,2,1,3],[0,2,3,1],[0,3,1,2],[0,3,2,1],
        [1,0,2,3],[1,0,3,2],[1,2,0,3],[1,2,3,0],[1,3,0,2],[1,3,2,0],
        [2,0,1,3],[2,0,3,1],[2,1,0,3],[2,1,3,0],[2,3,0,1],[2,3,1,0],
        [3,0,1,2],[3,0,2,1],[3,1,0,2],[3,1,2,0],[3,2,0,1],[3,2,1,0],
    ];
    let mut best: [u8; 6] = [u8::MAX; 6];
    for perm in PERMS.iter() {
        let mut relabeled: [u8; 3] = [0; 3];
        for (i, &c) in cards.iter().enumerate() {
            let r = c / 4;
            let s = c % 4;
            relabeled[i] = r * 4 + perm[s as usize];
        }
        relabeled.sort();
        let mut key = [0u8; 6];
        for (i, &c) in relabeled.iter().enumerate() {
            key[2 * i] = c / 4;
            key[2 * i + 1] = c % 4;
        }
        if key < best { best = key; }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cards::card_from_str;

    #[test]
    fn rank_buckets_groups_by_rank() {
        // 7s, 7h, 7d, 7c, As, 2c
        let cards: Vec<u8> = ["7s", "7h", "7d", "7c", "As", "2c"]
            .iter().map(|s| card_from_str(s).unwrap()).collect();
        let buckets = rank_buckets(&cards);
        // Should have 3 buckets: 2s, 7s, As
        assert_eq!(buckets.len(), 3);
        // Sorted by rank ascending
        assert_eq!((buckets[0][0] / 4) + 2, 2);  // 2x
        assert_eq!((buckets[1][0] / 4) + 2, 7);  // 7x (4 of them)
        assert_eq!(buckets[1].len(), 4);
        assert_eq!((buckets[2][0] / 4) + 2, 14); // Ax
    }

    #[test]
    fn rank_subsample_one_per_rank() {
        let all_cards: Vec<u8> = (0..52).collect();
        let reps = rank_subsample(&all_cards);
        assert_eq!(reps.len(), 13);
    }

    #[test]
    fn canonical_enumeration_has_1755_flops() {
        let flops = enumerate_canonical_flops();
        assert_eq!(flops.len(), 1755);
        // All unique
        let mut seen = std::collections::HashSet::new();
        for f in &flops {
            assert!(seen.insert(*f), "duplicate in enumeration: {:?}", f);
        }
    }

    #[test]
    fn iso_key_matches_canonical_reps() {
        // Each enumerated flop should map to a unique iso key
        let flops = enumerate_canonical_flops();
        let mut keys = std::collections::HashSet::new();
        for f in &flops {
            let k = flop_iso_key(*f);
            assert!(keys.insert(k), "canonical reps should all have distinct iso keys: {:?}", f);
        }
        assert_eq!(keys.len(), 1755);
    }

    #[test]
    fn iso_key_invariant_under_suit_permutation() {
        // Two flops that differ only by suit relabeling must produce the same key
        let f1 = ["3c", "7h", "Ts"].map(|s| card_from_str(s).unwrap());
        let f2 = ["3h", "7d", "Tc"].map(|s| card_from_str(s).unwrap());
        let f3 = ["3s", "7s", "Th"].map(|s| card_from_str(s).unwrap()); // DIFFERENT class (2-tone)
        assert_eq!(flop_iso_key(f1), flop_iso_key(f2));
        assert_ne!(flop_iso_key(f1), flop_iso_key(f3));
    }

    #[test]
    fn every_real_flop_maps_to_canonical_key() {
        // Spot-check: exhaustive test — every C(52,3) flop has its iso key
        // appear in our 1755 canonical set.
        let flops = enumerate_canonical_flops();
        let canonical_keys: std::collections::HashSet<_> =
            flops.iter().map(|f| flop_iso_key(*f)).collect();
        let mut total = 0;
        for a in 0u8..52 {
            for b in (a + 1)..52 {
                for c in (b + 1)..52 {
                    total += 1;
                    let key = flop_iso_key([a, b, c]);
                    assert!(canonical_keys.contains(&key),
                        "flop [{},{},{}] iso_key {:?} not in canonical set", a, b, c, key);
                }
            }
        }
        assert_eq!(total, 22100);
    }
}
