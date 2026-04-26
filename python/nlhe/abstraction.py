"""Card abstraction for river subgames.

On the river there are no more cards to come, so a combo's "hand strength"
reduces to a single scalar per board: its equity vs. a uniform random villain
hand (with board-conflict combos excluded). We call this E[HS].

Bucketing: sort combos by E[HS] and split into ``n_buckets`` equal-weight bins
(weights taken from the side's range). Every combo in the same bucket shares
one CFR infoset — regrets and average strategy are pooled.

This keeps card conflicts correct (they're still resolved per-combo at terminals)
while shrinking the number of infosets from O(n_combos) to O(n_buckets).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .cards import INDEX_TO_CARD, INDEX_TO_COMBO, NUM_COMBOS, card_to_index
from .hand_eval import evaluate_seven


def compute_river_ehs(board: Sequence[str]) -> Dict[int, float]:
    """Return {global_combo_idx: E[HS]} for every combo compatible with ``board``.

    E[HS] = P(win) + 0.5 * P(tie) vs. a uniform random opponent combo drawn
    from the remaining 47 cards.
    """
    board_indices = {card_to_index(c) for c in board}
    board_tuple = tuple(board)

    # Precompute hand rank for every compatible combo.
    combo_ranks: Dict[int, Tuple] = {}
    for idx, (a, b) in enumerate(INDEX_TO_COMBO):
        if a in board_indices or b in board_indices:
            continue
        seven = list(board_tuple) + [INDEX_TO_CARD[a], INDEX_TO_CARD[b]]
        combo_ranks[idx] = evaluate_seven(seven)

    ehs: Dict[int, float] = {}
    for hero_idx, hero_rank in combo_ranks.items():
        ha, hb = INDEX_TO_COMBO[hero_idx]
        wins = 0
        ties = 0
        total = 0
        for villain_idx, villain_rank in combo_ranks.items():
            va, vb = INDEX_TO_COMBO[villain_idx]
            if va == ha or va == hb or vb == ha or vb == hb:
                continue
            total += 1
            if hero_rank > villain_rank:
                wins += 1
            elif hero_rank == villain_rank:
                ties += 1
        ehs[hero_idx] = (wins + 0.5 * ties) / total if total > 0 else 0.0
    return ehs


def bucket_by_ehs(
    combo_indices: Sequence[int],
    combo_weights: Sequence[float],
    ehs: Dict[int, float],
    n_buckets: int,
) -> List[int]:
    """Assign each combo in ``combo_indices`` to a bucket (0..n_buckets-1).

    Buckets are equal-weight quantiles on E[HS]. Returns a list of bucket ids
    parallel to ``combo_indices``.

    If ``n_buckets`` >= number of distinct E[HS] values, each combo still gets
    a bucket in [0, n_buckets) but some buckets may be empty — this is fine
    for the downstream solver, which only indexes buckets that actually appear.
    """
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")
    n = len(combo_indices)
    if n == 0:
        return []
    if len(combo_weights) != n:
        raise ValueError("combo_indices and combo_weights length mismatch")

    # Sort combos by E[HS] (ascending = weakest-first)
    order = sorted(range(n), key=lambda k: ehs.get(combo_indices[k], 0.0))
    total_w = sum(combo_weights)
    if total_w <= 0:
        # degenerate — everyone in bucket 0
        return [0] * n

    target = total_w / n_buckets
    bucket_of = [0] * n
    cum = 0.0
    current_bucket = 0
    for rank, k in enumerate(order):
        bucket_of[k] = current_bucket
        cum += combo_weights[k]
        # Advance bucket if we've accumulated at least this bucket's share
        # AND there are more combos left to fill later buckets.
        if (
            cum >= target * (current_bucket + 1)
            and current_bucket + 1 < n_buckets
            and rank + 1 < n
        ):
            current_bucket += 1
    return bucket_of
