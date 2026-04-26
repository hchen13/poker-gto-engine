"""Query interface for precomputed flop/turn GTO strategies.

Usage:
    result = query(
        action_line="3bet_called",   # limped | sr_called | 3bet_called | 4bet_called
        stack_bb=200,                 # 200 or 500
        flop="K62r",                  # rank-texture notation, OR exact "KdQh2c"
        hero_hand="KQo",              # hand type OR specific combo "KhQs"
        position="ip",                # "ip" (SB) or "oop" (BB)
        action_path="",               # "" = root, "check" = after BB checks, etc.
        turn_card=None,               # e.g. "Ts" — for turn node lookup
    )
    print(result["table"])            # markdown table
    print(result["node_info"])        # street/path/player context
"""

from __future__ import annotations

import gzip
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRECOMPUTE_ROOT = Path(
    os.environ.get("POKER_PRECOMPUTE_DIR") or (_PROJECT_ROOT / "precompute_out")
).resolve()


def _load_json(f: Path) -> Dict:
    """Load a precompute file of any supported format.
    Named `_load_json` for back-compat with existing call sites; now also
    handles .mpk.zst (msgpack + u8-quantized + zstd, ~5x smaller than .json.gz).
    """
    name = f.name
    if name.endswith(".mpk.zst"):
        # Lazy import: only pay the dep cost when this format is used
        from .compressed_format import decode
        return decode(f.read_bytes())
    if name.endswith(".gz"):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(f.read_text())

RANK_ORDER = "23456789TJQKA"
SUIT_CHARS = "shdc"

ACTION_LINE_DIRS = {
    "limped":                  "full_limped",
    "sr_called":               "full_sr_called",
    "sr_called_ip_caller":     "full_sr_called_ip_caller",
    "3bet_called":             "full_3bet_called",
    "3bet_called_ip3bet":      "full_3bet_called_ip3bet",
    "4bet_called":             "full_4bet_called",
    "4bet_called_ip_caller":   "full_4bet_called_ip_caller",
}

# Pot sizes by action line (from fixtures)
ACTION_LINE_POTS = {
    "limped": 2.0,
    "sr_called": 12.0,
    "sr_called_ip_caller": 12.0,
    "3bet_called": 36.0,
    "3bet_called_ip3bet": 36.0,
    "4bet_called": 108.0,
    "4bet_called_ip_caller": 108.0,
}


def _rank(card: str) -> str:
    """Return rank char from a 2-char card like 'Kd' → 'K'."""
    return card[0].upper()


def _rank_idx(card: str) -> int:
    return RANK_ORDER.index(_rank(card))


def _parse_flop_texture(flop_str: str) -> Tuple[List[str], Optional[str]]:
    """Parse flop string. Returns (rank_list, texture) where texture is
    'rainbow'/'monotone'/'flush_draw' or None if exact cards given.

    Accepted formats:
    - "K62r"    → ranks K,6,2 + rainbow
    - "K62"     → ranks K,6,2, any texture
    - "K62ss"   → ranks K,6,2, two-suited (flush draw)
    - "K62s"    → same as ss
    - "K62m"    → monotone
    - "KdQh2c"  → exact cards
    """
    # Exact card notation: 6 chars with suit letters
    if len(flop_str) == 6 and all(flop_str[i + 1] in SUIT_CHARS for i in range(0, 6, 2)):
        cards = [flop_str[0:2], flop_str[2:4], flop_str[4:6]]
        return cards, None

    # Rank+texture notation
    ranks = []
    texture = None
    i = 0
    while i < len(flop_str):
        c = flop_str[i]
        if c.upper() in RANK_ORDER:
            ranks.append(c.upper())
        elif c.lower() in ('r', 'm'):
            texture = 'rainbow' if c.lower() == 'r' else 'monotone'
        elif c.lower() == 's':
            texture = 'flush_draw'
        i += 1
    return ranks, texture


def _suit_count(cards: List[str]) -> int:
    """Number of distinct suits in 2-char card list."""
    return len(set(c[1] for c in cards))


def _texture_match(cards: List[str], texture: Optional[str]) -> bool:
    if texture is None:
        return True
    sc = _suit_count(cards)
    if texture == 'rainbow':
        return sc == 3
    if texture == 'monotone':
        return sc == 1
    if texture == 'flush_draw':
        return sc == 2
    return True


# ---------- canonical iso-class lookup ----------
# Mirrors src/nlhe/abstraction.rs :: flop_iso_key


_SUIT_PERMS: List[Tuple[int, int, int, int]] = [
    (0,1,2,3),(0,1,3,2),(0,2,1,3),(0,2,3,1),(0,3,1,2),(0,3,2,1),
    (1,0,2,3),(1,0,3,2),(1,2,0,3),(1,2,3,0),(1,3,0,2),(1,3,2,0),
    (2,0,1,3),(2,0,3,1),(2,1,0,3),(2,1,3,0),(2,3,0,1),(2,3,1,0),
    (3,0,1,2),(3,0,2,1),(3,1,0,2),(3,1,2,0),(3,2,0,1),(3,2,1,0),
]


def _card_index(card: str) -> int:
    """Card string like 'Ks' → int 0..51 (rank*4 + suit, suits=s,h,d,c)."""
    r = RANK_ORDER.index(card[0].upper())
    s = SUIT_CHARS.index(card[1].lower())
    return r * 4 + s


def flop_iso_key(cards: List[str]) -> Tuple[int, ...]:
    """Canonical iso-class key for a 3-card flop (suit-symmetric).
    Minimum over all 24 suit permutations of (sorted_cards) → 6-byte tuple.
    """
    card_ints = [_card_index(c) for c in cards]
    best: Optional[Tuple[int, ...]] = None
    for perm in _SUIT_PERMS:
        relabeled = [(c // 4) * 4 + perm[c % 4] for c in card_ints]
        relabeled.sort()
        key = tuple(v for c in relabeled for v in (c // 4, c % 4))
        if best is None or key < best:
            best = key
    return best  # type: ignore


# Cached iso-key indexes: {job_dir_path_str: {iso_key: flop_file_path}}
_ISO_INDEX_CACHE: Dict[str, Dict[Tuple[int, ...], Path]] = {}


def _enumerate_precompute_files(job_dir: Path) -> List[Path]:
    """List all flop files in a job_dir regardless of format."""
    out: List[Path] = []
    out.extend(job_dir.glob("flop_*.mpk.zst"))   # new compressed format
    out.extend(job_dir.glob("flop_*.json.gz"))   # gzipped JSON
    out.extend(job_dir.glob("flop_*.json"))      # uncompressed JSON
    return out


def _extract_flop_label(path: Path) -> str:
    """From 'flop_3c7hTh.mpk.zst' or 'flop_3c7hTh.json.gz' → '3c7hTh'."""
    name = path.name
    for suffix in (".mpk.zst", ".json.gz", ".json"):
        if name.endswith(suffix):
            return name[len("flop_"):-len(suffix)]
    return name.replace("flop_", "")


def _build_iso_index(job_dir: Path) -> Dict[Tuple[int, ...], Path]:
    key_str = str(job_dir)
    if key_str in _ISO_INDEX_CACHE:
        return _ISO_INDEX_CACHE[key_str]
    index: Dict[Tuple[int, ...], Path] = {}
    for f in _enumerate_precompute_files(job_dir):
        fname = _extract_flop_label(f)
        if len(fname) != 6:
            continue
        cards = [fname[0:2], fname[2:4], fname[4:6]]
        try:
            k = flop_iso_key(cards)
        except (ValueError, IndexError):
            continue
        index.setdefault(k, f)
    _ISO_INDEX_CACHE[key_str] = index
    return index


def find_flop_file(
    action_line: str,
    stack_bb: int,
    flop: str,
) -> Optional[Path]:
    """Return path to the best-matching precompute JSON for this spot."""
    dir_prefix = ACTION_LINE_DIRS.get(action_line)
    if dir_prefix is None:
        raise ValueError(f"Unknown action_line: {action_line!r}. Use: {list(ACTION_LINE_DIRS)}")
    stack_key = f"{stack_bb}bb"
    job_dir = PRECOMPUTE_ROOT / f"{dir_prefix}_{stack_key}"
    if not job_dir.exists():
        raise FileNotFoundError(f"Precompute dir not found: {job_dir}")

    rank_list, texture = _parse_flop_texture(flop)

    # Exact 3-card specification: look up directly
    if texture is None and len(rank_list) == 0:
        # was already parsed as exact cards in rank_list
        pass

    def _find_in_dir(job_dir: Path, stem: str) -> Optional[Path]:
        for ext in (".mpk.zst", ".json.gz", ".json"):
            p = job_dir / (stem + ext)
            if p.exists():
                return p
        return None

    if texture is None and all(len(r) == 2 for r in rank_list):
        # Exact cards. Primary lookup: iso-class canonical key.
        try:
            k = flop_iso_key(rank_list)
            idx = _build_iso_index(job_dir)
            if k in idx:
                return idx[k]
        except (ValueError, IndexError):
            pass
        # Secondary: try direct filename match (cheap fallback for legacy dirs)
        cards_sorted = sorted(rank_list, key=lambda c: _rank_idx(c))
        stem = "flop_" + "".join(cards_sorted)
        exact = _find_in_dir(job_dir, stem)
        if exact is not None:
            return exact
        # Tertiary: fall through to rank+texture search (handles partially-sampled legacy precomputes)
        texture = {1: 'monotone', 2: 'flush_draw', 3: 'rainbow'}.get(_suit_count(rank_list))
        rank_list = [_rank(c) for c in rank_list]

    # Rank-based search (for rank+texture-only queries like "T73r")
    target_ranks = sorted(rank_list, key=lambda r: RANK_ORDER.index(r))
    candidates = []
    for f in _enumerate_precompute_files(job_dir):
        fname = _extract_flop_label(f)
        if len(fname) != 6:
            continue
        file_cards = [fname[0:2], fname[2:4], fname[4:6]]
        file_ranks = sorted([_rank(c) for c in file_cards], key=lambda r: RANK_ORDER.index(r))
        if file_ranks == target_ranks and _texture_match(file_cards, texture):
            candidates.append(f)

    if not candidates:
        return None
    # Prefer rainbow for readability, otherwise first match
    for c in candidates:
        fname = c.stem.replace("flop_", "")
        cards = [fname[0:2], fname[2:4], fname[4:6]]
        if _suit_count(cards) == 3:
            return c
    return candidates[0]


def _hand_type(combo: str) -> str:
    """Convert specific combo 'KhQs' → 'KQo', 'KhQh' → 'KQs', 'KhKs' → 'KK'."""
    combo = combo.strip()
    if len(combo) == 4:
        r1, s1, r2, s2 = combo[0].upper(), combo[1].lower(), combo[2].upper(), combo[3].lower()
        ri1, ri2 = RANK_ORDER.index(r1), RANK_ORDER.index(r2)
        if ri1 < ri2:
            r1, s1, r2, s2 = r2, s2, r1, s1
        if ri1 == ri2:
            return r1 + r2
        return r1 + r2 + ('s' if s1 == s2 else 'o')
    # Already hand type like "KQo", "AKs", "TT"
    return combo.strip()


def find_bucket_for_hand(
    data: Dict,
    hero_hand: str,
    position: str = "oop",
) -> Tuple[int, List[str]]:
    """Return (bucket_idx, hands_in_bucket) for the given hand.

    position: "oop" (BB/hero) uses bucket_hands;
              "ip" (SB/villain) uses villain_bucket_hands.
    hero_hand can be a specific combo ('KhQs') or hand type ('KQo').

    If hand not in the precomputed range, falls back to EHS-based
    approximation: compute the hand's EHS on the precompute's rep5 board,
    match it to the nearest bucket by EHS range. The returned bucket will
    have a list beginning with "~{ht}" to indicate the approximation.
    """
    ht = _hand_type(hero_hand)
    is_ip = position.lower() in ("ip", "sb", "1")

    if is_ip:
        bucket_hands: List[List[str]] = data.get("villain_bucket_hands", [])
        bucket_ehs: List[List[float]] = data.get("villain_bucket_ehs_range", [])
        label = "villain_bucket_hands"
    else:
        bucket_hands = data.get("bucket_hands", [])
        bucket_ehs = data.get("bucket_ehs_range", [])
        label = "bucket_hands"

    if not bucket_hands:
        raise ValueError(f"{label} missing from file. Run patch_villain_buckets first.")

    for b, hands in enumerate(bucket_hands):
        if ht in hands:
            return b, hands

    # Fallback: compute EHS on rep5, find nearest bucket.
    approx = _approximate_bucket_by_ehs(data, ht, bucket_ehs)
    if approx is None:
        raise ValueError(
            f"Hand type {ht!r} fully conflicts with the flop or EHS fallback failed. "
            f"Bucket 0 sample: {bucket_hands[0][:5] if bucket_hands else 'empty'}"
        )
    bucket_idx, hand_ehs = approx
    hands = [f"~{ht} (EHS≈{hand_ehs:.3f}, approximated — not in {label})"] + list(bucket_hands[bucket_idx])
    return bucket_idx, hands


def _approximate_bucket_by_ehs(
    data: Dict,
    ht: str,
    bucket_ehs: List[List[float]],
) -> Optional[Tuple[int, float]]:
    """Compute EHS for ``ht`` on the precompute's rep5 board and find the
    nearest bucket. Returns (bucket_idx, avg_ehs) or None on failure.
    """
    try:
        import sys as _sys
        _project_root = str(Path(__file__).resolve().parents[3])
        if _project_root not in _sys.path:
            _sys.path.insert(0, _project_root)
        from python.nlhe.abstraction import compute_river_ehs  # type: ignore
        from python.nlhe.cards import INDEX_TO_CARD, INDEX_TO_COMBO, CARD_TO_INDEX  # type: ignore
    except Exception:
        return None

    flop_label = data.get("flop_label", "")
    if len(flop_label) != 6:
        return None
    flop_cards_str = [flop_label[0:2], flop_label[2:4], flop_label[4:6]]
    try:
        flop_indices = [CARD_TO_INDEX[c] for c in flop_cards_str]
    except KeyError:
        return None

    rep5_indices = _rep5_for_flop(flop_indices)
    rep5_str = [INDEX_TO_CARD[i] for i in rep5_indices]

    # Compute EHS for all non-conflicting combos on rep5
    try:
        ehs_map = compute_river_ehs(rep5_str)
    except Exception:
        return None

    # Extract combos matching hand type ht
    board_set = set(rep5_indices)
    target_ehs: List[float] = []
    for combo_idx, (a, b) in enumerate(INDEX_TO_COMBO):
        if a in board_set or b in board_set:
            continue
        combo_str = INDEX_TO_CARD[a] + INDEX_TO_CARD[b]
        if _hand_type(combo_str) == ht and combo_idx in ehs_map:
            target_ehs.append(ehs_map[combo_idx])

    if not target_ehs:
        return None

    avg_ehs = sum(target_ehs) / len(target_ehs)

    # Find bucket whose [lo, hi] contains avg_ehs; else closest bucket
    best_i = 0
    best_dist = float("inf")
    for i, rng in enumerate(bucket_ehs):
        if len(rng) < 2:
            continue
        lo, hi = rng[0], rng[1]
        if lo <= avg_ehs <= hi:
            return i, avg_ehs
        d = min(abs(avg_ehs - lo), abs(avg_ehs - hi))
        if d < best_dist:
            best_dist, best_i = d, i
    return best_i, avg_ehs


def _rep5_for_flop(flop_indices: List[int]) -> List[int]:
    """Reconstruct the rep5 5-card board used by precompute bucketing.
    Matches Rust ``rep5`` construction (rank_subsample + middle + middle+1).
    """
    board = set(flop_indices)
    remaining = [c for c in range(52) if c not in board]
    # rank_subsample: one representative per rank, ascending rank, min card of rank
    by_rank: Dict[int, List[int]] = {}
    for c in remaining:
        r = c // 4
        by_rank.setdefault(r, []).append(c)
    subset = [min(by_rank[r]) for r in sorted(by_rank)]
    rep5 = list(flop_indices)
    rep5.append(subset[len(subset) // 2])
    ri = (len(subset) // 2 + 1) % len(subset)
    if subset[ri] == rep5[3]:
        ri = 0
    rep5.append(subset[ri])
    return rep5


def find_node(
    data: Dict,
    position: str,
    action_path: str,
    turn_card: Optional[str] = None,
) -> Optional[Dict]:
    """Find the node matching position + action_path (+ turn_card for turn).

    position: "ip" (SB=player 1) or "oop" (BB=player 0)
    action_path: "" for root, "check" after OOP checks, "check/bet_11.88" etc.
    turn_card: e.g. "Ts" — only relevant for turn nodes
    """
    player_id = 1 if position.lower() in ("ip", "sb", "1") else 0
    nodes: List[Dict] = data["nodes"]

    matches = []
    for node in nodes:
        if node["player"] != player_id:
            continue
        if node["path"] != action_path:
            continue
        if turn_card is not None:
            if node.get("street") != "turn":
                continue
            if node.get("turn_card", "").lower() != turn_card.lower():
                continue
        matches.append(node)

    if not matches:
        return None
    # Prefer flop > turn if no turn_card specified
    if turn_card is None:
        flop_matches = [n for n in matches if n["street"] == "flop"]
        if flop_matches:
            return flop_matches[0]
    return matches[0]


def list_action_paths(data: Dict, position: str, street: str = "flop") -> List[str]:
    """List available action paths for a given position/street (for debugging)."""
    player_id = 1 if position.lower() in ("ip", "sb", "1") else 0
    return [
        n["path"] for n in data["nodes"]
        if n["player"] == player_id and n["street"] == street
    ]


def _pot_after_path(base_pot: float, path: str) -> float:
    """Rough pot estimation by tracing action_path string."""
    pot = base_pot
    actions = [a for a in path.split("/") if a]
    i = 0
    while i < len(actions):
        a = actions[i]
        if a == "check" or a == "fold":
            pass
        elif a == "call":
            # find preceding bet amount
            if i > 0 and "_" in actions[i - 1]:
                try:
                    bet = float(actions[i - 1].split("_")[1])
                    pot += bet * 2
                except ValueError:
                    pass
        elif "_" in a:
            pass  # just a bet sizing label
        i += 1
    return pot


def format_strategy_table(
    node: Dict,
    bucket_idx: int,
    bucket_hands: List[str],
    base_pot: float,
    action_path: str,
) -> str:
    """Render markdown strategy table for a specific bucket."""
    strategy: List[List[float]] = node["strategy"]
    action_labels: List[str] = node["action_labels"]
    probs = strategy[bucket_idx]

    pot = _pot_after_path(base_pot, action_path)

    rows = []
    for a, p in zip(action_labels, probs):
        if p < 0.005:
            continue
        # Convert action label to readable format
        label = _humanize_action(a, pot)
        rows.append((label, p))

    rows.sort(key=lambda r: -r[1])

    lines = [
        f"**Pot:** {pot:.1f} BB  |  **Bucket {bucket_idx}** ({', '.join(bucket_hands[:6])}{', ...' if len(bucket_hands) > 6 else ''})",
        "",
        "| 操作 | 频率 | 理由 |",
        "|------|------|------|",
    ]
    for label, p in rows:
        lines.append(f"| {label} | {p*100:.0f}% | |")

    return "\n".join(lines)


def _humanize_action(action: str, pot: float) -> str:
    """Convert 'bet_11.88' to 'bet 33% pot (11.9 BB)'."""
    if action in ("check", "fold", "call"):
        return action
    if "_" in action:
        kind, amt_str = action.split("_", 1)
        try:
            amt = float(amt_str)
            if kind == "allin":
                return f"all-in ({amt:.0f} BB)"
            pct = int(round(amt / pot * 100)) if pot > 0 else 0
            return f"{kind} {pct}% pot ({amt:.1f} BB)"
        except ValueError:
            return action
    return action


def _hero_has_flush_draw(hero_hand: str, flop_cards: List[str]) -> bool:
    """Hero has FD if both hero cards share a suit AND board has 2+ that suit.
    Only meaningful when hero_hand is a specific combo (e.g. 'AsKs', not 'AKs')."""
    if len(hero_hand) != 4:
        return False
    s1, s2 = hero_hand[1], hero_hand[3]
    if s1 not in "shdc" or s2 not in "shdc" or s1 != s2:
        return False
    return sum(1 for c in flop_cards if c[1] == s1) >= 2


def bucket_dispersion_note(
    hero_hand: str,
    flop_label: str,
    bucket_hands: List[str],
) -> Optional[str]:
    """Detect bucket heterogeneity that materially distorts hero's strategy.

    Strategy in a precomputed table is per-bucket. When a bucket lumps together
    hands with very different draw / made-hand structure (e.g. AKs with NFD
    alongside A7o with no draw on a 2-suited flop), the displayed frequency is
    a bucket average that may not reflect hero's specific combo. Returns a
    short markdown warning (or None if the bucket is structurally tight).
    """
    if len(bucket_hands) < 4:
        return None

    flop_cards = [flop_label[i:i+2] for i in range(0, 6, 2)]
    suit_counts = {s: sum(1 for c in flop_cards if c[1] == s) for s in "shdc"}
    flushy_suit = max(suit_counts.values()) >= 2
    paired_board = len({c[0] for c in flop_cards}) < 3

    suited_types = sum(1 for h in bucket_hands if len(h) == 3 and h.endswith("s"))
    offsuit_types = sum(1 for h in bucket_hands if len(h) == 3 and h.endswith("o"))
    pair_types = sum(1 for h in bucket_hands if len(h) == 2 and h[0] == h[1])

    hero_has_fd = _hero_has_flush_draw(hero_hand, flop_cards)

    notes = []
    # Hero-specific: hero has FD but bucket includes hands that can't have FD here
    if hero_has_fd and offsuit_types >= 2:
        notes.append(
            f"Hero ({hero_hand}) has a flush draw on this board, but the bucket "
            f"includes {offsuit_types} offsuit hand-types that cannot have FD on "
            "this flop. The displayed frequency is averaged across the bucket."
        )
    # Generic: suited + offsuit + flushy → draw structure differs across bucket
    elif suited_types >= 2 and offsuit_types >= 2 and flushy_suit:
        notes.append(
            f"Bucket spans both suited ({suited_types}) and offsuit ({offsuit_types}) "
            "hand-types on a flushy board — flush-draw presence differs across the "
            "bucket and the strategy is a bucket average."
        )
    # Pocket pair vs high-card on unpaired board → set/overpair vs nothing
    elif pair_types >= 2 and (suited_types + offsuit_types) >= 4 and not paired_board:
        notes.append(
            f"Bucket mixes {pair_types} pocket-pair types with high-card hands on an "
            "unpaired board — set/overpair vs no-pair structure differs across the bucket."
        )

    if not notes:
        return None
    return (
        "⚠️ **Bucket dispersion warning** — " + " ".join(notes) +
        " For a hand whose structure differs from the bucket norm, "
        "`poker solve-flop-manual` (or `solve-turn-manual` / `solve-river-manual`) "
        "computes a hand-specific strategy."
    )


def query(
    action_line: str,
    stack_bb: int,
    flop: str,
    hero_hand: str,
    position: str,
    action_path: str = "",
    turn_card: Optional[str] = None,
) -> Dict[str, Any]:
    """Full query: find file → find bucket → find node → return table + context."""
    f = find_flop_file(action_line, stack_bb, flop)
    if f is None:
        raise FileNotFoundError(
            f"No precomputed file for {action_line}/{stack_bb}bb/{flop}. "
            "Check flop rank spelling or texture."
        )

    data = _load_json(f)
    flop_label = data["flop_label"]

    bucket_idx, bucket_hands = find_bucket_for_hand(data, hero_hand, position)
    ehs_key = "villain_bucket_ehs_range" if position.lower() in ("ip", "sb", "1") else "bucket_ehs_range"
    ehs_range = data.get(ehs_key, data.get("bucket_ehs_range", [[0, 1]] * 16))[bucket_idx]

    node = find_node(data, position, action_path, turn_card)
    if node is None:
        available = list_action_paths(data, position, "flop" if turn_card is None else "turn")
        raise ValueError(
            f"No node found for position={position!r}, path={action_path!r}, turn={turn_card!r}.\n"
            f"Available paths for {position}: {available[:20]}"
        )

    base_pot = ACTION_LINE_POTS.get(action_line, 12.0)
    table_md = format_strategy_table(node, bucket_idx, bucket_hands, base_pot, action_path)

    warning = bucket_dispersion_note(hero_hand, flop_label, bucket_hands)
    if warning:
        table_md = warning + "\n\n" + table_md

    ht = _hand_type(hero_hand)
    pos_label = "IP (SB)" if position.lower() in ("ip", "sb", "1") else "OOP (BB)"

    context = (
        f"**Flop:** {flop_label}  |  **Line:** {action_line}/{stack_bb}bb  |  "
        f"**Hero:** {ht} ({pos_label})  |  "
        f"**EHS range:** {ehs_range[0]:.3f}–{ehs_range[1]:.3f}  |  "
        f"**Node path:** {node['path'] or '(root)'}  |  **Street:** {node['street']}"
    )
    if turn_card:
        context += f"  |  **Turn:** {turn_card}"

    return {
        "table": table_md,
        "context": context,
        "flop_label": flop_label,
        "bucket_idx": bucket_idx,
        "bucket_hands": bucket_hands,
        "dispersion_warning": warning,
        "node": node,
        "ehs_range": ehs_range,
        "file": str(f),
    }


def available_paths_summary(
    action_line: str,
    stack_bb: int,
    flop: str,
) -> Dict[str, Any]:
    """Return all available action paths in a file (for user to browse)."""
    f = find_flop_file(action_line, stack_bb, flop)
    if f is None:
        return {"error": f"No file for {action_line}/{stack_bb}bb/{flop}"}
    data = _load_json(f)
    nodes = data["nodes"]
    flop_oop = sorted({n["path"] for n in nodes if n["street"] == "flop" and n["player"] == 0})
    flop_ip = sorted({n["path"] for n in nodes if n["street"] == "flop" and n["player"] == 1})
    turn_cards = sorted({n.get("turn_card", "") for n in nodes if n["street"] == "turn" and n.get("turn_card")})
    return {
        "flop_label": data["flop_label"],
        "flop_oop_paths": flop_oop,
        "flop_ip_paths": flop_ip,
        "turn_cards_available": turn_cards[:10],
        "total_nodes": len(nodes),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 5:
        action_line, stack_bb, flop, hero_hand = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
        position = sys.argv[5] if len(sys.argv) > 5 else "oop"
        action_path = sys.argv[6] if len(sys.argv) > 6 else ""
        turn_card = sys.argv[7] if len(sys.argv) > 7 else None
        try:
            r = query(action_line, stack_bb, flop, hero_hand, position, action_path, turn_card)
            print(r["context"])
            print()
            print(r["table"])
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Demo: KQo on K62r, 3bet pot, OOP
        try:
            r = query("3bet_called", 200, "K62r", "KQo", "oop", "")
            print(r["context"])
            print()
            print(r["table"])
            print()
            paths = available_paths_summary("3bet_called", 200, "K62r")
            print("OOP flop paths:", paths["flop_oop_paths"][:10])
            print("IP  flop paths:", paths["flop_ip_paths"][:10])
        except Exception as e:
            print(f"Demo error: {e}")
