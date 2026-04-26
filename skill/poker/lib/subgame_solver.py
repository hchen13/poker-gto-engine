"""Shared helpers for on-demand subgame solvers (flop / turn / river).

Each solver CLI spawns a Rust binary via stdin/stdout JSON. Common logic:
- parse range strings → 1326-weight vectors
- parse board cards, zero board-conflicting combos
- format hero's bucket strategy as markdown table
- locate hero node by (player, path) for IP response lookup
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from python.nlhe.range_parser import parse_range, RangeParseError  # noqa: E402
from python.nlhe.cards import (  # noqa: E402
    CARD_TO_INDEX, INDEX_TO_COMBO, NUM_COMBOS,
)

RANK_ORDER = "23456789TJQKA"


def find_binary(name: str) -> Path:
    for cand in [
        PROJECT_ROOT / "target" / "release" / name,
        PROJECT_ROOT / "target" / "debug" / name,
    ]:
        if cand.exists() and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError(
        f"{name} binary not found. Build: cd {PROJECT_ROOT} && cargo build --release --bin {name}"
    )


def parse_board(raw: str, expected_count: int) -> List[int]:
    tokens = raw.replace(",", " ").split()
    if len(tokens) != expected_count:
        raise ValueError(f"expected {expected_count} cards, got {len(tokens)}: {tokens}")
    try:
        return [CARD_TO_INDEX[t] for t in tokens]
    except KeyError as e:
        raise ValueError(f"Invalid card {e}. Use like 'Ac Kh 7s'.")


def board_mask(indices: List[int]) -> int:
    m = 0
    for i in indices:
        m |= (1 << i)
    return m


def zero_board_conflicts(weights: List[float], mask: int) -> List[float]:
    w = list(weights)
    for idx, (a, b) in enumerate(INDEX_TO_COMBO):
        if mask & ((1 << a) | (1 << b)):
            w[idx] = 0.0
    return w


def hand_type(combo: str) -> str:
    combo = combo.strip()
    if len(combo) == 2:
        return combo.upper()
    if len(combo) == 3:
        return combo[0].upper() + combo[1].upper() + combo[2].lower()
    if len(combo) == 4:
        r1, s1, r2, s2 = combo[0].upper(), combo[1].lower(), combo[2].upper(), combo[3].lower()
        ri1, ri2 = RANK_ORDER.index(r1), RANK_ORDER.index(r2)
        if ri1 < ri2:
            r1, s1, r2, s2 = r2, s2, r1, s1
            ri1, ri2 = ri2, ri1
        if ri1 == ri2:
            return r1 + r2
        return r1 + r2 + ("s" if s1 == s2 else "o")
    return combo


def humanize_action(action: str, pot: float) -> str:
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


def format_table(
    action_labels: List[str],
    strategy_row: List[float],
    hands_in_bucket: List[str],
    pot: float,
    ehs_range: Optional[List[float]] = None,
) -> str:
    rows = [(humanize_action(l, pot), p) for l, p in zip(action_labels, strategy_row) if p >= 0.005]
    rows.sort(key=lambda r: -r[1])
    hands_str = ", ".join(hands_in_bucket[:6]) if hands_in_bucket else "bucket"
    if len(hands_in_bucket) > 6:
        hands_str += ", ..."
    ehs_str = ""
    if ehs_range and len(ehs_range) == 2:
        ehs_str = f"  |  EHS {ehs_range[0]:.3f}–{ehs_range[1]:.3f}"
    lines = [
        f"**Pot:** {pot:.1f} BB  |  {hands_str}{ehs_str}",
        "",
        "| 操作 | 频率 | 理由 |",
        "|------|------|------|",
    ]
    for h, p in rows:
        lines.append(f"| {h} | {p*100:.0f}% | |")
    return "\n".join(lines)


def call_solver(binary: Path, input_spec: Dict[str, Any], timeout_s: float = 180) -> Dict[str, Any]:
    proc = subprocess.run(
        [str(binary)],
        input=json.dumps(input_spec), capture_output=True, text=True, timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{binary.name} failed: {proc.stderr.strip() or proc.stdout[:200]}")
    return json.loads(proc.stdout)


def find_hero_bucket(
    result: Dict[str, Any],
    hero_hand: str,
    is_ip: bool,
) -> Tuple[int, List[str], Optional[List[float]]]:
    """Return (bucket_idx, bucket_hands, ehs_range_for_bucket). bucket_idx=-1 if not found."""
    ht = hand_type(hero_hand)
    bucket_hands_list = result.get("villain_bucket_hands" if is_ip else "bucket_hands", [])
    ehs_key = "villain_ehs_range" if is_ip else "hero_ehs_range"
    ehs_all = result.get(ehs_key, [])
    for bi, hands in enumerate(bucket_hands_list):
        if isinstance(hands, list) and ht in hands:
            return bi, hands, (ehs_all[bi] if bi < len(ehs_all) else None)
    return -1, [], None


def find_primary_node(
    result: Dict[str, Any],
    hero_player: int,
    *,
    street: Optional[str] = None,
    turn_card: Optional[str] = None,
    path: Optional[str] = None,
    vs_action: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Locate the node at which hero acts.
    Returns (node, resolved_vs_action). resolved_vs_action is non-None for IP hero
    when path wasn't fixed but we defaulted to OOP's most-frequent action.
    """
    nodes = result.get("nodes", [])
    hero_nodes = [n for n in nodes if n["player"] == hero_player]
    if street is not None:
        hero_nodes = [n for n in hero_nodes if n.get("street") == street]
    if turn_card is not None:
        tc = turn_card.lower()
        hero_nodes = [n for n in hero_nodes if (n.get("turn_card") or "").lower() == tc]

    # If caller gave explicit path, use that
    if path is not None:
        match = next((n for n in hero_nodes if n["path"] == path), None)
        return match, vs_action

    # For IP hero without explicit path, try vs_action first
    if hero_player == 1 and vs_action:
        match = next((n for n in hero_nodes if n["path"].endswith(vs_action)), None)
        if match:
            return match, vs_action

    # OOP: pick root (empty path) or shortest path
    if hero_player == 0:
        match = next((n for n in hero_nodes if n["path"] == ""), None)
        if match:
            return match, None
        if hero_nodes:
            return min(hero_nodes, key=lambda n: len(n["path"])), None
        return None, None

    # IP default: find OOP's most frequent action at root, look up IP's response
    oop_root = next((n for n in nodes if n["player"] == 0 and n["path"] == ""), None)
    if oop_root:
        totals = [sum(b[i] for b in oop_root["strategy"]) for i in range(len(oop_root["action_labels"]))]
        for idx in sorted(range(len(totals)), key=lambda i: -totals[i]):
            cand = oop_root["action_labels"][idx]
            match = next((n for n in hero_nodes if n["path"] == cand), None)
            if match:
                return match, cand
    return (hero_nodes[0] if hero_nodes else None), None


def parse_ranges_or_exit(oop_range: str, ip_range: str) -> Tuple[List[float], List[float]]:
    try:
        oop_w = parse_range(oop_range)
        ip_w = parse_range(ip_range)
    except RangeParseError as e:
        print(json.dumps({"error": f"Range parse failed: {e}"}), file=sys.stderr)
        sys.exit(1)
    return oop_w, ip_w


def board_to_str(indices: List[int]) -> str:
    RANKS = "23456789TJQKA"; SUITS = "shdc"
    return " ".join(f"{RANKS[c//4]}{SUITS[c%4]}" for c in indices)
