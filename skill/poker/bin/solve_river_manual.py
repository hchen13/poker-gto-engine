#!/usr/bin/env python3
"""River subgame solver with MANUALLY specified ranges (no precompute lookup).

Use this when precompute doesn't cover your spot — e.g., 100BB, multiway
reduced to HU, limp-raise pots, exotic action paths. The LLM (or you) provides
OOP and IP ranges as text; the solver returns GTO strategy against those ranges.

Usage:
    python solve_river_manual.py \\
      --board "<5 cards>" \\
      --oop-range "<range string>" \\
      --ip-range "<range string>" \\
      --pot <BB> --oop-stack <BB> --ip-stack <BB> \\
      [--hand <hero_hand> --position <oop|ip>] \\
      [--vs-action <action>] \\
      [--first-to-act 0|1] \\
      [--iterations N] [--buckets K]

Example:
    # 100BB 3bet pot, AK7r-Ts-2h, hero is IP BTN with AQo facing EP OOP cbet
    python solve_river_manual.py \\
      --board "Ac Kh 7s Td 2h" \\
      --oop-range "JJ+, AKs, AKo, AQs, A5s-A2s, KQs" \\
      --ip-range "22-TT, AQs, AJs, KQs, QJs, JTs, T9s, 98s" \\
      --pot 72 --oop-stack 46 --ip-stack 46 \\
      --hand AQo --position ip

Output: JSON with context/table/strategy same shape as solve_river.py.

Exit 0 = success, 1 = error (JSON to stderr).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from python.nlhe.range_parser import parse_range, RangeParseError  # noqa: E402
from python.nlhe.cards import (  # noqa: E402
    CARD_TO_INDEX, INDEX_TO_COMBO, NUM_COMBOS,
)

SOLVER_BINARY = PROJECT_ROOT / "target" / "release" / "solve_river_subgame"


def _parse_board(raw: str) -> list[int]:
    tokens = raw.replace(",", " ").split()
    if len(tokens) != 5:
        raise ValueError(f"Need 5 cards in board, got {len(tokens)}: {tokens}")
    try:
        return [CARD_TO_INDEX[t] for t in tokens]
    except KeyError as e:
        raise ValueError(f"Invalid card {e}. Use format like 'Ac Kh 7s Td 2h'.")


def _zero_conflicts(weights: list[float], board_mask: int) -> list[float]:
    w = list(weights)
    for idx, (a, b) in enumerate(INDEX_TO_COMBO):
        if board_mask & ((1 << a) | (1 << b)):
            w[idx] = 0.0
    return w


def _hand_type(combo: str) -> str:
    RANK_ORDER = "23456789TJQKA"
    combo = combo.strip()
    if len(combo) == 2:  # pair like "JJ"
        return combo.upper()
    if len(combo) == 3:  # hand type like "AQo", "KQs"
        return combo[0].upper() + combo[1].upper() + combo[2].lower()
    if len(combo) == 4:  # specific combo like "AhKd"
        r1, s1, r2, s2 = combo[0].upper(), combo[1].lower(), combo[2].upper(), combo[3].lower()
        ri1, ri2 = RANK_ORDER.index(r1), RANK_ORDER.index(r2)
        if ri1 < ri2:
            r1, s1, r2, s2 = r2, s2, r1, s1
            ri1, ri2 = ri2, ri1
        if ri1 == ri2:
            return r1 + r2
        return r1 + r2 + ("s" if s1 == s2 else "o")
    return combo


def _humanize_action(action: str, pot: float) -> str:
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


def _format_table(action_labels, strategy_row, hands_in_bucket, pot, ehs_range):
    rows = []
    for label, p in zip(action_labels, strategy_row):
        if p < 0.005:
            continue
        rows.append((_humanize_action(label, pot), p))
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
    for human, p in rows:
        lines.append(f"| {human} | {p*100:.0f}% | |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="River subgame solver with manually specified ranges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--board", required=True, help="5 cards space/comma-separated, e.g. 'Ac Kh 7s Td 2h'")
    parser.add_argument("--oop-range", required=True, help="OOP range string, e.g. 'JJ+, AKs, AKo, AQs'")
    parser.add_argument("--ip-range", required=True, help="IP range string")
    parser.add_argument("--pot", type=float, required=True, help="Pot size in BB at river")
    parser.add_argument("--oop-stack", type=float, required=True, help="OOP effective stack in BB")
    parser.add_argument("--ip-stack", type=float, required=True, help="IP effective stack in BB")
    parser.add_argument("--first-to-act", type=int, default=0, choices=[0, 1], help="0=OOP, 1=IP (default 0)")
    parser.add_argument("--hand", default=None, help="Hero hand (e.g. AQo) to display their bucket's strategy")
    parser.add_argument("--position", default=None, help="oop or ip (required if --hand given)")
    parser.add_argument("--vs-action", default=None, help="For IP hero: which OOP action to respond to")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--buckets", type=int, default=16)
    args = parser.parse_args()

    if not SOLVER_BINARY.exists():
        print(json.dumps({
            "error": f"solver binary not found at {SOLVER_BINARY}",
            "hint": "Run: cd ~/projects/poker-gto-engine && cargo build --release --bin solve_river_subgame"
        }), file=sys.stderr)
        sys.exit(1)
    if args.hand and not args.position:
        print(json.dumps({"error": "--position required when --hand is given"}), file=sys.stderr)
        sys.exit(1)

    try:
        board = _parse_board(args.board)
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

    board_mask = 0
    for idx in board:
        board_mask |= (1 << idx)

    try:
        oop_w = parse_range(args.oop_range)
        ip_w = parse_range(args.ip_range)
    except RangeParseError as e:
        print(json.dumps({"error": f"Range parse failed: {e}"}), file=sys.stderr)
        sys.exit(1)

    oop_w = _zero_conflicts(oop_w, board_mask)
    ip_w = _zero_conflicts(ip_w, board_mask)

    if sum(oop_w) == 0 or sum(ip_w) == 0:
        print(json.dumps({
            "error": "After zeroing board conflicts, one side has empty range. Check your ranges cover non-conflicting combos."
        }), file=sys.stderr)
        sys.exit(1)

    solver_input = {
        "board": board,
        "oop_weights": oop_w,
        "ip_weights": ip_w,
        "pot": args.pot,
        "oop_stack": max(0.0, args.oop_stack),
        "ip_stack": max(0.0, args.ip_stack),
        "first_to_act": args.first_to_act,
        "iterations": args.iterations,
        "k_buckets": args.buckets,
    }

    proc = subprocess.run(
        [str(SOLVER_BINARY)],
        input=json.dumps(solver_input), capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        print(json.dumps({"error": f"solver failed: {proc.stderr.strip()}"}), file=sys.stderr)
        sys.exit(1)
    result = json.loads(proc.stdout)

    # Locate hero node if --hand + --position given
    is_ip = args.position and args.position.lower() in ("ip", "sb", "1")
    nodes = result.get("nodes", [])
    hero_player = 1 if is_ip else 0
    hero_nodes = [n for n in nodes if n["player"] == hero_player]
    vs_action = args.vs_action

    primary_node = None
    if not is_ip:
        primary_node = next((n for n in hero_nodes if n["path"] == ""), None)
    else:
        if vs_action:
            primary_node = next((n for n in hero_nodes if n["path"] == vs_action), None)
        else:
            oop_root = next((n for n in nodes if n["player"] == 0 and n["path"] == ""), None)
            if oop_root:
                totals = [sum(b[i] for b in oop_root["strategy"]) for i in range(len(oop_root["action_labels"]))]
                for idx in sorted(range(len(totals)), key=lambda i: -totals[i]):
                    cand = oop_root["action_labels"][idx]
                    match = next((n for n in hero_nodes if n["path"] == cand), None)
                    if match:
                        primary_node = match
                        vs_action = cand
                        break

    hero_bucket_idx = -1
    hero_bucket_hands = []
    ehs_range = None
    if args.hand and args.position:
        ht = _hand_type(args.hand)
        bucket_hands_list = result.get("villain_bucket_hands" if is_ip else "bucket_hands", [])
        ehs_key = "villain_ehs_range" if is_ip else "hero_ehs_range"
        ehs_all = result.get(ehs_key, [])
        for bi, hands in enumerate(bucket_hands_list):
            if isinstance(hands, list) and ht in hands:
                hero_bucket_idx = bi
                hero_bucket_hands = hands
                if bi < len(ehs_all):
                    ehs_range = ehs_all[bi]
                break

    # Format table
    if primary_node and hero_bucket_idx >= 0:
        table_md = _format_table(
            primary_node["action_labels"],
            primary_node["strategy"][hero_bucket_idx],
            hero_bucket_hands, args.pot, ehs_range
        )
    elif primary_node:
        display_idx = 0
        table_md = _format_table(
            primary_node["action_labels"],
            primary_node["strategy"][display_idx],
            result.get("villain_bucket_hands" if is_ip else "bucket_hands", [[]])[display_idx],
            args.pot, None
        )
    else:
        table_md = "(no primary node found; see nodes field for full strategy tree)"

    # IP response breakdown
    ip_responses = {}
    if is_ip and hero_bucket_idx >= 0:
        for n in hero_nodes:
            if "/" in n["path"]:
                continue
            if hero_bucket_idx < len(n["strategy"]):
                ip_responses[n["path"]] = {
                    "action_labels": n["action_labels"],
                    "strategy": n["strategy"][hero_bucket_idx],
                }

    board_str = " ".join(sorted([f"{'23456789TJQKA'[c//4]}{'shdc'[c%4]}" for c in board[:3]]) +
                        [f"{'23456789TJQKA'[c//4]}{'shdc'[c%4]}" for c in board[3:]])
    pos_label = ""
    if args.hand and args.position:
        pos_name = "IP" if is_ip else "OOP"
        pos_label = f"  |  **Hero:** {_hand_type(args.hand)} ({pos_name})"
        if hero_bucket_idx < 0:
            pos_label += " *(hand not in specified range)*"
        if is_ip and vs_action:
            pos_label += f"  |  **Facing:** OOP {vs_action}"

    context = (
        f"**Board:** {board_str}  |  **Mode:** manual ranges (no precompute)  |  "
        f"**Pot:** {args.pot:.1f} BB  |  **Effective stack:** {min(args.oop_stack, args.ip_stack):.1f} BB"
        f"{pos_label}"
    )

    output = {
        "context": context,
        "table": table_md,
        "action_labels": primary_node["action_labels"] if primary_node else result["action_labels"],
        "hero_strategy": primary_node["strategy"] if primary_node else result["hero_strategy"],
        "bucket_hands": result.get("bucket_hands", []),
        "villain_bucket_hands": result.get("villain_bucket_hands", []),
        "hero_ehs_range": result.get("hero_ehs_range", []),
        "villain_ehs_range": result.get("villain_ehs_range", []),
        "hero_value": result["hero_value"],
        "hero_bucket_idx": hero_bucket_idx,
        "vs_action": vs_action if is_ip else None,
        "ip_responses": ip_responses,
        "pot": args.pot,
        "iterations": result["iterations"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
