#!/usr/bin/env python3
"""Flop subgame solver with manually specified ranges (no precompute lookup).

Solves the full flop → turn → river tree with bucketed CFR+. Typical wall
time 15-60s with K=16 and default subsets.

Usage:
    python solve_flop_manual.py \\
      --board "<3 cards>" \\
      --oop-range "<range>" --ip-range "<range>" \\
      --pot <BB> --oop-stack <BB> --ip-stack <BB> \\
      [--hand <hero_hand> --position <oop|ip>] \\
      [--path <action_path>] [--turn <card>] [--vs-action <action>] \\
      [--iterations 200] [--buckets 16] [--no-subset]

Example:
    python solve_flop_manual.py \\
      --board "Tc 7h 3d" \\
      --oop-range "22-JJ, A2s-ATs, ..." --ip-range "22+, A2s+, ..." \\
      --pot 12 --oop-stack 194 --ip-stack 194 \\
      --hand JTs --position oop --path ""

For a turn-level query after flop solve, pass --turn <card> and --path
<full_path_including_flop_actions>.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from skill.poker.lib.subgame_solver import (  # noqa: E402
    find_binary, parse_board, board_mask, zero_board_conflicts,
    parse_ranges_or_exit, call_solver, find_hero_bucket, find_primary_node,
    format_table, hand_type, board_to_str,
)


def main():
    p = argparse.ArgumentParser(
        description="Flop subgame solver (manual ranges)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    p.add_argument("--board", required=True, help="3 cards: 'Tc 7h 3d'")
    p.add_argument("--oop-range", required=True)
    p.add_argument("--ip-range",  required=True)
    p.add_argument("--pot", type=float, required=True)
    p.add_argument("--oop-stack", type=float, required=True)
    p.add_argument("--ip-stack",  type=float, required=True)
    p.add_argument("--first-to-act", type=int, default=0, choices=[0, 1])
    p.add_argument("--hand", default=None)
    p.add_argument("--position", default=None)
    p.add_argument("--path", default=None, help="Action path to target decision node (default: flop root)")
    p.add_argument("--turn", default=None, dest="turn_card", help="Turn card if querying a turn node")
    p.add_argument("--vs-action", default=None)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--buckets", type=int, default=16)
    p.add_argument("--max-raises", type=int, default=1)
    p.add_argument("--turn-max-raises", type=int, default=1)
    p.add_argument("--river-max-raises", type=int, default=1)
    p.add_argument("--no-subset", action="store_true",
                   help="Use all turn/river cards (slower; default uses rank subsample)")
    args = p.parse_args()

    if args.hand and not args.position:
        print(json.dumps({"error": "--position required when --hand given"}), file=sys.stderr); sys.exit(1)

    try:
        binary = find_binary("solve_flop_subgame")
        board = parse_board(args.board, 3)
    except (FileNotFoundError, ValueError) as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr); sys.exit(1)

    mask = board_mask(board)
    oop_w, ip_w = parse_ranges_or_exit(args.oop_range, args.ip_range)
    oop_w = zero_board_conflicts(oop_w, mask)
    ip_w  = zero_board_conflicts(ip_w, mask)
    if sum(oop_w) == 0 or sum(ip_w) == 0:
        print(json.dumps({"error": "after board conflicts, one side has empty range"}), file=sys.stderr); sys.exit(1)

    spec = {
        "board": board, "oop_weights": oop_w, "ip_weights": ip_w,
        "pot": args.pot, "oop_stack": max(0.0, args.oop_stack),
        "ip_stack": max(0.0, args.ip_stack),
        "first_to_act": args.first_to_act,
        "iterations": args.iterations, "k_buckets": args.buckets,
        "max_raises": args.max_raises,
        "turn_max_raises": args.turn_max_raises,
        "river_max_raises": args.river_max_raises,
        "use_subset": not args.no_subset,
    }
    try:
        result = call_solver(binary, spec, timeout_s=300)  # flop can take a bit longer
    except RuntimeError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr); sys.exit(1)

    is_ip = args.position and args.position.lower() in ("ip", "sb", "1")
    hero_player = 1 if is_ip else 0

    hero_bucket_idx, hero_bucket_hands, hero_ehs = (-1, [], None)
    if args.hand:
        hero_bucket_idx, hero_bucket_hands, hero_ehs = find_hero_bucket(result, args.hand, is_ip)

    street = "turn" if args.turn_card else "flop"
    node, resolved_vs = find_primary_node(
        result, hero_player, street=street, turn_card=args.turn_card,
        path=args.path, vs_action=args.vs_action,
    )

    if node and hero_bucket_idx >= 0:
        table_md = format_table(
            node["action_labels"], node["strategy"][hero_bucket_idx],
            hero_bucket_hands, args.pot, hero_ehs,
        )
    elif node:
        table_md = format_table(
            node["action_labels"], node["strategy"][0],
            result.get("villain_bucket_hands" if is_ip else "bucket_hands", [[]])[0],
            args.pot, None,
        )
    else:
        table_md = "(no matching decision node found)"

    pos_label = ""
    if args.hand and args.position:
        pos_label = f"  |  **Hero:** {hand_type(args.hand)} ({'IP' if is_ip else 'OOP'})"
        if hero_bucket_idx < 0:
            pos_label += " *(hand not in specified range)*"
        if is_ip and resolved_vs:
            pos_label += f"  |  **Facing:** OOP {resolved_vs}"
    if args.turn_card:
        pos_label += f"  |  **Turn:** {args.turn_card}"

    context = (
        f"**Board:** {board_to_str(board)}  |  **Mode:** flop on-demand (manual ranges)  |  "
        f"**Pot:** {args.pot:.1f} BB  |  **Eff stack:** {min(args.oop_stack, args.ip_stack):.1f} BB"
        f"{pos_label}"
    )

    out = {
        "context": context,
        "table": table_md,
        "action_labels": node["action_labels"] if node else result["action_labels"],
        "hero_strategy": node["strategy"] if node else result["hero_strategy"],
        "bucket_hands": result.get("bucket_hands", []),
        "villain_bucket_hands": result.get("villain_bucket_hands", []),
        "hero_ehs_range": result.get("hero_ehs_range", []),
        "villain_ehs_range": result.get("villain_ehs_range", []),
        "hero_value": result["hero_value"],
        "hero_bucket_idx": hero_bucket_idx,
        "vs_action": resolved_vs,
        "pot": args.pot,
        "iterations": result["iterations"],
        "street": street,
        "turn_card": args.turn_card,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
