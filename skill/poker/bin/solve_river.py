#!/usr/bin/env python3
"""Solve the river subgame on-demand using bucketed CFR+.

Usage:
    python solve_river.py <action_line> <stack_bb> <flop> <action_path> --turn <card> --river <card> [options]

Arguments:
    action_line   : limped | sr_called | 3bet_called | 4bet_called
    stack_bb      : 200 or 500
    flop          : rank+texture — e.g. AK7r, QT9s
    action_path   : postflop actions through end of turn — e.g. "check/check/check/check"
                    Use "" or "check/check" etc. Must include all flop + turn actions.

Options:
    --turn <card>      Turn card, e.g. Ts (REQUIRED)
    --river <card>     River card, e.g. 2h (REQUIRED)
    --hand <hand>      Hero's hand, e.g. AKo — shows hero's bucket strategy
    --position <pos>   Hero's position: oop or ip (required if --hand given)
    --iterations N     CFR+ iterations (default 500)
    --buckets K        Number of range buckets (default 16)

Output (JSON):
    {
      "context": "Board: ... | Pot: ... | ...",
      "table": "**Pot:** 36.0 BB | Bucket 12 ...",
      "action_labels": ["fold","call","bet_X",...],
      "hero_strategy": [[f32 × n_actions] × k_buckets],
      "bucket_hands": [["AKo","AKs",...], ...],
      "hero_value": f32,
      "hero_bucket_idx": int,    (-1 if hand not given or not found)
      ...
    }

Examples:
    # River after flop/turn check-check, turn=Ts, river=2h (OOP leads river)
    python solve_river.py 3bet_called 500 AK7r "check/check/check/check" --turn Ts --river 2h

    # With hero hand to see their specific bucket
    python solve_river.py 3bet_called 500 AK7r "check/check/check/check" --turn Ts --river 2h \\
      --hand AKo --position oop

    # After OOP bets turn, IP calls
    python solve_river.py 3bet_called 500 AK7r "check/check/bet_18.00/call" --turn Ts --river 2h \\
      --hand KQo --position ip
"""

import argparse
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from skill.poker.lib.river_solver import solve_river


def main():
    parser = argparse.ArgumentParser(
        description="On-demand river subgame solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("action_line", help="limped|sr_called|3bet_called|4bet_called")
    parser.add_argument("stack_bb", type=int, help="200 or 500")
    parser.add_argument("flop", help="Rank+texture: AK7r, QT9s, KK2m")
    parser.add_argument("action_path", help='Actions through end of turn: "" | "check/check/check/check"')
    parser.add_argument("--turn", dest="turn_card", required=True, help="Turn card: Ts, 2h, Kd")
    parser.add_argument("--river", dest="river_card", required=True, help="River card: 2h, 9s, As")
    parser.add_argument("--hand", dest="hero_hand", default=None, help="Hero hand: AKo, KQs, JJ")
    parser.add_argument("--position", dest="position", default=None, help="oop or ip")
    parser.add_argument("--vs-action", dest="vs_action", default=None,
                        help="For IP hero: which OOP river action to respond to (e.g. 'check' or 'bet_24.00'). "
                             "If omitted, defaults to OOP's most-frequent action.")
    parser.add_argument("--iterations", type=int, default=500, help="CFR+ iterations (default 500)")
    parser.add_argument("--buckets", type=int, default=16, help="Range buckets (default 16)")

    args = parser.parse_args()

    if args.hero_hand and not args.position:
        print(json.dumps({"error": "--position required when --hand is given"}), file=sys.stderr)
        sys.exit(1)

    try:
        result = solve_river(
            action_line=args.action_line,
            stack_bb=args.stack_bb,
            flop=args.flop,
            action_path=args.action_path,
            river_card=args.river_card,
            hero_hand=args.hero_hand,
            position=args.position,
            turn_card=args.turn_card,
            vs_action=args.vs_action,
            iterations=args.iterations,
            k_buckets=args.buckets,
        )
    except (FileNotFoundError, ValueError) as e:
        error_msg = str(e)
        hint = None
        if "No precomputed file" in error_msg:
            hint = "Run paths.py to verify the spot exists: python paths.py {} {} {}".format(
                args.action_line, args.stack_bb, args.flop)
        elif "missing" in error_msg and "bucket_of_combo" in error_msg:
            hint = "Run: cd ~/projects/poker-gto-engine && cargo run --release --bin patch_bucket_arrays"
        elif "binary not found" in error_msg:
            hint = "Run: cd ~/projects/poker-gto-engine && cargo build --release --bin solve_river_subgame"
        out = {"error": error_msg}
        if hint:
            out["hint"] = hint
        print(json.dumps(out, indent=2), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {e}"}), file=sys.stderr)
        sys.exit(1)

    output = {
        "context":              result["context"],
        "table":                result["table"],
        "action_labels":        result["action_labels"],
        "hero_strategy":        result["hero_strategy"],
        "bucket_hands":         result["bucket_hands"],
        "villain_bucket_hands": result["villain_bucket_hands"],
        "hero_ehs_range":       result["hero_ehs_range"],
        "villain_ehs_range":    result["villain_ehs_range"],
        "hero_value":           result["hero_value"],
        "hero_bucket_idx":      result["hero_bucket_idx"],
        "vs_action":            result.get("vs_action"),
        "ip_responses":         result.get("ip_responses", {}),
        "board":                result["board"],
        "pot":                  result["pot"],
        "iterations":           result["iterations"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
