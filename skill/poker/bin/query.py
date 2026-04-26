#!/usr/bin/env python3
"""Query GTO strategy for a specific hand in a specific spot.

Usage:
    python query.py <action_line> <stack_bb> <flop> <hand> <position> <action_path> [--turn <card>]

Arguments:
    action_line   : limped | sr_called | 3bet_called | 4bet_called
    stack_bb      : 200 or 500
    flop          : rank+texture — e.g. AK7r, QT9s, KK2m, T98r
                    (r=rainbow, s=flush_draw, m=monotone)
    hand          : hand type — e.g. KQo, AQs, JJ, TT
    position      : ip (BTN/SB, acts second) | oop (UTG/BB, acts first)
    action_path   : postflop actions so far — e.g. "" | "check" | "check/bet_11.88/call"
                    Use "" for root (hero acts first). Use paths.py to see valid options.
    --turn <card> : turn card if querying turn node — e.g. Ts, 2h, Kd

Examples:
    # OOP at flop root
    python query.py 3bet_called 500 AK7r AKo oop ""

    # IP after OOP checks
    python query.py 3bet_called 500 AK7r KQo ip check

    # OOP on turn after check/check, turn=Ts
    python query.py 3bet_called 500 AK7r AKo oop "check/check" --turn Ts

    # IP on turn after OOP checks turn
    python query.py 3bet_called 500 AK7r KQo ip "check/check/check" --turn Ts

Output (JSON):
    {
      "context": "Flop: 7cKhAd | Line: 3bet_called/500bb | Hero: AKo (OOP) | ...",
      "table": "**Pot:** 36.0 BB | Bucket 12 (K9s, AKo, AKs)\n\n| 操作 | 频率 | ...",
      "flop_label": "7cKhAd",
      "bucket_idx": 12,
      "bucket_hands": ["K9s", "AKo", "AKs"],
      "ehs_range": [0.908, 0.938],
      "node": { "player": 0, "street": "flop", "path": "", ... }
    }

Exit 0 = success, Exit 1 = error with message in stderr.
"""

import argparse
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from skill.poker.lib.query_precompute import query


def main():
    parser = argparse.ArgumentParser(
        description="Query precomputed GTO strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("action_line", help="limped|sr_called|3bet_called|4bet_called")
    parser.add_argument("stack_bb", type=int, help="200 or 500")
    parser.add_argument("flop", help="Rank+texture: AK7r, QT9s, KK2m")
    parser.add_argument("hand", help="Hand type: KQo, AQs, JJ")
    parser.add_argument("position", help="ip or oop")
    parser.add_argument("action_path", help='Action path: "" | "check" | "check/bet_11.88/call"')
    parser.add_argument("--turn", dest="turn_card", default=None,
                        help="Turn card: Ts, 2h, Kd")

    args = parser.parse_args()

    try:
        result = query(
            action_line=args.action_line,
            stack_bb=args.stack_bb,
            flop=args.flop,
            hero_hand=args.hand,
            position=args.position,
            action_path=args.action_path,
            turn_card=args.turn_card,
        )
    except (FileNotFoundError, ValueError) as e:
        error_msg = str(e)
        # Provide actionable hints
        hint = None
        if "No precomputed file" in error_msg:
            hint = "Run: python paths.py {} {} {} to verify the flop exists".format(
                args.action_line, args.stack_bb, args.flop)
        elif "not found in any" in error_msg and "bucket" in error_msg:
            hint = (
                "Hand '{}' is not in the {} range for {}. "
                "Check range table in SKILL.md.".format(
                    args.hand, args.position, args.action_line)
            )
        elif "No node found" in error_msg:
            hint = "Run: python paths.py {} {} {} to see valid action_paths".format(
                args.action_line, args.stack_bb, args.flop)

        out = {"error": error_msg}
        if hint:
            out["hint"] = hint
        print(json.dumps(out, indent=2), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {e}"}), file=sys.stderr)
        sys.exit(1)

    # Strip non-serializable node object, keep key fields
    output = {
        "context": result["context"],
        "table": result["table"],
        "flop_label": result["flop_label"],
        "bucket_idx": result["bucket_idx"],
        "bucket_hands": result["bucket_hands"],
        "dispersion_warning": result.get("dispersion_warning"),
        "ehs_range": result["ehs_range"],
        "node_path": result["node"]["path"],
        "node_street": result["node"]["street"],
        "node_player": result["node"]["player"],
        "action_labels": result["node"]["action_labels"],
        "strategy": result["node"]["strategy"][result["bucket_idx"]],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
