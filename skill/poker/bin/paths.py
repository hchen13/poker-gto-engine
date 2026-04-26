#!/usr/bin/env python3
"""List all valid action paths for a given spot.

Usage:
    python paths.py <action_line> <stack_bb> <flop>

Examples:
    python paths.py 3bet_called 500 AK7r
    python paths.py sr_called 200 QT9s

Output (JSON):
    {
      "flop_label": "7cKhAd",
      "action_line": "3bet_called",
      "stack_bb": 500,
      "pot_bb": 36.0,
      "oop_flop_paths": ["", "check/allin_182.00", "check/bet_11.88", ...],
      "ip_flop_paths":  ["check", "bet_11.88", "bet_18.00", ...],
      "turn_cards_sample": ["2s", "3s", "4s", "5s", "6s"],
      "turn_count": 44,
      "note": "For turn nodes: use action_path from oop/ip_flop_paths + turn_card=<card>"
    }

Exit 0 = success, Exit 1 = error.
"""

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from skill.poker.lib.query_precompute import (
    find_flop_file, _load_json, ACTION_LINE_POTS
)


def main():
    if len(sys.argv) < 4:
        print("Usage: paths.py <action_line> <stack_bb> <flop>", file=sys.stderr)
        print("Example: paths.py 3bet_called 500 AK7r", file=sys.stderr)
        sys.exit(1)

    action_line, stack_bb_str, flop = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        stack_bb = int(stack_bb_str)
    except ValueError:
        print(json.dumps({"error": f"stack_bb must be integer, got {stack_bb_str!r}"}))
        sys.exit(1)

    try:
        f = find_flop_file(action_line, stack_bb, flop)
    except (ValueError, FileNotFoundError) as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    if f is None:
        print(json.dumps({
            "error": f"No precomputed file for {action_line}/{stack_bb}bb/{flop}",
            "hint": "Try r/s/m suffix: AK7r=rainbow, AK7s=flush_draw, AK7m=monotone"
        }))
        sys.exit(1)

    data = _load_json(f)
    nodes = data["nodes"]

    # Strip any supported extension: .mpk.zst, .json.gz, .json
    fname = f.name
    for suffix in (".mpk.zst", ".json.gz", ".json"):
        if fname.endswith(suffix):
            label = fname[len("flop_"):-len(suffix)]
            break
    else:
        label = f.stem.replace("flop_", "")

    flop_oop = sorted({n["path"] for n in nodes if n["street"] == "flop" and n["player"] == 0})
    flop_ip  = sorted({n["path"] for n in nodes if n["street"] == "flop" and n["player"] == 1})
    turn_cards = sorted({n.get("turn_card", "") for n in nodes
                         if n["street"] == "turn" and n.get("turn_card")})

    # Also build turn paths per position (deduplicated)
    turn_oop = sorted({n["path"] for n in nodes if n["street"] == "turn" and n["player"] == 0})
    turn_ip  = sorted({n["path"] for n in nodes if n["street"] == "turn" and n["player"] == 1})

    print(json.dumps({
        "flop_label": label,
        "action_line": action_line,
        "stack_bb": stack_bb,
        "pot_bb": ACTION_LINE_POTS.get(action_line, 0),
        "oop_flop_paths": flop_oop,
        "ip_flop_paths": flop_ip,
        "oop_turn_paths": turn_oop[:20],
        "ip_turn_paths": turn_ip[:20],
        "turn_cards_sample": turn_cards[:10],
        "turn_count": len(turn_cards),
        "total_nodes": len(nodes),
        "note": (
            "action_path for turn nodes: pick from oop/ip_turn_paths above. "
            "Add turn_card=<card> to the query command."
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
