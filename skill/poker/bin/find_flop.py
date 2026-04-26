#!/usr/bin/env python3
"""Find the closest matching precomputed flop file.

Usage:
    python find_flop.py <action_line> <stack_bb> <flop>

Examples:
    python find_flop.py 3bet_called 500 AK7r
    python find_flop.py 3bet_called 500 AhKd7c
    python find_flop.py sr_called 200 QT9s

Output (JSON):
    {
      "flop_label": "7cKhAd",
      "file": "/path/to/flop_7cKhAd.json.gz",
      "texture": "rainbow",
      "action_line": "3bet_called",
      "stack_bb": 500
    }

Exit 0 = found, Exit 1 = not found (error in stderr).
"""

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from skill.poker.lib.query_precompute import find_flop_file, _suit_count


def texture_name(label: str) -> str:
    cards = [label[i:i+2] for i in range(0, 6, 2)]
    sc = _suit_count(cards)
    return {1: "monotone", 2: "flush_draw", 3: "rainbow"}.get(sc, "unknown")


def main():
    if len(sys.argv) < 4:
        print("Usage: find_flop.py <action_line> <stack_bb> <flop>", file=sys.stderr)
        print("Example: find_flop.py 3bet_called 500 AK7r", file=sys.stderr)
        sys.exit(1)

    action_line, stack_bb_str, flop = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        stack_bb = int(stack_bb_str)
    except ValueError:
        print(f"Error: stack_bb must be an integer (got {stack_bb_str!r})", file=sys.stderr)
        sys.exit(1)

    try:
        f = find_flop_file(action_line, stack_bb, flop)
    except (ValueError, FileNotFoundError) as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    if f is None:
        print(json.dumps({
            "error": f"No precomputed file found for {action_line}/{stack_bb}bb/{flop}",
            "hint": "Try a different texture suffix: r=rainbow, s=flush_draw, m=monotone"
        }))
        sys.exit(1)

    # Handle all supported suffixes: .mpk.zst, .json.gz, .json
    fname = f.name
    for suffix in (".mpk.zst", ".json.gz", ".json"):
        if fname.endswith(suffix):
            label = fname[len("flop_"):-len(suffix)]
            break
    else:
        label = f.stem.replace("flop_", "")

    print(json.dumps({
        "flop_label": label,
        "file": str(f),
        "texture": texture_name(label),
        "action_line": action_line,
        "stack_bb": stack_bb,
    }, indent=2))


if __name__ == "__main__":
    main()
