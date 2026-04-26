#!/usr/bin/env python3
"""Look up an opponent profile to inform strategy reasoning.

Profiles live in ``skill/poker/state/profiles/`` (archetype profiles that ship
with the skill) and ``skill/poker/state/`` (per-player profiles, filename =
player name lowercased). Both are markdown — free-form notes the LLM reads
and folds into the 理由 column.

The profile does NOT modify solver output. The GTO table stays the baseline;
the profile guides exploitative deviations in reasoning.

Usage:
    python get_profile.py <name_or_archetype>

Examples:
    python get_profile.py allen       # look for state/allen.md first, then state/profiles/allen.md
    python get_profile.py fish        # built-in fish archetype
    python get_profile.py nit

Output (JSON):
    {
      "name": "allen",
      "source": "/absolute/path/to/profile.md",
      "archetype": false,                  # true if matched a profiles/ archetype
      "markdown": "...full file contents..."
    }

Exit 1 if no profile found (error to stderr).
"""

import argparse
import json
import sys
from pathlib import Path


def profile_roots() -> list[Path]:
    root = Path(__file__).resolve().parents[1] / "state"
    return [root, root / "profiles"]


def find_profile(name: str) -> tuple[Path, bool] | None:
    """Return (path, is_archetype). Case-insensitive match on filename stem."""
    name_lc = name.lower()
    roots = profile_roots()
    for i, root in enumerate(roots):
        if not root.exists():
            continue
        for p in root.glob("*.md"):
            if p.stem.lower() == name_lc:
                return p, (i == 1)
        # Also allow "EXAMPLE_<name>.md" style
        for p in root.glob(f"EXAMPLE_{name_lc}*.md"):
            return p, False
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Look up opponent profile markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("name", help="Player name or archetype (fish/nit/reg/maniac)")
    args = parser.parse_args()

    result = find_profile(args.name)
    if result is None:
        available = []
        for root in profile_roots():
            if root.exists():
                available.extend(sorted(p.stem for p in root.glob("*.md")))
        hint = (
            f"No profile found for '{args.name}'. Available: {available}. "
            f"Archetypes: fish, nit, reg, maniac."
        )
        print(json.dumps({"error": hint}), file=sys.stderr)
        sys.exit(1)

    path, is_archetype = result
    try:
        markdown = path.read_text(encoding="utf-8")
    except Exception as e:
        print(json.dumps({"error": f"Read failed: {e}"}), file=sys.stderr)
        sys.exit(1)

    print(json.dumps({
        "name": args.name,
        "source": str(path),
        "archetype": is_archetype,
        "markdown": markdown,
    }, indent=2))


if __name__ == "__main__":
    main()
