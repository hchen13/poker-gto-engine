"""`poker` CLI — single dispatch entry point for the skill.

After `uv tool install --editable .` (or pip install) at the project root,
this exposes the `poker` command on PATH. All subcommands wrap the existing
`bin/*.py` scripts and reuse their argv parsing — adding a subcommand is just
adding a row to `SUBCOMMANDS`.

Usage:
    poker <subcommand> [args...]
    poker --help               # list subcommands
    poker <subcommand> --help  # subcommand-specific help
"""

from __future__ import annotations

import importlib
import sys
import textwrap

# subcommand → (module under skill.poker.bin, one-line help)
SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "find-flop":          ("find_flop",          "Find the closest precomputed flop file for a spot"),
    "paths":              ("paths",              "List valid action paths for a spot"),
    "query":              ("query",              "Query precomputed GTO strategy for a flop/turn node"),
    "solve-river":        ("solve_river",        "Solve a river decision on-demand (~100ms)"),
    "solve-river-manual": ("solve_river_manual", "Solve river with manually-specified ranges"),
    "solve-turn-manual":  ("solve_turn_manual",  "Solve turn with manually-specified ranges (~500ms)"),
    "solve-flop-manual":  ("solve_flop_manual",  "Solve flop with manually-specified ranges (~30-60s)"),
    "profile":            ("get_profile",        "Look up an opponent profile"),
}


def _print_help() -> None:
    width = max(len(c) for c in SUBCOMMANDS)
    lines = ["poker — NLHE GTO hand analyzer\n",
             "Usage:  poker <subcommand> [args...]\n",
             "Subcommands:"]
    for cmd, (_mod, desc) in SUBCOMMANDS.items():
        lines.append(f"  {cmd:<{width}}  {desc}")
    lines.append("")
    lines.append("Run `poker <subcommand> --help` for subcommand-specific arguments.")
    print("\n".join(lines))


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        sys.exit(0)

    cmd = argv[0]
    if cmd not in SUBCOMMANDS:
        print(f"poker: unknown subcommand {cmd!r}\n", file=sys.stderr)
        _print_help()
        sys.exit(2)

    module_name, _desc = SUBCOMMANDS[cmd]
    # Rewrite argv so the wrapped script sees its own args; the program name
    # is set so any usage/help messages show the canonical form.
    sys.argv = [f"poker {cmd}", *argv[1:]]
    module = importlib.import_module(f"skill.poker.bin.{module_name}")
    module.main()


if __name__ == "__main__":
    main()
