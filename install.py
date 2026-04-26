#!/usr/bin/env python3
"""One-shot installer for the poker-gto-engine skill.

Idempotent. Run from the repo root:

    python3 install.py            # interactive, prompts on missing tools
    python3 install.py --yes      # non-interactive, fails fast on missing tools
    python3 install.py --skip-rust       # skip cargo build (use existing target/)
    python3 install.py --skip-precompute # skip precompute data download

After install, the `poker` command is on PATH (~/.local/bin/poker on macOS/Linux).
Run `poker --help` to verify.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RED, GREEN, YELLOW, RESET = "\033[31m", "\033[32m", "\033[33m", "\033[0m"
if not sys.stdout.isatty():
    RED = GREEN = YELLOW = RESET = ""


def step(n: int, total: int, msg: str) -> None:
    print(f"{GREEN}[{n}/{total}]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}warning:{RESET} {msg}", file=sys.stderr)


def die(msg: str, code: int = 1) -> None:
    print(f"{RED}error:{RESET} {msg}", file=sys.stderr)
    sys.exit(code)


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def check_python() -> None:
    if sys.version_info < (3, 10):
        die(f"Python ≥ 3.10 required, got {sys.version.split()[0]}")


def check_rust(yes: bool) -> None:
    if have("cargo"):
        return
    msg = ("cargo not found. Install Rust toolchain via:\n"
           "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh\n"
           "Then re-run this installer.")
    die(msg)


def check_uv(yes: bool) -> None:
    if have("uv"):
        return
    msg = ("uv not found. Install via:\n"
           "    curl -LsSf https://astral.sh/uv/install.sh | sh\n"
           "Then re-run this installer.")
    die(msg)


def build_rust() -> None:
    binaries = [
        "precompute_bucketed_parallel",
        "solve_river_subgame",
        "solve_turn_subgame",
        "solve_flop_subgame",
    ]
    args = ["cargo", "build", "--release"]
    for b in binaries:
        args += ["--bin", b]
    run(args)


def install_cli() -> None:
    run(["uv", "tool", "install", "--editable", str(ROOT)])


def smoke_test() -> bool:
    poker = shutil.which("poker")
    if not poker:
        warn("`poker` not on PATH. uv typically installs to ~/.local/bin — "
             "add it to PATH and re-run smoke test manually.")
        return False
    try:
        out = subprocess.run(
            [poker, "find-flop", "3bet_called", "200", "K62r"],
            capture_output=True, text=True, timeout=30, check=True,
        )
    except subprocess.CalledProcessError as e:
        warn(f"smoke test failed: {e.stderr.strip() or e.stdout.strip()}")
        return False
    if "flop_label" in out.stdout:
        return True
    warn(f"smoke test produced unexpected output: {out.stdout[:200]}")
    return False


def precompute_status() -> tuple[bool, int]:
    """Returns (has_data, file_count) for default precompute_out/."""
    d = ROOT / "precompute_out"
    if not d.is_dir():
        return False, 0
    n = sum(1 for _ in d.rglob("flop_*.mpk.zst")) + sum(1 for _ in d.rglob("flop_*.json.gz"))
    return n > 0, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true",
                    help="Non-interactive; fail fast on missing tools.")
    ap.add_argument("--skip-rust", action="store_true",
                    help="Skip cargo build (re-use existing target/release/ binaries).")
    ap.add_argument("--skip-precompute", action="store_true",
                    help="Skip precompute data status check (CLI install will still finish).")
    args = ap.parse_args()

    total = 5
    step(1, total, "Check Python ≥ 3.10")
    check_python()

    step(2, total, "Check Rust + uv toolchain")
    check_uv(args.yes)
    if not args.skip_rust:
        check_rust(args.yes)

    step(3, total, "Build Rust binaries (release)")
    if args.skip_rust:
        print("  skipped (--skip-rust)")
    else:
        build_rust()

    step(4, total, "Install `poker` CLI via uv tool install --editable")
    install_cli()

    step(5, total, "Verify install")
    if smoke_test():
        print(f"  {GREEN}✓{RESET} `poker find-flop` works")
    else:
        warn("smoke test inconclusive — check `poker --help` manually.")

    if not args.skip_precompute:
        has, n = precompute_status()
        print()
        if has:
            print(f"{GREEN}precompute data:{RESET} {n} files in precompute_out/ — ready.")
        else:
            print(f"{YELLOW}precompute data:{RESET} not found. The skill works on-demand without it,")
            print("  but precomputed flop/turn queries will be unavailable.")
            print("  Options:")
            print("    - Download release tarball (~1.4GB):")
            print("        gh release download v0.1 --pattern 'precompute_out_v2.tar.zst'")
            print("        zstd -d precompute_out_v2.tar.zst -c | tar -x")
            print("    - Generate locally (~14h, requires built binaries):")
            print("        ./target/release/precompute_bucketed_parallel "
                  "fixtures/precompute/v2/full_3bet_called_200bb.json")
            print("    - Override location with env var:")
            print("        export POKER_PRECOMPUTE_DIR=/path/to/precompute_out")

    print()
    print(f"{GREEN}Install complete.{RESET}")
    print()
    print("Next steps:")
    print("  - Try: poker query 3bet_called 200 K62r KQo ip check")
    print("  - Register skill (optional, manual):")
    print(f"    Claude Code: ln -sf {ROOT}/skill/poker ~/.claude/skills/poker")
    print(f"    Hermes:      cp -R {ROOT}/skill/poker ~/.hermes/skills/gaming/poker-gto")
    print()
    print("Agents: see AGENTS.md at the repo root for the structured operator guide.")


if __name__ == "__main__":
    main()
