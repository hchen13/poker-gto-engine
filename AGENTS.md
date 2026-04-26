# AGENTS.md — operator guide for LLM agents

This file is the structured "how to operate this repo" guide for an LLM agent.
Humans should read `README.md`. Agents working on this repo should read this.

## TL;DR

```
git clone <repo>
cd poker-gto-engine
./install.sh --yes
poker query 3bet_called 200 K62r KQo ip check
```

If those four lines run without error and the last one prints JSON
containing `"strategy"`, the skill is functional.

## Install

One command, idempotent:

```
./install.sh
```

Equivalent: `python3 install.py`. Add `--yes` for non-interactive mode.
The installer:
1. Verifies Python ≥ 3.10
2. Verifies `cargo` and `uv` (errors out with install URLs if missing)
3. Builds Rust binaries (~5–15 min cold; ~5s warm)
4. Installs the `poker` CLI via `uv tool install --editable .`
5. Smoke-tests `poker find-flop`
6. Reports precompute data status

After install, `poker --help` lists subcommands.

## Verify

```
poker query 3bet_called 200 K62r KQo ip check
```

Expected: JSON with `"context"`, `"table"`, `"strategy"` (a list of floats
summing to 1.0). If you get `{"error": ...}`, read the `"hint"` field — it
tells you exactly what to fix.

## Use

All skill instructions are in `skill/poker/SKILL.md`. The agent-facing
contract is the `poker` CLI — never edit Python in `skill/poker/lib/` or
`skill/poker/bin/` to "make a query work"; if a query fails, it's either
out of precompute coverage (→ use `poker solve-*-manual`) or the agent
parsed the spot wrong (→ re-read SKILL.md "Step 1: Parse the spot").

## Subcommands

```
poker find-flop          <action_line> <stack_bb> <flop>
poker paths              <action_line> <stack_bb> <flop>
poker query              <action_line> <stack_bb> <flop> <hand> <position> <action_path> [--turn TURN]
poker solve-river        <action_line> <stack_bb> <flop> <action_path> --turn T --river R --hand H --position P
poker solve-river-manual --board "..." --oop-range "..." --ip-range "..." --pot N --oop-stack N --ip-stack N --hand H --position P
poker solve-turn-manual  ... (same shape, board has 4 cards)
poker solve-flop-manual  ... (same shape, board has 3 cards; ~30–60s)
poker profile            <name_or_archetype>
```

Run `poker <sub> --help` for argument details on any subcommand.

## Precompute data

The `precompute_out/` directory holds ~1.4GB of CFR+ solutions across 14
preflop variants × 1755 canonical flops. Without it, `poker query` and
`poker paths` return errors; the on-demand `solve-*` subcommands still
work standalone.

Provision options (in order of preference):
1. Download from GitHub Release: see `install.py` step 5 output for the
   command. Extract to `precompute_out/`.
2. Generate locally: `./target/release/precompute_bucketed_parallel
   fixtures/precompute/v2/<spec>.json`. ~1h per spec, 14 specs total.
3. Override location with env var: `export POKER_PRECOMPUTE_DIR=/path/...`

## Skill registration (optional, manual)

The installer does NOT modify `~/.claude/` or `~/.hermes/`. To wire the
skill into a specific agent platform:

**Claude Code:**
```
mkdir -p ~/.claude/skills
ln -sf "$(pwd)/skill/poker" ~/.claude/skills/poker
```

**Hermes:**
```
cp -R skill/poker ~/.hermes/skills/gaming/poker-gto
# Or for a specific profile: ~/.hermes/profiles/<name>/skills/gaming/poker-gto
```

These commands are also printed at the end of `install.py`.

## Common failure modes

- **`poker: command not found`**: `~/.local/bin` not on PATH. Fix:
  `export PATH="$HOME/.local/bin:$PATH"` (add to shell rc file).
- **`cargo: command not found`**: install Rust:
  `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **`No precomputed file for ...`**: precompute data not provisioned, or
  spot is outside coverage. Either download data, or fall back to
  `poker solve-*-manual`.
- **`Module not found: zstandard`**: shouldn't happen if you used
  `install.sh` — the CLI's deps are isolated in its own uv-managed env.
  If it does, re-run `uv tool install --editable .` from the repo root.

## Repository layout (what an agent should know)

```
poker-gto-engine/
├── install.sh, install.py    # one-shot installer
├── pyproject.toml            # declares deps + `poker` entry point
├── AGENTS.md                 # this file
├── README.md                 # human-facing
├── Cargo.toml, src/, tests/  # Rust solver
├── skill/poker/              # the Python skill
│   ├── SKILL.md              # skill instructions (the LLM reads this)
│   ├── INSTALL.md            # legacy; superseded by install.sh
│   ├── cli.py                # `poker` dispatcher
│   ├── bin/*.py              # one .py per subcommand
│   ├── lib/*.py              # shared helpers
│   └── state/                # opponent profiles
├── scripts/                  # ops scripts (compress, sync, etc.)
├── fixtures/                 # test inputs and precompute job specs
└── precompute_out/           # downloaded data, .mpk.zst files
```
