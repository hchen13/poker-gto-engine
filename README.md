# poker-gto-engine

NLHE GTO solver + agent skill for analyzing real hands.

Two layers:
- **Solver**: CFR+ implementation in Rust (with Python reference) for river / turn / flop / preflop subgames
- **`/poker` skill**: agent-portable wrapper (`skill/poker/`) that takes a natural-language hand description and returns a `{action, 频率, EV, 理由}` table

## Quickstart

```bash
# Build solver
cargo build --release --bin solve

# (Optional) Build precompute orchestrator
cargo build --release --bin precompute

# Run a river spot via Rust CLI
./target/release/solve --input-file fixtures/rust/river_polarized.json

# Or via Python wrapper (uses Rust under the hood)
python3 -m python.analyze_spot --input-file fixtures/nlhe-river/solve_with_exploitability_rust.json

# Run tests
python3 -m pytest tests/ -q     # 129 Python tests
cargo test --lib --release       # 50 Rust tests
```

## Repo layout

| Dir | Contents |
|---|---|
| `python/nlhe/` | Reference Python solver (river / turn / flop / preflop modules) |
| `src/nlhe/` | Rust port (10× – 70× faster, same numerical output) |
| `src/bin/` | `solve` (subgame CLI), `precompute` (offline orchestrator), `bench_*` |
| `skill/poker/` | Agent-portable `/poker` skill — see `skill/poker/INSTALL.md` |
| `fixtures/` | Sample input JSONs for solver + precompute |
| `precompute_out/` | Output of precompute runs (queryable cache for the skill) |
| `tests/` | Python unit tests; Rust tests live inside `src/` |
| `docs/` | `roadmap.md` (current plan), `architecture.md`, `increment1-progress.md` |

## Solver capabilities

- **River HU** subgame: ~70 ms per 1000-iter solve (Rust)
- **Turn HU** subgame: 10–60 s for typical spots; chance node averaging over 48 rivers
- **Flop HU** subgame: minutes; double chance averaging (turn × river); supports public-card abstraction
- **Preflop HU**: CFR+ with precomputed 169×169 equity table at terminals
- **Best-response / exploitability** validator for self-checking convergence
- **Card abstraction** (rank-based for now; equity-based is future work)

All solvers use the same JSON I/O via the `solve` binary, so the agent can talk to one CLI for any street.

## Precompute

The `precompute` binary builds GTO baseline tables that the `/poker` skill consults before falling back to on-demand solves:

```bash
./target/release/precompute fixtures/precompute/hu_200bb_full.json
```

Sample config in `fixtures/precompute/`. A single variant × single stack (1755 flops, 8 workers, k=16 buckets, 200 iterations) takes 1-5h wall clock depending on range width and SPR. The orchestrator (`precompute_bucketed_parallel`) is built to run unattended, with `progress.json` updated per-flop and per-thread profiling in the log.

After precompute finishes, run `patch_villain_buckets` and `patch_bucket_arrays` to add IP-side bucket hands and combo→bucket arrays (needed by the river on-demand solver). `scripts/ip3bet_post.sh` / `ip_variants_post.sh` demonstrate how to chain precompute + rolling gzip + patches in one background pipeline.

The skill auto-discovers `precompute_out/` and uses what it finds; missing spots fall back to on-demand.

## `/poker` skill

The skill lives entirely under `skill/poker/`. Install steps for each agent are in `skill/poker/INSTALL.md`.

End-user flow:
1. Describe a hand in natural language (positions, stacks, action history, board, hero cards, opponent type)
2. Skill parses → builds JSON spec → calls solver (cache or on-demand)
3. Output: a `{action, 频率, EV, 理由}` table for the user's specific decision point

Frequencies and EVs come from the solver. The 理由 column is LLM prose anchored on those numbers — never a guess.

## Status snapshot (2026-04-25)

- Increments 1-5 complete. Solver + precompute + skill all shipped, plus full on-demand fallback pipeline.
- Precomputed tables: 7 action-line variants × 2 stack depths × 1755 canonical iso-class flops = 24,570 files
  - **Canonical enumeration** (abstraction::enumerate_canonical_flops) guarantees one representative per suit-isomorphism class, 100% coverage of the 22,100 real flops
  - Core HU lines: `limped`, `sr_called`, `3bet_called`, `4bet_called`
  - 6-max role-swap variants (caller-in-position): `sr_called_ip_caller`, `3bet_called_ip3bet`, `4bet_called_ip_caller`
- **On-demand bucketed subgame solvers** (src/bin/solve_{river,turn,flop}_subgame):
  - River: ~100ms (K=16 CFR+)
  - Turn: ~500ms (K=16 + river chance subsample)
  - Flop: 30-60s (K=16 + turn+river chance subsamples) — within 2-min interactive budget
- Skill CLIs (skill/poker/bin/): `find_flop.py`, `paths.py`, `query.py`, `solve_river.py`, `solve_river_manual.py`, `solve_turn_manual.py`, `solve_flop_manual.py`, `get_profile.py`
- Query cascade: exact file → iso-key canonical lookup → rank+texture rainbow preference → `*_manual.py` escape hatch for anything precompute can't cover (exotic stacks, limp-raise, etc.)
- Skill installed at `~/.claude/skills/poker/` (hardlinked to project) and `~/.hermes/skills/gaming/poker-gto/` (sync via scripts/sync_hermes_skill.sh)
- Tests: 144 Python (including 20 skill-CLI + 2 iso-key invariance) + 50+ Rust

See `docs/roadmap.md` for historical plan; `skill/poker/SKILL.md` for usage.

## Old phase-0 demos

Kuhn / Leduc reference implementations and CLIs still live under `python/` and `src/` for regression-testing the CFR core:

```bash
./scripts/e2e_kuhn_demo.sh
./scripts/e2e_leduc_demo.sh
```
