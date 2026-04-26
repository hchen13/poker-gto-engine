# poker-gto-engine

**English** | [中文](README.CN.md)

**A real GTO solver, packaged as an LLM agent skill.**

Tell your AI friend in plain language — "I'm in BB, 3-bet pot, A♥K♠ on K62 rainbow, villain checks, what now?" — and instead of making up frequencies, the agent queries a precomputed CFR+ table (or runs an on-demand subgame solve) and shows you the GTO mixed strategy with reasoning anchored on the actual numbers.

```
You:  6-max 200BB cash. BTN opens, I 3-bet from BB, BTN calls. Flop K♦ 6♥ 2♣.
      I have A♥K♣ — top pair top kicker, out of position. How should I play?

Agent runs:  poker query 3bet_called 200 K62r AhKd oop ""

**Flop:** 2s6hKd  |  **Line:** 3bet_called/200bb  |  **Hero:** AKo (OOP/BB)
**Pot:** 36.0 BB  |  Bucket 9 (AA, AKo, KQs, AKs)  |  EHS 0.82–0.83

| Action                | Frequency | Reason                                                          |
|-----------------------|-----------|-----------------------------------------------------------------|
| bet 33% pot (11.9 BB) | 56%       | Standard small c-bet — denies villain's free draws              |
| bet 50% pot (18.0 BB) | 41%       | Bigger mix; deep stacks let OOP build pot for turn/river        |
| check                 |  3%       | Rare — protects the OOP checking range with a strong hand       |
| bet 67% pot (24.1 BB) |  1%       | Low-frequency overbet line                                      |
```

Frequencies come from the solver. The reasoning column is LLM prose anchored on the actual numbers — never invented.

> Note: the skill's default `result["table"]` uses Chinese column headers (`操作 | 频率 | 理由`) because the project was bootstrapped Chinese-first. To localize, edit the table renderer in `skill/poker/lib/query_precompute.py::format_strategy_table`.

## What's underneath that one command

- **24,570 precomputed CFR+ tables** — 14 preflop variants × 200/500 BB stack depths × 1755 canonical iso-class flops. ~1.4 GB compressed (msgpack + u8-quantized + zstd; ~5× smaller than gzip while preserving decision-level correctness).
- **On-demand bucketed subgame solvers** for spots the precompute doesn't cover — river (~100 ms), turn (~500 ms), flop (~36 s).
- **A single `poker` CLI** with 8 subcommands, designed for LLM agents to invoke via shell. Ships with a `SKILL.md` that Claude Code, Hermes, and other agent platforms can register.

The **postflop model is heads-up** — which fits the way most cash-game pots actually play out: multi-way preflop, then collapsed to a single villain by the flop. Real HU tables are also covered. Uses K=16 EHS-quantile bucketing. Practical tradeoffs for laptop hardware — not a PioSolver replacement, but real numbers, no hallucination.

## Install

```bash
git clone https://github.com/hchen13/poker-gto-engine
cd poker-gto-engine
./install.sh
```

The installer builds the Rust solver (5–15 min cold), installs the `poker` CLI to `~/.local/bin/`, runs a smoke test, and reports precompute data status. See [`AGENTS.md`](AGENTS.md) for the LLM-operator-friendly version.

Verify:

```bash
poker query 3bet_called 200 K62r KQo ip check
```

If you get JSON containing a `"strategy"` array, you're done.

## Subcommands

| Command                       | What it does                                                                  | Latency  |
|-------------------------------|-------------------------------------------------------------------------------|----------|
| `poker query`                 | Look up flop/turn GTO strategy in the precomputed tables                      | <100 ms  |
| `poker paths`                 | List valid action paths (bet sizes, lines) for a spot                         | <100 ms  |
| `poker find-flop`             | Find the canonical iso-class file for a flop                                  | <50 ms   |
| `poker solve-river`           | Solve a river decision on-demand using the precomputed turn root              | ~100 ms  |
| `poker solve-river-manual`    | Solve river with explicit ranges (no precompute lookup)                       | ~70 ms   |
| `poker solve-turn-manual`     | Solve turn subgame with explicit ranges                                       | ~500 ms  |
| `poker solve-flop-manual`     | Solve flop subgame with explicit ranges (within 2-min interactive budget)     | ~30–60 s |
| `poker profile`               | Look up an opponent profile (built-in `fish`/`nit`/`reg`/`maniac` or custom)  | <50 ms   |

`poker --help` lists everything; `poker <subcommand> --help` shows arguments.

## How it works

**Solver core** (Rust): CFR+ with K=16 EHS-quantile bucketing. Public-card chance subsampling for tractable flop/turn solves. Heads-up postflop tree; preflop format is whatever your real game is, as long as the pot is heads-up by the flop. The same flat-tree CFR+ kernel runs both precomputed jobs and on-demand subgames — `solve_river_subgame`, `solve_turn_subgame`, `solve_flop_subgame`, all in `src/bin/`.

**Canonical flop enumeration**: `abstraction::enumerate_canonical_flops()` returns exactly 1755 representatives — one per suit-isomorphism class — covering all 22,100 real flops. Lookup uses a 6-byte permutation-invariant key, so any specific flop maps to its canonical file in O(1).

**Compression**: precomputed strategies are quantized to u8 (1/255 precision) and stored as msgpack + zstd. Argmax preserved across all buckets; max per-cell error <0.005 (well below CFR+ convergence error). Result: ~5× smaller than gzipped JSON; the full library fits in 1.4 GB.

**Skill layer** (Python): `poker` CLI dispatches to either a precompute lookup or a Rust subgame solver. When a queried bucket is heterogeneous (e.g., AKs flush draw alongside A7o on a 2-suited flop), the result includes a `dispersion_warning` flagging that the displayed frequency is averaged across mixed structures, and steers the agent to `solve-*-manual` for hand-specific accuracy.

See [`docs/architecture.md`](docs/architecture.md) for deeper internals; [`SKILL.md`](skill/poker/SKILL.md) for the agent-facing skill manifest.

## Coverage

**Heads-up postflop only** (the most common cash-game flop is two-player after preflop folds). 200 BB and 500 BB stacks. Seven preflop variants — each precomputed at both stack depths:

| Variant                    | Story                                | Aggressor IP/OOP postflop |
|----------------------------|--------------------------------------|---------------------------|
| `limped`                   | SB limps, BB checks                  | —                         |
| `sr_called`                | SB opens, BB calls                   | SB = IP (HU)              |
| `sr_called_ip_caller`      | EP opens, IP calls (6-max)           | opener = OOP              |
| `3bet_called`              | BB 3-bets, SB calls                  | 3-bettor = OOP            |
| `3bet_called_ip3bet`       | IP 3-bets, EP calls (6-max)          | 3-bettor = IP             |
| `4bet_called`              | SB 4-bets, BB calls                  | 4-bettor = IP             |
| `4bet_called_ip_caller`    | EP 4-bets, IP calls (6-max)          | 4-bettor = OOP            |

Both flop and turn nodes are stored in each precompute file. River decisions are computed on-demand from the precomputed turn root. For multi-way preflop pots that collapse to HU postflop (typical 6-max/9-max cash), pick the variant whose IP/OOP layout matches your real spot — see [`SKILL.md`](skill/poker/SKILL.md) for a decision table.

## Tradeoffs (be honest)

- **K=16 EHS bucketing**. Bucket-level GTO is correct in expectation but flattens hands with similar EHS but different structure (e.g., AKs flush draw with A7o on a flushy board). The `dispersion_warning` field flags this; fall back to `solve-*-manual` for hand-specific solves.
- **Heads-up postflop only**. The solver assumes two players by the flop. Multi-way pots that stay multi-way past the flop aren't modeled — but the common cash-game case (multi-way preflop folding down to HU on the flop) IS the primary use case. Pick the variant whose IP/OOP layout matches your real spot.
- **Two stack depths**. 200 BB and 500 BB are precomputed; everything else uses on-demand manual solvers.
- **Range / pot approximation in 6-max+ formats**. Default ranges are HU-calibrated. 6-max actual ranges are tighter; multi-way pots are inflated. The on-demand solvers let you specify exact ranges and pot, which is the precise path for non-HU formats.
- **Not a PioSolver replacement**. This is a bucketed approximation tuned for laptop-grade hardware (8 GB RAM workable, 16 GB comfortable). For tournament-grade per-combo precision, use a commercial tool.

## For LLM agent builders

[`AGENTS.md`](AGENTS.md) is the structured operator guide for LLM agents that clone this repo. It covers install, verification, the `poker` CLI contract, common failure modes, and skill registration for Claude Code / Hermes. The skill manifest itself ([`skill/poker/SKILL.md`](skill/poker/SKILL.md)) is what the agent reads at runtime — it documents how to parse a hand description, pick the right `action_line` variant, run the query, and present the result with reasoning.

If you want to plug this into a different agent platform, the contract is just the `poker` CLI — JSON in, JSON out. No platform-specific hooks.

## Tests

```bash
python3 -m pytest tests/ -q     # ~144 Python tests (skill CLI, iso-key invariants, solver bridges)
cargo test --lib --release       # ~50 Rust tests (CFR+ correctness, abstraction, bucketing)
```

## Provisioning precompute data

The `precompute_out/` directory (~1.4 GB) is not tracked in git — it ships as a release tarball. After clone + install, either:

```bash
# Download the prebuilt tables (preferred):
gh release download v0.1 --pattern 'precompute_out_v2.tar.zst'
zstd -d precompute_out_v2.tar.zst -c | tar -x

# Or generate locally (~14 h on 8 cores):
./target/release/precompute_bucketed_parallel fixtures/precompute/v2/full_3bet_called_200bb.json
# (repeat for each of the 14 specs in fixtures/precompute/v2/)

# Or use a custom location:
export POKER_PRECOMPUTE_DIR=/path/to/precompute_out
```

Without precompute, `poker query` and `poker paths` return errors but `poker solve-*-manual` still works fully.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

CFR+ algorithm: Tammelin et al., *Solving Heads-Up Limit Hold'em Poker* (2015). Hand evaluator: standard 7-card hash-based ranking. The bucketing tradeoff and msgpack/zstd format are pragmatic engineering choices, not novel research.
