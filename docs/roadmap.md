# Poker GTO Engine — Roadmap

**Target**: a Claude Code `/poker` skill. User describes a hand in natural
language; agent returns `{action, frequency, EV, reasoning}` per legal action,
driven by a real GTO solver backed by precomputed tables.

## Architecture (final state)

**Offline** (runs for hours/days, one-time build):
- Rust-ported solver
- Preflop GTO solve at 200BB and 500BB HU, with fine bucket abstraction
- Flop GTO solve for the standard strategically-distinct flops, per
  preflop action line
- All results persisted to disk in a queryable format

**Online** (seconds-level response):
- LLM parses the user's natural-language hand description → structured spot
- LLM maps the spot to the corresponding node in the precomputed tree
- Range at that node is derived recursively from the precomputed solution,
  optionally offset by the opponent-profile layer for non-GTO opponents
- For turn/river decisions beyond the flop precompute depth, invoke the
  on-demand subgame solver seeded with the derived range
- Format output as `{action, frequency, EV, reasoning}`; frequency and EV
  come from the solver, reasoning is written by the LLM anchored on those
  numbers

## Increment plan

### Done — Increment 1: River HU subgame MVP
- Hand evaluator, range parser, river game tree, CFR+ solver
- Card abstraction (E[HS] bucketing)
- Best-response / exploitability validator
- CLI with JSON in/out
- 116 tests passing

### Increment 2 — Turn solver (Python) ✅ DONE
- Extended `tree.py` with `is_chance` Node flag and `build_turn_tree`
- New modules: `turn_showdown.py` (per-river outcome matrices), `turn_cfr.py` (chance-aware CFR+), `turn_best_response.py` (exploitability validator)
- Chance node averages over 48 remaining cards; per-river showdown tables keyed by `showdown_board_key` on terminal nodes
- Tests: 9 new (chip conservation, tree structure, showdown, sanity, exploitability monotone descent)
- CLI: `nlhe_turn_solve` game type, optional `compute_exploitability`
- End-to-end verified on a real hand (99 on KJ9r-5 vs fish limp-3bet range)
- 125 tests total passing

**Known gap** (polish for later): when `first_to_act=1` (hero IP), the root strategy output shows villain's strategy. To surface hero's response strategy, a subnode extractor is needed — deferred as it's not blocking for Inc 3.

**Perf notes**: single-street river + turn Python CFR+ handles small ranges (≤10 combos/side) in seconds, medium ranges (≤40 combos) in ~1 minute per 40 iters with max_raises=1. Rust port in Inc 4 will be needed for high-bucket precomputation on realistic ranges.

### Increment 3 — Flop solver (Python) ✅ DONE (MVP)
- New modules: `flop_showdown.py`, `flop_cfr.py`, `flop_best_response.py`
- `tree.py`: added `build_flop_tree`; deepest showdown leaves carry (turn_card, river_card) tuple keys
- Two-level chance handling reuses the chance-node primitive built in Inc 2
- Tests: 4 in `tests/nlhe/test_flop_solver.py` (tree structure, showdown sizing, sanity, run-out keys)
- CLI: `nlhe_flop_solve` game type
- 129 tests total passing

**Known gap deferred to Inc 4**: no public-card abstraction yet (turn × river enumeration is full 49×48). Python performance scales linearly with combo² × 2352 run-outs; tractable only for tiny ranges. Real-size flop solves will run in Rust with abstraction.

### Increment 4 — Rust port + preflop + offline precomputation ✅ DONE

All pure-function, tree, solver, and tooling modules shipped. SIMDified hot paths (terminal loop, reach scaling, regret update) with `wide` crate. Bucketed CFR+ with K=16 EHS quantiles gives ~100-6000× speedup over unbucketed on realistic ranges.

**Offline precompute executed**: 7 action-line variants × 2 stack depths × 1755 strategically-distinct flops = 24,570 files, ~5GB gzipped in `precompute_out/`:
- Core HU: `limped`, `sr_called`, `3bet_called`, `4bet_called`
- 6-max role-swap variants (caller-in-position approximations): `sr_called_ip_caller`, `3bet_called_ip3bet`, `4bet_called_ip_caller`

Each file carries: per-node strategies (indexed by K=16 buckets), `bucket_hands` + `villain_bucket_hands` (representative hands per bucket), `bucket_ehs_range` + `villain_bucket_ehs_range` (EHS bounds per bucket), `oop_bucket_of_combo` + `ip_bucket_of_combo` (1326-long arrays for filtering along action paths).

### Increment 5 — `/poker` Claude Code skill ✅ DONE

**Skills installed at both `~/.claude/skills/poker/` and `~/.hermes/skills/gaming/poker-gto/`**, triggered by natural-language hand descriptions.

**CLI tools** (`skill/poker/bin/`):
- `find_flop.py` — locate nearest precomputed flop file by rank+texture notation
- `paths.py` — list valid action_paths for a spot
- `query.py` — flop/turn strategy with EHS-based bucket fallback for out-of-range hands
- `solve_river.py` — on-demand river bucketed CFR+ via Rust `solve_river_subgame` binary; supports IP hero with per-OOP-action response breakdown (`ip_responses` field) and `--vs-action` for specific response queries
- All tools output `{context, table, ...}` JSON with `hint` field on errors

**Out-of-range approximation** (added after first hermes-agent test revealed HU ranges don't always match 6-max): when hero's hand isn't in the precomputed range, query.py computes its EHS on the precompute's rep5 board and maps to the nearest bucket by EHS. River solver injects the combos at weight 1.0 and lets the Rust binary re-bucket on the actual 5-card board. Output marked *"approximated — not in precomputed range"*.

**Still deferred**:
- True per-position 6-max ranges (current variants use HU ranges as approximation). Would require new range fixtures and rerunning 6 precompute runs.
- Opponent profile layer at solver level (adjusting ranges for non-GTO opponents). Currently scaffolded for LLM-context use only — profiles inform 理由 column but don't modify solver output.
- 100BB stack depth (explicit user decision — doesn't play at this depth).

## Deferred

- **Multiway (3+ handed)**: future hybrid — HU precomputed baseline + LLM heuristic adjustment for side-pot / coverage dynamics
- **Tournament / ICM adjustments**: not in scope for cash-game use case
- **Exotic formats (PLO, short deck)**: not in scope

## Why this shape

User plays low-stakes cash with friends. They're willing to wait on
offline precomputation but want fast, precise answers when analyzing hands
for learning. That pattern inverts the usual latency/accuracy tradeoff:
spend as much precompute time as needed; make queries effectively free.

Rust is re-added (originally deferred) because preflop + flop precompute
with fine abstraction is genuinely Python-infeasible on the target
hardware (M4, 16GB). Python remains the reference / testing implementation
in Inc 2-3 so correctness can be verified before the port.
