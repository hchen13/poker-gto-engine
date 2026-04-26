# Increment 1 — Progress Log

Scope: river HU subgame GTO solver MVP. See `architecture.md` for full plan.

## Completed

### Restructure + module skeleton
- New `python/nlhe/` package created
- `python/analyze_nlhe_river.py` replaced by `python/nlhe/equity.py` (function renamed `river_equity_vs_range`)
- All existing tests and CLI (`analyze_spot.py`, shell scripts) updated to new import paths
- Kuhn / Leduc code preserved as CFR regression baselines

### M1 — Hand evaluator (Python reference)
Files: `python/nlhe/cards.py`, `python/nlhe/hand_eval.py`
Tests: `tests/nlhe/test_hand_eval.py` (19 tests)
- 5-card and 7-card evaluator (`evaluate_seven`, `evaluate_any`)
- All 10 hand categories, wheel + steel wheel, tie-breaking kickers
- Card <-> index mappings, 1326-combo index (`combo_index`, `INDEX_TO_COMBO`)

### M2 — Range parser
File: `python/nlhe/range_parser.py`
Tests: `tests/nlhe/test_range_parser.py` (30 tests)
- Supported syntax: `AA`, `AKs`, `AKo`, `AK`, `TT+`, `A2s+`, `55-77`, `T9s-76s`, `AhKs` (specific combo), `AA:0.5` (weight)
- Returns 1326-dim weight vector

### M3 — River game tree builder
File: `python/nlhe/tree.py`
Tests: `tests/nlhe/test_tree.py` (14 tests)
- Action / Node dataclasses; terminal with `terminal_winner` (0/1 for fold, None for showdown) and `terminal_pot`
- Bet abstraction: `[0.33, 0.50, 0.67, 1.00, 1.50] * pot` + all-in (duplicates and oversize bets filtered)
- Raise: pot-sized raise + all-in
- `max_raises` cap for pathological loops
- **Chip conservation invariant verified in tests**: `terminal_pot + stacks[0] + stacks[1] == initial_total` at every leaf

### M4 — CFR+ solver on river subgame ← keystone, most complex
Files: `python/nlhe/showdown.py`, `python/nlhe/cfr.py`
Tests: `tests/nlhe/test_cfr.py` (6 tests)
- **Card-aware infosets**: regrets/strategy per `(node_id, local_combo_idx, action_idx)`
- CFR+ with regret clipping, linear averaging
- Precomputed showdown matrix with card-conflict sentinel
- Two-pass per iteration (one pass per updating player)
- **Verified behavior on known cases**:
  - Nut vs. air: hero EV ≈ initial pot, villain folds
  - Hero always loses: hero EV ≈ 0 (just check down)
  - Identical strength (tie): hero EV ≈ pot/2
  - Mixed range: solver polarizes — top set bets big, pure bluff shoves all-in (classic GTO)
- **Bug caught and fixed**: regret update was double-counting opponent reach factor. Found by comparing with known-correct Kuhn CFR. Keeping Kuhn/Leduc as regression baselines paid off.

### M7 — Exploitability-based solver validation
Files: `python/nlhe/best_response.py`, extended `python/nlhe/cfr.py` (bug fix + `build_and_train`), `python/nlhe/analyze.py`
Tests: `tests/nlhe/test_best_response.py` (4 tests)
Fixture: `fixtures/nlhe-river/solve_with_exploitability.json`
- Computes best-response (BR) value for each side vs the solver's average strategy
- Exploitability = BR_hero + BR_villain − initial_pot; at Nash it equals 0 (constant-sum property)
- This is the rigorous way to cross-check our solver — no need for PioSolver/GTOWizard output; the math self-validates
- **Bug caught & fixed**: CFR+ was normalizing EV by `sum(hero_w) * sum(villain_w)` which overcounts when there are card-conflict pairs. On AhAc/KsKc/QsJs vs JcJd/TcTd/AcQc (1 conflict out of 9 pairs), this made reported EVs off by 9/8. Fix: precompute `pair_weight_total` as the sum over non-conflict pairs. Caught while verifying BR sum ≥ pot invariant.
- **Result on sample spot**: 1000 iterations → exploitability 0.044 on pot 100 (0.04% of pot). Solver is finding near-exact Nash.
- CLI: set `"compute_exploitability": true` in JSON input to get BR + exploitability in output

### M5 — River card abstraction (E[HS] bucketing)
Files: `python/nlhe/abstraction.py`, extended `python/nlhe/cfr.py`, `python/nlhe/analyze.py`
Tests: `tests/nlhe/test_abstraction.py` (6 tests)
Fixture: `fixtures/nlhe-river/solve_bucketed.json`
- `compute_river_ehs(board)`: E[HS] per combo vs uniform random villain (board-conflict filtered)
- `bucket_by_ehs(combos, weights, ehs, n_buckets)`: equal-weight quantile bucketing
- `solve_river(..., hero_buckets, villain_buckets)`: CFR shares regret/strategy per bucket. Card conflicts and terminal payoffs stay per-combo (correctness-preserving).
- Backward compatible: without buckets, each combo is its own bucket → identical output (regression test verifies exact equivalence)
- CLI exposes `n_buckets` in JSON input
- **Known tradeoff**: coarse bucketing can merge semantically distinct hands (e.g. n_buckets=4 on AA/KK/AKs/AKo/AQs/QJs lumps some AQs combos with QJs). Higher bucket counts separate better.
- **Note on performance**: Python abstraction reduces infoset count and memory but per-iter cost is still dominated by terminal `outcome[i][j]` traversal. Real speedup comes in M8 (Rust port).

### M6 — End-to-end CLI
Files: `python/nlhe/analyze.py`, extended `python/analyze_spot.py`
Tests: `tests/nlhe/test_analyze.py` (5 tests)
Fixture: `fixtures/nlhe-river/solve_polarized_hero.json`
- New JSON game type `nlhe_river_solve` accepted by `analyze_spot.py`
- Input: board, pot, stacks, hero_range/villain_range (strings), first_to_act, hero_hand (optional), iterations, max_raises
- Output: root strategy per combo, hero EV, recommendation for a specific hero hand
- Text and JSON output formats supported
- IP hero (first_to_act=1) case: emits a note rather than a bogus recommendation
- End-to-end smoke verified: AhAc / KsKc / QsJs vs. JcJd / TcTd / AcQc produces polarized strategies

### Test totals
- 116 tests pass
- 0 failures / 0 errors

## Pending

| # | Milestone | Estimate | What it delivers |
|---|---|---|---|
| M8 | Port hot paths to Rust | days | 10×+ speedup on equity calc + CFR inner loop |

## Decision points deferred

- Exact bet/raise abstraction for Increment 2+ (turn + multi-street) — current 7-bet / 3-raise table is Increment-1 choice.
- Whether to keep Python CFR permanently or make Rust CFR the single source of truth after M8.
