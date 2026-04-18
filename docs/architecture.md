# Architecture

目标：先用 Python 建 correctness harness，验证 CFR / Kuhn / Leduc；随后把稳定的数据结构和 solver 主循环迁到 Rust。

当前目录职责：
- `python/`: reference implementation and correctness experiments
- `tests/`: Phase 0 correctness tests
- `src/`: future Rust solver core
- `docs/`: architecture, benchmarks, paper notes
- `fixtures/`: canonical spots and toy-game expectations

当前里程碑：
1. Kuhn CFR correctness
2. Leduc CFR correctness
3. River subgame baseline
4. Rust core migration
