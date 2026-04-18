# poker-gto-engine

自托管德州扑克 GTO 求解器工程。

当前状态：
- 仓库已初始化在 `~/projects/poker-gto-engine`
- Rust toolchain 与 CMake 已安装
- Phase 0 已跑通两个 correctness harness：Python Kuhn CFR 与 Rust Kuhn CFR

目录：
- `python/`：参考实现与算法验证
- `tests/`：Python 正确性测试与 Rust integration tests
- `src/`：Rust solver core
- `docs/`：架构与基准文档

当前可运行命令：
- `python3 -m unittest tests.test_kuhn_cfr -v`
- `python3 -m unittest discover -s tests -v`
- `cargo test rust_kuhn_cfr_converges_to_known_equilibrium_family -- --nocapture`
- `cargo test`

当前测试保证：
- Player 1 game value 接近 `-1/18`
- 起手策略满足 Kuhn equilibrium family 约束：J 的 bluff 频率在区间内，K 的 value bet 频率约为 `3 * alpha`，Q 起手近似纯 check

下一步：
1. 加 Leduc Poker correctness tests
2. 把 infoset / regret store 抽成独立模块
3. 做 Python 与 Rust 对拍输出
4. 进入 river subgame baseline
