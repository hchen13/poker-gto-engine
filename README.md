# poker-gto-engine

自托管德州扑克 GTO 求解器工程。

当前状态：
- 仓库已初始化在 `~/projects/poker-gto-engine`
- Rust toolchain 与 CMake 已安装
- Phase 0 已跑通 Python Kuhn CFR、Rust Kuhn CFR、Python/Rust 对拍
- e2e CLI 现在支持 Kuhn 的 root spot / facing-bet spot
- Leduc 规则层已经落地，最小 CFR 主循环也已经接上，并能通过 CLI 跑 root spot smoke
- Python 侧已经抽出了可复用的 `cfr_core.InfoSet`

目录：
- `python/`：参考实现、分析 CLI、算法验证
- `tests/`：Python 正确性测试与 Rust integration tests
- `src/`：Rust solver core
- `docs/`：架构与基准文档
- `fixtures/`：样例输入与参考值
- `scripts/`：e2e 演示脚本

当前可运行命令：
- `python3 -m unittest discover -s tests -v`
- `cargo test`
- `python3 -m python.analyze_kuhn --hero-card K --history '' --iterations 20000 --format text`
- `python3 -m python.analyze_kuhn --hero-card Q --history b --iterations 20000 --format json`
- `python3 -m python.analyze_spot --input-file fixtures/kuhn/root_k.json --format json`
- `python3 -m python.analyze_spot --input-file fixtures/leduc/root_k.json --format json`
- `./scripts/e2e_kuhn_demo.sh`
- `./scripts/e2e_leduc_demo.sh`

当前测试保证：
- Kuhn Player 1 game value 接近 `-1/18`
- 起手策略满足 Kuhn equilibrium family 约束
- Rust 与 Python 实现对拍一致
- 通用 CLI 能读取 JSON 输入并输出稳定分析结果
- facing-bet 节点输出真实语义动作标签：fold / call
- Leduc 规则状态机和终局 payoff 当前已有 9 个单测覆盖
- Leduc root spot 已能稳定返回有限数值、归一化策略和 infoset 数量

最短 e2e：
```bash
cd ~/projects/poker-gto-engine
./scripts/e2e_kuhn_demo.sh
./scripts/e2e_leduc_demo.sh
```

下一步：
1. 让 Leduc 不只支持 root spot，而是支持更多 infoset 查询
2. 对齐 `fixtures/leduc/reference.json`，把 smoke 提升成 correctness benchmark
3. 再决定是先做 Rust 版 Leduc，还是直接往更真实的 postflop 子博弈推进
