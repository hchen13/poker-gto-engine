# poker-gto-engine

自托管德州扑克 GTO 求解器工程。

当前状态：
- 仓库已初始化在 `~/projects/poker-gto-engine`
- Rust toolchain 与 CMake 已安装
- Phase 0 已跑通 Python Kuhn CFR、Rust Kuhn CFR、Python/Rust 对拍
- e2e CLI 现在支持 root spot 和 facing-bet spot，并且动作标签已经修正为 check/bet 或 fold/call
- Python 侧已经抽出了可复用的 `cfr_core.InfoSet`
- Leduc 规则层已经落地：起始状态、合法动作、轮次推进、公共牌发放、终局 payoff 都有测试

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
- `./scripts/e2e_kuhn_demo.sh`

当前测试保证：
- Kuhn Player 1 game value 接近 `-1/18`
- 起手策略满足 Kuhn equilibrium family 约束
- Rust 与 Python 实现对拍一致
- 通用 CLI 能读取 JSON 输入并输出稳定分析结果
- facing-bet 节点输出真实语义动作标签：fold / call
- Leduc 规则状态机和终局 payoff 当前已有 9 个单测覆盖

最短 e2e：
```bash
cd ~/projects/poker-gto-engine
./scripts/e2e_kuhn_demo.sh
```

下一步：
1. 把 `leduc_rules` 接到 CFR 主循环
2. 先做小迭代 smoke test，确认数值方向正常
3. 再把 Leduc 接到 `analyze_spot` CLI
4. 稳住后继续往更真实的 postflop 子博弈推进
