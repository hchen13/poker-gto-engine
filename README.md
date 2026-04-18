# poker-gto-engine

自托管德州扑克 GTO 求解器工程。

当前状态：
- 仓库已初始化在 `~/projects/poker-gto-engine`
- Rust toolchain 与 CMake 已安装
- Phase 0 已跑通 Python Kuhn CFR、Rust Kuhn CFR、Python/Rust 对拍
- 已有一个稳定可跑的 e2e 入口：输入 JSON 局面，输出动作频率与推荐动作

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
- `python3 -m python.analyze_spot --input-file fixtures/kuhn/root_k.json --format json`
- `./scripts/e2e_kuhn_demo.sh`

当前测试保证：
- Kuhn Player 1 game value 接近 `-1/18`
- 起手策略满足 Kuhn equilibrium family 约束
- Rust 与 Python 实现对拍一致
- 通用 CLI 能读取 JSON 输入并输出稳定分析结果

最短 e2e：
```bash
cd ~/projects/poker-gto-engine
./scripts/e2e_kuhn_demo.sh
```

下一步：
1. 把 Kuhn 的 infoset / regret store 抽成可复用模块
2. 以同一 CLI 契约接入 Leduc
3. 再往 river subgame 扩
4. 最终替换成真正的 NLHE postflop backend
