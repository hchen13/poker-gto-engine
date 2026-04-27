# poker-gto-engine

[English](README.md) | **中文**

[![CI](https://github.com/hchen13/poker-gto-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/hchen13/poker-gto-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**真 GTO solver，封装成 LLM agent 技能。**

跟你的 AI 朋友自然语言说一手牌——「我在 BB，3bet 底池，A♥K♠ on K62 rainbow，对手 check，怎么打？」——它不会瞎编频率，而是查一份预计算 CFR+ 表（或者 on-demand 跑一个 subgame solver），把 GTO 混合策略 + 用真实数字作锚点的理由列给你。

```
你：    6-max 200BB 现金桌。BTN 开池，我从 BB 3bet，BTN 跟注。Flop K♦ 6♥ 2♣，
        我手 A♥K♣（顶对顶踢），位置 OOP，怎么打？

Agent 调：  poker query 3bet_called 200 K62r AhKd oop ""

**Flop:** 2s6hKd  |  **Line:** 3bet_called/200bb  |  **Hero:** AKo (OOP/BB)
**Pot:** 36.0 BB  |  Bucket 9 (AA, AKo, KQs, AKs)  |  EHS 0.82–0.83

| 操作                   | 频率  | 理由                                                      |
|-----------------------|-------|-----------------------------------------------------------|
| bet 33% pot (11.9 BB) | 56%   | 标准小注 c-bet——拒绝对手听牌的免费实现                     |
| bet 50% pot (18.0 BB) | 41%   | 大尺寸混合，深筹下 OOP 给 turn/river 建池                  |
| check                 | 3%    | 偶发——用强牌保护 checking range                          |
| bet 67% pot (24.1 BB) | 1%    | 低频大注线                                                |
```

频率来自 solver。理由列是 LLM 基于真实数字写出来的解读——不是凭空捏造的。

## 一条命令背后是什么

- **24,570 张预计算 CFR+ 表**——14 种 preflop 变体 × 200/500 BB 深度 × 1755 张 canonical iso-class flop。压缩后 ~1.4 GB（msgpack + u8 量化 + zstd，比 gzip 小 ~5×，且决策层面无损）
- **on-demand bucketed subgame solver** 兜底任何预计算没覆盖的 spot——river ~100 ms，turn ~500 ms，flop ~36 s
- **一条 `poker` CLI**，8 个 subcommand，专门给 LLM agent 通过 shell 调用。附带 `SKILL.md`，Claude Code / Hermes 等 agent 平台可以直接注册

**Postflop 建模是 heads-up**——这正好对应大部分现金桌的真实节奏：preflop 多人入池，到 flop 时弃到只剩 2 人。真 HU 桌也覆盖。K=16 EHS 分位数 bucketing。给笔记本级硬件做的实战工程妥协，不是 PioSolver 替代品——但**输出的是真数字，不是幻觉**。

## 安装

```bash
git clone https://github.com/hchen13/poker-gto-engine
cd poker-gto-engine
./install.sh
```

安装脚本会编译 Rust solver（首次 5–15 分钟），把 `poker` CLI 装到 `~/.local/bin/`，跑一遍 smoke test，最后报告预计算数据状态。详情见 [`AGENTS.md`](AGENTS.md)（给 LLM agent 操作员看的版本）。

验证：

```bash
poker query 3bet_called 200 K62r KQo ip check
```

如果输出包含 `"strategy"` 数组的 JSON，就装好了。

## Subcommand 列表

| 命令                         | 用途                                                        | 延迟      |
|------------------------------|-----------------------------------------------------------|-----------|
| `poker query`                | 查 flop/turn 的 GTO 策略（预计算表）                          | <100 ms   |
| `poker paths`                | 列出某 spot 所有合法的 action 路径（注码、节点）                 | <100 ms   |
| `poker find-flop`            | 找到某 flop 对应的 canonical iso-class 文件                  | <50 ms    |
| `poker solve-river`          | on-demand 解 river 决策（拉预计算的 turn 根作 villain range）  | ~100 ms   |
| `poker solve-river-manual`   | 手动给 range 解 river（不查预计算）                           | ~70 ms    |
| `poker solve-turn-manual`    | 手动给 range 解 turn subgame                                 | ~500 ms   |
| `poker solve-flop-manual`    | 手动给 range 解 flop subgame（在 2 分钟交互预算内）            | ~30–60 s  |
| `poker profile`              | 查对手画像（内置 `fish`/`nit`/`reg`/`maniac` 或用户自定义）     | <50 ms    |

`poker --help` 列全部；`poker <subcommand> --help` 看具体参数。

## 工作原理

**Solver 内核**（Rust）：CFR+ + K=16 EHS 分位数 bucketing。flop/turn 用公共牌 chance 子采样让计算可控。Postflop 树是 heads-up——只要 flop 时已经收口到 2 人，preflop 几个人入池都行。同一个 flat-tree CFR+ kernel 跑预计算和 on-demand subgame——`solve_river_subgame`、`solve_turn_subgame`、`solve_flop_subgame` 都在 `src/bin/`。

**Canonical flop 枚举**：`abstraction::enumerate_canonical_flops()` 返回正好 1755 个代表元——每个 suit-isomorphism 等价类一个——覆盖全部 22,100 张真实 flop。查找用 6 字节排列不变 key，任何具体 flop 都能 O(1) 映射到它的 canonical 文件。

**压缩**：预计算策略量化到 u8（1/255 精度），用 msgpack + zstd 存。所有 bucket 的 argmax 保持一致，单格最大误差 <0.005（远低于 CFR+ 收敛误差）。结果：比 gzip JSON 小 ~5×，整个库 1.4 GB。

**Skill 层**（Python）：`poker` CLI 调度到预计算查找或 Rust subgame solver。当查到的 bucket 内手牌结构异质时（比如 AKs 同花听 + A7o 在双花板上），结果会带 `dispersion_warning`，提示 LLM 当前展示的是 bucket 平均频率，并建议跑 `solve-*-manual` 拿手牌专项解。

更深的内部细节见 [`docs/architecture.md`](docs/architecture.md)；agent 看的 skill manifest 见 [`SKILL.md`](skill/poker/SKILL.md)。

## 覆盖范围

**仅 heads-up postflop**（大部分现金桌的 flop 都是收口到 2 人后开打的）。200 BB 和 500 BB 两种深度。7 种 preflop 变体——每种都在两种深度下预计算：

| 变体                       | 故事                                | postflop 攻击方位置        |
|----------------------------|-------------------------------------|---------------------------|
| `limped`                   | SB limp，BB check                   | —                         |
| `sr_called`                | SB 开池，BB 跟注                     | SB = IP（真 HU）           |
| `sr_called_ip_caller`      | EP 开池，IP 跟注（6-max）            | opener = OOP              |
| `3bet_called`              | BB 3bet，SB 跟注                     | 3-bettor = OOP            |
| `3bet_called_ip3bet`       | IP 3bet，EP 跟注（6-max）            | 3-bettor = IP             |
| `4bet_called`              | SB 4bet，BB 跟注                     | 4-bettor = IP             |
| `4bet_called_ip_caller`    | EP 4bet，IP 跟注（6-max）            | 4-bettor = OOP            |

每个预计算文件存 flop 和 turn 节点。River 决策是 on-demand 跑出来的（从 turn 根拉对手 range）。如果是 6-max/9-max 桌 preflop 多人入池但 postflop 收缩到 HU，挑那个 IP/OOP 布局匹配你实际场景的变体——决策表见 [`SKILL.md`](skill/poker/SKILL.md)。

## 妥协与诚实

- **K=16 EHS bucketing**。Bucket 层面的 GTO 在期望上正确，但会把 EHS 接近、结构不同的手压扁（比如 AKs 同花听 + A7o 在双花板）。`dispersion_warning` 字段会提示，建议改用 `solve-*-manual` 算手牌专项策略。
- **仅 heads-up postflop**。Solver 假设 flop 时已经只剩 2 人。多人池一直打到 turn/river 不建模——但实际现金桌最常见的「preflop 多人，flop 弃到只剩 2 人」**就是**主用例。挑 IP/OOP 布局匹配你实际场景的变体即可。
- **两种深度**。200 BB 和 500 BB 预计算了；其他深度用 on-demand manual solver。
- **6-max+ 格式下的 range / 底池近似**。默认 range 是 HU 校准的，6-max 实际更紧、多人池底池更大。on-demand solver 让你显式给 range 和底池，是非 HU 场景的精确路径。
- **不是 PioSolver 替代品**。这是给笔记本级硬件（8 GB RAM 能跑、16 GB 舒服）调的 bucketed 近似。要锦标赛级 per-combo 精度，请用商业工具。

## LLM agent 开发者

[`AGENTS.md`](AGENTS.md) 是给 clone 此 repo 的 LLM agent 操作员的结构化指南：安装、验证、`poker` CLI 契约、常见失败模式、Claude Code / Hermes 的 skill 注册。Skill manifest 本身（[`skill/poker/SKILL.md`](skill/poker/SKILL.md)）是 agent 运行时读的——它写明了怎么解析手牌描述、挑 `action_line` 变体、跑 query、用理由列展示结果。

要把这个挂到别的 agent 平台上，契约就是 `poker` CLI——JSON 入 JSON 出。不需要平台特定的 hook。

## 测试

```bash
python3 -m pytest tests/ -q     # ~144 个 Python 测试（skill CLI、iso-key 不变量、solver 桥接）
cargo test --lib --release       # ~50 个 Rust 测试（CFR+ 正确性、abstraction、bucketing）
```

## 准备预计算数据

`precompute_out/` 目录（~1.4 GB）不入 git——以 release tarball 形式发布。clone + install 后，三选一：

```bash
# 直接下预编译表（推荐），不需要 gh：
curl -L https://github.com/hchen13/poker-gto-engine/releases/download/v0.1/precompute_out_v2.tar.zst | zstd -d | tar -x

# 或者用 gh CLI：
gh release download v0.1 --pattern 'precompute_out_v2.tar.zst'
zstd -d precompute_out_v2.tar.zst -c | tar -x

# 本地生成（8 核 ~14 小时）：
./target/release/precompute_bucketed_parallel fixtures/precompute/v2/full_3bet_called_200bb.json
# （对 fixtures/precompute/v2/ 下 14 个 spec 各跑一次）

# 或者放别的位置：
export POKER_PRECOMPUTE_DIR=/path/to/precompute_out
```

没有预计算数据时，`poker query` 和 `poker paths` 会返回错误，但 `poker solve-*-manual` 仍然完全可用。

## License

MIT。见 [LICENSE](LICENSE)。

## 致谢

CFR+ 算法：Tammelin 等人，*Solving Heads-Up Limit Hold'em Poker*（2015）。手牌评估器：标准 7 张牌哈希排名。Bucketing 妥协和 msgpack/zstd 存储格式都是实战工程选择，不是新研究。
