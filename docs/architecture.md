# Architecture

## 1. 项目目标

构建一个自托管的 NLHE GTO solver，作为 Claude Code skill 的后端，用来分析实际手牌（通常是 turn / river 面临 all-in 的决策），输出 GTO 动作 + 频率 + EV + 文字解释。

**MVP 范围**：
- 规则：No-Limit Texas Hold'em
- 人数：**Heads-up only**（多人用 HU 解 + LLM 启发式调整的混合方案，在 skill 层处理，不在 solver 层）
- 有效筹码：**200BB 和 500BB** 两套独立预计算
- 下注尺寸：
  - Postflop bet: `check / 33% / 50% / 67% / 100% / 150% / all-in`
  - Postflop raise: 相对前注的倍数（`2x / 3x / all-in`，具体最终值调整时确定）
  - Preflop: `3x / 4x / 6x / 8x / limp / all-in`
  - 支持 donk bet

**非目标（至少 Phase 1 内）**：
- Multiway GTO 求解
- 实时对战机器人
- GUI
- 支持 rake 建模（Phase 3+ 考虑）
- Tournament ICM

## 2. 语言与性能策略

Python 作为参考实现和 CLI orchestration 层足够用，但 solver 热路径必须用 Rust（现有项目已经同时有 Python / Rust 双实现，基础设施已在）。

| 层 | 语言 | 理由 |
|---|---|---|
| Solver 热循环（CFR inner loop） | Rust | 无 GC、内联优化、`unsafe` 局部使用做 SIMD |
| Hand evaluator（5/7 张牌排名） | Rust，SIMD 优化；可选集成 C++ OMPEval FFI | 预计算 bucket equity 会调用千万次 |
| Card / Bet abstraction（离线 pipeline） | Rust + Python 辅助脚本 | 数据处理用 Python 更灵活，矩阵计算用 Rust |
| GPU equity precompute（可选） | Metal compute shader（M4 GPU） | 若 CPU 路径不够快再上，Phase 2+ 评估 |
| CLI / JSON IO / skill 接入 | Python | 开发效率高，非热路径 |
| Preflop / Flop 策略库 | Rust 生成 + 磁盘存储（memmap） | 存完后 Python 和 Rust 都能查 |

**原则**：只要某个操作在 solve 过程中被调用 >10 万次，必须 Rust 化。低频的 IO / CLI / 调度留在 Python。

## 3. 总体架构

```
┌──────────────────────────────────────────────────────────────────┐
│  离线预计算（运行一次，产物存盘，跨会话复用）                         │
│                                                                    │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐ │
│  │ Equity matrix   │→  │ Card abstraction │→  │ Preflop solver  │ │
│  │ (169×169 等)   │   │ (bucket mapping) │   │ → preflop DB    │ │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘ │
│                                │                      │           │
│                                ↓                      ↓           │
│                        ┌─────────────────────────────────────┐   │
│                        │  Flop solver (1755 同构类 × 深度)    │   │
│                        │  → flop DB                          │   │
│                        └─────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ↓ （查询时作为输入 range）
┌──────────────────────────────────────────────────────────────────┐
│  在线查询（用户提交手牌 spot → 秒/分钟级返回）                       │
│                                                                    │
│   JSON 输入                                                         │
│      ↓                                                             │
│   Spot parser（位置 / 筹码 / 动作历史 / 牌）                          │
│      ↓                                                             │
│   Range reconstructor（从 preflop/flop DB 查出到达当前街时的双方 range）│
│      ↓                                                             │
│   Turn/River subgame builder（构建具体子博弈树）                     │
│      ↓                                                             │
│   CFR+ solver（现场 solve，目标 30s–5min 内收敛到可用精度）           │
│      ↓                                                             │
│   Output decoder（抽象策略 → 人类可读频率 + EV + 解释）               │
│      ↓                                                             │
│   JSON 输出（供 skill 消费）                                         │
└──────────────────────────────────────────────────────────────────┘
```

**为什么要这样分**：
- Preflop + flop 树太大，不可能在用户等待时现场 solve
- 但 preflop / flop 策略一旦 solve 完就是静态的，可以反复用
- Turn / river 依赖具体 board 和剩余 stack，必须现场 solve（但树已经很小）

## 4. 核心模块

### 4.1 Range 表示
- 基础单位：**combo**（具体两张牌，如 `AhKs`），共 1326 个
- 显示单位：**hand class**（如 `AKs` / `AKo` / `AA`），共 169 个
- Range = `[1326]` 概率向量，每个 combo 一个权重
- 可从文字（`"TT+, AK, KQs"`）解析为 combo 向量

### 4.2 Card abstraction（手牌抽象 / bucketing）
作用：把 1326 个 combo 根据某条 street 的 board 映射到 N 个桶，让 CFR 可解。

- **River**：用 `E[HS]`（对对手 range 的期望赢率）等距切桶，200–500 桶
- **Turn**：用 `E[HS²]`（加入方差捕捉听牌价值），200–500 桶
- **Flop**：potential-aware（OCHS 风格，多维赢率向量 + k-means），100–200 桶
- 产物：`bucket_map[street][board_hash][combo_idx] -> bucket_id`

桶数是精度 vs. 内存/时间的 trade-off。Phase 1 从低桶数开始（200），确认端到端跑通后再往上加。

### 4.3 Bet abstraction（下注抽象）
把连续的下注金额离散化为固定的动作集合。

- 当前节点 pot 和 stack 已知 → 每个离散动作展开为具体金额
- Postflop bet：`[check, bet33, bet50, bet67, bet100, bet150, allin]`（某些动作在某些 SPR 下不存在，需过滤）
- Postflop raise：`[fold, call, raise2x, raise3x, allin]`
- Preflop: 按上面定义的 preflop 表
- Donk bet：非 preflop aggressor 在新街第一个行动时也获得 bet 动作集合（而不是只有 check）

### 4.4 Game tree builder
输入：起始 pot、起始 stacks、起始 player to act、当前 street、board、双方 range
输出：博弈树（节点 = 决策点；叶子 = 终局或下一 street 进入点）

节点类型：
- `ActionNode`：某方行动，有 N 个子节点（按 bet abstraction 给出）
- `ChanceNode`：发牌节点（turn / river）
- `TerminalNode`：fold 结束 / showdown 结束
- `StreetTransitionNode`：进入下一街的锚点（查上一层策略库用）

### 4.5 CFR solver
Phase 1：**CFR+**（简单易实现，比 vanilla 快 ~10x）
- 正后悔匹配
- Linear CFR weighting
- Best response / exploitability 作为收敛判据

Phase 2+：**External Sampling MCCFR** 或 **Discounted CFR**，大幅加速大树

### 4.6 Hand evaluator
7 选 5 最强手判定。复用现有 `python/analyze_nlhe_river.py` 里的纯 Python 版本作为正确性基线，新写 Rust 版本用 SIMD 加速。长期可以接 OMPEval（C++ 的 world-class 实现）通过 FFI 调用。

### 4.7 Preflop / Flop solver & storage
- 策略库格式：`memmap` 的 `[infoset_id] -> [action_probs]` 扁平数组
- Infoset ID = hash(position, stack_depth, history, bucket)
- 查询 API：`lookup(position, stack_depth, history, combo) -> action_distribution`

### 4.8 Spot parser / output decoder
- Parser：接收 JSON（位置 / 筹码 / 动作序列 / 牌）→ 标准化内部表示
- Decoder：CFR 输出的抽象节点策略 → 按 combo 展开 → 聚合到 hand class → 加上 EV 和简短文字解释

## 5. Phase 1 MVP 详细 Scope

**Goal**：给定一个写死的起始 range（不依赖 preflop/flop DB），在**河牌** HU 子博弈上跑通全链路，输出 GTO 策略。

**Why 先只做河牌**：
- 最小可运行闭环，最快验证 CFR+ / equity / abstraction / game tree 都没 bug
- 河牌 tree 最小，收敛最快，容易对照外部 solver 验证结果
- 做完这一步就已经有个能用的 skill（你可以输入任意河牌 spot 得到 GTO 建议，只需手工指定双方 range）

**交付物**：

1. `nlhe::river::solve(spot) -> strategy` Rust API
2. `analyze_nlhe` CLI：读 JSON 输入（双方 range + board + pot + stacks + action history），输出 JSON（每个 combo 在当前决策点的动作分布 + 期望 EV）
3. 对一个已知参考值的测试 case（跑一个简单河牌 2bet pot，和公开 solver 输出对比）
4. 性能 baseline：200 桶河牌子博弈在 M4 上 < 30 秒收敛到 < 1% pot exploitability

**Phase 1 里 Explicit NOT doing**：
- Preflop / flop / turn solver
- Bucket map 预计算（河牌阶段桶很少，现场算都行）
- Multi-depth（Phase 1 只支持 100BB effective，调参时暂时用，后面换成 200/500BB）
- 中文解释生成（输出里先只有数字，解释让 skill 的 LLM 层做）

**Phase 1 里程碑（可验收）**：

| Milestone | 内容 | 验收标准 |
|---|---|---|
| M1 | Hand evaluator Rust 实现 | 100万次 random 7-card eval 在 M4 单核 < 3 秒；和 Python reference 结果 100% 一致 |
| M2 | Range 解析器 | `"TT+, AKs, KQo"` 等文字 → combo 权重向量；覆盖所有标记语法 |
| M3 | River game tree builder | 给定 spot，能展开完整博弈树；节点数、动作集合正确 |
| M4 | River CFR+ solver | 已知的 pot-limit 简化 case（写死 range）收敛到合理策略，和手算理论值对上 |
| M5 | Card abstraction（river 版） | E[HS] 切桶；同桶内 combo 策略差异验证在合理范围 |
| M6 | End-to-end CLI | 输入 JSON → 输出 JSON；跑一个"河牌面 all-in"的例子，人工检查合理性 |
| M7 | 参考 solver 对拍 | 和 PioSolver / GTO+ 的公开输出对比（选一两个简单 spot），误差 < 5% |

每个 M 完成都有对应的 test，全部 pass 才进入下一步。

## 6. 后续阶段预览

| Phase | 内容 | 预估 |
|---|---|---|
| **Phase 1** | River HU subgame MVP | 2–3 周 |
| **Phase 2** | Turn + River 子博弈（加 turn 层 CFR，加 turn abstraction） | 2 周 |
| **Phase 3** | Flop 预计算（1755 同构类 × 常用线） + preflop 默认表（先用开源 range） | 2–3 周 |
| **Phase 4** | 自建 preflop solver（200BB，跑到收敛存 DB） | 2–4 周（含离线跑 solve 的几天到一周） |
| **Phase 5** | 500BB 深筹码补齐；性能优化（MCCFR / Metal GPU） | 持续 |

## 7. 风险与已知未决

| 风险 | 缓解 |
|---|---|
| Preflop 500BB solve 在 16GB M4 上可能 OOM | Phase 4 再确认，必要时做 stack depth interpolation，先只 solve 200BB |
| Card abstraction 设计错误导致 solver 输出垃圾 | 从小桶数起步 + 和公开 solver 对拍 + 人工抽查 |
| 自己写的 hand evaluator 比 OMPEval 慢几十倍 | Phase 2 若性能不够直接集成 OMPEval FFI |
| Multiway 在 skill 层用 LLM 启发式"调整"效果不佳 | 接受它只是近似，作为 MVP 之后的持续优化方向 |
| Phase 1 的"写死 range"不实用 | 这是刻意 scope，目的是最快拿到可运行端到端链路；Phase 2/3 接入 range 库 |

## 8. 现有代码处置

- `python/cfr_core.py`：保留，在 CFR+ 实现里继续用（算法抽象通用）
- `python/kuhn_cfr.py` / `python/leduc_*`：**保留作为回归测试基线**，不删——未来升级到 CFR+ / MCCFR 时验证基础算法未被破坏
- `src/kuhn.rs` / `src/leduc.rs`：同上，保留
- `python/analyze_nlhe_river.py`：**改名重构为 `python/equity_calculator.py`**，只负责 equity 计算（它本身逻辑没错，名字有误导）
- 所有现有测试：保留继续跑
- 新增模块（抽象、博弈树、solver、storage）：新目录 `src/nlhe/` 和 `python/nlhe/`，不和现有 toy games 混

## 9. 目录结构（Phase 1 末期预期）

```
poker-gto-engine/
├── src/
│   ├── lib.rs
│   ├── kuhn.rs                 # 保留
│   ├── leduc.rs                # 保留
│   └── nlhe/                   # 新
│       ├── mod.rs
│       ├── card.rs             # 牌 / 牌面表示
│       ├── hand_eval.rs        # 7选5 evaluator
│       ├── range.rs            # combo / range 表示 & 解析
│       ├── abstraction.rs      # card / bet 抽象
│       ├── tree.rs             # game tree builder
│       ├── cfr.rs              # CFR+ solver
│       └── river.rs            # river-specific entry point
├── python/
│   ├── cfr_core.py             # 保留
│   ├── kuhn_*.py               # 保留
│   ├── leduc_*.py              # 保留
│   ├── equity_calculator.py    # 从 analyze_nlhe_river.py 改名
│   └── nlhe/                   # 新
│       ├── __init__.py
│       ├── cli.py              # analyze_nlhe CLI
│       └── decode.py           # 抽象策略 → 人类可读
├── tests/
│   ├── ... (全部保留)
│   └── nlhe/                   # 新，对应 Phase 1 各 milestone 的测试
├── fixtures/
│   ├── ... (保留)
│   └── nlhe/                   # 新，参考 spot + 对拍预期
├── docs/
│   ├── architecture.md         # 本文件
│   └── leduc-reference.md      # 保留
└── scripts/
    └── ... (保留 + 新增 nlhe demo)
```

---

**本文档会随开发推进持续更新**。每个 Phase 开始前会补充该 Phase 的详细设计；Phase 1 的第一周工作前还会补一份 `docs/phase1-week1-plan.md` 细化到文件级。
