# Leduc reference target

当前采用的第一版 correctness 参考值：

- Player 0 expected value: `-0.08553`
- 来源：<https://cs.stackexchange.com/questions/169593/nash-equilibrium-details-for-leduc-holdem>
- 背景：回答者声称基于 CFR 评估了 500,000,000 games，并给出了完整策略摘录。

这个值当前只作为外部 benchmark target。
后续还要补两件事：
1. 交叉核对 `brianberns/CFR-Explained` 或其他公开实现
2. 用我们自己的 Leduc solver 跑到稳定后，对比是否收敛到同一区间
