# Temporal Hand Labanotation Plan

## Problem Restatement

现有 Hand Labanotation 主要是逐帧姿态离散化。它适合“记录当前手形/空间状态”，但对以下问题表达力不够：

- 同一帧序列的运动方向和速度信息没有进入表示本体。
- 模型只能学 `pose -> symbol`，学不到 “how it moves”。
- 评测只看单帧离散标签，缺少真正的时序约束。

这正好对应导师提到的“在表示层面引入时序”。

## Proposed Representation: Temporal-HL

把原始 HL 扩展成三元组：

1. `Static token`
   每帧、每个区域向量的空间离散符号，保留原 HL 的静态姿态表达。
2. `Motion token`
   相邻帧区域向量的变化方向，量化为 `26 directional tokens + 1 hold token`。
3. `Keyframe mask`
   基于区域向量整体动能的局部极小值，标记“停顿/转折/边界”。

形式上：

`Temporal-HL_t = {(s_t^i, m_t^i)}_{i=1}^{40}, k_t`

其中 40 表示双手 20 个区域向量。

## Why This Is A Publishable Angle

- 不是单纯“给模型加 Transformer”，而是先改变 notation 本身。
- 允许研究两个问题：
  - `motion-to-notation`: 从连续运动到静态+动态符号；
  - `notation-to-motion`: 从时序手谱恢复运动骨架。
- 兼容机器人/控制场景，因为 motion token 比纯姿态 token 更接近控制信号。

## Prototype Scope In This Repo

当前代码已经提供：

- `temporal_hl/notation.py`
  - 区域向量提取
  - 静态符号量化
  - 动态符号量化
  - 关键帧检测
- `temporal_hl/preprocess.py`
  - 从 `annotations.jsonl` 生成 `Temporal-HL` 标签
- `temporal_hl/model.py`
  - 多任务时序基线
- `train_temporal_hl.py`
  - 静态符号、动态符号、关键帧联合训练入口

## Immediate Next Experiments

1. 先跑标签生成，检查静态/动态 token 分布是否均衡。
2. 训练 baseline，拿到：
   - static accuracy
   - motion accuracy
   - keyframe F1
3. 做两个关键 ablation：
   - 只预测 static
   - static + motion
4. 如果 motion 分支显著提升 static 预测或 reconstruction，就足以形成论文主线。

## Likely Paper Framing

标题方向可以是：

- `Temporal Hand Labanotation: A Motion-Aware Symbolic Representation for Hand Movement Documentation`
- `Beyond Framewise Hand Notation: Motion Tokens for Temporal Hand Labanotation`

核心贡献可以写成：

1. 第一个显式编码手部时序变化的 hand notation 表示。
2. 一个从 3D joints 自动生成 Temporal-HL 标签的弱监督管线。
3. 一个验证“时序符号优于纯静态符号”的多任务基线和评测协议。
