# 本周计划（2026-08-31 ~ 2026-09-06，W35）

> 本周两个主线任务，目标都是「吃透原理 + 产出结构化交付物」，周五前完成首轮产出并更新到本页面。

---

## 任务一：理解 OPD（On-Policy Distillation，在策略蒸馏）

**目标**：系统理解在策略蒸馏的原理、方法谱系和工程实现，能讲清「为什么需要 on-policy」「什么时候选它」「代价是什么」。

### 理解路径

1. **动机——off-policy 蒸馏的天花板**
   - 传统 KD / SFT-on-teacher-data：学生在教师的数据分布上学习，推理时却面对自己的输出分布
   - 长程任务（推理链、工具调用）中误差累积（exposure bias），学生越偏离教师轨迹，信号越失效
2. **核心思想**
   - 学生模型自己采样（on-policy rollout）
   - 教师模型对学生轨迹做逐 token 打分（dense supervision）——教师 logprob 充当「比 RL 标量奖励密得多」的信号
   - 训练目标：在学生自己的样本分布上向教师靠拢（天然对应 reverse KL / mode-seeking）
   - 训练循环：student sample → teacher 逐 token logprobs → 计算 advantage（如 teacher−student logprob 差）→ GRPO/PPO 式更新
3. **关键文献 / 方法谱系**
   - Thinking Machines: *On-Policy Distillation*（2025，实践向，GRPO 式实现，"蒸馏压缩数据的价值"）
   - GKD（Agarwal et al. 2023）：on-policy KD 的通用框架，混合散度目标
   - MiniLLM（Gu et al. 2023）：reverse KL + 策略梯度（白盒蒸馏开山作）
   - DistiLLM（2024）：Schulman KL 目标，训练更稳
   - 延伸关注：OPD 与 RL 的混合（先蒸馏冷启动再 RL 精修）
4. **需要吃透的技术点**
   - KL 方向选择：forward KL（mode-covering）vs reverse KL（mode-seeking），各自的失败模式
   - 计算开销：每步多一次教师前向；教师是否需要可访问 logprobs（白盒 vs 黑盒）
   - 适用场景：推理能力蒸馏（o1-style → 小模型）、工具调用 / agentic 行为蒸馏、弱到强
   - 与 SFT、RL 的边界：什么任务上 OPD 优于二者
5. **交付物**
   - 精读笔记（中文）
   - 方法对比表：方法 | KL 目标 | on-policy? | 教师白盒? | 典型场景 | 关键结果

### 排期

| 时间 | 内容 |
|------|------|
| 8/31（一） | 读 Thinking Machines 博客 + GKD |
| 9/1（二） | MiniLLM / DistiLLM + 技术细节（KL 选择、采样、训练循环） |
| 9/2（三） | 输出对比表 + 笔记初稿 |

---

## 任务二：学习构建大模型评测集

**目标**：掌握一套可复用的评测集构建方法论，并做出一个能真正跑起来的 mini 评测集 demo。

### 理解路径

1. **好评测集的五条标准**
   - 覆盖度：任务类型、领域、输入长度、语言
   - 难度分层：易/中/难（或连续难度），必须有区分度——全对或全错的题集没有信号
   - 判分明确：每题答案可验证或 rubric 无歧义
   - 无污染：防数据泄漏（cutoff、动态换题、held-out）
   - 可复现：固定版本、固定判分流程、分数可追溯
2. **数据来源**
   - 人工标注（金标准，贵）
   - 合成生成（LLM 出题 + 独立验证 / 过滤）
   - 真实数据改造（日志、工单、领域语料脱敏改写）
3. **评测协议**
   - 自动判分：exact match / F1 / pass@k（代码）/ 规则校验（IFEval 式指令约束）/ 工具调用成功率（τ-bench 式）
   - LLM-as-judge：对照金标准打分；控 bias——位置偏差（A/B 对调）、冗长偏好、自族偏好（judge 与被测同族）、prompt 敏感性
   - 人工抽检：抽样 + 双标注，一致性指标（Cohen's κ / Krippendorff's α）
4. **参考 benchmark 拆解**：MMLU、GPQA、HumanEval、IFEval、BFCL、τ-bench、LiveBench（动态新题防污染的代表）
5. **交付物**
   - 评测集构建 checklist（从需求定义到上线迭代）
   - mini 评测集 demo：15~20 题，分 3 档难度，含题目 schema（id / prompt / reference / rubric / difficulty / tags）、判分脚本、首轮结果表

### 排期

| 时间 | 内容 |
|------|------|
| 9/2（三） | 方法论整理 + benchmark 调研 |
| 9/3（四） | 定义 demo 评测集（主题、难度、判分协议） |
| 9/4（五） | 搭 demo + 跑首轮判分（可先用本地 qwen3.8-27b 或 lv_server 试跑），记录发现 |
| 9/5~6（周末） | 缓冲 / 深挖 / 可选：写一篇短分享 |

---

## 状态跟踪

| 任务 | 状态 | 交付物 | 备注 |
|------|------|--------|------|
| OPD（在策略蒸馏） | ⬜ 待启动 | 笔记 + 方法对比表 | 文献已圈定，待精读 |
| 大模型评测集 | ⬜ 待启动 | checklist + mini demo | 含判分脚本，可实际跑分 |

## 启动方式

两个任务都先委托 kcc（korea 上的 Claude Code）做深度调研，产出结构化报告后闭环回传：
- 任务一：kcc 出 OPD 文献精读 + 方法对比初稿 → 我审校 → 中文笔记落库
- 任务二：kcc 出评测集方法论 + demo 骨架 → 我审校 → 本地搭 demo 跑分

---
> 本文件由 Hermes Agent 通过微信指令 + ssh korea 远程提交
> 工作目录：/home/ubuntu/workspace/claw_bot/test/
