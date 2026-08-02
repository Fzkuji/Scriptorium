# 实验规划

> 2026-06-29 更新：本文件保留早期 LoCoMo/qwen3.6-flash 阶段的实验规划。面向 NativeMem 预写论文和后续提交的新版实验路线已经迁移到 `../experiments/runs/refine-logs/EXPERIMENT_PLAN.md`，具体 run tracker 在 `../experiments/runs/refine-logs/EXPERIMENT_TRACKER.md`。后续不要把这里的早期真实结果和预写论文里的理想完整结果直接混用。

## 1. 数据集

| 数据集 | 来源 | 规模 | 谁用了 |
|---|---|---|---|
| **LoCoMo** | ACL 2024 | 10 样本, ~1540 QA, 5 类 | Mem0 论文、A-Mem 论文都用了 |
| **LongMemEval** | ICLR 2025 | 500 题, 5 类 | Mem0 benchmark repo 用了(非论文) |

**结论：主实验用 LoCoMo**。两篇对标论文都在这个上面报了数字，直接可比。LongMemEval 作为补充实验。

---

## 2. 统一模型

两篇论文都用 **GPT-4o-mini** 作为统一模型。我们也应该用 GPT-4o-mini。

但我们没有 OpenAI API 额度。替代方案：
- **MiniMax M2.5**：已验证可用，但能力不够稳定（样本 1 崩塌）
- **Codex (ChatGPT 订阅)**：用 GPT-5.5，能力强但太慢
- **需要解决**：要么充一点 OpenAI API 额度跑 GPT-4o-mini（和论文完全一致），要么用其他模型但需要同时跑所有 baseline 保证公平

---

## 3. 评测指标

两篇论文都报了以下指标：
- **F1**（token 级别，主指标）
- **BLEU-1**
- **J-score / LLM judge**（Mem0 论文额外报了）

**结论：主指标用 F1**，和两篇论文对齐。BLEU-1 作为辅助。可选 LLM judge 作为补充。

---

## 4. 需要对标的方法 + 已有数字

### LoCoMo 上的已有结果（GPT-4o-mini, F1）

来源：Mem0 论文 Table

| 方法 | Single-Hop | Multi-Hop | Open-Domain | Temporal | 论文 | 有代码 |
|---|---|---|---|---|---|---|
| LoCoMo baseline | 25.02 | 12.04 | 40.36 | 18.41 | LoCoMo | ? |
| MemoryBank | 5.00 | 5.56 | 6.61 | 9.68 | Mem0 | ? |
| ReadAgent | 9.15 | 5.31 | 9.67 | 12.60 | Mem0 | ? |
| MemGPT | 26.65 | 9.15 | 41.04 | 25.52 | 两篇都有 | 有 |
| A-Mem | 27.02* | 12.14* | 44.65* | 45.85* | A-Mem | **有，已下载** |
| LangMem | 35.51 | 26.04 | 40.91 | 30.75 | Mem0 | ? |
| Zep | 35.74 | 19.37 | 49.56 | 42.00 | Mem0 | ? |
| OpenAI Memory | 34.30 | 20.09 | 39.31 | 14.04 | Mem0 | ? |
| **Mem0** | **38.72** | **28.64** | **47.65** | **48.93** | Mem0 | 有但需部署 |
| Mem0ᵍ (with graph) | 38.09 | 24.32 | 49.27 | 51.55 | Mem0 | 有但需部署 |

*A-Mem 自己论文的数字和 Mem0 论文里跑的 A-Mem 数字不完全一致，以各自论文为准。

### 哪些需要自己跑

| 方法 | 是否需要自己跑 | 原因 |
|---|---|---|
| LoCoMo baseline | 不需要 | 直接引用 Mem0 论文数字 |
| MemoryBank | 不需要 | 直接引用 |
| ReadAgent | 不需要 | 直接引用 |
| MemGPT | 不需要 | 直接引用（两篇论文数字一致） |
| A-Mem | **可自己跑验证** | 有代码，已下载 |
| LangMem | 不需要 | 直接引用 |
| Zep | 不需要 | 直接引用 |
| OpenAI Memory | 不需要 | 直接引用 |
| Mem0 | 不需要 | 直接引用论文数字（不是 v3 产品数字） |
| **Ours (Wiki)** | **必须自己跑** | — |
| **Ours (Flat baseline)** | **必须自己跑** | 消融对照 |
| **No memory** | **必须自己跑** | 下界 |
| **Full context** | **必须自己跑** | 上界参考 |

### 总结：需要自己跑的

1. **Wiki Memory (ours)** — 核心方法
2. **Flat Memory** — 消融 baseline（提取同样的记忆但不做 wiki 组织）
3. **No Memory** — LLM 直接回答，下界
4. **Full Context** — 全部对话塞 context，上界
5. **A-Mem（可选）** — 如果用和论文不同的模型，需要自己跑保证公平

---

## 5. 实验矩阵

### 主实验：LoCoMo 全集（10 样本, ~1540 QA）

| 方法 | 来源 | 模型 |
|---|---|---|
| Wiki Memory (ours) | 自己跑 | GPT-4o-mini 或统一模型 |
| Flat Memory | 自己跑 | 同上 |
| No Memory | 自己跑 | 同上 |
| Full Context | 自己跑 | 同上 |
| A-Mem | 引用论文 or 自己跑 | GPT-4o-mini |
| MemGPT | 引用论文 | GPT-4o-mini |
| Mem0 | 引用论文 | GPT-4o-mini |
| Zep | 引用论文 | GPT-4o-mini |
| LangMem | 引用论文 | GPT-4o-mini |
| OpenAI Memory | 引用论文 | GPT-4o-mini |

### 消融实验

| 消融 | 目的 |
|---|---|
| Wiki vs Flat | 证明 wiki 结构化组织有效 |
| RSP vs 随机放置 | 证明检索模拟定位有效 |
| 有反馈重组 vs 无反馈 | 证明 RFR 有效 |
| 不同 wiki 粒度 | 文件多/少对效果的影响 |

### 补充实验

| 实验 | 目的 |
|---|---|
| LongMemEval | 第二个 benchmark，增加说服力 |
| 不同模型 | GPT-4o-mini / Qwen / Llama 跨模型泛化 |
| 大规模 wiki 导航 | 当 wiki 太大塞不进 context 时，导航 vs 全塞 |

---

## 6. 框架选择

| 组件 | 方案 |
|---|---|
| 数据加载 | A-Mem 的 `load_dataset.py`（已有） |
| 评测指标 | A-Mem 的 `utils.py`（F1/BLEU/ROUGE/BERTScore，已有） |
| LLM judge | Mem0 的 `prompts.py`（J-score，已有） |
| A-Mem baseline | `test_advanced_robust.py`（已有，需改后端） |
| 我们的方法 | `run_locomo_v4.py` 基础上优化 |

---

## 7. 关键问题待解决

1. **模型选择**：用 GPT-4o-mini（和论文一致但要花钱）还是用 MiniMax（免费但不一致）？
   - 如果用 MiniMax，必须同时跑 A-Mem 在 MiniMax 上的数字，不能引用论文数字
   - 如果用 GPT-4o-mini，可以直接引用所有论文数字

2. **速度问题**：LoCoMo 全集 1540 题 × 每题多次 LLM 调用，总计几千次调用
   - MiniMax：每次 1-2 秒，总计 1-2 小时
   - Codex：每次 10-30 秒，总计 10+ 小时
   - GPT-4o-mini API：每次 1-2 秒但要花钱

3. **存储质量**：当前最大瓶颈。MiniMax 在样本 1 上只生成了 8 个文件，信息大量丢失。需要优化存储 prompt 或换更强模型。

---

## 8. 建议的执行顺序

1. 解决模型问题（确定用什么模型）
2. 优化存储 prompt（解决信息丢失问题）
3. 跑 LoCoMo 全集（我们的 4 个方法）
4. 用 A-Mem 的 F1 指标 + Mem0 的 J-score 双指标评测
5. 整理结果，和论文数字对标
6. 跑消融实验
7. 跑 LongMemEval 补充实验
