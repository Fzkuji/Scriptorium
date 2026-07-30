# 开源 Memory 框架评测方式：可比性分析与复用决策

> 面向决策的分析文档。输入：`prompts_collection/` 全部 17 份调研文件 + `unified_evaluation_protocol.md`。
> 核心问题：(1) 哪些框架的自报数字互相可比；(2) 我们按统一协议跑出的数字能和谁放同一张表；(3) 各家评测代码里哪些值得直接复用。
> 日期：2026-07-02

---

## 1. 可比性的两个决定轴

LoCoMo 类 J-score（LLM judge 二元准确率）的可比性由两个变量决定，其余（top-k、embedder 等）是次级因素：

1. **Answer prompt 是否有 "less than 5-6 words" 硬字数限制**。judge 是"touches on the same topic 即对"的宽松判分，答案越长越容易蹭对——去掉字数限制会系统性抬高 J-score。这是各家"刷分"最隐蔽的一招。
2. **Judge 谱系**。绝大多数框架用的是同一份 Mem0 `ACCURACY_PROMPT`（gpt-4o-mini、0/1、"be generous"、日期同时段即对无数值窗口、跳过 cat5），judge 侧差异反而小；真正拉开口径的是词面 F1 系（无 judge）与 judge 系之间、以及 cat5 的三种互斥处理。

## 2. LoCoMo 可比性分组

### A 组：标准 Mem0 协议（短答 prompt + Mem0 宽松 0/1 judge + 跳过 cat5，1540 题）——组内互相可比

| 框架 | 自报数字 | 备注 |
|---|---|---|
| Mem0 | 论文 66.9（gpt-4o-mini 系口径） | 谱系源头 |
| Nemori | v4: 80.8（judge gpt-4.1-mini） | answer prompt 含 5-6 words |
| LightMem | ACC 71.95–73.90 | 同款 prompt + judge |
| Memobase | v0.0.37: 75.78 | 整体 fork 自 mem0 evaluation |
| MemMachine（memory 模式） | 0.9123（gpt-4.1-mini 应答）/ 0.8487（gpt-4o-mini） | adapted from Mem0，逐文件注明 |
| MemOS | 75.80（3-run 均值） | **边缘成员**：answer prompt 保留 "under 5-6 words" 但多两条放松（可用世界知识、跨条目综合），baseline 却用 Mem0 原版 prompt——自家数字略占便宜 |
| Memory-R1 | J 62.7 | **弱成员**：judge 模型未披露、官方零代码、RL reward=EM 使模型本身向短答优化 |

组内比较仍需注明：judge 模型不完全一致（gpt-4o-mini vs gpt-4.1-mini）、应答模型不一致（gpt-4o-mini vs gpt-4.1-mini，MemMachine 换 gpt-4.1-mini 提了 6+ 个点）。

### B 组：无字数限制 + 宽松 0/1 judge + 跳过 cat5——组内勉强互比，与 A 组不可比

| 框架 | 自报数字 | answer prompt 约束 |
|---|---|---|
| Zep | ≈0.803（10-run 均值，仓库内可复算） | 仅 "concise"，且大量 timestamp 推理指导 |
| MIRIX | 85.4%（3-run 均值，分支内可复算） | "only state the answer"，无词数上限 |
| MemMachine（agent 模式） | **0.9169**（博客 91.7） | "yours may be longer" + 最多 10+ 轮重检索（每轮 top-30），检索预算远超其他家 |

与 A 组不可比的原因：同一 generous judge 下，去掉字数限制单方面利好 B 组。MemMachine 自己就是活对照——同一仓库同一 judge，memory 模式（5-6 words）0.9123 vs agent 模式（放开）0.9169，还叠加了多轮检索。**91.7、85.4、80.3 这三个数不能和 A 组的 66.9–80.8 放同一列**。

### C 组：词面 F1 系（无 judge 或 judge 关闭）——与 A/B 全部不可比，组内也互不可比

| 框架 | 自报数字 | F1 实现 | cat5 处理 |
|---|---|---|---|
| SimpleMem | F1=43.24 | set-based（去重）token F1 | **计入 overall**：有提示二选一，gold 固定弃答句；Omni 版硬编码恒满分 |
| MemoryOS | F1 提升 49.11%（相对） | set-based F1（`re.findall(\w+)`） | **计入**：gold=adversarial_answer；另有 one-shot 示例疑似取自数据集 |
| A-Mem | 论文 12 自动指标 | set-based F1 | **计入**：gold=adversarial_answer 二选一（奖励幻觉） |
| LoCoMo 官方 Table 2 | gpt-4-turbo 51.6 | **multiset** F1 + stem + 冠词归一 | 计入：关键词匹配 |

三个互斥的 cat5 口径 + set/multiset 之差 + answer prompt 松紧不一，任何两家的 overall F1 都不在同一口径上。**LoCoMo 官方 51.6 与 SimpleMem 43.24 之间没有可比关系。**

### D 组：无评测代码，只能引论文数字并强注不可比

- **Memory-R1**：官方仓库只有 README（"Code coming soon"）。
- **Letta (MemGPT)**：main 无任何 LoCoMo/LME 代码；文献里"MemGPT 在 LoCoMo 的分数"全部出自第三方 harness（Mem0 evaluation / Zep harness），引用时必须标注来源 harness 而非 Letta。
- **MemoryBank**：只有自造数据 + 从未调用的 answer prompt，论文分数为人工评。

## 3. LongMemEval 可比性

**官方协议组**（官方 5 分题型 anscheck judge，temporal off-by-one、abstention 单独规则）：

- LongMemEval 官方论文（judge gpt-4o-2024-08-06）
- Zep（`zep_longmem_eval.py`，judge gpt-4o，官方模板）
- MemMachine main（`longmemeval_evaluate.py`，官方模板逐字照搬，gpt-4o，abstention 也评）
- LightMem、Nemori（官方模板，judge 模型换小）
- MIRIX（`evals/mab/llm_judge_mab.py`，官方 judge 经 MemoryAgentBench 移植，gpt-4o；注意数据经 HF `ai-hyz/MemoryAgentBench` 转手）

这一组 judge 协议同源，可放同表（脚注 judge 模型差异）。LME 官方协议本身不限答案长度，answer prompt 差异的影响小于 LoCoMo，但仍应注明。

**不可比**：

- MemOS（LME +40.43%）：judge 用通用模板（LoCoMo judge 换例子），**非官方分题型模板**，abstention/temporal 规则全丢。
- Mem0 benchmarks：统一放宽版 judge（"When in doubt, lean toward yes"、off-by-one 泛化到所有题型）+ 数据集特调 answer prompt（"chandelier counts as jewelry"）。
- SimpleMem EvolveMem：默认 token F1 评 LME，adapter 注释自认非官方协议。

## 4. 我们（unified_evaluation_protocol.md）能和谁同表

我们的协议：LoCoMo cat1-4 全量 1540 题，**Mem0 原版 5-6 words answer prompt** + **Mem0 ACCURACY_PROMPT judge（GPT-5.5）**；LongMemEval-S 500 题用**官方 5 模板 judge**；主指标 J-score，辅 set-based F1 + BLEU-1。

### 4.1 LoCoMo 主表（J-score）——可同表，加脚注

| 对方数字 | 脚注内容 |
|---|---|
| Mem0（论文/自跑） | judge 模型不同（gpt-4o-mini → GPT-5.5）；最好用本地 `mem0-benchmarks/` 重跑统一 |
| Nemori v4 80.8 | judge gpt-4.1-mini；建议重跑统一 |
| LightMem 71.95–73.90 | judge gpt-4o-mini；prompt 同款 |
| Memobase 75.78 | judge gpt-4o-mini；其 README 中 baseline 数字摘自 mem0 论文非重跑，引用时只用 Memobase 自报值 |
| MemMachine memory 模式 0.9123 | 应答模型 gpt-4.1-mini（比我们统一的应答模型强）；agent 模式 91.7 **不进此表** |
| MemOS 75.80 | 其自家 answer prompt 有两条放松；3-run 均值 |
| Memory-R1 J 62.7 | judge 模型未披露、官方无代码、测试集是 1307 题子集（非 1540）——最弱的可比性，只作参考行 |

### 4.2 LoCoMo——完全不能同表（只能文字对比 + 说明口径）

- **Zep 0.803 / MIRIX 85.4 / MemMachine agent 91.7**：answer prompt 无字数限制（MemMachine 还叠加 agent 多轮检索）。若要对比，唯一正路是用我们的统一 answerer/judge **重跑**它们的记忆系统。
- **SimpleMem 43.24 / MemoryOS / A-Mem 论文值**：词面 F1 不同口径 + cat5 计入且口径互斥。我们的 set-based F1 辅表可与 SimpleMem/A-Mem/Nemori/LightMem 的 F1 数字并列（同一份 A-Mem `utils.py` 实现），但 SimpleMem 的 overall 含 cat5 白送分，须取其 cat1-4 分项或加脚注；MemoryOS 的 F1 建议不比（cat5=adversarial_answer + one-shot 疑似数据泄漏）。
- **LoCoMo 官方 Table 2（51.6 等）**：multiset F1 + 含 cat5，与所有以上都不可比，只作背景引用。

### 4.3 LongMemEval——可同表，加脚注

可同表：官方论文数字、Zep、MemMachine main、LightMem、Nemori、MIRIX（MAB 口径注明数据转手）。统一脚注：judge 模型（官方 gpt-4o-2024-08-06 / 各家 gpt-4o 或更小 / 我们 GPT-5.5）+ 各家 answer prompt 不统一。
不可同表：MemOS、Mem0 benchmarks、SimpleMem（原因见 §3）。

## 5. 各框架评测代码中值得复用的部分（文件路径级）

已在协议中确定复用的（LightMem judge、A-Mem utils.py F1/BLEU、LongMemEval 官方 judge、Mem0 三段式 runner）不重复列，以下是本轮 9 家调研新发现的可复用件：

| 来源 | 路径 | 复用价值 |
|---|---|---|
| Zep | `getzep/zep: benchmarks/locomo/evaluation.py`（`evaluate_context_completeness`） | **检索质量与答案质量分离评测**：三档 COMPLETE/PARTIAL/INSUFFICIENT judge + `accuracy_with_complete_context`。可移植为我们的分析指标（wiki 检索是否含答案 vs 答案是否对） |
| Zep | `benchmarks/locomo/experiments/`（5 组 ×10 runs 完整 JSON） | 现成对照数据：可用我们的 judge 对其答案 rejudge，验证 judge 口径差多少 |
| MemOS | `MemTensor/MemOS: evaluation/scripts/locomo/`（`--lib` 多框架适配器 + `run_locomo_eval.sh` num_runs=3） | 多框架接入结构 + **judge 多 run 取均值±std** 的做法值得采纳（我们协议目前 1 run）；`locomo_rag.py` 是现成 chunk-500 RAG baseline |
| MemMachine | `MemMachine/MemMachine: evaluation/episodic_memory/longmemeval_evaluate.py` | LongMemEval 官方 anscheck 的**逐字移植**（含 abstention 处理），比我们从官方 repo 抠代码省事，可直接抄 |
| MIRIX | `Mirix-AI/MIRIX: evals/organize_results.py` | 指标之外的**成本维度**：延迟分解（构建/检索/作答）、credit 成本、记忆库 token 数（tiktoken）——论文效率对比表可直接对齐这套输出 |
| MIRIX | `public_evaluation` 分支 `public_evaluations/baselines/`（vendored Mem0 harness + zep-papers + 各家结果 json） | 现成的 gpt-4.1-mini 口径 Mem0/LangMem/Zep/RAG 结果 json，可 rejudge 做交叉验证 |
| MIRIX | `evals/mab/llm_judge_substring.py` | MAB `substring_exact_match` 的干净实现（normalize + 子串），可作我们 F1 之外的免 LLM 快检指标 |
| Memobase | `memodb-io/memobase: docs/experiments/locomo-benchmark/fixture/memobase/`（两版预测+judge 结果） | 同上，可 rejudge；`compute_p95_latency.py` 是现成检索延迟统计 |
| SimpleMem | `aiming-lab/SimpleMem: OmniSimpleMem/benchmarks/locomo/run_locomo.py` L60-156 | 全网唯一第三方实现的 **LoCoMo 官方 task_eval 风格 F1**（normalize+Porter stem+multiset+cat1 逗号拆分），可用来交叉验证我们如果要报官方口径 F1 |
| Zep | `benchmarks/longmemeval/zep_memgpt_eval.ipynb` | 若需补 DMR（MSC）实验，这是唯一现成 harness（MemGPT 官方从未发布 DMR 代码） |

**明确不复用**：Memory-R1（无代码）、Letta（已删）、MemoryBank（无打分代码）、MemoryOS `eval/`（协议有缺陷：one-shot 疑似泄漏 + cat5 词面对 adversarial_answer）、SimpleMem EvolveMem 的 prompt 进化层（本质是对 gold 格式过拟合，反面教材，可在论文 related work 里作为"评测博弈"例证引用）。

## 6. 决策要点汇总

1. **主表阵容**（LoCoMo J-score，统一协议重跑）：NativeMem、Mem0、A-Mem、Nemori、LightMem + full-context/RAG 下界，可选加 Memobase、MemMachine memory 模式（两者接入成本低，harness 与 Mem0 同构）。Zep 需商业 cloud API，接入成本高，建议只引其自报 0.803 并注明口径不可比。
2. **论文引用表**分两块：块一（可同表，带 judge 模型脚注）= A 组自报值；块二（口径不可比，单独一表或文字）= Zep/MIRIX/MemMachine-agent/SimpleMem/MemoryOS/官方 Table 2。绝不混排。
3. **协议增强建议**（低成本采纳）：judge 3-run 均值±std（学 MemOS）；报 context completeness（学 Zep）；报构建/检索延迟与记忆库 token 数（学 MIRIX）。
4. **写作素材**：MemMachine memory vs agent 模式（0.9123 vs 0.9169）、MemOS 自家 prompt 与 baseline prompt 的不对称、SimpleMem cat5 白送分、MemoryOS one-shot 泄漏——这些是"benchmark 口径博弈"的现成证据，可支撑我们统一协议一节的动机论证。
