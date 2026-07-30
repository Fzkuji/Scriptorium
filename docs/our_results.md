# NativeMem 实验结果汇总

更新时间: 2026-07-03

---

## 2026-07-12 汇总（论文表格底稿，全部从 eval_full.json 重算）

口径：**标准** = judge_score≥1；**严格** = judge_score≥1 且答案不匹配弃答正则 `not mentioned|no information|cannot be determined|not specified|unknown`（大小写不敏感）。F1 = set-based token F1（A-Mem/Nemori 同款实现），BLEU-1 同参考实现，均从 records 的 question/gold/answer 离线补算。类别号：1=multi-hop，2=temporal，3=open-domain，4=single-hop。

### 主结果 — LoCoMo cat1-4，gpt-4o-mini 全管线，10 样本 1540 题（`results/v88-4omini-official`）

| 口径 | single-hop | multi-hop | temporal | open-domain | **overall** |
|---|---|---|---|---|---|
| 标准 J | 82.9 | 77.3 | 68.2 | 70.8 | **78.1** |
| 严格 J | 82.4 | 77.0 | 67.0 | 69.8 | **77.4** |
| F1 | 45.1 | 32.6 | 46.7 | 14.8 | **41.2** |
| BLEU-1 | 39.9 | 25.9 | 41.9 | 12.8 | **36.1** |

题数：single-hop 841 / multi-hop 282 / temporal 321 / open-domain 96。open-domain F1 极低（14.8）是回答冗长导致 precision 惩罚，J 达 70.8 说明答案实为正确。

每样本（标准 J overall）：s0=79.6 · s1=87.7 · s2=82.2 · s3=81.4 · s4=80.3 · s5=75.6 · s6=70.0 · s7=74.9 · s8=74.4 · s9=77.8。

### 版本演进 — qwen3.6-flash，s0 单样本 152 题，严格 J

| 版本 | 改动 | overall | SH | MH | TMP | OD | 建库时间(s) | 调用 | tok_out |
|---|---|---|---|---|---|---|---|---|---|
| v8.3 promptfix | 修指令遵循 | 90.8 | 87.1 | 87.5 | 97.3 | 100.0 | 1469 | 83 | 0.34M |
| v8.4 article | 文章化事件文件 | 88.8 | 88.6 | 87.5 | 89.2 | 92.3 | 5230 | 247 | 1.21M |
| v8.5 sections | 分节+去重 | 96.1 | 98.6 | 90.6 | 94.6 | 100.0 | 7143 | 344 | 1.63M |
| v8.6 speedup | 合并调用提速 | 94.1 | 91.4 | 93.8 | 97.3 | 100.0 | 3661 | 256 | 1.24M |
| v8.7 fix | 答题侧修复 | 92.8 | 94.3 | 87.5 | 91.9 | 100.0 | 4070 | 366 | 1.62M |
| v8.8 final | 引用格式统一 | 94.1 | 92.9 | 96.9 | 97.3 | 84.6 | 4583 | 333 | 1.46M |
| **v9.0 2-stage** | **两级转写/谱曲** | 92.8 | 92.9 | 87.5 | 97.3 | 92.3 | **1553** | **96** | **0.41M** |

v9.0 与 v8.8 同水平（标准 94.1 / 严格 92.8）但建库成本约 1/3：调用 0.29×、tok_out 0.28×、时间 0.34×。v8.8 标准与严格一致（无弃答），v9.0 标准 94.1 / 严格 92.8（差一题）。

### 切块策略对照 — v8.8，s0，严格 J（证明对切块不敏感）

| 切块 | overall | SH | MH | TMP | OD |
|---|---|---|---|---|---|
| 原子逐句 | 94.1 | 92.9 | 96.9 | 97.3 | 84.6 |
| 固定 10 句块 | 93.4 | 92.9 | 93.8 | 97.3 | 84.6 |
| 话题边界 | 93.4 | 94.3 | 90.6 | 91.9 | 100.0 |

overall 三者差 ≤0.7 点。

### 主表基线（共识值，来自 [表 1a](#表-1alocomo-llm-judge--gpt-4o-mini) / [published_results_matrix.md](published_results_matrix.md)）

Full-context 72.7 · OpenAI-Memory 52.8 · A-Mem 54.1 · LangMem 55.6 · MemoryOS 56.6 · MemU 58.9 · Zep 61.2 · Mem0 61.8 · MemOS 72.6（均 ≥2 家中立重跑）。自报参考区（‡，vendor-reported / 未独立复现）：LightMem 72.0 · TiMem 75.3 · Mnemis 73.0（judge 4.1-mini）· MIRIX 85.4（judge 4.1-mini）· MemMachine 84.9 · MemU 自报 92.1。

论文表见 `paper/4Experiments.tex`：表 `tab:locomo_main`（主 J 对比）、`tab:locomo_lexical`（F1/BLEU-1 分项）、`tab:progression`（版本演进+成本）、`tab:chunking`（切块对照）。

---

## ★ 最终对比总表（论文主表底稿）

策略（详见 [selected_results.md](selected_results.md)）：被 ≥2 家中立论文重跑的方法直接取**重跑均值**（不自己跑）；只有自报的我们复现；争议的我们仲裁。数字明细全部可溯源到 [published_results_matrix.md](published_results_matrix.md)。

### 模型配置

| 阶段 | Builder / Answerer | Judge | 状态 |
|---|---|---|---|
| 筛查轮（当前） | gpt-5.4-mini（订阅，¥0）| gpt-5.5（订阅）| Full-ctx 校准 72.4 ≈ 文献 4o-mini 组 72.7，**Δ≈0 可直接对表** |
| 正式轮（最终报数） | gpt-4o-mini（API，全套约 $40）| gpt-4o-mini + gpt-5.5 双 judge | 待跑 |

### 表结构说明

3 个 benchmark × 2 个模型配置（gpt-4o-mini / gpt-4.1-mini）= 6 张表。行 = 方法，列 = 子集 + overall。
- 数值 = 中立重跑均值（≥2 家）或单源重跑（标 ¹）；**空白 = 无中立数据，待我们复现**
- ‡ = 仅自报（放表底参考区，不与共识值同列比较）
- 我们的实测最终填入空白格；正式轮全部用同款模型

---

### 表 1a：LoCoMo LLM-Judge @ gpt-4o-mini

| 方法 | single-hop | multi-hop | temporal | open-domain | **overall** | 源数 |
|---|---|---|---|---|---|---|
| Full-context | 84.8 | 67.8 | 53.2 | 52.4 | **72.7** | 4 |
| OpenAI-Memory | 63.8¹ | 42.9¹ | 21.7¹ | 62.3¹ | **52.8** | 2 |
| LangMem | 61.8 | 50.2 | 24.2 | 59.4 | **55.6** | 4 |
| A-Mem | 65.1 | 49.8 | 57.3 | 27.1 | **54.1** | 4 |
| MemoryOS | 64.8 | 54.6 | 39.1 | 45.8 | **56.6** | 4 |
| MemU | | | | | **58.9** | 2 |
| Mem0 | 67.3 | 58.3 | 54.8 | 42.2 | **61.8** | 6 |
| Zep (graphiti OSS) | 62.5 | 45.9 | 54.1 | 58.1 | **61.2** | 3 |
| Memobase | | | | | **72.0**¹ | 1 |
| MemOS | | | | | **72.6** | 2 |
| MIRIX | | | | | **64.3**¹ | 1 |
| RAG/BM25 | | | | | （30.2~61.0 配置各异）| — |
| Nemori | | | | | | 0 → 我们复现 |
| LightMem | | | | | | 0 → 我们复现 |
| EverMemOS | | | | | | 0 → 我们复现 |
| MemMachine | | | | | | 0 → 我们复现 |
| E-Mem | | | | | | 0 → 我们复现 |
| SimpleMem | | | | | | 0 → 我们复现 |
| **NativeMem** | | | | | | 我们 |
| *自报参考区* | Nemori 73.0‡ · LightMem 72.0~73.0‡ · TiMem 75.3‡ · MemOS 75.8‡ · EverMemOS 86.8‡ · MemMachine 87.5‡ · E-Mem 78.0‡ | | | | | |

### 表 1b：LoCoMo LLM-Judge @ gpt-4.1-mini

| 方法 | single-hop | multi-hop | temporal | open-domain | **overall** | 源数 |
|---|---|---|---|---|---|---|
| Full-context | | | | | **80.6**¹ | 1 [Nemori] |
| LangMem | | | | | **73.4**¹ | 1 [Nemori] |
| A-Mem | | | | | **61.4**¹ | 1 [Nemori] |
| MemoryOS | | | | | **60.4** | 2 |
| Mem0 | | | | | **65.3** | 2 |
| Zep | | | | | 冲突：61.6 [Nemori] vs 85.2 [EverMemOS⚠] | 2 |
| MemOS | | | | | **80.8**¹ | 1 [EverMemOS⚠] |
| MemU | | | | | **66.7**¹ | 1 [EverMemOS⚠] |
| **NativeMem** | | | | | | 我们 |
| *自报参考区* | Nemori 80.8‡ · MIRIX 85.4‡ · EverMemOS 93.1‡ · MemMachine 91.2/91.7‡ · Mnemis 93.3‡（judge 同为 4.1-mini）| | | | | |

⚠ EverMemOS 作为报数方系统性偏高（其报的 Zep/MemOS 比其他家高 15~25），其单源值谨慎使用。

### 表 2a：LongMemEval-S @ gpt-4o-mini（官方 5 模板 judge）

| 方法 | ss-user | ss-assistant | ss-preference | multi-session | temporal | knowledge-update | **overall** | 源数 |
|---|---|---|---|---|---|---|---|---|
| Full-context | 78.6~87.1 | 89.3 | 6.7~36.7 | 38.3~45.5 | 76.9~78.2 | 31.6~42.1 | **55.7** | 3 |
| NaiveRAG | 90.0¹ | 98.2¹ | 53.3¹ | 48.5¹ | 68.0¹ | 39.9¹ | **61.0**¹ | 1 [LightMem] |
| LangMem | 60.0¹ | 46.4¹ | 60.0¹ | 20.3¹ | 66.7¹ | 15.8¹ | **37.2**¹ | 1 [LightMem] |
| A-Mem | 87.9 | 92.0 | 43.0 | 44.6 | 68.5 | 41.7 | **59.0** | 2 |
| MemoryOS | 80.6 | 71.2 | 40.7 | 37.9 | 52.4 | 42.9 | **51.4** | 2 |
| Mem0 | 85.9 | 39.9 | 66.7 | 58.5 | 71.8 | 54.1 | **61.7** | 3 |
| Memobase | 92.9¹ | 23.2¹ | 80.1¹ | 66.9¹ | 89.7¹ | 75.9¹ | **72.4**¹ | 1 [MemOS] |
| MemU | 67.1¹ | 19.6¹ | 76.7¹ | 42.1¹ | 41.0¹ | 17.3¹ | **38.4**¹ | 1 [MemOS] |
| MIRIX | 72.9¹ | 63.6¹ | 53.3¹ | 30.1¹ | 52.6¹ | 25.6¹ | **43.5**¹ | 1 [MemOS] |
| Supermemory | 85.7¹ | 58.9¹ | 89.9¹ | 52.6¹ | 55.1¹ | 44.4¹ | **58.4**¹ | 1 [MemOS] |
| MemOS | 93.7¹ | 67.9¹ | 50.7¹ | 58.8¹ | 76.7¹ | 65.1¹ | **68.7**¹ | 1 [TiMem] |
| MemoryBank | 29.7¹ | 50.0¹ | 12.0¹ | 9.8¹ | 21.8¹ | 17.1¹ | **21.0**¹ | 1 [TiMem] |
| Zep | | | | | | | | 0（自报 63.8‡）→ 我们复现 |
| Nemori | | | | | | | | 0（自报 64.2‡）→ 我们复现 |
| LightMem | | | | | | | | 弱验证（自报 68.6‡ ≈ SimpleMem 宏平均 68.7）|
| ES-Mem | | | | | | | | 0（自报 72.4‡）→ 我们复现 |
| **NativeMem** | | | | | | | | 我们 |
| *自报参考区* | MemOS 77.8‡ · TiMem 76.9‡ · Mnemis 87.2‡ | | | | | | | |

### 表 2b：LongMemEval-S @ gpt-4.1-mini

数据极少（仅 SimpleMem 一篇用 4.1-mini judge 且报的是 6 类宏平均，口径不同）：Full-ctx 39.6 / Mem0 59.8 / LightMem 68.7 / SimpleMem 自报‡。**此表基本靠我们正式轮填充。**

### 表 3a/3b：BEAM @ 两模型

公开数字仅 Mem0-Platform 自报（GPT-5 配置，口径不可比）。**两张表完全靠我们实测**：按协议先 100K 冒烟，再 1M 全量 700 题；行 = NativeMem / Mem0 / Full-ctx / BM25（可行子集）。

---

### 旧版表 1（保留原稿供参考）：LoCoMo（cat1-4, LLM-Judge）

| 方法 | 共识均值† | 重跑来源数 | 范围 | 我们实测 (s0, 5.4-mini) | 结论/状态 |
|---|---|---|---|---|---|
| Full-context | 72.7 | 4 家 | 71.6~73.8 | **72.4** ✓ | 校准通过，引用+实测双列 |
| BM25/RAG | 配置差异大 | — | 30.2~68.2 | 待跑 | 我们实测为准 |
| OpenAI-Memory | 52.8 | 2 家 | 52.8±0.1 | 不跑 | 引用† |
| LangMem | 55.6 | 4 家 | 51.3~58.1 | 不跑 | 引用† |
| A-Mem | 54.1 | 4 家 | 48.4~64.2 | 不跑（太慢）| 引用† + 波动脚注 |
| MemoryOS | 56.6 | 4 家 | 54.5~60.8 | 不跑 | 引用† |
| MemU | 58.9 | 2 家 | 56.6~61.2 | 不跑 | 引用† |
| Mem0 | 61.8 | 6 家 | 57.8~64.6 | 待跑（仲裁）| 引用† + 我们实测 |
| Zep (graphiti OSS) | 61.2 | 3 家 | 58.5~66.0 | **48.0**（5.4-mini 建图上界）| 引用† + 实测脚注：云版自称 75.1 无中立复现 |
| Memobase | 72.0 | 1 家 | — | 可选 | 单源引用，标注 |
| MemOS | 72.6 | 2 家 | 69.2~75.9 | 不跑 | 引用† |
| MIRIX | 64.3 | 1 家 | 自报 85.4 | 待跑（仲裁）| Δ21 争议 |
| Nemori | 自报 73.0‡ | 0 | — | 待跑（首复现）| 桶 C |
| LightMem | 自报 72.0~73.0‡ | 0 | — | 待跑（首复现）| 桶 C |
| EverMemOS | 自报 86.8/93.1‡ | 0 | — | 待跑（首复现）| 桶 C |
| MemMachine | 自报 87.5~91.7‡ | 0 | — | 待跑（首复现）| 桶 C |
| Hindsight | 自报 89.6‡ | 0 | — | 待跑（首复现）| 桶 C |
| E-Mem | 转录 78.0~85.3‡ | 0 | — | 待跑（首复现）| 桶 C |
| **NativeMem（我们）** | — | — | — | **待跑** | 主角 |
| *引用组（无代码）* | Mnemis 93.3‡ / TiMem 75.3‡ / Memory-R1 62.7‡ / Synthius 94.4‡ / ByteRover 96.1‡ / HORMA 51.6‡ | | | 不可复现 | 表注不可比 |

†=中立重跑均值（排除自报与转录）；‡=仅自报，未经独立验证。

### 表 2：LongMemEval-S（LLM-Judge, 官方 5 模板组）

| 方法 | 共识均值† | 重跑来源数 | 范围 | 状态 |
|---|---|---|---|---|
| Full-context | 55.7 | 3 家 | 55.0~56.8 | 引用†（正式轮我们校准实测）|
| NaiveRAG | 61.0~67.2 | 2 家（配置不同）| — | 我们实测为准 |
| LangMem | 37.2 | 1 家 | — | 单源引用 |
| MemoryOS | 51.4 | 2 家 | 44.8~58.0 | 引用† + 波动脚注 |
| Mem0 | 61.7 | 3 家 | 53.6~66.4 | 引用† + 波动脚注 |
| A-Mem | 59.0 | 2 家 | 55.4~62.6 | 引用† |
| MemU | 38.4 | 1 家 | — | 单源引用 |
| Memobase | 72.4 | 1 家 | — | 单源引用 |
| MIRIX | 43.5 | 1 家 | — | 单源引用 |
| MemOS | 68.7 | 1 家（TiMem）| 自报 77.8 | 引用重跑值 |
| Zep | 自报 63.8‡（MemOS 列为转录）| 0 | — | 待实测 |
| Nemori | 自报 64.2‡ | 0 | — | 待跑（首复现）|
| LightMem | 自报 68.6‡（SimpleMem 宏平均重跑 68.7 ≈ 吻合）| ~1 | — | 弱验证 |
| TiMem | 自报 76.9‡ | 0 | — | 引用组 |
| ES-Mem | 自报 72.4‡（其 baseline 全为转录 LightMem）| 0 | — | 待跑（首复现）|
| Mnemis | 自报 87.2‡（judge 用 4.1-mini，口径不同）| 0 | — | 引用组 |
| **NativeMem（我们）** | — | — | — | 待跑 |

### 表 3：F1 / BLEU-1（辅指标）

引用源：Mem0 论文 Table 1（唯一 per-category F1+BLEU-1 全报，⚠ 标签错位脚注）+ A-Mem 论文（12 指标）。我们所有实测系统由 evaluate.py 自动产出全指标（judge/f1/f1_official/bleu1-4/rouge/em），正式轮直接填入。

### BEAM

可选扩展：只有 Mem0-Platform 公开过基线（自报口径）。按协议 §6.1：冒烟 100K → 正式 1M 700 题，Phase 2 之后视时间决定。

### 待跑队列（更新）

1. ~~Full-ctx（校准）~~ ✓ 72.4
2. ~~Zep 仲裁~~ ✓ 48.0（上界）
3. NativeMem（主角，s0 → 全量）
4. Mem0（仲裁，预期 ~61.8）、BM25（下界）、MIRIX（仲裁 64 vs 85）
5. 桶 C 首复现：Nemori、LightMem、EverMemOS、MemMachine、Hindsight、SimpleMem、E-Mem
6. 正式轮：全套换 gpt-4o-mini 重跑（需 OpenAI API key，~$40）

---

## 0. 旧版结果 (memory_builder.py, qwen3.6-flash, 全 10 sample)

旧版代码使用 qwen3.6-flash 模型，10 轮拆分 + 5W1H 框架 + copy-don't-generate。跑了全部 10 个 LoCoMo sample（1986 题）。

### Overall (1986 题, LJ 0-100, judge=qwen3.6-flash)

| 指标 | 值 |
|---|---|
| **Overall LJ** | **66.3** |
| **Accuracy (LJ>=50)** | **70.1%** |

### Per-Category

| Category | n | LJ | Accuracy |
|---|---|---|---|
| Open-domain | 841 | **80.6** | 83.4% |
| Temporal | 321 | **81.4** | 83.2% |
| Single-hop | 282 | **74.3** | 84.8% |
| Multi-hop | 96 | 51.8 | 52.1% |
| Adversarial | 446 | 26.6 | 30.3% |

### Per-Sample

| Sample | n | LJ |
|---|---|---|
| 0 | 199 | 68.0 |
| 1 | 105 | 65.5 |
| 2 | 193 | 72.9 |
| 3 | 260 | 66.8 |
| 4 | 242 | 64.5 |
| 5 | 158 | 70.4 |
| 6 | 190 | 73.3 |
| 7 | 239 | 40.0 |
| 8 | 196 | 75.4 |
| 9 | 204 | 72.6 |

### 说明
- LJ 是 0-100 连续分（非 0/1 binary），judge 模型为 qwen3.6-flash
- 构建模型和检索模型均为 qwen3.6-flash
- 与其他论文的 LJ（0/1 binary + GPT-4o judge）不可直接比较
- Sample 7 异常低（40.0），其余 sample LJ 在 64-75 之间

---

## 1. 新版配置 (memory_builder_v2.py, deepseek-v4-flash)

| 项目 | 配置 |
|---|---|
| Benchmark | LoCoMo Sample 0 (199 题, 19 sessions) |
| 构建/检索模型 | deepseek-v4-flash (阿里云 API) |
| Judge 模型 | GPT-5.5 (ChatGPT proxy, 待完成) / deepseek-v4-flash (已完成部分) |
| 评测指标 | F1 (LoCoMo 官方 token-level) + LLM Judge (0/1 binary, 待跑) |

### 公平性说明

- 三个方法（NativeMem、Mem0、A-Mem）全部使用 **deepseek-v4-flash** 构建记忆和检索回答
- Judge 使用同一个模型评分
- **当前未统一 answerer**：每个方法用自己的方式回答，导致回答长度差异大（NativeMem 25 词 vs Mem0 4 词 vs A-Mem 5 词），F1 受回答长度影响
- 标准做法应统一 answerer，后续需要改进

---

## 2. LoCoMo F1 结果 (Sample 0, 199 题)

### 总览

| 方法 | Overall F1 | 平均回答长度 | 构建模型 | 记忆条数/文件数 |
|---|---|---|---|---|
| **A-Mem** | **0.406** | 5 词 | deepseek-v4-flash | 199 条笔记 |
| **NativeMem v3** | 0.376 | 25 词 | deepseek-v4-flash | 25 文件 |
| Mem0 | 0.176 | 4 词 | deepseek-v4-flash | 20 条记忆 |

### Per-Category F1

| Category | n | NativeMem v3 | A-Mem | Mem0 |
|---|---|---|---|---|
| Multi-hop | 32 | 0.140 | **0.196** | 0.176 |
| Temporal | 37 | 0.202 | **0.255** | 0.065 |
| Open-domain | 13 | 0.161 | **0.456** | 0.091 |
| Single-hop | 70 | 0.382 | **0.445** | 0.369 |
| Adversarial | 47 | **0.723** | 0.596 | 0.000 |

### 分析

- **A-Mem 的 F1 略高于 NativeMem**（0.406 vs 0.376），但 A-Mem 回答更短（5 词 vs 25 词），F1 的 precision 天然有利
- **NativeMem 在 Adversarial 上远超其他方法**（0.723 vs 0.596 vs 0.000），因为 prompt 里加了 "No information available" 的表述
- **Mem0 在 Adversarial 上完全失败**（0.000），因为模型从不回答 "no information available"
- **NativeMem 回答长度是 gold 的 5 倍**（25 词 vs 5 词），严重影响 F1 precision，需要统一 answerer 才能公平比较

---

## 3. 构建阶段对比

| 指标 | NativeMem v3 | Mem0 | A-Mem |
|---|---|---|---|
| 构建时间 | 1,231s | 626s | ~14,400s (4h) |
| LLM 调用次数 | 480 | 19 | ~2,000+ |
| 构建 tokens | ~3.3M | 206K | 未统计 |
| 记忆规模 | 25 文件 | 20 条 | 199 条笔记 |
| 重复标题 | 0 | N/A | N/A |
| 需要 embedding | 否 | 是 (MiniLM 384d) | 是 |

---

## 4. NativeMem 版本演进

| 版本 | Overall F1 | 找到率 | 验证通过率 | 重复标题 | 文件数 | 关键改动 |
|---|---|---|---|---|---|---|
| v2 (自定义工具) | — | 59% | 65% | 10 | 50 | Python 自定义工具 |
| v2 修复后 | — | 67% | 90% | 10 | 50 | 轨迹传递 + 多轮修复 |
| **v3 (bash 工具)** | **0.376** | **86%** | **100%** | **0** | **25** | 标准 shell 命令 + 100 行拆分 |

### v3 关键改动
1. 把 6 个自定义 Python 工具替换为单一 `bash` 工具，模型直接写 shell 命令
2. Prompt 教模型用 `ls -lhR`、`grep '^#'`、`cat`、`sed` 等标准命令
3. 文件超过 100 行自动拆分为子文件
4. 写入前必须检查已有标题结构，禁止重复标题
5. 检索 prompt 里加了 cost-aware 策略（小文件直接读，大文件先看标题）

---

## 5. LongMemEval 结果 (30 题, 每 category 5 题)

| Category | LJ (0-100) | n |
|---|---|---|
| single-session-user | 100.0 | 5 |
| single-session-assistant | 80.0 | 5 |
| single-session-preference | 80.0 | 5 |
| temporal-reasoning | 80.0 | 5 |
| knowledge-update | 60.0 | 5 |
| multi-session | 40.0 | 5 |
| **Overall** | **73.3** | 30 |

注: 此结果使用 deepseek-v4-flash 自己做 judge (0-100)，非标准 0/1 binary judge。后续需要用 GPT-5.5 重新评。

---

## 6. 待完成

### 评测
1. [ ] 用 GPT-5.5 做 0/1 Judge 评分（NativeMem + Mem0 + A-Mem）
2. [ ] 统一 answerer：所有方法提取 context → 同一个 LLM 回答 → 同一个 judge 评分
3. [ ] 跑全 10 个 LoCoMo sample（目前只跑了 sample 0）
4. [ ] LongMemEval 用标准 0/1 Judge 评

### 方法优化
5. [ ] NativeMem 检索改为返回 context（而不是直接回答），支持统一 answerer
6. [ ] 写入粒度优化（提取更完整的信息）
7. [ ] 减少检索步数（当前 7.1 步，Adversarial 无效搜索 9.9 步）

---

## 7. 参考：其他论文报告的 LoCoMo 结果

详见 [benchmark_survey.md](benchmark_survey.md) 和 [experiment_settings_survey.md](experiment_settings_survey.md)。

关键参考值（LLM Judge，不同 judge 模型/不同回答模型，不可直接比较）：
- ByteRover: 96.1%
- Mnemis: 93.9%
- MemMachine: 91.7%
- Mem0 (2026 algo): 92.5%
- Mem0 (original): 66.9%

关键参考值（F1，可跨论文比较）：
- Human: 87.9
- SimpleMem (GPT-4.1-mini): 43.24
- Memory-R1 (Qwen2.5-7B): 45.0
- Mem0 (GPT-4o-mini): 38.72
- A-Mem (GPT-4o-mini): ~35
- **NativeMem v3 (deepseek-v4-flash): 37.6** ← 我们
