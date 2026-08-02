# Prompts Collection 索引

各记忆系统 / 官方评测的 prompt 与评测实现调研汇总，供横向对比。共 17 个文件、18 个项目（Letta 与 MemoryBank 合并一个文件）。

- **第一批 8 个文件**（nativemem / mem0_core / mem0_benchmarks / locomo_official / longmemeval_official / amem / nemori / lightmem）：逐字 prompt 全量汇总，共 146 个 prompt，全部为源码逐字原文（verbatim，保留 f-string 占位符与源码拼写错误）。
- **第二批 9 个文件**（2026-07-02 新增，memos / memoryos / zep / memmachine / mirix / simplemem / memory-r1 / memobase / letta-membank）：开源框架**自带评测代码**调研，统一体裁（评测脚本位置 / Answer prompt / Judge 配置 / 指标计算 / 评测范围），关键 prompt 逐字收录但不做全量 prompt 计数。

## 文件清单

### Benchmark 官方

| 文件 | 项目 | 内容 | 来源 |
|---|---|---|---|
| [locomo_official.md](locomo_official.md) | LoCoMo 官方评测 | 构建 6 + Answer 14 + 其他 4 = 24 个 prompt；无 LLM judge（规则指标：F1/EM/BERTScore/ROUGE-L） | 本地代码 `code/locomo/` |
| [longmemeval_official.md](longmemeval_official.md) | LongMemEval 官方评测 | 构建 7 + Answer 8 + Judge 5 + CoN 1 = 25 个 prompt | 本地代码 `code/longmemeval/src/` |

### 我们的

| 文件 | 项目 | 内容 | 来源 |
|---|---|---|---|
| [nativemem.md](nativemem.md) | NativeMem（我们的方法） | 构建 7 + Answer 3 + Judge 3 + 其他 1 = 14 个 prompt | 本地代码（`memory_builder_v2.py` / `eval_standard.py` / `run_judge.py` 等） |

### 开源框架自带评测

| 文件 | 项目 | 自带评测代码？ | 内容 | 来源 |
|---|---|---|---|---|
| [mem0_core.md](mem0_core.md) | Mem0 核心库 | 否（核心库不含评测） | 构建 6 + update 2 + Answer 1 + 其他 4 = 13 个 prompt | 本地代码 `code/mem0/mem0/` |
| [mem0_benchmarks.md](mem0_benchmarks.md) | Mem0 官方 benchmarks | **是**（LoCoMo / LongMemEval / BEAM） | Answer 3 + Judge 9 + 其他 2 = 14 个 prompt | 本地代码 `code/mem0-benchmarks/` |
| [amem.md](amem.md) | A-Mem | 是（LoCoMo，自动指标） | 构建 7 + Answer 12 + 其他 2 = 21 个 prompt | 本地代码 `baselines/A-Mem/` |
| [nemori.md](nemori.md) | Nemori | 是（LoCoMo / LongMemEval） | 构建 7 + Answer 2 + Judge 5 + 其他 3 = 15 个 prompt | GitHub `nemori-ai/nemori` main（2026-07-02） |
| [lightmem.md](lightmem.md) | LightMem | 是（LoCoMo / LongMemEval） | 构建 6 + Answer 8 + Judge 8 = 20 个 prompt | GitHub `zjunlp/LightMem` main（2026-07-02） |
| [memos.md](memos.md) | MemOS | **是**（LoCoMo / LongMemEval / PrefEval / PersonaMem / LongBench-v2 全流程） | 自带评测调研；README 的 LoCoMo 75.80 出自该 pipeline | GitHub `MemTensor/MemOS` main @ `a4f1b5be` |
| [memoryos.md](memoryos.md) | MemoryOS (EMNLP 2025) | 是（仅 LoCoMo，无 judge，仅词级 F1） | 自带评测调研；README 宣称的 BLEU-1 无对应代码 | GitHub `BAI-LAB/MemoryOS` main（2026-07-02） |
| [zep.md](zep.md) | Zep / Graphiti | **是**（LoCoMo / LongMemEval / DMR + 图构建质量） | 自带评测调研；仓库内含已提交的 10-run LoCoMo 实验结果（accuracy≈0.803） | GitHub `getzep/zep` + `getzep/graphiti`（2026-07-02） |
| [memmachine.md](memmachine.md) | MemMachine | **是**（LoCoMo；main 加 LongMemEval/HotpotQA/2Wiki/BEAM） | 自带评测调研；逐文件注明 "adapted from Mem0"；博客 91.7 = agent 模式 | GitHub `MemMachine/MemMachine` v0.2.x + main |
| [mirix.md](mirix.md) | MIRIX | **是**（两代：论文版 + 新版 MemoryAgentBench） | 自带评测调研；论文 85.4% 可由 public_evaluation 分支三个 run json 复算 | GitHub `Mirix-AI/MIRIX` main + `public_evaluation` 分支 |
| [simplemem.md](simplemem.md) | SimpleMem | 是（三代各一套：主 pipeline / EvolveMem / Omni） | 自带评测调研；论文 F1=43.24 是词面 F1，judge 代码默认关闭 | GitHub `aiming-lab/SimpleMem` @ `60a48e83` |
| [memory-r1.md](memory-r1.md) | Memory-R1 | **否**（"Code coming soon"，仓库只有 README+图） | prompt 取自论文 arXiv:2508.19828 附录 Figures 9–12 | GitHub `yansikuan/memory-r1` + 论文附录 |
| [memobase.md](memobase.md) | Memobase | 是（LoCoMo，整体 fork 自 mem0 evaluation） | 自带评测调研；含官方两版结果 fixture | GitHub `memodb-io/memobase` main（2026-07-02） |
| [letta-membank.md](letta-membank.md) | Letta (MemGPT) + MemoryBank | **均否**（Letta 历史 doc-QA/nested-KV 代码已删；MemoryBank 只有数据无打分代码） | 历史 prompt 逐字收录（Letta @ commit `5511a080`） | 本地 clone `baselines/MemGPT/`、`baselines/MemoryBank/` |

## 大对比表：自带评测 × answer prompt 约束 × judge 形式

一行一个项目。"字数限制"特指 "less than 5-6 words" 这类硬性词数上限。

| 项目 | 自带评测代码 | 评测 benchmark | Answer prompt 字数限制 | Judge 模型 | Judge 评分制 | 日期容差 | Adversarial (LoCoMo cat5) |
|---|---|---|---|---|---|---|---|
| **LoCoMo 官方** | —（benchmark 本身） | LoCoMo | **强**："short answers ... in a few words"、"exact words from the conversations" | 无 LLM judge | 规则指标：multiset token-F1（stem+normalize）/EM/BERTScore/ROUGE-L | 无 | **评**：关键词匹配（含 "no information available"/"not mentioned" 即对），计入 overall |
| **LongMemEval 官方** | —（benchmark 本身） | LongMemEval | 无 | gpt-4o-2024-08-06（有 assert） | yes/no 0/1（5 个分题型模板） | temporal 题天数 off-by-one 容忍 | —（30 个 `_abs` abstention 题单独另报） |
| **NativeMem（我们）** | 是（本地） | LoCoMo / LongMemEval | 偏简短无上限："Answer concisely" / "as briefly as possible" | GPT-5.5（proxy） | 双轨：CORRECT/WRONG 0/1 + run_judge 0-100 连续（0/25/50/75/100 锚点） | **显式**：日期 14 天内、时长误差 50% 内算对 | 主表不评（1540 题口径） |
| **Mem0 核心库** | 否 | — | "clear, concise" 无限制 | 无 judge | — | — | — |
| **Mem0 benchmarks** | 是 | LoCoMo / LongMemEval / BEAM | LoCoMo 7 步 prompt **无**字数限制；LME "Be direct and concise"；（注：老 `mem0ai/mem0` 的 `evaluation/` 目录 ANSWER_PROMPT 才含 "less than 5-6 words"，是各家 fork 的源头） | 默认 gpt-5 | LoCoMo CORRECT/WRONG 0/1；LME yes/no 0/1；BEAM nugget 0/0.5/1 三档 | LoCoMo 14 天 + 时长 50%；LME off-by-one 泛化到所有题型 | 跳过（1540 题） |
| **A-Mem** | 是 | LoCoMo | **强**："Short answer:" / "shortest possible answer" | 无 LLM judge | 12 个自动指标（set-based F1、BLEU、ROUGE、BERTScore、METEOR、SBERT） | 无 | **评**：参考=**adversarial_answer**，二选一 prompt（选中对抗答案得分） |
| **Nemori** | 是 | LoCoMo / LongMemEval | **有**："less than 5-6 words"（两个 benchmark 的 answer prompt 均含） | LoCoMo gpt-4.1-mini（v4；v1 为 gpt-4o-mini）；LME 官方 4 模板 | 0/1 | LoCoMo prompt 宽判无数值窗口；LME off-by-one | 跳过 |
| **LightMem** | 是 | LoCoMo / LongMemEval | LoCoMo **有** "less than 5-6 words"（全部变体）；LME 通用模板无 | gpt-4o-mini | 0/1（LME 用官方 5 模板 + exact-match） | LoCoMo 宽判无窗口；LME off-by-one | 跳过 |
| **MemOS** | 是 | LoCoMo / LongMemEval / PrefEval / PersonaMem / LongBench-v2 | LoCoMo **有**："brief (under 5-6 words)"（Mem0 改写版，但多"可用世界知识"+"跨条目综合"两条放松）；LME 无 | gpt-4o-mini（EVAL_MODEL 默认；LME 硬编码） | 0/1；LoCoMo 判 3 runs 取均值±std，LME 1 run；LME judge 是通用模板**非官方分题型模板** | 无数值窗口，prompt 宽判 | 跳过（responses 阶段即过滤） |
| **MemoryOS** | 是（仅 LoCoMo） | LoCoMo | 无 "5-6 words"，但两次 "extremely concise" + 实体短语 one-shot（示例疑似取自 LoCoMo 数据本身） | **无 judge** | 唯一指标：set-based 词级 F1（逐 category 平均）；README 的 BLEU-1 无代码 | 无（但 prompt 强制 "15 July 2023" 日期格式，为词面 F1 优化） | **评**：cat5 用 adversarial_answer 当参考算 F1 |
| **Zep / Graphiti** | 是 | LoCoMo / LongMemEval / DMR(MSC)；Graphiti 另评图构建质量 | **无**（最强约束仅 "briefly"/"concise"） | LoCoMo gpt-4o-mini；LME gpt-4o；Test Harness gpt-4.1-mini | 0/1（LoCoMo judge 明文 "be generous"）；另有 context completeness 三档（COMPLETE/PARTIAL/INSUFFICIENT） | LoCoMo 同日/同时段即对无窗口；LME temporal off-by-one（官方规则） | 跳过 |
| **MemMachine** | 是（adapted from Mem0，逐文件注明） | LoCoMo（v0.2）；main 加 LongMemEval / HotpotQA / 2Wiki / BEAM | memory 模式**有**："less than 5-6 words"（Mem0 原文）；agent 模式（博客 91.7）无硬限（提示 gold <6 词但 "yours may be longer"） | gpt-4o-mini（LoCoMo）；gpt-4o（LME 官方 anscheck） | 0/1，只报 judge 均值（无 F1/BLEU） | 宽判无窗口；LME off-by-one | 跳过（三处代码均跳） |
| **MIRIX** | 是（两代） | LoCoMo + ScreenshotVQA（论文版）；LoCoMo + MemoryAgentBench（LME-S/RULER/LRU，新版） | **无** "5-6 words"："as brief as possible / only state the answer"（论文版）；新版 "Be VERY CONCISE" + max_completion_tokens=128；"5-6 words" 只在其 vendored Mem0 baseline 里 | gpt-4o-mini（LoCoMo，Mem0 judge 同文）；gpt-4o（MAB）；gpt-4o-2024-05-13（摘要） | 0/1；摘要 judge 为连续 F1（fluency×2RP/(R+P)）；论文版另报 BLEU-1 + set-based F1 | 宽判无窗口；MAB temporal off-by-one | 排除（两代汇总均剔） |
| **SimpleMem** | 是（三代各一套） | LoCoMo（主）；LongMemEval / MemBench（EvolveMem）；Mem-Gallery（Omni） | **无**硬限："very CONCISE answer (short phrase)" + 日期 'DD Month YYYY'；EvolveMem 层有明确 1-5/1-10/1-15 词限制并显式拟合 gold 格式 | judge 代码存在但**默认关闭**（config 自注 "not used yet"；示例 gpt-4.1-mini，t=0.3） | 主指标 = set-based 词面 F1（论文 43.24 即此，非 judge 分）；judge 启用时 0/1 | judge prompt 明文 ±1-2 天（未启用）；F1 无容差 | **评且计入 overall**：改为有提示二选一（gold 固定 "Not mentioned..."）；Omni 版硬编码弃答句恒满分；EvolveMem 版 gold=adversarial_answer |
| **Memory-R1** | **否**（官方仓库无任何代码） | LoCoMo（主）；MSC / LongMemEval 零样本（均论文自报） | **有**："The answer should be less than 5-6 words."（论文 Fig 11）；且 RL reward 用 EM 直接强化短答案 | **未披露** | 0/1（论文 Fig 12，Mem0 系宽松 judge）；另报 token-F1 + BLEU-1 | 宽判（同时段即对、格式不罚） | 剔除（152/81/1307 切分，测试 1307 题） |
| **Memobase** | 是（整体 fork 自 mem0 evaluation） | LoCoMo | **有**："less than 5-6 words"（prompts.py，与 mem0 相同，三个变体均含） | gpt-4o-mini（硬编码；README 说 gpt-4o，不一致） | 0/1；另算 set-based F1 + BLEU-1 | 宽判无窗口 | 跳过（三层代码跳过） |
| **Letta (MemGPT)** | **否**（main 无；历史 paper_experiments 已删） | —（历史：NQ doc-QA / nested-KV，非 LoCoMo/LME） | 无（历史 prompt 要求附证据文档） | 历史：gpt-4-0613 | 二值 CORRECT/INCORRECT（substring 先行、两级判分） | 无 | —（无 category 概念） |
| **MemoryBank** | **否**（只有评测数据 + 从未被调用的 answer prompt） | 自造 15 用户 ×100 probing questions | 无（反而要求 "provide detailed answers"） | 无（论文分数应为人工评） | — | — | — |

## Mem0 evaluation 谱系注记

老 `mem0ai/mem0` 仓库的 `evaluation/` 目录（ANSWER_PROMPT 含 "less than 5-6 words" + gpt-4o-mini `ACCURACY_PROMPT` 宽松 0/1 judge + 跳过 cat5）是**事实上的行业标准 harness**，被以下项目直接继承：

- **整体 fork / vendored**：Memobase（README 自述 fork 自 mem0 commit `393a4fd`）、MIRIX（`baselines/mem0/evaluation/` 整目录拷贝，judge 与 Mem0 逐字同文）、MemMachine（逐文件注明 "adapted from Mem0"）
- **同款 prompt**：Nemori、LightMem（answer + judge 均 Mem0 同款）、Memory-R1（论文自述 adapted from Chhikara et al. 2025）
- **改写**：MemOS（`ANSWER_PROMPT_MEMOS` 加两条放松、保留 5-6 words；judge 同源）、MIRIX 自家 answer prompt（去掉 5-6 words 换成 "only state the answer"）

## 覆盖情况（缺失标注）

- **第一批 8 文件**：均声明逐字全量收录，未发现"没抓到"的 prompt。表中标"无"的类别是源项目本身不存在该类 prompt，不是提取遗漏。
- **有意排除**（lightmem.md 已注明）：`baselines/` 下 langmem、mem0 的第三方 prompt 副本未收录；em2mem 多模态子系统 prompt 未收录。
- **逐字重复去重**：lightmem 的 `LoCoMo_Event_Binding_factual` 与核心版逐字相同只引用；mem0_benchmarks 的 BEAM f-string 与常量相同不重复列出。
- **非评测用 prompt**：locomo_official 的 4 个数据集构建 prompt、amem 的死代码 prompt 也一并收录并标注状态。
- **第二批 9 文件的特殊情况**：
  - memory-r1：官方仓库零代码，prompt 全部取自论文 arXiv:2508.19828 附录（Figures 9–12），judge 模型论文未披露。
  - letta-membank：Letta 历史 doc-QA/nested-KV 代码在 PR #2929 删除，最后可见 commit `5511a080`，已逐字收录；MemGPT 论文 DMR 代码从未随仓库发布。MemoryBank 的 eval answer prompt 在仓库中无任何调用者。
  - mirix：论文评测代码在 `public_evaluation` 分支（main 已删该目录），需 `git fetch origin public_evaluation`。
  - memoryos：README/论文宣称的 BLEU-1 在仓库无对应代码，不可复现。
  - simplemem：一个 repo 含三代系统三套评测，judge 配置自注 "not used yet"。

## 相关文档

- 统一评测协议（我们怎么跑）：[`../experiments/protocols/unified_evaluation_protocol.md`](../experiments/protocols/unified_evaluation_protocol.md)
- 可比性分析与代码复用决策：[`../related-work/evidence/framework_eval_methods.md`](../related-work/evidence/framework_eval_methods.md)
