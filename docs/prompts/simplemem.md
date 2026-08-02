# SimpleMem

- 仓库：https://github.com/aiming-lab/SimpleMem（aiming-lab，调研时 HEAD = `60a48e83`，2026-06-23）
- 论文：arXiv 2601.02553 "SimpleMem: Efficient Lifelong Memory for LLM Agents"
- 论文主表：GPT-4.1-mini backbone，LoCoMo overall **F1 = 43.24**（第二名 Mem0 34.20，即 README 里的 "+26.4%"）
- 仓库结构特殊：一个 repo 里装了三代系统，各有独立评测代码
  - 主 pipeline（论文 SimpleMem 文本记忆）：根目录 `test_locomo10.py`
  - Omni-SimpleMem（多模态 v2.0，README 称 LoCoMo F1=0.613）：`OmniSimpleMem/benchmarks/locomo/run_locomo.py`
  - EvolveMem（自进化 v3.0，README 称 +25.7% relative）：`EvolveMem/`（同代码复制在 `simplemem/evolver/`）

## 评测脚本位置

| 路径 | 作用 |
|---|---|
| `test_locomo10.py`（根目录，1075 行） | 论文主 LoCoMo 评测：加载数据、F1/BLEU/ROUGE/BERTScore/METEOR/SBERT、可选 LLM judge、cat5 特殊处理 |
| `simplemem/core/answer_generator.py` | 主 pipeline 的 answer prompt（`_build_answer_prompt`） |
| `test_ref/load_dataset.py`, `test_ref/utils.py`, `test_ref/test_advanced.py` | **A-Mem 的评测代码副本**（import `from memory_layer import AgenticMemorySystem`），作为对照参考；metrics 与主脚本相同 |
| `simplemem/evolver/benchmarks/{locomo,longmemeval,membench,metrics,base}.py`（= `EvolveMem/evolvemem/benchmarks/`） | EvolveMem 的 LoCoMo / LongMemEval / MemBench 适配器 + 指标 |
| `EvolveMem/run_benchmark.py`, `EvolveMem/run_evolution.py` | EvolveMem 运行入口 |
| `OmniSimpleMem/benchmarks/locomo/run_locomo.py` | 多模态版 LoCoMo 评测（官方 task_eval 风格 F1） |
| `OmniSimpleMem/benchmarks/memgallery/` | Mem-Gallery 评测（含 10 个 prompt txt） |
| `simplemem/multimodal/evaluation/{metrics,evaluator,benchmarks}.py` | 多模态评测的包内版本 |

## Answer prompt（原文）

### 1. 主 pipeline（论文 F1=43.24 用的）—— `simplemem/core/answer_generator.py` `_build_answer_prompt()`

**没有 "5-6 words" 式的硬字数限制**，只要求 "very CONCISE answer (short phrase)"，外加日期格式约束 `DD Month YYYY`。system prompt：`"You are a professional Q&A assistant. Extract concise answers from context. You must output valid JSON format."`，temperature=0.1，JSON 输出：

```
Answer the user's question based on the provided context.

User Question: {query}

Relevant Context:
{context_str}

Requirements:
1. First, think through the reasoning process
2. Then provide a very CONCISE answer (short phrase about core information)
3. Answer must be based ONLY on the provided context
4. All dates in the response must be formatted as 'DD Month YYYY' but you can output more or less details if needed
5. Return your response in JSON format

Output Format:
```json
{{
  "reasoning": "Brief explanation of your thought process",
  "answer": "Concise answer in a short phrase"
}}
```

Example:
Question: "When will they meet?"
Context: "Alice suggested meeting Bob at 2025-11-16T14:00:00..."

Output:
```json
{{
  "reasoning": "The context explicitly states the meeting time as 2025-11-16T14:00:00",
  "answer": "16 November 2025 at 2:00 PM"
}}
```

Now answer the question. Return ONLY the JSON, no other text.
```

### 2. Category 5（adversarial）专用 prompt —— `test_locomo10.py` `generate_category5_answer()`（约 L663）

**cat5 不做自由问答，改成二选一选择题**：把 adversarial_answer 和 "Not mentioned in the conversation" 随机排序给模型选；同时检索关 reflection（`enable_reflection=False`），temperature=0.5：

```
Based on the context below, answer the following question.

Context:
{context_str}

Question: {question}

Select the correct answer from the following two options. If the given answer is wrong or not answerable based on the context, you should choose "Not mentioned in the conversation".

Option A: {options[0]}
Option B: {options[1]}

Requirements:
1. Choose the option that best matches the context
2. If neither answer is supported by the context, or if the provided specific answer is incorrect, choose "Not mentioned in the conversation"
3. Return your response in JSON format
...
Return ONLY the JSON, no other text.
```

失败 3 次后默认返回 `"Not mentioned in the conversation"`（即默认正确答案）。

### 3. EvolveMem 的 per-category prompt —— `simplemem/evolver/benchmarks/locomo.py` `build_answer_prompt()`

这一层是**显式按 LoCoMo gold 答案格式做 prompt 拟合**（自进化的"动作"之一），字数限制明确（1-5 / 1-10 / 1-12 词不等）。例子：

- Cat1 counting：`"LoCoMo gold counts are SPELLED-OUT words for small counts. You MUST match this format. 1 -> 'once' / 2 -> 'twice' / 3 -> 'three times' ... If the count from context is ambiguous, pick the smallest plausible count (LoCoMo gold counts skew low: once/twice are typical)."`
- Cat3 yes/no：`"Many LoCoMo gold answers for yes/no are literally just 'yes' or 'Yes', so adding a clause HURTS."`
- Cat5（`locomo_cat5_mcq` flag 开时）：`"One option is the swap trap ('Not mentioned in the conversation') and the other is a concrete factual claim... When in doubt, ALWAYS prefer the concrete option."` —— 注意：EvolveMem 加载数据时 `ref = qa.get("answer") or qa.get("adversarial_answer", "")`（locomo.py L72），cat5 没有 answer 字段，所以 **reference 变成 adversarial_answer 本身**，prompt 再教模型选具体选项——与原始 LoCoMo（cat5 gold=弃答）方向相反。
- 默认（无 flag）：`"1. Answer in 1-10 words. Use EXACT words from context. 'how many' -> number; 'when' -> date; 'where' -> place. 2. Be specific, not vague."`
- LongMemEval（`longmemeval.py`）：`"Answer in 1-15 words. Use EXACT words/phrases from context."` + 按 qtype 的提示（multi-session → "output ONLY the final integer"；knowledge-update → "use the LATEST stated fact"），且 `"If the context genuinely lacks the answer, still give a best inference — do not say 'not mentioned'."`

### 4. Omni-SimpleMem —— `OmniSimpleMem/omni_memory/orchestrator.py` L986

`"Provide a CONCISE answer (short phrase, ideally under 10 words). Use exact words and phrases from the context"` + `"For counting questions, answer with just the number (e.g., '2' not 'twice')"` + `"All dates ... 'DD Month YYYY'. NEVER use dates from 2025 or 2026."`。runner 里 cat2 问题追加 `" Use the dates in the conversation to answer with an approximate date."`（`run_locomo.py` L194），cat2 预测出现 2025-2029 年份则强改 'unknown'（L520）。

## Judge 配置

### 主 pipeline judge（`test_locomo10.py` `llm_judge_answers()`，L366-490）

- **默认关闭**，需 `--llm-judge` flag；`config.py.example` L113 明确标注该节 `"LLM-as-Judge Configuration (not used yet)"` → **论文的 43.24 是词面 F1，不是 judge 分**
- judge 模型：`JUDGE_MODEL`（示例默认 `gpt-4.1-mini`），未配置则回落主 `LLM_MODEL`；`JUDGE_TEMPERATURE = 0.3`
- **0/1 二值分**（score 1.0 pass / 0.0 fail），JSON 输出带 reasoning
- **日期容差：judge prompt 里明文 "+/- 1-2 days"**，另接受粒度差异（"Afternoon" vs "14:05"）、答案子集、同义词

当前生效的 judge prompt 原文：

```
You are an expert Relevance & Accuracy Evaluator. Your task is to determine if the Predicted Answer successfully retrieves the necessary information to answer the Question, based on the Reference Answer.

Question: {question}
Reference Answer: {reference}
Predicted Answer: {prediction}

Evaluation Criteria:

1. **Responsiveness to Query**: 
   The predicted answer must directly address the specific question asked. It must contain highly relevant information that is topically aligned with the user's intent.

2. **Core Fact Preservation**: 
   The prediction must capture the "Key Signal" or "Core Entity" from the reference. The primary subject (Who), event (What), or outcome must be factually grounded in the reference text.

3. **Informational Utility**: 
   The answer must provide actionable or meaningful value. Even if brief, it must convey the essential message required by the question context.

4. **Acceptable Representational Variances (Robustness Protocol)**:
   To ensure fair evaluation of semantic meaning over syntactic rigidity, you must accept the following variations as **Valid Matches**:
   - **Temporal & Numerical Margins**: Accept timestamps within a reasonable proximity (e.g., +/- 1-2 days due to timezone/reporting differences) and rounded numerical approximations.
   - **Granularity Independence**: Accept answers at different levels of abstraction (e.g., "Afternoon" vs. "14:05", "Late October" vs. "Oct 25th") provided they encompass the truth.
   - **Information Subsetting**: A valid subset of the reference (e.g., mentioning 1 out of 3 reasons) is acceptable if it answers the core of the question.
   - **Synonymy**: Recognize domain-specific synonyms and different formats as equivalent.

Grading Logic:
- Score 1.0 (Pass): The prediction contains relevant core information, answers the question with sufficient utility, OR falls within the acceptable representational variances defined in criterion #4.
- Score 0.0 (Fail): The prediction contains NO relevant information, fails to identify the core subject/event, or provides no key info that matches the question's intent.

Output your evaluation in JSON format:
{{
  "score": 1.0, 
  "reasoning": "Brief assessment focusing on information relevance and core match."
}}

Return ONLY the JSON, no other text.
```

文件里还留着一版被注释掉的更宽松的旧 judge prompt（L376-408，`"being generous in your evaluation"`、`"If reference says '2 PM' and prediction says 'afternoon' → Accept"`、`"Only score 0.0 if the predicted answer is clearly wrong"`）。

### EvolveMem judge（`simplemem/evolver/benchmarks/metrics.py` L186，可选，默认不启用）

严格版 YES/NO，temperature 0.0，max_tokens 8：

```
You are a strict grader. Given a question, a gold answer, and a predicted answer, decide if the prediction is correct.

Question: {question}
Gold answer: {reference}
Predicted answer: {prediction}

Return ONLY one token: YES if the prediction conveys the same factual content as the gold answer (minor paraphrase ok), otherwise NO.
```

LongMemEval adapter 注释承认：`"LongMemEval paper uses GPT-4-judge; we implement both — default F1 for cheap iteration"` —— 即他们的 LongMemEval 数默认也是 F1 而非官方 judge 协议。

## 指标计算

### 主 pipeline（`test_locomo10.py` `calculate_metrics()`，L492；`test_ref/utils.py` 同款）

- **F1 是 set-based token overlap**：`simple_tokenize`（lower + 去 .,!? + split）后 `set(pred) & set(ref)`，precision/recall 用**去重集合**大小 → 与 SQuAD/LoCoMo 官方的 multiset F1 不同（重复词只算一次；无冠词剔除、无 stemming）。`simplemem/evolver/benchmarks/metrics.py` L29 自证：`"Token-overlap F1 (set-based, matches SimpleMem paper)"`
- 其余指标：exact_match（全串小写相等）、ROUGE-1/2/L（rouge_scorer, stemmer on）、BLEU-1..4（nltk, smoothing method1）、BERTScore、METEOR、SBERT cosine（all-MiniLM-L6-v2）
- **Category 5 计入 overall 平均**：cat5 reference 被强制改为 `"Not mentioned in the conversation"`（L867），模型做二选一后若选对，与 reference 逐词一致 → F1=1.0。也就是说 cat5 被简化成有提示的二分类，得分直接抬高 overall F1
- 聚合：overall + per-category mean/std/median（`aggregate_metrics()` L558）

### EvolveMem（`simplemem/evolver/benchmarks/base.py` `token_f1` L163）

- multiset F1，但 tokenize 时做**数字词归一化**：`once→1, twice→2, thrice→3, first→1...`（`_NUMBER_WORDS` L141），让 "twice" 和 "2" 也能 match —— 又一处对 LoCoMo gold 格式的适配
- `metrics.py` 提供 f1_multiset（主指标）+ f1_set + EM + contains + ROUGE-L + BLEU-1/4，BERTScore/METEOR/SBERT 是 opt-in

### Omni-SimpleMem（`OmniSimpleMem/benchmarks/locomo/run_locomo.py` L60-156）

- 唯一一处采用 **LoCoMo 官方 task_eval 风格 F1**：normalize（去冠词/标点）+ Porter stemming + multiset F1；cat1 comma-split multi-answer F1；cat3 取分号前第一个答案
- **cat5 完全不评模型输出**：runner 里 category==5 时预测被**硬编码**为 `"The information is not mentioned in the provided memories."`（`run_locomo.py` L509），而 cat5 评分是 refusal 短语匹配（含 "not mentioned"）→ **cat5 恒 1.0 满分**，且计入 overall F1。README 宣称的 LoCoMo F1=0.613 含这部分白送分

## 评测范围

- **LoCoMo**（locomo10.json，10 conversations / ~1986 QA）：主脚本默认全量跑（`--num-samples` 可截取前 N 个会话）；cat1-5 全评（cat5 用上述特殊协议）
- **LongMemEval**：仅 EvolveMem adapter（oracle / s / m 三个 split，500 题；`--max-samples` 支持分层抽样），默认 token F1 评分而非官方 GPT-4 judge
- **MemBench**：EvolveMem adapter（`membench.py`，按 agent × category 子集选取）
- **Mem-Gallery**：仅 Omni-SimpleMem（多模态）
- **Baseline 对比不在仓库内**：repo 里没有 Mem0/Zep 等 baseline 的运行代码，唯一例外是 `test_ref/test_advanced.py` —— 它是 **A-Mem 的评测脚本副本**（import `AgenticMemorySystem`，cat5 二选一、cat2 "shortest possible answer" 等 prompt 与 A-Mem/LoCoMo 官方风格一致），说明 SimpleMem 的评测框架是从 A-Mem 的 harness 改出来的
- 论文（arXiv 2601.02553）对比的 baseline：LoCoMo(原方法)、ReadAgent、MemoryBank、MemGPT、A-Mem、LightMem、Mem0；backbone 覆盖 GPT-4o / GPT-4.1-mini / Qwen-Plus / Qwen2.5-1.5B/3B / Qwen3-1.7B/8B；指标 F1 + BLEU-1 + Adversarial Success Rate + Token Cost；主表按 cat1-4 分列，cat5 以 "Adversarial Success Rate" 单独报

### 关键结论（对我们复现对比的含义）

1. 论文 F1=43.24 是 **set-based（去重）token F1**，不是 SQuAD multiset F1，也不是 judge 分；与其他框架论文的 "LLM-as-a-Judge (J)" 完全不可比
2. Answer prompt **无 "5-6 words" 限制**（对比 Mem0 论文式短答约束），但要求 short phrase + 精确日期格式，实际输出同样是短答风格
3. cat5 三个版本三种玩法：主脚本=有提示二选一（reference 固定弃答句）、Omni=硬编码弃答句恒满分、EvolveMem=reference 干脆用 adversarial_answer 并教模型选它——三者都显著偏离 LoCoMo 官方 cat5 协议（自由生成+refusal 检测）
4. LLM judge 代码存在但默认关闭、config 自注 "not used yet"；启用时是宽松 0/1 判分（±1-2 天日期容差、答案子集算对）
