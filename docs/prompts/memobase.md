# Memobase（memodb-io/memobase）自带 benchmark 评测实现调研

调研日期：2026-07-02。仓库：https://github.com/memodb-io/memobase（默认分支 `main`）。

**结论概要**：仓库自带 LoCoMo 评测，位于 `docs/experiments/locomo-benchmark/`，README 第一行自述 **fork 自 mem0 官方 evaluation 目录**（commit `393a4fd5a6cfeb754857a2229726f567a9fadf36`），因此 answer prompt、judge prompt、指标代码与 mem0 evaluation 基本一致（含 **"less than 5-6 words" 字数限制**）。没有 LongMemEval 评测。

## 评测脚本位置

- `docs/experiments/locomo-benchmark/` — LoCoMo 评测（核心）
  - `run_experiments.py` — 统一入口，`--technique_type {mem0, rag, langmem, zep, openai, memobase} --method {add, search}`
  - `prompts.py` — answer prompt（3 个变体：`ANSWER_PROMPT`、`ANSWER_PROMPT_GRAPH`、`ANSWER_PROMPT_ZEP`）
  - `evals.py` — 对 results.json 逐题算 BLEU/F1/LLM judge
  - `generate_scores.py` — 按 category 汇总均分
  - `metrics/llm_judge.py` — judge prompt + judge 调用
  - `metrics/utils.py` — F1/BLEU/ROUGE 等（注释标明借自 A-Mem `WujiangXu/AgenticMemory/utils.py`）
  - `src/memobase_client/{memobase_add.py, memobase_search.py, config.yaml}` — Memobase 自家的记忆写入/检索适配
  - `src/{memzero, zep, openai, rag.py, langmem.py}` — 各 baseline 适配（从 mem0 fork 来）
  - `fixture/memobase/` — 官方跑出的结果工件（v0.0.32 与 v0.0.37 两版，各含预测 answers 与 judge 结果 json）
  - `Makefile`、`compute_p95_latency.py`
- `docs/experiments/900-chats/` — 非 QA benchmark：用 ShareGPT 900 轮对话对比 Memobase vs mem0 的**插入成本/耗时**（gpt-4o-mini，Memobase $0.042/270-300s vs mem0 $0.24/1683s）
- `docs/experiments/chat_sessions/` — 小型 mock 抽取示例，非 benchmark

## Answer prompt（原文）

`docs/experiments/locomo-benchmark/prompts.py`，Memobase 用的是 `ANSWER_PROMPT`（`src/memobase_client/memobase_search.py` 第 44 行 `self.ANSWER_PROMPT = ANSWER_PROMPT`）。**第 8 条指令有字数限制："The answer should be less than 5-6 words."** 与 mem0 evaluation 原文相同：

```python
ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Question: {{question}}

    Answer:
    """
```

另有 `ANSWER_PROMPT_GRAPH`（mem0-graph 用，多了 knowledge graph relations 段和第 9 条指令）和 `ANSWER_PROMPT_ZEP`（单一 memories 段），三者都含同一条 "less than 5-6 words"。

Answer 生成配置（`src/memobase_client/memobase_search.py`）：
- 模型 `os.getenv("MODEL", "gpt-4o")`，`temperature=0.0`，prompt 以 **system** 消息发送
- Memobase 检索：`user.context(max_token_size=3000, chats=[{"role":"user","content":query}], event_similarity_threshold=0.2, fill_window_with_events=True)`，即上下文预算 3000 tokens（fixture 文件名里的 `_3000` 即此）
- 两个 speaker 各建一个 Memobase user，各检索一次 context，填入模板

## Judge 配置（原文）

`docs/experiments/locomo-benchmark/metrics/llm_judge.py`。**judge 模型硬编码 `gpt-4o-mini`**，`temperature=0.0`，`response_format={"type":"json_object"}`，**0/1 二值**（CORRECT=1 / WRONG=0），无连续分。日期无数值容差窗口，只有 prompt 内的宽松措辞（同日期不同格式算对、相对时间指同一时段算对）：

```python
ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it’s time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""
```

```python
def evaluate_llm_judge(question, gold_answer, generated_answer):
    """Evaluate the generated answer against the gold answer using an LLM judge."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[...],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    label = json.loads(response.choices[0].message.content)["label"]
    return 1 if label == "CORRECT" else 0
```

注意：README 的解释文字说 judge 用 "LLM(e.g. OpenAI gpt-4o)"，但代码实际硬编码 `gpt-4o-mini`，两处不一致。

## 指标计算

- 入口 `evals.py::process_item`：逐题算三个分数并存 json
  - `f1_score` = `metrics/utils.py::calculate_metrics` 的 token-level F1：`simple_tokenize`（小写 + 去 .,!? + split）后取**集合**交集算 precision/recall（set-based，重复 token 不计多次）
  - `bleu_score` = `metrics/utils.py::calculate_bleu_scores` 的 **BLEU-1**（nltk `sentence_bleu`，SmoothingFunction().method1；bleu2-4 也算但只报 bleu1）
  - `llm_score` = 上述 judge 的 0/1
  - ROUGE/BERTScore/METEOR/SBERT 代码存在但被注释禁用
- 汇总 `generate_scores.py`：pandas 按 category groupby 取均值，overall 是**全题直接平均**（不是各类均值再平均）
- **Category 5 (adversarial) 不评**：三层跳过 —— `memobase_search.py::process_data_file(exclude_category={5})` 答题阶段就排除；`evals.py` 里 `if category == "5": continue`；`metrics/llm_judge.py::main` 里 `# Skip category 5`
- category → 名称映射（`generate_scores.py`）：`1=single_hop, 2=temporal, 3=multi_hop, 4=open_domain`。注意这是沿用 mem0 fork 的映射，与 LoCoMo 官方论文的类别编号命名不一致（LoCoMo 原始定义中 1=multi-hop、4=single-hop），跨论文对数字时要小心

## 评测范围

- **Benchmark**：只有 LoCoMo（`locomo10.json`，10 组长对话）。无 LongMemEval、无其他 QA benchmark
- **Sample**：locomo10 全部 10 组对话，排除 category 5 后共 1540 题（fixture 里 v0.0.37 计数：single_hop 282 / temporal 321 / multi_hop 96 / open_domain 841）
- **对比 baseline**：README 表格里 Mem0、Mem0-Graph、LangMem、Zep、OpenAI（full-context/naive LLM）—— 但这些数字**直接摘自 mem0 论文**（arXiv 2504.19413），Memobase 只实际跑了自己（框架代码里 baseline 适配脚本都在，但官方没重跑）。Memobase 自报两版：v0.0.32 overall LLM Judge 70.91%，v0.0.37 overall 75.78%（Temporal 85.05% 最突出）。后来 Zep 团队在 issue #101 提交了更高的 Zep* 数字（overall 75.14），README 附注收录
- **产物**：`fixture/memobase/` 下有两版完整预测与 judge 结果 json，可用 `generate_scores.py` 复现表格数字
- 另有 `compute_p95_latency.py` 算检索 p95 延迟；`docs/experiments/900-chats/` 是与 mem0 的插入成本/速度对比（非 QA 精度）

**来源可信度提示**：该评测 harness 整体 fork 自 mem0 evaluation，answer prompt（含 5-6 words 限制）、judge prompt、跳过 category 5、指标实现都是 mem0 的做法；Memobase 自己的贡献是 `src/memobase_client/` 适配层与自家结果，且拿自己实测数字对比 mem0 论文里的静态数字。
