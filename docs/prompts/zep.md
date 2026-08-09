# Zep / Graphiti（getzep/zep + getzep/graphiti）

调研日期：2026-07-02。Zep 是商业记忆服务（zep-cloud SDK 调 API），开源部分主要是 SDK/示例仓库 `getzep/zep` 和开源图引擎 `getzep/graphiti`。**两个仓库都自带评测代码**。

## 评测脚本位置

### getzep/zep（主评测仓库，`benchmarks/` 目录）

| 路径 | 内容 |
|---|---|
| `benchmarks/README.md` | 指向论文 "Zep: A Temporal Knowledge Graph Architecture for Agent Memory"（blog + arXiv），说代码在 `kg_architecture_agent_memory`（已改名为 `longmemeval/`，README 未更新） |
| `benchmarks/locomo/` | 完整 LoCoMo 评测 harness：`benchmark.py`（CLI 入口）、`prompts.py`（全部 prompt）、`evaluation.py`（评测管线）、`ingestion.py`、`config.py` + `benchmark_config.yaml`、`common.py`（Pydantic 结果模型）、`persistence.py`、`tests/` |
| `benchmarks/locomo/experiments/` | **提交在仓库里的 5 组实验结果**（2025-12-03 ~ 2025-12-07，每组 10 个 run 的完整 JSON） |
| `benchmarks/longmemeval/zep_longmem_eval.py` | LongMemEval 评测脚本（论文用，ingest + eval + baseline 三合一） |
| `benchmarks/longmemeval/zep_memgpt_eval.ipynb` | DMR（MemGPT 论文的 Deep Memory Retrieval，基于 MSC）评测 notebook |
| `benchmarks/longmemeval/Zep Test Harness/zep_responses.py` + `zep_eval.py` | 另一套 LoCoMo 风格 harness（读 `test_data.json`，跳过 category 5，user 前缀 `rivian_experiment_user_`），推测是回应 Mem0 LoCoMo 对比争议时用的复评脚本 |

### getzep/graphiti

| 路径 | 内容 |
|---|---|
| `graphiti_core/prompts/eval.py` | QA prompt、judge prompt、query expansion、图构建质量对比 judge（都是库内置 prompt） |
| `tests/evals/eval_e2e_graph_building.py` + `eval_cli.py` | 用 LongMemEval oracle 数据做**图构建质量**的 pairwise LLM 评测（baseline 图 vs candidate 图），不是 benchmark QA 准确率 |
| `tests/evals/data/longmemeval_data/` | 自带 `longmemeval_oracle.json` 数据 |

## Answer prompt

### LoCoMo harness（`benchmarks/locomo/prompts.py`，逐字原文）

**无字数限制**。System prompt：

```python
RESPONSE_SYSTEM_PROMPT = """
You are a helpful expert assistant answering questions based on the provided context.
"""
```

User prompt（要求 "precise, concise answer"，大量篇幅在教模型用 timestamp 换算日期）：

```python
RESPONSE_PROMPT = """
# CONTEXT:
You have access to facts and entities from a conversation.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. Always convert relative time references to specific dates, months, or years.
6. Be as specific as possible when talking about people, places, and events
7. Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.

Clarification:
When interpreting memories, use the timestamp to determine when the described event happened, not when someone talked about the event.

Example:

Memory: (2023-03-15T16:33:00Z) I went to the vet yesterday.
Question: What day did I go to the vet?
Correct Answer: March 15, 2023
Explanation:
Even though the phrase says "yesterday," the timestamp shows the event was recorded as happening on March 15th. Therefore, the actual vet visit happened on that date, regardless of the word "yesterday" in the text.


# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Context:

{context}

Question: {question}
Answer:
"""
```

Context 模板（facts 带 event_time + entities 摘要）：

```python
CONTEXT_TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts for the conversation along with the datetime of the event that the fact refers to.
# If a fact mentions something happening a week ago, then the datetime will be the date time of last week and not the datetime
# of when the fact was stated.
# Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.

<FACTS>
{facts}
</FACTS>

# These are the most relevant entities
# ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>
"""
```

### LongMemEval（`benchmarks/longmemeval/zep_longmem_eval.py`，`lme_response`，逐字原文）

只有 "briefly answer" + 允许弃答，**无字数限制**：

```python
system_prompt = """
You are a helpful expert assistant answering questions from lme_experiment users based on the provided context.
"""

prompt = f"""
Your task is to briefly answer the question. You are given the following context from the previous conversation. If you don't know how to answer the question, abstain from answering.

Context: {context}

Question: {question}
"""
```

问题送入时前面拼了日期：`question = f"(date: {question_date}) {question}"`（temporal reasoning 靠这个）。

### Zep Test Harness（`benchmarks/longmemeval/Zep Test Harness/zep_responses.py`）

LoCoMo 数据 + "briefly answer... abstain from answering" + 与 LoCoMo harness 相同的 timestamp step-by-step 说明，模型 `gpt-4.1-mini`，temp 0。同样**无字数限制**。

### Graphiti（`graphiti_core/prompts/eval.py`，`qa_prompt`，逐字原文）

```python
sys_prompt = """You are Alice and should respond to all questions from the first person perspective of Alice"""

user_prompt = f"""
Your task is to briefly answer the question in the way that you think Alice would answer the question.
You are given the following entity summaries and facts to help you determine the answer to your question.
<ENTITY_SUMMARIES>
{to_prompt_json(context['entity_summaries'])}
</ENTITY_SUMMARIES>
<FACTS>
{to_prompt_json(context['facts'])}
</FACTS>
<QUESTION>
{context['query']}
</QUESTION>
"""
```

**结论：所有 answer prompt 都没有 "5-6 words" 之类的字数限制**，最强约束只是 "briefly" / "concise"。

## Judge 配置

### LoCoMo（`benchmarks/locomo/prompts.py` GRADER_PROMPT + `evaluation.py` `_grade_response`）

- Judge 模型：`gpt-4o-mini`，temperature 0（`benchmark_config.yaml`）
- 二元判分：结构化输出 `Grade(is_correct: "CORRECT or WRONG", reasoning: str)`，`is_correct.strip().lower() == "correct"` → bool
- **明确要求 judge 宽松**（"be generous with your grading"），日期只要求同一日期/时间段、格式不同也算对（无 ±N 天数值容差）

逐字原文：

```python
GRADER_SYSTEM_PROMPT = """
You are an expert grader that determines if answers to questions match a gold standard answer.
"""

GRADER_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {response}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.
"""
```

另有第二个 judge：**context completeness**（`evaluation.py` `evaluate_context_completeness`），同模型，判 retrieval context 是否含答案所需信息，三档 COMPLETE / PARTIAL / INSUFFICIENT + missing/present elements。这是 Zep 自称的 "PRIMARY evaluation metric"（评检索质量，与答案无关）。

### LongMemEval（`zep_longmem_eval.py` GRADING_PROMPTS）

- Judge 模型：`gpt-4o`（`GRADER_MODEL = "gpt-4o"`），temperature 0；response 模型也是 `gpt-4o`
- 二元 yes/no：`Grade(is_correct: str)`，`== "yes"` → bool
- **按 question_type 用不同 judge prompt**，与 LongMemEval 官方 judge prompt 基本一致，共 4 个：`temporal-reasoning`（**明确容忍天数 off-by-one**："do not penalize off-by-one errors for the number of days ... predicting 19 days when the answer is 18, the model's response is still correct"）、`knowledge-update`（旧信息+新答案也算对）、`single-session-preference`（按 rubric 判个性化回答）、`default`（"answer yes if the response contains the correct answer ... If the response only contains a subset of the information required by the answer, answer no"）

default 版逐字原文：

```python
"default": """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>
""",
```

### Zep Test Harness（`zep_eval.py`）

Judge 模型 `gpt-4.1-mini`，temp 0，用上面 LongMemEval default prompt 判 LoCoMo 的 QA，二元。

### Graphiti（`graphiti_core/prompts/eval.py` `eval_prompt`，逐字原文）

```python
sys_prompt = (
    """You are a judge that determines if answers to questions match a gold standard answer"""
)

user_prompt = f"""
Given the QUESTION and the gold standard ANSWER determine if the RESPONSE to the question is correct or incorrect.
Although the RESPONSE may be more verbose, mark it as correct as long as it references the same topic 
as the gold standard ANSWER. Also include your reasoning for the grade.
<QUESTION>
{context['query']}
</QUESTION>
<ANSWER>
{context['answer']}
</ANSWER>
<RESPONSE>
{context['response']}
</RESPONSE>
"""
```

输出 `EvalResponse(is_correct: bool, reasoning: str)`，二元。同文件还有 `eval_add_episode_results`：pairwise judge，判 candidate 图构建结果是否不差于 baseline（"If ... nearly identical in quality, return True"，即 tie 判给 candidate）。

## 指标计算

- **没有 F1 / BLEU，全部是 LLM-judge 二元准确率**（accuracy = correct/total）
- LoCoMo：`evaluation.py` `evaluate_locomo` 累计 correct_count；`common.py` `BenchmarkMetrics` 汇总 overall accuracy、by_category、by_difficulty、completeness 三档占比、`accuracy_with_complete_context`（context 完整子集上的准确率）、retrieval/response 延迟分布（p50/p90/p95/p99）、context token 统计（tiktoken 计数）
- **Category 5 (adversarial) 不评**：`evaluation.py` L87-89 逐字为
  ```python
  # Skip category 5 as golds are not provided for this category
  if qa.get("category") == 5:
      continue
  ```
  `Zep Test Harness/zep_responses.py` 同样 `qa.get("category") != 5` 过滤
- LongMemEval：`run_evaluation` 里 accuracy = correct_count / num_sessions，另记平均 response/retrieval 时延；结果按 question_type 存 jsonl
- Graphiti e2e：图构建质量分 = 每 user 的（candidate 不差于 baseline 的消息比例）再对 user 平均，不算 QA 准确率

## 评测范围

- **LoCoMo**：locomo10 全量 10 users（官方 snap-research 原始 URL 下载），除 category 5 外全部 QA。仓库里提交了 2025-12 的 5 组实验，每组 10 runs；最新一组（`experiments/experiment_20251207_215609/`，edge_limit=30, node_limit=30, 双 cross_encoder，gpt-4o-mini 应答+判分）**accuracy mean = 0.8032**（min 0.798, max 0.812），completeness COMPLETE 率 ~0.765。检索方式：graph.search 分别取 edges 和 nodes（各带 reranker），拼 context，不用整段对话
- **LongMemEval**：`longmemeval_s`（500 题，Google Drive 下载），user/session 逐条 ingest 进 Zep cloud；可 `--question-type` 过滤。这是其论文（arXiv 2501.13956）和 LongMemEval 博客结果的代码
- **DMR（MSC 500 题）**：`zep_memgpt_eval.ipynb`，复现 MemGPT 论文实验
- **对比的 baseline（仓库内）**：只有 full-context baseline——LongMemEval 脚本 `--baseline` 用全部 haystack 对话直接喂 LLM；DMR notebook baseline 用全对话和 speaker summaries 两种。**仓库内没有 Mem0/MemGPT/Letta 等其他记忆框架的运行代码**；博客里与 Mem0、full-context 的对比数字来自各自报告或 Zep 复评（Zep Test Harness 那套脚本疑似即 Mem0 争议复评所用）
- Graphiti 仓库只评自身图构建质量（LongMemEval oracle 数据），不产出 benchmark 榜单数字

## 要点总结

1. answer prompt 无字数限制（无 "5-6 words"），只有 "briefly"/"concise"；LoCoMo 应答 prompt 反而给了大量 timestamp 推理指导，对时间类题目有利
2. Judge 全部二元 0/1，且 LoCoMo judge 明文要求 "be generous"；日期容差是"同一日期/时段即可、格式不限"（LoCoMo）和"天数 off-by-one 不罚"（LongMemEval temporal-reasoning，官方原版规则）
3. LoCoMo 跳过 category 5（理由：数据集未提供 gold）——与 Mem0 等一致
4. Zep 额外引入 context completeness 这一自定义指标，把"检索质量"与"答案质量"分开报告
