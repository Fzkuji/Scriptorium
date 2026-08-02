# MemMachine（v0.2.x，博客报 LoCoMo 91.7）

- GitHub: https://github.com/MemMachine/MemMachine
- 调研对象：tag `v0.2.0` / `v0.2.6`（91.7 博客对应版本）+ `main` 分支（2026-07 时点）
- 博客：《MemMachine v0.2 Delivers Top Scores and Efficiency on LoCoMo Benchmark》（memmachine.ai, 2025-12）报 **0.9169**（gpt-4.1-mini, agent 模式）/ 0.9123（memory 模式）
- **结论先行：仓库自带完整评测代码，LoCoMo 评测链路直接从 Mem0 的 evaluation 代码改造而来（文件头逐一注明 "adapted from Mem0"），answer prompt 就是 Mem0 那个含 "less than 5-6 words" 的 prompt，judge 也是 Mem0 的 gpt-4o-mini 二值 judge，category 5 (adversarial) 全线跳过。**

## 评测脚本位置

v0.2.x（v0.2.0 ~ v0.2.6）目录 `evaluation/locomo/`：

```
evaluation/
├── README.md                          # 评测总指南（含示例分数表 overall 0.8487）
└── locomo/
    ├── locomo10.json                  # LoCoMo 数据集直接放在仓库里
    ├── episodic_memory/               # "memory 模式"：检索→单次 LLM 作答
    │   ├── locomo_ingest.py / locomo_search.py / locomo_evaluate.py
    │   ├── llm_judge.py / generate_scores.py / locomo_delete.py
    │   └── restapiv2_locomo_search.py # REST API v2 版（v0.2.6 有，v0.2.0 无），prompt 不同
    ├── episodic_agent/                # "agent 模式"：OpenAI Agents SDK 多轮工具调用
    │   ├── locomo_agent.py / memmachine_locomo.py / run_experiments.py
    │   └── evals.py / llm_judge.py / generate_scores.py
    └── utils/                         # helper（v0.2.6）
```

main 分支（2026-07）重组为 `evaluation/episodic_memory/`（LoCoMo + LongMemEval）和 `evaluation/retrieval_agent/`（LoCoMo、LongMemEval、HotpotQA、2WikiMultihopQA、BEAM），数据在 `evaluation/data/`。LongMemEval 脚本是 2026-03-13 才加的（PR #1216 "Add LongMemEval evaluation scripts"），**v0.2 时期只有 LoCoMo**。

## Answer prompt

### 1. LoCoMo memory 模式（`evaluation/locomo/episodic_memory/locomo_search.py`，v0.2.0 至 main 均在，main 路径 `evaluation/episodic_memory/locomo_search.py`）

文件头注释：`This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/prompts.py). It is modified to work with MemMachine.` —— 即 Mem0 的 ANSWER_PROMPT 原样搬来。**第 8 条有 "less than 5-6 words" 字数限制**：

```python
ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories to answer a question.

    # CONTEXT:
    You have access to memories from a conversation. These memories contain
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
       names mentioned in memories with the speakers.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    <MEMORIES>

    {conversation_memories}

    </MEMORIES>

    Question: {question}

    Answer:
    """
```

作答配置：`query_memory(query=question, limit=30, expand_context=3)` 取 top-30 记忆（长期+短期 episode + working memory summary），作答 LLM `gpt-4o-mini`，`temperature=0.0, top_p=1, max_output_tokens=4096`。记忆格式为 `[timestamp] speaker: content [ATTACHED: blip_caption]`。

### 2. LoCoMo REST API v2 版（`evaluation/locomo/episodic_memory/restapiv2_locomo_search.py`，v0.2.6；main 同名文件）

自写 prompt，**无 5-6 words 限制**，改为 "no more than a couple of sentences"：

```python
ANSWER_PROMPT = """
You are asked to answer a question based on your memories of a conversation.

<instructions>
1. Prioritize memories that answer the question directly. Be meticulous about recalling details.
2. When there may be multiple answers to the question, think hard to remember and list all possible answers. Do not become satisfied with just the first few answers you remember.
3. When asked about time intervals or to count items, do not rush to answer immediately. Instead, carefully enumerate the items or subtract the times using numbers.
4. Your memories are episodic, meaning that they consist of only your raw observations of what was said. You may need to reason about or guess what the memories imply in order to answer the question.
5. The question may contain typos or be based on the asker's own unreliable memories. Do your best to answer the question using the most relevant information in your memories.
6. Your memories may include small or large jumps in time or context. You are not confused by this. You just did not bother to remember everything in between.
7. Your memories are ordered from earliest to latest.
</instructions>

<memories>
{memories}
</memories>

Question: {question}
Your short response to the question without fluff (no more than a couple of sentences):
"""
```

### 3. LoCoMo agent 模式（`evaluation/locomo/episodic_agent/locomo_agent.py`，博客 0.9169 的模式）

executor agent 的 system instructions（可多轮调用 `search_conversation_session_memory` 工具，每轮 `limit=30, expand_context=3`，max_turns=30，`ModelSettings(max_tokens=2000, temperature=0.2)`；`memmachine_locomo.py` 里默认模型 `gpt-4o-mini`，博客用 gpt-4.1-mini）。**没有硬性字数限制，但明确提示 ground truth 少于 6 个词**：

```python
LOCOMO_EXECUTOR_INSTRUCTIONS = """
You are the executor agent. As the executor, your role is to answer the user's quesiton using the memories in the context and by querying for more memories if necessary.

# ENVIRONMENT
You will see episodic memories in your context:
    - Episodic memories come directly from the conversation source text and serve as the ground truth.
    - Each episodic memory has a timestamp for the message and a speaker associated with the message.
    - Some episodic memories may contain a blip caption for an attached image.

# ACTION SPACE
There are 2 actions available to you:
- You may query for extra memories using the search_conversation_session_memory tool.
    - The search_conversation_session_memory tool takes a single argument:
        - query (string): your search query (it will be embedded and compared to memory embeddings)
- You may generate your final output.

Proceed with the following program:

# PROGRAM
1. Based on the existing memories, decide whether to call search_conversation_session_memory (proceed to step 2) or generate the final output (go to step 3).
    - If the provided TURN is greater than 5, you should prefer to generate the final output (step 3).
    - If the provided TURN is greater than 10 at any point, you must generate the final output (step 4).
2. Call search_conversation_session_memory using hints from the speakers. Go to step 1.
3. Generate final response.

# FINAL RESPONSE GUIDELINES
- If there is a memory that contains relative time references (like "last year", "two months ago", etc.),
  calculate the actual time based on the timestamp of the episode.
  For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021 and the answer would be 2021.
  If the metadata is "12 April 2023", and the memory is "went to the beach yesterday", then the date is "11 April 2023".
- Time answers must include an absolute reference point. Do not assume that the current datetime is near any of the memory timestamps. The recipient will not get the context, and may read the answer far in the future.
- If the memories contain contradictory information, prioritize the most recent memory.
- The correct (ground truth) answer will be less than 6 words, but yours may be longer.
"""
```

v0.2.6 的 `restapiv2_locomo_search_agent.py` 另有一版：`LOCOMO_INSTRUCTIONS1`（直接抄 OpenAI GPT-5 prompting guide 的 context_gathering/persistence 段）+ `LOCOMO_INSTRUCTIONS2`（"Your final response to the question should be no more than a couple of sentences."）。

### 4. LongMemEval（仅 main 分支，`evaluation/episodic_memory/longmemeval_search.py`）

注释：`Parts of prompt borrowed from Mastra's OM`（Mastra observational-memory prompt）。作答模型默认 `gpt-5-mini`，无字数限制：

```python
ANSWER_PROMPT = """
You are a helpful assistant with access to extensive conversation history.
When answering questions, carefully review the conversation history to identify and use any relevant user preferences, interests, or specific details they have mentioned.

<history>
{memories}
</history>

IMPORTANT: When responding, reference specific details from these observations. Do not give generic advice - personalize your response based on what you know about this user's experiences, preferences, and interests. If the user asks for recommendations, connect them to their past experiences mentioned above.

KNOWLEDGE UPDATES: When asked about current state (e.g., "where do I currently...", "what is my current..."), always prefer the MOST RECENT information. Observations include dates - if you see conflicting information, the newer observation supersedes the older one. Look for phrases like "will start", "is switching", "changed to", "moved to" as indicators that previous information has been updated.

PLANNED ACTIONS: If the user stated they planned to do something (e.g., "I'm going to...", "I'm looking forward to...", "I will...") and the date they planned to do it is now in the past (check the relative time like "3 weeks ago"), assume they completed the action unless there's evidence they didn't. For example, if someone said "I'll start my new diet on Monday" and that was 2 weeks ago, assume they started the diet.

Current date: {question_timestamp}
Question: {question}
"""
```

main 分支 `evaluation/retrieval_agent/longmemeval_test.py` 还有一版引 Agent Lightning（arXiv:2508.03680）的 ANSWER_PROMPT，允许 open-domain fallback（记忆答不出时用世界知识兜底）。

## Judge 配置

### LoCoMo judge（`evaluation/locomo/episodic_memory/llm_judge.py`，v0.2.0 至 main 不变；episodic_agent 版仅引号/排版差异）

文件头注释：`This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/metrics/llm_judge.py).` —— **就是 Mem0 的 judge**。judge 模型 **gpt-4o-mini**（硬编码），`temperature=0.0`，JSON 输出，**0/1 二值**（CORRECT=1 否则 0），**"be generous with your grading"**，时间题容差宽（同一时间段即可、格式不同也算对）：

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
def evaluate_llm_judge(question, gold_answer, generated_answer) -> int:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[...],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    label = json_repair.loads(response.choices[0].message.content)["label"]
    return 1 if label == "CORRECT" else 0
```

**Category 5 跳过**，三处都跳：`llm_judge.py` main 循环里 `if int(category) == 5: continue`；`locomo_evaluate.py` 里 `if key == "5": continue`；main 分支 `evaluation/episodic_memory/generate_scores.py` 里 `if category == "5": continue`。

main 分支 `retrieval_agent/llm_judge.py` 把 judge 模型改为配置化（`retrieval_agent.judge_llm_model`，回退 `llm_model`），prompt 同上。

### LongMemEval judge（仅 main，`evaluation/episodic_memory/longmemeval_evaluate.py`）

直接用 **LongMemEval 官方的 anscheck prompt**（`get_anscheck_prompt`，逐字照搬官方五个模板：single-session-user/assistant/multi-session 通用版、temporal-reasoning 版含 **off-by-one 天数容差**、knowledge-update 版、single-session-preference rubric 版、abstention 版），judge 模型 **gpt-4o**，`temperature=0.0`，判定 `1 if "yes" in llm_response.lower() else 0`，`main()` 里 `exclude_abstention=False`（abstention 题也评）。

## 指标计算

- **只有 LLM-judge 准确率，没有 F1/BLEU**。`locomo_evaluate.py` 文件头明说：`It is modified to only report LLM judge scores and to be simpler.`（把 Mem0 evals.py 里的 F1/BLEU 部分删了）。整个 evaluation/ 目录 grep 不到 BLEU；唯一的 F1 在 main 分支 BEAM 评测（`retrieval_agent/beam/beam_evaluate.py`，fact-alignment precision/recall/F1 × normalized Kendall tau），与 LoCoMo 无关。
- LoCoMo 分数 = 0/1 judge 标签的均值。`generate_scores.py` 用 pandas `df.groupby("category").agg({"llm_score": "mean"})` 出各 category 均值 + `df.agg({"llm_score": "mean"})` 出 overall（按题数加权的总均值，1540 题）。
- LongMemEval 分数 = 各 question_type 的 yes 比例 + overall 均值，另记 latency/token 统计（`longmemeval_evaluate.py`），`lme_generate.py` 打印 7 类分数。

## 评测范围

- **v0.2 时期（91.7 博客对应）只评 LoCoMo**：`locomo10.json`（随仓库分发），10 段对话共 1986 问（cat1 multi-hop 282 / cat2 temporal 321 / cat3 open-domain 96 / cat4 single-hop 841 / cat5 adversarial 446），**剔除 cat5 后实评 1540 问**。
- 博客数字（2025-12《MemMachine v0.2 Delivers Top Scores and Efficiency on LoCoMo Benchmark》）：agent 模式 gpt-4.1-mini overall **0.9169**（cat1 0.8830 / cat2 0.9159 / cat3 0.7188 / cat4 0.9512），memory 模式 0.9123；judge gpt-4o-mini；embedder text-embedding-3-small；博客自述评测代码 "obtained from Mem0 evaluation of the LoCoMo benchmark"。
- **baseline 只对比了 Mem0**（自跑，gpt-4.1-mini，0.8000），没有 Zep/Letta 等其他系统。注意仓库 README/docs 示例给的是 memory 模式 gpt-4o-mini 的 overall 0.8487 —— 91.7 靠的是换 gpt-4.1-mini + agent 多轮检索模式。
- main 分支后续新增：LongMemEval（s/m/oracle 三个 cleaned json，官方 anscheck judge）、HotpotQA、2WikiMultihopQA、BEAM（均在 `evaluation/retrieval_agent/`），与 91.7 博客无关。

## 与 91.7 数字相关的关键观察

1. 整条评测链路（answer prompt、judge prompt、judge 模型、cat5 剔除、打分聚合）都继承自 Mem0 的 evaluation 代码，所以与 Mem0 报告的数字在协议上同源可比；但 agent 模式给了模型最多 10+ 轮重检索机会（每轮 top-30 + expand_context=3），检索预算远大于 Mem0 单次检索。
2. memory 模式沿用 "less than 5-6 words" 短答案 prompt；agent 模式反而放开（"yours may be longer"），配合 generous judge，对 recall 型长答案有利。
3. Judge 是 0/1 二值、gpt-4o-mini、宽松时间容差；无连续分、无人工复核。
4. cat5 (adversarial) 全部不评，overall 是 4 类 1540 题的加权均值 —— 与 Mem0/Zep 系博客同一口径。
