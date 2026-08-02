# MemOS (MemTensor/MemOS) 自带 benchmark 评测实现调研

- 仓库：https://github.com/MemTensor/MemOS
- 调研版本：main @ `a4f1b5be94b3` (2026-06-18)
- 结论：**仓库自带完整评测代码**，位于 `evaluation/` 目录。README 报的 LoCoMo 75.80 就是用这套 pipeline（LLM-as-a-Judge，gpt-4o-mini，3 次判分取均值）得到的。

## 评测脚本位置

```
evaluation/
├── data/locomo/locomo10.json              # LoCoMo 数据（自带）
├── data/locomo/locomo10_rag.json
├── scripts/
│   ├── locomo/                            # LoCoMo 全流程
│   │   ├── locomo_ingestion.py            # 写入记忆
│   │   ├── locomo_search.py               # 检索（默认 top_k=15，run 脚本用 20）
│   │   ├── locomo_responses.py            # 生成回答（此处过滤 category 5）
│   │   ├── locomo_eval.py                 # LLM judge + NLP 指标
│   │   ├── locomo_metric.py               # 汇总分数（按 category / user）
│   │   ├── locomo_rag.py                  # RAG baseline（chunk 500，k 可调）
│   │   ├── locomo_openai.py               # OpenAI 原生 memory baseline
│   │   └── prompts.py                     # Answer prompts（mem0 / zep / memos 各一套）
│   ├── longmemeval/                       # LongMemEval（longmemeval_s）
│   │   ├── lme_ingestion.py / lme_search.py / lme_responses.py
│   │   ├── lme_eval.py                    # judge 硬编码 gpt-4o-mini
│   │   ├── lme_metric.py
│   │   └── lme_rag.py
│   ├── PrefEval/                          # PrefEval（4 个 gpt-4o-mini 判分维度）
│   ├── personamem/                        # PersonaMem（多选题，选项匹配算 accuracy）
│   ├── long_bench-v2/                     # LongBench-v2
│   ├── utils/prompts.py                   # LME/PersonaMem/PrefEval answer prompt + LME judge prompt
│   ├── run_locomo_eval.sh                 # LoCoMo 一键脚本（num_runs=3）
│   └── run_lme_eval.sh                    # LME 一键脚本（num_runs 用默认 1）
└── docs/en/open_source/evaluation/overview.md   # 评测说明文档
```

## Answer prompt（原文）

### LoCoMo — MemOS 自己用的 prompt（`evaluation/scripts/locomo/prompts.py`，`ANSWER_PROMPT_MEMOS`）

**有字数限制："The answer must be brief (under 5-6 words)"（第 8 条）。**

```python
ANSWER_PROMPT_MEMOS = """
    You are a knowledgeable and helpful AI assistant.

   # CONTEXT:
   You have access to memories from two speakers in a conversation. These memories contain
   timestamped information that may be relevant to answering the question.

   # INSTRUCTIONS:
   1. Carefully analyze all provided memories. Synthesize information across different entries if needed to form a complete answer.
   2. Pay close attention to the timestamps to determine the answer. If memories contain contradictory information, the **most recent memory** is the source of truth.
   3. If the question asks about a specific event or fact, look for direct evidence in the memories.
   4. Your answer must be grounded in the memories. However, you may use general world knowledge to interpret or complete information found within a memory (e.g., identifying a landmark mentioned by description).
   5. If the question involves time references (like "last year", "two months ago", etc.), you **must** calculate the actual date based on the memory's timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
   6. Always convert relative time references to specific dates, months, or years in your final answer.
   7. Do not confuse character names mentioned in memories with the actual users who created them.
   8. The answer must be brief (under 5-6 words) and direct, with no extra description.

   # APPROACH (Think step by step):
   1. First, examine all memories that contain information related to the question.
   2. Synthesize findings from multiple memories if a single entry is insufficient.
   3. Examine timestamps and content carefully, looking for explicit dates, times, locations, or events.
   4. If the answer requires calculation (e.g., converting relative time references), perform the calculation.
   5. Formulate a precise, concise answer based on the evidence from the memories (and allowed world knowledge).
   6. Double-check that your answer directly addresses the question asked and adheres to all instructions.
   7. Ensure your final answer is specific and avoids vague time references.

   {context}

   Question: {question}

   Answer:
   """
```

注意：这是 Mem0 官方 LoCoMo prompt 的**改写版**。相比 Mem0 原版（同文件里的 `ANSWER_PROMPT_MEM0`，"The answer should be less than 5-6 words."），MemOS 版多了两处放松：允许"use general world knowledge to interpret or complete information"（第 4 条）、要求"Synthesize information across different entries"（第 1 条）。baseline（mem0/mem0_graph）用 `ANSWER_PROMPT_MEM0`，zep 用 `ANSWER_PROMPT_ZEP`（无字数限制），其余框架（memobase/memu/supermemory 等）也走 `ANSWER_PROMPT_MEMOS`（`locomo_responses.py` 的 else 分支）。prompt 作为 **system message** 发送，`temperature=0`，模型由环境变量 `CHAT_MODEL` 指定（`.env-example` 里是 `gpt-4o-mini`）。

### LongMemEval（`evaluation/scripts/utils/prompts.py`，`LME_ANSWER_PROMPT`）

**无字数限制**，但注入了 question_date 作为 Current Date：

```python
LME_ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from a conversation. These memories contain timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories.
    2. Pay special attention to the timestamps to determine the answer.
    3. If the question asks about a specific event or fact, look for direct evidence in the memories.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question.
    2. Examine the timestamps and content of these memories carefully.
    3. Look for explicit mentions of dates, times, locations, or events that answer the question.
    4. If the answer requires calculation (e.g., converting relative time references), show your work.
    5. Formulate a precise, concise answer based solely on the evidence in the memories.
    6. Double-check that your answer directly addresses the question asked.
    7. Ensure your final answer is specific and avoids vague time references.

    {context}

    Current Date: {question_date}

    Question: {question}

    Answer:
    """
```

搜索阶段还把 `question_date` 作为 `reference_time` 传给 MemOS 后端（`lme_search.py` 的 `memos_search(..., reference_time=question_date)`），docs 明确说 MemOS Cloud 不支持该参数、建议用开源 server 跑 LME 才有可比数字。

## Judge 配置

### LoCoMo judge（`evaluation/scripts/locomo/locomo_eval.py`，`locomo_grader`）

- Judge 模型：`os.getenv("EVAL_MODEL", "gpt-4o-mini")`，temperature=0
- 打分：**二值 CORRECT/WRONG**（0/1），无部分分
- 每题判 **num_runs 次**（`run_locomo_eval.sh` 传 `--num_runs 3`，argparse 默认也是 3），最终报 3 次 run 各自 accuracy 的 **均值 ± 标准差**
- 日期容差：没有程序化容差，全靠 prompt 里的 "be generous"（同一日期/时间段即 CORRECT，格式不同如 "May 7th" vs "7 May" 也算对）
- 这套 judge prompt 与 Mem0 的 LoCoMo judge prompt 基本一致（源头是 LoCoMo/Mem0 的宽松版 judge）

```python
system_prompt = """
    You are an expert grader that determines if answers to questions match a gold standard answer
    """

accuracy_prompt = f"""
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
Generated answer: {response}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""
```

输出用正则 `\{\s*"label"\s*:\s*["\']([^"\']*)["\']\s*\}` 抽取 JSON label，`label.lower() == "correct"` 记 1 分。

### LongMemEval judge（`evaluation/scripts/utils/prompts.py` 的 `LME_JUDGE_MODEL_TEMPLATE`，由 `lme_eval.py` 的 `lme_grader` 调用）

- Judge 模型：**硬编码 `gpt-4o-mini`**，temperature=0
- 同样二值 CORRECT/WRONG；`run_lme_eval.sh` 未传 `--num_runs`，argparse 默认 **1 次判分**
- **不是 LongMemEval 官方的 per-question-type judge prompt**——官方对 abstention/temporal 等题型有专用 judge 模板，MemOS 用的是一份通用模板（就是 LoCoMo judge prompt 换了例子）：

```python
LME_JUDGE_MODEL_TEMPLATE = """
    Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a ’gold’ (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Where did I buy my new tennis racket from?
    Gold answer: the sports store downtown
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it’s time for the real question:
    Question: {question}
    Gold answer: {golden_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """
```

## 指标计算

主指标是 **LLM-as-a-Judge accuracy**（judge 判 CORRECT 的比例，多 run 取均值±std）。README 的 LoCoMo 75.80 即此。

辅助 NLP 指标在 `locomo_eval.py` / `lme_eval.py` 的 `calculate_nlp_metrics`（两个文件里各有一份几乎相同的实现）：

- **F1**（`calculate_f1_score`）：token **集合**级——`nltk.word_tokenize(lower())` 后取 set，precision = |交|/|response_set|，recall = |交|/|gold_set|，F1 = 调和平均。注意是去重的 set overlap，不是 SQuAD 式 bag-of-tokens F1。
- **BLEU-1/2/3/4**（`calculate_bleu_scores`）：NLTK `sentence_bleu` + SmoothingFunction().method1。
- **ROUGE-1/2/L**（`rouge_scorer`，use_stemmer=True）、**METEOR**。
- **semantic**（可选，默认关）：`Qwen/Qwen3-Embedding-0.6B` 余弦相似度 + BERTScore F1。
- 默认 `--options lexical`，即实际只算 lexical 指标。
- 还记录 context_tokens（tiktoken cl100k_base）和 search/response 时延 P50/P95。

汇总在 `locomo_metric.py` / `lme_metric.py`：按 category、按 user 分组求均值，导出 JSON + Excel。LoCoMo category 映射（`locomo_metric.py` 第 42 行）：

```python
category_mapping = {
    "4": "single hop",
    "1": "multi hop",
    "2": "temporal reasoning",
    "3": "open domain",
}
```

### Category 5 (adversarial) 不评

`locomo_responses.py` 第 97 行，在生成回答阶段就把 category 5 的问题过滤掉了：

```python
qa_set_filtered = [qa for qa in qa_set if qa.get("category") != 5]
```

所以 MemOS 的 LoCoMo 分数是 **category 1-4** 上的 judge accuracy（与 Mem0/Zep 等的通行做法一致）。

## 评测范围

| Benchmark | 数据 | 样本 | 备注 |
|---|---|---|---|
| LoCoMo | `data/locomo/locomo10.json`（仓库自带） | 10 个对话全部 QA，剔除 category 5 | judge 3 runs 取均值 |
| LongMemEval | `longmemeval_s`（需自行从 HF `xiaowu0162/longmemeval-cleaned` 下载） | 500 题（每 user 1 题） | search 时传 reference_time；judge 1 run |
| PrefEval | `filtered_inter_turns.json`（需从 amazon-science/PrefEval 下载） | topk=10 | gpt-4o-mini 判 violate/acknowledge/hallucinate/helpful 四维，报 preference-following accuracy |
| PersonaMem | personamem 数据 | — | 多选题，`<final_answer>(a)` 格式抽取，直接 accuracy |
| LongBench-v2 | scripts/long_bench-v2/ | — | 多选题 |

对比 baseline（脚本 `--lib` choices）：`mem0`、`mem0_graph`、`memobase`、`memu`、`supermemory`，加上 `memos-api`（本地 server）和 `memos-api-online`（云服务）；另有 `locomo_rag.py`（chunk-500 RAG / 全文 context）和 `locomo_openai.py`（OpenAI 原生 memory）两个 baseline。docs 说 zep 也有非官方实现（prompts 里留有 `ANSWER_PROMPT_ZEP`），但当前 HEAD 的 `--lib` choices 已不含 zep。所有 baseline 均为 MemOS 团队的"unofficial implementations"（docs 原话）。

回答模型：`.env-example` 里 `CHAT_MODEL="gpt-4o-mini"`；检索 top_k：run 脚本统一 20。

## 与论文数字的关系

README（LoCoMo 75.80、LongMemEval +40.43% 等）和 arXiv 2507.03724 报的分数就出自这套 `evaluation/` pipeline：gpt-4o-mini 回答 + gpt-4o-mini 宽松 judge + 二值判分 + 排除 category 5 + LoCoMo 3-run 平均。需要注意的可比性问题：(1) MemOS 自己的 answer prompt 比给 baseline 用的 Mem0 prompt 多了"可用世界知识补全"和"跨条目综合"两条放松；(2) LME judge 不是官方 per-type judge；(3) 各家 baseline 是 MemOS 自行实现的接入。
