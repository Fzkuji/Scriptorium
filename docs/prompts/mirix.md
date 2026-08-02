# MIRIX（Mirix-AI/MIRIX，多模态多智能体记忆）

调研时间：2026-07-02。仓库：https://github.com/Mirix-AI/MIRIX（论文 arXiv:2507.07957）。

MIRIX 仓库里有**两代自带评测代码**：

1. **`public_evaluation` 分支的 `public_evaluations/`** —— 论文（LoCoMo + ScreenshotVQA，对比 Mem0/Zep/LangMem/长上下文）用的原始评测代码。main 分支已删除该目录，但分支还在（`git fetch origin public_evaluation`）。
2. **main 分支的 `evals/`** —— 新一代评测 runner（2026 年活跃开发中），跑 LoCoMo + MemoryAgentBench（LongMemEval-S / RULER / LRU），带四种 judge。

## 评测脚本位置

### main 分支 `evals/`

| 文件 | 作用 |
|---|---|
| `evals/main_eval.py` | LoCoMo runner（`data/locomo10.json`，数据文件不随仓库分发） |
| `evals/task_agent.py` | 回答 agent：gpt-4.1-mini + `search_memory`/`check_raw_item` 工具循环（最多 5 轮），含 answer 系统 prompt |
| `evals/mirix_memory_system.py` | 记忆写入（`add_chunk`）与检索包装（`wrap_user_prompt`） |
| `evals/llm_judge.py` | 通用 LoCoMo judge（gpt-4o-mini，CORRECT/WRONG） |
| `evals/organize_results.py` | 汇总指标 → `metrics.json`（accuracy、分类别 accuracy、延迟、成本、记忆 token 数） |
| `evals/mab/longmem_eval.py` | LongMemEval-S runner（HF `ai-hyz/MemoryAgentBench`，split `Accurate_Retrieval`，source `longmemeval_s*`） |
| `evals/mab/ruler_eval.py` | RULER QA runner（`ruler_qa1_197K` 单跳 SQuAD / `ruler_qa2_421K` 多跳 HotpotQA） |
| `evals/mab/lru_eval.py` | Long_Range_Understanding runner（`infbench_sum_eng_shots2` 小说摘要 / `detective_qa`） |
| `evals/mab/llm_judge_mab.py` | MemoryAgentBench 官方 judge 移植（gpt-4o，按题型分 prompt，yes/no） |
| `evals/mab/llm_judge_substring.py` | substring_exact_match（无 LLM，RULER 用） |
| `evals/mab/llm_judge_mab_summary.py` | MAB 摘要 judge（gpt-4o-2024-05-13，fluency/recall/precision→F1） |
| `evals/mab/run_mab_{longmem,ruler,lru}_eval.sh` | 一键脚本 |

### `public_evaluation` 分支 `public_evaluations/`（论文结果来源）

| 文件 | 作用 |
|---|---|
| `main.py` / `run_instance.py` | LoCoMo & ScreenshotVQA runner（`--agent_name mirix/gpt-long-context/gemini-long-context/siglip`） |
| `conversation_creator.py` | 数据加载 + ingestion prompt + **answer question prompt** |
| `agent.py` | 各 agent 封装（MIRIX 走 `AgentWrapper.send_message`，问答前更新 persona 要求极简回答） |
| `evals.py` + `metrics/llm_judge.py` + `metrics/utils.py` | 指标：LLM judge + BLEU + token-F1（utils.py 注明借自 A-Mem/AgenticMemory） |
| `evaluation_metrics/evaluation_metrics_run{1,2,3}.json` | 论文 LoCoMo 三次 run 的原始打分 |
| `baselines/mem0/evaluation/` | 整个 Mem0 评测 harness 的 vendored 拷贝（含 mem0/langmem/rag/zep 结果 json） |
| `baselines/zep-papers/` | Zep 官方 LoCoMo 评测拷贝 |
| `run.sh` | 逐个 `global_idx 0..9` 跑完 10 个 LoCoMo 对话，每个前 reset postgresql |

## Answer prompt（原文）

### 1) 论文版（public_evaluation 分支）—— **无 "5-6 words" 数字限制，但有强简短要求**

每个 LoCoMo 问题被包进这个 prompt（`public_evaluations/conversation_creator.py`, `get_query_and_answer()`，L178-189）：

```python
question = f"""You will be given a question and you need to answer the question based on the memories.
# APPROACH (Think step by step):
1. First, search and check the memories that might contain information related to the question.
2. Examine the timestamps and content of these memories carefully.
3. Look for explicit mentions of dates, times, locations, or events that answer the question.
4. If the answer requires calculation (e.g., converting relative time references), show your work.
5. Formulate a precise, concise answer based solely on the evidence in the memories.
6. Double-check that your answer directly addresses the question asked.
7. Ensure your final answer is specific and avoids vague time references like "yesterday", "last year" but with specific dates.
8. The answer should be as brief as possible, you should **only state the answer** WITHOUT repeating the question. For example, if asked 'When did Mary go to the store?', you should simply answer 'June 1st'. Do NOT say 'Mary went to the store on June 1st' which is redundant and strictly forbidden. Your answer should be as short as possible.

Question: {question}"""
```

问答开始前还把 MIRIX 的 core memory persona 改成极简应答（`public_evaluations/agent.py` L87）：

```python
self.agent.update_core_memory_persona("Is a helpful assistant that answers questions with extreme conciseness.\nIs persistent and tries to find the answerr using different queries and different search methods. Never uses unnecessary words or repeats the question in the answer. Always provides the shortest answer possible and tries to utter the fewest words possible.")
```

注意：**"The answer should be less than 5-6 words." 只出现在 vendored 的 Mem0 baseline prompt 里**（`public_evaluations/baselines/mem0/evaluation/prompts.py` L27/L85/L130，Mem0 官方原文），MIRIX 自己的 prompt 是同一 Mem0 模板的改写，把 5-6 词的硬限制换成了 "as brief as possible / only state the answer"。ScreenshotVQA 的问题则不加包装直接问。

### 2) 新版（main 分支 `evals/task_agent.py`，`TaskAgent.answer()` 系统 prompt，L211-268 节选关键部分）

回答模型 gpt-4.1-mini，`max_completion_tokens=128`，最多 5 轮 `search_memory` 工具调用。同样无数字词数限制，但要求只输出答案、禁止说不知道、找不到也要猜：

```python
system_prompt = (
    "You are the Chat Agent, a component of the personal assistant system. "
    ...
    "\n\nMessage Processing Protocol:\n"
    "1. Analyze the user's query and use `search_memory` to gather necessary context.\n"
    "   - If a result includes `raw_input_id` and you need the original text, call `check_raw_item`.\n"
    "2. Provide a helpful and concise answer based on the retrieved information.\n"
    "3. Only inform the user that you don't know the answer if at least three consecutive searches with different parameters have failed to yield relevant information.\n"
    "4. Be VERY CONCISE in your response, only output the answer and nothing else.\n"
    "5. There are some open-ended questions where you may not find explicit evidences, you still need to answer it based on your understanding. Never say you don't know or 'there is no specific information', ...\n"
    "6. If there is no information found, you still need to answer it. Guess an answer if you don't have enough information.\n"
    "\n\nAnswer Format Guidelines (CRITICAL):\n"
    "- For list questions (What books, What instruments, What activities, etc.), provide a simple comma-separated list or use 'and' between items.\n"
    "  Example: \"clarinet and violin\" NOT \"She plays clarinet\"\n"
    ...
    "- For simple fact questions (What is X's relationship status?, How old?, etc.), provide direct factual answers.\n"
    "  Example: \"Single\" NOT \"She experienced a breakup but is...\"\n"
    ...
    "- ALWAYS extract the minimal, direct answer that matches what's being asked. Do NOT add ANY additional information!\n"
    "- If the question asks for multiple items, search until you find ALL items, not just the first one."
)
```

检索上下文以 system 消息注入（`mirix_memory_system.py` `wrap_user_prompt`）：`"These are the high-level memories retrieved automatically according to the user's query:\n<episodic_memory>...（每条记忆 [timestamp] summary）...</episodic_memory>"`，后跟原始问题。

## Judge 配置

### 1) LoCoMo judge（两代通用；`evals/llm_judge.py` 与 `public_evaluations/metrics/llm_judge.py` 完全同文）

- **模型：gpt-4o-mini**，temperature=0.0，`response_format={"type": "json_object"}`
- **0/1 二值**（CORRECT=1 / WRONG=0），无连续分
- Prompt 就是 **Mem0 LoCoMo judge 原文**（宽松打分 + 日期容差：同一日期不同格式算对，无 ±N 天数值容差）：

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
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": ACCURACY_PROMPT.format(...)}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    label = json.loads(response.choices[0].message.content)["label"]
    return 1 if label == "CORRECT" else 0
```

### 2) MemoryAgentBench judge（main 分支 `evals/mab/llm_judge_mab.py`，官方 `longmem_qa_evaluate.py` 移植）

- **模型：gpt-4o**，temperature=0，max_tokens=10；**判分规则 `1 if "yes" in text.lower() else 0`**（0/1）
- 按题型分 5 个 prompt：single/multi-session 通用、temporal（**明确允许天数 off-by-one**）、knowledge-update（旧+新信息只要更新值对就算对）、preference（不必命中全部 rubric）、abstention（`question_id` 带 `_abs` 后缀时路由）。temporal prompt 原文关键句：

```
... In addition, do not penalize off-by-one errors for the number of days. If the question asks for
the number of days/weeks/months, etc., and the model makes off-by-one errors
(e.g., predicting 19 days when the answer is 18), the model's response is still correct.
...
Is the model response correct? Answer yes or no only.
```

### 3) substring judge（`evals/mab/llm_judge_substring.py`）

无 LLM。normalize（小写、去标点、去冠词、压空白）后判断预测是否包含任一 gold 子串，0/1，即 MAB 官方 `substring_exact_match`。RULER / detective_qa 用。

### 4) 摘要 judge（`evals/mab/llm_judge_mab_summary.py`）

模型钉死 **gpt-4o-2024-05-13**（temperature=0.1），每条 3 次调用：fluency（0/1）、recall（keypoints 命中数）、precision（句子被专家摘要支持数），`F1 = fluency * 2RP/(R+P)` —— **连续分 [0,1]**。仅用于 infbench 摘要任务。

## 指标计算

### 论文版（`public_evaluations/evals.py` + `metrics/utils.py`）

- 每题算三个分：`bleu_score`（BLEU-1，nltk `sentence_bleu` + SmoothingFunction().method1）、`f1_score`（**token 集合 F1**：`simple_tokenize` 小写去标点后按 set 求 P/R/F1，代码借自 A-Mem `WujiangXu/AgenticMemory/utils.py`）、`llm_score`（上面的 gpt-4o-mini judge，0/1）
- 汇总：overall 平均 + 按 category 平均；**`calculate_category_stats` 里 `categories = ["1", "2", "3", "4"]` —— category 5 (adversarial) 不进分类汇总**（run1 的 detailed_results 里只混进 2 条 cat-5，1540 条为 cat 1-4，说明上游数据就基本剔除了 cat 5）
- 论文 LoCoMo 主指标 = 三次 run 的 overall_avg_llm_score 平均：run1 0.8398 / run2 0.8734 / run3 0.8482 → **均值 0.8538 ≈ 论文报的 85.4%（LLM-as-a-Judge）**

### 新版（`evals/organize_results.py`）

- `--judge default|mab|substring|mab_summary` 四选一；16 进程并行调 judge
- accuracy = score 之和 / 判分题数，**显式排除 category 5**（`if result.get("category") != 5`，两处）；另输出 `accuracy_by_category`（同样 `category == 5: continue`）
- 无 BLEU/F1（QA 只有 judge accuracy）；额外输出延迟分解（add_chunk / wrap_user_prompt / answer）、credit 成本、记忆库 token 数（tiktoken 数 summary+details）
- mab_summary 模式额外输出 `gpt-4-fluency/recall/precision/f1` 均值（对齐 MAB 官方 `averaged_metrics`）

## 评测范围

### 论文版（public_evaluation 分支，= arXiv:2507.07957 的实验）

- **LoCoMo**：`locomo10.json` 全部 10 个对话（`run.sh` global_idx 0-9），cat 1-4 共 ~1540 题（cat 5 adversarial 排除），跑 3 遍取平均；回答/记忆模型 gpt-4.1-mini（README 注明为公平对比从 gemini-2.5-flash 换成 gpt-4.1-mini），检索用 embedding（text-embedding-3-small）
- **ScreenshotVQA**：3 个 student 的截图流 + qa_pairs.json（数据不在 repo 内），对比 agent：`gpt-long-context`、`gemini-long-context`、`siglip`（SigLIP top-50 检索 + Gemini 作答）
- **Baselines**：Mem0、LangMem（vendored Mem0 harness 重跑，model 改 gpt-4.1-mini）、Zep（官方 zep-papers 代码重跑 gpt-4.1-mini）、RAG（mem0 harness 里的 rag_results）；结果 json 都存在 `baselines/` 下

### 新版（main 分支 evals/，进行中）

- **LoCoMo**（main_eval.py，locomo10.json 需自备）
- **LongMemEval-S**：经 HF `ai-hyz/MemoryAgentBench`（split Accurate_Retrieval，source `longmemeval_s*`），每对话 ~60 题，按 session 的 Chat Time 传 `occurred_at` 锚定时间；judge 用 `--judge mab` 才与 MAB 榜可比
- **RULER QA1/QA2**、**LRU（infbench_sum / detective_qa）**：同一 HF 数据集其它 split/source
- 只评 MIRIX 自身（不同 config/prompt 版本 0130a…0208b 互相对比），**不含外部 baseline**

## 关键结论

1. MIRIX 仓库自带完整评测代码（两代），论文 LoCoMo 85.4% 可由 `public_evaluation` 分支的三个 run json 复算验证。
2. **MIRIX 自己的 answer prompt 没有 "5-6 words" 硬性字数限制**；它把 Mem0 模板的第 8 条 "The answer should be less than 5-6 words." 改成了 "The answer should be as brief as possible, you should **only state the answer**"（论文版）/ "Be VERY CONCISE ... only output the answer and nothing else" + `max_completion_tokens=128`（新版）。"5-6 words" 原文只存在于其 vendored 的 Mem0 baseline prompts 里。
3. Judge 是 Mem0 同款 gpt-4o-mini CORRECT/WRONG 二值 judge（宽松、日期同日即对）；category 5 (adversarial) 在两代指标汇总里都被排除。
4. 论文版还报 BLEU-1 与 token-F1（借 A-Mem 代码），但主结论用 LLM judge 分。
