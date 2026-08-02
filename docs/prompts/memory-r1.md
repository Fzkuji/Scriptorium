# Memory-R1（RL 训练记忆管理，arXiv:2508.19828）

**官方仓库：https://github.com/yansikuan/memory-r1（第一作者 Sikuan Yan）**

## 评测脚本位置

**该仓库无自带评测代码。** 截至 2026-07-02，官方仓库 main 分支全部文件为（GitHub trees API 递归列举确认）：

```
.DS_Store
LICENSE
README.md
assets/.DS_Store
assets/memory_r1_framework.png
assets/motivation.png
```

README 明确写着 "Code coming soon. Please star the repo to stay updated!"。没有 evaluation/、benchmarks/、experiments/、tests/、examples/ 任何目录，没有一行 Python。

论文里的结果怎么来的：论文（arXiv:2508.19828，v5 HTML）Appendix C 开头自述 "In developing our Memory Manager Prompt, answer generation agent prompt, and LLM-as-a-Judge prompt, we adapt elements from the prompt released by prior work (Packer et al., 2023; Chhikara et al., 2025)"，Appendix D 又说 "Prompts for memory operations and memory-augmented answer generation are adapted from Chhikara et al. (2025)"。即评测协议直接沿用 **Mem0 的 LoCoMo 评测流水线**（MemGPT + Mem0 的 prompt），训练用 VERL 框架（PPO/GRPO）。下面的 prompt 全部取自论文附录（Figures 9–12），是目前唯一的一手来源。

第三方复刻仓库（非官方，可作交叉参考）：https://github.com/pradyutnair-prosus/memory-r1 ，其中 `src/agents_memory/prompts_r1.py` 的 `ANSWER_AGENT_PROMPT` 与论文 Figure 11 逐字一致（含 "less than 5-6 words"），另有 `scripts/benchmark_locomo.py`、`scripts/benchmark_mem0.py`、`src/agents_memory/evaluation.py`（judge 默认 `gpt-4.1`，三维 PASS/FAIL 0/1 打分——注意这是复刻者自创的 retrieval judge，不是论文的 answer judge）。

## Answer prompt（原文，论文 Figure 11 / Appendix C.2）

**有字数限制：第 8 条 "The answer should be less than 5-6 words."** —— 与 Mem0 的 LoCoMo answer prompt 同源。

```
You are an intelligent memory assistant tasked with retrieving
accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation.
These memories contain timestamped information that may be relevant
to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago"),
   calculate the actual date based on the memory timestamp.
6. Always convert relative time references to specific dates, months, or years.
7. Focus only on the content of the memories. Do not confuse character names
8. The answer should be less than 5-6 words.
9. IMPORTANT: Select memories you found that are useful for answering the questions,
and output it before you answer questions.
10. IMPORTANT: Output the final answer after **Answer:**

# APPROACH (Think step by step):
1. Examine all relevant memories
2. Examine the timestamps carefully
3. Look for explicit mentions that answer the question
4. Convert relative references if needed
5. Formulate a concise answer
6. Double-check the answer correctness
7. Ensure the final answer is specific
8. First output the memories that you found are important before you answer questions

Memories for user John:
- 7:20 pm on 16 June, 2023: ...
... (In total 30 most relevant memories from John's Memory Bank are provided) ...
Memories for user Maria:
- 6:29 pm on 7 July, 2023: ...
... (In total 30 most relevant memories from Maria's Memory Bank are provided) ...
Question: {question}
```

注意第 9 条：Answer Agent 先输出"记忆蒸馏"（选出有用的记忆条目）再回答，这是 Memory-R1 相对 Mem0 answer prompt 的主要改动；RL 训练也是对这个格式做的。检索规模是每个说话人 top-30、共 60 条候选记忆。

## Judge 配置（原文，论文 Figure 12 / Appendix C.3）

- **Judge 模型：论文全文（正文 + 附录）未指明具体 judge 模型**，只说 "J uses a separate LLM to assess semantic correctness, relevance, completeness, and contextual appropriateness"。prompt 声明改编自 MemGPT (Packer et al., 2023) 与 Mem0 (Chhikara et al., 2025)——即 LoCoMo/Mem0 系的 judge prompt。
- **打分：二元 0/1（CORRECT / WRONG）**，非连续分；要求 JSON 输出 `{"label": ...}`。
- **日期容差：宽松**。prompt 明确要求时间类问题只要指同一时间段就算对，格式不同（"May 7th" vs "7 May"）也算对。
- 内容匹配也宽松："as long as it touches on the same topic as the gold answer, it should be counted as CORRECT"。

```
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.
You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer,
which you will score as CORRECT or WRONG.

The point of the question is to ask about something one user should know about the other user based on their
prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be longer, but you should be generous with your grading — as long as it touches
on the same topic as the gold answer, it should be counted as CORRECT.

For time-related questions, the gold answer will be a specific date, month, or year. The generated answer
might include relative references (e.g., "last Tuesday"), but you should be generous — if it refers to
the same time period as the gold answer, mark it CORRECT, even if the format differs (e.g., "May 7th" vs.
"7 May").

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.
Return the label in JSON format with the key as "label".
```

## 指标计算

无代码可考，以下为论文描述：

- **三个指标：token-level F1、BLEU-1 (B1)、LLM-as-a-Judge (J)**。"F1 and B1 measure lexical overlap with ground-truth answers"——即 LoCoMo 官方式的词级重叠计算（与 Mem0 评测同一套）。
- **Category 5 (adversarial) 不评**："Following prior work (Chhikara et al., 2025), we exclude the adversarial subset"（跟 Mem0 论文一样把 LoCoMo 的 adversarial 类剔除）。
- 一个重要细节（论文 Reward Design Analysis 一节）：作者试过用 LLM-as-a-Judge 当 RL reward，J 分最高（63.58）但 F1/BLEU-1 变差，因为答案变长；最终 Answer Agent 的 reward 用 **Exact Match（EM）**，倾向产出短答案——这正好和 answer prompt 的 "less than 5-6 words" 约束以及词重叠指标互相咬合。也就是说 Memory-R1 的模型是**直接朝着短答案指标优化过的**，F1/BLEU-1 数字与 prompt 字数限制强耦合。
- 测试时 greedy decoding（temperature=0），max tokens 2048。

## 评测范围

- **主 benchmark：LoCoMo**（Maharana et al., 2024）。剔除 adversarial 后按 1:1:8 切 train/val/test = **152 / 81 / 1307 个问题**；只报 Single Hop、Multi-Hop、Open Domain、Temporal 四类 + Overall。训练只用那 152 个 QA。
- **零样本泛化：MSC**（MemGPT 改造版）与 **LongMemEval**（Wu et al., 2024），均不参与训练，只在 Figure 4 报聚合分，无逐类明细。
- **Baselines（全部由作者用 LLaMA-3.1-8B-Instruct 和 Qwen-2.5-7B-Instruct 重新实现**，temperature=0，max tokens 2048）：LoCoMo（RAG 式基线）、A-Mem、Mem0、MemoryOS、Memory-SFT（GPT-4o 指导的 SFT 变体消融）。
- 骨干模型：LLaMA-3.1-8B-Instruct；Qwen-2.5-3B/7B/14B-Instruct（scaling 实验）。
- 记忆库构建：GPT-4o-mini 从对话轮抽取事实建 temporal memory bank；RL 用 VERL（PPO + GRPO）。

## 与本项目对比要点

- Answer prompt 有 "5-6 words" 硬约束，且 RL reward 用 EM，模型被训练成输出极短答案——其 F1/BLEU-1 与不加字数限制的系统不可直接比。
- Judge 是 0/1 二元、宽松匹配、宽松日期容差，judge 模型未披露，复现无锚点。
- 官方零代码，所有数字仅论文自报；baseline 全部是作者重实现（非各家官方实现），Mem0 等的分数与其官方论文报告值不同源。
