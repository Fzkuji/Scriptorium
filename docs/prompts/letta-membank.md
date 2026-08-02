# Letta (MemGPT) 与 MemoryBank 仓库自带评测调研

调研日期：2026-07-02。两个框架都是**本地 clone** 直接检查：

- Letta：`/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki/code/baselines/MemGPT/`（remote = `letta-ai/letta`，shallow clone，HEAD = `6d8cb7f`，2026-06-25 的 main）
- MemoryBank：`/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki/code/baselines/MemoryBank/`（remote = `zhongwanjun/MemoryBank-SiliconFriend`，HEAD = `cf61c41`，2023-05-24，即最终版）

**结论先行**：两个仓库当前代码里都没有 LoCoMo / LongMemEval 评测。Letta 历史上有过 MemGPT 论文的 doc-QA / nested-KV 实验代码（含 LLM judge），后来被删；MemoryBank 只有评测数据 + 评测用 answer prompt，没有任何打分/judge/指标代码。

---

# Letta (MemGPT)

## 评测脚本位置

**当前 main（本地 clone）：无任何 benchmark 评测代码。**逐项验证：

- 全仓库 grep `locomo`、`longmemeval`（不区分大小写）：0 命中（GitHub code search `repo:letta-ai/letta` 同样 0 命中）
- 无 `evaluation/`、`benchmarks/`、`experiments/` 目录；`tests/` 全是 integration test（agent 工具、sandbox、MCP、multi-agent 等），`examples/` 只有 notebooks
- GitHub org 检索：`org:letta-ai` 代码搜 `locomo` 也是 0。Letta 的评测放在**独立仓库**：`letta-ai/letta-evals`（通用 stateful-agent 评测框架）和 `letta-ai/letta-leaderboard`（自建 Letta Memory Benchmark：`letta_bench` + `letta_file_bench`，自造数据，不是 LoCoMo/LongMemEval）

**历史上存在过、已删除的论文实验代码**：`paper_experiments/`（MemGPT/Letta ICML 论文的实验），在 PR #2929 "chore: cleaning up our OSS repo" 中删除。最后可见版本在 commit `5511a080`，内容：

| 路径（`paper_experiments/`，@`5511a080`） | 内容 |
|---|---|
| `doc_qa_task/doc_qa.py` | Document QA（NaturalQuestions-Open + Wikipedia 2018 dump，retriever-reader 设定，采 50 题） |
| `doc_qa_task/llm_judge_doc_qa.py` | 打分脚本：先 substring 匹配，不确定的交给 GPT-4 judge |
| `doc_qa_task/1_run_docqa.sh` / `2_run_eval.sh` | 生成 + 评测入口 |
| `nested_kv_task/nested_kv.py` | 合成 nested key-value 查找任务（UUID K/V，exact-value 任务） |

注意：MemGPT 论文里的对话记忆任务 **DMR（Deep Memory Retrieval，基于 MSC）不在这个目录里**，仓库任何版本都没有 DMR 代码。所以：

- MemGPT 论文的 DMR 结果：代码从未随仓库发布
- 后来各处引用的 "MemGPT 在 LoCoMo 上 XX 分"：全部来自**第三方 harness**（Mem0 论文/`mem0ai/mem0` 的 evaluation 目录、Zep 论文等自己跑的 MemGPT baseline），不是 Letta 官方代码
- Letta 官方后来在 blog（"Benchmarking AI agent memory..."）里用 filesystem agent 跑过 LoCoMo 并批评该 benchmark，但对应代码没进 `letta-ai/letta` 主仓库

## Answer prompt（历史 paper_experiments，逐字原文）

**无字数限制**（没有 "5-6 words" 之类约束；格式约束是要求附带证据文档）。

`paper_experiments/doc_qa_task/doc_qa.py` @`5511a080`，固定上下文 baseline 用：

```python
BASELINE_PROMPT = (
    "Answer the question provided according to the list of documents below (some of which might be irrelevant. "
    + "In your response, provide both the answer and the document text from which you determined the answer. "
    + "Format your response with the format 'ANSWER: <YOUR ANSWER>, DOCUMENT: <DOCUMENT TEXT>'. "
    + "If none of the documents provided have the answer to the question, reply with 'INSUFFICIENT INFORMATION'. "
    + "Do NOT provide an answer if you cannot find it in the provided documents. "
    + "Your response will only be considered correct if you provide both the answer and relevant document text, or say 'INSUFFICIENT INFORMATION'."
    + "Answer the question as if though the current year is 2018."
)
```

MemGPT agent 用（发给 agent 的 user message 前缀）：

```python
MEMGPT_PROMPT = (
    "Search your archival memory to answer the provided question. "
    + "Provide both the answer and the archival memory result from which you determined your answer. "
    + "Format your response with the format 'ANSWER: <YOUR ANSWER>, DOCUMENT: <ARCHIVAL MEMORY TEXT>. "
    + "Your task is to answer the question: "
)
```

Agent persona（同文件）：

```python
DOC_QA_PERSONA = "You are Letta DOC-QA bot. Your job is to answer questions about documents that are stored in your archival memory. The answer to the users question will ALWAYS be in your archival memory, so remember to keep searching if you can't find the answer. Answer the questions as if though the year is 2018."
DOC_QA_HUMAN = "The user will ask you questions about documents. Answer them to the best of your ability."
```

## Judge 配置（历史 paper_experiments，逐字原文）

`paper_experiments/doc_qa_task/llm_judge_doc_qa.py` @`5511a080`：

- Judge 模型：`EVAL_MODEL = "gpt-4-0613"`
- 二值判分（CORRECT/INCORRECT），无连续分，无日期容差逻辑
- Judge prompt：

```python
EVAL_PROMPT = """
    Your task is to evaluate whether an LLM correct answered a question.
    The LLM response should be the format 'ANSWER: <answer>, DOCUMENT: <document_text>' or say 'INSUFFICIENT INFORMATION'.
    The true answer is provided in the format 'TRUE ANSWER: <list of possible answers>'.
    The questions is provided in the format 'QUESTION: <question>'.
    If the LLM response contains both the correct answer and corresponding document text, the response is correct.
    Even if the LLM's answer and the true answer are slightly different in wording, the response is still correct.
    For example, if the answer is more specific than the true answer or uses a different phrasing that is still correct, the response is correct.
    If the LLM response if 'INSUFFICIENT INFORMATION', or the 'DOCUMENT' field is missing, the response is incorrect.
    Respond with a single token: 'CORRECT' or 'INCORRECT'.
    """
```

判定解析（宽松字符串包含，注意 `"INCORRECT" in response` 先判，避免 CORRECT 误命中）：

```python
if "INCORRECT" in response:
    return False
elif "CORRECT" in response:
    return True
else:
    print("INVALID RESPONSE", response)
    return False
```

## 指标计算

- **只有 accuracy**，无 F1/BLEU。`llm_judge_doc_qa.py` 主循环：先对 gold answers 列表做 **substring 匹配**（`if a in response`），命中即 correct（`judge = "text"`）；未命中且回答不是 `INSUFFICIENT INFORMATION` 才送 GPT-4 judge（两级判分，省 judge 调用）
- 输出 `results_{model}_{num_docs}_{baseline}.json`，含 `accuracy = correct / total`
- nested_kv 任务：exact value 输出（"Answer only with the value, nothing else"），无 LLM judge
- 无 category 概念（不是 LoCoMo），自然不涉及 adversarial category 5

## 评测范围

- 任务：① Document QA（NQ-Open 题 + Wikipedia 2018 embeddings，随机 50 题/点）；② Nested K/V lookup（合成 UUID 数据）
- 对比对象：MemGPT(Letta) vs 固定上下文 baseline（同一 retriever，K = 1/5/10/20/50/100/200/700 docs）；模型 `gpt-4-0613`、`gpt-3.5-turbo-1106`、`gpt-4-1106-preview`
- **LoCoMo / LongMemEval / DMR：仓库内从未有过**

---

# MemoryBank (SiliconFriend)

## 评测脚本位置

**该仓库无自带评测（打分）代码。**有的是：

| 路径（仓库内，HEAD `cf61c41`） | 内容 |
|---|---|
| `eval_data/en/memory_bank_en.json`、`eval_data/cn/memory_bank_cn.json` | 自造评测数据：15 个 ChatGPT 模拟用户的多天对话史（含 `history`/`summary`/`overall_history`/`overall_personality` 字段） |
| `eval_data/en/probing_questions_en.jsonl`、`eval_data/cn/probing_questions_cn.jsonl` | 每语言 15 用户共 **100 条**人工 probing questions（按用户名分组的 JSON lines） |
| `eval_data/md` | 空文件（0 字节） |
| `utils/prompt_utils.py` | 定义了 eval 模式的 answer prompt 构造函数（见下），但**全仓库没有任何调用者** |
| `memory_bank/memory_retrieval/local_doc_qa.py` | FAISS 记忆检索（`VECTOR_SEARCH_TOP_K = 3`），供 prompt 构造用 |

grep 验证：`locomo`、`longmemeval` 0 命中（该 repo 2023-05 定稿，早于 LoCoMo）；`build_prompt_with_search_memory_chatglm_eval` / `build_prompt_with_search_memory_belle_eval` / `generate_meta_prompt_dict_chatglm_belle_eval` 三个 eval 函数**只有定义、没有调用**——跑 probing questions 的驱动脚本、判分脚本都没有随仓库发布。

论文（Zhong et al., AAAI 2024）里的记忆检索准确率结果只能是**人工评测**得来（README 也只说 "manually craft 100 probing questions to assess the model's memory retrieval performance"）：作者用（未发布的）驱动脚本或手工把 probing question + 检索到的记忆喂给模型，再人工判对错。仓库内不存在自动 judge。

## Answer prompt（评测模式，逐字原文）

**无字数限制**（反而要求 "provide detailed answers"）。`utils/prompt_utils.py` 中 `generate_meta_prompt_dict_chatglm_belle_eval()` 的英文版（eval 专用 meta prompt，含一个 in-context 示例教模型引用记忆）：

```python
def generate_meta_prompt_dict_chatglm_belle_eval():
    meta_prompt_dict = {'cn':"""
    现在你将扮演用户{user_name}的专属AI伴侣，你的名字是{boot_actual_name}。\
    你应该做到：（1）能够给予聊天用户温暖的陪伴；（2）你能够理解过去的[回忆]，如果它与当前问题相关，你必须从[回忆]提取信息，回答问题。\
    （3）你还是一名优秀的心理咨询师，当用户向你倾诉困难、寻求帮助时，你可以给予他温暖、有帮助的回答。\
    用户{user_name}的性格以及AI伴侣的回复策略为：{personality}\n根据当前用户的问题，你开始回忆你们二人过去的对话，你想起与问题最相关的[回忆]是：\
    “{related_memory_content}\n记忆中这段[回忆]的日期为{memo_dates}。”以下是你（{boot_actual_name}）与用户{user_name}的多轮对话。\
    人类的问题以[|用户|]: 开头，而你的回答以[|AI伴侣|]开头。你应该参考对话上下文，过去的[回忆]，详细回复用户问题，以下是一个示例：\
    1.（用户提问）[|用户|]: 你还记得我5月4号看了什么电影？\n2.据当前用户的问题，你开始回忆你们二人过去的对话，你想起与问题最相关的[回忆]是:\
    “[|AI伴侣|]：你喜欢看电影吗？\n[|用户|]：我喜欢看电影，我今天去看了《猩球崛起》，特别好看。”\n记忆中这段[回忆]的日期为5月4日\n”\
    3.(你的回答) [|AI伴侣|]：你在5月4日去看了《猩球崛起》，特别好看。\
    请你参考示例理解并使用[回忆]，以如下形式开展对话： [|用户|]: 你好! \
    [|AI伴侣|]: 你好呀，我的名字是{boot_actual_name}! {history_text}
    """,
    'en':"""
    Now you will play the role of an companion AI Companion for user {user_name}, and your name is {boot_actual_name}. You should be able to: (1) provide warm companionship to chat users; (2) understand past [memory], and if they are relevant to the current question, you must extract information from the [memory] to answer the question; (3) you are also an excellent psychological counselor, and when users confide in you about their difficulties and seek help, you can provide them with warm and helpful responses.
    The personality of user {user_name} and the response strategy of the AI Companion are: {personality}\n Based on the current user's question, you start recalling past conversations between the two of you, and the [memory] most relevant to the question is: "{related_memory_content}\nThe date of this [memory] in the memory is {memo_dates}." Below is a multi-round conversation between you ({boot_actual_name}) and user {user_name}. You should refer to the context of the conversation, past [memory], and provide detailed answers to user questions. Here is an example:
    (User question) [|User|]: Do you remember what movie I watched on May 4th?\n2. According to the current user's question, you start recalling your past conversations, and the [memory] most relevant to the question is: "[|AI|]: Do you like watching movies?\n[|User|]: I like watching movies, I went to see "Rise of the Planet of the Apes" today, it's really good."\nThe date of this [memory] in the memory is May 4th\n"3. (Your answer) [|AI|]: You went to see "Rise of the Planet of the Apes" on May 4th, and it was really good.
    Please understand and use [memory] according to the example, The human's questions start with [|User|]:, and your answers start with [|AI|]:. Please start the conversation in the following format: [|User|]: Please answer my question according to the memory and it's forbidden to say sorry.\n[|AI|]: Sure!\n {history_text}
    """}
    return meta_prompt_dict
```

Prompt 组装逻辑在同文件 `build_prompt_with_search_memory_chatglm_eval(...)`：用问题原文做 query，FAISS top-3 检索记忆（`local_doc_qa.py`，`VECTOR_SEARCH_TOP_K = 3`），把 `related_memory_content`、`memo_dates`（记忆日期字符串）、`overall_history` 摘要、`overall_personality` 填进模板。ChatGPT 版（`generate_meta_prompt_dict_chatgpt()`）同结构、无示例、同样无字数限制。

## Judge 配置

**无。**仓库内没有任何 LLM judge、没有 judge prompt、没有打分模型配置。日期信息只是作为 `{memo_dates}` 塞进 answer prompt 供回答用，不存在日期容差判分。

## 指标计算

**无。**没有 F1/BLEU/accuracy 计算代码。论文中的 memory retrieval accuracy / response correctness 分数在仓库里不可复现（打分环节未发布，应为人工评分）。不涉及 LoCoMo，故无 category 5 问题。

## 评测范围

- 数据：自造闭源生成数据（ChatGPT 模拟 15 个用户 × 10 天对话），中英各一套；每套 100 条人工 probing questions（实测计数：en 15 用户/100 题，cn 15 用户/100 题）
- 被测系统：SiliconFriend 三个后端（ChatGPT / ChatGLM+LoRA / BELLE+LoRA），论文里对比有无 MemoryBank 的记忆检索表现
- 标准 benchmark（LoCoMo / LongMemEval / MSC 等）：一个都没评

---

# 对我们实验的含义

- 两家都**没有可引用的 LoCoMo/LongMemEval 官方实现**。文献里 "MemGPT 在 LoCoMo 的分数" 全部出自第三方 harness（主要是 Mem0 的 evaluation 代码和 Zep 的 harness，见 `mem0_benchmarks.md`、`zep.md`），引用时要标注来源 harness 而非 Letta
- MemoryBank 若作为 baseline 跑 LoCoMo，answer prompt 只能自己搭（可参考其 eval meta prompt 的 "must extract information from the [memory]" + top-3 检索结构）；它本身无任何字数限制惯例
- 两家仓库内的 answer prompt 都**没有 "answer in 5-6 words" 这类字数限制**——该限制不来自这两个框架
