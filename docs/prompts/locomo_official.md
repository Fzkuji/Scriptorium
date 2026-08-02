# LoCoMo 官方评测（locomo）

代码根目录：`code/locomo/`。QA 评测入口为 `task_eval/evaluate_qa.py`，按模型分发到 `gpt_utils.py` / `claude_utils.py` / `gemini_utils.py` / `hf_llm_utils.py`。以下 prompt 均为逐字原文（含源码中的拼写错误如 "wriiten"、行尾空格、`\n` 转义符按源码原样保留）。

## 构建/记忆提取 prompt

LoCoMo 官方评测本身不做"记忆构建"，但 RAG 评测模式（`--rag-mode summary/observation`）需要先离线生成 session 摘要和 observation（事实）库，使用以下 prompt。

### Session 摘要 prompt（RAG summary 模式）

来源：`locomo/task_eval/get_session_summaries.py`，函数 `get_summary_query`（拼接在对话文本之前的指令，源码中以 `query` 变量拼接）

```text
Generate a concise summary of the following conversation using exact words from the conversation wherever possible. The summary should contain all facts about the two speakers, as well as references to time.\n
```

（源码为 `query = "Generate a concise summary of ... references to time.\n"`，随后 `query = query + conv + "\n"`，即指令在前、对话在后。）

### Observation（事实）提取 prompt（RAG observation 模式）

来源：`locomo/generative_agents/memory_utils.py`，变量 `CONVERSATION2FACTS_PROMPT`（由 `task_eval/get_facts.py` 经 `get_session_facts` 调用；few-shot 示例来自 `prompt_examples/fact_generation_examples_new.json`，其 `input_prefix` 为 `CONVERSATION: `，`output_prefix` 为 `OBSERVATIONS:`）

```text
Write a concise and short list of all possible OBSERVATIONS about each speaker that can be gathered from the CONVERSATION. Each dialog in the conversation contains a dialogue id within square brackets. Each observation should contain a piece of information about the speaker, and also include the dialog id of the dialogs from which the information is taken. The OBSERVATIONS should be objective factual information about the speaker that can be used as a database about them. Avoid abstract observations about the dynamics between the two speakers such as 'speaker is supportive', 'speaker appreciates' etc. Do not leave out any information from the CONVERSATION. Important: Escape all double-quote characters within string output with backslash.\n\n
```

### 反思（reflection）prompt（生成式 agent 记忆，评测未直接使用）

来源：`locomo/generative_agents/memory_utils.py`，变量 `REFLECTION_INIT_PROMPT`

```text
{}\n\nGiven the information above, what are the three most salient insights that {} has about {}? Give concise answers in the form of a json list where each entry is a string.
```

来源：`locomo/generative_agents/memory_utils.py`，变量 `REFLECTION_CONTINUE_PROMPT`

```text
{} has the following insights about {} from previous interactions.{}\n\nTheir next conversation is as follows:\n\n{}\n\nGiven the information above, what are the three most salient insights that {} has about {} now? Give concise answers in the form of a json list where each entry is a string.
```

来源：`locomo/generative_agents/memory_utils.py`，变量 `SELF_REFLECTION_INIT_PROMPT`

```text
{}\n\nGiven the information above, what are the three most salient insights that {} has about self? Give concise answers in the form of a json list where each entry is a string.
```

来源：`locomo/generative_agents/memory_utils.py`，变量 `SELF_REFLECTION_CONTINUE_PROMPT`

```text
{} has the following insights about self.{}\n\n{}\n\nGiven the information above, what are the three most salient insights that {} has about self now? Give concise answers in the form of a json list where each entry is a string.
```

## Answer prompt（QA 回答）

### 对话上下文开头 prompt（所有 API 模型共用，RAG 与非 RAG 均在非 RAG 时使用；RAG 模式下上下文换成检索结果，不加此前缀）

来源：`locomo/task_eval/gpt_utils.py`、`claude_utils.py`、`gemini_utils.py`、`hf_llm_utils.py`，变量 `CONV_START_PROMPT`（四个文件中逐字相同）

```text
Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n
```

### 单题 QA prompt（batch_size=1，非 RAG 与 RAG 共用；最终 query 为 `上下文 + '\n\n' + QA_PROMPT.format(question)`）

来源：`locomo/task_eval/gpt_utils.py`、`claude_utils.py`、`gemini_utils.py`，变量 `QA_PROMPT`（三个文件中逐字相同）

```text
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
```

### 单题 QA prompt —— 第 5 类（对抗/无法回答）问题专用

来源：`locomo/task_eval/gpt_utils.py`、`claude_utils.py`、`gemini_utils.py`，变量 `QA_PROMPT_CAT_5`（三个文件中逐字相同）

```text
Based on the above context, answer the following question.

Question: {} Short answer:
```

### 批量 QA prompt（batch_size>1，仅非 RAG）—— GPT 版

来源：`locomo/task_eval/gpt_utils.py`，变量 `QA_PROMPT_BATCH`（注意第一、二行行尾各有一个空格，按原文保留）

```text
Based on the above conversations, write short answers for each of the following questions in a few words. 
Write the answers in the form of a json dictionary where each entry contains the question number as "key" and the short answer as "value". 
Use single-quote characters for named entities and double-quote characters for enclosing json elements. Answer with exact words from the conversations whenever possible.

```

### 批量 QA prompt —— Claude 版

来源：`locomo/task_eval/claude_utils.py`，变量 `QA_PROMPT_BATCH`

```text
Based on the above conversations, write short answers for each of the following questions in a few words. Write the answers in the form of a json dictionary where each entry contains the string format of question number as 'key' and the short answer as value. Use single-quote characters for named entities. Answer with exact words from the conversations whenever possible.

```

### 批量 QA prompt —— Gemini 版

来源：`locomo/task_eval/gemini_utils.py`，变量 `QA_PROMPT_BATCH`

```text
Based on the above conversations, write short answers for each of the following questions in a few words. Write the answers in the form of a json dictionary where each entry contains the question number as 'key' and the short answer as value. Use single-quote characters for named entities. Answer with exact words from the conversations whenever possible.

```

（批量模式下问题列表拼接方式为 `QA_PROMPT_BATCH + "\n".join(["%s: %s" % (k, q) for k, q in enumerate(questions)])`，最终 query 为 `query_conv + '\n' + question_prompt`。）

### 单题 QA prompt —— HuggingFace 开源模型版（llama/mistral/gemma）

来源：`locomo/task_eval/hf_llm_utils.py`，变量 `QA_PROMPT`

```text
Based on the above conversations, write a short answer for the following question in a few words. Do not write complete and lengthy sentences. Answer with exact words from the conversations whenever possible.

Question: {}
```

### 批量 QA prompt —— HuggingFace 版（代码中定义但批量分支 `raise NotImplementedError`，实际未启用）

来源：`locomo/task_eval/hf_llm_utils.py`，变量 `QA_PROMPT_BATCH`

```text
Based on the above conversations, write short answers for each of the following questions in a few words. Write the answers in the form of a json dictionary where each entry contains the question number as 'key' and the short answer as value. Answer with exact words from the conversations whenever possible.

```

### HuggingFace 模型 system prompt

来源：`locomo/task_eval/hf_llm_utils.py`，变量 `LLAMA2_CHAT_SYSTEM_PROMPT`（`LLAMA3_CHAT_SYSTEM_PROMPT` 与之逐字相同）

```text
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant whose job is to understand the following conversation and answer questions based on the conversation.
If you don't know the answer to a question, please don't share false information.
<</SYS>>

{} [/INST]
```

来源：`locomo/task_eval/hf_llm_utils.py`，函数 `run_llama` 中 `apply_chat_template` 的 system 消息（实际生效的版本）

```text
You are a helpful, respectful and honest assistant whose job is to understand the following conversation and answer questions based on the conversation. If you don't know the answer to a question, please don't share false information.
```

来源：`locomo/task_eval/hf_llm_utils.py`，变量 `MISTRAL_INSTRUCT_SYSTEM_PROMPT`

```text
<s>[INST] {} [/INST]
```

来源：`locomo/task_eval/hf_llm_utils.py`，变量 `GEMMA_INSTRUCT_PROMPT`

```text
<bos><start_of_turn>user
{}<end_of_turn>
```

### 按问题类别附加的后缀（拼接到问题文本上）

来源：`locomo/task_eval/gpt_utils.py` / `claude_utils.py` / `gemini_utils.py` / `hf_llm_utils.py`，函数 `get_gpt_answers` / `get_claude_answers` / `get_gemini_answers` / `get_hf_answers` 内的 category 分支。

第 2 类（时间推理）问题后缀（四个文件逐字相同）：

```text
 Use DATE of CONVERSATION to answer with an approximate date.
```

第 5 类（对抗）问题后缀 —— GPT/Claude/Gemini 版（`{}` 处随机填入 `Not mentioned in the conversation` 与真实答案）：

```text
 Select the correct answer: (a) {} (b) {}. 
```

第 5 类问题后缀 —— HuggingFace 版（`{}` 处随机填入 `No information available` 与真实答案）：

```text
 (a) {} (b) {}. Select the correct answer by writing (a) or (b).
```

### RAG 变体说明

RAG 模式（`--use-rag`，仅 `gpt_utils.py` 实现，batch_size 必须为 1）没有独立的 QA prompt，复用上面的 `QA_PROMPT` / `QA_PROMPT_CAT_5`；区别仅在上下文来源：由 `get_rag_context`（`locomo/task_eval/gpt_utils.py`）检索 top-k 条目，按 `日期 + ': ' + 条目` 拼接（dialog/observation 模式用 `\n` 连接，summary 模式用 `\n\n` 连接），不加 `CONV_START_PROMPT` 前缀。Claude/Gemini 的 RAG 分支为 `raise NotImplementedError`。

## Judge prompt

无。LoCoMo 官方评测没有 LLM judge。`locomo/task_eval/evaluation.py` 全部使用规则指标：token 级 F1（Porter 词干化，`f1_score`/`f1`）、exact match（`exact_match_score`）、BERTScore（`bert_score`）、ROUGE-L（`rougel_score`）；第 5 类对抗问题用子串规则判定（预测中含 `no information available` 或 `not mentioned` 记 1 分，否则 0 分）。`evaluation_stats.py` 只做分数聚合统计，也不含任何 LLM 调用。

## 其他（数据集构建 prompt，非评测用）

`locomo/prompt_examples/` 目录下的文件主要是数据集构建阶段的 few-shot 示例库（`persona_generation_examples.json`、`event_generation_examples.json`、`sub_event_generation_examples.json`、`graph_generation_examples.json`、`visual_graph_generation_examples.json`、`fact_generation_examples.json`、`fact_generation_examples_new.json`），这些文件本身只含 `input_prefix`/`output_prefix`（如 `PERSONA: `、`CONVERSATION: `/`FACT:`、`INPUT: `/`OUTPUT:`、`Attributes: `/`Persona: `）加示例数据，指令文本在调用代码（`generative_agents/` 下）中。`chatgpt_instructions.json` 是早期用 ChatGPT 网页手工构建数据的指令记录，未被任何 Python 代码引用。与评测无关，此处仅收录两个在代码中实际使用的独立指令 prompt：

### 图片分享对话改写 prompt

来源：`locomo/prompt_examples/image_sharing_examples.json`，键 `prompt`（由 `generative_agents/conversation_utils.py` 使用）

```text
Modify speaker's DIALOG to include context from the CAPTION of the photo shared the speaker with a certain INTENT.
```

其输入格式（键 `input_format`）：

```text
DIALOG: {}\nCAPTION: {}\nINTENT:{}\nMODIFIED_DIALOG: 
```

### 去上下文重写 prompt

来源：`locomo/prompt_examples/remove_context_examples.json`，键 `prompt`（由 `generative_agents/generate_conversations.py` 使用；注意第 1 条规则行尾有一个空格，按原文保留）

```text
Rewrite the DIALOG according to the following rules:
1. If some words or information in DIALOG can be inferred from CONTEXT, remove or replace those words in DIALOG with indirect references like 'that', 'it' etc. 
2. Replace references to objects mentioned in CAPTION using indirect references like 'this', 'that' etc.
- Remove repetitions of questions already asked in CONTEXT
```

其输入格式（键 `input_format` / `input_format_w_image`）：

```text
CONTEXT: {}\n\nDIALOG: {}\n\nOUTPUT: 
```

```text
CONTEXT: {}\n\nDIALOG: {}\n\nCAPTION: {}\n\nOUTPUT: 
```
