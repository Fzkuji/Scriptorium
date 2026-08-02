# LongMemEval 官方评测

代码根目录：`code/longmemeval/src/`。所有 prompt 均为逐字原文（Python 字符串已解转义，`{}` 为 `str.format` 占位符）。

## 构建/记忆提取 prompt

LongMemEval 官方没有"构建记忆库"阶段，但 `index_expansion/` 目录下有一组用于检索索引扩展的提取 prompt（会话摘要、关键词、用户事实、时间事件），性质等同于记忆提取。

### 会话摘要（session summary）

来源：`longmemeval/src/index_expansion/batch_expansion_session_summ.py`，函数 `summarize_session` 内变量 `summarization_prompt`（第 23 行，拼接在对话文本之前）

```text
Below is a transcript of a conversation between a human user and an AI assistant. Please summarize the following dialogue as concisely as possible in a short paragraph, extracting the main themes and key information. In your summary, focus more on what the user mentioned or asked for. Dialogue content:
```

### 会话关键词（session keyphrases）

来源：`longmemeval/src/index_expansion/batch_expansion_session_keyphrases.py`，变量 `summarization_prompt`（第 21 行）

```text
Below is a transcript of a conversation between a human user and an AI assistant. Generate a list of keyphrases for the session. Separate each keyphrase with a semicolon. Dialogue content:
```

### 轮次关键词（turn keyphrases）

来源：`longmemeval/src/index_expansion/batch_expansion_turn_keyphrases.py`，变量 `summarization_prompt`（第 20 行）

```text
Below is a transcript of a round of conversation between a human user and an AI assistant. Generate a list of keyphrases for the round. Separate each keyphrase with a semicolon. Dialogue content:
```

### 会话级用户事实提取（session user facts）

来源：`longmemeval/src/index_expansion/batch_expansion_session_userfact.py`，函数 `extract_session_userfact` 内变量 `system_prompt`（第 20 行）

```text
You will be given a list of messages from a human user to an AI assistant. Extract all the personal information, life events, experience, and preferences related to the user. Make sure you include all details such as life events, personal experience, preferences, specific numbers, locations, or dates. State each piece of information in a simple sentence. Put these sentences in a json list, each element being a standalone personal fact about the user. Minimize the coreference across the facts, e.g., replace pronouns with actual entities. If there is no specific events, personal information, or preference mentioned, just generate an empty list.
```

来源：同文件，变量 `user_prompt`（第 22 行）

```text
Human user messages:
{}

Personal facts about the user (a list of strings in json format; do not generate anything else):
```

### 轮次级用户事实提取（turn user facts）

来源：`longmemeval/src/index_expansion/batch_expansion_turn_userfact.py`，变量 `system_prompt`（第 21 行）

```text
You will be given a message from a human user to an AI assistant. Extract all the personal information, life events, experience, and preferences related to the user. Make sure you include all details such as life events, personal experience, preferences, specific numbers, locations, or dates. State each piece of information in a simple sentence. Put these sentences in a json list, each element being a standalone personal fact about the user. Minimize the coreference across the facts, e.g., replace pronouns with actual entities. If there is no specific events, personal information, or preference mentioned, just generate an empty list.
```

来源：同文件，变量 `user_prompt`（第 23 行）

```text
Human user message:
{}

Personal facts about the user (a list of strings in json format; do not generate anything else):
```

### 会话时间事件提取（temporal events）

来源：`longmemeval/src/index_expansion/batch_expansion_session_temp_event.py`，变量 `system_prompt`（第 20 行）

```text
You will be given a list of messages from a human user to an AI assistant, as well as the time the conversation took place. Extract all events related to the user as long as its date is specified or could be inferred. If the time some event took place cannot be inferred, do not extract that event. Return the events in a json list where each item contains two fields: "date" and "event". Write date in the form YYYY/MM/DD. If there is no specific event, just write an empty list.
```

来源：同文件，变量 `user_prompt`（第 22 行）

```text
Conversation date: {}
Human user messages:
{}

Personal facts about the user with dates (a list of dicts in json format; do not generate anything else):
```

### 时间范围查询剪枝（temporal query pruning）

来源：`longmemeval/src/index_expansion/temp_query_search_pruning.py`，变量 `system_prompt`（第 40 行；原文含拼写错误 "referencea"，保留）

```text
You will be given a question from a human user asking about some prvious events, as well as the time the question is asked. Infer a potential time range such that the events happening in this range is likely to help to answer the question (a start date and an end date). Write a json dict two fields: "start" and "end". Write date in the form YYYY/MM/DD. If the question does not have any temporal referencea, do not attempt to guess a time range. Instead, just say N/A.
```

来源：同文件，变量 `user_prompt`（第 42 行）

```text
Question date: {}
Question:
{}

Relevant Date Range(dict in json format; do not generate anything else):
```

## Answer prompt（QA 回答 prompt）

均来自 `longmemeval/src/generation/run_generation.py`，函数 `prepare_prompt` 内变量 `answer_prompt_template`。占位符依次为 history_string、question_date、question（no-retrieval 模式只有 question）。

### no-retrieval（无检索，直接回答）

来源：`longmemeval/src/generation/run_generation.py`，`prepare_prompt`，`retriever_type == 'no-retrieval'` 分支（第 48 行）

```text
{}
```

CoT 变体（第 50 行，直接拼接在上面模板之后，即完整模板为 `{}Answer step by step.`）：

```text
Answer step by step.
```

### 标准检索模式（merge_key_expansion_into_value == 'none'），非 CoT

来源：`longmemeval/src/generation/run_generation.py`，`prepare_prompt`（第 57 行）

```text
I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.


History Chats:

{}

Current Date: {}
Question: {}
Answer:
```

### 标准检索模式，CoT

来源：同文件，`prepare_prompt`（第 55 行）

```text
I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.


History Chats:

{}

Current Date: {}
Question: {}
Answer (step by step):
```

### 事实扩展 merge 模式（原始会话 + 提取的用户事实），非 CoT

来源：同文件，`prepare_prompt`（第 62 行）

```text
I will give you several history chats between you and a user, as well as the relevant user facts extracted from the chat history. Please answer the question based on the relevant chat history and the user facts


History Chats:

{}

Current Date: {}
Question: {}
Answer:
```

### 事实扩展 merge 模式，CoT

来源：同文件，`prepare_prompt`（第 60 行）

```text
I will give you several history chats between you and a user, as well as the relevant user facts extracted from the chat history. Please answer the question based on the relevant chat history and the user facts. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.


History Chats:

{}

Current Date: {}
Question: {}
Answer (step by step):
```

### 事实扩展 replace 模式（仅用提取的事实替代原始会话），非 CoT

来源：同文件，`prepare_prompt`（第 67 行）

```text
I will give you several facts extracted from history chats between you and a user. Please answer the question based on the relevant facts.


History Chats:

{}

Current Date: {}
Question: {}
Answer:
```

### 事实扩展 replace 模式，CoT

来源：同文件，`prepare_prompt`（第 65 行）

```text
I will give you several facts extracted from history chats between you and a user. Please answer the question based on the relevant facts. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.


History Chats:

{}

Current Date: {}
Question: {}
Answer (step by step):
```

## Judge prompt

均来自 `longmemeval/src/evaluation/evaluate_qa.py`，函数 `get_anscheck_prompt(task, question, answer, response, abstention=False)` 内变量 `template`。三个占位符依次为 question、answer（abstention 时为 explanation，preference 时为 rubric）、model response。判定方式：判 "yes" 是否出现在裁判回复（小写）中。

### single-session-user / single-session-assistant / multi-session

来源：`longmemeval/src/evaluation/evaluate_qa.py`，`get_anscheck_prompt`（第 27 行）

```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. 

Question: {}

Correct Answer: {}

Model Response: {}

Is the model response correct? Answer yes or no only.
```

### temporal-reasoning

来源：同文件，`get_anscheck_prompt`（第 30 行）

```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. 

Question: {}

Correct Answer: {}

Model Response: {}

Is the model response correct? Answer yes or no only.
```

### knowledge-update

来源：同文件，`get_anscheck_prompt`（第 33 行）

```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {}

Correct Answer: {}

Model Response: {}

Is the model response correct? Answer yes or no only.
```

### single-session-preference

来源：同文件，`get_anscheck_prompt`（第 36 行）

```text
I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: {}

Rubric: {}

Model Response: {}

Is the model response correct? Answer yes or no only.
```

### abstention（question_id 含 `_abs` 时使用，覆盖所有题型）

来源：同文件，`get_anscheck_prompt`（第 41 行，`abstention=True` 分支）

```text
I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {}

Explanation: {}

Model Response: {}

Does the model correctly identify the question as unanswerable? Answer yes or no only.
```

## 其他

### CoN（Chain-of-Note）阅读笔记提取 prompt

回答阶段可选步骤（`--con true`）：先对每个检索到的会话提取与问题相关的笔记，再把笔记作为 history 喂给 answer prompt。占位符依次为 session date、session content（JSON）、question date、question。原文含笔误 "answering the answer"，保留。

来源：`longmemeval/src/generation/run_generation.py`，函数 `prepare_prompt` 内变量 `con_prompt`（第 195 行）

```text
I will give you a chat history between you and a user, as well as a question from the user. Write reading notes to extract all the relevant user information relevant to answering the answer. If no relevant information is found, just output "empty". 


Chat History:
Session Date: {}
Session Content:
{}

Question Date: {}
Question: {}
Extracted note (information relevant to answering the question):
```
