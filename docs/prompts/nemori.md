# Nemori (nemori-ai/nemori)

来源仓库：https://github.com/nemori-ai/nemori（main 分支，通过 raw.githubusercontent.com 抓取原文，2026-07-02）

Nemori 是面向 agentic LLM 工作流的长期记忆系统：把对话切分成 episode（情景记忆），再蒸馏出 semantic 语义知识，提供统一检索。核心 prompt 集中在 `nemori/llm/prompts.py` 的 `PromptTemplates` 类；评测（LoCoMo / LongMemEval）的 answer / judge prompt 在 `evaluation/` 目录。

---

## 构建 / 记忆提取 prompt

### 1. Episode 生成 prompt

来源：`nemori/llm/prompts.py`，`PromptTemplates.EPISODE_GENERATION_PROMPT`

```text
You are an episodic memory generation expert. Please convert the following conversation into an episodic memory.

Conversation content:
{conversation}

Boundary detection reason:
{boundary_reason}

Please analyze the conversation to extract time information and generate a structured episodic memory. Return only a JSON object containing the following three fields:
{{
    "title": "A concise, descriptive title that accurately summarizes the theme (10-20 words)",
    "content": "A detailed description of the conversation in third-person narrative. It must include all important information: who participated in the conversation at what time, what was discussed, what decisions were made, what emotions were expressed, and what plans or outcomes were formed. Write it as a coherent story so that the reader can clearly understand what happened. Ensure that time information is precise to the hour, including year, month, day, and hour.",
    "timestamp": "YYYY-MM-DDTHH:MM:SS format timestamp representing when this episode occurred (analyze from message timestamps or content)"
}}

Time Analysis Instructions:
1. **Primary Source**: Look for explicit timestamps in the message metadata or content
2. **Secondary Source**: Analyze temporal references in the conversation content ("yesterday", "last week", "this morning", etc.)
3. **Fallback**: If no time information is available, use a reasonable estimate based on context
4. **Format**: Always return timestamp in ISO format: "2024-01-15T14:30:00"

Requirements:
1. The title should be specific and easy to search (including key topics/activities).
2. The content must include all important information from the conversation.
3. Convert the dialogue format into a narrative description.
4. Maintain chronological order and causal relationships.
5. Use third-person unless explicitly first-person.
6. Include specific details that aid keyword search.
7. Notice the time information, and write the time information in the content.
8. When relative times (e.g., last week, next month, etc.) are mentioned in the conversation, you need to convert them to absolute dates (year, month, day). Write the converted time in parentheses after the original time reference.
9. **IMPORTANT**: Analyze the actual time when the conversation happened from the message timestamps or content, not the current time.

Example:
If the conversation is about someone planning to go hiking and the messages have timestamps from March 14, 2024 at 3:00 PM:
{{
    "title": "Weekend Hiking Plan March 16, 2024: Sunrise Trip to Mount Rainier",
    "content": "On March 14, 2024 at 3:00 PM, the user expressed interest in going hiking on the upcoming weekend (March 16, 2024) and sought advice. They particularly wanted to see the sunrise at Mount Rainier, having heard the scenery is beautiful. When asked about gear, they received suggestions including hiking boots, warm clothing (as it's cold at the summit), a flashlight, water, and high-energy food. The user decided to leave at 4:00 AM on Saturday, March 16, 2024 to catch the sunrise and planned to invite friends for the adventure. They were very excited about the trip, hoping to connect with nature.",
    "timestamp": "2024-03-14T15:00:00"
}}

Return only the JSON object, do not add any other text:
```

调用时的 system message（来源：`nemori/llm/generators/episode.py`，`EpisodeGenerator.generate`）：

```text
You are an episodic memory generation expert.
```

### 2. 批量对话分段 prompt

来源：`nemori/llm/prompts.py`，`PromptTemplates.BATCH_SEGMENTATION_PROMPT`

```text
You are an intelligent conversation segmentation expert. Your task is to analyze a batch of messages and group them into coherent episodes.

## Input Messages
You will receive {count} messages numbered from 1 to {count}:

{messages}

## Your Task
Analyze these messages and group them into coherent episodes with **HIGH SENSITIVITY** to topic shifts. Be strict and create NEW episodes when detecting:

1. **Topic Change** (Highest Priority):
   - Do the new messages introduce a completely different topic?
   - Is there a shift from one specific event to another?
   - Has the conversation moved from one question to an unrelated new question?

2. **Intent Transition**:
   - Has the purpose of the conversation changed? (e.g., from casual chat to seeking help, from discussing work to discussing personal life)
   - Has the core question or issue of the current topic been answered or fully discussed?

3. **Temporal Markers**:
   - Are there temporal transition markers ("earlier", "before", "by the way", "oh right", "also", etc.)?
   - Is the time gap between messages more than 30 minutes?

4. **Structural Signals**:
   - Are there explicit topic transition phrases ("changing topics", "speaking of which", "quick question", etc.)?
   - Are there concluding statements indicating the current topic is finished?

5. **Content Relevance**:
   - How related is the new message to the previous discussion? (Consider splitting if relevance < 30%)
   - Does it involve completely different people, places, or events?

Decision Principles:
- **Prioritize topic independence**: Each episode should revolve around one core topic or event
- **When in doubt, split**: When uncertain, lean towards starting a new episode
- **Maintain reasonable length**: A single episode typically shouldn't exceed 10-15 messages

## Output Format
Return a JSON object with episodes, where each episode contains:
- `indices`: List of message numbers (1-based) belonging to this episode
- `topic`: Brief, specific description of what this episode is about

Example output:
{{
    "episodes": [
        {{
            "indices": [1, 2, 3, 4],
            "topic": "Discussion about weekend hiking plans"
        }},
        {{
            "indices": [5, 6, 7],
            "topic": "Questions about Python programming"
        }},
        {{
            "indices": [8, 9],
            "topic": "Work schedule discussion"
        }}
    ]
}}

## Important Guidelines
- Episodes can have non-consecutive indices if messages are interleaved
- An episode should typically contain 2-15 messages
- Focus on topical coherence over strict chronological order
- When in doubt, prefer smaller, more focused episodes

Return only the JSON object, no additional text.
```

### 3. 预测 prompt（Prediction-Correction-Refinement 第一步）

来源：`nemori/llm/prompts.py`，`PromptTemplates.PREDICTION_PROMPT`

```text
You are a knowledge-based episode prediction system. Your task is to reconstruct a complete conversation episode based on limited clues and your knowledge base.

IMPORTANT: You are predicting the ACTUAL CONTENT and KNOWLEDGE of what happened, not the writing style or format.

## Input Information

**Episode Title/Summary**: {episode_title}

**Relevant Knowledge Statements** (your current world model):
{knowledge_statements}

## Your Task

Based on the above clues, reconstruct what you believe happened in this episode. Focus on:
1. **Core Facts**: What specific information was discussed?
2. **Key Decisions**: What choices or conclusions were made?
3. **Knowledge Exchange**: What knowledge was shared or learned?
4. **Logical Flow**: How did the conversation progress?

## What to IGNORE
- Writing style or level of detail
- Specific formatting or structure
- Exact phrasing or word choices
- Whether timestamps are included in the text
- How formal or casual the language is

## Output Format

Generate a natural narrative that captures what you predict happened. Write it as if you're describing the episode to someone else. Focus on the SUBSTANCE, not the STYLE.

Your prediction:
```

### 4. 对比提取知识 prompt（Prediction-Correction-Refinement 第二步）

来源：`nemori/llm/prompts.py`，`PromptTemplates.EXTRACT_KNOWLEDGE_FROM_COMPARISON_PROMPT`

```text
You are extracting valuable knowledge by comparing original conversation with predicted content.

## Original Conversation:
{original_messages}

## Predicted Summary:
{predicted_episode}

## Your Task:
Extract ONLY the valuable knowledge that exists in the original but is missing or misrepresented in the prediction.

## CRITICAL: Focus on HIGH-VALUE Knowledge Only

Extract ONLY knowledge that passes these criteria:
- **Persistence Test**: Will this still be true in 6 months?
- **Specificity Test**: Does it contain concrete, searchable information?
- **Utility Test**: Can this help predict future user needs or preferences?
- **Independence Test**: Can this be understood without the conversation context?

## HIGH-VALUE Knowledge Categories (EXTRACT THESE):
1. **Identity & Background**: Names, professions, companies, education
2. **Persistent Preferences**: Favorite books/movies/tools, long-term likes/dislikes
3. **Technical Details**: Technologies, versions, methodologies, architectures
4. **Relationships**: Family, colleagues, team members, mentors
5. **Goals & Plans**: Career objectives, learning goals, project plans
6. **Beliefs & Values**: Principles, philosophies, strong opinions
7. **Habits & Patterns**: Regular activities, workflows, schedules

## LOW-VALUE Knowledge (SKIP THESE):
- Temporary emotions or reactions
- Single conversation acknowledgments
- Vague statements without specifics
- Context-dependent information

## Guidelines:
1. Each statement should be self-contained and atomic
2. Include ALL specific details (names, versions, titles)
3. Use present tense for persistent facts
4. Focus on facts that help understand the user long-term
5. DO NOT include time/date information in the statement
6. Quality over quantity - fewer valuable statements are better

## Examples:
GOOD: "Caroline's favorite book is 'Becoming Nicole' by Amy Ellis Nutt"
GOOD: "The user works at ByteDance as a senior ML engineer"
BAD: "The user thanked the assistant"
BAD: "The user was happy about the response"

## Output Format (JSON):
{{
    "statements": [
        "First factual statement extracted from the gap",
        "Second factual statement extracted from the gap",
        "..."
    ]
}}

Important:
- Each statement should be self-contained and understandable without context
- Use present tense for persistent facts
- Include specific names, titles, and details
- Focus on quality over quantity - only extract truly valuable knowledge
```

### 5. 语义记忆生成 prompt（fallback 直接模式）

来源：`nemori/llm/prompts.py`，`PromptTemplates.SEMANTIC_GENERATION_PROMPT`

```text
You are an AI memory system. Extract HIGH-VALUE, PERSISTENT semantic memories from the following episodes.

CRITICAL: Focus on extracting LONG-TERM VALUABLE KNOWLEDGE, not temporary conversation details.

Episodes to analyze:
{episodes}

## HIGH-VALUE Knowledge Criteria

Extract ONLY knowledge that passes these tests:
- **Persistence Test**: Will this still be true in 6 months?
- **Specificity Test**: Does it contain concrete, searchable information?
- **Utility Test**: Can this help predict future user needs?
- **Independence Test**: Can be understood without conversation context?

## HIGH-VALUE Categories (FOCUS ON THESE):

1. **Identity & Professional**
   - Names, titles, companies, roles
   - Education, qualifications, skills

2. **Persistent Preferences**
   - Favorite books, movies, music, tools
   - Technology preferences with reasons
   - Long-term likes and dislikes

3. **Technical Knowledge**
   - Technologies used (with versions)
   - Architectures, methodologies
   - Technical decisions and rationales

4. **Relationships**
   - Names of family, colleagues, friends
   - Team structure, reporting lines
   - Professional networks

5. **Goals & Plans**
   - Career objectives
   - Learning goals
   - Project plans

6. **Patterns & Habits**
   - Regular activities
   - Workflows, schedules
   - Recurring challenges

## Examples:

HIGH-VALUE (Extract these):
- "Caroline's favorite book is 'Becoming Nicole' by Amy Ellis Nutt"
- "The user works at ByteDance as a senior ML engineer"
- "The user prefers PyTorch over TensorFlow for debugging"
- "The user's team lead is named Sarah"
- "The user is learning Rust for systems programming"
- "The user has been practicing yoga since March 2021"
- "The user joined Amazon in August 2020 as a data scientist"
- "The user plans to relocate to Seattle in January 2025"

LOW-VALUE (Skip these):
- "The user thanked the assistant"
- "The user was confused about X"
- "The user appreciated the help"
- "The conversation was productive"
- Any temporary emotions or reactions

## Output Format

Return ONLY high-value knowledge in JSON format:
{{
    "statements": [
        "First high-value persistent fact...",
        "Second high-value persistent fact...",
        "Third high-value persistent fact..."
    ]
}}

Quality over quantity - extract only knowledge that truly helps understand the user long-term.
```

### 6. Episode 合并决策 prompt

来源：`nemori/llm/prompts.py`，`PromptTemplates.MERGE_DECISION_PROMPT`

```text
You are an episodic memory merge decision expert. Determine if a new episode should be merged with an existing similar episode.

## New Episode
Time Range: {new_time_range}
Content: {new_content}

## Candidate Episodes to Merge With
{candidates}

## Your Task
Decide whether the new episode should:
1. **merge**: Merge with one of the candidates (they describe the same event/topic)
2. **new**: Keep as a separate new episode (it's a distinct event)

## Merge Criteria
Merge ONLY if:
- Both episodes describe the SAME event or conversation session
- They have significant temporal overlap or are very close in time
- The content is clearly a continuation or different perspective of the same topic
- Merging would create a more complete picture without mixing different events

Do NOT merge if:
- They are different events/conversations even if on similar topics
- They are separated by significant time gaps (>1 hour)
- They involve different contexts or participants

## Output Format
Return JSON:
{{
    "decision": "merge" or "new",
    "merge_target_id": "episode_id_to_merge_with" (only if decision is "merge", otherwise null),
    "reason": "Brief explanation of your decision"
}}

Return only the JSON object, no additional text.
```

### 7. Episode 合并内容生成 prompt

来源：`nemori/llm/prompts.py`，`PromptTemplates.MERGE_CONTENT_PROMPT`

```text
You are an episodic memory merge content generator. Combine two related episodes into a single, coherent episode.

## Original Episode
Time Range: {original_time_range}
Title: {original_title}
Content: {original_content}

## New Episode to Merge
Time Range: {new_time_range}
Title: {new_title}
Content: {new_content}

## Combined Event Details
{combined_events}

## Your Task
Generate a merged episode that:
1. Combines information from both episodes without duplication
2. Maintains chronological flow of events
3. Preserves all important details from both episodes
4. Creates a coherent narrative

## Output Format
Return JSON:
{{
    "title": "Merged episode title that captures the complete topic",
    "content": "Detailed narrative combining both episodes chronologically. Include all participants, key decisions, emotions, and outcomes.",
    "timestamp": "ISO format timestamp of when the merged episode occurred (use earliest time)"
}}

Return only the JSON object, no additional text.
```

---

## Answer prompt（QA 回答）

**重点确认：两个 answer prompt 均显式要求 short answer —— "The answer should be less than 5-6 words."**

### 8. LoCoMo answer prompt

来源：`evaluation/locomo/search.py`，模块级变量 `ANSWER_PROMPT`（Jinja2 Template，以 user message 发送，无 system message）

```text
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

    Episodic Memories:
    {{ episodic }}

    Semantic Memories:
    {{ semantic }}

    Question: {{ question }}

    Answer:
```

### 9. LongMemEval answer prompt

来源：`evaluation/longmemeval/search.py`，模块级变量 `ANSWER_PROMPT`（Jinja2 Template，与 LoCoMo 版本几乎相同，仅在末尾多一行 `Question Date`）

```text
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

    Episodic Memories:
    {{ episodic }}

    Semantic Memories:
    {{ semantic }}

    Question: {{ question }}
    Question Date: {{ question_date }}

    Answer:
```

---

## Judge prompt（LLM 评分）

### 10. LoCoMo judge prompt

来源：`evaluation/locomo/metrics/llm_judge.py`，模块级变量 `ACCURACY_PROMPT`（默认 judge 模型 `openai/gpt-4.1-mini`，temperature=0，JSON 输出；注意原文中的弯引号 `’` 为逐字保留）

```text
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
```

配套 system message（同文件，`evaluate_llm_judge` 函数内）：

```text
You are an expert grader that determines if answers to questions match a gold standard answer.
```

### 11. LongMemEval judge prompt — temporal-reasoning

来源：`evaluation/longmemeval/evals.py`，`LongMemEvalEvaluator.__init__` 中的 `self.TEMPORAL_REASONING_PROMPT`

```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>
```

### 12. LongMemEval judge prompt — knowledge-update

来源：`evaluation/longmemeval/evals.py`，`LongMemEvalEvaluator.__init__` 中的 `self.KNOWLEDGE_UPDATE_PROMPT`

```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>
```

### 13. LongMemEval judge prompt — single-session-preference

来源：`evaluation/longmemeval/evals.py`，`LongMemEvalEvaluator.__init__` 中的 `self.SINGLE_SESSION_PREFERENCE_PROMPT`

```text
I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

<QUESTION>
B: {question}
</QUESTION>
<RUBRIC>
{gold_answer}
</RUBRIC>
<RESPONSE>
A: {response}
</RESPONSE>
```

### 14. LongMemEval judge prompt — 默认（其他题型）

来源：`evaluation/longmemeval/evals.py`，`LongMemEvalEvaluator.__init__` 中的 `self.DEFAULT_PROMPT`

```text
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
```

配套 system prompt（同文件，`evaluate_single_response` 函数内的 `system_prompt`，判定通过结构化输出 `Grade.is_correct == 'yes'`）：

```text
You are an expert grader that determines if answers to questions match a gold standard answer
```

---

## 其他

### 15. 多模态补充指引（追加到 episode 生成 prompt 末尾）

来源：`nemori/llm/generators/episode.py`，模块级变量 `_MULTIMODAL_GUIDANCE`（当对话包含图片时，通过 `_build_multimodal_prompt` 拼接在 episode 生成 prompt 之后）

```text
If images are included in this conversation:
1. Use the images to enrich your understanding of what the user was doing or discussing.
2. Describe the visual context naturally within the narrative.
3. Do NOT reference technical details like "image_url" or "screenshot #3".
4. Integrate visual information chronologically with the text conversation.
```

---

## 备注

- 检索（search）模块不使用 LLM prompt，是向量 / 关键词检索，未发现 query rewrite 或 rerank prompt。
- LoCoMo 评测跳过 category 5（`evaluation/locomo/evals.py`、`llm_judge.py` 中 `if category == "5": continue`），指标含 BLEU-1、F1 和 LLM judge 二值分。
- `nemori/llm/generators/segmenter.py`、`semantic.py`、`merger.py` 均直接复用 `PromptTemplates` 中的模板（user role，无额外 system message），除 episode.py 的 system message 与多模态指引外没有其他内联 prompt。
- 所有类别（构建/记忆提取、answer、judge、其他）均已找到，无缺失。
