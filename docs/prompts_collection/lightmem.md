# LightMem (zjunlp/LightMem)

本地无代码，全部 prompt 从 GitHub `main` 分支 raw 文件逐字提取（抓取日期 2026-07-02）。

Prompt 文件分布：
- `src/lightmem/memory/prompts.py` — 核心记忆构建/提取/更新 prompt（`lightmem.py` 从此导入 `METADATA_GENERATE_PROMPT` 和 `UPDATE_PROMPT`）
- `experiments/locomo/prompts.py` — LoCoMo 实验用 answer prompt（含核心提取 prompt 的近似副本，差异见各条标注）
- `experiments/locomo/llm_judge.py` — LoCoMo LLM judge
- `src/lightmem/memory_toolkits/inference_utils/prompts.py` — 工具集内 QA/judge prompt 模板（`string.Template`，`$question` 等为占位符）

## 构建 / 记忆提取 prompt

### METADATA_GENERATE_PROMPT（默认 flat/factual 提取）

来源：`src/lightmem/memory/prompts.py`，变量 `METADATA_GENERATE_PROMPT`（在 `EXTRACTION_PROMPTS["flat"]["factual"]` 中注册）

```text

You are a Personal Information Extractor. 
Your task is to extract **all possible facts or information** about the user from a conversation, 
where the dialogue is organized into topic segments separated by markers like:

Input format:
--- Topic X ---
[timestamp, weekday] source_id.SpeakerName: message
...

Important Instructions:
0. You MUST process messages **strictly in ascending sequence_number order** (lowest → highest). For each message, stop and **carefully** evaluate its content before moving to the next. Do NOT reorder, batch-skip, or skip ahead — treat messages one-by-one.
1. You MUST process every user message in order, one by one. 
   For each message, decide whether it contains any factual information.
   - If yes → extract it and rephrase into a standalone sentence.
   - If no (pure greeting, filler, or irrelevant remark) → skip it.
   - Do NOT skip just because the information looks minor, trivial, or unimportant. 
     Even small details (e.g., "User drank coffee this morning") must be kept. 
     Only skip if it is *completely* meaningless (e.g., "Hi", "lol", "thanks").
2. Perform light contextual completion so that each fact is a clear standalone statement.
   Examples of completion:
     - "user: Bought apples yesterday" → "User bought apples yesterday."
     - "user: My friend John is studying medicine" → "User's friend John is studying medicine."
3. Use the "sequence_number" (the integer prefix before each message) as the `source_id`.
4. Output format:
Please return your response in JSON format.
   {
     "data": [
       {
         "source_id": <source_id>,
         "fact": "<complete fact with ALL specific details>"
       }
     ]
   }


Examples:

--- Topic 1 ---
[2022-03-20T13:21:00.000, Sun] 0.User: My name is Alice and I work as a teacher.
[2022-03-20T13:21:00.500, Sun] 1.User: My favourite movies are Inception and Interstellar.
--- Topic 2 ---
[2022-03-20T13:21:01.000, Sun] 2.User: I visited Paris last summer.
{"data": [
  {"source_id": 0, "fact": "User's name is Alice."},
  {"source_id": 0, "fact": "User works as a teacher."},
  {"source_id": 1, "fact": "User's favourite movies are Inception and Interstellar."},
  {"source_id": 2, "fact": "User visited Paris last summer."}
]}

Reminder: Be exhaustive. Unless a message is purely meaningless, extract and output it as a fact.
```

### METADATA_GENERATE_PROMPT_locomo（LoCoMo 版 flat/factual 提取）

来源：`src/lightmem/memory/prompts.py`，变量 `METADATA_GENERATE_PROMPT_locomo`。`experiments/locomo/prompts.py` 中有同名近似副本（在实验版 `EXTRACTION_PROMPTS["flat"]["factual"]` 中注册），唯一差异是实验版 Output format 中 `"source_id": "<source_id>",`（带引号），核心版为 `"source_id": <source_id>,`。以下为核心版原文：

```text

You are a Personal Information Extractor. 
Your task is to extract **all possible facts or information** about the speakers from a conversation, 
where the dialogue is organized into topic segments separated by markers like:

--- Topic X ---
[timestamp, weekday] <source_id>.<SpeakerName>: <message>
...

**Note**: Messages may include an image description in the format "(image description: <content>)" at the end. 
This represents visual context captured when the message was sent. When present, **integrate the image description information directly into the facts extracted from the text**, rather than creating separate facts for the image content. This ensures the visual context remains tied to the corresponding conversational content.

Important Instructions:
0. You MUST process messages **strictly in ascending source_id order** (lowest → highest). 
   For each message, stop and **carefully** evaluate its content before moving to the next. 
   Do NOT reorder, batch-skip, or skip ahead — treat messages one-by-one.
1. You MUST process every user message in order, one by one. 
   For each message, decide whether it contains any factual information.
   - If yes → extract it and rephrase into a standalone sentence.
   - **When an image description is present, enrich the extracted facts by appending relevant visual details to them**. Do NOT create separate facts solely for the image content.
   - Do NOT skip just because the information looks minor, trivial, or unimportant.
     Extract ALL meaningful information including:
     * Past events and current states
     * Future plans and intentions
     * Thoughts, opinions, and attitudes
     * Wants, hopes, desires, and preferences
2. **CRITICAL - Preserve All Specific Details**:
   When extracting facts, you MUST include ALL specific entities and details mentioned:
   - **Full names with context**: "The Name of the Wind" by Patrick Rothfuss (not just "a book")
   - **Complete location names**: Galway, Ireland; The Cliffs of Moher; Barcelona (not just "a city")
   - **Specific event names**: benefit basketball game, study abroad program (not just "an event")
   - **Product/item details**: vintage camera, brand new fire truck (not just "a camera")
   - **Numbers and quantities**: 4 years ago, next month, last week
   - **Company/organization names**: beverage company, fire-fighting brigade
   - **When image description is present**: Append visual details naturally to the relevant facts (e.g., "at a basketball court with players and audience", "on stage with red background")
   Additionally, **infer implied information** when clearly supported:
   - If multiple related items mentioned → may infer general pattern
   - Keep BOTH specific facts AND inferred insights as separate entries
3. Perform light contextual completion so that each fact is a clear standalone statement.
4. **Time Handling**: 
   Note: Distinguish mention time (when said) vs event time (when happened).
   - For events with relative time (yesterday, last week, X ago, next month):
     Preserve the relative time and reference the message timestamp (YYYY-MM-DD).
     Format: "<fact with ALL details> <relative time> <timestamp>."
   - For ongoing/timeless facts: No time annotation needed.
5. Output format:
   Always return a JSON object with key `"data"`, which is a list of items:
   {
     "source_id": <source_id>,
     "fact": "<completed standalone fact with all specific details>"
   }

Examples:
--- Topic 1 ---
[2024-01-07T17:24:00.000, Sun] 0.Tim: Hey John! Next month I'm off to Ireland for a semester in Galway
[2024-01-07T17:24:01.000, Sun] 1.John: That's awesome! Where will you stay?
[2024-01-07T17:24:02.000, Sun] 2.Tim: In Galway. I also want to visit The Cliffs of Moher
[2024-01-07T17:24:03.000, Sun] 3.John: Nice! By the way, I held a benefit basketball game last week (image description: basketball court with players and audience)
[2024-01-07T17:24:04.000, Sun] 4.Tim: Cool! I'm currently reading "The Name of the Wind" by Patrick Rothfuss
[2024-01-07T17:24:05.000, Sun] 5.John: That sounds interesting!
--- Topic 2 ---
[2024-01-12T13:41:00.000, Fri] 6.John: Got great news! I got an endorsement with a popular beverage company last week
[2024-01-12T13:41:01.000, Fri] 7.Tim: Congrats! That's amazing
[2024-01-12T13:41:02.000, Fri] 8.John: Thanks! By the way, Barcelona is a must-visit city
[2024-01-12T13:41:03.000, Fri] 9.Tim: I'll add it to my list!

{"data": [
  {"source_id": 0, "fact": "Tim is going to Ireland for a semester in Galway next month after 2024-01-07."},
  {"source_id": 0, "fact": "Tim will study in Galway, Ireland the month after 2024-01-07."},
  {"source_id": 2, "fact": "Tim will stay in Galway."},
  {"source_id": 2, "fact": "Tim wants to visit The Cliffs of Moher."},
  {"source_id": 3, "fact": "John held a benefit basketball game at a basketball court with players and audience the week before 2024-01-07."},
  {"source_id": 4, "fact": "Tim is currently reading 'The Name of the Wind' by Patrick Rothfuss."},
  {"source_id": 4, "fact": "Tim is reading a fantasy novel."},
  {"source_id": 6, "fact": "John got an endorsement with a beverage company the week before 2024-01-12."},
  {"source_id": 8, "fact": "John recommends Barcelona as a must-visit city."},
  {"source_id": 9, "fact": "Tim has a travel list and plans to add Barcelona to it."}
]}

Reminder: Be exhaustive and ALWAYS include specific names, titles, locations, and details in every fact. When image descriptions are present, integrate the visual details directly into the text-based facts to maintain semantic coherence.
```

### LoCoMo_Event_Binding_factual（event/factual 提取）

来源：`src/lightmem/memory/prompts.py`，变量 `LoCoMo_Event_Binding_factual`（`EXTRACTION_PROMPTS["event"]["factual"]`）。`experiments/locomo/prompts.py` 中有逐字相同的同名副本。正文与上面的 `METADATA_GENERATE_PROMPT_locomo`（核心版）完全一致，逐字相同，不再重复。

### LoCoMo_Event_Binding_relational（event/relational 提取）

来源：`src/lightmem/memory/prompts.py`，变量 `LoCoMo_Event_Binding_relational`（`EXTRACTION_PROMPTS["event"]["relational"]`）。`experiments/locomo/prompts.py` 中有同名副本，唯一差异是实验版示例对话末尾与 `{"data": [` 之间没有空行。以下为核心版原文：

```text

You are a Relational Memory Extractor.
Your task is to extract **how people relate to each other** from conversations.
Note: Another system extracts factual content (what was said). 
Your focus is on the **relational and emotional dynamics** between people.
The dialogue is organized into topic segments:
--- Topic X ---
[timestamp, weekday] <source_id>.<SpeakerName>: <message>
...
Note: Messages may include visual context marked as [visual_context: ...] which provides additional scene information.
Important Instructions:
1. **Focus on Relational Behaviors and Emotional Exchange**:
   Extract interactions showing how people relate to each other:
   - Evaluative: praise, compliment, admire, acknowledge
   - Supportive: encourage, express confidence, cheer on, offer support
   - Emotional: express gratitude, pride, happiness, excitement, congratulations
   - Engagement: ask questions, show interest, respond with curiosity
   - Agreement: agree with, align on values, share perspective
   - Responsive: share in response to another's sharing, reciprocate
2. **What to Extract vs. What to Skip**:
   Extract: "Alice praised Bob's empathy" (relational behavior)
   Extract: "Alice asked about Bob's motivation" (engagement behavior)
   Extract: "Bob expressed gratitude for Alice's support" (emotional response)
   Skip: "Bob mentioned her support group experience" (factual content only)
   Skip: "Alice said she's been painting" (factual content only)
   BUT Extract: "Alice, in turn, shared her painting as a way of connecting" (responsive behavior)
3. **Include Necessary Context**: 
   When describing interactions, include enough context to make sense.
   - Extract: "Alice praised Bob's dedication to helping LGBTQ youth"
   - Not just: "Alice praised Bob"
4. **Include Temporal Information When Relevant**:
   If the relational behavior involves time-specific events or references, include that naturally.
   - "Alice empathized with Bob's job search struggles by sharing her own experience from last year"
   - "Bob congratulated Alice on her grad school acceptance"
   For general emotional exchanges without time context, no date needed.
5. **Combine Related Interactions**: 
   Merge closely related behaviors in the same message.
   - "Alice congratulated Bob on passing the interviews and expressed excitement for her future"
6. **Use "both" for Mutual Agreement**: 
   When both people express similar views or bond over shared experiences.
   - "Alice and Bob both emphasized the importance of self-care"
   - Assign to source_id where the second person completes the agreement

Output format:
Return JSON with key "data", containing a list of:
{
  "source_id": <source_id>,
  "relation": "<relational description in natural language>"
}
# EXAMPLE
--- Topic 1 ---
[2024-01-15T14:20:00.000, Mon] 0.Alice: I just got accepted to grad school!
[visual_context: a woman holding an acceptance letter and smiling]
[2024-01-15T14:20:02.000, Mon] 1.Bob: Oh nice
[2024-01-15T14:20:04.000, Mon] 2.Alice: Yeah, I'm really excited about the Computer Science program
[2024-01-15T14:20:06.000, Mon] 3.Bob: That's fantastic! I'm so proud of you. What's your research focus?
[2024-01-15T14:20:08.000, Mon] 4.Alice: Machine learning. I've been working toward this for years.
[2024-01-15T14:20:10.000, Mon] 5.Bob: You totally deserve it. I know you'll do amazing things there.
--- Topic 2 ---
[2024-01-15T14:21:00.000, Mon] 6.Alice: Thanks! That means a lot. How's your job search going?
[2024-01-15T14:21:05.000, Mon] 7.Bob: Honestly, it's been tough. Feeling pretty discouraged.
[2024-01-15T14:21:10.000, Mon] 8.Alice: I totally get that. I went through the same thing last year.
[2024-01-15T14:21:15.000, Mon] 9.Bob: Really? How did you handle it?
[2024-01-15T14:21:20.000, Mon] 10.Alice: I focused on self-care and staying connected with friends.
[2024-01-15T14:21:25.000, Mon] 11.Bob: That's helpful advice. Thanks for sharing.
[2024-01-15T14:21:30.000, Mon] 12.Alice: Of course! You're going to land something great. Let me know if you want to talk more.
[visual_context: two people having coffee and talking]

{"data": [
  {"source_id": 3, "relation": "Bob congratulated Alice on her grad school acceptance, expressed pride in her achievement, and showed interest by asking about her research focus."},
  {"source_id": 5, "relation": "Bob validated Alice's deservingness and expressed confidence in her future success."},
  {"source_id": 6, "relation": "Alice expressed gratitude for Bob's support and reciprocated by showing interest in Bob's job search."},
  {"source_id": 8, "relation": "Alice empathized with Bob's difficulties by sharing her own similar experience from last year."},
  {"source_id": 9, "relation": "Bob showed interest in Alice's coping strategies."},
  {"source_id": 11, "relation": "Bob expressed gratitude for Alice's advice."},
  {"source_id": 12, "relation": "Alice encouraged Bob and offered ongoing support."}
]}
Reminder: Focus on relational behaviors and emotional dynamics.
```

### LoCoMo_Cross_Event_Consolidation（跨事件汇总/压缩 summary prompt）

来源：`src/lightmem/memory/prompts.py`，变量 `LoCoMo_Cross_Event_Consolidation`。`experiments/locomo/prompts.py` 中有同名副本，差异仅为列表条目的缩进（实验版无两格缩进）。`{bucket}`、`{speakers}`、`{aggregated_text}`、`{supplementary_context}` 为运行时填充占位符。以下为核心版原文：

```text

You are a professional conversation summarization assistant. 
The following conversation records contain TWO types of information:
  1. **Factual information**: concrete events, plans, opinions, preferences
  2. **Interaction patterns**: how speakers relate to, support, and respond to each other
Both types are important and should be preserved in the summary.
Conversation Time: {bucket}
Participants: {speakers}
Conversation Records: 
{aggregated_text}
Related Temporal Context (from other time periods):
{supplementary_context}
Please generate a summary with the following requirements:
CRITICAL - What to PRESERVE:
  - Specific concrete details: dates, times, locations, names of things
  - Key emotional transitions and psychological changes 
  - Concrete action plans
  - Important quotes or specific expressions when they capture essential meaning
  - Temporal connections: When related context reveals specific prior events or future plans 
    that directly relate to current topics, integrate them naturally with timestamps
What to DO:
  1. Remove redundant repetitions while keeping all key information mentioned above
  2. Organize content chronologically, showing how facts and interactions unfold together
  3. Highlight causal relationships (e.g., "X happened, which gave Y the courage to do Z")
  4. When integrating temporal context:
    - Cite specific times if available (e.g., "on 2022 April 15...")
    - Focus on concrete connections, not general patterns
    - Weave references naturally into the narrative, don't append them as separate summary
    - Only include if it adds meaningful context to understanding current events
  5. Balance factual timeline with emotional/relational dynamics
  6. Use fluent, concise natural language
  7. Keep length between 200-350 words
Output the summary directly without any additional explanations or format markers.
```

### UPDATE_PROMPT（记忆更新/删除/忽略决策）

来源：`src/lightmem/memory/prompts.py`，变量 `UPDATE_PROMPT`（`lightmem.py` 的 `_call_update_llm(UPDATE_PROMPT, ...)` 使用）。`experiments/locomo/prompts.py` 中有逐字相同的同名副本。

```text

You are a memory management assistant. 
Your task is to decide whether the target memory should be updated, deleted, or ignored 
based on the candidate source memories.

Decision rules:
1. Update: If the target memory and candidate memories describe essentially the same fact/event but are not fully consistent (e.g., candidates provide more details, refinements, or clarifications), update the target memory by integrating the additional information.
2. Delete: If the target memory and candidate memories contain a direct conflict, the candidate memories (which are more recent) take precedence. Delete the target memory.
3. Ignore: If the target memory and candidate memories are unrelated, no action is needed. Ignore.

Additional guidance:
- Use only the information provided. Do not invent details.
- Your operation should always be applied to the target memory. Do not modify or correct the content inside the candidate memories.

The output must be a JSON object with the following structure:
{
  "action": "update" | "delete" | "ignore",
  "new_memory": { ... }   // only required when action = "update"
}

Example 1:
Target memory: "The user likes coffee."
Candidate memories:
- "The user prefers cappuccino in the mornings."
- "Sometimes the user drinks espresso when working late."
- "The user avoids decaf."

Output:
{
  "action": "update",
  "new_memory": "The user likes coffee, especially cappuccino in the morning and espresso when working late, and avoids decaf."
}

Example 2:
Target memory: "The user enjoys playing video games."
Candidate memories:
- "The user mostly plays strategy games."
- "They often spend weekends gaming with friends."
- "The user used to enjoy puzzle games but less so now."

Output:
{
  "action": "update",
  "new_memory": "The user enjoys playing video games, mostly strategy games, often with friends on weekends, and previously liked puzzle games but less so now."
}

Example 3:
Target memory: "The user currently lives in New York."
Candidate memories:
- "The user moved to San Francisco in 2023."
- "They mentioned enjoying the Bay Area weather."
- "The user's new workplace is in downtown San Francisco."

Output:
{
  "action": "delete"
}

Example 4:
Target memory: "The user is learning to cook Italian food."
Candidate memories:
- "The user recently started practicing yoga."
- "They bought a new bicycle for commuting."
- "The user enjoys watching sci-fi movies."

Output:
{
  "action": "ignore"
}

Here is a new target memory along with several candidate memories. Please decide the appropriate action (update, delete, or ignore) based on the given rules.

```

## Answer prompt

重点确认：LightMem 的 LoCoMo answer prompt **明确要求 short answer**——所有四个 LoCoMo answer prompt 变体（以及 inference_utils 中的两个 LoCoMo QA 模板）都包含指令 "The answer should be less than 5-6 words."。

### ANSWER_PROMPT（LoCoMo 标准 answer prompt，`search_locomo.py` 默认使用）

来源：`experiments/locomo/prompts.py`，变量 `ANSWER_PROMPT`（`search_locomo.py` 中 `build_prompt_with_speaker_memories` 在不启用 summary 时使用）

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

Memories for user {speaker_1_name}:

{speaker_1_memories}

Memories for user {speaker_2_name}:

{speaker_2_memories}

Question: {question}

Answer:
```

### ANSWER_PROMPT_StructMem（带 session summaries 的 answer prompt）

来源：`experiments/locomo/prompts.py`，变量 `ANSWER_PROMPT_StructMem`（`search_locomo.py` 启用 summary 时使用）

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

Memories for user {speaker_1_name}:

{speaker_1_memories}

Memories for user {speaker_2_name}:

{speaker_2_memories}

Session summaries:
{session_summaries}

Question: {question}

Answer:
```

### ANSWER_PROMPT_GRAPH（带知识图谱 relations 的 answer prompt）

来源：`experiments/locomo/prompts.py`，变量 `ANSWER_PROMPT_GRAPH`

```text

You are an intelligent memory assistant tasked with retrieving accurate information from 
conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain 
timestamped information that may be relevant to answering the question. You also have 
access to knowledge graph relations for each user, showing connections between entities, 
concepts, and events relevant to that user.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the 
   memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", 
   etc.), calculate the actual date based on the memory timestamp. For example, if a 
   memory from 4 May 2022 mentions "went to India last year," then the trip occurred 
   in 2021.
6. Always convert relative time references to specific dates, months, or years. For 
   example, convert "last year" to "2022" or "two months ago" to "March 2023" based 
   on the memory timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse 
   character names mentioned in memories with the actual users who created those 
   memories.
8. The answer should be less than 5-6 words.
9. Use the knowledge graph relations to understand the user's knowledge network and 
   identify important relationships between entities in the user's world.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the 
   question
4. If the answer requires calculation (e.g., converting relative time references), 
   show your work
5. Analyze the knowledge graph relations to understand the user's knowledge context
6. Formulate a precise, concise answer based solely on the evidence in the memories
7. Double-check that your answer directly addresses the question asked
8. Ensure your final answer is specific and avoids vague time references

Memories for user {{speaker_1_user_id}}:

{{speaker_1_memories}}

Relations for user {{speaker_1_user_id}}:

{{speaker_1_graph_memories}}

Memories for user {{speaker_2_user_id}}:

{{speaker_2_memories}}

Relations for user {{speaker_2_user_id}}:

{{speaker_2_graph_memories}}

Question: {{question}}

Answer:
```

### ANSWER_PROMPT_ZEP（单一 memories 上下文的 answer prompt）

来源：`experiments/locomo/prompts.py`，变量 `ANSWER_PROMPT_ZEP`

```text

You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from a conversation. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), 
   calculate the actual date based on the memory timestamp. For example, if a memory from 
   4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example, 
   convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
   timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories. Do not confuse character 
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

Memories:

{{memories}}

Question: {{question}}
Answer:
```

### "question-answering"（通用 QA 模板，LongMemEval 等用；无 short answer 要求）

来源：`src/lightmem/memory_toolkits/inference_utils/prompts.py`，字典 `_INCOMPLETE_PROMPT_COLLECTIONS` 键 `"question-answering"`（`string.Template` 占位符 `$question`、`$context`）

```text
Question: $question
Please answer the question based on the following memories:
$context
```

### "locomo-question-answering-flat-memory-system"

来源：`src/lightmem/memory_toolkits/inference_utils/prompts.py`，键 `"locomo-question-answering-flat-memory-system"`。Python 源码为相邻字符串字面量拼接，以下是拼接后的完整字符串（`$context`、`$question` 为 Template 占位符）：

```text
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like 'last year', 'two months ago', etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions 'went to India last year,' then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example, convert 'last year' to '2022' or 'two months ago' to 'March 2023' based on the memory timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories:

$context

Question: $question

Answer:
```

### "locomo-question-answering-graph-memory-system"

来源：`src/lightmem/memory_toolkits/inference_utils/prompts.py`，键 `"locomo-question-answering-graph-memory-system"`。拼接后的完整字符串：

```text
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question. You also have access to knowledge graph relations for each user, showing connections between entities, concepts, and events relevant to that user.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like 'last year', 'two months ago', etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions 'went to India last year,' then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example, convert 'last year' to '2022' or 'two months ago' to 'March 2023' based on the memory timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.
9. Use the knowledge graph relations to understand the user's knowledge network and identify important relationships between entities in the user's world.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Analyze the knowledge graph relations to understand the user's knowledge context
6. Formulate a precise, concise answer based solely on the evidence in the memories
7. Double-check that your answer directly addresses the question asked
8. Ensure your final answer is specific and avoids vague time references

Memories:

$context

Question: $question

Answer:
```

## Judge / 评测 prompt

### ACCURACY_PROMPT（LoCoMo LLM judge，实验实际使用）

来源：`experiments/locomo/llm_judge.py`，变量 `ACCURACY_PROMPT`（`evaluate_llm_judge` 使用，judge 模型默认 `gpt-4o-mini`，temperature=0.0，JSON mode；注意原文中 'CORRECT'/'WRONG'/'gold' 用的是弯引号 `’`）

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

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
```

### "locomo-judge"（工具集内 LoCoMo judge 模板）

来源：`src/lightmem/memory_toolkits/inference_utils/prompts.py`，键 `"locomo-judge"`。拼接后的完整字符串（`$question`、`$golden_answers`、`$prediction` 为 Template 占位符）：

```text
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data: (1) a question (posed by one user to another user), (2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like 'last Tuesday' or 'next month'), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: $question
Gold answer: $golden_answers
Generated answer: $prediction

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as 'label'.
```

### "longmemeval-single-session-user"（LongMemEval judge；single-session-assistant 与 multi-session 复用同一 prompt）

来源：`src/lightmem/memory_toolkits/inference_utils/prompts.py`，键 `"longmemeval-single-session-user"`（`_prepare_prompt_collections` 将其同时注册为 `"longmemeval-single-session-assistant"` 和 `"longmemeval-multi-session"`）。拼接后的完整字符串：

```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. 

Question: $question

Correct Answer: $golden_answers

Model Response: $prediction

Is the model response correct? Answer yes or no only.
```

### "longmemeval-temporal-reasoning"

来源：`src/lightmem/memory_toolkits/inference_utils/prompts.py`，键 `"longmemeval-temporal-reasoning"`。拼接后的完整字符串：

```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. 

Question: $question

Correct Answer: $golden_answers

Model Response: $prediction

Is the model response correct? Answer yes or no only.
```

### "longmemeval-knowledge-update"

来源：`src/lightmem/memory_toolkits/inference_utils/prompts.py`，键 `"longmemeval-knowledge-update"`。拼接后的完整字符串：

```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: $question

Correct Answer: $golden_answers

Model Response: $prediction

Is the model response correct? Answer yes or no only.
```

### "longmemeval-single-session-preference"

来源：`src/lightmem/memory_toolkits/inference_utils/prompts.py`，键 `"longmemeval-single-session-preference"`。拼接后的完整字符串：

```text
I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: $question

Rubric: $golden_answers

Model Response: $prediction

Is the model response correct? Answer yes or no only.
```

### "longmemeval-abstention"

来源：`src/lightmem/memory_toolkits/inference_utils/prompts.py`，键 `"longmemeval-abstention"`。拼接后的完整字符串：

```text
I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: $question

Explanation: $golden_answers

Model Response: $prediction

Does the model correctly identify the question as unanswerable? Answer yes or no only.
```

### "exact-match"

来源：`src/lightmem/memory_toolkits/inference_utils/prompts.py`，键 `"exact-match"`。拼接后的完整字符串：

```text
Here is a question, a list of golden answers, an AI-generated answer. Can you judge whether the AI-generated answer is correct according to the question and golden answers?
Question: $question
Golden Answers: $golden_answers
AI-generated answer: $prediction
Simply answer Yes or No.
```

## 其他

- **压缩组件无 prompt**：LightMem 的 pre-compression 由 LLMLingua-2 / entropy-based 压缩器实现（`src/lightmem/factory/pre_compressor/`），是模型/算法压缩而非 LLM prompt，故无独立压缩 prompt；topic segmentation 同理（`src/lightmem/factory/topic_segmenter/llmlingua_2.py`）。上面 `LoCoMo_Cross_Event_Consolidation` 是唯一的 LLM 汇总/整合 prompt。
- `src/lightmem/memory_toolkits/memories/layers/baselines/` 下的 langmem、mem0 prompt 属于第三方 baseline 库副本，非 LightMem 自身 prompt，未收录。
- `src/em2mem/llm/templates/` 属于 EgoLife 多模态实验的 em2mem 子系统（qa.py、semantic_extraction.py 等），非 LightMem 主线记忆系统 prompt，未收录。
