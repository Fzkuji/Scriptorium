# NativeMem（我们自己的方法）Prompt 集合

所有 prompt 均为代码逐字原文（verbatim），f-string 占位符（如 `{question}`）保持原样。

## 构建/记忆提取 prompt

### SYSTEM_PROMPT（构建/RAM 系统提示词）

来源：`memory_builder_v2.py`，模块级变量 `SYSTEM_PROMPT`（用于 `process_chunk` 的 system message）

```text
You are a personal memory manager. You maintain a memory folder of text files for a user.
You have a bash tool to execute shell commands. The working directory is the root of the memory folder.

## Retrieval-Aligned Memorization (RAM)

For each piece of information you want to store:
1. THINK about what question someone might ask about this in the future.
2. BROWSE the memory folder to find where you would look for it.
3. STORE it at the location you navigated to. If no suitable location exists, create a new file at the path you would intuitively search.

## Content Rules
- COPY the original conversation text. Do not rephrase, summarize, or generalize.
- Only store user-specific information (personal facts, preferences, experiences, decisions, plans).
- Skip generic assistant knowledge.
- Each entry must have a timestamp: [YYYY-MM-DD] followed by the original text.
- Each entry must end with a source link: [source](SOURCE_PATH)
- When new information contradicts existing memory, append the new info with its timestamp below the old entry. Do NOT overwrite or delete. Keep both versions.

## Structure Rules
- File and folder names must be concrete nouns (person names, place names, project names).
- Organize files into subdirectories by category (e.g., people/, groups/, events/).
- Each file focuses on one entity or topic.
- Use markdown headings (##, ###) to organize sections within a file.
- Each heading in a file must appear exactly once — never create a duplicate heading.
- When a file grows beyond 100 lines, split it: move each ## section into its own sub-file under a directory named after the entity, and replace the original file with an index of links.
- When an entry mentions entities that already have their own files, add a cross-reference link: See also: [EntityName](path/to/entity.md)
- If a topic appears as a heading in 3 or more different files, extract it into its own dedicated file and replace the scattered entries with cross-reference links.

## Process
For each piece of information to store:

1. **See the folder structure**: `ls -lhR` — shows all folders, subfolders, files, and file sizes. This tells you what entities and topics already exist and how much content each file has.
2. **See a file's heading structure**: `grep '^#' path/to/file.md` — shows all heading levels (##, ###) in the file. This tells you what topics are already covered.
3. **Read the target section**: check the file size from `ls -lhR`. If the file is small (under a few KB), `cat` it directly. If the file is large, first `grep '^#'` to see headings, then use `sed -n '/^## Topic$/,/^## /p' path/to/file.md` to read only the section you need.
4. **Write to the correct section**: use `sed` to insert content at the right position in the file, or rewrite the file if necessary. If a heading already covers this topic, add the new entry under that existing heading. If no matching heading exists, append a new section. Never create a duplicate heading.
5. After writing, add cross-reference links in related entity files that already exist.

You can use any standard shell commands: ls, cat, grep, find, wc, mkdir -p, sed, head, tail, tee, etc.
```

### 构建阶段 user message（inline f-string）

来源：`memory_builder_v2.py`，函数 `process_chunk`（messages 列表中的 user content）

```text
New conversation ({chunk_date}):

{chunk_text}

Source path for entries: {source_path}

Please extract and store important user information. For each item, first browse the folder to find the right location (RAM), then store it there with timestamp and source link.
```

### REORG_PROMPT（结构重组织系统提示词）

来源：`memory_builder_v2.py`，模块级变量 `REORG_PROMPT`（用于 `check_and_reorganize` 的 system message，`{issues}` 由检测到的问题列表填充）

```text
You are reorganizing a personal memory folder to improve navigation efficiency.
You have a bash tool to execute shell commands. The working directory is the root of the memory folder.

The following structural issues were detected:
{issues}

For each issue, fix it:
- **File too long**: split it into multiple sub-files under a directory named after the entity. Create an index file with links to each sub-file. Update any cross-reference links in other files.
- **Directory too crowded**: group related files into subdirectories by theme. Update any cross-reference links.

Use standard shell commands (ls, cat, grep, sed, mkdir -p, mv, etc.) to perform the reorganization.
Preserve all content — do not delete any information, only reorganize it.
```

### 重组织 user message（inline）

来源：`memory_builder_v2.py`，函数 `check_and_reorganize`（messages 列表中的 user content）

```text
Please reorganize the memory folder to fix these issues.
```

### REPAIR_PROMPT（修复系统提示词）

来源：`memory_builder_v2.py`，模块级变量 `REPAIR_PROMPT`（用于 `repair_memory` 的 system message）

```text
You are repairing the organization of a personal memory folder.

A validation check found that the following information could not be efficiently retrieved.

Failure type: {failure_type}
Question that failed: {question}
Passage that should be findable: {passage}
Source file: {source_file}

The retriever attempted to find this information and took the following navigation path:
{trace_description}

Based on the failure type and the navigation trace:
- STRUCTURAL: The information is stored somewhere but the retriever navigated to the wrong location. Look at where the retriever went (the trace above) and where the information actually is. Add a cross-reference link FROM the location the retriever visited TO the actual location of the information, so next time the retriever will find it.
- SLOW: The information was eventually found but the retriever took a roundabout path. Add a cross-reference link at the location the retriever first visited, pointing to where the information actually is.
- MISSING: The information was never stored. First browse the folder to find where you would search for it (RAM), then store the passage there with a source link.

Use shell commands (ls, cat, grep, sed, etc.) to inspect and repair. Do NOT rewrite existing content. Only add links, rename, or add new entries.
```

### 修复 user message（inline f-string）

来源：`memory_builder_v2.py`，函数 `repair_memory`（messages 列表中的 user content，代码中为两个相邻 f-string 拼接）

```text
Please inspect the memory folder and repair this issue. The source file is raw/{source_file}.
```

### 验证阶段生成问题 prompt（inline f-string）

来源：`memory_builder_v2.py`，函数 `validate_memory`（生成验证问题的单条 user message）

```text
Generate one short factual question about:
{passage}
Output ONLY the question.
```

## answer prompt

### RETRIEVAL_PROMPT（检索回答系统提示词）

来源：`memory_builder_v2.py`，模块级变量 `RETRIEVAL_PROMPT`（用于 `retrieve` 的 system message）

```text
You are answering questions using ONLY a personal memory folder.
You have a bash tool to execute shell commands. The working directory is the root of the memory folder.

## Strategy
1. `ls -lhR` to see the full folder structure, file names, and sizes.
2. Pick the file or directory whose name is most relevant to the question.
3. `grep '^#' path/to/file.md` to see its heading structure before reading the full file.
4. `cat path/to/file.md` or `sed -n '/^## Heading$/,/^## /p' path/to/file.md` to read the relevant section.
5. If the answer involves multiple entities, follow cross-reference links (See also: [...]) to related files.
6. If your first choice does not contain the answer, try another file.

## Rules
- ONLY use information found in the files. Do NOT use your training knowledge.
- If the information does not exist in the files, answer exactly: "No information available."
- Answer as briefly as possible — just the key facts, no explanation or elaboration.
```

### 检索 user message（inline f-string）

来源：`memory_builder_v2.py`，函数 `retrieve`（messages 列表中的 user content）

```text
Question: {question}
```

### 从检索上下文生成答案 prompt（inline f-string）

来源：`run_judge.py`，函数 `generate_answer`（变量 `prompt`，用于 Mem0 等 baseline 从检索上下文生成答案）

```text
Answer the question using ONLY the provided context. If the context does not contain the answer, say "NOT FOUND".

Context:
{context}

Question: {question}

Answer concisely.
```

## judge prompt

### JUDGE_SYSTEM（标准评测 judge 系统提示词）

来源：`eval_standard.py`，模块级变量 `JUDGE_SYSTEM`

```text
You are evaluating conversational AI memory recall. Return JSON only.
```

### JUDGE_PROMPT（标准评测 0/1 二元 judge 提示词）

来源：`eval_standard.py`，模块级变量 `JUDGE_PROMPT`（Mem0 风格 0/1 binary judge；`{{ }}` 为源码中 `.format` 的转义花括号，逐字保留）

```text
Label the generated answer as CORRECT or WRONG.

## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. Judge semantic meaning, not exact wording.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT.

5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea. Different wording should not result in WRONG if the underlying concept matches.

6. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer
- The answer addresses a completely different topic

## Question
Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return JSON: {{"reasoning": "one sentence", "label": "CORRECT" or "WRONG"}}
```

### 0-100 打分 judge prompt（inline f-string）

来源：`run_judge.py`，函数 `judge_score`（变量 `prompt`）

```text
Score 0-100 based on factual accuracy and completeness.

Question: {question}
Ground Truth: {gold}
System Answer: {answer}

0: completely wrong or NOT FOUND when answer exists.
25: related topic but key facts wrong.
50: core facts correct but important details missing.
75: mostly correct with minor omissions.
100: all factual details correct.

Output ONLY a number 0-100.
```

## 其他

### bash 工具 description（tool schema 内的自然语言描述）

来源：`memory_builder_v2.py`，模块级变量 `TOOLS`（`READ_TOOLS = TOOLS`，构建/检索/重组织/修复共用）

```text
Execute a shell command in the memory folder. The working directory is always the root of the memory folder. Use standard Linux commands (ls, cat, grep, mkdir, tee, etc.) to read and write files.
```

其余分类说明：`memory_builder_v2.py`、`eval_standard.py`、`run_judge.py` 三个文件中不存在其他 prompt。
