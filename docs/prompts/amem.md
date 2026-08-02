# A-Mem

代码根目录：`baselines/A-Mem/`（相对 `model-aligned-wiki/code/`）。所有 prompt 为逐字原文，含源码中的缩进。A-Mem 有两套实现：主实现（`memory_layer.py` + `test_advanced.py`，JSON schema 输出）和 robust 变体（`memory_layer_robust.py` + `llm_text_parsers.py` + `test_advanced_robust.py`，纯文本输出）。

## 构建/记忆提取 prompt

### 1. Note construction：内容分析（keywords/context/tags）

来源：`baselines/A-Mem/memory_layer.py`，`MemoryNote.analyze_content()` 中的变量 `prompt`（第 315 行，后接 `+ content`）

```text
Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            
```

### 2. Memory evolution（link generation + neighbor update，合一）

来源：`baselines/A-Mem/memory_layer.py`，`AgenticMemorySystem.__init__()` 中的 `self.evolution_system_prompt`（第 685 行），在 `process_memory()` 中用 `.format(context=..., content=..., keywords=..., nearest_neighbors_memories=..., neighbor_number=...)` 填充

```text

                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.  

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories:
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"], 
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                
```

### 3. Note construction（robust 变体）：内容分析

来源：`baselines/A-Mem/llm_text_parsers.py`，变量 `ANALYZE_CONTENT_PROMPT`（第 128 行），在 `memory_layer_robust.py` 第 320 行使用

```text
Analyze the following content and provide:
1. KEYWORDS: The most important keywords (nouns, verbs, key concepts). Order from most to least important. At least three keywords. Do not include speaker names or time references.
2. CONTEXT: One sentence summarizing the main topic, key points, and purpose.
3. TAGS: Broad categories/themes for classification (domain, format, type). At least three tags.

Respond using EXACTLY this format (one section per header):

KEYWORDS: keyword1, keyword2, keyword3, ...
CONTEXT: A single sentence summarizing the content.
TAGS: tag1, tag2, tag3, ...

Content for analysis:
{content}
```

### 4. Note construction 重试（robust 变体）：只提取关键词

来源：`baselines/A-Mem/llm_text_parsers.py`，变量 `FOCUSED_KEYWORDS_PROMPT`（第 204 行），在 `memory_layer_robust.py` 第 328 行作为重试 prompt 使用

```text
List exactly 5 keywords that capture the main concepts of the following text. Output only the keywords, comma-separated, nothing else.

Text: {content}
```

### 5. Memory evolution 决策（robust 变体）

来源：`baselines/A-Mem/llm_text_parsers.py`，变量 `EVOLUTION_DECISION_PROMPT`（第 143 行），在 `memory_layer_robust.py` 第 478 行使用

```text
You are an AI memory evolution agent. Analyze the new memory note and its nearest neighbors to decide if evolution is needed.

New memory:
- Context: {context}
- Content: {content}
- Keywords: {keywords}

Nearest neighbor memories:
{nearest_neighbors_memories}

Based on the relationships between the new memory and its neighbors, decide:
- NO_EVOLUTION: The memory stands alone, no changes needed.
- STRENGTHEN: The new memory should be linked to some neighbors and its tags updated.
- UPDATE_NEIGHBOR: The neighbors' context/tags should be updated based on new understanding.
- STRENGTHEN_AND_UPDATE: Both strengthen and update neighbors.

Respond using EXACTLY this format:
DECISION: <one of NO_EVOLUTION, STRENGTHEN, UPDATE_NEIGHBOR, STRENGTHEN_AND_UPDATE>
REASON: <brief explanation>
```

### 6. Link generation / strengthen（robust 变体）

来源：`baselines/A-Mem/llm_text_parsers.py`，变量 `STRENGTHEN_DETAILS_PROMPT`（第 164 行），在 `memory_layer_robust.py` 第 496 行使用

```text
Given the new memory and its neighbors, provide updated connections and tags.

New memory:
- Content: {content}
- Keywords: {keywords}

Neighbor memories:
{nearest_neighbors_memories}

Which neighbor indices should the new memory connect to? What tags best describe this memory?

Respond using EXACTLY this format:
CONNECTIONS: 0, 2, 3
TAGS: tag1, tag2, tag3, ...
```

### 7. Neighbor update（robust 变体）

来源：`baselines/A-Mem/llm_text_parsers.py`，变量 `UPDATE_NEIGHBORS_PROMPT`（第 180 行），在 `memory_layer_robust.py` 第 511 行使用

```text
Given the new memory and its neighbor memories, update each neighbor's context and tags based on a holistic understanding of all these memories together.

New memory:
- Content: {content}
- Context: {context}

Neighbor memories:
{nearest_neighbors_memories}

For each neighbor (indexed 0 to {max_neighbor_idx}), provide updated context and tags. If no change is needed, repeat the original values.

Respond using EXACTLY this format (one block per neighbor):

NEIGHBOR 0:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

NEIGHBOR 1:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

(continue for all {neighbor_count} neighbors)
```

## Answer prompt（QA 回答）

### 8. 查询关键词生成

来源：`baselines/A-Mem/test_advanced.py`，`AMemLocomoEval.generate_query_llm()` 中的变量 `prompt`（第 96 行，f-string，`{question}` 为运行时填充）

```text
Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text. 

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}
```

### 9. 记忆相关片段筛选（代码中定义，answer_question 主流程中该调用已被注释掉）

来源：`baselines/A-Mem/test_advanced.py`，`AMemLocomoEval.retrieve_memory_llm()` 中的变量 `prompt`（第 64 行，f-string）

```text
Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

                Conversation memories:
                {memories_text}

                Question: {query}

                Return only the relevant parts of the conversation that would help answer this specific question. Format your response as a JSON object with a "relevant_parts" field containing the selected text. 
                If no parts are relevant, do not do any things just return the input.

                Example response format:
                {{"relevant_parts": "2024-01-01: Speaker A said something relevant..."}}
```

### 10. 回答 prompt：初始默认（死代码，随后总会被各 category 分支覆盖）

来源：`baselines/A-Mem/test_advanced.py`，`AMemLocomoEval.answer_question()` 中的变量 `user_prompt`（第 140 行，f-string）

```text
Context:
                {context}

                Question: {question}

                Answer the question based only on the information provided in the context above.
```

### 11. 回答 prompt：category 5（对抗性问题，二选一）

来源：`baselines/A-Mem/test_advanced.py`，`AMemLocomoEval.answer_question()` 中的变量 `user_prompt`（第 155 行，f-string；`{answer_tmp[0]}`/`{answer_tmp[1]}` 为正确答案与 "Not mentioned in the conversation" 的随机排列）

```text

                            Based on the context: {context}, answer the following question. {question} 
                            
                            Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:
                            
```

### 12. 回答 prompt：category 2（时间类问题）

来源：`baselines/A-Mem/test_advanced.py`，`AMemLocomoEval.answer_question()` 中的变量 `user_prompt`（第 162 行，f-string）

```text

                            Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
                            Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.   

                            Question: {question} Short answer:
                            
```

### 13. 回答 prompt：category 3

来源：`baselines/A-Mem/test_advanced.py`，`AMemLocomoEval.answer_question()` 中的变量 `user_prompt`（第 169 行，f-string）

```text

                            Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            
```

### 14. 回答 prompt：category 1/4（默认分支）

来源：`baselines/A-Mem/test_advanced.py`，`AMemLocomoEval.answer_question()` 中的变量 `user_prompt`（第 175 行，f-string）

```text
Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            
```

### 15. 查询关键词生成（robust 变体）

来源：`baselines/A-Mem/test_advanced_robust.py`，`generate_query_llm()` 中的变量 `prompt`（第 98 行，f-string）

```text
Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:
```

### 16. 记忆相关片段筛选（robust 变体）

来源：`baselines/A-Mem/test_advanced_robust.py`，`retrieve_memory_llm()` 中的变量 `prompt`（第 83 行，f-string）

```text
Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

Conversation memories:
{memories_text}

Question: {query}

Return only the relevant parts of the conversation that would help answer this specific question.
If no parts are relevant, return the input unchanged.
```

### 17. 回答 prompt：category 5（robust 变体）

来源：`baselines/A-Mem/test_advanced_robust.py`，`answer_question()` 中的变量 `user_prompt`（第 125 行，f-string）

```text
Based on the context: {context}, answer the following question. {question}

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:
```

### 18. 回答 prompt：category 2（robust 变体）

来源：`baselines/A-Mem/test_advanced_robust.py`，`answer_question()` 中的变量 `user_prompt`（第 130 行，f-string）

```text
Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:
```

### 19. 回答 prompt：category 3 与 category 1/4 默认分支（robust 变体，两分支文本相同）

来源：`baselines/A-Mem/test_advanced_robust.py`，`answer_question()` 中的变量 `user_prompt`（第 136 行与第 141 行，f-string）

```text
Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:
```

## Judge prompt

无。评测（`baselines/A-Mem/utils.py`）使用 ROUGE / BLEU / BERTScore / METEOR / SBERT 相似度等自动指标，不含 LLM judge prompt。

## 其他

### 20. System message（主实现，所有 LLM 调用统一使用）

来源：`baselines/A-Mem/memory_layer.py`，`get_completion()` 中 messages 的 system role（第 52、98、220 行）

```text
You must respond with a JSON object.
```

### 21. System message（robust 变体，所有 LLM 调用统一使用）

来源：`baselines/A-Mem/memory_layer_robust.py`，类常量 `SYSTEM_MESSAGE`（第 76 行）

```text
Follow the format specified in the prompt exactly. Do not add extra commentary.
```
