# Mem0 官方 benchmarks 仓库 Prompt 汇总

来源代码根目录：`code/mem0-benchmarks/`

涉及三个 benchmark 的 prompt 文件：

- `benchmarks/locomo/prompts.py`（LOCOMO）
- `benchmarks/longmemeval/prompts.py`（LongMemEval）
- `benchmarks/beam/prompts.py`（BEAM）

所有 prompt 均为代码逐字原文（含 `.format()` / f-string 的 `{placeholder}` 与 `{{}}` 转义写法）。

## 构建/记忆提取 prompt

无。此仓库的这三个 prompt 文件只包含答案生成（answer generation）和评判（judge）prompt，不含记忆构建/提取 prompt。

## Answer prompt

### LOCOMO — `ANSWER_GENERATION_PROMPT`

来源：`benchmarks/locomo/prompts.py`，模块级常量 `ANSWER_GENERATION_PROMPT`（由 `get_answer_generation_prompt()` 填充 `{memories}`、`{question}`、`{reference_date}`；`ANSWERER_MEMORY_LIMIT = 200`）

```text
You are answering a question using retrieved memories from past conversations. Follow these reasoning steps IN ORDER.

## Step 1: SCAN ALL MEMORIES
Read EVERY memory below from first to last. For each one that contains information relevant to the question, note it. Do NOT stop after finding the first relevant memory — important details are often scattered across many memories, including ones far down the list. Give equal weight to ALL memories regardless of position — a memory near the end is just as likely to contain the answer as one near the beginning. In these memories, "User" refers to the main person whose memories these are.

## Step 2: ENTITY VERIFICATION
Confirm each relevant memory is about the correct person/entity. If the question asks "What does Person A like?" and a memory says "Person B likes X", do NOT use that memory to answer about Person A. In two-person conversations, both speakers' actions are relevant — if the question asks about person A and a memory attributes an action to person B (the other speaker), that information is still valid evidence from their shared conversations, but always check the attribution is correct.

## Step 3: COMBINE AND CROSS-REFERENCE
- COMBINE facts from multiple memories about the same topic. If one memory says "won first place" and another says "performed a piece titled X," those describe the same event — connect them.
- For listing/counting questions, extract EVERY distinct item from ALL memories. A single memory may contain multiple items. Think about what CATEGORIES of answers the question could have, then re-scan specifically for each category.
- For counting questions ("how many times", "how many X"), enumerate each distinct instance explicitly with its date or context BEFORE giving a final count. Do not estimate — list them out, then count the list.
- DECOMPOSE complex sentences: "an immersive X with Y, enjoys Z" contains multiple distinct facts. Each could be the answer.
- Connect related facts across memories: if one says "nearby lake" and another says "Lake Tahoe is great for kayaking", the nearby lake IS Lake Tahoe. If one says "bought X in Paris", infer the country is France.

## Step 4: SELECT THE BEST ANSWER
- Do NOT assume the highest-ranked memory is correct. Multiple memories may describe different events for the same topic. Compare each candidate's relevance to the SPECIFIC question, not its retrieval score. A lower-ranked memory that directly answers the question beats a higher-ranked one that is only tangentially related.
- ALWAYS choose the MOST SPECIFIC detail available. A proper name, title, or number beats a generic description. Rate each candidate as HIGH specificity (name, title, number, specific activity) or LOW (generic description), and prefer HIGH.
- Report what someone actually DID, not what was offered or available to them. "Has not tried X yet" means X was NOT done — disqualify it. "Joined X" or "has done X" means it WAS done — prefer it.
- When multiple memories repeat the same generic fact, that repetition does NOT make it more correct than a single memory with a more specific answer.
- Photos depict what was IN the photo, not facts about someone's daily life. Prefer direct statements over photo descriptions for inferences.
- Re-read the question carefully before answering. If it asks "what aspect/type/kind", answer with the specific aspect. If it asks "what did they discover they both enjoy", answer with the specific thing, not the setting.

## Step 5: TEMPORAL GROUNDING
These conversations took place around {reference_date}. All events occurred in 2022-2024.
- Calculate time relative to this date, NOT today. Never output 2025 or 2026.
- Use dates explicitly stated in memory text. Do not invent or estimate dates.
- When a question asks what someone "shared" or "mentioned" on a date, that date is when they TALKED about it — look for events shortly BEFORE that date.
- For "how long" questions, find the start and end dates explicitly, then compute the duration. Do not guess.
- TEMPORAL DISAMBIGUATION: When you find MULTIPLE instances of similar events at different dates, enumerate them all with their dates before picking. If the question uses past tense + "the" → select the instance closest to (and before) the reference date. If future tense ("plans to", "going to") → select the earliest planned date. NEVER default to the first-mentioned or highest-scored instance — the DATE determines the answer.

## Step 6: INCLUSION CHECK (for lists and counts)
If you found items during reasoning that you're tempted to exclude from your answer — STOP. Include them unless you have STRONG evidence they are wrong. The most common mistake is finding relevant items but then dropping them due to overly strict filtering. More items is better than fewer when there is supporting evidence.
- For counting: after enumerating, re-verify each item. Check for duplicates (same event described differently) and ensure you haven't missed items from memories late in the list.
- The question assumes something happened. Find WHAT happened, don't say nothing happened.

## Step 7: COMMIT AND ANSWER
Give a direct, specific answer. NEVER say "not specified", "not mentioned", "no record", or "the memories don't say" — if ANY memory contains relevant information, give the best answer from available evidence. No hedging, no caveats. If the question asks for a list, include ALL items found. NEVER return an empty answer when relevant memories exist.
- NEVER generate specific names, titles, places, or dates that do not appear in any memory above. If no memory contains the specific detail the question asks for, answer with what the memories DO contain rather than guessing.
- For open-domain/opinion questions ("Would X do Y?", "Is X considered Z?"):
  * Follow the DIRECT causal reasoning in the memories. Do NOT construct elaborate counter-arguments.
  * "Would X still do Y without Z?" — If memories show X does Y BECAUSE of Z, then without Z, answer "likely no."
  * "Would X do Y again soon?" — If the most recent attempt involved a bad experience (accident, scare, trauma), answer "likely no." A recent negative experience outweighs historical positive patterns.
  * For trait questions ("Is X considered Z?"): weigh ALL evidence including symbolic/indirect references. If there is SOME but not strong evidence, answer with a qualified degree ("somewhat") rather than flat "no."

# Instructions

## Misc

1. Make reasonable deductions based on your memories. Memory shows store with a lot of working people -> store employs a lot of people
2. If a memory describes something recognizable (e.g., "romantic drama about memory and relationships"), you may name it (e.g., "Eternal Sunshine of the Spotless Mind").
3. Use domain knowledge to connect facts: a game exclusive to one platform implies ownership of that platform. An unnamed company deal can be linked to a previously expressed brand preference.

{memories}

Question: {question}

Work through Steps 1-7, then give your final answer after "ANSWER:".
```

### LongMemEval — `ANSWER_GENERATION_PROMPT`

来源：`benchmarks/longmemeval/prompts.py`，模块级常量 `ANSWER_GENERATION_PROMPT`（由 `get_answer_generation_prompt()` 填充 `{memories}`、`{question_date}`、`{question}`）

```text
You are a personal assistant with access to memories from past conversations with a user. Answer the question using information from the memories below. Be direct and concise.

IMPORTANT: Today's date is {question_date}. All relative time expressions MUST be computed relative to this date.

IMPORTANT: If memories indicate the user wants to avoid something, your answer must NOT contain it — not as primary, secondary, or context.

IMPORTANT: If memories contain the numbers needed to compute the answer (ages to subtract, prices, dates to diff), DO the computation. NEVER abstain when the raw data exists — even scattered across different conversations.

IMPORTANT: Keep your responses short. No need to go into too much detail, no need to describe things at the lowest level. You can generally describe events and ideas abstractly.

IMPORTANT: Pay close attention to the EXACT entity in the question. If the question asks about a specific variant and memories only mention a DIFFERENT variant (e.g., "electric guitar" vs "acoustic guitar"), abstain — these are talking about different things!

IMPORTANT: For comparison/savings questions, BOTH costs must come from USER-stated facts (or user-relayed, e.g., "my friend said"). Do NOT use assistant-provided general info. If only one side has a user-stated cost, abstain.

IMPORTANT: If the query uses a specific but WRONG role/title/entity (e.g., asks about experience as a "Sales Manager" but memories say "Senior Sales Engineer"), do NOT answer as if they match — instead say you don't have the information! Always lean towards abstention in these cases! Do not mix up different role titles, they are not the same roles and you should say you don't have information.

Before answering, reason step-by-step inside <mem_thinking> tags:
- List every relevant memory; try to list all memories relevant to what the user wants to do! Eg. List memory of Payment management apps if query is about paying someone; list memory of travel management apps if query is about going somewhere.

- For counting: enumerate each item with date. Apply the question's EXACT verb/qualifier strictly (e.g., "LED" = leader only, "BAKED" = completed baking only, "RAISED" = total from events user participated in (include team/event totals), "COMPLETED writing" = each distinct finished piece). Count multiple items in a single memory separately. Do a SECOND full scan of all memories after initial count — items at positions 30-200 are commonly missed. Verify each item is a completed action (past tense), not a plan ("plans to", "intends to").
- For cross-topic computation: scan ALL memories for each needed fact independently — they're often in unrelated conversations. List: (a) what you need, (b) where each appears, (c) the computation.
- For temporal questions: identify dates, compute intervals from {question_date}
- CONTEXT CHECK: Before using a memory's value, verify it applies to the SAME context as the question. A wake-up time "while traveling" is NOT the same as a regular weekday wake-up time. A "general daily" schedule may conflict with a "specific weekday" schedule — always prefer the more specific memory that matches the question's context. List the context of each memory (weekday routine vs. travel vs. weekend vs. specific day) and only use values from the matching context.
- For time-bounded counting: compute the INCLUSIVE date window first, then check EVERY item's date. Err on inclusion for ambiguous dates.
- For "where is X": trace location chronologically through memories
- For suggestions: list (a) what user has/does, (b) what they avoid/dislike, (c) what they want to explore. Check every suggestion against (b) before including.
- State your conclusion

The user will only see text outside the <mem_thinking> tags.

Rules:

1. **Always try to answer**: If the topic appears in any memory — even indirectly — answer using what you have. Don't refuse for one missing detail.

2. **Most recent wins**: For conflicting values of the same fact, use the most recent memory. But: (a) memories about different people/contexts aren't conflicting; (b) for historical event dates, use the memory recorded closest to the event; (c) for current counts/scores/status, the latest value REPLACES all earlier ones — don't sum or average.

Similarly, when memories give two numbers for the same metric (e.g., "has 1,250 followers" and "close to 1,300 followers") on the same date, treat the HIGHER/UPDATED value as current — "close to 1,300" means the count has grown from 1,250 to approximately 1,300.

3. **Time-bounded questions**: Compute the date window from {question_date}. Show date arithmetic in <mem_thinking>. Scan EVERY memory for events in range. "Last weekend" is imprecise — could mean up to 10 days ago as people sometimes mean weekend before the latest one. "Last 3 months" can include boundary days of the 4th month back.

"Last month" includes the current month so far as well as the previous month. Eg. "last month" in Late May includes all of April. If the literal window yields nothing, check the immediately preceding period.

4. **Temporal reference points**: "How many days ago did X when Y happened" — compute interval between X and Y, NOT between X and today.

5. **Counting and ordering**: Scan ALL memories first to last. Build a numbered list in <mem_thinking> with date and position. Deduplicate by matching dates/descriptions. Count items in a single memory separately.
Any addition to a list on the same day as a stated count is already included in the count

When asked to count all instances of an event *before* a specific one, obviously don't include the specific one in the count. Eg. "how many restaurants did i visit before eating at Pizza Hut?". Obviously don't include Pizza Hut in the count

6. **Use only the memories**: Don't invent numbers, prices, or addresses.

7. **When to abstain**: Say "The information provided is not enough" when:
   - The topic is genuinely unmentioned

- The question asks about a specific event that doesn't exist, even if a related topic does

- IMPORTANT: If the query uses a specific but WRONG role/title/entity (e.g., asks about experience as a "Sales Manager" but memories say "Senior Sales Engineer"), do NOT answer as if they match — instead say you don't have the information! Always lean towards abstention in these cases! Do not mix up different role titles, they are not the same roles and you should say you don't have information.

   - For comparison/ordering, BOTH items must be present as completed events
   If query asks to compare timings of two tasks and one of them did not even happen, abstain.
   Before abstaining, do a keyword scan of ALL memories (they're chronological, not relevance-sorted — check positions 1-200). Only abstain if NO keywords match.
   EXCEPTIONS: For suggestion questions, don't abstain for lack of real-time info — recommend based on known preferences. If you lack exact brand but have the store, output the store.

8. **Yes/no and comparison**: "Did I ever do X?" with no matching memory = "No." For comparisons, find both values across all memories and compare directly.

9. **Actions vs intentions**: Use the date of actual execution, not the plan date. "Decided to" or "took X for servicing" = action initiated. Only treat as plan if explicit future-tense ("plans to", "will"). A plan with a specified date and no update = assume completed on that date. If a later memory confirms execution, use the execution date — it supersedes the earlier plan.

When a query asks: "when I decided to do X", it means they are asking when X was actually done.

10. **User facts vs assistant advice**: "User..." = actual experience. "Assistant..." = advice. Prefer user-stated facts for personal questions. Don't convert currencies unless user stated the conversion.

11. **Connect memories across topics**: Facts needed for computation are often in unrelated conversations (age in travel advice + relative's age in birthday discussion; cashback rate in membership talk + purchase amount in expense tracking). Search ALL memories for each fact independently.

12. **Personalization**: For suggestions/recommendations:
   - Prioritize personal preferences over informational content
   - Apply known preferences to new contexts — don't abstain for unfamiliar destinations
   - Acknowledge prior work before suggesting next steps
   - Respect anti-preferences — check every suggestion against known dislikes
   - Reference existing tools owned, not to acquire
   - Lead with personalization, don't pad with generic alternatives
   - Suggest similar things to the user as their habits. Eg. Logging basketball scores in a app they do usually. Eg. Adding travel logs to a travel logging app they use usually.
   - IMPORTANT: Scan ALL top memories for user-owned tools, apps, and resources relevant to the question. If the user has a travel card (Suica), a trip organizer app (TripIt), a budgeting tool, etc., mention ALL of them — not just the most obvious one. Do a SECOND pass of the top 30 memories specifically looking for apps, tools, and resources the user has mentioned owning or using.

13. **Reasonable deduction**:
- Infer from patterns
IMPORTANT: Assume that similar items referenced in the same sentence have the same type.
Eg. "User ate lunch, which was the third meal with this chicken fajitas". This means the other meals with these chicken fajitas were lunch meals too, should be treated as explicit lunches.

14. IMPORTANT: If two pieces of memory directly contradict each other (not just an update, a direct contradiction), then assume that the memory that was created later is true. Doesn't matter if a different one "appears" more reliable. If on the same day, trust the one at a later time.

- Chronological actions:
If the user is watching the 11th episode of a series is watching it normally, assume they have completed the earlier 10 too.

- If you lack a name but have a description, answer with the description.

**Memory grouping rules**: Memories under the same date heading are from the same conversation.
- A count + "added X items" on the SAME date = count already includes them
- "Aims to beat X" = X is the current value
- "Previous" = the value superseded by a more recent one
- Events described as just completed ("attended", "went to", "just got back from", "completed") = happened on/near that date. Undated actions = assume the event happened on the memory's date.

# Misc Rules
- Count class projects too when asked about users' projects. Class projects = projects.
- Most old (Eg. ancestral, vintage, heritage) items count as antiques too!
- If you don't have chords for a song (but have notes), output the notes. Song notes count as chord progressions.
- Starting a *diorama project* (eg. diorama work, working on terrain) EXPLICITLY COUNTS AS working on that model kit; these are equivalent! Always count such items.
- Running into someone at a coffee shop and exchanging numbers DOES NOT count as meeting them; lunch meetings do count.
- Potlucks/feasts/birthday parties count as dinner parties (BBQ doesn't).
- chandelier counts as jewelry
- Always assume birthdays cleanly follow years. Ie. User was 22 in 2022; they will be 23 in 2023.
- "scratch grains" count as "new layer feed", always include them when interpreting "new layer feed"

Memories (sorted newest-first, grouped by date):
{memories}

Today's Date: {question_date}
Question: {question}

IMPORTANT: You MUST provide your full thinking in <mem_thinking> tags BEFORE giving your answer.; Reasoning and answer:
```

### BEAM — `ANSWER_GENERATION_PROMPT`

来源：`benchmarks/beam/prompts.py`，模块级常量 `ANSWER_GENERATION_PROMPT`。同文件的 `get_beam_answer_generation_prompt()` 函数内嵌了一份内容完全相同的 f-string（仅 `{memories}` 写作 `{memories_text}`），不再重复列出。

```text
You are an AI assistant with access to stored memories from prior conversations with a user.
Use these memories to answer the following question as accurately and completely as possible.

IMPORTANT RULES:
1. Scan ALL provided memories before answering — do not stop after the first relevant one.
2. If multiple memories contain relevant information, combine and cross-reference them.
3. If the memories contain contradictory information, prefer the more recent one.
4. If the memories don't contain enough information to answer, say exactly: "I don't have enough information to answer this question."
5. For temporal questions: pay attention to dates and relative time references.
6. For ordering questions: present events in chronological order.
7. For preference questions: use the most recently stated preference.
8. Be specific and direct — include exact names, dates, numbers, and details from the memories.
9. Do NOT invent or assume information that isn't in the memories.

QUESTION: {question}

RETRIEVED MEMORIES:
{memories}

ANSWER:
```

## Judge prompt

### LOCOMO — `JUDGE_SYSTEM_PROMPT`

来源：`benchmarks/locomo/prompts.py`，模块级常量 `JUDGE_SYSTEM_PROMPT`

```text
You are evaluating conversational AI memory recall. Return JSON only with the format requested.
```

### LOCOMO — `_JUDGE_TEMPLATE`

来源：`benchmarks/locomo/prompts.py`，模块级常量 `_JUDGE_TEMPLATE`（统一 judge 模板；`{evidence_section}`、`{evidence_rule}`、`{evidence_wrong_clause}` 由 `_build_judge_prompt()` 注入，`{{question}}`、`{{answer}}`、`{{response}}` 为二次 format 的转义占位符；无 evidence 版本即模块常量 `JUDGE_PROMPT`）

```text
Label the generated answer as CORRECT or WRONG.
{evidence_section}
## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. "Chocolate raspberry tart" = "chocolate cake with raspberries". "Shelter meal service" = "volunteering at a homeless shelter". Emotions and sentiments in the same positive/negative family count as paraphrases: "proud" = "fulfilled" = "accomplished"; "huge success" = "relieved" = "thrilled" (all express positive achievement). Judge semantic meaning, not exact wording.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed or specific. If the generated answer adds extra descriptive details beyond the gold answer while still referencing the same core entity or concept, mark CORRECT.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT (e.g., "5 months" matches "six months"; "19 days" matches "two weeks"). Relative dates ("few days before November") match specific dates in the same window. A specific date (e.g., "February 2020") that is consistent with a vague reference (e.g., "a few years ago" relative to 2023) is CORRECT. Converting "last year" to the actual year (e.g., "2022" when conversations are in 2023) is CORRECT.
{evidence_rule}
5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches. For EMOTIONS and FEELINGS questions, answers expressing sentiments in the same valence (positive/negative) about the same event are CORRECT — do not require the exact same emotion word.

6. **SAME REFERENT**: If the generated answer mentions or references the same named entity, character, person, or concept as the gold answer, mark CORRECT — even if the generated answer provides a different physical description or includes additional details. The key question is: does the generated answer identify the same core entity? If yes, it is CORRECT.

7. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact. Minor differences in specificity, phrasing, or scope should not result in WRONG. Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer{evidence_wrong_clause}
- The answer addresses a completely different topic

## Question
Question: {{question}}
Gold answer: {{answer}}
Generated answer: {{response}}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels.
```

### LOCOMO — `_EVIDENCE_CHUNK`

来源：`benchmarks/locomo/prompts.py`，模块级常量 `_EVIDENCE_CHUNK`（有 evidence 时注入 `_JUDGE_TEMPLATE` 的 `{evidence_section}`）

```text

## Evidence (actual conversation messages containing the answer)
{evidence_context}

```

### LOCOMO — `_EVIDENCE_RULE`

来源：`benchmarks/locomo/prompts.py`，模块级常量 `_EVIDENCE_RULE`（有 evidence 时注入 `_JUDGE_TEMPLATE` 的 `{evidence_rule}`，随后 `_build_judge_prompt()` 会把原第 5/6/7 条规则重新编号为 6/7/8）

```text

5. **EVIDENCE SUPPORTS ANSWER**: If the evidence corroborates the generated answer, mark CORRECT — even when the generated answer diverges from the gold answer. The gold answer may be wrong or oversimplified; if the generated answer provides a more accurate or better-supported conclusion based on the evidence, that is acceptable. Use evidence only to ACCEPT answers, never to reject them more strictly.

```

### LOCOMO — `_EVIDENCE_WRONG_CLAUSE`

来源：`benchmarks/locomo/prompts.py`，模块级常量 `_EVIDENCE_WRONG_CLAUSE`（有 evidence 时注入 `_JUDGE_TEMPLATE` 的 `{evidence_wrong_clause}`）

```text
 AND is not supported by evidence
```

### LongMemEval — `JUDGE_PROMPT`

来源：`benchmarks/longmemeval/prompts.py`，模块级常量 `JUDGE_PROMPT`（由 `get_judge_prompt()` 填充 `{question}`、`{answer}`、`{response}`；所有 question type 共用此统一 prompt）

```text
I will give you a question, a correct answer (or rubric), and a model response. Decide whether the model response is correct.

CORE PRINCIPLE — Semantic equivalence: Judge by MEANING, not exact words. Answer "yes" if every concept in the correct answer is addressed in the response, even with different vocabulary, more specific terms, or restructured phrasing.

IMPORTANT BIAS CHECK: You have a tendency to say "no" too quickly. Before concluding "no", you MUST verify the answer is truly wrong, not just differently worded. When in doubt, lean toward "yes".

Rules:

**Equivalence & Supersets**
- Equivalent or superset responses are correct. Extra details are fine unless proven to be factually wrong. Extra qualifiers are fine unless proven to be wrong. E.g., "a blue dress and a matching necklace" is correct when the answer is "a blue dress."
- If a response captures the most specific part (exact item/place/name) but omits a broader container, it's correct.
- Same factual meaning with different phrasing = correct (e.g., "No, you did not visit with a friend" ≈ "You didn't mention going with anyone").
- Adding scope qualifiers like "regular-season" or "excluding X" is fine as long as the core value is correct. The qualifier may narrow the context but does NOT make the answer wrong unless the correct answer explicitly includes the excluded items.

**Lists & Compound Terms**
- For list answers, match each item by semantic meaning. A concept is covered if restated via synonyms, sub-concepts, or related terms. Adding methodological detail or rewording verbs to near-synonyms is acceptable.
- A broad term like "A and B significance" is covered if the response addresses the topic area through related specific terms, even without naming each component literally.
- If some items as listed as "or"s, "maybe"s and potential answers, it's okay if the answer does not include those.
- If two items in a list achieve the same purpose, listing just one of them is fine.

IMPORTANT: The "anti-preference" items are very specific!
Eg. Someone "not interested in general AI topics" could be very interested in specific AI topics in general AI *conferences*; those are not the same thing and should be accepted! topics != conferences

**Numbers & Precision**
- Hedging ("at least 3", "approximately") is fine if the core number matches. A range that includes the correct answer is correct.
Generally, if the user themself would be satisfied by the response, it is acceptable. Ie. If the answer is conditional on information they would have (eg. their birthday, some hidden dependent information), and would be correct with that information, that is acceptable.
- More precise answers are correct: "22 days" matches "3 weeks"; "over $270" matches "$270."; "9 1/2 months" matches "9 months";

- Rough answers are correct: "about nine months" ≈ "9 months; "8 months and 20 days" matches "9 months";

- Off-by-one errors on days/weeks/months are acceptable.
- Approximate unit conversions are equivalent: "14 weeks" ≈ "3 months", "6 months" ≈ "half a year."
- Round time ranges generously: 7 months and 16 days ≈ 8 months.
- Notes instead of chords are acceptable when justified
- A correct number with added context (e.g., "about 5 months ago (around December 2022)") is correct — the parenthetical date is supplementary, not a contradiction.

**Dates & Temporal**
- Date format variations are equivalent: "February 1st" = "Feb 1, 2023" = "on February 1."
- Same-day event ordering swaps are acceptable.
- Outdated info alongside the correct updated answer is acceptable if the current value is identified.
- "recent" is upto 6 years ago, which means 2017+
- References like "last weekend", "last Wednesday", etc. are imprecise - people sometimes mean the weekend/Wednesday before the latest one if they're near it. "Last 3 months" can include boundary days of the 4th month back. "Last month" includes the current month so far. Be flexible with such timestamps

**Counting Edge Cases**
- If correct answer is "0" or "nothing found," model saying "not enough information" is also correct.
- Similarly, If correct answer is "not enough information", model saying "0" or "nothing found," is also correct.

**Preference/Personalization Rubrics** (apply in order):
1. Correct if the response demonstrates awareness of user's personal context (preferences, habits, interests). Need not satisfy every rubric point.
2. Primary criterion: do main suggestions align with what the user WANTS?
3. Anti-preferences: evaluate the OVERALL thrust, not keyword scanning. If the response largely suggests correct options, minor incidental references to "not-preferred" things are fine.
4. Mentioning a phone app as a MEANS to a preferred activity (e.g., meditation app for sleep) is not "suggesting phone use." Judge by the activity, not delivery mechanism.
5. "May not prefer" = mild preference, not hard prohibition. Secondary/context-dependent inclusion is fine.
6. Explicit acknowledgment of anti-preferences (e.g., "keep screens off") strengthens correctness.
7. Context-dependent suggestions are acceptable (reading is fine on a bus even if rubric flags visual attention activities). Adjacent genres alongside preferred ones are additive, not contradictory.
8. If the rubric mentions specific user resources/tools (e.g., "Suica card", "TripIt app"), the response is correct if it demonstrates awareness of the user's MAIN personal context even if it does not name every specific tool. The rubric is a guide, not a checklist.

**Abstention Matching**
- If correct answer = unanswerable/abstention, ANY phrasing that conveys "I don't have this information" is correct, regardless of what partial context is mentioned or omitted.
- Saying "not enough information" while mentioning partial related context = correct abstention.
- Saying "no record of X" or "only have plans for X, not actual dates" = correct abstention.
- The key test: does the response REFUSE to answer the question? If yes, it matches an abstention ground truth, period.

FINAL CHECK: Before answering "no," you MUST reason through these steps:
1. What is the core factual claim or intent of the correct answer?
2. Does the model response address that same claim, even in different words?
3. Is the response a superset (correct answer + extra details)?
4. For numbers: does the core number match, ignoring hedging/qualifiers?
5. For abstentions: does the response effectively decline to answer?
Only answer "no" if, after this analysis, a core concept is entirely unaddressed or contradicted.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Think step-by-step in <judge_thinking> tags, then give your final verdict as exactly "yes" or "no" on a new line after the closing tag.
```

### BEAM — `BEAM_JUDGE_SYSTEM_PROMPT`

来源：`benchmarks/beam/prompts.py`，模块级常量 `BEAM_JUDGE_SYSTEM_PROMPT`

```text
You are an expert evaluator assessing whether an AI assistant's response satisfies specific rubric criteria. You must be objective, fair, and consistent. Return ONLY valid JSON with the exact format requested.
```

### BEAM — `JUDGE_PROMPT`

来源：`benchmarks/beam/prompts.py`，模块级常量 `JUDGE_PROMPT`（供 EvalsManager Prompt Playground 提取用的常量版本；`{question}`、`{response}`、`{answer}` 为占位符，`{{...}}` 为 format 转义）

```text
Evaluate whether the following LLM response demonstrates compliance with the specified RUBRIC CRITERION.

QUESTION:
{question}

LLM RESPONSE:
{response}

RUBRIC CRITERION:
{answer}

SCORING GUIDELINES:

First, determine whether the rubric criterion is a POSITIVE requirement (the response SHOULD include something) or a NEGATIVE constraint (the response SHOULD NOT include something).

**For POSITIVE requirements** (response should contain, mention, or demonstrate something):
- **1.0 (Complete Compliance)**: The required element is present, accurate, and complete. The response fully and clearly satisfies the rubric criterion.
- **0.5 (Partial Compliance)**: The required element is partially present, has minor inaccuracies, or is incomplete. The core intent is present but not fully realized.
- **0.0 (No Compliance)**: The required element is missing, incorrect, or the response is entirely off-topic / non-responsive.

**For NEGATIVE constraints** (response should NOT contain or should avoid something):
- **1.0 (Complete Compliance)**: The response is responsive to the question AND the prohibited element is absent.
- **0.5 (Partial Compliance)**: The response is responsive but contains a borderline or ambiguous reference to the prohibited element.
- **0.0 (No Compliance)**: The prohibited element is present in the response, OR the response is non-responsive (off-topic, refusal, empty).

**Compound statement handling**: If the rubric criterion contains "and" or commas connecting multiple required elements:
- All elements present and correct = 1.0
- Some (but not all) elements present and correct = 0.5
- No elements present or correct = 0.0

EVALUATION RULES:
1. **Semantic tolerance**: Paraphrases and synonyms are acceptable. The response does not need to use the exact same words as the rubric.
2. **Numeric and date equivalence**: Treat equivalent representations as identical. "$68,000" = "68k" = "sixty-eight thousand dollars". "2 years" = "24 months". Prefer normalized comparison for numbers, currencies, dates, and durations.
3. **Case / punctuation / whitespace tolerance**: Differences in capitalization, punctuation, and whitespace must be ignored when comparing content.
4. **Hedging tolerance**: Do not penalize hedging language ("I think", "probably", "it seems"), passive voice, or verbosity if the substantive content satisfies the rubric criterion.
5. **Style neutrality**: Do not penalize for tone, formatting, or length unless the rubric criterion specifically requires a particular format.
6. **Responsiveness**: If the LLM response is completely off-topic or refuses to answer, score 0.0.
7. **Independence**: Evaluate this criterion in isolation — do not consider other rubric items.
8. **Specificity matters**: Vague or generic answers that could apply to any question score lower than specific, detailed answers.

STEP-BY-STEP EVALUATION:
Follow these steps in order:
1. **Understand the Requirement**: Read the rubric criterion and classify it as a positive requirement or a negative constraint.
2. **Parse Compound Statements**: If the criterion contains multiple sub-requirements joined by "and" or commas, identify each element separately.
3. **Check Compliance**: Compare the LLM response against each element, applying the tolerance rules above (semantic, numeric, case, hedging).
4. **Assign Score**: Use the appropriate scoring table (positive or negative) and compound-statement rule to determine the score.
5. **Provide Reasoning**: Write a concise explanation referencing which elements were or were not satisfied.

Return your evaluation as a JSON object with exactly two fields:
{{"score": <0.0 or 0.5 or 1.0>, "reason": "<one concise sentence explaining your score>"}}
```

### BEAM — `get_beam_nugget_judge_prompt()`（实际运行使用的 rubric nugget judge）

来源：`benchmarks/beam/prompts.py`，函数 `get_beam_nugget_judge_prompt(question, nugget, llm_response)` 返回的 f-string。与上面的 `JUDGE_PROMPT` 常量几乎相同，差异有两处：占位符名（`{llm_response}`、`{nugget}`），以及 EVALUATION RULES 第 6 条末尾多了 "for all criteria"。以下为 f-string 代码原文（`{question}` 等在运行时被实参插值，`{{...}}` 为 f-string 转义）：

```text
Evaluate whether the following LLM response demonstrates compliance with the specified RUBRIC CRITERION.

QUESTION:
{question}

LLM RESPONSE:
{llm_response}

RUBRIC CRITERION:
{nugget}

SCORING GUIDELINES:

First, determine whether the rubric criterion is a POSITIVE requirement (the response SHOULD include something) or a NEGATIVE constraint (the response SHOULD NOT include something).

**For POSITIVE requirements** (response should contain, mention, or demonstrate something):
- **1.0 (Complete Compliance)**: The required element is present, accurate, and complete. The response fully and clearly satisfies the rubric criterion.
- **0.5 (Partial Compliance)**: The required element is partially present, has minor inaccuracies, or is incomplete. The core intent is present but not fully realized.
- **0.0 (No Compliance)**: The required element is missing, incorrect, or the response is entirely off-topic / non-responsive.

**For NEGATIVE constraints** (response should NOT contain or should avoid something):
- **1.0 (Complete Compliance)**: The response is responsive to the question AND the prohibited element is absent.
- **0.5 (Partial Compliance)**: The response is responsive but contains a borderline or ambiguous reference to the prohibited element.
- **0.0 (No Compliance)**: The prohibited element is present in the response, OR the response is non-responsive (off-topic, refusal, empty).

**Compound statement handling**: If the rubric criterion contains "and" or commas connecting multiple required elements:
- All elements present and correct = 1.0
- Some (but not all) elements present and correct = 0.5
- No elements present or correct = 0.0

EVALUATION RULES:
1. **Semantic tolerance**: Paraphrases and synonyms are acceptable. The response does not need to use the exact same words as the rubric.
2. **Numeric and date equivalence**: Treat equivalent representations as identical. "$68,000" = "68k" = "sixty-eight thousand dollars". "2 years" = "24 months". Prefer normalized comparison for numbers, currencies, dates, and durations.
3. **Case / punctuation / whitespace tolerance**: Differences in capitalization, punctuation, and whitespace must be ignored when comparing content.
4. **Hedging tolerance**: Do not penalize hedging language ("I think", "probably", "it seems"), passive voice, or verbosity if the substantive content satisfies the rubric criterion.
5. **Style neutrality**: Do not penalize for tone, formatting, or length unless the rubric criterion specifically requires a particular format.
6. **Responsiveness**: If the LLM response is completely off-topic or refuses to answer, score 0.0 for all criteria.
7. **Independence**: Evaluate this criterion in isolation — do not consider other rubric items.
8. **Specificity matters**: Vague or generic answers that could apply to any question score lower than specific, detailed answers.

STEP-BY-STEP EVALUATION:
Follow these steps in order:
1. **Understand the Requirement**: Read the rubric criterion and classify it as a positive requirement or a negative constraint.
2. **Parse Compound Statements**: If the criterion contains multiple sub-requirements joined by "and" or commas, identify each element separately.
3. **Check Compliance**: Compare the LLM response against each element, applying the tolerance rules above (semantic, numeric, case, hedging).
4. **Assign Score**: Use the appropriate scoring table (positive or negative) and compound-statement rule to determine the score.
5. **Provide Reasoning**: Write a concise explanation referencing which elements were or were not satisfied.

Return your evaluation as a JSON object with exactly two fields:
{{"score": <0.0 or 0.5 or 1.0>, "reason": "<one concise sentence explaining your score>"}}
```

## 其他

### BEAM — `get_beam_fact_extraction_prompt()`（event ordering 评测：事件抽取）

来源：`benchmarks/beam/prompts.py`，函数 `get_beam_fact_extraction_prompt(response)` 返回的 f-string（用于 Kendall tau-b 计算前从回答中按出现顺序抽取事件）

```text
Extract all distinct events or facts mentioned in the following response,
in the exact order they are presented. Return ONLY a JSON array of short event descriptions.

RESPONSE:
{response}

Return format: ["event 1 description", "event 2 description", ...]
```

### BEAM — `get_beam_event_alignment_prompt()`（event ordering 评测：事件对齐）

来源：`benchmarks/beam/prompts.py`，函数 `get_beam_event_alignment_prompt(extracted_event, rubric_events)` 返回的 f-string（`{events_list}` 由 rubric events 编号列表拼接而成；`{{...}}` 为 f-string 转义）

```text
Given the following extracted event from an LLM response, determine which
reference event it best corresponds to. Return ONLY a JSON object.

EXTRACTED EVENT:
{extracted_event}

REFERENCE EVENTS:
{events_list}

If the extracted event matches one of the reference events (even approximately or paraphrased),
return the 0-based index. If it doesn't match any, return -1.

Return format: {{"index": <integer>, "reason": "<brief explanation>"}}
```
