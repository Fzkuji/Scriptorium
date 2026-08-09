"""Verbatim prompts for the unified evaluation protocol.

Sources (see docs/experiments/protocols/unified_evaluation_protocol.md §6 and docs/prompts/):
- ANSWER_PROMPT: Mem0 original short-answer prompt, identical copy used by
  Nemori (evaluation/locomo/search.py) and LightMem (search_locomo.py).
  Key constraint: "The answer should be less than 5-6 words."
- ACCURACY_PROMPT: LoCoMo binary judge (CORRECT/WRONG), identical copy in
  Nemori evaluation/locomo/metrics/llm_judge.py and LightMem
  experiments/locomo/llm_judge.py. Curly quotes preserved verbatim.
- LongMemEval judge templates: official get_anscheck_prompt from
  benchmarks/longmemeval/src/evaluation/evaluate_qa.py (ICLR 2025 repo).
"""

# =============================================================================
# Unified answerer (LoCoMo / LongMemEval) — Mem0 original, short-answer
# =============================================================================

ANSWER_PROMPT = """    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

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

    Memories:
    {memories}

    Question: {question}

    Wrap your final answer in <answer></answer> tags, e.g., <answer>A shell necklace</answer>.

    Answer:"""

# NOTE: the <answer></answer> output-format line is OUR single addition to the
# otherwise verbatim Mem0 prompt. It constrains only formatting (deterministic
# extraction), not content; all systems under comparison use the identical
# prompt, so internal comparability is unaffected. Documented in the paper's
# experimental setup.


# =============================================================================
# LoCoMo judge — binary CORRECT/WRONG (Mem0/Nemori/LightMem identical copy)
# =============================================================================

JUDGE_SYSTEM_LOCOMO = (
    "You are an expert grader that determines if answers to questions match a "
    "gold standard answer."
)

ACCURACY_PROMPT = """Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
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

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


# =============================================================================
# LongMemEval judge — official per-type templates (evaluate_qa.py verbatim)
# =============================================================================

LME_TEMPLATE_DEFAULT = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

LME_TEMPLATE_TEMPORAL = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

LME_TEMPLATE_KNOWLEDGE_UPDATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

LME_TEMPLATE_PREFERENCE = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

LME_TEMPLATE_ABSTENTION = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    """Official LongMemEval judge prompt dispatch (evaluate_qa.py logic, verbatim)."""
    if abstention:
        return LME_TEMPLATE_ABSTENTION.format(question, answer, response)
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        return LME_TEMPLATE_DEFAULT.format(question, answer, response)
    if task == "temporal-reasoning":
        return LME_TEMPLATE_TEMPORAL.format(question, answer, response)
    if task == "knowledge-update":
        return LME_TEMPLATE_KNOWLEDGE_UPDATE.format(question, answer, response)
    if task == "single-session-preference":
        return LME_TEMPLATE_PREFERENCE.format(question, answer, response)
    raise NotImplementedError(f"Unknown LongMemEval task type: {task}")
