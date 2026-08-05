"""LLM judges for the unified evaluation protocol (all 0/1 binary).

- judge_locomo: Mem0/Nemori/LightMem ACCURACY_PROMPT, JSON {"label": ...}
  parsed to 1 (CORRECT) / 0 (WRONG). cat 3 gold uses only text before ';'
  (LoCoMo convention, applied by the caller or here via preprocess).
- judge_longmemeval: official per-type anscheck templates; strict yes/no
  parsing (whole-word match on the first line, not the official substring
  'yes' in response — see protocol §6.4).
"""

import json
import re

from .prompts import (ACCURACY_PROMPT, JUDGE_SYSTEM_LOCOMO,
                      get_anscheck_prompt)
from .llm_clients import LLMCallError, emit_attempt_event, judge_chat


class JudgeCallError(RuntimeError):
    """A judge failure carrying all completed and failed provider attempts."""

    def __init__(self, message, usage):
        super().__init__(message)
        self.usage = usage


def preprocess_locomo_gold(gold, category):
    """cat 3 (open-domain): official convention keeps only text before ';'."""
    gold = str(gold)
    if category == 3 and ";" in gold:
        return gold.split(";")[0].strip()
    return gold


def _parse_label(text):
    """Parse an exact CORRECT/WRONG label from a JSON object."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            f"unparseable LoCoMo judge response: {str(text)[:200]!r}"
        ) from exc
    if isinstance(obj, dict) and obj.get("label") in ("CORRECT", "WRONG"):
        return 1 if obj["label"] == "CORRECT" else 0
    raise ValueError(f"unparseable LoCoMo judge response: {text[:200]!r}")


def _new_usage_total():
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "parse_attempts": 0,
        "logical_judge_calls": 0,
        "request_attempts": 0,
        "physical_http_attempts": 0,
        "failed_request_attempts": 0,
        "unknown_token_attempts": 0,
        "request_attempt_details": [],
        "requested_models": [],
        "response_models": [],
        "response_ids": [],
        "finish_reasons": [],
        "refusals": [],
        "choice_counts": [],
    }


def _accumulate_usage(usage_total, usage):
    logical_call = usage_total["logical_judge_calls"] + 1
    usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
    usage_total["completion_tokens"] += usage.get("completion_tokens", 0) or 0
    request_attempts = usage.get("request_attempts", 0)
    usage_total["request_attempts"] += request_attempts
    usage_total["physical_http_attempts"] += usage.get(
        "physical_http_attempts", request_attempts
    )
    usage_total["failed_request_attempts"] += usage.get(
        "failed_request_attempts", 0
    )
    usage_total["unknown_token_attempts"] += usage.get(
        "unknown_token_attempts", 0
    )
    details = usage.get("request_attempt_details", [])
    for detail in details:
        recorded = dict(detail)
        recorded["parse_attempt"] = logical_call
        recorded["logical_judge_call"] = logical_call
        usage_total["request_attempt_details"].append(recorded)
    usage_total["logical_judge_calls"] = logical_call
    usage_total["parse_attempts"] += 1
    for source, target in (
        ("requested_model", "requested_models"),
        ("response_model", "response_models"),
        ("response_id", "response_ids"),
        ("finish_reason", "finish_reasons"),
        ("refusal", "refusals"),
        ("choice_count", "choice_counts"),
    ):
        usage_total[target].append(usage.get(source))


def _raise_with_failed_usage(usage_total, exc):
    failed = exc.usage
    logical_call = usage_total["logical_judge_calls"] + 1
    usage_total["prompt_tokens"] += failed.get("prompt_tokens", 0) or 0
    usage_total["completion_tokens"] += failed.get("completion_tokens", 0) or 0
    for key in ("request_attempts", "failed_request_attempts", "unknown_token_attempts"):
        usage_total[key] += failed.get(key, 0) or 0
    usage_total["physical_http_attempts"] += failed.get(
        "physical_http_attempts", failed.get("request_attempts", 0)
    ) or 0
    for detail in failed.get("request_attempt_details", []):
        recorded = dict(detail)
        recorded["parse_attempt"] = logical_call
        recorded["logical_judge_call"] = logical_call
        usage_total["request_attempt_details"].append(recorded)
    usage_total["logical_judge_calls"] = logical_call
    emit_attempt_event({
        "event": "logical_judge_call",
        "logical_judge_call": logical_call,
        "status": "http_failed",
        "usage": failed,
    })
    raise JudgeCallError(str(exc), usage_total) from exc


def _parse_or_record(parser, text, usage_total):
    logical_call = usage_total["logical_judge_calls"]
    try:
        score = parser(text)
    except ValueError as exc:
        emit_attempt_event({
            "event": "logical_judge_call",
            "logical_judge_call": logical_call,
            "status": "parse_rejected",
            "response_id": usage_total["response_ids"][-1],
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
        })
        raise
    emit_attempt_event({
        "event": "logical_judge_call",
        "logical_judge_call": logical_call,
        "status": "parse_accepted",
        "response_id": usage_total["response_ids"][-1],
        "score": score,
    })
    return score


def judge_locomo(question, gold, answer, category=None):
    """Binary LoCoMo judge. Returns (score, raw_judge_text, usage)."""
    if category is not None:
        gold = preprocess_locomo_gold(gold, category)
    prompt = ACCURACY_PROMPT.format(
        question=question, gold_answer=gold, generated_answer=answer)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_LOCOMO},
        {"role": "user", "content": prompt},
    ]
    usage_total = _new_usage_total()
    last_error = None
    for _ in range(3):
        try:
            text, usage = judge_chat(messages)
        except LLMCallError as exc:
            _raise_with_failed_usage(usage_total, exc)
        _accumulate_usage(usage_total, usage)
        try:
            return _parse_or_record(_parse_label, text, usage_total), text, usage_total
        except ValueError as exc:
            last_error = exc
    raise JudgeCallError(str(last_error), usage_total) from last_error


def judge_locomo_abstention(question, gold, answer):
    """Judge LoCoMo category-5 abstention without exposing its distractor."""
    prompt = (
        "Evaluate whether the generated answer correctly abstains because the "
        "conversation does not contain the requested information. Mark CORRECT "
        "only if it declines to provide an unsupported answer and does not assert "
        "a specific answer as fact. Return exactly a JSON object with one field: "
        '{"label": "CORRECT"} or {"label": "WRONG"}.\n\n'
        f"Question: {question}\n"
        f"Canonical reference: {gold}\n"
        f"Generated answer: {answer}"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an abstention evaluator. Judge only whether the answer "
                "appropriately recognizes missing conversational evidence."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    usage_total = _new_usage_total()
    last_error = None
    for _ in range(3):
        try:
            text, usage = judge_chat(messages)
        except LLMCallError as exc:
            _raise_with_failed_usage(usage_total, exc)
        _accumulate_usage(usage_total, usage)
        try:
            return _parse_or_record(_parse_label, text, usage_total), text, usage_total
        except ValueError as exc:
            last_error = exc
    raise JudgeCallError(str(last_error), usage_total) from last_error


def judge_locomo_cat5(question, gold, answer, *, abstention):
    """Use abstention semantics only for category-5 records without an answer."""
    if abstention:
        return judge_locomo_abstention(question, gold, answer)
    return judge_locomo(question, gold, answer, category=5)


def judge_beam(question, gold, answer, *, rubric=None, abstention=False):
    """BEAM judge. Returns (score, raw_judge_text, usage).

    BEAM ships a rubric with each question — the specific facts an answer has
    to contain. Where one exists it is the standard, because the gold text is
    one phrasing of many that satisfy it. Abstention questions are judged the
    way LoCoMo's are: declining is right, asserting an unsupported fact is not.
    """
    if abstention:
        return judge_locomo_abstention(question, gold, answer)
    checklist = "\n".join(f"- {item}" for item in (rubric or []) if str(item).strip())
    prompt = (
        "Judge whether the generated answer is CORRECT or WRONG for the "
        "question. Judge meaning, not wording: a different phrasing, order or "
        "level of detail is fine, and equivalent date formats are equal "
        "('2024-03-15' = '15 March 2024'). Mark WRONG when a required fact is "
        "missing, contradicted, or replaced by a different value. Return "
        'exactly a JSON object with one field: {"label": "CORRECT"} or '
        '{"label": "WRONG"}.\n\n'
        f"Question: {question}\n"
        f"Reference answer: {gold}\n"
        + (f"Required by the rubric:\n{checklist}\n" if checklist else "")
        + f"Generated answer: {answer}"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an evaluator for long-conversation memory. Judge only "
                "whether the answer conveys what the reference and rubric "
                "require."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    usage_total = _new_usage_total()
    last_error = None
    for _ in range(3):
        try:
            text, usage = judge_chat(messages)
        except LLMCallError as exc:
            _raise_with_failed_usage(usage_total, exc)
        _accumulate_usage(usage_total, usage)
        try:
            return _parse_or_record(_parse_label, text, usage_total), text, usage_total
        except ValueError as exc:
            last_error = exc
    raise JudgeCallError(str(last_error), usage_total) from last_error


def _parse_yes_no_strict(text):
    """Parse an exact yes/no verdict from the first non-empty line."""
    first_line = next(
        (line.strip() for line in text.splitlines() if line.strip()), ""
    )
    match = re.fullmatch(r"(yes|no)\.?", first_line, re.IGNORECASE)
    if match is None:
        raise ValueError(f"unparseable LongMemEval judge response: {text[:200]!r}")
    return 1 if match.group(1).lower() == "yes" else 0


def judge_longmemeval(question_type, question, gold, answer, abstention=False):
    """Official LongMemEval judge. Returns (score, raw_judge_text, usage)."""
    prompt = get_anscheck_prompt(
        question_type, question, gold, answer, abstention=abstention)
    messages = [{"role": "user", "content": prompt}]
    usage_total = _new_usage_total()
    last_error = None
    for _ in range(3):
        try:
            text, usage = judge_chat(messages)
        except LLMCallError as exc:
            _raise_with_failed_usage(usage_total, exc)
        _accumulate_usage(usage_total, usage)
        try:
            return (
                _parse_or_record(_parse_yes_no_strict, text, usage_total),
                text,
                usage_total,
            )
        except ValueError as exc:
            last_error = exc
    raise JudgeCallError(str(last_error), usage_total) from last_error
