import pytest

import src.evaluation.judges as judges
from src.evaluation.judges import _parse_label, _parse_yes_no_strict


def _chat_usage(prompt_tokens, requested, response, response_id):
    detail = {
        "request_attempt": 1,
        "status": "accepted",
        "failure_type": None,
        "error_type": None,
        "error_message": None,
        "requested_model": requested,
        "response_model": response,
        "response_id": response_id,
        "finish_reason": "stop",
        "refusal": None,
        "choice_count": 1,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 1,
    }
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 1,
        "request_attempts": 1,
        "failed_request_attempts": 0,
        "unknown_token_attempts": 0,
        "request_attempt_details": [detail],
        "requested_model": requested,
        "response_model": response,
        "response_id": response_id,
        "finish_reason": "stop",
        "refusal": None,
        "choice_count": 1,
    }


def test_locomo_judge_parser_rejects_missing_verdict():
    with pytest.raises(ValueError, match="unparseable LoCoMo"):
        _parse_label("I cannot decide")
    with pytest.raises(ValueError, match="unparseable LoCoMo"):
        _parse_label('{"label":"CORRECTNESS"}')
    with pytest.raises(ValueError, match="unparseable LoCoMo"):
        _parse_label('{"label":"CORRECT-or-WRONG"}')
    with pytest.raises(ValueError, match="unparseable LoCoMo"):
        _parse_label('{"label":"UNKNOWN","explanation":"CORRECT"}')
    with pytest.raises(ValueError, match="unparseable LoCoMo"):
        _parse_label('{"label":"correct"}')
    with pytest.raises(ValueError, match="unparseable LoCoMo"):
        _parse_label('prefix {"label":"CORRECT"}')
    with pytest.raises(ValueError, match="unparseable LoCoMo"):
        _parse_label('"CORRECT"')


def test_longmemeval_judge_parser_rejects_missing_verdict():
    with pytest.raises(ValueError, match="unparseable LongMemEval"):
        _parse_yes_no_strict("uncertain")
    with pytest.raises(ValueError, match="unparseable LongMemEval"):
        _parse_yes_no_strict("Yes or no cannot be determined")
    with pytest.raises(ValueError, match="unparseable LongMemEval"):
        _parse_yes_no_strict("not yes")
    with pytest.raises(ValueError, match="unparseable LongMemEval"):
        _parse_yes_no_strict("Answer: yes")
    with pytest.raises(ValueError, match="unparseable LongMemEval"):
        _parse_yes_no_strict("No, it is not correct")


def test_judge_parsers_accept_expected_labels():
    assert _parse_label('{"label":"CORRECT"}') == 1
    assert _parse_label('{"label":"WRONG"}') == 0
    assert _parse_yes_no_strict("yes") == 1
    assert _parse_yes_no_strict("Yes.") == 1
    assert _parse_yes_no_strict("\n  No.  \nExplanation follows") == 0


def test_longmemeval_judge_retries_unparseable_output(monkeypatch):
    outputs = iter([
        ("uncertain", _chat_usage(2, "requested-1", "response-1", "id-1")),
        ("yes", _chat_usage(3, "requested-2", "response-2", "id-2")),
    ])
    monkeypatch.setattr(judges, "judge_chat", lambda _messages: next(outputs))

    score, raw, usage = judges.judge_longmemeval(
        "multi-session", "question", "gold", "answer"
    )

    assert score == 1 and raw == "yes"
    assert usage == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "parse_attempts": 2,
        "logical_judge_calls": 2,
        "request_attempts": 2,
        "physical_http_attempts": 2,
        "failed_request_attempts": 0,
        "unknown_token_attempts": 0,
        "request_attempt_details": [
            {**_chat_usage(2, "requested-1", "response-1", "id-1")[
                "request_attempt_details"
            ][0], "parse_attempt": 1, "logical_judge_call": 1},
            {**_chat_usage(3, "requested-2", "response-2", "id-2")[
                "request_attempt_details"
            ][0], "parse_attempt": 2, "logical_judge_call": 2},
        ],
        "requested_models": ["requested-1", "requested-2"],
        "response_models": ["response-1", "response-2"],
        "response_ids": ["id-1", "id-2"],
        "finish_reasons": ["stop", "stop"],
        "refusals": [None, None],
        "choice_counts": [1, 1],
    }


def test_parse_retry_exhaustion_preserves_all_usage(monkeypatch):
    outputs = iter([
        ("uncertain", _chat_usage(2, "requested", "response", f"id-{index}"))
        for index in range(3)
    ])
    monkeypatch.setattr(judges, "judge_chat", lambda _messages: next(outputs))

    with pytest.raises(judges.JudgeCallError) as captured:
        judges.judge_longmemeval(
            "multi-session", "question", "gold", "answer"
        )

    usage = captured.value.usage
    assert usage["parse_attempts"] == 3
    assert usage["logical_judge_calls"] == 3
    assert usage["physical_http_attempts"] == 3
    assert usage["response_ids"] == ["id-0", "id-1", "id-2"]


def test_cat5_abstention_prompt_contains_no_distractor_sentinel(monkeypatch):
    distractor = "DISTRACTOR_SENTINEL_MUST_NOT_REACH_JUDGE"
    captured = []

    def fake_chat(messages):
        captured.extend(messages)
        assert distractor not in repr(messages)
        return (
            '{"label":"CORRECT"}',
            _chat_usage(3, "requested", "response", "cat5-response"),
        )

    monkeypatch.setattr(judges, "judge_chat", fake_chat)
    score, _, _ = judges.judge_locomo_abstention(
        "What happened?",
        "Not mentioned in the conversation",
        "The conversation does not say.",
    )
    assert score == 1
    assert captured


def test_cat5_explicit_answer_uses_regular_locomo_judge(monkeypatch):
    captured = []

    def fake_chat(messages):
        captured.extend(messages)
        return (
            '{"label":"CORRECT"}',
            _chat_usage(3, "requested", "response", "cat5-explicit-response"),
        )

    monkeypatch.setattr(judges, "judge_chat", fake_chat)
    score, _, _ = judges.judge_locomo_cat5(
        "Did Caroline make the bowl?", "No", "No", abstention=False
    )

    assert score == 1
    assert captured
    assert "Gold answer: No" in captured[-1]["content"]
    assert "abstention evaluator" not in repr(captured)
