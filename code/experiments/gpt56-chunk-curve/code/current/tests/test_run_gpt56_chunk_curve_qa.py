from __future__ import annotations

import copy
import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest


class _HealthySubscriptionUpstream(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        payload = json.dumps(
            {
                "status": "ok",
                "auth_readable": True,
                "max_concurrency": 2,
                "max_attempts": 2,
                "connect_timeout_s": 30,
                "read_timeout_s": 300,
                "requested_reasoning_effort": "none",
                "code_sha256": "a" * 64,
                "request_log": "/private/upstream.jsonl",
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def test_default_cli_is_plan_only(monkeypatch, capsys, tmp_path: Path) -> None:
    from scripts import run_gpt56_chunk_curve_qa as runner

    plan = {
        "protocol_id": "gpt56-w32-screening-qa-v1",
        "run_ids": ["run-1"],
        "question_count": 3,
    }
    monkeypatch.setattr(runner, "prepare_plan", lambda args: plan)

    assert runner.main(
        [
            "--build-root",
            str(tmp_path / "builds"),
            "--output-dir",
            str(tmp_path / "qa"),
            "--run-ids",
            "run-1",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "planned QA: 1 runs / 3 questions" in output
    assert "model requests sent: 0" in output
    assert not (tmp_path / "qa").exists()


def test_plan_and_execution_scope_freeze_the_same_upstream(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import run_gpt56_chunk_curve_qa as runner

    binding = contract.RunBinding("run-1", {"run_id": "run-1"}, tmp_path)
    validated = {"unit": {"run_id": "run-1"}}
    monkeypatch.setattr(
        runner.contract,
        "select_completed_runs",
        lambda **kwargs: [binding],
    )
    monkeypatch.setattr(
        runner.contract,
        "validate_unit_binding",
        lambda value: validated,
    )
    calls = []

    def fake_preregistration(**kwargs):
        calls.append(kwargs)
        return {
            "subscription_upstream": {
                "origin": kwargs.get("upstream"),
                "code_sha256": kwargs.get("upstream_code_sha256"),
            }
        }

    monkeypatch.setattr(
        runner.contract, "build_preregistration", fake_preregistration
    )
    args = runner.parse_args(
        [
            "--build-root",
            str(tmp_path / "builds"),
            "--output-dir",
            str(tmp_path / "qa"),
            "--run-ids",
            "run-1",
            "--upstream",
            "http://127.0.0.1:8205",
            "--upstream-code-sha256",
            "a" * 64,
        ]
    )

    assert runner.prepare_plan(args) == runner.prepare_scope(args)[2]
    assert len(calls) == 2
    assert all(call["upstream"] == "http://127.0.0.1:8205" for call in calls)
    assert all(call["upstream_code_sha256"] == "a" * 64 for call in calls)


def test_execute_cli_requires_frozen_upstream_hash(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts import run_gpt56_chunk_curve_qa as runner

    monkeypatch.setattr(
        runner,
        "prepare_plan",
        lambda args: {
            "protocol_id": "gpt56-w32-screening-qa-v1",
            "run_ids": ["run-1"],
            "question_count": 1,
        },
    )

    with pytest.raises(runner.QARunnerError, match="upstream code SHA-256"):
        runner.main(
            [
                "--build-root",
                str(tmp_path / "builds"),
                "--output-dir",
                str(tmp_path / "qa"),
                "--run-ids",
                "run-1",
                "--execute",
                "--allow-model-requests",
            ]
        )
    assert not (tmp_path / "qa").exists()


def test_formal_scope_requires_exact_six_w32_runs_and_356_questions() -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import run_gpt56_chunk_curve_qa as runner

    def scope(tier: str):
        unit_specs = [
            ("locomo", "conv-44", 123),
            ("locomo", "conv-48", 191),
            ("longmemeval-s", "2318644b", 1),
            ("longmemeval-s", "gpt4_6dc9b45b", 1),
            ("beam-100k", "100K-conv-1", 20),
            ("beam-100k", "100K-conv-2", 20),
        ]
        bindings = []
        units = []
        for benchmark, unit_id, count in unit_specs:
            run_id = runner.FORMAL_RUN_IDS_BY_TIER[tier][
                (benchmark, unit_id)
            ]
            row = {
                "run_id": run_id,
                "benchmark": benchmark,
                "tier": tier,
                "model": f"gpt-5.6-{tier}",
                "write_turns": 32,
                "reasoning_effort": "none",
                "unit_id": unit_id,
            }
            bindings.append(contract.RunBinding(run_id, row, Path(".")))
            units.append(
                {
                    "unit": {"run_id": run_id},
                    "questions": [
                        {"question_id": f"q{index}"}
                        for index in range(count)
                    ],
                }
            )
        return bindings, units

    bindings, units = scope("terra")
    runner.validate_formal_scope(bindings=bindings, validated_units=units)

    with pytest.raises(runner.QARunnerError, match="exactly six"):
        runner.validate_formal_scope(
            bindings=bindings[:-1],
            validated_units=units[:-1],
        )

    wrong_count = copy.deepcopy(units)
    wrong_count[0]["questions"].pop()
    with pytest.raises(runner.QARunnerError, match="356 questions"):
        runner.validate_formal_scope(
            bindings=bindings,
            validated_units=wrong_count,
        )


def test_execute_cli_checks_formal_scope_before_creating_output(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import run_gpt56_chunk_curve_qa as runner

    plan = {
        "protocol_id": contract.PROTOCOL_ID,
        "run_ids": ["run-1"],
        "question_count": 1,
        "qa": {"tokenizer": {}},
    }
    binding = contract.RunBinding(
        "run-1",
        {
            "run_id": "run-1",
            "benchmark": "locomo",
            "tier": "terra",
            "model": "gpt-5.6-terra",
            "write_turns": 32,
            "reasoning_effort": "none",
            "unit_id": "conv-44",
        },
        tmp_path,
    )
    validated = {
        "unit": {"run_id": "run-1"},
        "questions": [{"question_id": "q0"}],
    }
    monkeypatch.setattr(runner, "prepare_plan", lambda args: plan)
    monkeypatch.setattr(
        runner, "prepare_scope", lambda args: ([binding], [validated], plan)
    )
    output = tmp_path / "qa"

    with pytest.raises(runner.QARunnerError, match="exactly six"):
        runner.main(
            [
                "--build-root",
                str(tmp_path / "builds"),
                "--output-dir",
                str(output),
                "--run-ids",
                "run-1",
                "--upstream-code-sha256",
                "a" * 64,
                "--execute",
                "--allow-model-requests",
            ]
        )
    assert not output.exists()


def test_cli_accepts_preflight_only_mode(tmp_path: Path) -> None:
    from scripts import run_gpt56_chunk_curve_qa as runner

    args = runner.parse_args(
        [
            "--build-root",
            str(tmp_path / "builds"),
            "--output-dir",
            str(tmp_path / "qa"),
            "--run-ids",
            "run-1",
            "--preflight-only",
        ]
    )
    assert args.preflight_only is True


def test_execute_cli_validates_tokenizer_before_creating_output(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts import run_gpt56_chunk_curve_qa as runner

    plan = {
        "protocol_id": "gpt56-w32-screening-qa-v1",
        "run_ids": ["run-1"],
        "question_count": 1,
        "qa": {"tokenizer": {"implementation": "tiktoken"}},
    }
    monkeypatch.setattr(runner, "prepare_plan", lambda args: plan)
    monkeypatch.setattr(runner, "prepare_scope", lambda args: ([], [], plan))
    monkeypatch.setattr(runner, "validate_formal_scope", lambda **kwargs: None)
    monkeypatch.setattr(
        runner.answer_contract,
        "formal_token_counter",
        lambda: (_ for _ in ()).throw(RuntimeError("tokenizer unavailable")),
    )
    monkeypatch.setattr(
        runner,
        "start_qa_proxy",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("proxy started")),
    )
    output = tmp_path / "qa"

    with pytest.raises(RuntimeError, match="tokenizer unavailable"):
        runner.main(
            [
                "--build-root",
                str(tmp_path / "builds"),
                "--output-dir",
                str(output),
                "--run-ids",
                "run-1",
                "--upstream-code-sha256",
                "a" * 64,
                "--execute",
                "--allow-model-requests",
            ]
        )
    assert not output.exists()


def test_initialize_output_is_no_clobber(tmp_path: Path) -> None:
    from scripts import run_gpt56_chunk_curve_qa as runner

    output = tmp_path / "qa"
    plan = {
        "schema_version": 1,
        "protocol_id": "gpt56-w32-screening-qa-v1",
        "run_ids": ["run-1"],
    }

    path = runner.initialize_output(output, plan)
    assert path == output / "preregistration.json"
    assert json.loads(path.read_text(encoding="utf-8")) == plan

    with pytest.raises(FileExistsError):
        runner.initialize_output(output, plan)


def test_execute_short_answer_question_is_read_only(tmp_path: Path) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import readonly_nativemem_control as readonly
    from scripts import run_gpt56_chunk_curve_qa as runner
    from scripts.run_controlled_locomo_answers import FakeAnswerClient
    from src.evaluation.visible_token_budget import TokenCounter

    run_dir = tmp_path / "build"
    memory = run_dir / "memory"
    (memory / "topics").mkdir(parents=True)
    (memory / "timeline").mkdir()
    (memory / "topics" / "fact.md").write_text(
        "The answer is recorded here [D1:1].\n",
        encoding="utf-8",
    )
    row = {
        "run_id": "locomo-terra-w32",
        "benchmark": "locomo",
        "tier": "terra",
        "model": "gpt-5.6-terra",
        "write_turns": 32,
        "unit_id": "conv-test",
        "selection": {"sample_index": 5, "sample_id": "conv-test"},
    }
    binding = contract.RunBinding(
        run_id=row["run_id"],
        row=row,
        run_dir=run_dir,
    )
    question = {
        "question_id": "q0",
        "question": "What is the answer?",
        "gold": "recorded",
        "category": 1,
        "evidence": ["D1:1"],
    }
    validated = {
        "conversation": {
            "session_1": [
                {
                    "dia_id": "D1:1",
                    "speaker": "A",
                    "text": "The answer is recorded.",
                }
            ],
            "session_1_date_time": "2026-01-01",
        },
        "questions": [question],
        "metadata": {"sample_id": "conv-test"},
        "unit": {"run_id": row["run_id"]},
    }
    proxy_log = tmp_path / "proxy.jsonl"
    answer_client = FakeAnswerClient(
        proxy_log=proxy_log,
        run_id="qa-test",
        response_text="<answer>recorded</answer>",
    )
    completion = readonly.ScriptedCompletionResource(
        [
            [{"name": "read_memory_file", "arguments": {"path": "topics/fact.md"}}],
            [],
        ]
    )

    record = runner.execute_short_answer_question(
        binding=binding,
        validated_unit=validated,
        question=question,
        output_dir=tmp_path / "qa",
        qa_run_id="qa-test",
        completion_resource=completion,
        answer_client=answer_client,
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        proxy_log=proxy_log,
        formal=False,
    )

    assert record["answer"] == "recorded"
    assert record["question_id"] == "conv-test::q0"
    assert record["selection"] == {"sample_index": 5, "sample_id": "conv-test"}
    assert record["result"]["memory"]["unchanged"] is True
    assert record["result"]["diagnostics"]["mapped_source_recall"] == 1.0
    assert record["result"]["prompt"]["answer_completion_reservation_tokens"] == 10_000
    assert record["result"]["prompt"]["kind"] == "locomo-lme-short-answer-v1"
    assert "less than 5-6 words" in answer_client.prompts[0]


def test_longmemeval_question_date_is_visible_to_retrieval_and_answer(
    tmp_path: Path,
) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import readonly_nativemem_control as readonly
    from scripts import run_gpt56_chunk_curve_qa as runner
    from scripts.run_controlled_locomo_answers import FakeAnswerClient
    from src.evaluation.visible_token_budget import TokenCounter

    class RecordingCompletion(readonly.ScriptedCompletionResource):
        def __init__(self) -> None:
            super().__init__([[]])
            self.requests: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.requests.append(dict(kwargs))
            return super().create(**kwargs)

    run_dir = tmp_path / "build"
    memory = run_dir / "memory"
    (memory / "topics").mkdir(parents=True)
    (memory / "timeline").mkdir()
    (memory / "topics" / "fact.md").write_text(
        "The event happened in June 2021 [D1:1].\n",
        encoding="utf-8",
    )
    row = {
        "run_id": "lme-terra-w32",
        "benchmark": "longmemeval-s",
        "tier": "terra",
        "model": "gpt-5.6-terra",
        "write_turns": 32,
        "unit_id": "lme-temporal",
        "selection": {"dataset_index": 0, "question_id": "lme-temporal"},
    }
    binding = contract.RunBinding(
        run_id=row["run_id"],
        row=row,
        run_dir=run_dir,
    )
    question = {
        "question_id": "lme-temporal",
        "question": "How many months ago did the event happen?",
        "question_date": "2021/10/02 (Sat) 00:00",
        "gold": "4 months ago",
        "question_type": "temporal-reasoning",
    }
    validated = {
        "conversation": {
            "session_1": [
                {
                    "dia_id": "D1:1",
                    "speaker": "A",
                    "text": "The event happened in June 2021.",
                }
            ],
            "session_1_date_time": "2021/06/02 (Wed) 00:00",
        },
        "questions": [question],
        "metadata": {"question_id": "lme-temporal"},
        "unit": {"run_id": row["run_id"]},
    }
    proxy_log = tmp_path / "proxy.jsonl"
    answer_client = FakeAnswerClient(
        proxy_log=proxy_log,
        run_id="qa-test",
        response_text="<answer>4 months ago</answer>",
    )
    completion = RecordingCompletion()

    record = runner.execute_short_answer_question(
        binding=binding,
        validated_unit=validated,
        question=question,
        output_dir=tmp_path / "qa",
        qa_run_id="qa-test",
        completion_resource=completion,
        answer_client=answer_client,
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        proxy_log=proxy_log,
        formal=False,
    )

    model_question = (
        "Current Date: 2021/10/02 (Sat) 00:00\n"
        "Question: How many months ago did the event happen?"
    )
    retrieval_prompt = completion.requests[0]["messages"][0]["content"]
    assert model_question in retrieval_prompt
    assert model_question in answer_client.prompts[0]
    assert record["question"] == "How many months ago did the event happen?"
    assert record["question_date"] == "2021/10/02 (Sat) 00:00"
    assert record["model_question"] == model_question


def test_publish_short_answer_evaluator_inputs(tmp_path: Path) -> None:
    from scripts import run_gpt56_chunk_curve_qa as runner

    records = [
        {
            "benchmark": "locomo",
            "tier": "terra",
            "selection": {"sample_index": 5, "sample_id": "conv-44"},
            "original_question_id": "q0",
            "question": "LoCoMo question",
            "gold": "gold",
            "category": 1,
            "answer": "answer",
        },
        {
            "benchmark": "longmemeval-s",
            "tier": "terra",
            "selection": {"dataset_index": 9, "question_id": "lme-9"},
            "original_question_id": "lme-9",
            "question": "LME question",
            "gold": "gold",
            "question_type": "knowledge-update",
            "answer": "answer",
        },
    ]

    outputs = runner.publish_short_answer_evaluator_inputs(
        records=records,
        output_dir=tmp_path / "qa",
        tier="terra",
    )

    locomo = json.loads(outputs["locomo"][0].read_text(encoding="utf-8"))
    assert locomo == [
        {
            "question_id": "q0",
            "question": "LoCoMo question",
            "gold": "gold",
            "category": 1,
            "answer": "answer",
        }
    ]
    hypotheses = [
        json.loads(line)
        for line in outputs["longmemeval-s"].read_text(encoding="utf-8").splitlines()
    ]
    assert hypotheses == [{"question_id": "lme-9", "hypothesis": "answer"}]
    assert runner.publish_short_answer_evaluator_inputs(
        records=records,
        output_dir=tmp_path / "qa",
        tier="terra",
    ) == outputs


def test_publish_beam_predictions(tmp_path: Path) -> None:
    from scripts import run_gpt56_chunk_curve_qa as runner

    records = [
        {
            "benchmark": "beam-100k",
            "tier": "sol",
            "original_question_id": "1-q0",
            "answer": "first",
        },
        {
            "benchmark": "beam-100k",
            "tier": "sol",
            "original_question_id": "2-q0",
            "answer": "second",
        },
    ]

    path = runner.publish_beam_predictions(
        records=records,
        output_dir=tmp_path / "qa",
        tier="sol",
    )

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"question_id": "1-q0", "answer": "first"},
        {"question_id": "2-q0", "answer": "second"},
    ]
    assert runner.publish_beam_predictions(
        records=records,
        output_dir=tmp_path / "qa",
        tier="sol",
    ) == path


def test_execute_beam_question_preserves_rubric_fields(tmp_path: Path) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import readonly_nativemem_control as readonly
    from scripts import run_gpt56_chunk_curve_qa as runner
    from scripts.run_controlled_locomo_answers import FakeAnswerClient
    from src.evaluation.visible_token_budget import TokenCounter

    run_dir = tmp_path / "build"
    memory = run_dir / "memory"
    (memory / "topics").mkdir(parents=True)
    (memory / "timeline").mkdir()
    (memory / "topics" / "fact.md").write_text(
        "The interface uses a dark theme [D1:1].\n",
        encoding="utf-8",
    )
    row = {
        "run_id": "beam-sol-w32",
        "benchmark": "beam-100k",
        "tier": "sol",
        "model": "gpt-5.6-sol",
        "write_turns": 32,
        "unit_id": "100K-conv-1",
        "selection": {"conversation_index": 0, "conversation_id": "1"},
    }
    binding = contract.RunBinding(row["run_id"], row, run_dir)
    question = {
        "question_id": "1-q0",
        "question": "Which theme is used?",
        "question_text": "Which theme is used?",
        "question_type": "information_extraction",
        "gold_field": "answer",
        "gold": "dark",
        "rubric_nuggets": ["states that the theme is dark"],
    }
    validated = {
        "conversation": {
            "session_1": [
                {
                    "dia_id": "D1:1",
                    "speaker": "A",
                    "text": "The interface uses a dark theme.",
                }
            ],
            "session_1_date_time": "2026-01-01",
        },
        "questions": [question],
        "metadata": {"conversation_id": "1"},
        "unit": {"run_id": row["run_id"]},
    }
    proxy_log = tmp_path / "proxy.jsonl"
    answer_client = FakeAnswerClient(
        proxy_log=proxy_log,
        run_id="qa-test",
        response_text="<answer>dark</answer>",
    )
    completion = readonly.ScriptedCompletionResource(
        [
            [{"name": "read_memory_file", "arguments": {"path": "topics/fact.md"}}],
            [],
        ]
    )

    record = runner.execute_beam_question(
        binding=binding,
        validated_unit=validated,
        question=question,
        output_dir=tmp_path / "qa",
        qa_run_id="qa-test",
        completion_resource=completion,
        answer_client=answer_client,
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        proxy_log=proxy_log,
        formal=False,
        attempt=2,
    )

    assert record["answer"] == "dark"
    assert record["original_question_id"] == "1-q0"
    assert record["question_type"] == "information_extraction"
    assert record["gold_field"] == "answer"
    assert record["rubric"] == ["states that the theme is dark"]
    assert record["result"]["memory"]["unchanged"] is True
    assert record["result"]["prompt"]["kind"] == "beam-r115-answer-v1"
    assert (
        record["result"]["prompt"]["template_sha256"]
        == contract.BEAM_ANSWER_PROMPT_SHA256
    )
    assert "less than 5-6 words" not in answer_client.prompts[0]
    assert "Do not impose a short-answer limit" in answer_client.prompts[0]
    assert "Follow every format" in answer_client.prompts[0]
    assert (
        "I don't have enough information to answer this question."
        in answer_client.prompts[0]
    )
    assert (
        tmp_path
        / "qa"
        / "sol"
        / "beam-100k"
        / "100K-conv-1"
        / "questions"
        / "100K-conv-1_1-q0"
        / "attempt-0002"
        / "result.json"
    ).is_file()


def test_execute_pending_questions_rejects_unapproved_incomplete_attempt(
    tmp_path: Path,
) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import readonly_nativemem_control as readonly
    from scripts import run_gpt56_chunk_curve_qa as runner
    from scripts.run_controlled_locomo_answers import FakeAnswerClient
    from src.evaluation.visible_token_budget import TokenCounter

    run_dir = tmp_path / "build"
    memory = run_dir / "memory"
    (memory / "topics").mkdir(parents=True)
    (memory / "timeline").mkdir()
    (memory / "topics" / "fact.md").write_text(
        "The answer is recorded here [D1:1].\n",
        encoding="utf-8",
    )
    row = {
        "run_id": "locomo-terra-w32",
        "benchmark": "locomo",
        "tier": "terra",
        "model": "gpt-5.6-terra",
        "write_turns": 32,
        "unit_id": "conv-test",
        "selection": {"sample_index": 5, "sample_id": "conv-test"},
    }
    binding = contract.RunBinding(row["run_id"], row, run_dir)
    question = {
        "question_id": "q0",
        "question": "What is the answer?",
        "gold": "recorded",
        "category": 1,
        "evidence": ["D1:1"],
    }
    validated = {
        "conversation": {
            "session_1": [
                {
                    "dia_id": "D1:1",
                    "speaker": "A",
                    "text": "The answer is recorded.",
                }
            ],
            "session_1_date_time": "2026-01-01",
        },
        "questions": [question],
        "metadata": {"sample_id": "conv-test"},
        "unit": {"run_id": row["run_id"]},
    }
    output = tmp_path / "qa"
    question_root = (
        output
        / "terra"
        / "locomo"
        / "conv-test"
        / "questions"
        / "conv-test_q0"
    )
    interrupted = question_root / "attempt-0001"
    interrupted.mkdir(parents=True)
    (interrupted / "interrupted.txt").write_text("preserve", encoding="utf-8")
    proxy_log = tmp_path / "proxy.jsonl"
    answer_client = FakeAnswerClient(
        proxy_log=proxy_log,
        run_id="qa-test",
        response_text="<answer>recorded</answer>",
    )
    completion = readonly.ScriptedCompletionResource(
        [
            [{"name": "read_memory_file", "arguments": {"path": "topics/fact.md"}}],
            [],
        ]
    )

    with pytest.raises(runner.QARunnerError, match="retry authorization"):
        runner.execute_pending_questions(
            bindings=[binding],
            validated_units=[validated],
            output_dir=output,
            qa_run_id="qa-test",
            completion_resource=completion,
            answer_client=answer_client,
            tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
            proxy_log=proxy_log,
            formal=False,
        )
    (interrupted / "interrupted.txt").unlink()
    interrupted.rmdir()

    records = runner.execute_pending_questions(
        bindings=[binding],
        validated_units=[validated],
        output_dir=output,
        qa_run_id="qa-test",
        completion_resource=completion,
        answer_client=answer_client,
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        proxy_log=proxy_log,
        formal=False,
    )

    assert len(records) == 1
    assert records[0]["answer"] == "recorded"
    assert (question_root / "attempt-0001" / "result.json").is_file()
    assert records[0]["result"]["run_id"] == "qa-test:attempt-0001"
    manifest = json.loads(
        (question_root / "attempt-0001" / "attempt_manifest.json").read_text()
    )
    assert manifest == {
        "schema_version": 1,
        "qa_run_id": "qa-test",
        "execution_run_id": "qa-test:attempt-0001",
        "attempt": 1,
        "question_id": "conv-test::q0",
        "benchmark": "locomo",
        "answer_prompt_kind": "locomo-lme-short-answer-v1",
        "answer_prompt_template_sha256": (
            contract.SHORT_ANSWER_PROMPT_SHA256
        ),
    }
    proxy_events = [json.loads(line) for line in proxy_log.read_text().splitlines()]
    assert all(
        event["logical_call_id"].startswith("qa-test:attempt-0001:")
        for event in proxy_events
    )
    assert (question_root / "record.json").is_file()

    manifest_path = question_root / "attempt-0001/attempt_manifest.json"
    wrong_manifest = copy.deepcopy(manifest)
    wrong_manifest["question_id"] = "different::q0"
    manifest_path.write_text(json.dumps(wrong_manifest), encoding="utf-8")
    with pytest.raises(runner.QARunnerError, match="attempt manifest contract"):
        runner.execute_pending_questions(
            bindings=[binding],
            validated_units=[validated],
            output_dir=output,
            qa_run_id="qa-test",
            completion_resource=object(),
            answer_client=object(),
            tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
            proxy_log=proxy_log,
            formal=False,
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    (question_root / "record.json").unlink()
    result_path = question_root / "attempt-0001/result.json"
    original_result = json.loads(result_path.read_text())
    wrong_result = copy.deepcopy(original_result)
    wrong_result["run_id"] = "qa-test:attempt-0002"
    result_path.write_text(json.dumps(wrong_result), encoding="utf-8")
    with pytest.raises(runner.QARunnerError, match="attempt identity"):
        runner.execute_pending_questions(
            bindings=[binding],
            validated_units=[validated],
            output_dir=output,
            qa_run_id="qa-test",
            completion_resource=object(),
            answer_client=object(),
            tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
            proxy_log=proxy_log,
            formal=False,
        )
    result_path.write_text(json.dumps(original_result), encoding="utf-8")
    recovered_record = runner.execute_pending_questions(
        bindings=[binding],
        validated_units=[validated],
        output_dir=output,
        qa_run_id="qa-test",
        completion_resource=object(),
        answer_client=object(),
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        proxy_log=proxy_log,
        formal=False,
    )
    assert recovered_record == records
    assert (question_root / "record.json").is_file()
    assert not (question_root / "attempt-0002").exists()

    resumed = runner.execute_pending_questions(
        bindings=[binding],
        validated_units=[validated],
        output_dir=output,
        qa_run_id="qa-test",
        completion_resource=object(),
        answer_client=object(),
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        proxy_log=proxy_log,
        formal=False,
    )
    assert resumed == records
    assert not (question_root / "attempt-0002").exists()

    record_path = question_root / "record.json"
    record_path.unlink()
    accepted_attempt = question_root / "attempt-0001"
    over_limit_attempt = question_root / "attempt-0009"
    accepted_attempt.rename(over_limit_attempt)
    with pytest.raises(runner.QARunnerError, match="attempt limit"):
        runner.execute_pending_questions(
            bindings=[binding],
            validated_units=[validated],
            output_dir=output,
            qa_run_id="qa-test",
            completion_resource=object(),
            answer_client=object(),
            tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
            proxy_log=proxy_log,
            formal=False,
        )
    over_limit_attempt.rename(accepted_attempt)
    restored = runner.execute_pending_questions(
        bindings=[binding],
        validated_units=[validated],
        output_dir=output,
        qa_run_id="qa-test",
        completion_resource=object(),
        answer_client=object(),
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        proxy_log=proxy_log,
        formal=False,
    )
    assert restored == records

    tampered = json.loads(record_path.read_text())
    tampered["question"] = "changed question"
    record_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(runner.QARunnerError, match="question payload differs"):
        runner.execute_pending_questions(
            bindings=[binding],
            validated_units=[validated],
            output_dir=output,
            qa_run_id="qa-test",
            completion_resource=object(),
            answer_client=object(),
            tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
            proxy_log=proxy_log,
            formal=False,
        )

    tampered["question"] = question["question"]
    tampered["result"]["answer"]["response_model"] = "changed-model"
    record_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(runner.QARunnerError, match="result contract differs"):
        runner.execute_pending_questions(
            bindings=[binding],
            validated_units=[validated],
            output_dir=output,
            qa_run_id="qa-test",
            completion_resource=object(),
            answer_client=object(),
            tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
            proxy_log=proxy_log,
            formal=False,
        )


def test_resume_validates_every_prior_attempt_before_execution(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import run_gpt56_chunk_curve_qa as runner
    from src.evaluation.visible_token_budget import TokenCounter

    run_dir = tmp_path / "build"
    (run_dir / "memory/topics").mkdir(parents=True)
    (run_dir / "memory/timeline").mkdir()
    row = {
        "run_id": "locomo-terra-w32",
        "benchmark": "locomo",
        "tier": "terra",
        "model": "gpt-5.6-terra",
        "write_turns": 32,
        "unit_id": "conv-test",
        "selection": {"sample_index": 5, "sample_id": "conv-test"},
    }
    binding = contract.RunBinding(row["run_id"], row, run_dir)
    validated = {
        "conversation": {
            "session_1": [],
            "session_1_date_time": "2026-01-01",
        },
        "questions": [
            {
                "question_id": "q0",
                "question": "What is the answer?",
                "gold": "recorded",
                "category": 1,
                "evidence": [],
            }
        ],
        "metadata": {"sample_id": "conv-test"},
        "unit": {"run_id": row["run_id"]},
    }
    question_root = (
        tmp_path
        / "qa/terra/locomo/conv-test/questions/conv-test_q0"
    )
    (question_root / "attempt-0001").mkdir(parents=True)
    (question_root / "attempt-0002").mkdir()
    checked: list[int] = []

    def validate(**kwargs):
        checked.append(kwargs["attempt"])
        if kwargs["attempt"] == 1:
            raise runner.QARunnerError("prior retry authorization differs")
        return {"to_attempt": kwargs["attempt"] + 1}

    monkeypatch.setattr(runner, "_validate_retry_authorization", validate)
    monkeypatch.setattr(
        runner,
        "execute_short_answer_question",
        lambda **kwargs: pytest.fail("execution began before chain validation"),
    )

    with pytest.raises(runner.QARunnerError, match="prior retry authorization"):
        runner.execute_pending_questions(
            bindings=[binding],
            validated_units=[validated],
            output_dir=tmp_path / "qa",
            qa_run_id="qa-test",
            completion_resource=object(),
            answer_client=object(),
            tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
            proxy_log=None,
            formal=True,
        )
    assert checked == [1]


def test_execute_pending_questions_retries_transient_empty_output(
    tmp_path: Path,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import readonly_nativemem_control as readonly
    from scripts import run_gpt56_chunk_curve_qa as runner
    from scripts.run_controlled_locomo_answers import FakeAnswerClient
    from src.evaluation.visible_token_budget import TokenCounter

    class EmptyOutputError(RuntimeError):
        status_code = 500

    class OneTransientFailure:
        def __init__(self) -> None:
            self.calls = 0
            self.success = readonly.ScriptedCompletionResource([[]])

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise EmptyOutputError(
                    "upstream completed with empty output"
                )
            return self.success.create(**kwargs)

    run_dir = tmp_path / "build"
    memory = run_dir / "memory"
    (memory / "topics").mkdir(parents=True)
    (memory / "timeline").mkdir()
    (memory / "topics" / "fact.md").write_text(
        "The answer is recorded here [D1:1].\n",
        encoding="utf-8",
    )
    row = {
        "run_id": "locomo-sol-w32",
        "benchmark": "locomo",
        "tier": "sol",
        "model": "gpt-5.6-sol",
        "write_turns": 32,
        "unit_id": "conv-test",
        "selection": {"sample_index": 5, "sample_id": "conv-test"},
    }
    binding = contract.RunBinding(row["run_id"], row, run_dir)
    question = {
        "question_id": "q0",
        "question": "What is the answer?",
        "gold": "recorded",
        "category": 1,
        "evidence": ["D1:1"],
    }
    validated = {
        "conversation": {
            "session_1": [
                {
                    "dia_id": "D1:1",
                    "speaker": "A",
                    "text": "The answer is recorded.",
                }
            ],
            "session_1_date_time": "2026-01-01",
        },
        "questions": [question],
        "metadata": {"sample_id": "conv-test"},
        "unit": {"run_id": row["run_id"]},
    }
    output = tmp_path / "qa"
    proxy_log = tmp_path / "proxy.jsonl"
    completion = OneTransientFailure()

    records = runner.execute_pending_questions(
        bindings=[binding],
        validated_units=[validated],
        output_dir=output,
        qa_run_id="qa-test",
        completion_resource=completion,
        answer_client=FakeAnswerClient(
            proxy_log=proxy_log,
            run_id="qa-test",
            response_text="<answer>recorded</answer>",
        ),
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        proxy_log=proxy_log,
        formal=False,
    )

    question_root = (
        output
        / "sol"
        / "locomo"
        / "conv-test"
        / "questions"
        / "conv-test_q0"
    )
    assert completion.calls == 2
    assert (question_root / "attempt-0001" / "attempt_manifest.json").is_file()
    assert not (question_root / "attempt-0001" / "result.json").exists()
    assert (question_root / "attempt-0002" / "result.json").is_file()
    assert records[0]["result"]["run_id"] == "qa-test:attempt-0002"
    authorization = json.loads(
        (question_root / "attempt-0001" / "retry_authorization.json").read_text()
    )
    assert authorization["schema_version"] == 1
    assert authorization["qa_run_id"] == "qa-test"
    assert authorization["question_id"] == "conv-test::q0"
    assert authorization["from_attempt"] == 1
    assert authorization["to_attempt"] == 2
    assert authorization["reason"] == "upstream_completed_empty_output"
    assert authorization["failure"]["http_status"] == 500
    assert authorization["failure"]["upstream_http_attempts"] is None
    audited_authorization = auditor._audit_retry_authorization(
        attempt_root=question_root / "attempt-0001",
        manifest=json.loads(
            (question_root / "attempt-0001" / "attempt_manifest.json").read_text()
        ),
        formal=False,
    )
    assert audited_authorization == authorization
    manifests = auditor._attempt_manifests(
        qa_root=output,
        qa_run_id="qa-test",
    )
    chain_report = auditor._audit_attempt_chains(
        manifests=manifests,
        records=records,
        formal=False,
    )
    assert chain_report["accepted_attempts"] == 1
    assert chain_report["authorized_retries"] == 1

    authorization_path = question_root / "attempt-0001/retry_authorization.json"
    hidden_authorization = question_root / "attempt-0001/retry_authorization.hidden"
    authorization_path.rename(hidden_authorization)
    with pytest.raises(runner.QARunnerError, match="retry authorization"):
        runner.execute_pending_questions(
            bindings=[binding],
            validated_units=[validated],
            output_dir=output,
            qa_run_id="qa-test",
            completion_resource=object(),
            answer_client=object(),
            tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
            proxy_log=proxy_log,
            formal=False,
        )
    hidden_authorization.rename(authorization_path)

    with pytest.raises(runner.QARunnerError, match="formal retry proxy"):
        runner._retry_authorization_payload(
            attempt_root=question_root / "attempt-0001",
            qa_run_id="qa-test",
            question_id="conv-test::q0",
            attempt=1,
            exc=EmptyOutputError("upstream completed with empty output"),
            formal=True,
        )
    with pytest.raises(runner.QARunnerError, match="formal retry proxy"):
        runner._validate_retry_authorization(
            attempt_root=question_root / "attempt-0001",
            qa_run_id="qa-test",
            question_id="conv-test::q0",
            attempt=1,
            formal=True,
        )

    manifest_path = question_root / "attempt-0001" / "attempt_manifest.json"
    tampered_manifest = json.loads(manifest_path.read_text())
    tampered_manifest["execution_run_id"] = "qa-test:attempt-9999"
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    with pytest.raises(runner.QARunnerError, match="failure binding"):
        runner._retry_authorization_payload(
            attempt_root=question_root / "attempt-0001",
            qa_run_id="qa-test",
            question_id="conv-test::q0",
            attempt=1,
            exc=EmptyOutputError("upstream completed with empty output"),
            formal=False,
        )
    with pytest.raises(auditor.QAAuditError, match="failure binding"):
        auditor._audit_retry_authorization(
            attempt_root=question_root / "attempt-0001",
            manifest=tampered_manifest,
            formal=False,
        )


def test_retry_authorization_rejects_answer_ledger_without_empty_output_marker(
    tmp_path: Path,
) -> None:
    from scripts import controlled_locomo_answer_contract as answer_contract
    from scripts import run_gpt56_chunk_curve_qa as runner

    class EmptyOutputError(RuntimeError):
        status_code = 500

    attempt_root = tmp_path / "attempt-0001"
    attempt_root.mkdir()
    question_id = "conv-test::q0"
    manifest = {
        "schema_version": 1,
        "qa_run_id": "qa-test",
        "execution_run_id": "qa-test:attempt-0001",
        "attempt": 1,
        "question_id": question_id,
        "benchmark": "locomo",
    }
    (attempt_root / "attempt_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    logical_call_id = "qa-test:attempt-0001:conv-test::q0:dual_source:answer"
    with answer_contract.DurableLedger(
        attempt_root / "answer_ledger.jsonl",
        run_id=logical_call_id,
    ) as ledger:
        ledger.append(
            "physical_http_attempt_finished",
            {
                "status": "retryable_empty_output",
                "error": "unrelated failure",
                "http_status": 500,
                "upstream_http_attempts": None,
                "proxy_event_id": "event-1",
            },
        )

    with pytest.raises(runner.QARunnerError, match="exact empty-output evidence"):
        runner._retry_authorization_payload(
            attempt_root=attempt_root,
            qa_run_id="qa-test",
            question_id=question_id,
            attempt=1,
            exc=EmptyOutputError("upstream completed with empty output"),
            formal=False,
        )


def test_http_answer_empty_output_retries_as_a_new_question_attempt(
    tmp_path: Path,
) -> None:
    from scripts import controlled_locomo_answer_contract as answer_contract
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import readonly_nativemem_control as readonly
    from scripts import run_gpt56_chunk_curve_qa as runner
    from scripts.run_controlled_locomo_answers import HttpAnswerClient
    from src.evaluation.visible_token_budget import TokenCounter

    class AnswerUpstream(BaseHTTPRequestHandler):
        calls = 0

        def do_POST(self) -> None:  # noqa: N802
            type(self).calls += 1
            body = self.rfile.read(int(self.headers["Content-Length"]))
            question_id = self.headers["X-Controlled-Question-ID"]
            logical_call_id = self.headers["X-Controlled-Logical-Call-ID"]
            linkage = {
                "event_id": f"event-{type(self).calls}",
                "question_id": question_id,
                "logical_call_id": logical_call_id,
                "request_sha256": answer_contract.sha256_bytes(body),
                "client_http_attempts": 1,
                "upstream_http_attempts": (
                    None if type(self).calls == 1 else 1
                ),
                "unsupported_parameters": [],
            }
            if type(self).calls == 1:
                payload = {
                    "error": "upstream completed with empty output",
                    "exclusive_proxy_meta": linkage,
                }
                status = 500
            else:
                payload = {
                    "id": "response-2",
                    "model": "gpt-5.5",
                    "choices": [
                        {
                            "message": {"content": "<answer>recorded</answer>"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                    "proxy_meta": {
                        "attempts": 1,
                        "unsupported_parameters": [],
                    },
                    "exclusive_proxy_meta": linkage,
                }
                status = 200
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    run_dir = tmp_path / "build"
    memory = run_dir / "memory"
    (memory / "topics").mkdir(parents=True)
    (memory / "timeline").mkdir()
    (memory / "topics/fact.md").write_text(
        "The answer is recorded [D1:1].\n", encoding="utf-8"
    )
    row = {
        "run_id": "locomo-sol-w32",
        "benchmark": "locomo",
        "tier": "sol",
        "model": "gpt-5.6-sol",
        "write_turns": 32,
        "unit_id": "conv-test",
        "selection": {"sample_index": 5, "sample_id": "conv-test"},
    }
    binding = contract.RunBinding(row["run_id"], row, run_dir)
    validated = {
        "conversation": {
            "session_1": [
                {
                    "dia_id": "D1:1",
                    "speaker": "A",
                    "text": "The answer is recorded.",
                }
            ],
            "session_1_date_time": "2026-01-01",
        },
        "questions": [
            {
                "question_id": "q0",
                "question": "What is the answer?",
                "gold": "recorded",
                "category": 1,
                "evidence": ["D1:1"],
            }
        ],
        "metadata": {"sample_id": "conv-test"},
        "unit": {"run_id": row["run_id"]},
    }
    upstream = HTTPServer(("127.0.0.1", 0), AnswerUpstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        records = runner.execute_pending_questions(
            bindings=[binding],
            validated_units=[validated],
            output_dir=tmp_path / "qa",
            qa_run_id="qa-test",
            completion_resource=readonly.ScriptedCompletionResource([[], []]),
            answer_client=HttpAnswerClient(
                base_url=f"http://127.0.0.1:{upstream.server_port}/v1",
                retries=3,
                answer_max_tokens=4_096,
            ),
            tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
            proxy_log=None,
            formal=False,
        )
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    question_root = (
        tmp_path / "qa/sol/locomo/conv-test/questions/conv-test_q0"
    )
    authorization = json.loads(
        (question_root / "attempt-0001/retry_authorization.json").read_text()
    )
    first_events = answer_contract.audit_ledger(
        question_root / "attempt-0001/answer_ledger.jsonl"
    )
    assert AnswerUpstream.calls == 2
    assert authorization["failure"]["stage"] == "answer"
    assert any(
        event.get("event") == "physical_http_attempt_finished"
        and event.get("payload", {}).get("status") == "retryable_empty_output"
        for event in first_events
    )
    assert records[0]["result"]["run_id"] == "qa-test:attempt-0002"
    assert (question_root / "attempt-0002/result.json").is_file()


def test_start_and_stop_exclusive_qa_proxy(tmp_path: Path) -> None:
    from scripts import run_gpt56_chunk_curve_qa as runner

    upstream = HTTPServer(("127.0.0.1", 0), _HealthySubscriptionUpstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        process, record, process_log = runner.start_qa_proxy(
            output_dir=tmp_path / "qa",
            python=Path(sys.executable),
            upstream=f"http://127.0.0.1:{upstream.server_port}",
            upstream_code_sha256="a" * 64,
            run_id="qa-run",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        health = json.loads(
            opener.open(record["base_url"].removesuffix("/v1") + "/healthz").read()
        )
        assert health["status"] == "ok"
        assert Path(record["ready_path"]).is_file()
        stopped = runner.stop_qa_proxy(
            process=process,
            record=record,
            process_log=process_log,
        )
        assert stopped["returncode"] is not None
        assert Path(stopped["stop_path"]).is_file()
        assert process.poll() is not None
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)


def test_recover_stale_exclusive_qa_proxy(tmp_path: Path) -> None:
    from scripts import run_gpt56_chunk_curve_qa as runner

    upstream = HTTPServer(("127.0.0.1", 0), _HealthySubscriptionUpstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    process = process_log = None
    try:
        process, record, process_log = runner.start_qa_proxy(
            output_dir=tmp_path / "qa",
            python=Path(sys.executable),
            upstream=f"http://127.0.0.1:{upstream.server_port}",
            upstream_code_sha256="a" * 64,
            run_id="qa-stale",
        )

        recovered = runner.recover_stale_qa_proxies(
            output_dir=tmp_path / "qa",
            run_id="qa-stale",
        )

        process.wait(timeout=10)
        assert len(recovered) == 1
        assert recovered[0]["status"] == "terminated_stale_proxy"
        assert Path(recovered[0]["recovery_path"]).is_file()
        assert process.poll() is not None
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        if process_log is not None and not process_log.closed:
            process_log.close()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)


def test_run_execution_publishes_and_resumes_complete_scope(tmp_path: Path) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import readonly_nativemem_control as readonly
    from scripts import run_gpt56_chunk_curve_qa as runner
    from scripts.run_controlled_locomo_answers import FakeAnswerClient
    from src.evaluation.visible_token_budget import TokenCounter

    run_dir = tmp_path / "build"
    memory = run_dir / "memory"
    (memory / "topics").mkdir(parents=True)
    (memory / "timeline").mkdir()
    (memory / "topics" / "fact.md").write_text(
        "The answer is recorded here [D1:1].\n",
        encoding="utf-8",
    )
    row = {
        "run_id": "locomo-terra-w32",
        "benchmark": "locomo",
        "tier": "terra",
        "model": "gpt-5.6-terra",
        "write_turns": 32,
        "unit_id": "conv-test",
        "selection": {"sample_index": 5, "sample_id": "conv-test"},
    }
    binding = contract.RunBinding(row["run_id"], row, run_dir)
    question = {
        "question_id": "q0",
        "question": "What is the answer?",
        "gold": "recorded",
        "category": 1,
        "evidence": ["D1:1"],
    }
    validated = {
        "conversation": {
            "session_1": [
                {
                    "dia_id": "D1:1",
                    "speaker": "A",
                    "text": "The answer is recorded.",
                }
            ],
            "session_1_date_time": "2026-01-01",
        },
        "questions": [question],
        "metadata": {"sample_id": "conv-test"},
        "unit": {"run_id": row["run_id"]},
    }
    plan = {
        "schema_version": 1,
        "protocol_id": "gpt56-w32-screening-qa-v1",
        "run_ids": [row["run_id"]],
        "question_count": 1,
    }
    output = tmp_path / "qa"
    proxy_log = tmp_path / "proxy.jsonl"
    answer_client = FakeAnswerClient(
        proxy_log=proxy_log,
        run_id="qa-test",
        response_text="<answer>recorded</answer>",
    )
    completion = readonly.ScriptedCompletionResource(
        [
            [{"name": "read_memory_file", "arguments": {"path": "topics/fact.md"}}],
            [],
        ]
    )

    records = runner.run_execution(
        plan=plan,
        bindings=[binding],
        validated_units=[validated],
        output_dir=output,
        qa_run_id="qa-test",
        completion_resource=completion,
        answer_client=answer_client,
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        proxy_log=proxy_log,
        formal=False,
    )

    assert len(records) == 1
    assert (output / "preregistration.json").is_file()
    assert (output / "terra/evaluator_inputs/locomo/sample5_questions.json").is_file()
    completion_record = json.loads((output / "completion.json").read_text())
    assert completion_record["status"] == "complete"
    assert completion_record["question_count"] == 1

    resumed = runner.run_execution(
        plan=plan,
        bindings=[binding],
        validated_units=[validated],
        output_dir=output,
        qa_run_id="qa-test",
        completion_resource=object(),
        answer_client=object(),
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        proxy_log=proxy_log,
        formal=False,
    )
    assert resumed == records


def test_run_preflight_selects_one_question_per_tier_and_benchmark(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import run_gpt56_chunk_curve_qa as runner

    rows = [
        {
            "run_id": "terra-locomo",
            "benchmark": "locomo",
            "tier": "terra",
            "unit_id": "conv-a",
        },
        {
            "run_id": "terra-locomo-second",
            "benchmark": "locomo",
            "tier": "terra",
            "unit_id": "conv-b",
        },
        {
            "run_id": "sol-beam",
            "benchmark": "beam-100k",
            "tier": "sol",
            "unit_id": "beam-a",
        },
    ]
    bindings = [
        contract.RunBinding(row["run_id"], row, tmp_path / row["run_id"])
        for row in rows
    ]
    units = [
        {
            "unit": {"run_id": row["run_id"]},
            "conversation": {},
            "questions": [
                {"question_id": "q0", "question": "first"},
                {"question_id": "q1", "question": "second"},
            ],
        }
        for row in rows
    ]
    captured: dict[str, object] = {}

    def fake_execute_pending_questions(**kwargs):
        selected_bindings = kwargs["bindings"]
        selected_units = kwargs["validated_units"]
        captured["run_ids"] = [binding.run_id for binding in selected_bindings]
        captured["question_counts"] = [
            len(unit["questions"]) for unit in selected_units
        ]
        return [
            {
                "run_id": binding.run_id,
                "tier": binding.row["tier"],
                "benchmark": binding.row["benchmark"],
                "question_id": f"{binding.row['unit_id']}::q0",
                "original_question_id": "q0",
                "answer": "answer",
            }
            for binding in selected_bindings
        ]

    monkeypatch.setattr(
        runner, "execute_pending_questions", fake_execute_pending_questions
    )
    output = tmp_path / "qa"
    records = runner.run_preflight(
        plan={
            "schema_version": 1,
            "protocol_id": "gpt56-w32-screening-qa-v1",
            "run_ids": [row["run_id"] for row in rows],
            "question_count": 6,
        },
        bindings=bindings,
        validated_units=units,
        output_dir=output,
        qa_run_id="qa-preflight",
        completion_resource=object(),
        answer_client=object(),
        tokenizer=object(),
        proxy_log=None,
        formal=False,
    )

    assert captured["run_ids"] == ["terra-locomo", "sol-beam"]
    assert captured["question_counts"] == [1, 1]
    assert len(records) == 2
    preflight = json.loads((output / "preflight.json").read_text())
    assert preflight["status"] == "complete"
    assert preflight["question_count"] == 2
    assert not (output / "completion.json").exists()
    assert not list(output.rglob("evaluator_inputs"))
