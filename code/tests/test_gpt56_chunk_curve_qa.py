from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import pytest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _canonical_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _matrix_row(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "benchmark": "locomo",
        "tier": "terra",
        "model": "gpt-5.6-terra",
        "write_turns": 32,
        "reasoning_effort": "none",
        "unit_id": run_id,
        "output_dir": f"results/formal/gpt56-chunk-curve/locomo/{run_id}",
        "selection": {"sample_index": 0, "sample_id": "sample-0"},
    }


def test_completed_run_selection_rejects_incomplete_run(tmp_path: Path) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract

    complete = _matrix_row("complete-terra-w32")
    incomplete = _matrix_row("incomplete-terra-w32")
    matrix = tmp_path / "matrix.jsonl"
    matrix.write_text(
        "\n".join(json.dumps(row) for row in (complete, incomplete)) + "\n",
        encoding="utf-8",
    )
    build_root = tmp_path / "builds"
    complete_dir = build_root / "locomo" / "complete-terra-w32"
    _write_json(
        complete_dir / "build.json",
        {
            "status": "complete",
            "run_id": complete["run_id"],
            "builder_model": complete["model"],
            "write_turns": complete["write_turns"],
        },
    )
    _write_json(
        complete_dir / "memory" / "_SUCCESS.json",
        {"run_id": complete["run_id"]},
    )

    selected = contract.select_completed_runs(
        matrix_path=matrix,
        build_root=build_root,
        run_ids=[str(complete["run_id"])],
    )
    assert [binding.run_id for binding in selected] == [complete["run_id"]]

    with pytest.raises(contract.QAContractError, match="not complete"):
        contract.select_completed_runs(
            matrix_path=matrix,
            build_root=build_root,
            run_ids=[str(incomplete["run_id"])],
        )


def test_completed_run_selection_accepts_analysis_windows(tmp_path: Path) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract

    row = _matrix_row("locomo-terra-w16")
    row["write_turns"] = 16
    matrix = tmp_path / "matrix.jsonl"
    matrix.write_text(json.dumps(row) + "\n", encoding="utf-8")
    build_root = tmp_path / "builds"
    run_dir = build_root / "locomo" / "locomo-terra-w16"
    _write_json(
        run_dir / "build.json",
        {
            "status": "complete",
            "run_id": row["run_id"],
            "builder_model": row["model"],
            "write_turns": row["write_turns"],
        },
    )
    _write_json(run_dir / "memory" / "_SUCCESS.json", {"run_id": row["run_id"]})

    selected = contract.select_completed_runs(
        matrix_path=matrix,
        build_root=build_root,
        run_ids=[row["run_id"]],
    )

    assert [binding.run_id for binding in selected] == [row["run_id"]]


def test_completed_run_selection_accepts_session_groups(tmp_path: Path) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract

    row = _matrix_row("locomo-terra-s8")
    row["write_turns"] = "session"
    row["session_group_size"] = 8
    matrix = tmp_path / "matrix.jsonl"
    matrix.write_text(json.dumps(row) + "\n", encoding="utf-8")
    build_root = tmp_path / "builds"
    run_dir = build_root / "locomo" / "locomo-terra-s8"
    _write_json(
        run_dir / "build.json",
        {
            "status": "complete",
            "run_id": row["run_id"],
            "builder_model": row["model"],
            "write_turns": row["write_turns"],
            "session_group_size": row["session_group_size"],
        },
    )
    _write_json(run_dir / "memory" / "_SUCCESS.json", {"run_id": row["run_id"]})

    selected = contract.select_completed_runs(
        matrix_path=matrix,
        build_root=build_root,
        run_ids=[row["run_id"]],
    )

    assert [binding.run_id for binding in selected] == [row["run_id"]]


def test_completed_run_selection_uses_explicit_run_directory_manifest(
    tmp_path: Path,
) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract

    row = _matrix_row("locomo-terra-w8")
    row["write_turns"] = 8
    matrix = tmp_path / "matrix.jsonl"
    matrix.write_text(json.dumps(row) + "\n", encoding="utf-8")
    run_dir = tmp_path / "scattered-results" / "actual-run"
    _write_json(
        run_dir / "build.json",
        {
            "status": "complete",
            "run_id": row["run_id"],
            "builder_model": row["model"],
            "write_turns": row["write_turns"],
        },
    )
    _write_json(run_dir / "memory" / "_SUCCESS.json", {"run_id": row["run_id"]})
    manifest = tmp_path / "completed.jsonl"
    manifest.write_text(
        json.dumps({"run_id": row["run_id"], "run_dir": str(run_dir)}) + "\n",
        encoding="utf-8",
    )

    selected = contract.select_completed_runs(
        matrix_path=matrix,
        completed_runs_manifest=manifest,
        run_ids=[row["run_id"]],
    )

    assert selected[0].run_dir == run_dir.resolve()


def test_completed_run_selection_requires_reasoning_none(tmp_path: Path) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract

    row = _matrix_row("locomo-terra-reasoning")
    row["reasoning_effort"] = "medium"
    matrix = tmp_path / "matrix.jsonl"
    matrix.write_text(json.dumps(row) + "\n", encoding="utf-8")
    build_root = tmp_path / "builds"
    run_dir = build_root / "locomo" / "locomo-terra-reasoning"
    _write_json(
        run_dir / "build.json",
        {
            "status": "complete",
            "run_id": row["run_id"],
            "builder_model": row["model"],
            "write_turns": row["write_turns"],
        },
    )
    _write_json(run_dir / "memory" / "_SUCCESS.json", {"run_id": row["run_id"]})

    with pytest.raises(contract.QAContractError, match="reasoning_effort must be none"):
        contract.select_completed_runs(
            matrix_path=matrix,
            build_root=build_root,
            run_ids=[row["run_id"]],
        )


def test_source_token_count_uses_recorded_fallback_method() -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract

    conversation = {
        "session_1": [
            {"text": "abcdefgh"},
            {"text": "ijklmnop"},
        ]
    }

    assert contract.source_token_count_for_method(
        conversation,
        "chars_div_4_fallback",
    ) == 4

    with pytest.raises(contract.QAContractError, match="unsupported source tokenizer"):
        contract.source_token_count_for_method(conversation, "changed-tokenizer")


def test_beam_arrow_loader_reads_ipc_row_without_datasets(tmp_path: Path) -> None:
    import pyarrow as pa

    from scripts import gpt56_chunk_curve_qa_contract as contract

    arrow_path = tmp_path / "beam.arrow"
    table = pa.table(
        {
            "conversation_id": ["first", "second"],
            "payload": ["a", "b"],
        }
    )
    with arrow_path.open("wb") as handle:
        with pa.ipc.new_stream(handle, table.schema) as writer:
            writer.write_table(table)

    assert contract.load_arrow_row(arrow_path, 1) == {
        "conversation_id": "second",
        "payload": "b",
    }

    with pytest.raises(contract.QAContractError, match="outside Arrow table"):
        contract.load_arrow_row(arrow_path, 2)


def test_validate_locomo_unit_rebuilds_recorded_payload(tmp_path: Path) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract

    row = _matrix_row("locomo-terra-w32")
    matrix = tmp_path / "matrix.jsonl"
    matrix.write_text(json.dumps(row) + "\n", encoding="utf-8")
    build_root = tmp_path / "builds"
    run_dir = build_root / "locomo" / "locomo-terra-w32"
    _write_json(
        run_dir / "build.json",
        {
            "status": "complete",
            "run_id": row["run_id"],
            "builder_model": row["model"],
            "write_turns": row["write_turns"],
        },
    )
    _write_json(run_dir / "memory" / "_SUCCESS.json", {"run_id": row["run_id"]})
    conversation = {
        "session_1": [
            {"dia_id": "D1:1", "speaker": "A", "text": "abcdefgh"},
            {"dia_id": "D1:2", "speaker": "B", "text": "ijklmnop"},
        ],
        "session_1_date_time": "2026-01-01",
    }
    locomo_data = tmp_path / "locomo.json"
    _write_json(
        locomo_data,
        [
            {
                "sample_id": "sample-0",
                "conversation": conversation,
                "qa": [
                    {
                        "question": "What happened?",
                        "answer": "A fact",
                        "category": 1,
                        "evidence": ["D1:1"],
                    },
                    {
                        "question": "Adversarial?",
                        "adversarial_answer": "No",
                        "category": 5,
                    },
                ],
            }
        ],
    )
    questions = [
        {
            "question_id": "q0",
            "question": "What happened?",
            "gold": "A fact",
            "category": 1,
            "evidence": ["D1:1"],
        }
    ]
    _write_json(
        run_dir / "unit.json",
        {
            "run_id": row["run_id"],
            "benchmark": "locomo",
            "unit_id": row["unit_id"],
            "selection": row["selection"],
            "metadata": {"sample_id": "sample-0"},
            "messages": 2,
            "source_tokens": 4,
            "source_tokenizer": "chars_div_4_fallback",
            "questions": questions,
        },
    )

    binding = contract.select_completed_runs(
        matrix_path=matrix,
        build_root=build_root,
        run_ids=[str(row["run_id"])],
    )[0]
    rebuilt = contract.validate_unit_binding(binding, locomo_data=locomo_data)

    assert rebuilt["conversation"] == conversation
    assert rebuilt["questions"] == questions
    assert rebuilt["raw_dataset"] == {
        "path": str(locomo_data.resolve()),
        "sha256": hashlib.sha256(locomo_data.read_bytes()).hexdigest(),
    }

    plan = contract.build_preregistration(
        bindings=[binding],
        validated_units=[rebuilt],
        matrix_path=matrix,
        upstream="http://127.0.0.1:8205",
        upstream_code_sha256="a" * 64,
    )
    assert plan["protocol_id"] == "gpt56-w32-screening-qa-v1"
    assert plan["evidence_class"] == "screening_build"
    assert plan["qa"] == {
        "answer_model": "gpt-5.5",
        "visible_token_limit": 20_000,
        "max_retrieval_rounds": 12,
        "condition": "dual_source",
        "answer_max_tokens": 10_000,
        "accepted_answer_max_tokens": [4_096, 10_000],
        "completion_cap_enforcement": "posthoc_usage_validation",
        "upstream_ignored_client_parameters": ["max_output_tokens"],
        "answer_retries": 3,
        "question_max_attempts": 8,
        "retryable_question_errors": [
            "upstream_completed_empty_output",
            "upstream_transient_provider_server_error",
        ],
        "answer_prompts": contract.ANSWER_PROMPTS,
        "tokenizer": {
            "implementation": "tiktoken",
            "implementation_version": "0.12.0",
            "encoding_name": "o200k_base",
            "requested_model": "gpt-5.5",
            "resolution": "tiktoken_named_fallback",
            "fallback_encoding": "o200k_base",
            "fallback_reason": "requested_model_not_in_tiktoken_mapping",
            "provider_exact": False,
            "counting_note": (
                "Local tiktoken count used for an enforceable experiment budget; "
                "it is not a provider-reported exact count."
            ),
        },
    }
    assert plan["run_ids"] == [row["run_id"]]
    assert plan["question_count"] == 1
    assert plan["inputs"][0]["session_group_size"] is None
    assert plan["subscription_upstream"] == {
        "origin": "http://127.0.0.1:8205",
        "code_sha256": "a" * 64,
    }
    assert plan["execution_runtime"] == {
        "python_implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("tiktoken", "openai", "httpx")
        },
    }
    preregistered_input = plan["inputs"][0]
    assert preregistered_input["memory_before"]["file_count"] == 1
    assert preregistered_input["canonical_conversation_sha256"] == (
        _canonical_sha256(conversation)
    )
    assert preregistered_input["turn_index_sha256"] == _canonical_sha256(
        {
            "D1:1": {
                "date": "2026-01-01",
                "speaker": "A",
                "text": "abcdefgh",
            },
            "D1:2": {
                "date": "2026-01-01",
                "speaker": "B",
                "text": "ijklmnop",
            },
        }
    )
    assert preregistered_input["normalized_questions_sha256"] == (
        _canonical_sha256(questions)
    )
    assert preregistered_input["raw_dataset"] == rebuilt["raw_dataset"]
    assert {
        "qa_runner",
        "qa_auditor",
        "build_runner",
        "controlled_qa_proxy",
        "controlled_answer_contract",
        "evaluation_prompts",
        "controlled_answer_runner",
        "gpt55_run_proxy",
        "readonly_auditor",
        "visible_token_audit",
        "longmemeval_answer_runner",
        "beam_converter",
        "beam_screening_evaluator",
        "beam_screening_semantic_judge",
        "beam_answer_contract",
        "longmemeval_evaluator",
        "locomo_evidence_launcher",
    } <= set(plan["source_hashes"])
    assert all(
        len(value) == 64 for value in plan["source_hashes"].values()
    )

    original_dataset = locomo_data.read_bytes()
    locomo_data.write_bytes(original_dataset + b"\n")
    with pytest.raises(
        contract.QAContractError,
        match="raw dataset binding differs",
    ):
        contract.build_preregistration(
            bindings=[binding],
            validated_units=[rebuilt],
            matrix_path=matrix,
        )
    locomo_data.write_bytes(original_dataset)

    unit = json.loads((run_dir / "unit.json").read_text(encoding="utf-8"))
    unit["source_tokens"] = 5
    _write_json(run_dir / "unit.json", unit)
    with pytest.raises(contract.QAContractError, match="source token count differs"):
        contract.validate_unit_binding(binding, locomo_data=locomo_data)


def test_validate_longmemeval_unit_rebuilds_recorded_payload(tmp_path: Path) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts.run_v88_gpt55_longmemeval import longmemeval_to_locomo

    fixture = Path("tests/fixtures/longmemeval_s_tiny.json").resolve()
    item = json.loads(fixture.read_text(encoding="utf-8"))[0]
    row = {
        "run_id": "lme-terra-w32",
        "benchmark": "longmemeval-s",
        "tier": "terra",
        "model": "gpt-5.6-terra",
        "write_turns": 32,
        "reasoning_effort": "none",
        "unit_id": item["question_id"],
        "output_dir": "results/formal/gpt56-chunk-curve/longmemeval-s/lme-terra-w32",
        "selection": {"dataset_index": 0, "question_id": item["question_id"]},
    }
    matrix = tmp_path / "matrix.jsonl"
    matrix.write_text(json.dumps(row) + "\n", encoding="utf-8")
    build_root = tmp_path / "builds"
    run_dir = build_root / "longmemeval-s" / "lme-terra-w32"
    _write_json(
        run_dir / "build.json",
        {
            "status": "complete",
            "run_id": row["run_id"],
            "builder_model": row["model"],
            "write_turns": row["write_turns"],
        },
    )
    _write_json(run_dir / "memory" / "_SUCCESS.json", {"run_id": row["run_id"]})
    conversation = longmemeval_to_locomo(item, 0)
    question = {
        "question_id": item["question_id"],
        "question": item["question"],
        "question_date": item["question_date"],
        "gold": item["answer"],
        "question_type": item["question_type"],
        "answer_session_ids": item["answer_session_ids"],
    }
    source_tokens = contract.source_token_count_for_method(
        conversation,
        "chars_div_4_fallback",
    )
    _write_json(
        run_dir / "unit.json",
        {
            "run_id": row["run_id"],
            "benchmark": row["benchmark"],
            "unit_id": row["unit_id"],
            "selection": row["selection"],
            "metadata": {"dataset_index": 0},
            "messages": sum(
                len(value)
                for key, value in conversation.items()
                if key.startswith("session_") and isinstance(value, list)
            ),
            "source_tokens": source_tokens,
            "source_tokenizer": "chars_div_4_fallback",
            "questions": [question],
        },
    )

    binding = contract.select_completed_runs(
        matrix_path=matrix,
        build_root=build_root,
        run_ids=[row["run_id"]],
    )[0]
    rebuilt = contract.validate_unit_binding(
        binding,
        longmemeval_data=fixture,
    )

    assert rebuilt["conversation"] == conversation
    assert rebuilt["questions"] == [question]
    assert rebuilt["raw_dataset"] == {
        "path": str(fixture.resolve()),
        "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
    }


def test_validate_beam_unit_uses_arrow_ipc_fallback(tmp_path: Path) -> None:
    import pyarrow as pa

    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts.run_v88_gpt55_beam import conversation_to_native, extract_questions

    fixture = json.loads(
        Path("tests/fixtures/beam_hf_fixture.json").read_text(encoding="utf-8")
    )
    item = fixture["100K"][0]
    arrow_path = tmp_path / "beam.arrow"
    table = pa.Table.from_pylist([item])
    with arrow_path.open("wb") as handle:
        with pa.ipc.new_stream(handle, table.schema) as writer:
            writer.write_table(table)

    row = {
        "run_id": "beam-terra-w32",
        "benchmark": "beam-100k",
        "tier": "terra",
        "model": "gpt-5.6-terra",
        "write_turns": 32,
        "reasoning_effort": "none",
        "unit_id": "beam-fixture",
        "output_dir": "results/formal/gpt56-chunk-curve/beam-100k/beam-terra-w32",
        "selection": {
            "conversation_index": 0,
            "conversation_id": item["conversation_id"],
            "split": "100K",
        },
    }
    matrix = tmp_path / "matrix.jsonl"
    matrix.write_text(json.dumps(row) + "\n", encoding="utf-8")
    build_root = tmp_path / "builds"
    run_dir = build_root / "beam-100k" / "beam-terra-w32"
    _write_json(
        run_dir / "build.json",
        {
            "status": "complete",
            "run_id": row["run_id"],
            "builder_model": row["model"],
            "write_turns": row["write_turns"],
        },
    )
    _write_json(run_dir / "memory" / "_SUCCESS.json", {"run_id": row["run_id"]})
    conversation, metadata = conversation_to_native(item)
    questions = extract_questions(item)
    for question_index, question in enumerate(questions):
        question["question_id"] = f"{item['conversation_id']}-q{question_index}"
    _write_json(
        run_dir / "unit.json",
        {
            "run_id": row["run_id"],
            "benchmark": row["benchmark"],
            "unit_id": row["unit_id"],
            "selection": row["selection"],
            "metadata": {
                "conversation_id": item["conversation_id"],
                "source_id_count": len(metadata["source_id_map"]),
            },
            "messages": sum(
                len(value)
                for key, value in conversation.items()
                if key.startswith("session_") and isinstance(value, list)
            ),
            "source_tokens": contract.source_token_count_for_method(
                conversation,
                "chars_div_4_fallback",
            ),
            "source_tokenizer": "chars_div_4_fallback",
            "questions": questions,
        },
    )

    binding = contract.select_completed_runs(
        matrix_path=matrix,
        build_root=build_root,
        run_ids=[row["run_id"]],
    )[0]
    rebuilt = contract.validate_unit_binding(binding, beam_arrow=arrow_path)

    assert rebuilt["conversation"] == conversation
    assert rebuilt["questions"] == questions
    assert rebuilt["raw_dataset"] == {
        "path": str(arrow_path.resolve()),
        "sha256": hashlib.sha256(arrow_path.read_bytes()).hexdigest(),
    }


def test_locomo_evaluator_must_match_locked_sha(tmp_path: Path) -> None:
    from scripts import gpt56_chunk_curve_qa_contract as contract

    evaluator = Path("scripts/eval_full.py").resolve()
    assert contract.verify_locomo_evaluator(evaluator) == (
        "f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd"
    )

    changed = tmp_path / "eval_full.py"
    changed.write_bytes(evaluator.read_bytes() + b"\n")
    with pytest.raises(contract.QAContractError, match="locked SHA-256"):
        contract.verify_locomo_evaluator(changed)


def test_transient_provider_server_error_gets_bounded_retry_authorization(
    tmp_path: Path,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import run_gpt56_chunk_curve_qa as runner
    from src.evaluation.durable_model_ledger import HashChainLedger

    class HttpError(RuntimeError):
        def __init__(self, status_code: int, message: str) -> None:
            super().__init__(message)
            self.status_code = status_code

    qa_run_id = "qa-test"
    question_id = "conv-test::q0"
    attempt_root = tmp_path / "attempt-0001"
    attempt_root.mkdir()
    manifest = {
        "schema_version": 1,
        "qa_run_id": qa_run_id,
        "execution_run_id": f"{qa_run_id}:attempt-0001",
        "attempt": 1,
        "question_id": question_id,
        "benchmark": "locomo",
    }
    _write_json(attempt_root / "attempt_manifest.json", manifest)

    logical_call_id = (
        f"{qa_run_id}:attempt-0001:retrieval:{question_id}:dual_source:call-0002"
    )
    error_text = (
        "InternalServerError: Error code: 500 - {'error': "
        "'upstream failed after retries: upstream stream failed: "
        '{"code": "server_error", "message": "Please retry."}'
        "'}"
    )
    ledger = HashChainLedger(
        attempt_root / "retrieval_model_ledger.jsonl",
        run_id=f"{qa_run_id}:attempt-0001",
    )
    ledger.append(
        "model_call_failed",
        operation_id=f"retrieval:{question_id}:dual_source",
        logical_call_id=logical_call_id,
        error=error_text,
        proxy_evidence={
            "events": [
                {
                    "status": "error",
                    "http_status": 500,
                    "upstream_http_attempts": None,
                    "logical_call_id": logical_call_id,
                    "event_id": "event-server-error",
                }
            ]
        },
    )

    transient = HttpError(500, error_text)
    authorization = runner._retry_authorization_payload(
        attempt_root=attempt_root,
        qa_run_id=qa_run_id,
        question_id=question_id,
        attempt=1,
        exc=transient,
        formal=False,
    )
    assert authorization["reason"] == "upstream_transient_provider_server_error"
    assert authorization["from_attempt"] == 1
    assert authorization["to_attempt"] == 2
    _write_json(attempt_root / "retry_authorization.json", authorization)
    assert auditor._audit_retry_authorization(
        attempt_root=attempt_root,
        manifest=manifest,
        formal=False,
    ) == authorization

    with pytest.raises(runner.QARunnerError, match="not allowed"):
        runner._retry_authorization_payload(
            attempt_root=attempt_root,
            qa_run_id=qa_run_id,
            question_id=question_id,
            attempt=contract.QUESTION_MAX_ATTEMPTS,
            exc=transient,
            formal=False,
        )

    non_retryable = [
        HttpError(401, error_text),
        HttpError(429, error_text),
        HttpError(500, "HTTP 401: unauthorized"),
        HttpError(500, "HTTP 429: usage_limit_reached"),
        HttpError(
            500,
            "upstream completed with empty output: usage_limit_reached",
        ),
        HttpError(500, "upstream completed with empty output: unauthorized"),
        HttpError(500, "upstream failed after retries: HTTP 500"),
    ]
    for error in non_retryable:
        with pytest.raises(runner.QARunnerError, match="not allowed"):
            runner._retry_authorization_payload(
                attempt_root=attempt_root,
                qa_run_id=qa_run_id,
                question_id=question_id,
                attempt=1,
                exc=error,
                formal=False,
            )


@pytest.mark.parametrize(
    "error_text",
    [
        (
            'Error code: 500 - upstream stream failed: '
            '{"code": "server_is_overloaded", "message": "retry"}'
        ),
        (
            "Error code: 500 - upstream failed after retries: ProxyError: "
            "Unable to connect to proxy; Remote end closed connection "
            "without response"
        ),
        (
            "Error code: 500 - upstream failed after retries: "
            "ConnectionError: HTTPSConnectionPool(host='chatgpt.com', "
            "port=443): Read timed out."
        ),
    ],
)
def test_additional_transient_gateway_failures_are_retryable(
    error_text: str,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import run_gpt56_chunk_curve_qa as runner

    expected = contract.TRANSIENT_PROVIDER_SERVER_ERROR_RETRY_REASON
    assert runner._retryable_error_reason(
        status_code=500,
        error_text=error_text,
    ) == expected
    assert auditor._retryable_error_reason(
        status_code=500,
        error_text=error_text,
    ) == expected


def test_answer_stage_transient_gateway_failure_gets_retry_authorization(
    tmp_path: Path,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor
    from scripts import controlled_locomo_answer_contract as answer_contract
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import run_gpt56_chunk_curve_qa as runner

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
    _write_json(attempt_root / "attempt_manifest.json", manifest)
    logical_call_id = "qa-test:attempt-0001:conv-test::q0:dual_source:answer"
    error = (
        "upstream failed after retries: ProxyError: Unable to connect to proxy; "
        "Connection reset by peer"
    )
    with answer_contract.DurableLedger(
        attempt_root / "answer_ledger.jsonl",
        run_id=logical_call_id,
    ) as ledger:
        ledger.append(
            "physical_http_attempt_finished",
            {
                "status": "unaccountable_error",
                "error": error,
                "http_status": 500,
                "upstream_http_attempts": None,
                "proxy_event_id": "event-answer-transient",
            },
        )

    exc = answer_contract.ControlledAnswerError(
        "failed answer request has unknown provider attempts"
    )
    authorization = runner._retry_authorization_payload(
        attempt_root=attempt_root,
        qa_run_id="qa-test",
        question_id=question_id,
        attempt=1,
        exc=exc,
        formal=True,
    )
    assert authorization["reason"] == (
        contract.TRANSIENT_PROVIDER_SERVER_ERROR_RETRY_REASON
    )
    assert authorization["failure"]["stage"] == "answer"
    _write_json(attempt_root / "retry_authorization.json", authorization)
    assert auditor._audit_retry_authorization(
        attempt_root=attempt_root,
        manifest=manifest,
        formal=True,
    ) == authorization
