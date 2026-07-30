from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest


def test_independent_auditor_expands_compact_gold_source_specs() -> None:
    from scripts import audit_readonly_nativemem_control as auditor

    assert auditor.expand_source_specs(["D1:1, 2-3", "D2:4"]) == [
        "D1:1",
        "D1:2",
        "D1:3",
        "D2:4",
    ]


def test_readonly_auditor_reconstructs_the_declared_beam_prompt(
    tmp_path: Path,
) -> None:
    from scripts import audit_readonly_nativemem_control as generic_auditor
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import r115_beam_control_contract as beam_answer_contract
    from scripts import readonly_nativemem_control as readonly
    from scripts.run_controlled_locomo_answers import FakeAnswerClient
    from src.evaluation.visible_token_budget import TokenCounter

    memory = tmp_path / "memory"
    (memory / "topics").mkdir(parents=True)
    (memory / "timeline").mkdir()
    (memory / "topics" / "fact.md").write_text(
        "The target answer is blue [D1:1].\n",
        encoding="utf-8",
    )
    turn_index = {
        "D1:1": {
            "date": "2026-01-01",
            "speaker": "A",
            "text": "The target answer is blue.",
        }
    }
    artifact_dir = tmp_path / "question"
    question = "What color is the target?"
    readonly.execute_question(
        artifact_dir=artifact_dir,
        run_id="beam-test",
        method=contract.PROTOCOL_ID,
        condition="dual_source",
        memory_root=memory,
        turn_index=turn_index,
        question_id="beam-q0",
        question=question,
        gold_source_ids=[],
        source_recall_eligible=False,
        completion_resource=readonly.ScriptedCompletionResource([[]]),
        answer_client=FakeAnswerClient(
            proxy_log=tmp_path / "proxy.jsonl",
            run_id="beam-test",
            response_text="blue",
        ),
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        formal=False,
        proxy_log=tmp_path / "proxy.jsonl",
        answer_prompt_builder=contract.assemble_beam_answer_prompt,
        answer_prompt_kind=contract.BEAM_ANSWER_PROMPT_KIND,
        answer_prompt_template_sha256=contract.BEAM_ANSWER_PROMPT_SHA256,
    )

    with pytest.raises(
        generic_auditor.ReadOnlyAuditError,
        match="fixed answer prompt reconstruction differs",
    ):
        generic_auditor.audit_question(
            artifact_dir=artifact_dir,
            memory_root=memory,
            method=contract.PROTOCOL_ID,
            condition="dual_source",
            question_id="beam-q0",
            question=question,
            gold_source_ids=[],
            source_recall_eligible=False,
            formal=False,
            turn_index=turn_index,
        )

    report = generic_auditor.audit_question(
        artifact_dir=artifact_dir,
        memory_root=memory,
        method=contract.PROTOCOL_ID,
        condition="dual_source",
        question_id="beam-q0",
        question=question,
        gold_source_ids=[],
        source_recall_eligible=False,
        formal=False,
        turn_index=turn_index,
        answer_prompt_template=beam_answer_contract.ANSWER_PROMPT,
        answer_prompt_kind=contract.BEAM_ANSWER_PROMPT_KIND,
        answer_prompt_template_sha256=contract.BEAM_ANSWER_PROMPT_SHA256,
    )
    assert report["status"] == "passed"


def test_readonly_auditor_accepts_terminal_budget_rejections(
    tmp_path: Path,
) -> None:
    from scripts import audit_readonly_nativemem_control as auditor
    from scripts import readonly_nativemem_control as readonly
    from scripts.run_controlled_locomo_answers import FakeAnswerClient
    from src.evaluation.visible_token_budget import TokenCounter

    memory = tmp_path / "memory"
    (memory / "topics").mkdir(parents=True)
    (memory / "timeline").mkdir()
    (memory / "topics" / "large.md").write_text(
        "x" * 25_000,
        encoding="utf-8",
    )
    (memory / "topics" / "later.md").write_text(
        "This tool result is rejected after exhaustion.\n",
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "question"
    question = "What was recorded?"
    readonly.execute_question(
        artifact_dir=artifact_dir,
        run_id="budget-terminal-test",
        method="test-method",
        condition="dual_source",
        memory_root=memory,
        turn_index={},
        question_id="q0",
        question=question,
        gold_source_ids=[],
        source_recall_eligible=False,
        completion_resource=readonly.ScriptedCompletionResource(
            [
                [
                    {
                        "name": "read_memory_file",
                        "arguments": {"path": "topics/large.md"},
                    },
                    {
                        "name": "read_memory_file",
                        "arguments": {"path": "topics/later.md"},
                    },
                ]
            ]
        ),
        answer_client=FakeAnswerClient(
            proxy_log=tmp_path / "proxy.jsonl",
            run_id="budget-terminal-test",
            response_text="unknown",
        ),
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        formal=False,
        proxy_log=tmp_path / "proxy.jsonl",
    )

    trace = [
        json.loads(line)
        for line in (artifact_dir / "visible_tokens.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(
        record.get("decision") == "rejected_budget_exhausted"
        and record.get("delivered") is None
        for record in trace
    )

    report = auditor.audit_question(
        artifact_dir=artifact_dir,
        memory_root=memory,
        method="test-method",
        condition="dual_source",
        question_id="q0",
        question=question,
        gold_source_ids=[],
        source_recall_eligible=False,
        formal=False,
        turn_index={},
    )
    assert report["status"] == "passed"


def test_audit_question_records_verifies_artifact_hashes(tmp_path: Path) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor
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
        "unit": {"run_id": row["run_id"]},
    }
    qa_root = tmp_path / "qa"
    proxy_log = tmp_path / "proxy.jsonl"
    plan = {
        "schema_version": 1,
        "protocol_id": "gpt56-w32-screening-qa-v1",
        "run_ids": [row["run_id"]],
        "question_count": 1,
    }
    records = runner.run_execution(
        plan=plan,
        bindings=[binding],
        validated_units=[validated],
        output_dir=qa_root,
        qa_run_id="qa-test",
        completion_resource=readonly.ScriptedCompletionResource(
            [
                [
                    {
                        "name": "read_memory_file",
                        "arguments": {"path": "topics/fact.md"},
                    }
                ],
                [],
            ]
        ),
        answer_client=FakeAnswerClient(
            proxy_log=proxy_log,
            run_id="qa-test",
            response_text="<answer>recorded</answer>",
        ),
        tokenizer=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
        proxy_log=proxy_log,
        formal=False,
    )
    record = records[0]

    report = auditor.audit_question_records(
        records=[record],
        qa_root=qa_root,
        formal=False,
        bindings=[binding],
        validated_units=[validated],
    )
    assert report["question_count"] == 1
    assert report["artifact_count"] == 5
    assert report["independently_reconstructed_questions"] == 1
    assert report["all_memory_unchanged"] is True

    root_report = auditor.audit_qa_root(
        plan=plan,
        bindings=[binding],
        validated_units=[validated],
        qa_root=qa_root,
        formal=False,
    )
    assert root_report["status"] == "verified_complete"
    assert root_report["question_count"] == 1

    extra = qa_root / "terra" / "evaluator_inputs" / "unexpected.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(auditor.QAAuditError, match="unexpected evaluator input"):
        auditor.audit_qa_root(
            plan=plan,
            bindings=[binding],
            validated_units=[validated],
            qa_root=qa_root,
            formal=False,
        )
    extra.unlink()

    from src.evaluation import durable_model_ledger as durable

    attempt_root = next(qa_root.rglob("attempt-*/result.json")).parent
    ledger_path = attempt_root / "retrieval_model_ledger.jsonl"
    ledger_records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    first_start = next(
        item for item in ledger_records if item.get("event") == "model_call_started"
    )
    request_path = attempt_root / first_start["request_path"]
    result_path = attempt_root / "result.json"
    record_path = result_path.parent.parent / "record.json"
    originals = {
        path: path.read_bytes()
        for path in (request_path, ledger_path, result_path, record_path)
    }
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["model_visible_payload"]["messages"][0]["content"] = (
        "semantically tampered retrieval prompt"
    )
    request_path.write_text(
        durable.canonical_json(request) + "\n", encoding="utf-8"
    )
    for ledger_record in ledger_records:
        if ledger_record.get("request_path") == first_start["request_path"]:
            ledger_record["request_sha256"] = durable.sha256_file(request_path)
    previous = durable.ZERO_HASH
    for ledger_record in ledger_records:
        ledger_record["previous_event_sha256"] = previous
        ledger_record.pop("event_sha256", None)
        event_sha = durable.sha256_bytes(
            durable.canonical_json(ledger_record).encode("utf-8")
        )
        ledger_record["event_sha256"] = event_sha
        previous = event_sha
    ledger_path.write_text(
        "".join(
            durable.canonical_json(ledger_record) + "\n"
            for ledger_record in ledger_records
        ),
        encoding="utf-8",
    )
    semantic_tamper = copy.deepcopy(record)
    semantic_tamper["result"]["retrieval"]["ledger_state"] = (
        durable.ledger_state(durable.read_ledger(ledger_path))
    )
    semantic_tamper["result"]["artifacts"][
        "retrieval_model_ledger_sha256"
    ] = durable.sha256_file(ledger_path)
    result_path.write_text(
        durable.canonical_json(semantic_tamper["result"]) + "\n",
        encoding="utf-8",
    )
    record_path.write_text(
        durable.canonical_json(semantic_tamper) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(auditor.QAAuditError, match="retrieval prompt reconstruction"):
        auditor.audit_question_records(
            records=[semantic_tamper],
            qa_root=qa_root,
            formal=False,
            bindings=[binding],
            validated_units=[validated],
        )
    for path, payload in originals.items():
        path.write_bytes(payload)

    ledger_records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    starts = [
        item for item in ledger_records if item.get("event") == "model_call_started"
    ]
    second_start = starts[1]
    second_request_path = attempt_root / second_start["request_path"]
    continuation_originals = {
        path: path.read_bytes()
        for path in (second_request_path, ledger_path, result_path, record_path)
    }
    second_request = json.loads(second_request_path.read_text(encoding="utf-8"))
    second_request["model_visible_payload"]["messages"][-1]["content"] = (
        "tampered continuation"
    )
    second_request_path.write_text(
        durable.canonical_json(second_request) + "\n", encoding="utf-8"
    )
    for ledger_record in ledger_records:
        if ledger_record.get("request_path") == second_start["request_path"]:
            ledger_record["request_sha256"] = durable.sha256_file(
                second_request_path
            )
    previous = durable.ZERO_HASH
    for ledger_record in ledger_records:
        ledger_record["previous_event_sha256"] = previous
        ledger_record.pop("event_sha256", None)
        event_sha = durable.sha256_bytes(
            durable.canonical_json(ledger_record).encode("utf-8")
        )
        ledger_record["event_sha256"] = event_sha
        previous = event_sha
    ledger_path.write_text(
        "".join(
            durable.canonical_json(ledger_record) + "\n"
            for ledger_record in ledger_records
        ),
        encoding="utf-8",
    )
    continuation_tamper = copy.deepcopy(record)
    continuation_tamper["result"]["retrieval"]["ledger_state"] = (
        durable.ledger_state(durable.read_ledger(ledger_path))
    )
    continuation_tamper["result"]["artifacts"][
        "retrieval_model_ledger_sha256"
    ] = durable.sha256_file(ledger_path)
    result_path.write_text(
        durable.canonical_json(continuation_tamper["result"]) + "\n",
        encoding="utf-8",
    )
    record_path.write_text(
        durable.canonical_json(continuation_tamper) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        auditor.QAAuditError,
        match="retrieval message chain reconstruction",
    ):
        auditor.audit_question_records(
            records=[continuation_tamper],
            qa_root=qa_root,
            formal=False,
            bindings=[binding],
            validated_units=[validated],
        )
    for path, payload in continuation_originals.items():
        path.write_bytes(payload)

    ledger_records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    first_start = next(
        item for item in ledger_records if item.get("event") == "model_call_started"
    )
    request_path = attempt_root / first_start["request_path"]
    tool_originals = {
        path: path.read_bytes()
        for path in (request_path, ledger_path, result_path, record_path)
    }
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["model_visible_payload"]["tools"][0]["function"]["name"] = (
        "tampered_tool"
    )
    request_path.write_text(
        durable.canonical_json(request) + "\n", encoding="utf-8"
    )
    for ledger_record in ledger_records:
        if ledger_record.get("request_path") == first_start["request_path"]:
            ledger_record["request_sha256"] = durable.sha256_file(request_path)
    previous = durable.ZERO_HASH
    for ledger_record in ledger_records:
        ledger_record["previous_event_sha256"] = previous
        ledger_record.pop("event_sha256", None)
        event_sha = durable.sha256_bytes(
            durable.canonical_json(ledger_record).encode("utf-8")
        )
        ledger_record["event_sha256"] = event_sha
        previous = event_sha
    ledger_path.write_text(
        "".join(
            durable.canonical_json(ledger_record) + "\n"
            for ledger_record in ledger_records
        ),
        encoding="utf-8",
    )
    tool_tamper = copy.deepcopy(record)
    tool_tamper["result"]["retrieval"]["ledger_state"] = durable.ledger_state(
        durable.read_ledger(ledger_path)
    )
    tool_tamper["result"]["artifacts"][
        "retrieval_model_ledger_sha256"
    ] = durable.sha256_file(ledger_path)
    result_path.write_text(
        durable.canonical_json(tool_tamper["result"]) + "\n",
        encoding="utf-8",
    )
    record_path.write_text(
        durable.canonical_json(tool_tamper) + "\n", encoding="utf-8"
    )
    with pytest.raises(
        auditor.QAAuditError,
        match="retrieval tool definition reconstruction",
    ):
        auditor.audit_question_records(
            records=[tool_tamper],
            qa_root=qa_root,
            formal=False,
            bindings=[binding],
            validated_units=[validated],
        )
    for path, payload in tool_originals.items():
        path.write_bytes(payload)

    wrong_prompt = copy.deepcopy(record)
    wrong_prompt["result"]["prompt"]["kind"] = "beam-r115-answer-v1"
    with pytest.raises(auditor.QAAuditError, match="answer prompt differs"):
        auditor.audit_question_records(
            records=[wrong_prompt],
            qa_root=qa_root,
            formal=False,
            bindings=[binding],
            validated_units=[validated],
        )

    result_path = next(qa_root.rglob("attempt-*/result.json"))
    original_result_text = result_path.read_text(encoding="utf-8")
    wrong_prompt_sha = copy.deepcopy(record)
    wrong_prompt_sha["result"]["prompt"]["sha256"] = "0" * 64
    result_path.write_text(
        json.dumps(wrong_prompt_sha["result"], sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(auditor.QAAuditError, match="independent question audit failed"):
        auditor.audit_question_records(
            records=[wrong_prompt_sha],
            qa_root=qa_root,
            formal=False,
            bindings=[binding],
            validated_units=[validated],
        )
    result_path.write_text(original_result_text, encoding="utf-8")

    trace = next(qa_root.rglob("visible_tokens.jsonl"))
    trace.write_text(trace.read_text() + "tamper\n", encoding="utf-8")
    with pytest.raises(auditor.QAAuditError, match="artifact SHA-256 differs"):
        auditor.audit_question_records(
            records=[record],
            qa_root=qa_root,
            formal=False,
            bindings=[binding],
            validated_units=[validated],
        )


def test_audit_proxy_logs_requires_success_and_zero_reasoning(tmp_path: Path) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor

    invocation = tmp_path / "qa/proxy/invocation-0001"
    invocation.mkdir(parents=True)
    log = invocation / "requests.jsonl"
    event = {
        "run_id": "qa-run",
        "status": "success",
        "http_status": 200,
        "requested_model": "gpt-5.5",
        "actual_model": "gpt-5.5",
        "response_id": "response-1",
        "event_id": "event-1",
        "question_id": "question-1",
        "logical_call_id": "logical-1",
        "request_sha256": "a" * 64,
        "response_sha256": "b" * 64,
        "client_http_attempts": 1,
        "child_upstream_http_attempts": 1,
        "upstream_http_attempts": 1,
        "unsupported_parameters": [],
        "ignored_client_parameters": ["max_output_tokens"],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        "error": None,
    }
    log.write_text(json.dumps(event) + "\n", encoding="utf-8")
    (invocation / "stop.json").write_text(
        json.dumps({"run_id": "qa-run", "returncode": -15}),
        encoding="utf-8",
    )

    report = auditor.audit_proxy_logs(qa_root=tmp_path / "qa", qa_run_id="qa-run")
    assert report["proxy_events"] == 1
    assert report["response_ids"] == 1
    assert report["reasoning_tokens"] == 0

    failed = {
        **event,
        "status": "error",
        "http_status": 502,
        "actual_model": None,
        "response_id": None,
        "event_id": "event-0",
        "ignored_client_parameters": [],
        "usage": None,
        "error": "temporary upstream error",
    }
    log.write_text(
        json.dumps(failed) + "\n" + json.dumps(event) + "\n",
        encoding="utf-8",
    )
    retried = auditor.audit_proxy_logs(qa_root=tmp_path / "qa", qa_run_id="qa-run")
    assert retried["proxy_events"] == 2
    assert retried["failed_events"] == 1
    assert retried["response_ids"] == 1

    unknown_provider_count = copy.deepcopy(failed)
    unknown_provider_count["child_upstream_http_attempts"] = 1
    unknown_provider_count["upstream_http_attempts"] = None
    log.write_text(
        json.dumps(unknown_provider_count) + "\n", encoding="utf-8"
    )
    unknown_report = auditor.audit_proxy_logs(
        qa_root=tmp_path / "qa", qa_run_id="qa-run"
    )
    assert unknown_report["upstream_http_attempts"] is None
    assert unknown_report["known_upstream_http_attempts"] == 0
    assert unknown_report["unknown_upstream_attempt_events"] == 1

    bad_child_count = copy.deepcopy(event)
    bad_child_count["child_upstream_http_attempts"] = 0
    log.write_text(json.dumps(bad_child_count) + "\n", encoding="utf-8")
    with pytest.raises(auditor.QAAuditError, match="proxy event contract differs"):
        auditor.audit_proxy_logs(qa_root=tmp_path / "qa", qa_run_id="qa-run")

    bad_provider_count = copy.deepcopy(event)
    bad_provider_count["upstream_http_attempts"] = 3
    log.write_text(json.dumps(bad_provider_count) + "\n", encoding="utf-8")
    with pytest.raises(auditor.QAAuditError, match="proxy event contract differs"):
        auditor.audit_proxy_logs(qa_root=tmp_path / "qa", qa_run_id="qa-run")

    bad_usage = copy.deepcopy(event)
    bad_usage["usage"]["total_tokens"] = 99
    log.write_text(json.dumps(bad_usage) + "\n", encoding="utf-8")
    with pytest.raises(auditor.QAAuditError, match="proxy event contract differs"):
        auditor.audit_proxy_logs(qa_root=tmp_path / "qa", qa_run_id="qa-run")

    event["usage"]["completion_tokens_details"]["reasoning_tokens"] = 1
    log.write_text(
        json.dumps(failed) + "\n" + json.dumps(event) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(auditor.QAAuditError, match="proxy event contract differs"):
        auditor.audit_proxy_logs(qa_root=tmp_path / "qa", qa_run_id="qa-run")


def test_audit_proxy_logs_attributes_abandoned_success_to_its_attempt(
    tmp_path: Path,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor
    from scripts import gpt56_chunk_curve_qa_contract as contract

    qa_root = tmp_path / "qa"
    question_root = (
        qa_root
        / "terra"
        / "locomo"
        / "conv-test"
        / "questions"
        / "conv-test_q0"
    )

    def write_attempt(attempt: int) -> tuple[Path, str]:
        root = question_root / f"attempt-{attempt:04d}"
        root.mkdir(parents=True)
        execution_run_id = f"qa-run:attempt-{attempt:04d}"
        manifest = {
            "schema_version": 1,
            "qa_run_id": "qa-run",
            "execution_run_id": execution_run_id,
            "attempt": attempt,
            "question_id": "conv-test::q0",
            "benchmark": "locomo",
            "answer_prompt_kind": contract.SHORT_ANSWER_PROMPT_KIND,
            "answer_prompt_template_sha256": (
                contract.SHORT_ANSWER_PROMPT_SHA256
            ),
        }
        (root / "attempt_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return root, execution_run_id

    abandoned_root, abandoned_run = write_attempt(1)
    accepted_root, accepted_run = write_attempt(2)
    abandoned_call = f"{abandoned_run}:conv-test::q0:dual_source:answer"
    accepted_call = f"{accepted_run}:conv-test::q0:dual_source:answer"
    for root, logical_call_id in (
        (abandoned_root, abandoned_call),
        (accepted_root, accepted_call),
    ):
        calls = root / "calls"
        calls.mkdir()
        (calls / "answer.request.json").write_text(
            json.dumps({"logical_call_id": logical_call_id}),
            encoding="utf-8",
        )

    def event(*, event_id: str, response_id: str, logical_call_id: str) -> dict:
        return {
            "run_id": "qa-run",
            "status": "success",
            "http_status": 200,
            "requested_model": "gpt-5.5",
            "actual_model": "gpt-5.5",
            "response_id": response_id,
            "event_id": event_id,
            "question_id": "conv-test::q0",
            "logical_call_id": logical_call_id,
            "request_sha256": "a" * 64,
            "response_sha256": "b" * 64,
            "client_http_attempts": 1,
            "requested_completion_token_cap": 4_096,
            "child_upstream_http_attempts": 1,
            "upstream_http_attempts": 1,
            "unsupported_parameters": [],
            "ignored_client_parameters": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            "error": None,
        }

    abandoned_event = event(
        event_id="event-abandoned",
        response_id="response-abandoned",
        logical_call_id=abandoned_call,
    )
    accepted_event = event(
        event_id="event-accepted",
        response_id="response-accepted",
        logical_call_id=accepted_call,
    )
    invocation = qa_root / "proxy" / "invocation-0001"
    invocation.mkdir(parents=True)
    log = invocation / "requests.jsonl"
    log.write_text(
        json.dumps(abandoned_event) + "\n" + json.dumps(accepted_event) + "\n",
        encoding="utf-8",
    )
    (invocation / "stop.json").write_text(
        json.dumps({"run_id": "qa-run", "returncode": -15}),
        encoding="utf-8",
    )
    record = {
        "question_id": "conv-test::q0",
        "benchmark": "locomo",
        "result": {
            "run_id": accepted_run,
            "answer": {
                "proxy_evidence": {"events": [accepted_event]},
            },
        },
    }

    report = auditor.audit_proxy_logs(
        qa_root=qa_root,
        qa_run_id="qa-run",
        records=[record],
    )
    assert report["accepted_successes"] == 1
    assert report["abandoned_successes"] == 1
    assert report["all_successes_accounted"] is True

    wrong_cap = copy.deepcopy(accepted_event)
    wrong_cap["requested_completion_token_cap"] = 1_200
    log.write_text(
        json.dumps(abandoned_event) + "\n" + json.dumps(wrong_cap) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(auditor.QAAuditError, match="completion token cap differs"):
        auditor.audit_proxy_logs(
            qa_root=qa_root,
            qa_run_id="qa-run",
            records=[record],
        )

    log.write_text(
        json.dumps(abandoned_event) + "\n" + json.dumps(accepted_event) + "\n",
        encoding="utf-8",
    )
    unbound = event(
        event_id="event-unbound",
        response_id="response-unbound",
        logical_call_id="qa-run:attempt-9999:conv-test::q0:dual_source:answer",
    )
    log.write_text(
        log.read_text(encoding="utf-8") + json.dumps(unbound) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(auditor.QAAuditError, match="proxy event attempt binding differs"):
        auditor.audit_proxy_logs(
            qa_root=qa_root,
            qa_run_id="qa-run",
            records=[record],
        )


def test_attempt_manifest_rejects_attempt_above_preregistered_limit(
    tmp_path: Path,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor
    from scripts import gpt56_chunk_curve_qa_contract as contract

    root = (
        tmp_path
        / "qa/sol/locomo/conv-test/questions/conv-test_q0/attempt-0009"
    )
    root.mkdir(parents=True)
    (root / "attempt_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "qa_run_id": "qa-run",
                "execution_run_id": "qa-run:attempt-0009",
                "attempt": 9,
                "question_id": "conv-test::q0",
                "benchmark": "locomo",
                "answer_prompt_kind": contract.SHORT_ANSWER_PROMPT_KIND,
                "answer_prompt_template_sha256": (
                    contract.SHORT_ANSWER_PROMPT_SHA256
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(auditor.QAAuditError, match="attempt limit"):
        auditor._attempt_manifests(qa_root=tmp_path / "qa", qa_run_id="qa-run")


def test_attempt_request_ids_include_answer_ledger_logical_call(
    tmp_path: Path,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor
    from scripts import controlled_locomo_answer_contract as answer_contract

    logical_call_id = "qa-run:attempt-0001:conv-test::q0:dual_source:answer"
    with answer_contract.DurableLedger(
        tmp_path / "answer_ledger.jsonl", run_id=logical_call_id
    ) as ledger:
        ledger.append(
            "physical_http_attempt_started",
            {
                "logical_call_id": logical_call_id,
                "attempt": 1,
            },
        )

    assert logical_call_id in auditor._attempt_request_ids(tmp_path)


def test_retry_proxy_bindings_require_the_exact_failed_proxy_event() -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor

    binding = {
        "question_id": "conv-test::q0",
        "from_attempt": 1,
        "to_attempt": 2,
        "stage": "answer",
        "logical_call_id": "qa-run:attempt-0001:conv-test::q0:dual_source:answer",
        "proxy_event_id": "failed-event-1",
    }
    event = {
        "event_id": "failed-event-1",
        "question_id": "conv-test::q0",
        "logical_call_id": binding["logical_call_id"],
        "status": "error",
        "http_status": 500,
        "upstream_http_attempts": None,
        "error": "upstream HTTP 500",
    }

    report = auditor._audit_retry_proxy_bindings(
        retry_bindings=[binding],
        events=[event],
    )
    assert report == {
        "bound_retry_events": 1,
        "retry_event_ids": ["failed-event-1"],
    }

    wrong = copy.deepcopy(event)
    wrong["logical_call_id"] = "different-call"
    with pytest.raises(auditor.QAAuditError, match="retry proxy event differs"):
        auditor._audit_retry_proxy_bindings(
            retry_bindings=[binding],
            events=[wrong],
        )


def test_formal_proxy_audit_enforces_authorized_retry_chain(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor
    from scripts import controlled_locomo_answer_contract as answer_contract
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import run_gpt56_chunk_curve_qa as runner

    class EmptyOutputError(RuntimeError):
        status_code = 500

    qa_root = tmp_path / "qa"
    question_root = (
        qa_root
        / "terra/locomo/conv-test/questions/conv-test_q0"
    )
    question_id = "conv-test::q0"

    def write_manifest(attempt: int) -> tuple[Path, str]:
        root = question_root / f"attempt-{attempt:04d}"
        root.mkdir(parents=True)
        execution_run_id = f"qa-run:attempt-{attempt:04d}"
        (root / "attempt_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "qa_run_id": "qa-run",
                    "execution_run_id": execution_run_id,
                    "attempt": attempt,
                    "question_id": question_id,
                    "benchmark": "locomo",
                    "answer_prompt_kind": contract.SHORT_ANSWER_PROMPT_KIND,
                    "answer_prompt_template_sha256": (
                        contract.SHORT_ANSWER_PROMPT_SHA256
                    ),
                }
            ),
            encoding="utf-8",
        )
        return root, execution_run_id

    failed_root, failed_run = write_manifest(1)
    accepted_root, accepted_run = write_manifest(2)
    failed_call = f"{failed_run}:{question_id}:dual_source:answer"
    accepted_call = f"{accepted_run}:{question_id}:dual_source:answer"
    with answer_contract.DurableLedger(
        failed_root / "answer_ledger.jsonl", run_id=failed_call
    ) as ledger:
        ledger.append(
            "physical_http_attempt_finished",
            {
                "status": "retryable_empty_output",
                "error": "upstream completed with empty output",
                "http_status": 500,
                "upstream_http_attempts": None,
                "proxy_event_id": "failed-event",
            },
        )
    authorization = runner._retry_authorization_payload(
        attempt_root=failed_root,
        qa_run_id="qa-run",
        question_id=question_id,
        attempt=1,
        exc=EmptyOutputError("upstream completed with empty output"),
        formal=True,
    )
    authorization_path = failed_root / "retry_authorization.json"
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    with answer_contract.DurableLedger(
        accepted_root / "answer_ledger.jsonl", run_id=accepted_call
    ) as ledger:
        ledger.append("answer_completed", {"response_id": "response-ok"})

    failed_event = {
        "run_id": "qa-run",
        "status": "error",
        "http_status": 500,
        "requested_model": "gpt-5.5",
        "actual_model": None,
        "response_id": None,
        "event_id": "failed-event",
        "question_id": question_id,
        "logical_call_id": failed_call,
        "request_sha256": "a" * 64,
        "response_sha256": "b" * 64,
        "client_http_attempts": 1,
        "requested_completion_token_cap": 4_096,
        "child_upstream_http_attempts": 1,
        "upstream_http_attempts": None,
        "unsupported_parameters": [],
        "ignored_client_parameters": [],
        "usage": None,
        "error": "upstream HTTP 500",
    }
    accepted_event = {
        **failed_event,
        "status": "success",
        "http_status": 200,
        "actual_model": "gpt-5.5",
        "response_id": "response-ok",
        "event_id": "accepted-event",
        "logical_call_id": accepted_call,
        "request_sha256": "c" * 64,
        "response_sha256": "d" * 64,
        "upstream_http_attempts": 1,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        "error": None,
    }
    invocation = qa_root / "proxy/invocation-0001"
    invocation.mkdir(parents=True)
    (invocation / "requests.jsonl").write_text(
        json.dumps(failed_event) + "\n" + json.dumps(accepted_event) + "\n",
        encoding="utf-8",
    )
    (invocation / "stop.json").write_text("{}", encoding="utf-8")
    record = {
        "question_id": question_id,
        "benchmark": "locomo",
        "result": {
            "run_id": accepted_run,
            "answer": {"proxy_evidence": {"events": [accepted_event]}},
        },
    }
    monkeypatch.setattr(auditor, "_audit_proxy_invocation", lambda **kwargs: None)

    report = auditor.audit_proxy_logs(
        qa_root=qa_root,
        qa_run_id="qa-run",
        records=[record],
        plan={},
    )
    assert report["attempt_chains"]["accepted_attempts"] == 1
    assert report["attempt_chains"]["authorized_retries"] == 1
    assert report["attempt_chains"]["bound_retry_events"] == 1

    authorization_path.unlink()
    with pytest.raises(auditor.QAAuditError, match="retry authorization"):
        auditor.audit_proxy_logs(
            qa_root=qa_root,
            qa_run_id="qa-run",
            records=[record],
            plan={},
        )


def test_audit_proxy_logs_verifies_ready_start_and_stop_contract(
    tmp_path: Path,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor

    qa_root = tmp_path / "qa"
    invocation = qa_root / "proxy" / "invocation-0001"
    invocation.mkdir(parents=True)
    ready_path = invocation / "ready.json"
    log_path = invocation / "requests.jsonl"
    process_log_path = invocation / "process.log"
    upstream = "http://127.0.0.1:8205"
    upstream_sha = "a" * 64
    base_url = "http://127.0.0.1:19001/v1"
    ready = {
        "run_id": "qa-run",
        "pid": 12345,
        "port": 19001,
        "base_url": base_url,
        "upstream": upstream,
        "upstream_code_sha256": upstream_sha,
        "log": str(log_path.resolve()),
        "started_at": "2026-07-19T00:00:00+00:00",
    }
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    process_log_path.write_text("", encoding="utf-8")
    start = {
        "run_id": "qa-run",
        "pid": 12345,
        "base_url": base_url,
        "upstream": upstream,
        "upstream_code_sha256": upstream_sha,
        "ready_path": str(ready_path.resolve()),
        "log_path": str(log_path.resolve()),
        "process_log_path": str(process_log_path.resolve()),
        "started_at": ready["started_at"],
        "health": {
            "status": "ok",
            "run_id": "qa-run",
            "exclusive_log": str(log_path.resolve()),
            "upstream": upstream,
            "upstream_code_sha256": upstream_sha,
        },
    }
    (invocation / "start.json").write_text(
        json.dumps(start), encoding="utf-8"
    )
    wrapper_sha = auditor.build_runner.sha256_file(
        auditor.qa_runner.CONTROLLED_QA_PROXY
    )
    stop = {
        **start,
        "finished_at": "2026-07-19T00:01:00+00:00",
        "returncode": -15,
        "wrapper_sha256": wrapper_sha,
    }
    stop_path = invocation / "stop.json"
    stop_path.write_text(json.dumps(stop), encoding="utf-8")
    event = {
        "run_id": "qa-run",
        "status": "success",
        "http_status": 200,
        "requested_model": "gpt-5.5",
        "actual_model": "gpt-5.5",
        "response_id": "response-1",
        "event_id": "event-1",
        "question_id": "question-1",
        "logical_call_id": "logical-1",
        "request_sha256": "b" * 64,
        "response_sha256": "c" * 64,
        "client_http_attempts": 1,
        "child_upstream_http_attempts": 1,
        "upstream_http_attempts": 1,
        "unsupported_parameters": [],
        "ignored_client_parameters": [],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        "error": None,
    }
    log_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    plan = {
        "subscription_upstream": {
            "origin": upstream,
            "code_sha256": upstream_sha,
        },
        "source_hashes": {"controlled_qa_proxy": wrapper_sha},
    }

    report = auditor.audit_proxy_logs(
        qa_root=qa_root,
        qa_run_id="qa-run",
        plan=plan,
    )
    assert report["verified_invocations"] == 1

    stop["upstream"] = "http://127.0.0.1:9999"
    stop_path.write_text(json.dumps(stop), encoding="utf-8")
    with pytest.raises(auditor.QAAuditError, match="proxy invocation contract differs"):
        auditor.audit_proxy_logs(
            qa_root=qa_root,
            qa_run_id="qa-run",
            plan=plan,
        )


def _proxy_generation(
    *, origin: str, code_sha256: str, wrapper_sha256: str
) -> dict[str, str]:
    return {
        "origin": origin,
        "code_sha256": code_sha256,
        "wrapper_sha256": wrapper_sha256,
    }


def _proxy_plan(
    *,
    current: dict[str, str],
    accepted: list[dict[str, str]],
    recovered_invocations: list[str] | None = None,
    superseded_attempt_archives: list[dict[str, object]] | None = None,
    unbound_recovered_proxy_events: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "subscription_upstream": {
            "origin": current["origin"],
            "code_sha256": current["code_sha256"],
        },
        "source_hashes": {
            "controlled_qa_proxy": current["wrapper_sha256"],
        },
        "execution_recovery": {
            "schema_version": 1,
            "posthoc": True,
            "accepted_proxy_generations": accepted,
            "recovered_invocations_without_wrapper_sha": (
                recovered_invocations or []
            ),
            "superseded_attempt_archives": superseded_attempt_archives or [],
            "unbound_recovered_proxy_events": (
                unbound_recovered_proxy_events or []
            ),
        },
    }


def _write_proxy_invocation_metadata(
    *,
    qa_root: Path,
    invocation_name: str,
    origin: str,
    code_sha256: str,
    wrapper_sha256: str | None,
) -> None:
    invocation = qa_root / "proxy" / invocation_name
    invocation.mkdir(parents=True)
    ready_path = invocation / "ready.json"
    log_path = invocation / "requests.jsonl"
    process_log_path = invocation / "process.log"
    base_url = "http://127.0.0.1:19001/v1"
    ready = {
        "run_id": "qa-run",
        "pid": 12345,
        "port": 19001,
        "base_url": base_url,
        "upstream": origin,
        "upstream_code_sha256": code_sha256,
        "log": str(log_path.resolve()),
        "started_at": "2026-07-19T00:00:00+00:00",
    }
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    process_log_path.write_text("", encoding="utf-8")
    start = {
        "run_id": "qa-run",
        "pid": 12345,
        "base_url": base_url,
        "upstream": origin,
        "upstream_code_sha256": code_sha256,
        "ready_path": str(ready_path.resolve()),
        "log_path": str(log_path.resolve()),
        "process_log_path": str(process_log_path.resolve()),
        "started_at": ready["started_at"],
        "health": {
            "status": "ok",
            "run_id": "qa-run",
            "exclusive_log": str(log_path.resolve()),
            "upstream": origin,
            "upstream_code_sha256": code_sha256,
        },
    }
    (invocation / "start.json").write_text(
        json.dumps(start), encoding="utf-8"
    )
    event = {
        "run_id": "qa-run",
        "status": "success",
        "http_status": 200,
        "requested_model": "gpt-5.5",
        "actual_model": "gpt-5.5",
        "response_id": "response-1",
        "event_id": "event-1",
        "question_id": "question-1",
        "logical_call_id": "logical-1",
        "request_sha256": "b" * 64,
        "response_sha256": "c" * 64,
        "client_http_attempts": 1,
        "child_upstream_http_attempts": 1,
        "upstream_http_attempts": 1,
        "unsupported_parameters": [],
        "ignored_client_parameters": [],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        "error": None,
    }
    log_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    if wrapper_sha256 is not None:
        stop = {
            **start,
            "finished_at": "2026-07-19T00:01:00+00:00",
            "returncode": -15,
            "wrapper_sha256": wrapper_sha256,
        }
        (invocation / "stop.json").write_text(
            json.dumps(stop), encoding="utf-8"
        )
        return
    recovery = {
        "run_id": "qa-run",
        "pid": 12345,
        "ready_path": str(ready_path.resolve()),
        "ready_sha256": hashlib.sha256(ready_path.read_bytes()).hexdigest(),
        "recovered_at": "2026-07-19T00:01:00+00:00",
        "status": "terminated_stale_proxy",
    }
    (invocation / "recovery.json").write_text(
        json.dumps(recovery), encoding="utf-8"
    )


def test_audit_proxy_logs_accepts_explicit_historical_proxy_generation(
    tmp_path: Path,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor

    current = _proxy_generation(
        origin="http://127.0.0.1:8206",
        code_sha256="2" * 64,
        wrapper_sha256=auditor.build_runner.sha256_file(
            auditor.qa_runner.CONTROLLED_QA_PROXY
        ),
    )
    historical = _proxy_generation(
        origin="http://127.0.0.1:8205",
        code_sha256="1" * 64,
        wrapper_sha256="a" * 64,
    )
    qa_root = tmp_path / "qa"
    _write_proxy_invocation_metadata(
        qa_root=qa_root,
        invocation_name="invocation-0001",
        origin=historical["origin"],
        code_sha256=historical["code_sha256"],
        wrapper_sha256=historical["wrapper_sha256"],
    )

    report = auditor.audit_proxy_logs(
        qa_root=qa_root,
        qa_run_id="qa-run",
        plan=_proxy_plan(current=current, accepted=[historical, current]),
    )

    assert report["verified_invocations"] == 1


@pytest.mark.parametrize("generation_kind", ["undeclared", "cross-spliced"])
def test_audit_proxy_logs_rejects_invalid_proxy_generation(
    tmp_path: Path,
    generation_kind: str,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor

    current = _proxy_generation(
        origin="http://127.0.0.1:8206",
        code_sha256="2" * 64,
        wrapper_sha256=auditor.build_runner.sha256_file(
            auditor.qa_runner.CONTROLLED_QA_PROXY
        ),
    )
    historical = _proxy_generation(
        origin="http://127.0.0.1:8205",
        code_sha256="1" * 64,
        wrapper_sha256="a" * 64,
    )
    observed = (
        _proxy_generation(
            origin="http://127.0.0.1:8207",
            code_sha256="3" * 64,
            wrapper_sha256="b" * 64,
        )
        if generation_kind == "undeclared"
        else _proxy_generation(
            origin=historical["origin"],
            code_sha256=historical["code_sha256"],
            wrapper_sha256=current["wrapper_sha256"],
        )
    )
    qa_root = tmp_path / "qa"
    _write_proxy_invocation_metadata(
        qa_root=qa_root,
        invocation_name="invocation-0001",
        origin=observed["origin"],
        code_sha256=observed["code_sha256"],
        wrapper_sha256=observed["wrapper_sha256"],
    )

    with pytest.raises(auditor.QAAuditError):
        auditor.audit_proxy_logs(
            qa_root=qa_root,
            qa_run_id="qa-run",
            plan=_proxy_plan(current=current, accepted=[historical, current]),
        )


def test_audit_proxy_logs_requires_current_generation_in_recovery_contract(
    tmp_path: Path,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor

    current = _proxy_generation(
        origin="http://127.0.0.1:8206",
        code_sha256="2" * 64,
        wrapper_sha256=auditor.build_runner.sha256_file(
            auditor.qa_runner.CONTROLLED_QA_PROXY
        ),
    )
    historical = _proxy_generation(
        origin="http://127.0.0.1:8205",
        code_sha256="1" * 64,
        wrapper_sha256="a" * 64,
    )
    qa_root = tmp_path / "qa"
    _write_proxy_invocation_metadata(
        qa_root=qa_root,
        invocation_name="invocation-0001",
        origin=current["origin"],
        code_sha256=current["code_sha256"],
        wrapper_sha256=current["wrapper_sha256"],
    )

    with pytest.raises(auditor.QAAuditError):
        auditor.audit_proxy_logs(
            qa_root=qa_root,
            qa_run_id="qa-run",
            plan=_proxy_plan(current=current, accepted=[historical]),
        )


@pytest.mark.parametrize("recovery_kind", ["unlisted", "cross-spliced"])
def test_audit_proxy_logs_rejects_invalid_recovered_proxy_gap(
    tmp_path: Path,
    recovery_kind: str,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor

    current = _proxy_generation(
        origin="http://127.0.0.1:8206",
        code_sha256="2" * 64,
        wrapper_sha256=auditor.build_runner.sha256_file(
            auditor.qa_runner.CONTROLLED_QA_PROXY
        ),
    )
    historical = _proxy_generation(
        origin="http://127.0.0.1:8205",
        code_sha256="1" * 64,
        wrapper_sha256="a" * 64,
    )
    observed_origin = (
        current["origin"]
        if recovery_kind == "unlisted"
        else historical["origin"]
    )
    observed_code_sha = current["code_sha256"]
    recovered_invocations = (
        [] if recovery_kind == "unlisted" else ["invocation-0005"]
    )
    qa_root = tmp_path / "qa"
    _write_proxy_invocation_metadata(
        qa_root=qa_root,
        invocation_name="invocation-0005",
        origin=observed_origin,
        code_sha256=observed_code_sha,
        wrapper_sha256=None,
    )

    with pytest.raises(auditor.QAAuditError):
        auditor.audit_proxy_logs(
            qa_root=qa_root,
            qa_run_id="qa-run",
            plan=_proxy_plan(
                current=current,
                accepted=[historical, current],
                recovered_invocations=recovered_invocations,
            ),
        )


def test_audit_proxy_logs_reports_declared_recovered_wrapper_sha_gap(
    tmp_path: Path,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor

    current = _proxy_generation(
        origin="http://127.0.0.1:8206",
        code_sha256="2" * 64,
        wrapper_sha256=auditor.build_runner.sha256_file(
            auditor.qa_runner.CONTROLLED_QA_PROXY
        ),
    )
    historical = _proxy_generation(
        origin="http://127.0.0.1:8205",
        code_sha256="1" * 64,
        wrapper_sha256="a" * 64,
    )
    qa_root = tmp_path / "qa"
    _write_proxy_invocation_metadata(
        qa_root=qa_root,
        invocation_name="invocation-0005",
        origin=historical["origin"],
        code_sha256=historical["code_sha256"],
        wrapper_sha256=None,
    )

    report = auditor.audit_proxy_logs(
        qa_root=qa_root,
        qa_run_id="qa-run",
        plan=_proxy_plan(
            current=current,
            accepted=[historical, current],
            recovered_invocations=["invocation-0005"],
        ),
    )

    assert report["verified_invocations"] == 1
    assert report["recovered_invocations_without_wrapper_sha_count"] == 1


def _test_directory_descriptor_sha256(root: Path) -> str:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append(
            {
                "byte_count": path.stat().st_size,
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_execution_recovery_binds_superseded_proxy_event_to_archived_ledger(
    tmp_path: Path,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor

    qa_root = tmp_path / "campaign" / "sol-r5"
    archive = (
        tmp_path
        / "campaign/retry-archives/sol/snapshot/sol/beam/questions/q/attempt-0001"
    )
    archive.mkdir(parents=True)
    archived_event = {
        "event_id": "subscription-qa-archived",
        "status": "success",
        "logical_call_id": "qa-run:attempt-0001:retrieval:q:call-0001",
        "question_id": "q",
    }
    (archive / "retrieval_model_ledger.jsonl").write_text(
        json.dumps({"proxy_evidence": {"events": [archived_event]}}) + "\n",
        encoding="utf-8",
    )
    (archive / "attempt_manifest.json").write_text(
        json.dumps({"execution_run_id": "qa-run:attempt-0001"}),
        encoding="utf-8",
    )
    current = _proxy_generation(
        origin="http://127.0.0.1:8206",
        code_sha256="2" * 64,
        wrapper_sha256=auditor.build_runner.sha256_file(
            auditor.qa_runner.CONTROLLED_QA_PROXY
        ),
    )
    archive_contract = {
        "path": archive.relative_to(qa_root.parent).as_posix(),
        "descriptor_sha256": _test_directory_descriptor_sha256(archive),
        "proxy_event_ids": [archived_event["event_id"]],
    }
    plan = _proxy_plan(
        current=current,
        accepted=[current],
        superseded_attempt_archives=[archive_contract],
    )

    superseded, unbound = auditor._execution_recovery_event_contract(  # noqa: SLF001
        qa_root=qa_root,
        plan=plan,
    )

    assert superseded == {archived_event["event_id"]}
    assert unbound == {}

    archive_contract["descriptor_sha256"] = "0" * 64
    with pytest.raises(auditor.QAAuditError, match="archive descriptor"):
        auditor._execution_recovery_event_contract(  # noqa: SLF001
            qa_root=qa_root,
            plan=plan,
        )


def test_audit_proxy_logs_accepts_explicit_unbound_recovered_success(
    tmp_path: Path,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor

    current = _proxy_generation(
        origin="http://127.0.0.1:8206",
        code_sha256="2" * 64,
        wrapper_sha256=auditor.build_runner.sha256_file(
            auditor.qa_runner.CONTROLLED_QA_PROXY
        ),
    )
    qa_root = tmp_path / "qa"
    _write_proxy_invocation_metadata(
        qa_root=qa_root,
        invocation_name="invocation-0005",
        origin=current["origin"],
        code_sha256=current["code_sha256"],
        wrapper_sha256=None,
    )
    plan = _proxy_plan(
        current=current,
        accepted=[current],
        recovered_invocations=["invocation-0005"],
        unbound_recovered_proxy_events=[
            {
                "event_id": "event-1",
                "invocation": "invocation-0005",
                "status": "success",
            }
        ],
    )

    report = auditor.audit_proxy_logs(
        qa_root=qa_root,
        qa_run_id="qa-run",
        plan=plan,
    )

    assert report["unbound_recovered_proxy_events"] == 1


def test_root_audit_counts_only_accepted_successes_as_logical_results(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import run_gpt56_chunk_curve_qa as runner

    qa_root = tmp_path / "qa"
    qa_root.mkdir()
    plan = {
        "protocol_id": contract.PROTOCOL_ID,
        "run_ids": ["build-1"],
        "question_count": 1,
    }
    preregistration = qa_root / "preregistration.json"
    preregistration.write_text(json.dumps(plan), encoding="utf-8")
    evaluator_paths = {"terra": {}}
    evaluator_hashes = {"terra": {}}
    completion = {
        "status": "complete",
        "protocol_id": contract.PROTOCOL_ID,
        "qa_run_id": "qa-run",
        "run_ids": ["build-1"],
        "question_count": 1,
        "preregistration_sha256": auditor.build_runner.sha256_file(
            preregistration
        ),
        "evaluator_inputs": evaluator_paths,
        "evaluator_input_sha256": evaluator_hashes,
    }
    (qa_root / "completion.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    records = [{"question_id": "unit::q0"}]
    binding = contract.RunBinding("build-1", {"run_id": "build-1"}, tmp_path)
    validated = {"unit": {"run_id": "build-1"}, "questions": []}
    monkeypatch.setattr(
        runner, "execute_pending_questions", lambda **kwargs: records
    )
    monkeypatch.setattr(
        auditor,
        "audit_question_records",
        lambda **kwargs: {
            "question_count": 1,
            "artifact_count": 5,
            "all_memory_unchanged": True,
            "visible_tokens": 1,
            "retrieval_model_calls": 1,
            "answer_model_calls": 1,
        },
    )

    def fake_proxy_audit(**kwargs):
        assert kwargs["records"] is records
        return {
            "response_ids": 3,
            "accepted_successes": 2,
            "abandoned_successes": 1,
            "all_successes_accounted": True,
        }

    monkeypatch.setattr(auditor, "audit_proxy_logs", fake_proxy_audit)
    monkeypatch.setattr(
        auditor,
        "_audit_evaluator_inputs",
        lambda **kwargs: {"paths": evaluator_paths, "sha256": evaluator_hashes},
    )
    monkeypatch.setattr(
        auditor.answer_contract, "formal_token_counter", lambda: object()
    )

    report = auditor.audit_qa_root(
        plan=plan,
        bindings=[binding],
        validated_units=[validated],
        qa_root=qa_root,
        formal=True,
    )
    assert report["status"] == "verified_complete"
    assert report["proxy"]["abandoned_successes"] == 1


def test_preflight_audit_reconstructs_one_question_per_benchmark(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor
    from scripts import gpt56_chunk_curve_qa_contract as contract
    from scripts import run_gpt56_chunk_curve_qa as runner

    qa_root = tmp_path / "qa"
    qa_root.mkdir()
    plan = {
        "protocol_id": contract.PROTOCOL_ID,
        "run_ids": ["locomo-run", "beam-run"],
        "question_count": 4,
    }
    preregistration = qa_root / "preregistration.json"
    preregistration.write_text(json.dumps(plan), encoding="utf-8")
    bindings = [
        contract.RunBinding(
            "locomo-run",
            {"tier": "terra", "benchmark": "locomo"},
            tmp_path / "locomo",
        ),
        contract.RunBinding(
            "beam-run",
            {"tier": "terra", "benchmark": "beam-100k"},
            tmp_path / "beam",
        ),
    ]
    validated = [
        {
            "unit": {"run_id": "locomo-run"},
            "questions": [{"question_id": "l0"}, {"question_id": "l1"}],
        },
        {
            "unit": {"run_id": "beam-run"},
            "questions": [{"question_id": "b0"}, {"question_id": "b1"}],
        },
    ]
    records = [
        {"question_id": "locomo::l0", "answer": "alpha"},
        {"question_id": "beam::b0", "answer": "beta"},
    ]
    preflight = {
        "status": "complete",
        "protocol_id": contract.PROTOCOL_ID,
        "qa_run_id": "qa-preflight",
        "question_count": 2,
        "question_ids": [record["question_id"] for record in records],
        "answer_sha256": {
            record["question_id"]: hashlib.sha256(
                record["answer"].encode("utf-8")
            ).hexdigest()
            for record in records
        },
        "preregistration_sha256": auditor.build_runner.sha256_file(
            preregistration
        ),
    }
    (qa_root / "preflight.json").write_text(
        json.dumps(preflight), encoding="utf-8"
    )
    captured: dict[str, object] = {}

    def fake_execute(**kwargs):
        captured["validated_units"] = kwargs["validated_units"]
        return records

    monkeypatch.setattr(runner, "execute_pending_questions", fake_execute)
    monkeypatch.setattr(
        auditor,
        "audit_question_records",
        lambda **kwargs: {
            "question_count": 2,
            "artifact_count": 10,
            "retrieval_model_calls": 2,
            "answer_model_calls": 2,
        },
    )
    monkeypatch.setattr(
        auditor,
        "audit_proxy_logs",
        lambda **kwargs: {
            "accepted_successes": 4,
            "all_successes_accounted": True,
        },
    )
    monkeypatch.setattr(
        auditor.answer_contract, "formal_token_counter", lambda: object()
    )

    report = auditor.audit_preflight_root(
        plan=plan,
        bindings=bindings,
        validated_units=validated,
        qa_root=qa_root,
        formal=True,
    )

    assert report["status"] == "verified_preflight"
    assert report["question_count"] == 2
    assert [
        unit["questions"] for unit in captured["validated_units"]
    ] == [[{"question_id": "l0"}], [{"question_id": "b0"}]]


def test_audit_scope_rejects_an_incomplete_formal_run_set(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor
    from scripts import gpt56_chunk_curve_qa_contract as contract

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
    monkeypatch.setattr(
        auditor.contract,
        "select_completed_runs",
        lambda **kwargs: [binding],
    )
    monkeypatch.setattr(
        auditor.contract,
        "validate_unit_binding",
        lambda value: {
            "unit": {"run_id": "run-1"},
            "questions": [{"question_id": "q0"}],
        },
    )
    monkeypatch.setattr(
        auditor.contract,
        "build_preregistration",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("plan built before formal scope validation")
        ),
    )

    with pytest.raises(auditor.qa_runner.QARunnerError, match="exactly six"):
        auditor.audit_scope(
            matrix_path=tmp_path / "matrix.jsonl",
            build_root=tmp_path / "builds",
            qa_root=tmp_path / "qa",
            run_ids=["run-1"],
        )


def test_auditor_reads_the_frozen_upstream_from_preregistration(
    tmp_path: Path,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor

    qa_root = tmp_path / "qa"
    qa_root.mkdir()
    preregistration = qa_root / "preregistration.json"
    preregistration.write_text(
        json.dumps(
            {
                "subscription_upstream": {
                    "origin": "http://127.0.0.1:8205",
                    "code_sha256": "a" * 64,
                }
            }
        ),
        encoding="utf-8",
    )

    assert auditor._frozen_upstream(qa_root) == (  # noqa: SLF001
        "http://127.0.0.1:8205",
        "a" * 64,
    )

    preregistration.write_text(
        json.dumps(
            {
                "subscription_upstream": {
                    "origin": "http://127.0.0.1:8205",
                    "code_sha256": "invalid",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(auditor.QAAuditError, match="upstream contract"):
        auditor._frozen_upstream(qa_root)  # noqa: SLF001


def test_auditor_reads_the_posthoc_execution_recovery_from_preregistration(
    tmp_path: Path,
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor

    qa_root = tmp_path / "qa"
    qa_root.mkdir()
    recovery = {
        "schema_version": 1,
        "posthoc": True,
        "accepted_proxy_generations": [
            {
                "origin": "http://127.0.0.1:8206",
                "code_sha256": "a" * 64,
                "wrapper_sha256": "b" * 64,
            }
        ],
        "recovered_invocations_without_wrapper_sha": ["invocation-0005"],
    }
    (qa_root / "preregistration.json").write_text(
        json.dumps({"execution_recovery": recovery}), encoding="utf-8"
    )

    assert auditor._frozen_execution_recovery(qa_root) == recovery  # noqa: SLF001


def test_main_writes_and_validates_no_clobber_audit(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor

    report = {
        "schema_version": 1,
        "status": "verified_complete",
        "question_count": 356,
        "artifact_count": 1_424,
    }
    monkeypatch.setattr(auditor, "audit_scope", lambda **kwargs: report)
    output = tmp_path / "audit.json"
    argv = [
        "--build-root",
        str(tmp_path / "builds"),
        "--qa-root",
        str(tmp_path / "qa"),
        "--run-ids",
        "run-1",
        "--output",
        str(output),
    ]

    assert auditor.main(argv) == 0
    assert json.loads(output.read_text()) == report
    assert auditor.main(argv) == 0


def test_main_routes_and_publishes_preflight_audit(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts import audit_gpt56_chunk_curve_qa as auditor

    report = {
        "schema_version": 1,
        "status": "verified_preflight",
        "question_count": 3,
        "artifact_count": 15,
    }

    def fake_audit_scope(**kwargs):
        assert kwargs["preflight_only"] is True
        return report

    monkeypatch.setattr(auditor, "audit_scope", fake_audit_scope)
    qa_root = tmp_path / "qa"
    argv = [
        "--build-root",
        str(tmp_path / "builds"),
        "--qa-root",
        str(qa_root),
        "--run-ids",
        "run-1",
        "--preflight-only",
    ]

    assert auditor.main(argv) == 0
    output = qa_root / "preflight-audit.json"
    assert json.loads(output.read_text()) == report
    assert auditor.main(argv) == 0
