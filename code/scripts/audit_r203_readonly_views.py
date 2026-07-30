#!/usr/bin/env python3
"""Independently audit all four R203 read-only view/source conditions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_readonly_nativemem_control as generic  # noqa: E402
import controlled_locomo_answer_contract as answer_contract  # noqa: E402
import r203_formal_contract as formal_contract  # noqa: E402
import readonly_nativemem_control as readonly_control  # noqa: E402
from src.evaluation.durable_model_ledger import DurableLedgerError  # noqa: E402
from src.evaluation.durable_model_ledger import read_ledger  # noqa: E402
from src import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402


RUN_SCHEMA = "nativemem.r203-readonly-views-run.v1"
FORMAL_RUN_SCHEMA = "nativemem.r203-formal-run.v2"
FORMAL_CHECKPOINT_SCHEMA = "nativemem.r203-formal-question-checkpoint.v1"
FORMAL_COMPLETE_SCHEMA = "nativemem.r203-formal-complete.v1"
FORMAL_AUDIT_SCHEMA = "nativemem.r203-formal-audit.v2"
METHOD = "r203_readonly_views"
CONDITIONS = (
    "dual_source",
    "topic_source",
    "timeline_source",
    "dual_no_source",
)
SOURCE_PATHS = (
    "scripts/r203_formal_contract.py",
    "scripts/readonly_nativemem_control.py",
    "scripts/audit_readonly_nativemem_control.py",
    "scripts/run_r203_readonly_views.py",
    "scripts/audit_r203_readonly_views.py",
    "scripts/controlled_locomo_answer_contract.py",
    "scripts/run_controlled_locomo_answers.py",
    "src/evaluation/durable_model_ledger.py",
    "src/evaluation/visible_token_budget.py",
    "src/evaluation/visible_token_audit.py",
)


class R203AuditError(RuntimeError):
    pass


def _inside_path(root: Path, value: Any, *, label: str, directory: bool) -> Path:
    if not isinstance(value, str) or not value:
        raise R203AuditError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise R203AuditError(f"{label} path escapes output")
    candidate = root / relative
    answer_contract.reject_symlink_components(candidate)
    valid = candidate.is_dir() if directory else candidate.is_file()
    if candidate.is_symlink() or not valid:
        raise R203AuditError(f"{label} is not a regular path")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise R203AuditError(f"{label} path escapes output") from exc
    return resolved


def _audit_synthetic(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().absolute()
    answer_contract.reject_symlink_components(output_dir)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise R203AuditError("R203 output root is invalid")
    output_dir = output_dir.resolve()
    manifest = answer_contract.read_json(output_dir / "run_manifest.json")
    complete = answer_contract.read_json(output_dir / "complete.json")
    condition_records = manifest.get("conditions") if isinstance(manifest, dict) else None
    condition_ids = (
        [item.get("id") for item in condition_records]
        if isinstance(condition_records, list)
        and all(isinstance(item, dict) for item in condition_records)
        else None
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != RUN_SCHEMA
        or manifest.get("mode") != "synthetic_no_network"
        or manifest.get("method") != METHOD
        or condition_ids != list(CONDITIONS)
        or manifest.get("scope")
        != {
            "samples": [0],
            "questions_per_condition": 1,
            "formal_all_ten_required": True,
            "formal": False,
        }
        or manifest.get("config", {}).get("budget_tokens") != 20_000
        or manifest.get("config", {}).get("model") != "gpt-5.5"
        or manifest.get("config", {}).get("answerer") != "gpt-5.5"
        or manifest.get("config", {}).get("network_requests") != 0
        or manifest.get("config", {}).get("retrieval_read_only") is not True
    ):
        raise R203AuditError("R203 run manifest differs")
    generic.audit_current_source_hashes(
        manifest.get("source_hashes"), expected_paths=SOURCE_PATHS
    )
    if (
        not isinstance(complete, dict)
        or complete.get("schema_version") != RUN_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("run_id") != manifest.get("run_id")
        or complete.get("network_requests") != 0
        or complete.get("condition_count") != 4
        or complete.get("questions_per_condition") != 1
    ):
        raise R203AuditError("R203 complete manifest differs")
    question = manifest.get("question")
    if (
        not isinstance(question, dict)
        or question.get("question_id") != "s0_q0"
        or not isinstance(question.get("question"), str)
        or not isinstance(question.get("gold_source_ids"), list)
        or not all(
            isinstance(value, str) and value for value in question["gold_source_ids"]
        )
        or not isinstance(question.get("source_recall_eligible"), bool)
    ):
        raise R203AuditError("R203 question is invalid")
    conversation = manifest.get("conversation")
    if not isinstance(conversation, dict):
        raise R203AuditError("R203 conversation is absent")
    turn_index = generic.build_turn_index(conversation)
    provenance_record = manifest.get("stage_evidence_provenance")
    if not isinstance(provenance_record, dict):
        raise R203AuditError("R203 stage provenance is absent")
    if provenance_record.get("path") != "stage_evidence_provenance.json":
        raise R203AuditError("R203 stage provenance path differs")
    provenance_path = _inside_path(
        output_dir,
        provenance_record.get("path"),
        label="R203 stage provenance",
        directory=False,
    )
    if (
        answer_contract.sha256_file(provenance_path)
        != provenance_record.get("sha256")
        or complete.get("stage_evidence_provenance_sha256")
        != provenance_record.get("sha256")
    ):
        raise R203AuditError("R203 stage provenance hash differs")
    stage_provenance = answer_contract.read_json(provenance_path)
    if (
        not isinstance(stage_provenance, dict)
        or answer_contract.canonical_hash(stage_provenance)
        != provenance_record.get("record_sha256")
    ):
        raise R203AuditError("R203 stage provenance record differs")
    source_record = manifest.get("memory_source", {})
    if not isinstance(source_record, dict):
        raise R203AuditError("R203 source-memory record is invalid")
    if source_record.get("path") != "memory_source":
        raise R203AuditError("R203 source-memory path differs")
    source_hash = source_record.get("descriptor", {}).get("sha256")
    if source_hash != complete.get("memory_source_sha256"):
        raise R203AuditError("R203 source-memory hash differs")
    source_memory = _inside_path(
        output_dir,
        source_record.get("path"),
        label="R203 source memory",
        directory=True,
    )
    if generic.snapshot_memory_path(source_memory).descriptor != source_record.get(
        "descriptor"
    ):
        raise R203AuditError("R203 source-memory descriptor differs")
    reports: dict[str, dict[str, Any]] = {}
    copy_hashes: set[str] = set()
    for condition in CONDITIONS:
        condition_record = manifest.get("condition_copies", {}).get(condition)
        if not isinstance(condition_record, dict):
            raise R203AuditError(f"R203 condition copy is absent: {condition}")
        expected_memory_path = f"conditions/{condition}/memory"
        if condition_record.get("path") != expected_memory_path:
            raise R203AuditError("R203 condition copy path differs")
        memory = _inside_path(
            output_dir,
            condition_record.get("path"),
            label="R203 condition copy",
            directory=True,
        )
        live_descriptor = generic.snapshot_memory_path(memory).descriptor
        live_hash = live_descriptor["sha256"]
        if live_descriptor != condition_record.get("descriptor"):
            raise R203AuditError("R203 condition memory changed")
        copy_hashes.add(live_hash)
        checkpoint = answer_contract.read_json(
            output_dir / f"conditions/{condition}/completed/s0_q0.json"
        )
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("status") != "complete"
            or checkpoint.get("condition") != condition
        ):
            raise R203AuditError("R203 condition checkpoint differs")
        expected_artifact_path = (
            f"conditions/{condition}/questions/s0_q0/attempt-0001"
        )
        if checkpoint.get("artifact_dir") != expected_artifact_path:
            raise R203AuditError("R203 question artifact path differs")
        artifact_dir = _inside_path(
            output_dir,
            checkpoint.get("artifact_dir"),
            label="R203 question artifact",
            directory=True,
        )
        result = answer_contract.read_json(artifact_dir / "result.json")
        if (
            answer_contract.sha256_file(artifact_dir / "result.json")
            != checkpoint.get("result_sha256")
        ):
            raise R203AuditError("R203 condition result hash differs")
        if (
            not isinstance(result, dict)
            or result.get("budget", {}).get("tokenizer")
            != manifest.get("config", {}).get("tokenizer")
        ):
            raise R203AuditError("R203 tokenizer linkage differs")
        report = generic.audit_question(
            artifact_dir=artifact_dir,
            memory_root=memory,
            method=METHOD,
            condition=condition,
            question_id=question["question_id"],
            question=question["question"],
            gold_source_ids=question["gold_source_ids"],
            source_recall_eligible=question["source_recall_eligible"],
            formal=False,
            stage_provenance=stage_provenance,
            turn_index=turn_index,
        )
        expected_path = (
            "timeline/2025/02/03.md"
            if condition == "timeline_source"
            else "topics/travel.md"
        )
        if (
            report["mapped_source_recall"] != 1.0
            or report["first_relevant_file"] != expected_path
            or (
                condition == "dual_no_source"
                and report["source_resolution_tokens"] != 0
            )
            or (
                condition != "dual_no_source"
                and report["source_resolution_tokens"] <= 0
            )
        ):
            raise R203AuditError("R203 condition diagnostics differ")
        reports[condition] = report
    if copy_hashes != {source_hash}:
        raise R203AuditError("R203 conditions did not start from identical bytes")
    if reports != complete.get("condition_audits"):
        raise R203AuditError("R203 recorded condition audits differ")
    return {
        "schema_version": "nativemem.r203-readonly-views-audit.v1",
        "status": "passed",
        "mode": "synthetic_no_network",
        "run_id": manifest["run_id"],
        "network_requests": 0,
        "conditions": list(CONDITIONS),
        "condition_count": 4,
        "questions_per_condition": 1,
        "budget_tokens": 20_000,
        "model": "gpt-5.5",
        "byte_identical_start": True,
        "memory_unchanged": True,
        "condition_audits": reports,
        "formal_scope_required": "all_10_locomo_conversations",
        "formal_status": "blocked_waiting_new_all_ten_flex_source",
    }


def _validate_content_hash(payload: Mapping[str, Any], field: str) -> None:
    content = dict(payload)
    expected = content.pop(field, None)
    if expected != answer_contract.canonical_hash(content):
        raise R203AuditError(f"{field} differs")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise R203AuditError(f"JSONL artifact is absent or unsafe: {path}")
    payload = path.read_bytes()
    if not payload.endswith(b"\n"):
        raise R203AuditError(f"JSONL final line is incomplete: {path}")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise R203AuditError(f"invalid JSONL line {number}: {path}") from exc
        if not isinstance(value, dict):
            raise R203AuditError(f"JSONL line {number} is not an object: {path}")
        records.append(value)
    return records


def _formal_question_paths(
    output_dir: Path, condition: str, spec: Mapping[str, Any]
) -> dict[str, Path]:
    root = (
        output_dir
        / "conditions"
        / condition
        / "questions"
        / str(spec["artifact_id"])
    )
    return {
        "root": root,
        "input": root / "input_binding.json",
        "stage": root / "stage_provenance.json",
        "attempt": root / "attempt-0001",
        "question_audit": root / "question_audit.json",
        "checkpoint": root / "checkpoint.json",
    }


def _result_proxy_events(
    *, attempt: Path, result: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reconcile question ledgers with their exclusive-proxy evidence."""

    events: list[dict[str, Any]] = []
    response_ids: list[str] = []
    retrieval = read_ledger(attempt / "retrieval_model_ledger.jsonl")
    starts = {
        str(record["logical_call_id"]): record
        for record in retrieval
        if record.get("event") == "model_call_started"
    }
    finishes = {
        str(record["logical_call_id"]): record
        for record in retrieval
        if record.get("event") == "model_call_finished"
    }
    if set(starts) != set(finishes):
        raise R203AuditError("retrieval model-call lifecycle differs")
    for logical_id, finish in finishes.items():
        response_id = str(finish.get("response_id", ""))
        response_ids.append(response_id)
        evidence = finish.get("proxy_evidence")
        if not isinstance(evidence, Mapping) or evidence.get("mode") != "exclusive_proxy":
            raise R203AuditError("formal retrieval proxy evidence differs")
        rows = evidence.get("events")
        if not isinstance(rows, list) or not rows:
            raise R203AuditError("formal retrieval proxy events are absent")
        for event in rows:
            if not isinstance(event, dict):
                raise R203AuditError("retrieval proxy event is not an object")
            expected = {
                "logical_call_id": logical_id,
                "question_id": result["question_id"],
                "status": "success",
                "requested_model": formal_contract.MODEL,
                "actual_model": formal_contract.MODEL,
                "response_id": response_id,
            }
            if (
                any(event.get(key) != value for key, value in expected.items())
                or event.get("usage") != finish.get("usage")
            ):
                raise R203AuditError("retrieval proxy event linkage differs")
            events.append(event)
    answer = result.get("answer")
    if not isinstance(answer, Mapping):
        raise R203AuditError("formal answer record is absent")
    response_ids.append(str(answer.get("response_id", "")))
    evidence = answer.get("proxy_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("mode") != "exclusive_proxy":
        raise R203AuditError("formal answer proxy evidence differs")
    rows = evidence.get("events")
    if not isinstance(rows, list) or not rows:
        raise R203AuditError("formal answer proxy events are absent")
    logical_id = (
        f"{result['run_id']}:{result['question_id']}:"
        f"{result['condition']}:answer"
    )
    successes = 0
    for event in rows:
        if not isinstance(event, dict):
            raise R203AuditError("answer proxy event is not an object")
        if (
            event.get("logical_call_id") != logical_id
            or event.get("question_id") != result["question_id"]
            or event.get("requested_model") != formal_contract.MODEL
            or event.get("status") not in {"success", "error"}
        ):
            raise R203AuditError("answer proxy event linkage differs")
        if event.get("status") == "success":
            successes += 1
            if (
                event.get("actual_model") != formal_contract.MODEL
                or event.get("response_id") != answer["response_id"]
                or event.get("usage") != answer.get("usage")
            ):
                raise R203AuditError("answer success identity differs")
        events.append(event)
    if successes != 1:
        raise R203AuditError("answer does not have exactly one successful event")
    if any(not value for value in response_ids) or len(response_ids) != len(
        set(response_ids)
    ):
        raise R203AuditError("question response IDs are empty or duplicated")
    return events, response_ids


def audit_formal_question_checkpoint(
    *,
    output_dir: Path,
    condition: str,
    spec: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    copy_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently rebuild one formal R203 question and its M4 evidence."""

    output_dir = output_dir.expanduser().absolute()
    paths = _formal_question_paths(output_dir, condition, spec)
    checkpoint = answer_contract.read_json(paths["checkpoint"])
    if not isinstance(checkpoint, dict):
        raise R203AuditError("formal checkpoint is not an object")
    _validate_content_hash(checkpoint, "checkpoint_content_sha256")
    identity = {
        "schema_version": FORMAL_CHECKPOINT_SCHEMA,
        "status": "complete",
        "condition": condition,
        "question_id": spec["question_id"],
        "artifact_id": spec["artifact_id"],
        "dataset_index": spec["dataset_index"],
        "question_index": spec["question_index"],
        "category": spec["category"],
        "actual_model": formal_contract.MODEL,
        "source_recall_eligible": spec["source_recall_eligible"],
    }
    if any(checkpoint.get(key) != value for key, value in identity.items()):
        raise R203AuditError("formal checkpoint identity differs")
    expected_relative = {
        "input_binding": paths["input"],
        "stage_provenance": paths["stage"],
        "attempt": paths["attempt"],
        "question_audit": paths["question_audit"],
    }
    for key, expected_path in expected_relative.items():
        record = checkpoint.get(key)
        if not isinstance(record, Mapping):
            raise R203AuditError(f"formal checkpoint {key} is absent")
        actual = _inside_path(
            output_dir,
            record.get("path"),
            label=f"formal {key}",
            directory=key == "attempt",
        )
        if actual != expected_path.resolve():
            raise R203AuditError(f"formal {key} path differs")
        if key == "attempt":
            if record.get("result_sha256") != answer_contract.sha256_file(
                actual / "result.json"
            ):
                raise R203AuditError("formal result hash differs")
        elif record.get("sha256") != answer_contract.sha256_file(actual):
            raise R203AuditError(f"formal {key} hash differs")
    memory = formal_contract.condition_memory_path(
        output_dir, condition, int(spec["dataset_index"])
    )
    live_memory = readonly_control.memory_descriptor(memory)
    if checkpoint.get("memory_sha256") != live_memory.get("sha256"):
        raise R203AuditError("formal checkpoint memory hash differs")
    _conversation, turn_index = formal_contract.conversation_and_turn_index(spec)
    expected_stage = formal_contract.build_stage_provenance(
        memory_root=memory,
        spec=spec,
        source_binding=source_binding,
    )
    recorded_stage = answer_contract.read_json(paths["stage"])
    if recorded_stage != expected_stage:
        raise R203AuditError("formal stage provenance differs")
    expected_input = formal_contract.input_binding(
        preregistration=formal_contract.DEFAULT_PREREGISTRATION,
        source_binding=source_binding,
        copy_inventory_sha256=copy_record["inventory_content_sha256"],
        condition=condition,
        spec=spec,
        memory_root=memory,
        turn_index=turn_index,
        stage_provenance=recorded_stage,
    )
    if answer_contract.read_json(paths["input"]) != expected_input:
        raise R203AuditError("formal question input binding differs")
    question_report = generic.audit_question(
        artifact_dir=paths["attempt"],
        memory_root=memory,
        method=METHOD,
        condition=condition,
        question_id=str(spec["question_id"]),
        question=str(spec["question"]),
        gold_source_ids=list(spec["gold_source_ids"]),
        source_recall_eligible=bool(spec["source_recall_eligible"]),
        formal=True,
        stage_provenance=recorded_stage,
        turn_index=turn_index,
    )
    if answer_contract.read_json(paths["question_audit"]) != question_report:
        raise R203AuditError("recorded formal question audit differs")
    result = answer_contract.read_json(paths["attempt"] / "result.json")
    if not isinstance(result, dict):
        raise R203AuditError("formal question result is invalid")
    events, response_ids = _result_proxy_events(
        attempt=paths["attempt"], result=result
    )
    if checkpoint.get("response_ids") != response_ids:
        raise R203AuditError("formal checkpoint response IDs differ")
    return {
        "status": "passed",
        "condition": condition,
        "question_id": spec["question_id"],
        "artifact_id": spec["artifact_id"],
        "dataset_index": spec["dataset_index"],
        "category": spec["category"],
        "source_recall_eligible": spec["source_recall_eligible"],
        "mapped_source_recall": question_report["mapped_source_recall"],
        "visible_tokens": question_report["visible_tokens"],
        "source_resolution_tokens": question_report[
            "source_resolution_tokens"
        ],
        "retrieval_model_calls": question_report["retrieval_model_calls"],
        "response_ids": response_ids,
        "proxy_events": events,
        "stage_evidence": question_report["stage_evidence"],
    }


def _audit_formal_proxy_inventory(
    output_dir: Path,
    *,
    run_id: str,
    expected_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    proxy_root = output_dir / "proxy"
    if proxy_root.is_symlink() or not proxy_root.is_dir():
        raise R203AuditError("formal proxy root is absent")
    live_events: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    invocations = 0
    for directory in sorted(proxy_root.glob("invocation-*")):
        if directory.is_symlink() or not directory.is_dir():
            raise R203AuditError("formal proxy invocation is unsafe")
        manifest = answer_contract.read_json(directory / "manifest.json")
        if (
            not isinstance(manifest, dict)
            or manifest.get("run_id") != run_id
            or manifest.get("invocation_id") != directory.name
            or manifest.get("returncode") is None
            or manifest.get("provider_window_error") is not None
        ):
            raise R203AuditError("formal proxy invocation manifest differs")
        log = _inside_path(
            output_dir,
            manifest.get("log"),
            label="formal proxy log",
            directory=False,
        )
        if (
            manifest.get("log_sha256") != answer_contract.sha256_file(log)
            or manifest.get("log_bytes") != log.stat().st_size
        ):
            raise R203AuditError("formal proxy log binding differs")
        rows = _read_jsonl(log)
        try:
            windows.append(
                flex_evidence.audit_window(
                    manifest.get("provider_window"), consumer_records=rows
                )
            )
        except flex_evidence.EvidenceError as exc:
            raise R203AuditError(f"formal Flex provider window differs: {exc}") from exc
        live_events.extend(rows)
        invocations += 1
    if invocations < 1:
        raise R203AuditError("formal run has no closed proxy invocation")
    expected = list(expected_events)
    if Counter(str(row.get("event_id")) for row in live_events) != Counter(
        str(row.get("event_id")) for row in expected
    ):
        raise R203AuditError("formal proxy contains orphan or missing events")
    expected_by_id = {str(row.get("event_id")): row for row in expected}
    if len(expected_by_id) != len(expected):
        raise R203AuditError("formal expected proxy event IDs are duplicated")
    if any(expected_by_id.get(str(row.get("event_id"))) != row for row in live_events):
        raise R203AuditError("formal proxy event content differs")
    request_ids = [
        str(row.get("gateway_request_id"))
        for row in live_events
        if row.get("gateway_request_id") is not None
    ]
    if len(request_ids) != len(live_events) or len(request_ids) != len(
        set(request_ids)
    ):
        raise R203AuditError("formal gateway request IDs are absent or duplicated")
    return {
        "invocations": invocations,
        "events": len(live_events),
        "gateway_request_ids": len(request_ids),
        "upstream_http_attempts": sum(
            int(row.get("upstream_http_attempts", 0)) for row in live_events
        ),
        "actual_models": sorted({row.get("actual_model") for row in live_events}),
        "provider_windows": windows,
        "orphan_events": 0,
    }


def _audit_formal_outputs(
    output_dir: Path,
    *,
    specs: Sequence[Mapping[str, Any]],
    complete: Mapping[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    hypotheses: dict[str, list[dict[str, str]]] = {
        condition: [] for condition in formal_contract.CONDITIONS
    }
    for condition in formal_contract.CONDITIONS:
        for spec in specs:
            paths = _formal_question_paths(output_dir, condition, spec)
            result = answer_contract.read_json(paths["attempt"] / "result.json")
            record = {
                "condition": condition,
                "question_id": spec["question_id"],
                "artifact_id": spec["artifact_id"],
                "dataset_index": spec["dataset_index"],
                "question_index": spec["question_index"],
                "category": spec["category"],
                "question": spec["question"],
                "answer": result["answer"]["text"],
                "answer_response_id": result["answer"]["response_id"],
                "answer_response_model": result["answer"]["response_model"],
                "answer_usage": result["answer"]["usage"],
                "retrieval_model_calls": result["retrieval"]["model_calls"],
                "visible_tokens": result["budget"]["visible_tokens"],
                "source_resolution_tokens": result["budget"][
                    "source_resolution_tokens"
                ],
                "diagnostics": result["diagnostics"],
                "stage_evidence": result["stage_evidence"],
            }
            records.append(record)
            hypotheses[condition].append(
                {
                    "question_id": str(spec["question_id"]),
                    "hypothesis": str(result["answer"]["text"]),
                }
            )
    outputs = complete.get("outputs")
    if not isinstance(outputs, Mapping):
        raise R203AuditError("formal aggregate output binding is absent")
    results = _inside_path(
        output_dir,
        outputs.get("results", {}).get("path"),
        label="formal results",
        directory=False,
    )
    if (
        results != (output_dir / "results.json").resolve()
        or outputs["results"].get("sha256") != answer_contract.sha256_file(results)
        or answer_contract.read_json(results)
        != {"schema_version": FORMAL_RUN_SCHEMA, "records": records}
    ):
        raise R203AuditError("formal aggregate results differ")
    for condition, expected in hypotheses.items():
        record = outputs.get("hypotheses", {}).get(condition)
        if not isinstance(record, Mapping):
            raise R203AuditError("formal hypothesis binding is absent")
        path = _inside_path(
            output_dir,
            record.get("path"),
            label="formal hypotheses",
            directory=False,
        )
        if (
            path != (output_dir / "hypotheses" / f"{condition}.jsonl").resolve()
            or record.get("sha256") != answer_contract.sha256_file(path)
            or _read_jsonl(path) != expected
        ):
            raise R203AuditError("formal hypotheses differ")
    return {
        "results_sha256": answer_contract.sha256_file(results),
        "hypothesis_files": len(hypotheses),
    }


def _audit_formal(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().absolute()
    answer_contract.reject_symlink_components(output_dir)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise R203AuditError("formal R203 output root is invalid")
    manifest = answer_contract.read_json(output_dir / "run_manifest.json")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != FORMAL_RUN_SCHEMA
        or manifest.get("mode") != "formal"
        or manifest.get("method") != METHOD
        or manifest.get("conditions") != list(formal_contract.CONDITIONS)
    ):
        raise R203AuditError("formal R203 manifest identity differs")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise R203AuditError("formal R203 config is absent")
    try:
        provider_contract = flex_evidence.validate_recorded_contract(
            config.get("gateway_contract")
        )
    except flex_evidence.EvidenceError as exc:
        raise R203AuditError(f"formal gateway contract differs: {exc}") from exc
    python = Path(str(config.get("python", "")))
    expected_config = {
        "retrieval_model": formal_contract.MODEL,
        "answer_model": formal_contract.MODEL,
        "budget_policy": "hard_cap",
        "budget_tokens": formal_contract.BUDGET_TOKENS,
        "max_rounds": formal_contract.MAX_ROUNDS,
        "model_context_limit_tokens": formal_contract.MODEL_CONTEXT_LIMIT_TOKENS,
        "answer_completion_reservation_tokens": (
            formal_contract.ANSWER_COMPLETION_RESERVATION_TOKENS
        ),
        "answer_max_tokens": formal_contract.ANSWER_MAX_TOKENS,
        "answer_retries": formal_contract.ANSWER_RETRIES,
        "tokenizer": answer_contract.formal_token_counter().identity,
        "retrieval_read_only": True,
        "upstream": provider_contract["origin"],
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise R203AuditError("formal R203 config differs")
    if (
        python.is_symlink()
        or not python.is_file()
        or config.get("python_sha256") != answer_contract.sha256_file(python)
        or manifest.get("code_source_hashes") != formal_contract.source_hashes()
        or manifest.get("preregistration")
        != formal_contract.preregistration_binding()
    ):
        raise R203AuditError("formal code, Python, or preregistration binding differs")
    source_binding = answer_contract.read_json(output_dir / "source_binding.json")
    if not isinstance(source_binding, dict):
        raise R203AuditError("formal source binding is absent")
    _validate_content_hash(source_binding, "binding_sha256")
    if manifest.get("source_binding_sha256") != source_binding["binding_sha256"]:
        raise R203AuditError("formal source binding hash differs")
    live_preflight = formal_contract.run_preflight(
        source_root=ROOT / str(source_binding["source_root"]), write_path=None
    )
    live_binding = formal_contract.require_ready_binding(live_preflight)
    if live_binding != source_binding:
        raise R203AuditError("formal source binding differs from live audit")
    if manifest.get("preflight_content_sha256") != live_preflight.get(
        "preflight_content_sha256"
    ):
        raise R203AuditError("formal preflight binding differs")
    copy_record = answer_contract.read_json(
        output_dir / "condition_copy_inventory.json"
    )
    if not isinstance(copy_record, dict):
        raise R203AuditError("formal condition-copy inventory is absent")
    _validate_content_hash(copy_record, "inventory_content_sha256")
    if (
        manifest.get("condition_copy_inventory_sha256")
        != copy_record["inventory_content_sha256"]
        or copy_record.get("initialization")
        != "all_40_copies_completed_before_formal_questions"
        or len(copy_record.get("rows", [])) != 40
    ):
        raise R203AuditError("formal condition-copy binding differs")
    formal_contract.validate_copy_inventory(
        output_dir=output_dir,
        source_binding=source_binding,
        inventory=copy_record["rows"],
    )
    specs = formal_contract.question_specs()
    expected_scope = {
        "samples": formal_contract.SAMPLES,
        "primary_questions_per_condition": formal_contract.PRIMARY_QUESTIONS,
        "condition_artifacts": formal_contract.TOTAL_ARTIFACTS,
        "source_recall_denominator_per_condition": (
            formal_contract.SOURCE_RECALL_DENOMINATOR
        ),
        "category_counts": {
            str(key): value for key, value in formal_contract.CATEGORY_COUNTS.items()
        },
    }
    expected_run_id = (
        "r203-formal-"
        f"{str(source_binding['binding_sha256'])[:16]}-"
        f"{answer_contract.sha256_file(formal_contract.DEFAULT_PREREGISTRATION)[:16]}"
    )
    if (
        manifest.get("scope") != expected_scope
        or manifest.get("question_inventory_sha256")
        != formal_contract.question_inventory_sha256(specs)
        or manifest.get("run_id") != expected_run_id
        or manifest.get("provenance_claim_boundary")
        != source_binding["provenance_claim_boundary"]
    ):
        raise R203AuditError("formal scope, inventory, run ID, or claim boundary differs")
    expected_artifacts = {str(spec["artifact_id"]) for spec in specs}
    reports: list[dict[str, Any]] = []
    response_ids: list[str] = []
    proxy_events: list[dict[str, Any]] = []
    for condition in formal_contract.CONDITIONS:
        root = output_dir / "conditions" / condition / "questions"
        observed = (
            {path.name for path in root.iterdir() if path.is_dir()}
            if root.is_dir()
            else set()
        )
        if observed != expected_artifacts:
            raise R203AuditError(
                f"formal question artifact inventory differs: {condition}"
            )
        for spec in specs:
            report = audit_formal_question_checkpoint(
                output_dir=output_dir,
                condition=condition,
                spec=spec,
                source_binding=source_binding,
                copy_record=copy_record,
            )
            reports.append(report)
            response_ids.extend(report["response_ids"])
            proxy_events.extend(report["proxy_events"])
    if len(reports) != formal_contract.TOTAL_ARTIFACTS:
        raise R203AuditError("formal audited artifact count differs")
    if len(response_ids) != len(set(response_ids)):
        raise R203AuditError("formal response IDs are duplicated across questions")
    for condition in formal_contract.CONDITIONS:
        eligible = [
            report
            for report in reports
            if report["condition"] == condition
            and report["source_recall_eligible"]
        ]
        if len(eligible) != formal_contract.SOURCE_RECALL_DENOMINATOR:
            raise R203AuditError(
                f"formal source-recall denominator differs: {condition}"
            )
    proxy_report = _audit_formal_proxy_inventory(
        output_dir,
        run_id=expected_run_id,
        expected_events=proxy_events,
    )
    complete = answer_contract.read_json(output_dir / "complete.json")
    if (
        not isinstance(complete, dict)
        or complete.get("schema_version") != FORMAL_COMPLETE_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("run_id") != expected_run_id
        or complete.get("questions_per_condition")
        != formal_contract.PRIMARY_QUESTIONS
        or complete.get("condition_artifacts") != formal_contract.TOTAL_ARTIFACTS
        or complete.get("source_recall_denominator_per_condition")
        != formal_contract.SOURCE_RECALL_DENOMINATOR
        or complete.get("source_binding_sha256")
        != source_binding["binding_sha256"]
        or complete.get("condition_copy_inventory_sha256")
        != copy_record["inventory_content_sha256"]
        or complete.get("preregistration_sha256")
        != answer_contract.sha256_file(formal_contract.DEFAULT_PREREGISTRATION)
    ):
        raise R203AuditError("formal completion binding differs")
    progress_path = output_dir / "progress.json"
    progress = answer_contract.read_json(progress_path)
    if (
        not isinstance(progress, dict)
        or progress.get("status") != "complete"
        or progress.get("completed") != formal_contract.TOTAL_ARTIFACTS
        or progress.get("total") != formal_contract.TOTAL_ARTIFACTS
        or complete.get("progress_sha256")
        != answer_contract.sha256_file(progress_path)
    ):
        raise R203AuditError("formal progress binding differs")
    outputs = _audit_formal_outputs(
        output_dir, specs=specs, complete=complete
    )
    formal_contract.verify_source_unchanged(source_binding)
    formal_contract.validate_copy_inventory(
        output_dir=output_dir,
        source_binding=source_binding,
        inventory=copy_record["rows"],
    )
    return {
        "schema_version": FORMAL_AUDIT_SCHEMA,
        "status": "passed",
        "mode": "formal",
        "run_id": expected_run_id,
        "conditions": list(formal_contract.CONDITIONS),
        "questions_per_condition": formal_contract.PRIMARY_QUESTIONS,
        "condition_artifacts": len(reports),
        "source_recall_denominator_per_condition": (
            formal_contract.SOURCE_RECALL_DENOMINATOR
        ),
        "budget_tokens_per_question": formal_contract.BUDGET_TOKENS,
        "model": formal_contract.MODEL,
        "byte_identical_start": True,
        "memory_unchanged": True,
        "r203_maintenance_operation_count": 0,
        "native_builder_pre_maintenance_provenance_claimed": False,
        "response_ids": len(response_ids),
        "actual_model_evidence": proxy_report,
        "aggregate_outputs": outputs,
    }


def audit_run(output_dir: Path) -> dict[str, Any]:
    """Dispatch without importing the R203 runner."""

    output_dir = output_dir.expanduser().absolute()
    manifest = answer_contract.read_json(output_dir / "run_manifest.json")
    if isinstance(manifest, dict) and manifest.get("mode") == "formal":
        return _audit_formal(output_dir)
    return _audit_synthetic(output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(audit_run(args.artifact_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        R203AuditError,
        formal_contract.R203FormalError,
        generic.ReadOnlyAuditError,
        answer_contract.ControlledAnswerError,
        flex_evidence.EvidenceError,
        DurableLedgerError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
