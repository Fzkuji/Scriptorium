#!/usr/bin/env python3
"""Offline auditor for executed LongMemEval-S M1 Mem0/Graphiti plans."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import longmemeval_m1_backend_contract as backend_contract  # noqa: E402
import longmemeval_m1_contract as contract  # noqa: E402
from scripts.evaluation import durable_model_ledger as durable  # noqa: E402
from scripts.evaluation.visible_token_audit import audit_visible_token_trace  # noqa: E402
from scripts.evaluation.visible_token_budget import snapshot_memory_path  # noqa: E402
from scripts.gateways import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402


class BackendAuditError(RuntimeError):
    """Raised when an executed backend artifact is not independently verifiable."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BackendAuditError(message)


def _regular_file(path: Path, *, label: str) -> Path:
    try:
        contract.reject_symlink_components(path)
    except contract.ContractError as exc:
        raise BackendAuditError(str(exc)) from exc
    require(path.is_file() and not path.is_symlink(), f"{label} is missing")
    require(
        path.stat(follow_symlinks=False).st_nlink == 1,
        f"{label} must not be hardlinked",
    )
    return path


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    value = contract.read_json(path)
    require(isinstance(value, dict), f"{label} is not an object")
    return dict(value)


def _safe_operation_artifact(root: Path, relative: Any) -> Path:
    require(
        isinstance(relative, str) and relative, "operation artifact path is invalid"
    )
    candidate = Path(relative)
    require(
        not candidate.is_absolute() and ".." not in candidate.parts,
        "operation artifact path escapes model evidence",
    )
    path = root / candidate
    _regular_file(path, label="operation artifact")
    return path


def _audit_model_ledger(
    *,
    item_dir: Path,
    expected_session_count: int,
    formal: bool,
) -> dict[str, Any]:
    model_root = item_dir / "model_evidence"
    ledger_path = model_root / "ledger.jsonl"
    summary = backend_contract.model_ledger_summary(
        ledger_path=ledger_path,
        artifact_root=model_root,
        formal=formal,
    )
    records = durable.read_ledger(ledger_path)
    try:
        state = durable.ledger_state(records)
    except durable.DurableLedgerError as exc:
        raise BackendAuditError(str(exc)) from exc
    expected_operations = (
        ["initialize"]
        + [f"ingest-{index:04d}" for index in range(expected_session_count)]
        + ["retrieve", "close"]
    )
    require(
        list(state["committed_operations"]) == expected_operations,
        "model ledger operation inventory differs",
    )
    for operation_id, record in state["committed_operations"].items():
        artifact = _safe_operation_artifact(model_root, record.get("artifact_path"))
        require(
            contract.sha256_file(artifact) == record.get("artifact_sha256"),
            f"operation {operation_id} artifact hash differs",
        )
    call_operations = [
        str(record.get("operation_id"))
        for record in records
        if record.get("event") == "model_call_finished"
    ]
    require(
        all(value.startswith("ingest-") for value in call_operations),
        "builder model call occurred outside history ingest",
    )
    if formal:
        for index in range(expected_session_count):
            require(
                f"ingest-{index:04d}" in call_operations,
                f"history session {index} made no GPT-5.5 builder call",
            )
    summary["committed_operations"] = state["committed_operations"]
    return summary


def _formal_model_response_ids(items_root: Path) -> list[str]:
    response_ids: list[str] = []
    for ledger_path in sorted(items_root.glob("*/model_evidence/ledger.jsonl")):
        for record in durable.read_ledger(ledger_path):
            if record.get("event") != "model_call_finished":
                continue
            response_id = record.get("response_id")
            require(
                isinstance(response_id, str) and response_id,
                "formal model ledger response ID is invalid",
            )
            response_ids.append(response_id)
    require(
        len(response_ids) == len(set(response_ids)),
        "formal model response IDs are duplicated",
    )
    return response_ids


def _audit_provider_evidence(run_dir: Path, *, expected_requests: int) -> dict[str, Any]:
    root = run_dir / "provider_evidence"
    require(
        root.is_dir() and not root.is_symlink(),
        "formal backend provider evidence root is missing",
    )
    invocation_dirs = sorted(root.glob("invocation-*"))
    require(invocation_dirs, "formal backend has no Flex provider invocation")
    require(
        {path.name for path in root.iterdir()}
        == {f"invocation-{index:04d}" for index in range(1, len(invocation_dirs) + 1)},
        "Flex provider invocation directory inventory differs",
    )
    rows: list[dict[str, Any]] = []
    provider_response_ids: list[str] = []
    for index, directory in enumerate(invocation_dirs, start=1):
        require(
            directory.name == f"invocation-{index:04d}"
            and directory.is_dir()
            and not directory.is_symlink(),
            "Flex provider invocation directory differs",
        )
        expected_files = {
            flex_evidence.CHILD_READY_NAME,
            flex_evidence.CHILD_LOG_NAME,
            flex_evidence.WINDOW_NAME,
            flex_evidence.INVOCATION_NAME,
        }
        require(
            {path.name for path in directory.iterdir()} == expected_files,
            "Flex provider invocation file inventory differs",
        )
        record_path = directory / flex_evidence.INVOCATION_NAME
        record = _read_object(record_path, label="Flex provider invocation")
        for field, filename in (
            ("window", flex_evidence.WINDOW_NAME),
            ("consumer_log", flex_evidence.CHILD_LOG_NAME),
            ("child_ready", flex_evidence.CHILD_READY_NAME),
        ):
            binding = record.get(field)
            require(
                isinstance(binding, dict)
                and Path(str(binding.get("path"))).resolve() == directory / filename,
                f"Flex provider {field} path is not fixed",
            )
        try:
            report = flex_evidence.audit_invocation(record)
            consumers = flex_evidence.load_child_proxy_records(
                directory / flex_evidence.CHILD_LOG_NAME,
                expected_run_id=str(record.get("run_id")),
            )
        except flex_evidence.EvidenceError as exc:
            raise BackendAuditError(str(exc)) from exc
        provider_response_ids.extend(str(item["response_id"]) for item in consumers)
        rows.append(
            {
                "path": str(record_path.relative_to(run_dir)),
                "sha256": contract.sha256_file(record_path),
                "requests": report["requests"],
                "segment_sha256": report["segment_sha256"],
                "committed_cost_nanos": report["committed_cost_nanos"],
            }
        )
    require(
        len(provider_response_ids) == len(set(provider_response_ids)),
        "Flex provider response IDs are duplicated",
    )
    model_response_ids = _formal_model_response_ids(run_dir / "items")
    require(
        set(provider_response_ids) == set(model_response_ids),
        "model-ledger response IDs do not equal Flex provider response IDs",
    )
    result = {
        "schema": "longmemeval-m1-flex-provider-evidence/v1",
        "transport": "exclusive-child-proxy-to-openai-gpt55-flex-gateway",
        "invocations": rows,
        "requests": sum(int(row["requests"]) for row in rows),
    }
    require(
        result["requests"] == expected_requests,
        "Flex provider request count differs from model network calls",
    )
    return result


def _audit_source_trace(
    *,
    path: Path,
    method: str,
    item: Mapping[str, Any],
    item_index: int,
    model_root: Path,
) -> dict[str, Any]:
    trace = _read_object(path, label="source trace")
    sessions = contract.rendered_sessions(item)
    require(
        trace.get("schema_version") == backend_contract.BACKEND_SOURCE_TRACE_SCHEMA
        and trace.get("method") == method
        and trace.get("dataset_index") == item_index
        and trace.get("question_id") == item["question_id"]
        and trace.get("history_content_sha256") == contract.history_content_hash(item)
        and trace.get("history_owner_sha256") == contract.history_owner_hash(item)
        and trace.get("session_count") == len(sessions),
        "source trace identity differs",
    )
    rows = trace.get("sessions")
    require(
        isinstance(rows, list) and len(rows) == len(sessions),
        "source trace rows differ",
    )
    for index, (row, session) in enumerate(zip(rows, sessions)):
        require(isinstance(row, dict), f"source trace row {index} is invalid")
        require(
            row.get("dataset_session_index") == session["dataset_session_index"]
            and row.get("source_session_id") == session["source_session_id"]
            and row.get("date") == session["date"]
            and row.get("text_sha256") == session["text_sha256"]
            and row.get("text") is None,
            f"source trace row {index} binding differs",
        )
        receipt = row.get("ingest")
        require(
            isinstance(receipt, dict), f"source trace row {index} receipt is invalid"
        )
        operation = _read_object(
            model_root / "operations" / f"ingest-{index:04d}.json",
            label=f"ingest receipt {index}",
        )
        require(receipt == operation, f"source trace row {index} receipt differs")
        require(
            receipt.get("source_session_id") == session["source_session_id"]
            and receipt.get("dataset_session_index") == index,
            f"source trace row {index} backend receipt differs",
        )
    return trace


def _audit_retrieval(
    *,
    path: Path,
    method: str,
    item: Mapping[str, Any],
    item_index: int,
    checkpoint: Mapping[str, Any],
    model_root: Path,
) -> dict[str, Any]:
    value = _read_object(path, label="retrieval trace")
    require(
        value.get("schema_version") == backend_contract.BACKEND_RETRIEVAL_SCHEMA
        and value.get("method") == method
        and value.get("dataset_index") == item_index
        and value.get("question_id") == item["question_id"]
        and value.get("question_sha256") == contract.sha256_text(str(item["question"]))
        and value.get("retrieval_limit") == backend_contract.RETRIEVAL_LIMIT
        and value.get("workspace_after_build")
        == checkpoint.get("workspace_after_build"),
        "retrieval trace identity differs",
    )
    operation = _read_object(
        model_root / "operations/retrieve.json", label="retrieval operation artifact"
    )
    require(value == operation, "retrieval trace differs from committed operation")
    records = value.get("records")
    require(
        isinstance(records, list)
        and value.get("record_count") == len(records)
        and len(records) <= backend_contract.RETRIEVAL_LIMIT,
        "retrieval record count differs",
    )
    sessions = contract.rendered_sessions(item)
    source_by_id = {str(session["source_session_id"]): session for session in sessions}
    backend_ids: set[str] = set()
    for rank, record in enumerate(records):
        require(isinstance(record, dict), f"retrieval record {rank} is invalid")
        backend_id = record.get("backend_record_id")
        text = record.get("text")
        source_ids = record.get("source_session_ids")
        source_indices = record.get("source_dataset_session_indices")
        source_hashes = record.get("source_document_sha256s")
        require(
            record.get("rank") == rank
            and isinstance(backend_id, str)
            and backend_id
            and backend_id not in backend_ids
            and isinstance(text, str)
            and text.strip() != ""
            and record.get("text_sha256") == contract.sha256_text(text)
            and isinstance(source_ids, list)
            and len(source_ids) > 0
            and len(set(source_ids)) == len(source_ids)
            and isinstance(source_indices, list)
            and isinstance(source_hashes, list)
            and len(source_ids) == len(source_indices) == len(source_hashes),
            f"retrieval record {rank} fields differ",
        )
        backend_ids.add(backend_id)
        for source_id, source_index, source_hash in zip(
            source_ids, source_indices, source_hashes
        ):
            source = source_by_id.get(str(source_id))
            require(
                source is not None
                and source["dataset_session_index"] == source_index
                and source["text_sha256"] == source_hash,
                f"retrieval record {rank} source mapping differs",
            )
    return value


def _read_budget_deliveries(path: Path) -> list[dict[str, Any]]:
    deliveries: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BackendAuditError(
                f"budget trace line {line_number} is invalid"
            ) from exc
        require(isinstance(value, dict), "budget trace record is invalid")
        if value.get("record_type") == "delivery":
            deliveries.append(value)
    return deliveries


def _audit_answer_input(
    *,
    answer_path: Path,
    budget_path: Path,
    retrieval: Mapping[str, Any],
    method: str,
    item: Mapping[str, Any],
    item_index: int,
    checkpoint: Mapping[str, Any],
    tokenizer: contract.FormalTokenizer,
) -> dict[str, Any]:
    answer = _read_object(answer_path, label="answer input")
    try:
        contract.validate_answer_input_allowlist(answer)
    except contract.ContractError as exc:
        raise BackendAuditError(str(exc)) from exc
    require(
        answer.get("schema_version") == contract.SCHEMA_VERSION
        and answer.get("benchmark") == "LongMemEval-S"
        and answer.get("method") == method
        and answer.get("dataset_index") == item_index
        and answer.get("question_id") == item["question_id"]
        and answer.get("question_type") == item["question_type"]
        and answer.get("question") == item["question"]
        and answer.get("question_date") == item["question_date"]
        and answer.get("history_content_sha256") == contract.history_content_hash(item)
        and answer.get("history_owner_sha256") == contract.history_owner_hash(item),
        "answer input identity differs",
    )
    context = answer.get("context")
    require(isinstance(context, dict), "answer input context is invalid")
    deliveries = _read_budget_deliveries(budget_path)
    delivered_records = [
        record
        for record in deliveries
        if isinstance(record.get("delivered"), dict)
        and isinstance(record["delivered"].get("text"), str)
    ]
    expected_context = "".join(
        record["delivered"]["text"] for record in delivered_records
    )
    cumulative_delivered = sum(
        int(record["delivered"]["tokens"]) for record in delivered_records
    )
    require(
        context.get("text") == expected_context
        and context.get("text_sha256") == contract.sha256_text(expected_context)
        and context.get("utf8_bytes") == len(expected_context.encode("utf-8"))
        and context.get("visible_tokens") == cumulative_delivered
        and context.get("retokenized_context_tokens")
        == tokenizer.count(expected_context)
        and context.get("visible_tokens") <= contract.VISIBLE_BUDGET_TOKENS,
        "answer input visible context differs from the gate",
    )
    events = context.get("events")
    require(
        isinstance(events, list) and len(events) == len(delivered_records),
        "answer input context event count differs",
    )
    retrieval_records = retrieval.get("records")
    require(isinstance(retrieval_records, list), "retrieval records are invalid")
    cumulative = 0
    delivered_sources: list[str] = []
    for rank, (event, delivery) in enumerate(zip(events, delivered_records)):
        source = retrieval_records[rank]
        raw_text = delivery["raw"]["text"]
        delivered_text = delivery["delivered"]["text"]
        delivered_tokens = tokenizer.count(delivered_text)
        cumulative += delivered_tokens
        source_ids = source["source_session_ids"]
        require(
            isinstance(event, dict)
            and event.get("rank") == rank
            and event.get("backend_record_id") == source["backend_record_id"]
            and event.get("source_session_ids") == source_ids
            and event.get("source_dataset_session_indices")
            == source["source_dataset_session_indices"]
            and event.get("source_document_sha256s")
            == source["source_document_sha256s"]
            and event.get("raw_text_sha256") == contract.sha256_text(raw_text)
            and event.get("delivered_text_sha256")
            == contract.sha256_text(delivered_text)
            and event.get("raw_tokens") == tokenizer.count(raw_text)
            and event.get("delivered_tokens") == delivered_tokens
            and event.get("cumulative_visible_tokens") == cumulative,
            f"answer input context event {rank} differs",
        )
        delivered_sources.extend(source_ids)
    unique_sources = list(dict.fromkeys(delivered_sources))
    require(
        context.get("source_session_ids") == unique_sources,
        "answer input delivered source inventory differs",
    )
    budget = context.get("budget")
    require(
        isinstance(budget, dict)
        and budget.get("policy") == "hard_visible_total"
        and budget.get("configured_visible_budget_tokens")
        == contract.VISIBLE_BUDGET_TOKENS
        and budget.get("overflow_policy") == "truncate_current_then_stop"
        and budget.get("cumulative_visible_tokens") == cumulative,
        "answer input budget block differs",
    )
    prompt = contract.ANSWER_PROMPT.format(
        memories=expected_context, question=item["question"]
    )
    interface = answer.get("answer_protocol_interface")
    require(
        isinstance(interface, dict)
        and interface.get("status") == "reserved_not_executed"
        and interface.get("requested_model") == contract.EXPECTED_MODEL
        and interface.get("rendered_prompt_sha256") == contract.sha256_text(prompt)
        and interface.get("rendered_prompt_tokens") == tokenizer.count(prompt)
        and interface.get("model_calls") == 0,
        "answer protocol interface differs",
    )
    expected_source_mapping = contract._source_mapping(item, unique_sources)  # noqa: SLF001
    require(
        checkpoint.get("private_audit", {}).get("source_mapping")
        == expected_source_mapping,
        "checkpoint source mapping differs",
    )
    return answer


def _audit_item(
    *,
    plan: Mapping[str, Any],
    plan_record: Mapping[str, Any],
    item: Mapping[str, Any],
    preregistration_sha256: str,
    tokenizer: contract.FormalTokenizer,
    formal: bool,
) -> dict[str, Any]:
    method = str(plan["method"])
    item_index = int(plan_record["dataset_index"])
    item_dir = (
        Path(str(plan["output_root"]))
        / method
        / "items"
        / contract.item_directory_name(item_index, str(item["question_id"]))
    )
    require(
        item_dir.is_dir() and not item_dir.is_symlink(),
        f"item {item_index} directory is missing",
    )
    ledger_path = _regular_file(item_dir / "attempts.jsonl", label="attempt ledger")
    answer_path = _regular_file(item_dir / "answer_input.json", label="answer input")
    checkpoint_path = _regular_file(item_dir / "checkpoint.json", label="checkpoint")
    source_path = _regular_file(item_dir / "source_trace.json", label="source trace")
    retrieval_path = _regular_file(
        item_dir / "retrieval_raw.json", label="retrieval trace"
    )
    proxy_path = _regular_file(item_dir / "proxy_slice.json", label="proxy slice")
    budget_path = _regular_file(
        item_dir / "retrieval_budget.jsonl", label="budget trace"
    )
    budget_manifest_path = _regular_file(
        item_dir / "retrieval_budget.manifest.json", label="budget manifest"
    )
    events = contract.read_jsonl(ledger_path)
    terminal = contract.validate_item_ledger(
        events,
        method=method,
        item_index=item_index,
        question_id=str(item["question_id"]),
        require_complete=True,
    )
    require(
        terminal is not None and len(events) == 2, "attempt ledger terminal differs"
    )
    checkpoint = _read_object(checkpoint_path, label="checkpoint")
    require(
        checkpoint.get("schema_version") == contract.CHECKPOINT_SCHEMA_VERSION
        and checkpoint.get("backend_execution_schema")
        == backend_contract.BACKEND_RUN_SCHEMA
        and checkpoint.get("status") == "complete"
        and checkpoint.get("scope") == plan["scope"]
        and checkpoint.get("method") == method
        and checkpoint.get("dataset_index") == item_index
        and checkpoint.get("question_id") == item["question_id"]
        and checkpoint.get("attempt_id") == terminal.get("attempt_id")
        and checkpoint.get("terminal_event_id") == terminal.get("event_id")
        and checkpoint.get("history_owner_question_id") == item["question_id"]
        and checkpoint.get("history_content_sha256")
        == contract.history_content_hash(item)
        and checkpoint.get("history_owner_sha256") == contract.history_owner_hash(item)
        and checkpoint.get("preregistration_sha256") == preregistration_sha256
        and checkpoint.get("configuration_sha256")
        == contract.canonical_hash(plan["configuration"])
        and checkpoint.get("plan_content_sha256") == plan["plan_content_sha256"]
        and checkpoint.get("ledger_sha256") == contract.sha256_file(ledger_path),
        f"item {item_index} checkpoint identity differs",
    )
    workspace = Path(plan_record["workspace"]["workspace"])
    before_file = _read_object(
        item_dir / "workspace_before.json", label="workspace before"
    )
    require(
        before_file.get("snapshot") == checkpoint.get("workspace_before")
        and before_file.get("dataset_index") == item_index
        and before_file.get("question_id") == item["question_id"],
        f"item {item_index} initial workspace snapshot differs",
    )
    before = checkpoint.get("workspace_before")
    require(
        isinstance(before, dict)
        and before.get("kind") == "directory"
        and before.get("path") == str(workspace.resolve())
        and before.get("file_count") == 0
        and before.get("byte_count") == 0
        and before.get("entries") == [],
        f"item {item_index} workspace was not empty before construction",
    )
    final_snapshot = snapshot_memory_path(workspace).descriptor
    require(
        checkpoint.get("workspace_after_retrieval") == final_snapshot,
        f"item {item_index} final workspace hash differs",
    )
    model_summary = _audit_model_ledger(
        item_dir=item_dir,
        expected_session_count=len(item["haystack_sessions"]),
        formal=formal,
    )
    _audit_source_trace(
        path=source_path,
        method=method,
        item=item,
        item_index=item_index,
        model_root=item_dir / "model_evidence",
    )
    retrieval = _audit_retrieval(
        path=retrieval_path,
        method=method,
        item=item,
        item_index=item_index,
        checkpoint=checkpoint,
        model_root=item_dir / "model_evidence",
    )
    gate_report = audit_visible_token_trace(
        budget_path,
        manifest_path=budget_manifest_path,
        memory_after_path=workspace,
        require_complete=True,
    )
    require(
        gate_report.get("audit_status") == "pass",
        f"item {item_index} visible-token audit failed: {gate_report.get('errors')}",
    )
    require(
        gate_report.get("configured_budget_tokens") == contract.VISIBLE_BUDGET_TOKENS
        and gate_report.get("memory_before_sha256")
        == checkpoint.get("workspace_after_search", {}).get("sha256")
        and gate_report.get("memory_after_sha256") == final_snapshot["sha256"],
        f"item {item_index} visible-token workspace binding differs",
    )
    answer = _audit_answer_input(
        answer_path=answer_path,
        budget_path=budget_path,
        retrieval=retrieval,
        method=method,
        item=item,
        item_index=item_index,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
    )
    proxy_value = _read_object(proxy_path, label="proxy slice")
    try:
        backend_contract.validate_proxy_slice(proxy_value, formal=formal)
    except backend_contract.BackendContractError as exc:
        raise BackendAuditError(str(exc)) from exc
    expected_hashes = {
        "answer_input_sha256": contract.sha256_file(answer_path),
        "model_ledger_sha256": model_summary["ledger_sha256"],
        "source_trace_sha256": contract.sha256_file(source_path),
        "retrieval_sha256": contract.sha256_file(retrieval_path),
        "proxy_slice_sha256": contract.sha256_file(proxy_path),
        "budget_trace_sha256": contract.sha256_file(budget_path),
        "budget_manifest_sha256": contract.sha256_file(budget_manifest_path),
    }
    for field, expected in expected_hashes.items():
        require(
            checkpoint.get(field) == expected,
            f"item {item_index} checkpoint {field} differs",
        )
    require(
        terminal.get("answer_input_sha256") == expected_hashes["answer_input_sha256"]
        and terminal.get("model_ledger_sha256")
        == expected_hashes["model_ledger_sha256"]
        and terminal.get("source_trace_sha256")
        == expected_hashes["source_trace_sha256"]
        and terminal.get("retrieval_sha256") == expected_hashes["retrieval_sha256"]
        and terminal.get("proxy_slice_sha256") == expected_hashes["proxy_slice_sha256"]
        and terminal.get("model_calls") == model_summary["model_calls"]
        and terminal.get("network_calls") == model_summary["network_calls"],
        f"item {item_index} attempt terminal evidence differs",
    )
    if formal:
        require(
            checkpoint.get("requested_builder_model") == contract.EXPECTED_MODEL
            and checkpoint.get("actual_builder_model") == contract.EXPECTED_MODEL
            and checkpoint.get("model_calls") == model_summary["model_calls"] > 0
            and checkpoint.get("network_calls") == model_summary["network_calls"] > 0,
            f"item {item_index} formal model identity differs",
        )
    return {
        "dataset_index": item_index,
        "question_id": item["question_id"],
        "abstention": str(item["question_id"]).endswith("_abs"),
        "item_dir": item_dir.name,
        "answer_input_sha256": contract.sha256_file(answer_path),
        "checkpoint_sha256": contract.sha256_file(checkpoint_path),
        "ledger_sha256": contract.sha256_file(ledger_path),
        "history_content_sha256": checkpoint["history_content_sha256"],
        "history_owner_sha256": checkpoint["history_owner_sha256"],
        "visible_tokens": answer["context"]["visible_tokens"],
        "rendered_prompt_tokens": answer["answer_protocol_interface"][
            "rendered_prompt_tokens"
        ],
        "delivered_source_session_ids": answer["context"]["source_session_ids"],
        "source_mapping_session_recall": checkpoint["private_audit"]["source_mapping"][
            "session_recall"
        ],
        "model_calls": model_summary["model_calls"],
        "network_calls": model_summary["network_calls"],
        "workspace": str(workspace),
        "workspace_after_retrieval_sha256": final_snapshot["sha256"],
        "model_ledger_sha256": model_summary["ledger_sha256"],
        "proxy_slice_sha256": contract.sha256_file(proxy_path),
        "source_trace_sha256": contract.sha256_file(source_path),
        "retrieval_sha256": contract.sha256_file(retrieval_path),
        "budget_trace_sha256": contract.sha256_file(budget_path),
        "budget_manifest_sha256": contract.sha256_file(budget_manifest_path),
    }


def audit_backend_run(
    *,
    plan_path: Path,
    dataset_path: Path,
    preregistration_path: Path,
    output_path: Path | None = None,
    formal: bool = True,
) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    preregistration_path = preregistration_path.expanduser().resolve()
    try:
        plan, data, _ = backend_contract.validate_plan(
            plan_path=plan_path,
            dataset_path=dataset_path,
            preregistration_path=preregistration_path,
        )
    except backend_contract.BackendContractError as exc:
        raise BackendAuditError(str(exc)) from exc
    method = str(plan["method"])
    run_dir = Path(str(plan["output_root"])) / method
    manifest_path = _regular_file(run_dir / "run_manifest.json", label="run manifest")
    manifest = _read_object(manifest_path, label="run manifest")
    identity = manifest.get("identity")
    require(isinstance(identity, dict), "run manifest identity is invalid")
    require(
        manifest.get("schema_version") == contract.RUN_MANIFEST_SCHEMA_VERSION
        and manifest.get("backend_execution_schema")
        == backend_contract.BACKEND_RUN_SCHEMA
        and manifest.get("status") == "inputs_complete"
        and identity.get("method") == method
        and identity.get("scope") == plan["scope"]
        and identity.get("formal") == formal
        and identity.get("plan_sha256") == contract.sha256_file(plan_path)
        and identity.get("plan_content_sha256") == plan["plan_content_sha256"]
        and identity.get("dataset_sha256") == contract.sha256_file(dataset_path)
        and identity.get("preregistration_sha256")
        == contract.sha256_file(preregistration_path)
        and identity.get("configuration") == plan["configuration"]
        and identity.get("dependencies") == plan["dependencies"]
        and identity.get("retrieval_limit") == backend_contract.RETRIEVAL_LIMIT
        and identity.get("requested_builder_model") == contract.EXPECTED_MODEL
        and identity.get("accepted_actual_model_regex")
        == backend_contract.ACCEPTED_ACTUAL_MODEL_REGEX
        and identity.get("source_hashes") == backend_contract.source_hashes(),
        "run manifest frozen identity differs",
    )
    expected_names = [
        contract.item_directory_name(
            int(record["dataset_index"]), str(record["question_id"])
        )
        for record in plan["items"]
    ]
    items_root = run_dir / "items"
    actual_nodes = sorted(items_root.iterdir(), key=lambda path: path.name)
    require(
        [path.name for path in actual_nodes] == expected_names
        and all(path.is_dir() and not path.is_symlink() for path in actual_nodes),
        "backend item directory inventory differs",
    )
    tokenizer = contract.FormalTokenizer.resolve()
    audited = [
        _audit_item(
            plan=plan,
            plan_record=record,
            item=data[int(record["dataset_index"])],
            preregistration_sha256=contract.sha256_file(preregistration_path),
            tokenizer=tokenizer,
            formal=formal,
        )
        for record in plan["items"]
    ]
    require(manifest.get("items") == audited, "run manifest item inventory differs")
    inventory = contract.regular_tree_inventory(items_root)
    inventory_summary = {
        "entries": len(inventory),
        "bytes": sum(entry["bytes"] for entry in inventory),
        "root_sha256": contract.inventory_root(inventory),
    }
    require(
        manifest.get("item_tree_inventory") == inventory_summary,
        "run manifest item tree hash differs",
    )
    model_calls = sum(item["model_calls"] for item in audited)
    network_calls = sum(item["network_calls"] for item in audited)
    require(
        manifest.get("completed_items") == len(audited)
        and manifest.get("expected_items") == len(audited)
        and manifest.get("model_calls") == model_calls
        and manifest.get("network_calls") == network_calls,
        "run manifest completion accounting differs",
    )
    if formal:
        require(
            model_calls > 0 and network_calls > 0, "formal run made no model request"
        )
        provider_evidence = _audit_provider_evidence(
            run_dir, expected_requests=network_calls
        )
        require(
            manifest.get("provider_evidence") == provider_evidence,
            "run manifest Flex provider evidence differs",
        )
    else:
        require(
            manifest.get("provider_evidence") is None
            and not (run_dir / "provider_evidence").exists(),
            "synthetic backend run claims provider evidence",
        )
        provider_evidence = None
    report = {
        "schema_version": backend_contract.BACKEND_AUDIT_SCHEMA,
        "status": "passed",
        "audited_at": contract.utc_now(),
        "benchmark": "LongMemEval-S",
        "method": method,
        "scope": plan["scope"],
        "formal": formal,
        "plan_path": str(plan_path),
        "plan_sha256": contract.sha256_file(plan_path),
        "run_dir": str(run_dir),
        "run_manifest_sha256": contract.sha256_file(manifest_path),
        "dataset_sha256": contract.sha256_file(dataset_path),
        "preregistration_sha256": contract.sha256_file(preregistration_path),
        "items": len(audited),
        "abstention_items": sum(item["abstention"] for item in audited),
        "model_calls": model_calls,
        "network_calls": network_calls,
        "requested_builder_model": contract.EXPECTED_MODEL,
        "actual_builder_model": contract.EXPECTED_MODEL if formal else None,
        "workspace_paths": [item["workspace"] for item in audited],
        "unique_workspace_paths": len({item["workspace"] for item in audited}),
        "unique_history_hashes": len(
            {item["history_content_sha256"] for item in audited}
        ),
        "source_mapping": {
            "requirement": "R002",
            "granularity": "session",
            "turn_recall_claimed": False,
        },
        "visible_token_summary": {
            "min": min(item["visible_tokens"] for item in audited),
            "max": max(item["visible_tokens"] for item in audited),
            "sum": sum(item["visible_tokens"] for item in audited),
        },
        "item_tree_inventory": inventory_summary,
        "provider_evidence": provider_evidence,
    }
    require(
        report["unique_workspace_paths"] == len(audited)
        and report["unique_history_hashes"] == len(audited),
        "backend run reused a workspace or history",
    )
    if output_path is not None:
        output_path = Path(os.path.abspath(output_path.expanduser()))
        try:
            contract.atomic_json_no_clobber(output_path, report)
        except (contract.ContractError, FileExistsError) as exc:
            raise BackendAuditError(f"cannot publish backend audit: {exc}") from exc
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=contract.DEFAULT_DATASET)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_backend_run(
        plan_path=args.plan,
        dataset_path=args.dataset,
        preregistration_path=args.preregistration,
        output_path=args.output,
        formal=True,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
