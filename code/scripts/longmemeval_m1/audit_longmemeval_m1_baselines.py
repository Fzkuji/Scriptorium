#!/usr/bin/env python3
"""Independent read-only auditor for LongMemEval-S M1 baseline inputs/plans."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts import longmemeval_m1_contract as contract  # noqa: E402


class AuditError(RuntimeError):
    """Raised when an M1 artifact is incomplete, stale, or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _content_hash(payload: Mapping[str, Any], field: str) -> str:
    content = copy.deepcopy(dict(payload))
    content.pop(field, None)
    return contract.canonical_hash(content)


def audit_preregistration(
    preregistration_path: Path, dataset_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preregistration_path = preregistration_path.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    try:
        contract.ensure_distinct_paths(
            {"dataset": dataset_path, "preregistration": preregistration_path}
        )
        data = contract.validate_dataset(dataset_path)
        payload = contract.read_json(preregistration_path)
    except contract.ContractError as exc:
        raise AuditError(str(exc)) from exc
    require(isinstance(payload, dict), "preregistration is not an object")
    expected_scalars = {
        "schema_version": contract.PREREG_SCHEMA_VERSION,
        "artifact_type": "formal_experiment_matrix_preregistration",
        "status": "frozen",
        "benchmark": "LongMemEval-S",
        "milestone": "M1 controlled baselines",
        "task_scope": "retrieval_inputs_before_answer_generation",
        "formal_methods": list(contract.FORMAL_METHODS),
        "model_calls": 0,
        "network_calls": 0,
    }
    for key, expected in expected_scalars.items():
        require(payload.get(key) == expected, f"preregistration {key} differs")
    require(
        payload.get("preregistration_content_sha256")
        == _content_hash(payload, "preregistration_content_sha256"),
        "preregistration content hash differs",
    )
    dataset = payload.get("dataset")
    require(isinstance(dataset, dict), "preregistration dataset is invalid")
    require(
        dataset.get("sha256") == contract.EXPECTED_DATASET_SHA256,
        "preregistration dataset hash differs",
    )
    require(dataset.get("items") == 500, "preregistration item count differs")
    require(
        dataset.get("abstention_items") == 30,
        "preregistration abstention count differs",
    )
    require(
        dataset.get("question_types") == contract.EXPECTED_TYPES,
        "preregistration question types differ",
    )
    require(
        dataset.get("unique_history_hashes") == 500,
        "preregistration unique history count differs",
    )
    try:
        tokenizer = contract.FormalTokenizer.resolve()
        dependencies = contract.strict_dependency_snapshot()
    except contract.ContractError as exc:
        raise AuditError(str(exc)) from exc
    require(payload.get("tokenizer") == tokenizer.identity, "tokenizer identity differs")
    require(payload.get("dependencies") == dependencies, "dependency versions differ")
    expected_rows = [
        {
            "method": method,
            "formal_scope_items": 500,
            "configuration": contract.method_configuration(method),
        }
        for method in contract.FORMAL_METHODS
    ]
    require(payload.get("rows") == expected_rows, "formal matrix differs")
    require(
        payload.get("context_policy")
        == {
            "model_context_limit_tokens": 128_000,
            "answer_completion_reservation_tokens": 4_096,
            "max_rendered_prompt_tokens": 123_904,
            "full_context_no_truncation": True,
            "hard_visible_total_methods": ["bm25", "mem0", "graphiti"],
            "hard_visible_total_tokens": 20_000,
        },
        "context policy differs",
    )
    source_mapping = payload.get("source_mapping")
    require(
        source_mapping
        == {
            "requirement": "R002",
            "granularity": "session",
            "turn_recall_claimed": False,
            "answer_session_ids_are_audit_only": True,
        },
        "source mapping policy differs",
    )
    answer_protocol = payload.get("answer_protocol")
    require(isinstance(answer_protocol, dict), "answer protocol block is missing")
    require(answer_protocol.get("status") == "not_started", "answer stage has started")
    require(
        answer_protocol.get("requested_model") == "gpt-5.5",
        "answer model reservation differs",
    )
    require(
        answer_protocol.get("prompt_template_sha256")
        == contract.sha256_text(contract.ANSWER_PROMPT),
        "answer prompt template hash differs",
    )
    require(
        payload.get("code")
        == {
            "contract": {
                "path": str(Path(contract.__file__).resolve()),
                "sha256": contract.sha256_file(Path(contract.__file__).resolve()),
            },
            "preregister_runner": {
                "path": str(
                    (ROOT / "scripts/longmemeval_m1/run_longmemeval_m1_baselines.py").resolve()
                ),
                "sha256": contract.sha256_file(
                    ROOT / "scripts/longmemeval_m1/run_longmemeval_m1_baselines.py"
                ),
            },
        },
        "preregistration code binding differs",
    )
    return payload, data


def _expected_run_identity(
    *,
    method: str,
    dataset_path: Path,
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "benchmark": "LongMemEval-S",
        "milestone": "M1 controlled baselines",
        "method": method,
        "scope": "formal_500",
        "dataset_path": str(dataset_path),
        "dataset_sha256": contract.sha256_file(dataset_path),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": contract.sha256_file(preregistration_path),
        "preregistration_content_sha256": preregistration[
            "preregistration_content_sha256"
        ],
        "configuration": contract.method_configuration(method),
        "tokenizer": contract.FormalTokenizer.resolve().identity,
        "code_hashes": {
            "contract": contract.sha256_file(Path(contract.__file__).resolve()),
            "runner": contract.sha256_file(
                ROOT / "scripts/longmemeval_m1/run_longmemeval_m1_baselines.py"
            ),
            "answer_prompt_source": contract.sha256_file(
                ROOT / "src/evaluation/prompts.py"
            ),
        },
    }


def _root_allowlist(run_dir: Path, audit_output: Path | None) -> None:
    allowed = {".run.lock", "items", "run_manifest.json", "audit.json"}
    if audit_output is not None and audit_output.parent == run_dir:
        allowed.add(audit_output.name)
    observed = {path.name for path in run_dir.iterdir()}
    require(observed <= allowed, f"run directory has unexpected nodes: {sorted(observed - allowed)}")
    require({".run.lock", "items", "run_manifest.json"} <= observed, "run directory is incomplete")


def _item_allowlist(item_dir: Path) -> None:
    expected = {".item.lock", "answer_input.json", "attempts.jsonl", "checkpoint.json"}
    observed = {path.name for path in item_dir.iterdir()}
    require(observed == expected, f"{item_dir.name} node inventory differs: {sorted(observed)}")


def _audit_one_item(
    *,
    method: str,
    item: Mapping[str, Any],
    item_index: int,
    item_dir: Path,
    preregistration_sha256: str,
    configuration_sha256: str,
    tokenizer: contract.FormalTokenizer,
) -> dict[str, Any]:
    _item_allowlist(item_dir)
    answer_path = item_dir / "answer_input.json"
    checkpoint_path = item_dir / "checkpoint.json"
    ledger_path = item_dir / "attempts.jsonl"
    try:
        answer_input = contract.read_json(answer_path)
        checkpoint = contract.read_json(checkpoint_path)
        events = contract.read_jsonl(ledger_path)
        terminal = contract.validate_item_ledger(
            events,
            method=method,
            item_index=item_index,
            question_id=str(item["question_id"]),
            require_complete=True,
        )
        expected_input, expected_private = contract.prepare_item(
            method, item, item_index, tokenizer
        )
        contract.validate_answer_input_allowlist(answer_input)
    except contract.ContractError as exc:
        raise AuditError(str(exc)) from exc
    require(isinstance(answer_input, dict), f"item {item_index} answer input is invalid")
    require(isinstance(checkpoint, dict), f"item {item_index} checkpoint is invalid")
    require(answer_input == expected_input, f"item {item_index} answer input differs")
    require(
        checkpoint.get("schema_version") == contract.CHECKPOINT_SCHEMA_VERSION,
        f"item {item_index} checkpoint schema differs",
    )
    require(checkpoint.get("status") == "complete", f"item {item_index} is incomplete")
    require(checkpoint.get("method") == method, f"item {item_index} method differs")
    require(
        checkpoint.get("dataset_index") == item_index
        and checkpoint.get("question_id") == item["question_id"],
        f"item {item_index} checkpoint identity differs",
    )
    require(
        checkpoint.get("history_owner_question_id") == item["question_id"],
        f"item {item_index} history owner differs",
    )
    require(
        checkpoint.get("history_content_sha256") == contract.history_content_hash(item)
        and checkpoint.get("history_owner_sha256") == contract.history_owner_hash(item),
        f"item {item_index} history hash differs",
    )
    require(
        checkpoint.get("preregistration_sha256") == preregistration_sha256
        and checkpoint.get("configuration_sha256") == configuration_sha256,
        f"item {item_index} frozen binding differs",
    )
    require(
        checkpoint.get("answer_input_sha256") == contract.sha256_file(answer_path)
        and checkpoint.get("ledger_sha256") == contract.sha256_file(ledger_path),
        f"item {item_index} artifact hash differs",
    )
    require(
        checkpoint.get("private_audit") == expected_private,
        f"item {item_index} private source audit differs",
    )
    require(
        checkpoint.get("answer_stage") == "not_started"
        and checkpoint.get("model_calls") == 0
        and checkpoint.get("network_calls") == 0,
        f"item {item_index} no-model status differs",
    )
    require(terminal is not None, f"item {item_index} terminal is missing")
    starts = [event for event in events if event.get("event") == "attempt_started"]
    require(len(starts) == 1 and len(events) == 2, f"item {item_index} ledger count differs")
    start = starts[0]
    require(
        terminal.get("start_event_id") == start.get("event_id")
        and checkpoint.get("start_event_id") == start.get("event_id")
        and checkpoint.get("terminal_event_id") == terminal.get("event_id")
        and checkpoint.get("attempt_id") == terminal.get("attempt_id")
        and checkpoint.get("attempt_id") == start.get("attempt_id"),
        f"item {item_index} attempt linkage differs",
    )
    require(
        start.get("history_content_sha256") == contract.history_content_hash(item)
        and start.get("history_owner_sha256") == contract.history_owner_hash(item)
        and start.get("preregistration_sha256") == preregistration_sha256
        and start.get("configuration_sha256") == configuration_sha256,
        f"item {item_index} attempt binding differs",
    )
    require(
        terminal.get("answer_input_sha256") == contract.sha256_file(answer_path)
        and terminal.get("answer_input_content_sha256")
        == contract.canonical_hash(answer_input)
        and terminal.get("private_audit_sha256")
        == contract.canonical_hash(expected_private),
        f"item {item_index} terminal evidence differs",
    )
    require(
        start.get("model_calls") == terminal.get("model_calls") == 0
        and start.get("network_calls") == terminal.get("network_calls") == 0,
        f"item {item_index} ledger records a call",
    )
    context = answer_input["context"]
    events_payload = context["events"]
    source_ids = context["source_session_ids"]
    require(
        context["text_sha256"] == contract.sha256_text(context["text"])
        and context["retokenized_context_tokens"]
        == tokenizer.count(context["text"]),
        f"item {item_index} context hash/count differs",
    )
    require(
        source_ids == [event["source_session_id"] for event in events_payload],
        f"item {item_index} delivered source IDs differ",
    )
    require(
        all(
            event["source_session_id"] in item["haystack_session_ids"]
            and event["source_session_occurrence_key"]
            == (
                f"{int(event['dataset_session_index']):04d}:"
                f"{event['source_session_id']}"
            )
            and event["source_mapping_granularity"] == "session"
            and event["turn_recall_claimed"] is False
            and f"session_id={event['source_session_id']}" in context["text"]
            for event in events_payload
        ),
        f"item {item_index} source mapping is invalid",
    )
    prompt = contract.ANSWER_PROMPT.format(
        memories=context["text"], question=item["question"]
    )
    interface = answer_input["answer_protocol_interface"]
    require(
        interface.get("status") == "reserved_not_executed"
        and interface.get("model_calls") == 0
        and interface.get("rendered_prompt_sha256") == contract.sha256_text(prompt)
        and interface.get("rendered_prompt_tokens") == tokenizer.count(prompt)
        and interface.get("rendered_prompt_tokens") <= 123_904,
        f"item {item_index} rendered prompt evidence differs",
    )
    if method == "full_context":
        require(
            source_ids == item["haystack_session_ids"],
            f"item {item_index} full context omits a session",
        )
        require(
            len(events_payload) == len(item["haystack_sessions"])
            and all(event["decision"] == "delivered_full" for event in events_payload)
            and context["budget"]["truncation_allowed"] is False
            and context["budget"]["truncated_events"] == 0,
            f"item {item_index} full context was truncated",
        )
        require(
            context["visible_tokens"] == context["retokenized_context_tokens"],
            f"item {item_index} full-context token count differs",
        )
    else:
        require(
            context["visible_tokens"] <= 20_000
            and context["budget"]["configured_visible_budget_tokens"] == 20_000
            and context["budget"]["cumulative_visible_tokens"]
            == context["visible_tokens"]
            and sum(event["decision"] == "delivered_truncated" for event in events_payload)
            <= 1,
            f"item {item_index} BM25 budget differs",
        )
        if any(event["decision"] == "delivered_truncated" for event in events_payload):
            require(
                events_payload[-1]["decision"] == "delivered_truncated",
                f"item {item_index} continued after a truncated event",
            )
    source_mapping = checkpoint["private_audit"]["source_mapping"]
    require(
        source_mapping["granularity"] == "session"
        and source_mapping["turn_recall_claimed"] is False
        and source_mapping["reference_answer_session_ids"]
        == item["answer_session_ids"]
        and source_mapping["delivered_source_session_ids"]
        == list(dict.fromkeys(source_ids)),
        f"item {item_index} R002 mapping differs",
    )
    matched = [
        value for value in item["answer_session_ids"] if value in source_ids
    ]
    require(
        source_mapping["matched_answer_session_ids"] == matched
        and source_mapping["session_recall"]
        == len(matched) / len(item["answer_session_ids"]),
        f"item {item_index} session recall differs",
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
        "visible_tokens": context["visible_tokens"],
        "rendered_prompt_tokens": interface["rendered_prompt_tokens"],
        "delivered_source_session_ids": source_ids,
        "source_mapping_session_recall": source_mapping["session_recall"],
    }


def audit_run(
    *,
    run_dir: Path,
    dataset_path: Path,
    preregistration_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    preregistration_path = preregistration_path.expanduser().resolve()
    output_path = (
        Path(os.path.abspath(output_path.expanduser())) if output_path else None
    )
    try:
        paths = {
            "run_dir": run_dir,
            "dataset": dataset_path,
            "preregistration": preregistration_path,
        }
        if output_path is not None:
            paths["audit_output"] = output_path
        contract.ensure_distinct_paths(paths)
        contract.reject_symlink_components(run_dir)
    except contract.ContractError as exc:
        raise AuditError(str(exc)) from exc
    require(run_dir.is_dir() and not run_dir.is_symlink(), "run directory is invalid")
    _root_allowlist(run_dir, output_path)
    preregistration, data = audit_preregistration(
        preregistration_path, dataset_path
    )
    manifest_path = run_dir / "run_manifest.json"
    try:
        manifest = contract.read_json(manifest_path)
    except contract.ContractError as exc:
        raise AuditError(str(exc)) from exc
    require(isinstance(manifest, dict), "run manifest is not an object")
    method = manifest.get("identity", {}).get("method")
    require(method in contract.PREPARABLE_METHODS, "run method is invalid")
    expected_identity = _expected_run_identity(
        method=method,
        dataset_path=dataset_path,
        preregistration_path=preregistration_path,
        preregistration=preregistration,
    )
    require(manifest.get("identity") == expected_identity, "run identity differs")
    expected_manifest = {
        "schema_version": contract.RUN_MANIFEST_SCHEMA_VERSION,
        "status": "inputs_complete",
        "run_dir": str(run_dir),
        "completed_items": 500,
        "expected_items": 500,
        "abstention_items": 30,
        "model_calls": 0,
        "network_calls": 0,
        "answer_stage": "not_started",
    }
    for key, value in expected_manifest.items():
        require(manifest.get(key) == value, f"run manifest {key} differs")
    require(
        str(manifest.get("created_at", "")) >= str(preregistration.get("frozen_at", "")),
        "run predates its preregistration",
    )
    items_root = run_dir / "items"
    expected_names = [
        contract.item_directory_name(index, item["question_id"])
        for index, item in enumerate(data)
    ]
    actual_nodes = sorted(items_root.iterdir(), key=lambda path: path.name)
    require(
        [path.name for path in actual_nodes] == expected_names,
        "item directory inventory differs",
    )
    require(
        all(path.is_dir() and not path.is_symlink() for path in actual_nodes),
        "item inventory contains a non-directory",
    )
    try:
        tree_inventory = contract.regular_tree_inventory(items_root)
    except contract.ContractError as exc:
        raise AuditError(str(exc)) from exc
    tree_summary = manifest.get("item_tree_inventory")
    require(isinstance(tree_summary, dict), "item tree inventory is missing")
    require(
        tree_summary
        == {
            "entries": len(tree_inventory),
            "bytes": sum(entry["bytes"] for entry in tree_inventory),
            "root_sha256": contract.inventory_root(tree_inventory),
        },
        "item tree inventory hash differs",
    )
    tokenizer = contract.FormalTokenizer.resolve()
    preregistration_sha256 = contract.sha256_file(preregistration_path)
    configuration_sha256 = contract.canonical_hash(
        contract.method_configuration(method)
    )
    audited = [
        _audit_one_item(
            method=method,
            item=item,
            item_index=index,
            item_dir=items_root / expected_names[index],
            preregistration_sha256=preregistration_sha256,
            configuration_sha256=configuration_sha256,
            tokenizer=tokenizer,
        )
        for index, item in enumerate(data)
    ]
    require(manifest.get("items") == audited, "manifest item inventory differs")
    history_hashes = [item["history_content_sha256"] for item in audited]
    owner_hashes = [item["history_owner_sha256"] for item in audited]
    require(len(set(history_hashes)) == 500, "history content was reused")
    require(len(set(owner_hashes)) == 500, "history owner was reused")
    token_summary = {
        "min": min(item["visible_tokens"] for item in audited),
        "max": max(item["visible_tokens"] for item in audited),
        "sum": sum(item["visible_tokens"] for item in audited),
    }
    prompt_summary = {
        "min": min(item["rendered_prompt_tokens"] for item in audited),
        "max": max(item["rendered_prompt_tokens"] for item in audited),
        "sum": sum(item["rendered_prompt_tokens"] for item in audited),
    }
    require(
        manifest.get("visible_token_summary") == token_summary,
        "visible token summary differs",
    )
    require(
        manifest.get("rendered_prompt_token_summary") == prompt_summary,
        "rendered prompt token summary differs",
    )
    report = {
        "schema_version": "longmemeval-m1-audit-v1",
        "status": "passed",
        "audited_at": contract.utc_now(),
        "benchmark": "LongMemEval-S",
        "method": method,
        "run_dir": str(run_dir),
        "dataset_sha256": contract.sha256_file(dataset_path),
        "preregistration_sha256": preregistration_sha256,
        "run_manifest_sha256": contract.sha256_file(manifest_path),
        "items": len(audited),
        "abstention_items": sum(item["abstention"] for item in audited),
        "unique_history_hashes": len(set(history_hashes)),
        "source_mapping": {
            "requirement": "R002",
            "granularity": "session",
            "turn_recall_claimed": False,
        },
        "visible_token_summary": token_summary,
        "rendered_prompt_token_summary": prompt_summary,
        "full_context_truncations": (
            0 if method == "full_context" else None
        ),
        "item_tree_inventory": tree_summary,
        "model_calls": 0,
        "network_calls": 0,
        "answer_stage": "not_started",
    }
    if output_path is not None:
        try:
            contract.atomic_json_no_clobber(output_path, report)
        except (contract.ContractError, FileExistsError) as exc:
            raise AuditError(f"cannot publish audit: {exc}") from exc
    return report


def audit_backend_plan(
    *,
    plan_path: Path,
    dataset_path: Path,
    preregistration_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    preregistration_path = preregistration_path.expanduser().resolve()
    output_path = (
        Path(os.path.abspath(output_path.expanduser())) if output_path else None
    )
    try:
        paths = {
            "plan": plan_path,
            "dataset": dataset_path,
            "preregistration": preregistration_path,
        }
        if output_path is not None:
            paths["audit_output"] = output_path
        contract.ensure_distinct_paths(paths)
        payload = contract.read_json(plan_path)
        preregistration, data = audit_preregistration(
            preregistration_path, dataset_path
        )
    except contract.ContractError as exc:
        raise AuditError(str(exc)) from exc
    require(isinstance(payload, dict), "backend plan is not an object")
    method = payload.get("method")
    scope = payload.get("scope")
    require(method in contract.PLANNED_BACKENDS, "backend plan method is invalid")
    require(scope in {"formal", "smoke"}, "backend plan scope is invalid")
    require(
        payload.get("schema_version") == contract.BACKEND_PLAN_SCHEMA_VERSION
        and payload.get("status") == "planned_not_executed"
        and payload.get("benchmark") == "LongMemEval-S"
        and payload.get("model_calls") == 0
        and payload.get("network_calls") == 0,
        "backend plan status differs",
    )
    require(
        payload.get("plan_content_sha256")
        == _content_hash(payload, "plan_content_sha256"),
        "backend plan content hash differs",
    )
    require(
        payload.get("configuration") == contract.method_configuration(method),
        "backend plan configuration differs",
    )
    require(
        payload.get("dependencies") == contract.strict_dependency_snapshot(),
        "backend plan dependencies differ",
    )
    require(
        payload.get("dataset_sha256") == contract.EXPECTED_DATASET_SHA256
        and payload.get("preregistration_sha256")
        == contract.sha256_file(preregistration_path)
        and payload.get("preregistration_content_sha256")
        == preregistration["preregistration_content_sha256"],
        "backend plan frozen binding differs",
    )
    require(
        str(payload.get("created_at", "")) >= str(preregistration.get("frozen_at", "")),
        "backend plan predates its preregistration",
    )
    output_root_value = payload.get("output_root")
    require(
        isinstance(output_root_value, str) and output_root_value,
        "backend plan output root is missing",
    )
    output_root = Path(output_root_value)
    try:
        contract.reject_symlink_components(output_root)
    except contract.ContractError as exc:
        raise AuditError(str(exc)) from exc
    items = payload.get("items")
    require(isinstance(items, list), "backend plan items are invalid")
    if scope == "formal":
        expected_indices = list(range(500))
    else:
        require(len(items) == 1, "smoke plan must contain exactly one item")
        expected_indices = [items[0].get("dataset_index")]
        require(
            isinstance(expected_indices[0], int) and 0 <= expected_indices[0] < 500,
            "smoke plan item index is invalid",
        )
    require(
        [item.get("dataset_index") for item in items] == expected_indices,
        "backend plan item indices differ",
    )
    workspaces: set[str] = set()
    for record, index in zip(items, expected_indices):
        source = data[index]
        require(record.get("question_id") == source["question_id"], "plan question id differs")
        require(
            record.get("history_content_sha256") == contract.history_content_hash(source)
            and record.get("history_owner_sha256") == contract.history_owner_hash(source)
            and record.get("history_session_ids") == source["haystack_session_ids"],
            f"plan item {index} history binding differs",
        )
        workspace = record.get("workspace")
        require(isinstance(workspace, dict), f"plan item {index} workspace is invalid")
        require(
            workspace
            == contract.backend_workspace_descriptor(
                method=method,
                output_root=output_root,
                item_index=index,
                question_id=source["question_id"],
            ),
            f"plan item {index} workspace paths differ",
        )
        require(
            workspace.get("history_owner_question_id") == source["question_id"],
            f"plan item {index} workspace owner differs",
        )
        identity = contract.path_identity(Path(workspace["workspace"]))
        require(identity not in workspaces, "backend plan reuses a workspace")
        workspaces.add(identity)
        require(
            record.get("checkpoint") == "checkpoint.json"
            and record.get("attempt_ledger") == "attempts.jsonl"
            and record.get("visible_token_budget") == 20_000,
            f"plan item {index} execution interface differs",
        )
    require(payload.get("item_count") == len(items), "backend plan item count differs")
    report = {
        "schema_version": "longmemeval-m1-backend-plan-audit-v1",
        "status": "passed",
        "audited_at": contract.utc_now(),
        "method": method,
        "scope": scope,
        "item_count": len(items),
        "unique_workspaces": len(workspaces),
        "model_calls": 0,
        "network_calls": 0,
        "execution_status": "not_started",
    }
    if output_path is not None:
        try:
            contract.atomic_json_no_clobber(output_path, report)
        except (contract.ContractError, FileExistsError) as exc:
            raise AuditError(f"cannot publish plan audit: {exc}") from exc
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit LongMemEval-S M1 controlled baseline inputs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("run_dir", type=Path)
    run.add_argument("--dataset", type=Path, default=contract.DEFAULT_DATASET)
    run.add_argument("--preregistration", type=Path, required=True)
    run.add_argument("--output", type=Path)
    plan = subparsers.add_parser("plan")
    plan.add_argument("plan_path", type=Path)
    plan.add_argument("--dataset", type=Path, default=contract.DEFAULT_DATASET)
    plan.add_argument("--preregistration", type=Path, required=True)
    plan.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run":
        report = audit_run(
            run_dir=args.run_dir,
            dataset_path=args.dataset,
            preregistration_path=args.preregistration,
            output_path=args.output,
        )
    else:
        report = audit_backend_plan(
            plan_path=args.plan_path,
            dataset_path=args.dataset,
            preregistration_path=args.preregistration,
            output_path=args.output,
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
