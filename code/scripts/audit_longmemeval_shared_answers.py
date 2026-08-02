#!/usr/bin/env python3
"""Independently audit LongMemEval-S M1 shared-answer artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_longmemeval_m1_backends as backend_input_auditor  # noqa: E402
import audit_longmemeval_m1_baselines as base_input_auditor  # noqa: E402
import controlled_locomo_answer_contract as answer_contract  # noqa: E402
import longmemeval_m1_backend_contract as backend_contract  # noqa: E402
import longmemeval_m1_contract as input_contract  # noqa: E402
import longmemeval_shared_answer_contract as contract  # noqa: E402
from scripts.evaluation.durable_model_ledger import (  # noqa: E402
    DurableLedgerError,
    read_proxy_events,
)
from scripts.gateways import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402


def _inside(root: Path, relative: Any, *, label: str, directory: bool) -> Path:
    if not isinstance(relative, str) or not relative:
        raise contract.SharedAnswerError(f"{label} path is invalid")
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise contract.SharedAnswerError(f"{label} path escapes root")
    candidate = root / raw
    resolved = (
        contract.require_regular_directory(candidate, label=label)
        if directory
        else contract.require_regular_file(candidate, label=label)
    )
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise contract.SharedAnswerError(f"{label} path escapes root") from exc
    return resolved


def _recorded_input_audit_matches_live(
    recorded: Mapping[str, Any], live: Mapping[str, Any]
) -> None:
    for key, value in live.items():
        if key != "audited_at":
            contract.require(
                recorded.get(key) == value,
                f"recorded source input audit {key} differs",
            )


def _audit_base_formal_source(
    *,
    method: str,
    source_run_dir: Path,
    dataset_path: Path,
    preregistration_path: Path,
) -> dict[str, Any]:
    contract.require(
        method in contract.BASE_INPUT_METHODS, "base source method differs"
    )
    source_run_dir = contract.require_regular_directory(
        source_run_dir, label="formal source run"
    )
    dataset_path = contract.require_regular_file(dataset_path, label="dataset")
    preregistration_path = contract.require_regular_file(
        preregistration_path, label="preregistration"
    )
    try:
        live = base_input_auditor.audit_run(
            run_dir=source_run_dir,
            dataset_path=dataset_path,
            preregistration_path=preregistration_path,
            output_path=None,
        )
    except (base_input_auditor.AuditError, input_contract.ContractError) as exc:
        raise contract.SharedAnswerError(f"live source audit failed: {exc}") from exc
    audit_path = contract.require_regular_file(
        source_run_dir / "audit.json", label="recorded source audit"
    )
    manifest_path = contract.require_regular_file(
        source_run_dir / "run_manifest.json", label="source run manifest"
    )
    recorded = contract.read_json(audit_path)
    manifest = contract.read_json(manifest_path)
    contract.require(isinstance(recorded, Mapping), "recorded source audit is invalid")
    contract.require(isinstance(manifest, Mapping), "source manifest is invalid")
    _recorded_input_audit_matches_live(recorded, live)
    items = manifest.get("items")
    tree = manifest.get("item_tree_inventory")
    contract.require(
        manifest.get("status") == "inputs_complete"
        and manifest.get("identity", {}).get("method") == method
        and isinstance(items, list)
        and len(items) == input_contract.EXPECTED_ITEMS
        and isinstance(tree, Mapping),
        "formal source inventory differs",
    )
    descriptor = {
        "mode": "formal_recomputed_input_audit",
        "method": method,
        "source_auditor": contract.source_auditor_identity(method),
        "source_plan": None,
        "source_run_dir": str(source_run_dir),
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": contract.sha256_file(manifest_path),
        "input_audit_path": str(audit_path),
        "input_audit_sha256": contract.sha256_file(audit_path),
        "live_audit_content_sha256": contract.canonical_hash(
            {key: value for key, value in live.items() if key != "audited_at"}
        ),
        "item_tree_root_sha256": tree.get("root_sha256"),
        "item_tree_entries": tree.get("entries"),
        "dataset_path": str(dataset_path),
        "dataset_sha256": contract.sha256_file(dataset_path),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": contract.sha256_file(preregistration_path),
        "items": len(items),
        "abstention_items": manifest.get("abstention_items"),
        "source_item_records_sha256": contract.canonical_hash(items),
    }
    return {"descriptor": descriptor, "manifest": manifest, "items": items}


def _audit_backend_source(
    *,
    method: str,
    source_run_dir: Path,
    dataset_path: Path,
    preregistration_path: Path,
    formal: bool,
) -> dict[str, Any]:
    contract.require(
        method in contract.BACKEND_INPUT_METHODS, "backend source method differs"
    )
    source_run_dir = contract.require_regular_directory(
        source_run_dir, label="backend source run"
    )
    dataset_path = contract.require_regular_file(dataset_path, label="dataset")
    preregistration_path = contract.require_regular_file(
        preregistration_path, label="preregistration"
    )
    manifest_path = contract.require_regular_file(
        source_run_dir / "run_manifest.json", label="backend source run manifest"
    )
    manifest = contract.read_json(manifest_path)
    contract.require(
        isinstance(manifest, Mapping), "backend source manifest is invalid"
    )
    identity = manifest.get("identity")
    contract.require(
        isinstance(identity, Mapping), "backend source identity is missing"
    )
    raw_plan_path = identity.get("plan_path")
    contract.require(
        isinstance(raw_plan_path, str) and raw_plan_path,
        "backend source plan path is missing",
    )
    plan_path = contract.require_regular_file(
        Path(raw_plan_path), label="backend source plan"
    )
    contract.reject_overlapping_paths(
        {
            "source_run": source_run_dir,
            "dataset": dataset_path,
            "preregistration": preregistration_path,
            "backend_plan": plan_path,
        }
    )
    try:
        plan, _, _ = backend_contract.validate_plan(
            plan_path=plan_path,
            dataset_path=dataset_path,
            preregistration_path=preregistration_path,
        )
        live = backend_input_auditor.audit_backend_run(
            plan_path=plan_path,
            dataset_path=dataset_path,
            preregistration_path=preregistration_path,
            output_path=None,
            formal=formal,
        )
    except (
        backend_contract.BackendContractError,
        backend_input_auditor.BackendAuditError,
        input_contract.ContractError,
    ) as exc:
        raise contract.SharedAnswerError(
            f"live backend source audit failed: {exc}"
        ) from exc
    expected_scope = "formal" if formal else "smoke"
    expected_items = input_contract.EXPECTED_ITEMS if formal else 1
    contract.require(
        live.get("status") == "passed"
        and live.get("method") == method
        and live.get("formal") is formal
        and live.get("scope") == expected_scope
        and live.get("items") == expected_items
        and live.get("run_dir") == str(source_run_dir)
        and plan.get("scope") == expected_scope
        and identity.get("method") == method
        and identity.get("formal") is formal
        and identity.get("plan_sha256") == contract.sha256_file(plan_path)
        and identity.get("plan_content_sha256") == plan.get("plan_content_sha256")
        and manifest.get("status") == "inputs_complete"
        and manifest.get("completed_items") == expected_items
        and manifest.get("expected_items") == expected_items,
        "backend source identity differs",
    )
    if formal:
        contract.require(
            live.get("actual_builder_model") == contract.EXPECTED_MODEL
            and int(live.get("model_calls", 0)) > 0
            and int(live.get("network_calls", 0)) > 0,
            "formal backend source lacks model evidence",
        )
    audit_path = contract.require_regular_file(
        source_run_dir / "audit.json", label="recorded backend source audit"
    )
    recorded = contract.read_json(audit_path)
    contract.require(isinstance(recorded, Mapping), "recorded backend audit is invalid")
    _recorded_input_audit_matches_live(recorded, live)
    items = manifest.get("items")
    tree = manifest.get("item_tree_inventory")
    contract.require(
        isinstance(items, list)
        and len(items) == expected_items
        and isinstance(tree, Mapping),
        "backend source item inventory differs",
    )
    workspace_bindings = contract.backend_workspace_bindings(plan)
    workspace_paths = [
        str(binding["workspace"]["workspace"]) for binding in workspace_bindings
    ]
    contract.require(
        live.get("workspace_paths") == workspace_paths
        and live.get("unique_workspace_paths") == expected_items
        and [item.get("workspace") for item in items] == workspace_paths,
        "backend workspace binding differs",
    )
    source_auditor = (
        contract.source_auditor_identity(method)
        if formal
        else contract.synthetic_source_auditor_identity(backend=True)
    )
    descriptor = {
        "mode": (
            "formal_recomputed_backend_input_audit"
            if formal
            else "synthetic_backend_plan_no_network"
        ),
        "method": method,
        "source_auditor": source_auditor,
        "source_plan": {
            "path": str(plan_path),
            "sha256": contract.sha256_file(plan_path),
            "content_sha256": plan["plan_content_sha256"],
            "scope": plan["scope"],
            "workspace_count": len(workspace_bindings),
            "workspace_bindings_sha256": contract.canonical_hash(workspace_bindings),
        },
        "source_run_dir": str(source_run_dir),
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": contract.sha256_file(manifest_path),
        "input_audit_path": str(audit_path),
        "input_audit_sha256": contract.sha256_file(audit_path),
        "live_audit_content_sha256": contract.canonical_hash(
            {key: value for key, value in live.items() if key != "audited_at"}
        ),
        "item_tree_root_sha256": tree.get("root_sha256"),
        "item_tree_entries": tree.get("entries"),
        "dataset_path": str(dataset_path),
        "dataset_sha256": contract.sha256_file(dataset_path),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": contract.sha256_file(preregistration_path),
        "items": len(items),
        "abstention_items": manifest.get("abstention_items"),
        "source_item_records_sha256": contract.canonical_hash(items),
        "workspace_paths_sha256": contract.canonical_hash(workspace_paths),
    }
    return {"descriptor": descriptor, "manifest": manifest, "items": items}


def _audit_formal_source(
    *,
    method: str,
    source_run_dir: Path,
    dataset_path: Path,
    preregistration_path: Path,
) -> dict[str, Any]:
    """Dispatch the source to the auditor assigned by the formal contract."""

    contract.require(
        method in contract.FORMAL_EXECUTABLE_METHODS,
        f"formal source auditor is unavailable for {method}",
    )
    if method in contract.BASE_INPUT_METHODS:
        return _audit_base_formal_source(
            method=method,
            source_run_dir=source_run_dir,
            dataset_path=dataset_path,
            preregistration_path=preregistration_path,
        )
    return _audit_backend_source(
        method=method,
        source_run_dir=source_run_dir,
        dataset_path=dataset_path,
        preregistration_path=preregistration_path,
        formal=True,
    )


def _audit_synthetic_source(*, method: str, source_run_dir: Path) -> dict[str, Any]:
    source_run_dir = contract.require_regular_directory(
        source_run_dir, label="synthetic source run"
    )
    allowed_root = {"items", "run_manifest.json", "audit.json"}
    contract.require(
        {path.name for path in source_run_dir.iterdir()} == allowed_root,
        "synthetic source root inventory differs",
    )
    manifest_path = contract.require_regular_file(
        source_run_dir / "run_manifest.json", label="synthetic source manifest"
    )
    audit_path = contract.require_regular_file(
        source_run_dir / "audit.json", label="synthetic source audit"
    )
    manifest = contract.read_json(manifest_path)
    audit = contract.read_json(audit_path)
    contract.require(
        isinstance(manifest, Mapping)
        and manifest.get("schema_version") == contract.SYNTHETIC_SOURCE_SCHEMA
        and manifest.get("status") == "inputs_complete"
        and manifest.get("mode") == "synthetic_no_network"
        and manifest.get("method") == method
        and manifest.get("completed_items") == 1
        and manifest.get("expected_items") == 1
        and manifest.get("model_calls") == 0
        and manifest.get("network_calls") == 0,
        "synthetic source manifest differs",
    )
    items = manifest.get("items")
    tree = manifest.get("item_tree_inventory")
    contract.require(
        isinstance(items, list)
        and len(items) == 1
        and isinstance(tree, Mapping)
        and tree.get("root_sha256") == contract.canonical_hash(items[0]),
        "synthetic source item tree differs",
    )
    item = items[0]
    expected_item_name = contract.item_directory_name(
        int(item["dataset_index"]), str(item["question_id"])
    )
    item_nodes = list((source_run_dir / "items").iterdir())
    contract.require(
        len(item_nodes) == 1
        and item_nodes[0].name == expected_item_name
        and item.get("item_dir") == expected_item_name,
        "synthetic source item directory differs",
    )
    snapshot = contract.source_item_snapshot(
        source_run_dir=source_run_dir, source_item_record=item
    )
    answer_input = contract.read_json(source_run_dir / snapshot["answer_input_path"])
    contract.require(
        isinstance(answer_input, Mapping), "synthetic answer input is invalid"
    )
    tokenizer = input_contract.FormalTokenizer.resolve()
    contract.validate_frozen_answer_input(
        answer_input,
        method=method,
        tokenizer=tokenizer,
        expected_index=int(item["dataset_index"]),
        expected_question_id=str(item["question_id"]),
    )
    contract.require(
        isinstance(audit, Mapping)
        and audit.get("schema_version") == contract.SYNTHETIC_SOURCE_SCHEMA
        and audit.get("status") == "passed"
        and audit.get("mode") == "synthetic_no_network"
        and audit.get("method") == method
        and audit.get("items") == 1
        and audit.get("item_tree_root_sha256") == tree.get("root_sha256")
        and audit.get("run_manifest_sha256") == contract.sha256_file(manifest_path)
        and audit.get("model_calls") == 0
        and audit.get("network_calls") == 0,
        "synthetic source audit differs",
    )
    descriptor = {
        "mode": "synthetic_no_network",
        "method": method,
        "source_auditor": contract.synthetic_source_auditor_identity(backend=False),
        "source_plan": None,
        "source_run_dir": str(source_run_dir),
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": contract.sha256_file(manifest_path),
        "input_audit_path": str(audit_path),
        "input_audit_sha256": contract.sha256_file(audit_path),
        "live_audit_content_sha256": contract.canonical_hash(audit),
        "item_tree_root_sha256": tree.get("root_sha256"),
        "item_tree_entries": tree.get("entries"),
        "dataset_path": None,
        "dataset_sha256": input_contract.EXPECTED_DATASET_SHA256,
        "preregistration_path": str(
            (
                ROOT / "results/gpt55-longmemeval-m1-baselines-20260714/"
                "formal_matrix_preregistration.json"
            ).resolve()
        ),
        "preregistration_sha256": contract.sha256_file(
            ROOT / "results/gpt55-longmemeval-m1-baselines-20260714/"
            "formal_matrix_preregistration.json"
        ),
        "items": 1,
        "abstention_items": 0,
        "source_item_records_sha256": contract.canonical_hash(items),
    }
    return {"descriptor": descriptor, "manifest": manifest, "items": items}


def _audit_protocol(identity: Mapping[str, Any]) -> dict[str, Any]:
    source = identity.get("source")
    contract.require(isinstance(source, Mapping), "answer source descriptor is missing")
    preregistration_path = contract.require_regular_file(
        Path(str(source.get("preregistration_path"))), label="answer preregistration"
    )
    dataset_value = source.get("dataset_path")
    dataset_path = (
        Path(str(dataset_value))
        if dataset_value is not None
        else ROOT / "benchmarks/longmemeval/data/longmemeval_s_cleaned.json"
    )
    try:
        preregistration, _ = base_input_auditor.audit_preregistration(
            preregistration_path, dataset_path
        )
    except (base_input_auditor.AuditError, input_contract.ContractError) as exc:
        raise contract.SharedAnswerError(str(exc)) from exc
    protocol = contract.shared_protocol(preregistration)
    contract.require(
        identity.get("shared_protocol") == protocol, "shared protocol differs"
    )
    return protocol


def _audit_proxy_evidence(
    *,
    result: Mapping[str, Any],
    proxy_log: Path,
    used_event_ids: set[str],
) -> None:
    logical_call_id = str(result["logical_call_id"])
    answer = result["answer"]
    evidence = result.get("proxy_evidence")
    contract.require(isinstance(evidence, Mapping), "proxy evidence is missing")
    events = read_proxy_events(proxy_log, logical_call_id=logical_call_id)
    contract.require(
        evidence.get("mode") == "exclusive_proxy"
        and evidence.get("events") == events
        and evidence.get("client_http_attempts") == answer.get("client_http_attempts")
        and evidence.get("upstream_http_attempts")
        == answer.get("upstream_http_attempts")
        and evidence.get("unsupported_parameters")
        == answer.get("unsupported_parameters"),
        "stored proxy evidence differs from append-only log",
    )
    prefix = evidence.get("log_prefix")
    contract.require(
        isinstance(prefix, Mapping)
        and Path(str(prefix.get("path"))).resolve() == proxy_log
        and contract.proxy_prefix_matches(proxy_log, prefix),
        "proxy log prefix evidence differs",
    )
    contract.require(
        len(events) == answer.get("client_http_attempts")
        and [event.get("request_sha256") for event in events]
        == answer.get("request_sha256s")
        and [event.get("response_sha256") for event in events]
        == answer.get("response_sha256s")
        and [event.get("event_id") for event in events]
        == answer.get("exclusive_proxy_event_ids"),
        "proxy physical-attempt arrays differ",
    )
    for event in events:
        event_id = event.get("event_id")
        contract.require(
            isinstance(event_id, str)
            and event_id
            and event_id not in used_event_ids
            and event.get("question_id") == result.get("question_id")
            and event.get("logical_call_id") == logical_call_id
            and event.get("requested_model") == contract.EXPECTED_MODEL,
            "proxy event identity differs or is duplicated",
        )
        used_event_ids.add(event_id)
    successes = [event for event in events if event.get("status") == "success"]
    contract.require(
        len(successes) == 1, "proxy has an invalid accepted response count"
    )
    success = successes[0]
    contract.require(
        success.get("response_id") == answer.get("response_id")
        and success.get("actual_model") == answer.get("response_model")
        and success.get("usage") == answer.get("usage"),
        "proxy accepted response identity differs",
    )


def _audit_flex_provider_evidence(
    *,
    output_dir: Path,
    manifest: Mapping[str, Any],
    proxy_log: Path,
    response_ids: list[str],
) -> dict[str, Any]:
    identity = manifest["identity"]
    contract.require(
        identity.get("transport_contract")
        == "openai_gpt55_flex_exclusive_window",
        "formal shared-answer transport contract differs",
    )
    try:
        provider_contract = flex_evidence.validate_recorded_contract(
            identity.get("provider_contract")
        )
    except flex_evidence.EvidenceError as exc:
        raise contract.SharedAnswerError(str(exc)) from exc
    root = contract.require_regular_directory(
        output_dir / "provider_evidence", label="provider evidence root"
    )
    contract.require(
        {path.name for path in root.iterdir()} == {"invocation-0001"},
        "provider evidence invocation inventory differs",
    )
    directory = contract.require_regular_directory(
        root / "invocation-0001", label="provider evidence invocation"
    )
    expected_files = {
        "ready.json",
        "requests.jsonl",
        "gateway_window.json",
        "invocation.json",
    }
    contract.require(
        {path.name for path in directory.iterdir()} == expected_files,
        "provider evidence file inventory differs",
    )
    contract.require(
        proxy_log == directory / "requests.jsonl",
        "shared-answer proxy log is not the fixed provider consumer log",
    )
    invocation_path = contract.require_regular_file(
        directory / "invocation.json", label="provider invocation"
    )
    invocation = contract.read_json(invocation_path)
    contract.require(
        isinstance(invocation, Mapping)
        and invocation.get("schema")
        == "longmemeval-shared-answer-flex-invocation/v1",
        "provider invocation identity differs",
    )
    fixed_paths = {
        "ready": directory / "ready.json",
        "consumer_log": directory / "requests.jsonl",
        "window": directory / "gateway_window.json",
    }
    for field, path in fixed_paths.items():
        binding = invocation.get(field)
        contract.require(
            isinstance(binding, Mapping)
            and binding.get("path") == str(path.relative_to(output_dir))
            and binding.get("sha256") == contract.sha256_file(path),
            f"provider invocation {field} binding differs",
        )
    ready = contract.read_json(directory / "ready.json")
    contract.require(
        isinstance(ready, Mapping)
        and ready.get("run_id") == invocation.get("run_id")
        and ready.get("upstream") == provider_contract["origin"],
        "provider consumer readiness differs",
    )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(proxy_log.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise contract.SharedAnswerError(
                f"provider consumer log line {line_number} is invalid"
            ) from exc
        contract.require(
            isinstance(value, dict)
            and value.get("run_id") == invocation.get("run_id")
            and value.get("status") == "success"
            and value.get("http_status") == 200,
            "provider consumer log contains a failed or foreign request",
        )
        records.append(value)
    window = contract.read_json(directory / "gateway_window.json")
    try:
        report = flex_evidence.audit_window(window, consumer_records=records)
    except flex_evidence.EvidenceError as exc:
        raise contract.SharedAnswerError(str(exc)) from exc
    provider_response_ids = [str(row["response_id"]) for row in records]
    contract.require(
        len(provider_response_ids) == len(set(provider_response_ids))
        and set(provider_response_ids) == set(response_ids),
        "shared-answer responses do not equal the Flex provider window",
    )
    expected_invocation = {
        "provider_model": report["provider_model"],
        "returned_alias": report["returned_alias"],
        "service_tier": report["service_tier"],
        "billing": report["billing"],
        "requests": report["requests"],
        "gateway_request_ids": [row["gateway_request_id"] for row in records],
        "response_ids": provider_response_ids,
        "audit": report,
    }
    for field, expected in expected_invocation.items():
        contract.require(
            invocation.get(field) == expected,
            f"provider invocation {field} differs",
        )
    descriptor = {
        "schema": "longmemeval-shared-answer-flex-evidence/v1",
        "invocation": str(invocation_path.relative_to(output_dir)),
        "invocation_sha256": contract.sha256_file(invocation_path),
        "requests": report["requests"],
        "segment_sha256": report["segment_sha256"],
        "committed_cost_nanos": report["committed_cost_nanos"],
    }
    contract.require(
        manifest.get("provider_evidence") == descriptor,
        "shared-answer manifest provider evidence differs",
    )
    return {**report, "provider_contract": provider_contract}


def _audit_answer_item(
    *,
    output_dir: Path,
    manifest: Mapping[str, Any],
    source_run_dir: Path,
    source_descriptor: Mapping[str, Any],
    source_item_record: Mapping[str, Any],
    proxy_log: Path,
    tokenizer: input_contract.FormalTokenizer,
    used_event_ids: set[str],
) -> dict[str, Any]:
    method = str(manifest["identity"]["method"])
    mode = str(manifest["identity"]["mode"])
    run_id = str(manifest["run_id"])
    snapshot = contract.source_item_snapshot(
        source_run_dir=source_run_dir,
        source_item_record=source_item_record,
    )
    answer_input = contract.read_json(source_run_dir / snapshot["answer_input_path"])
    contract.require(
        isinstance(answer_input, Mapping), "source answer input is invalid"
    )
    validated = contract.validate_frozen_answer_input(
        answer_input,
        method=method,
        tokenizer=tokenizer,
        expected_index=int(source_item_record["dataset_index"]),
        expected_question_id=str(source_item_record["question_id"]),
    )
    item_name = contract.item_directory_name(
        int(validated["dataset_index"]), str(validated["question_id"])
    )
    item_dir = _inside(
        output_dir / "items", item_name, label="answer item", directory=True
    )
    allowed = {
        ".item.lock",
        "input_binding.json",
        "answer_ledger.jsonl",
        "result.json",
        "checkpoint.json",
    }
    contract.require(
        {path.name for path in item_dir.iterdir()} == allowed,
        f"{item_name} answer artifact inventory differs",
    )
    contract.require_regular_file(item_dir / ".item.lock", label=f"{item_name} lock")
    paths = {
        name: contract.require_regular_file(
            item_dir / name, label=f"{item_name} {name}"
        )
        for name in (
            "input_binding.json",
            "answer_ledger.jsonl",
            "result.json",
            "checkpoint.json",
        )
    }
    source_answer_path = source_run_dir / snapshot["answer_input_path"]
    for name, path in paths.items():
        contract.ensure_no_inode_alias(
            source_answer_path, path, label=f"source answer input/{name}"
        )
    binding = contract.read_json(paths["input_binding.json"])
    contract.require(isinstance(binding, Mapping), "input binding is invalid")
    contract.validate_binding_content(binding)
    expected_binding = contract.input_binding(
        run_id=run_id,
        method=method,
        source_descriptor=source_descriptor,
        source_item_record=source_item_record,
        snapshot=snapshot,
        answer_input=answer_input,
        validated=validated,
    )
    contract.require(binding == expected_binding, "input binding differs")
    binding_sha = contract.sha256_file(paths["input_binding.json"])

    logical_call_id = (
        f"{run_id}:{method}:{validated['dataset_index']:04d}:{validated['question_id']}"
    )
    records = answer_contract.audit_ledger(
        paths["answer_ledger.jsonl"], expected_run_id=logical_call_id
    )
    contract.require(
        records
        and records[0].get("event") == "question_started"
        and records[-1].get("event") == "question_completed"
        and sum(record.get("event") == "question_started" for record in records) == 1
        and sum(record.get("event") == "question_completed" for record in records) == 1,
        "answer ledger lifecycle differs",
    )
    started = [
        record
        for record in records
        if record.get("event") == "physical_http_attempt_started"
    ]
    finished = [
        record
        for record in records
        if record.get("event") == "physical_http_attempt_finished"
    ]
    result = contract.read_json(paths["result.json"])
    checkpoint = contract.read_json(paths["checkpoint.json"])
    contract.require(isinstance(result, Mapping), "answer result is invalid")
    contract.require(isinstance(checkpoint, Mapping), "answer checkpoint is invalid")
    answer = result.get("answer")
    prompt = result.get("prompt")
    frozen = result.get("frozen_context")
    contract.require(
        result.get("schema_version") == contract.RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("mode") == mode
        and result.get("run_id") == run_id
        and result.get("logical_call_id") == logical_call_id
        and result.get("method") == method
        and result.get("dataset_index") == validated["dataset_index"]
        and result.get("question_id") == validated["question_id"]
        and result.get("question_type") == validated["question_type"]
        and result.get("abstention") == validated["abstention"]
        and result.get("input_binding_sha256") == binding_sha
        and result.get("source_answer_input_sha256") == snapshot["answer_input_sha256"],
        "answer result identity differs",
    )
    contract.require(
        frozen
        == {
            "sha256": validated["context_sha256"],
            "visible_tokens": validated["context_visible_tokens"],
            "budget_policy": validated["budget_policy"],
            "consumed_verbatim": True,
            "retrieval_ranking_or_truncation_recomputed": False,
        },
        "frozen context consumption record differs",
    )
    contract.require(
        isinstance(prompt, Mapping)
        and prompt.get("sha256") == validated["rendered_prompt_sha256"]
        and prompt.get("local_tokens") == validated["rendered_prompt_tokens"]
        and prompt.get("model_context_limit_tokens")
        == contract.MODEL_CONTEXT_LIMIT_TOKENS
        and prompt.get("answer_completion_reservation_tokens")
        == contract.ANSWER_MAX_TOKENS
        and prompt.get("local_context_check_passed") is True
        and prompt.get("provider_context_check_passed") is True,
        "answer prompt accounting differs",
    )
    contract.require(isinstance(answer, Mapping), "answer response is missing")
    raw_output = answer.get("raw_output")
    usage = answer.get("usage")
    contract.require(
        isinstance(raw_output, str)
        and answer.get("raw_output_sha256") == contract.sha256_text(raw_output)
        and answer.get("text") == answer_contract.extract_answer(raw_output)
        and answer.get("requested_model") == contract.EXPECTED_MODEL
        and answer.get("response_model") == contract.EXPECTED_MODEL
        and isinstance(answer.get("response_id"), str)
        and bool(answer.get("response_id"))
        and answer.get("logical_answer_calls") == 1
        and isinstance(answer.get("client_http_attempts"), int)
        and answer.get("client_http_attempts") >= 1
        and isinstance(answer.get("upstream_http_attempts"), int)
        and answer.get("upstream_http_attempts") >= 1
        and isinstance(usage, Mapping),
        "answer response identity or attempt accounting differs",
    )
    provider_total = usage.get("total_tokens")
    contract.require(
        isinstance(provider_total, int)
        and not isinstance(provider_total, bool)
        and 0 <= provider_total <= contract.MODEL_CONTEXT_LIMIT_TOKENS
        and prompt.get("provider_reported_total_tokens") == provider_total,
        "provider token accounting differs",
    )
    for field in (
        "exclusive_proxy_event_ids",
        "request_sha256s",
        "response_sha256s",
        "unsupported_parameters",
    ):
        contract.require(
            isinstance(answer.get(field), list)
            and all(isinstance(value, str) and value for value in answer[field]),
            f"answer {field} differs",
        )
    attempts = int(answer["client_http_attempts"])
    contract.require(
        len(started) == len(finished) == attempts
        and [record["payload"].get("attempt") for record in started]
        == list(range(1, attempts + 1))
        and [record["payload"].get("attempt") for record in finished]
        == list(range(1, attempts + 1))
        and [record["payload"].get("request_sha256") for record in started]
        == answer.get("request_sha256s")
        and [record["payload"].get("response_sha256") for record in finished]
        == answer.get("response_sha256s")
        and [record["payload"].get("proxy_event_id") for record in finished]
        == answer.get("exclusive_proxy_event_ids")
        and finished[-1]["payload"].get("status") == "accepted"
        and all(record["payload"].get("status") == "error" for record in finished[:-1]),
        "answer ledger physical-attempt accounting differs",
    )
    first_payload = records[0]["payload"]
    last_payload = records[-1]["payload"]
    contract.require(
        first_payload.get("run_id") == run_id
        and first_payload.get("method") == method
        and first_payload.get("question_id") == validated["question_id"]
        and first_payload.get("input_binding_sha256") == binding_sha
        and first_payload.get("answer_input_sha256") == snapshot["answer_input_sha256"]
        and first_payload.get("rendered_prompt_sha256")
        == validated["rendered_prompt_sha256"]
        and first_payload.get("context_consumed_verbatim") is True
        and last_payload.get("run_id") == run_id
        and last_payload.get("question_id") == validated["question_id"]
        and last_payload.get("input_binding_sha256") == binding_sha
        and last_payload.get("rendered_prompt_sha256")
        == validated["rendered_prompt_sha256"]
        and last_payload.get("response_id") == answer.get("response_id")
        and last_payload.get("response_model") == answer.get("response_model")
        and last_payload.get("logical_answer_calls") == 1
        and last_payload.get("client_http_attempts") == attempts
        and last_payload.get("upstream_http_attempts")
        == answer.get("upstream_http_attempts"),
        "answer ledger input/response binding differs",
    )
    _audit_proxy_evidence(
        result=result, proxy_log=proxy_log, used_event_ids=used_event_ids
    )
    contract.require(
        result.get("answer_ledger_sha256")
        == contract.sha256_file(paths["answer_ledger.jsonl"]),
        "result answer-ledger hash differs",
    )
    expected_network = attempts if mode == "formal" else 0
    expected_simulated = attempts if mode == "synthetic_no_network" else 0
    contract.require(
        result.get("model_network_requests") == expected_network
        and result.get("simulated_client_attempts") == expected_simulated,
        "result request-mode accounting differs",
    )
    contract.require(
        checkpoint.get("schema_version") == contract.CHECKPOINT_SCHEMA
        and checkpoint.get("status") == "complete"
        and checkpoint.get("run_id") == run_id
        and checkpoint.get("method") == method
        and checkpoint.get("dataset_index") == validated["dataset_index"]
        and checkpoint.get("question_id") == validated["question_id"]
        and checkpoint.get("logical_call_id") == logical_call_id
        and checkpoint.get("input_binding_sha256") == binding_sha
        and checkpoint.get("answer_ledger_sha256")
        == contract.sha256_file(paths["answer_ledger.jsonl"])
        and checkpoint.get("result_sha256")
        == contract.sha256_file(paths["result.json"])
        and checkpoint.get("response_id") == answer.get("response_id"),
        "answer checkpoint differs",
    )
    contract.assert_source_unchanged(
        source_run_dir=source_run_dir,
        source_item_record=source_item_record,
        expected_snapshot=snapshot,
    )
    return {
        "dataset_index": validated["dataset_index"],
        "question_id": validated["question_id"],
        "question_type": validated["question_type"],
        "abstention": validated["abstention"],
        "response_id": answer["response_id"],
        "logical_answer_calls": 1,
        "client_http_attempts": attempts,
        "upstream_http_attempts": answer["upstream_http_attempts"],
        "network_requests": expected_network,
        "simulated_client_attempts": expected_simulated,
        "context_visible_tokens": validated["context_visible_tokens"],
        "rendered_prompt_tokens": validated["rendered_prompt_tokens"],
    }


def audit_answer_run(output_dir: Path) -> dict[str, Any]:
    output_dir = contract.require_regular_directory(
        output_dir.expanduser().absolute(), label="shared-answer output"
    )
    manifest_path = contract.require_regular_file(
        output_dir / "run_manifest.json", label="shared-answer manifest"
    )
    manifest = contract.read_json(manifest_path)
    contract.require(isinstance(manifest, Mapping), "shared-answer manifest is invalid")
    identity = manifest.get("identity")
    contract.require(isinstance(identity, Mapping), "shared-answer identity is missing")
    mode = identity.get("mode")
    method = identity.get("method")
    contract.require(
        manifest.get("schema_version") == contract.RUN_SCHEMA
        and manifest.get("status") == "complete"
        and mode in {"formal", "synthetic_no_network"}
        and method in contract.FORMAL_METHODS
        and identity.get("output_dir") == str(output_dir)
        and manifest.get("source_hashes") == contract.source_hashes(),
        "shared-answer manifest identity or source hashes differ",
    )
    _audit_protocol(identity)
    proxy_log = contract.require_regular_file(
        Path(str(identity.get("proxy_log"))), label="exclusive proxy log"
    )
    expected_root = {".run.lock", "run_manifest.json", "items"}
    if mode == "formal":
        expected_root.add("provider_evidence")
    if proxy_log.parent == output_dir:
        expected_root.add(proxy_log.name)
    if (output_dir / "audit.json").exists():
        expected_root.add("audit.json")
    contract.require(
        {path.name for path in output_dir.iterdir()} == expected_root,
        "shared-answer root inventory differs",
    )
    contract.require_regular_file(output_dir / ".run.lock", label="answer run lock")
    source_descriptor = identity.get("source")
    contract.require(
        isinstance(source_descriptor, Mapping), "source descriptor is missing"
    )
    source_run_dir = Path(str(source_descriptor.get("source_run_dir"))).resolve()
    contract.require(
        proxy_log != source_run_dir and source_run_dir not in proxy_log.parents,
        "exclusive proxy log overlaps the frozen source run",
    )
    source_mode = source_descriptor.get("mode")
    if mode == "formal":
        contract.require(
            identity.get("transport_contract")
            == "openai_gpt55_flex_exclusive_window"
            and isinstance(identity.get("provider_contract"), Mapping),
            "formal answer lacks its Flex provider contract",
        )
        contract.require(
            source_mode
            in {
                "formal_recomputed_input_audit",
                "formal_recomputed_backend_input_audit",
            },
            "formal answer is bound to a non-formal source",
        )
        dataset_path = Path(str(source_descriptor.get("dataset_path"))).resolve()
        preregistration_path = Path(
            str(source_descriptor.get("preregistration_path"))
        ).resolve()
        source = _audit_formal_source(
            method=str(method),
            source_run_dir=source_run_dir,
            dataset_path=dataset_path,
            preregistration_path=preregistration_path,
        )
    elif source_mode == "synthetic_backend_plan_no_network":
        source = _audit_backend_source(
            method=str(method),
            source_run_dir=source_run_dir,
            dataset_path=Path(str(source_descriptor.get("dataset_path"))).resolve(),
            preregistration_path=Path(
                str(source_descriptor.get("preregistration_path"))
            ).resolve(),
            formal=False,
        )
    else:
        contract.require(
            identity.get("transport_contract") == "deterministic_fake_no_network"
            and identity.get("provider_contract") is None
            and manifest.get("provider_evidence") is None,
            "synthetic answer claims a formal provider transport",
        )
        contract.require(
            source_mode == "synthetic_no_network",
            "synthetic answer source mode differs",
        )
        source = _audit_synthetic_source(
            method=str(method), source_run_dir=source_run_dir
        )
    contract.require(
        source["descriptor"] == source_descriptor,
        "live source descriptor differs from answer binding",
    )
    contract.reject_overlapping_paths(
        {"answer_output": output_dir, "source_run": source_run_dir}
    )
    items = source["items"]
    expected_names = [
        contract.item_directory_name(
            int(item["dataset_index"]), str(item["question_id"])
        )
        for item in items
    ]
    items_root = contract.require_regular_directory(
        output_dir / "items", label="answer items root"
    )
    actual_nodes = sorted(items_root.iterdir(), key=lambda path: path.name)
    contract.require(
        [path.name for path in actual_nodes] == expected_names,
        "answer item directory inventory differs",
    )
    tokenizer = input_contract.FormalTokenizer.resolve()
    contract.require(
        tokenizer.identity == identity.get("shared_protocol", {}).get("tokenizer"),
        "answer tokenizer identity differs",
    )
    used_event_ids: set[str] = set()
    reports = [
        _audit_answer_item(
            output_dir=output_dir,
            manifest=manifest,
            source_run_dir=source_run_dir,
            source_descriptor=source_descriptor,
            source_item_record=item,
            proxy_log=proxy_log,
            tokenizer=tokenizer,
            used_event_ids=used_event_ids,
        )
        for item in items
    ]
    response_ids = [report["response_id"] for report in reports]
    contract.require(
        len(response_ids) == len(set(response_ids)),
        "answer response IDs are duplicated",
    )
    provider_evidence = (
        _audit_flex_provider_evidence(
            output_dir=output_dir,
            manifest=manifest,
            proxy_log=proxy_log,
            response_ids=response_ids,
        )
        if mode == "formal"
        else None
    )
    inventory = contract.question_inventory_summary(reports)
    if mode == "formal":
        contract.require(
            inventory
            == {
                "items": input_contract.EXPECTED_ITEMS,
                "abstention_items": input_contract.EXPECTED_ABSTENTION,
                "question_types": input_contract.EXPECTED_TYPES,
            },
            "formal answer question inventory differs",
        )
    totals = {
        "completed_items": len(reports),
        "logical_answer_calls": sum(
            report["logical_answer_calls"] for report in reports
        ),
        "client_http_attempts": sum(
            report["client_http_attempts"] for report in reports
        ),
        "upstream_http_attempts": sum(
            report["upstream_http_attempts"] for report in reports
        ),
        "network_requests": sum(report["network_requests"] for report in reports),
        "simulated_client_attempts": sum(
            report["simulated_client_attempts"] for report in reports
        ),
    }
    contract.require(
        all(manifest.get(key) == value for key, value in totals.items())
        and manifest.get("expected_items") == len(items)
        and manifest.get("question_inventory") == inventory,
        "shared-answer manifest totals differ",
    )
    return {
        "schema_version": "longmemeval-m1-shared-answer-audit-v1",
        "status": "passed",
        "audited_at": input_contract.utc_now(),
        "benchmark": "LongMemEval-S",
        "mode": mode,
        "method": method,
        "run_id": manifest["run_id"],
        "source_input_unchanged": True,
        "frozen_context_consumed_verbatim": True,
        "retrieval_ranking_or_truncation_recomputed": False,
        "shared_protocol_status": "passed",
        "items": len(reports),
        "abstention_items": inventory["abstention_items"],
        "question_types": inventory["question_types"],
        "logical_answer_calls": totals["logical_answer_calls"],
        "client_http_attempts": totals["client_http_attempts"],
        "upstream_http_attempts": totals["upstream_http_attempts"],
        "network_requests": totals["network_requests"],
        "simulated_client_attempts": totals["simulated_client_attempts"],
        "exclusive_proxy_events": len(used_event_ids),
        "response_ids_unique": True,
        "provider_evidence": provider_evidence,
    }


def audit_synthetic_matrix(output_dir: Path) -> dict[str, Any]:
    output_dir = contract.require_regular_directory(
        output_dir.expanduser().absolute(), label="synthetic matrix"
    )
    expected_root = {
        "matrix_manifest.json",
        "complete.json",
        "sources",
        "answers",
        "backend_plans",
    }
    if (output_dir / "audit.json").exists():
        expected_root.add("audit.json")
    contract.require(
        {path.name for path in output_dir.iterdir()} == expected_root,
        "synthetic matrix root inventory differs",
    )
    manifest = contract.read_json(output_dir / "matrix_manifest.json")
    complete = contract.read_json(output_dir / "complete.json")
    contract.require(
        isinstance(manifest, Mapping)
        and manifest.get("schema_version") == contract.SYNTHETIC_MATRIX_SCHEMA
        and manifest.get("status") == "running"
        and manifest.get("mode") == "synthetic_no_network"
        and manifest.get("formal_methods") == list(contract.FORMAL_METHODS)
        and manifest.get("formal_executable_methods")
        == list(contract.FORMAL_EXECUTABLE_METHODS)
        and manifest.get("backend_source_validation")
        == "isolated_plan_plus_backend_auditor"
        and manifest.get("source_hashes") == contract.source_hashes()
        and manifest.get("model_calls") == 0
        and manifest.get("network_calls") == 0,
        "synthetic matrix manifest differs",
    )
    contract.require(
        isinstance(complete, Mapping)
        and complete.get("schema_version") == contract.SYNTHETIC_MATRIX_SCHEMA
        and complete.get("status") == "complete"
        and complete.get("mode") == "synthetic_no_network"
        and complete.get("formal_methods") == list(contract.FORMAL_METHODS)
        and complete.get("formal_executable_methods")
        == list(contract.FORMAL_EXECUTABLE_METHODS)
        and complete.get("questions") == len(contract.FORMAL_METHODS)
        and complete.get("logical_answer_calls_simulated")
        == len(contract.FORMAL_METHODS)
        and complete.get("model_calls") == 0
        and complete.get("network_calls") == 0,
        "synthetic matrix completion differs",
    )
    plans_root = contract.require_regular_directory(
        output_dir / "backend_plans", label="synthetic backend plans root"
    )
    expected_plan_names = sorted(
        name
        for method in contract.BACKEND_INPUT_METHODS
        for name in (
            f"{method}-formal-reserved.json",
            f"{method}-isolated-smoke.json",
        )
    )
    contract.require(
        sorted(path.name for path in plans_root.iterdir()) == expected_plan_names,
        "synthetic backend plan inventory differs",
    )
    plan_isolation: dict[str, Any] = {}
    for method in contract.BACKEND_INPUT_METHODS:
        formal_plan, _, _ = backend_contract.validate_plan(
            plan_path=plans_root / f"{method}-formal-reserved.json",
            dataset_path=input_contract.DEFAULT_DATASET,
            preregistration_path=(
                ROOT / "results/gpt55-longmemeval-m1-baselines-20260714/"
                "formal_matrix_preregistration.json"
            ),
        )
        smoke_plan, _, _ = backend_contract.validate_plan(
            plan_path=plans_root / f"{method}-isolated-smoke.json",
            dataset_path=input_contract.DEFAULT_DATASET,
            preregistration_path=(
                ROOT / "results/gpt55-longmemeval-m1-baselines-20260714/"
                "formal_matrix_preregistration.json"
            ),
        )
        plan_isolation[method] = contract.validate_backend_plan_isolation(
            formal_plan=formal_plan, smoke_plan=smoke_plan
        )
    contract.require(
        complete.get("backend_plan_isolation") == plan_isolation,
        "synthetic backend plan isolation evidence differs",
    )
    sources_root = contract.require_regular_directory(
        output_dir / "sources", label="synthetic sources root"
    )
    answers_root = contract.require_regular_directory(
        output_dir / "answers", label="synthetic answers root"
    )
    contract.require(
        sorted(path.name for path in sources_root.iterdir())
        == sorted(contract.FORMAL_METHODS)
        and sorted(path.name for path in answers_root.iterdir())
        == sorted(contract.FORMAL_METHODS),
        "synthetic four-method directory inventory differs",
    )
    reports = {
        method: audit_answer_run(answers_root / method)
        for method in contract.FORMAL_METHODS
    }
    contract.require(
        complete.get("method_audit_status")
        == {method: "passed" for method in contract.FORMAL_METHODS}
        and all(
            report.get("mode") == "synthetic_no_network"
            and report.get("method") == method
            and report.get("items") == 1
            and report.get("logical_answer_calls") == 1
            and report.get("network_requests") == 0
            and report.get("simulated_client_attempts") == 1
            for method, report in reports.items()
        ),
        "synthetic method reports differ",
    )
    protocols = [
        contract.read_json(answers_root / method / "run_manifest.json")["identity"][
            "shared_protocol"
        ]
        for method in contract.FORMAL_METHODS
    ]
    contract.require(
        all(protocol == protocols[0] for protocol in protocols[1:])
        and protocols[0] == manifest.get("shared_protocol"),
        "four methods do not share one answer protocol",
    )
    return {
        "schema_version": "longmemeval-m1-shared-answer-synthetic-audit-v1",
        "status": "passed",
        "audited_at": input_contract.utc_now(),
        "mode": "synthetic_no_network",
        "formal_methods": list(contract.FORMAL_METHODS),
        "formal_executable_methods": list(contract.FORMAL_EXECUTABLE_METHODS),
        "shared_protocol_status": "passed",
        "frozen_context_consumed_verbatim": True,
        "retrieval_ranking_or_truncation_recomputed": False,
        "questions": len(contract.FORMAL_METHODS),
        "logical_answer_calls_simulated": len(contract.FORMAL_METHODS),
        "model_calls": 0,
        "network_calls": 0,
        "method_reports": reports,
        "backend_plan_isolation": plan_isolation,
        "formal_status": "blocked_missing_formal_mem0_and_graphiti_runs",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    answer = subparsers.add_parser("answer")
    answer.add_argument("artifact_dir", type=Path)
    matrix = subparsers.add_parser("synthetic-matrix")
    matrix.add_argument("artifact_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = (
        audit_answer_run(args.artifact_dir)
        if args.command == "answer"
        else audit_synthetic_matrix(args.artifact_dir)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        contract.SharedAnswerError,
        answer_contract.ControlledAnswerError,
        input_contract.ContractError,
        base_input_auditor.AuditError,
        backend_input_auditor.BackendAuditError,
        backend_contract.BackendContractError,
        DurableLedgerError,
        FileExistsError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
