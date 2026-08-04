#!/usr/bin/env python3
"""Consume frozen LongMemEval-S M1 inputs with the shared GPT-5.5 answerer.

``preflight`` and ``synthetic-sanity`` make no model or network requests.
Formal ``answer`` execution requires an explicit authorization flag, an
already-running exclusive controlled proxy, and a live independent audit of
the frozen retrieval-input tree.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from baselines.longmemeval_m1 import audit_longmemeval_m1_backends as backend_input_auditor  # noqa: E402
from baselines.longmemeval_m1 import audit_longmemeval_m1_baselines as base_input_auditor  # noqa: E402
from baselines.controlled_locomo import controlled_locomo_answer_contract as answer_contract  # noqa: E402
from baselines.longmemeval_m1 import longmemeval_m1_backend_contract as backend_contract  # noqa: E402
from baselines.longmemeval_m1 import longmemeval_m1_contract as input_contract  # noqa: E402
from baselines.longmemeval_m1 import longmemeval_shared_answer_contract as contract  # noqa: E402
from baselines.controlled_locomo import run_controlled_locomo_answers as shared_answer  # noqa: E402
from baselines.longmemeval_m1 import run_longmemeval_m1_backends as backend_runner  # noqa: E402
from baselines.longmemeval_m1 import run_longmemeval_m1_baselines as base_input_runner  # noqa: E402
from scripts.evaluation.durable_model_ledger import (  # noqa: E402
    DurableLedgerError,
    proxy_evidence,
)
from baselines.gateways import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402


DEFAULT_ROOT = ROOT / "results/gpt55-longmemeval-m1-baselines-20260714"
DEFAULT_DATASET = ROOT / "benchmarks/longmemeval/data/longmemeval_s_cleaned.json"
DEFAULT_PREREGISTRATION = DEFAULT_ROOT / "formal_matrix_preregistration.json"
CONTROLLED_PROXY = SCRIPTS / "controlled_gpt55_run_proxy.py"


def _write_text_no_clobber(path: Path, text: str) -> None:
    try:
        input_contract.reject_symlink_components(path)
    except input_contract.ContractError as exc:
        raise contract.SharedAnswerError(str(exc)) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = text.encode("utf-8")
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise contract.SharedAnswerError(f"short write to {path}")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _audit_artifact_matches_live(
    recorded: Mapping[str, Any], live: Mapping[str, Any]
) -> None:
    for key, value in live.items():
        if key == "audited_at":
            continue
        contract.require(
            recorded.get(key) == value, f"recorded input audit {key} differs"
        )


def _audit_base_formal_source(
    *,
    method: str,
    source_run_dir: Path,
    dataset_path: Path,
    preregistration_path: Path,
) -> dict[str, Any]:
    """Re-audit a deterministic full-context or BM25 source tree."""

    contract.require(
        method in contract.BASE_INPUT_METHODS, "base source method differs"
    )
    source_run_dir = contract.require_regular_directory(
        source_run_dir.expanduser().absolute(), label=f"{method} source run"
    )
    dataset_path = contract.require_regular_file(
        dataset_path.expanduser().absolute(), label="LongMemEval-S dataset"
    )
    preregistration_path = contract.require_regular_file(
        preregistration_path.expanduser().absolute(), label="M1 preregistration"
    )
    contract.reject_overlapping_paths(
        {
            "source_run": source_run_dir,
            "dataset": dataset_path,
            "preregistration": preregistration_path,
        }
    )
    try:
        live = base_input_auditor.audit_run(
            run_dir=source_run_dir,
            dataset_path=dataset_path,
            preregistration_path=preregistration_path,
            output_path=None,
        )
    except (base_input_auditor.AuditError, input_contract.ContractError) as exc:
        raise contract.SharedAnswerError(
            f"{method} live input audit failed: {exc}"
        ) from exc
    contract.require(live.get("status") == "passed", f"{method} input audit failed")
    audit_path = contract.require_regular_file(
        source_run_dir / "audit.json", label=f"{method} recorded input audit"
    )
    recorded = contract.read_json(audit_path)
    contract.require(isinstance(recorded, Mapping), "recorded input audit is invalid")
    _audit_artifact_matches_live(recorded, live)
    manifest_path = contract.require_regular_file(
        source_run_dir / "run_manifest.json", label=f"{method} source manifest"
    )
    manifest = contract.read_json(manifest_path)
    contract.require(isinstance(manifest, Mapping), "source manifest is invalid")
    identity = manifest.get("identity")
    contract.require(
        isinstance(identity, Mapping)
        and identity.get("method") == method
        and manifest.get("status") == "inputs_complete"
        and manifest.get("completed_items") == input_contract.EXPECTED_ITEMS
        and manifest.get("expected_items") == input_contract.EXPECTED_ITEMS,
        f"{method} frozen source manifest differs",
    )
    items = manifest.get("items")
    contract.require(
        isinstance(items, list) and len(items) == input_contract.EXPECTED_ITEMS,
        f"{method} source item inventory differs",
    )
    tree = manifest.get("item_tree_inventory")
    contract.require(
        isinstance(tree, Mapping), "source item-tree descriptor is missing"
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
    return {
        "descriptor": descriptor,
        "manifest": manifest,
        "items": items,
        "live_audit": live,
        "recorded_audit": recorded,
    }


def _audit_backend_source(
    *,
    method: str,
    source_run_dir: Path,
    dataset_path: Path,
    preregistration_path: Path,
    formal: bool,
) -> dict[str, Any]:
    """Re-audit one Mem0/Graphiti run through the backend-only auditor."""

    contract.require(
        method in contract.BACKEND_INPUT_METHODS, "backend source method differs"
    )
    source_run_dir = contract.require_regular_directory(
        source_run_dir.expanduser().absolute(), label=f"{method} backend source run"
    )
    dataset_path = contract.require_regular_file(
        dataset_path.expanduser().absolute(), label="LongMemEval-S dataset"
    )
    preregistration_path = contract.require_regular_file(
        preregistration_path.expanduser().absolute(), label="M1 preregistration"
    )
    manifest_path = contract.require_regular_file(
        source_run_dir / "run_manifest.json", label=f"{method} backend source manifest"
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
        Path(raw_plan_path), label=f"{method} backend source plan"
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
            f"{method} live backend input audit failed: {exc}"
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
        f"{method} backend source identity differs",
    )
    if formal:
        contract.require(
            live.get("actual_builder_model") == contract.EXPECTED_MODEL
            and int(live.get("model_calls", 0)) > 0
            and int(live.get("network_calls", 0)) > 0,
            f"{method} formal backend source lacks model evidence",
        )
    audit_path = contract.require_regular_file(
        source_run_dir / "audit.json", label=f"{method} recorded backend input audit"
    )
    recorded = contract.read_json(audit_path)
    contract.require(isinstance(recorded, Mapping), "recorded backend audit is invalid")
    _audit_artifact_matches_live(recorded, live)
    items = manifest.get("items")
    tree = manifest.get("item_tree_inventory")
    contract.require(
        isinstance(items, list)
        and len(items) == expected_items
        and isinstance(tree, Mapping),
        f"{method} backend source item inventory differs",
    )
    workspace_bindings = contract.backend_workspace_bindings(plan)
    workspace_paths = [
        str(binding["workspace"]["workspace"]) for binding in workspace_bindings
    ]
    contract.require(
        live.get("workspace_paths") == workspace_paths
        and live.get("unique_workspace_paths") == expected_items
        and [item.get("workspace") for item in items] == workspace_paths,
        f"{method} backend workspace binding differs",
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
    return {
        "descriptor": descriptor,
        "manifest": manifest,
        "items": items,
        "live_audit": live,
        "recorded_audit": recorded,
    }


def audit_formal_source(
    *,
    method: str,
    source_run_dir: Path,
    dataset_path: Path,
    preregistration_path: Path,
) -> dict[str, Any]:
    """Dispatch a formal source to its exact independent auditor."""

    contract.require(
        method in contract.FORMAL_EXECUTABLE_METHODS,
        f"{method} has no complete frozen-input runner and independent auditor",
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


def _load_protocol(
    preregistration_path: Path, dataset_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        preregistration, _ = base_input_auditor.audit_preregistration(
            preregistration_path, dataset_path
        )
    except (base_input_auditor.AuditError, input_contract.ContractError) as exc:
        raise contract.SharedAnswerError(str(exc)) from exc
    return preregistration, contract.shared_protocol(preregistration)


def inspect_backend_plan_inventory(
    *,
    root: Path,
    method: str,
    dataset_path: Path,
    preregistration_path: Path,
) -> dict[str, Any]:
    """Read-only validation of canonical formal and legacy smoke plans."""

    contract.require(method in contract.BACKEND_INPUT_METHODS, "backend method differs")
    plan_root = root / "backend_plans"
    formal_path = plan_root / f"{method}-formal.json"
    smoke_path = plan_root / f"{method}-smoke-item0000.json"
    report: dict[str, Any] = {
        "formal_plan_path": str(formal_path),
        "legacy_smoke_plan_path": str(smoke_path),
    }
    plans: dict[str, Mapping[str, Any]] = {}
    for scope, path in (("formal", formal_path), ("smoke", smoke_path)):
        key = "formal_plan_status" if scope == "formal" else "legacy_smoke_plan_status"
        if not path.is_file():
            report[key] = "missing"
            continue
        try:
            plan, _, _ = backend_contract.validate_plan(
                plan_path=path,
                dataset_path=dataset_path,
                preregistration_path=preregistration_path,
            )
        except backend_contract.BackendContractError as exc:
            report[key] = "invalid"
            report[f"{scope}_plan_error"] = str(exc)
            continue
        if plan.get("method") != method or plan.get("scope") != scope:
            report[key] = "invalid_identity"
            continue
        plans[scope] = plan
        report[key] = "valid"
        report[f"{scope}_plan_sha256"] = contract.sha256_file(path)
        report[f"{scope}_plan_content_sha256"] = plan["plan_content_sha256"]
    if set(plans) == {"formal", "smoke"}:
        try:
            report["workspace_isolation"] = contract.validate_backend_plan_isolation(
                formal_plan=plans["formal"], smoke_plan=plans["smoke"]
            )
        except contract.SharedAnswerError as exc:
            report["legacy_smoke_plan_status"] = "rejected_workspace_overlap"
            report["workspace_isolation"] = {
                "status": "rejected_workspace_overlap",
                "reason": str(exc),
            }
    return report


def preflight_matrix(
    *,
    root: Path,
    dataset_path: Path,
    preregistration_path: Path,
) -> dict[str, Any]:
    root = root.expanduser().absolute().resolve()
    dataset_path = dataset_path.expanduser().absolute().resolve()
    preregistration_path = preregistration_path.expanduser().absolute().resolve()
    preregistration, protocol = _load_protocol(preregistration_path, dataset_path)
    methods: dict[str, Any] = {}
    for method in contract.FORMAL_METHODS:
        source_dir = root / method
        plan_preflight = (
            inspect_backend_plan_inventory(
                root=root,
                method=method,
                dataset_path=dataset_path,
                preregistration_path=preregistration_path,
            )
            if method in contract.BACKEND_INPUT_METHODS
            else None
        )
        expected_auditor = contract.source_auditor_identity(method)
        if source_dir.is_dir():
            source = audit_formal_source(
                method=method,
                source_run_dir=source_dir,
                dataset_path=dataset_path,
                preregistration_path=preregistration_path,
            )
            methods[method] = {
                "status": "ready_for_explicit_formal_answer_execution",
                "source": source["descriptor"],
                "expected_source_auditor": expected_auditor,
                "plan_preflight": plan_preflight,
                "model_calls": 0,
                "network_calls": 0,
            }
        elif not source_dir.exists():
            methods[method] = {
                "status": "blocked_missing_frozen_input",
                "expected_source_run_dir": str(source_dir),
                "expected_source_auditor": expected_auditor,
                "plan_preflight": plan_preflight,
                "model_calls": 0,
                "network_calls": 0,
            }
        else:
            methods[method] = {
                "status": "blocked_invalid_source_run_path",
                "source_run_dir": str(source_dir),
                "expected_source_auditor": expected_auditor,
                "plan_preflight": plan_preflight,
                "model_calls": 0,
                "network_calls": 0,
            }
    ready = [
        method
        for method, row in methods.items()
        if row["status"] == "ready_for_explicit_formal_answer_execution"
    ]
    blocked = [method for method in contract.FORMAL_METHODS if method not in ready]
    return {
        "schema_version": contract.PREFLIGHT_SCHEMA,
        "status": "blocked" if blocked else "ready",
        "audited_at": input_contract.utc_now(),
        "benchmark": "LongMemEval-S",
        "formal_methods": list(contract.FORMAL_METHODS),
        "formal_executable_methods": list(contract.FORMAL_EXECUTABLE_METHODS),
        "backend_source_validation": "isolated_plan_plus_backend_auditor",
        "shared_protocol_status": "passed",
        "shared_protocol": protocol,
        "preregistration_content_sha256": preregistration[
            "preregistration_content_sha256"
        ],
        "methods": methods,
        "ready_methods": ready,
        "blocked_methods": blocked,
        "model_calls": 0,
        "network_calls": 0,
    }


def _synthetic_source_input(
    *, method: str, tokenizer: input_contract.FormalTokenizer
) -> dict[str, Any]:
    question_id = f"synthetic-{method}"
    question = "What is the cat's name?"
    context_text = (
        "[session_index=0000 session_id=synthetic-session date=2025-01-01]\n"
        "user: Ari adopted a cat named Pixel."
    )
    context_tokens = tokenizer.count(context_text)
    event = {
        "rank": 0,
        "dataset_session_index": 0,
        "source_session_id": "synthetic-session",
        "source_session_occurrence_key": "0000:synthetic-session",
        "source_mapping_granularity": "session",
        "turn_recall_claimed": False,
        "raw_text_sha256": contract.sha256_text(context_text),
        "delivered_text_sha256": contract.sha256_text(context_text),
        "raw_tokens": context_tokens,
        "delivered_tokens": context_tokens,
        "decision": "delivered_full",
        "cumulative_visible_tokens": context_tokens,
    }
    if method == "full_context":
        budget = {
            "policy": "full_context_unbounded_accounted",
            "configured_visible_budget_tokens": None,
            "truncation_allowed": False,
            "truncated_events": 0,
            "cumulative_visible_tokens": context_tokens,
        }
    else:
        budget = {
            "policy": "hard_visible_total",
            "configured_visible_budget_tokens": contract.VISIBLE_BUDGET_TOKENS,
            "overflow_policy": "truncate_current_then_stop",
            "truncated_events": 0,
            "cumulative_visible_tokens": context_tokens,
            "exhausted": False,
        }
    prompt = input_contract.ANSWER_PROMPT.format(
        memories=context_text, question=question
    )
    return {
        "schema_version": input_contract.SCHEMA_VERSION,
        "benchmark": "LongMemEval-S",
        "method": method,
        "dataset_index": 0,
        "question_id": question_id,
        "question_type": "multi-session",
        "question": question,
        "question_date": "2025-01-02",
        "abstention": False,
        "history_owner_question_id": question_id,
        "history_content_sha256": contract.sha256_text(f"synthetic-history:{method}"),
        "history_owner_sha256": contract.sha256_text(
            f"synthetic-owner:{method}:{question_id}"
        ),
        "context": {
            "rendering": "session_header_role_content_v1",
            "text": context_text,
            "text_sha256": contract.sha256_text(context_text),
            "utf8_bytes": len(context_text.encode("utf-8")),
            "visible_tokens": context_tokens,
            "retokenized_context_tokens": context_tokens,
            "source_session_ids": ["synthetic-session"],
            "events": [event],
            "budget": budget,
        },
        "answer_protocol_interface": {
            "status": "reserved_not_executed",
            "requested_model": contract.EXPECTED_MODEL,
            "prompt_template": "scripts.evaluation.prompts.ANSWER_PROMPT",
            "prompt_template_sha256": contract.sha256_text(
                input_contract.ANSWER_PROMPT
            ),
            "rendered_prompt_sha256": contract.sha256_text(prompt),
            "rendered_prompt_tokens": tokenizer.count(prompt),
            "model_context_limit_tokens": contract.MODEL_CONTEXT_LIMIT_TOKENS,
            "answer_completion_reservation_tokens": contract.ANSWER_MAX_TOKENS,
            "max_rendered_prompt_tokens": contract.MAX_RENDERED_PROMPT_TOKENS,
            "model_calls": 0,
        },
    }


class _SyntheticBackendAdapter:
    """Deterministic backend fixture; it performs no model or network request."""

    observer = None

    def __init__(
        self,
        *,
        workspace: Mapping[str, Any],
        item: Mapping[str, Any],
        item_index: int,
        proxy_base_url: str,
        proxy_log: Path | None,
        ledger: Any,
        artifact_root: Path,
        tokenizer: input_contract.FormalTokenizer,
        formal: bool,
    ) -> None:
        del item, proxy_base_url, proxy_log, ledger, artifact_root, tokenizer, formal
        self.method = "synthetic"
        self.workspace = Path(str(workspace["workspace"]))
        self.item_index = item_index
        self.sessions: list[dict[str, Any]] = []
        self.closed = False

    def operation(self, operation_id: str) -> Any:
        del operation_id
        return nullcontext()

    def ingest(self, session: Mapping[str, Any]) -> dict[str, Any]:
        self.sessions.append(dict(session))
        input_contract.atomic_json_replace(
            self.workspace / "synthetic_store.json",
            {
                "item_index": self.item_index,
                "source_session_ids": [
                    value["source_session_id"] for value in self.sessions
                ],
            },
        )
        return {
            "backend": self.method,
            "source_session_id": session["source_session_id"],
            "dataset_session_index": session["dataset_session_index"],
            "memory_ids": [f"synthetic-{int(session['dataset_session_index']):04d}"],
        }

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        del question
        records: list[dict[str, Any]] = []
        for rank, session in enumerate(self.sessions[:2]):
            text = (
                "literal <|endoftext|> synthetic source fact"
                if rank == 0
                else " x" * 30_000
            )
            records.append(
                {
                    "rank": rank,
                    "backend_record_id": f"synthetic-record-{rank:04d}",
                    "text": text,
                    "score": 1.0 - rank / 10,
                    "source_session_ids": [session["source_session_id"]],
                    "source_dataset_session_indices": [
                        session["dataset_session_index"]
                    ],
                    "source_document_sha256s": [session["text_sha256"]],
                    "backend_trace": {"synthetic_no_network": True},
                }
            )
        return records

    def close(self) -> None:
        if self.closed:
            return
        input_contract.atomic_json_replace(
            self.workspace / "synthetic_store.json",
            {
                "item_index": self.item_index,
                "source_session_ids": [
                    value["source_session_id"] for value in self.sessions
                ],
                "closed": True,
            },
        )
        self.closed = True


def _synthetic_backend_factory(method: str, **kwargs: Any) -> Any:
    adapter = _SyntheticBackendAdapter(**kwargs)
    adapter.method = method
    return adapter


def create_synthetic_backend_source(
    *, matrix_root: Path, method: str
) -> dict[str, Any]:
    """Execute and independently audit an isolated one-item backend plan."""

    contract.require(method in contract.BACKEND_INPUT_METHODS, "backend method differs")
    plans_root = matrix_root / "backend_plans"
    plans_root.mkdir(parents=True, exist_ok=True)
    formal_plan_path = plans_root / f"{method}-formal-reserved.json"
    smoke_plan_path = plans_root / f"{method}-isolated-smoke.json"
    base_input_runner.create_backend_plan(
        method=method,
        scope="formal",
        item_index=None,
        dataset_path=DEFAULT_DATASET,
        preregistration_path=DEFAULT_PREREGISTRATION,
        output_root=matrix_root / "formal_reserved" / method,
        output_path=formal_plan_path,
    )
    base_input_runner.create_backend_plan(
        method=method,
        scope="smoke",
        item_index=0,
        dataset_path=DEFAULT_DATASET,
        preregistration_path=DEFAULT_PREREGISTRATION,
        output_root=matrix_root / "sources",
        output_path=smoke_plan_path,
    )
    formal_plan, _, _ = backend_contract.validate_plan(
        plan_path=formal_plan_path,
        dataset_path=DEFAULT_DATASET,
        preregistration_path=DEFAULT_PREREGISTRATION,
    )
    smoke_plan, _, _ = backend_contract.validate_plan(
        plan_path=smoke_plan_path,
        dataset_path=DEFAULT_DATASET,
        preregistration_path=DEFAULT_PREREGISTRATION,
    )
    plan_isolation = contract.validate_backend_plan_isolation(
        formal_plan=formal_plan, smoke_plan=smoke_plan
    )
    backend_runner.execute_backend_plan(
        plan_path=smoke_plan_path,
        dataset_path=DEFAULT_DATASET,
        preregistration_path=DEFAULT_PREREGISTRATION,
        adapter_factory=_synthetic_backend_factory,
        proxy_base_url="http://127.0.0.1:9/v1",
        proxy_log=None,
        formal=False,
    )
    source_run_dir = matrix_root / "sources" / method
    backend_input_auditor.audit_backend_run(
        plan_path=smoke_plan_path,
        dataset_path=DEFAULT_DATASET,
        preregistration_path=DEFAULT_PREREGISTRATION,
        output_path=source_run_dir / "audit.json",
        formal=False,
    )
    source = _audit_backend_source(
        method=method,
        source_run_dir=source_run_dir,
        dataset_path=DEFAULT_DATASET,
        preregistration_path=DEFAULT_PREREGISTRATION,
        formal=False,
    )
    source["plan_isolation"] = plan_isolation
    return source


def create_synthetic_source(
    *, source_run_dir: Path, method: str, tokenizer: input_contract.FormalTokenizer
) -> dict[str, Any]:
    source_run_dir.mkdir(parents=True, exist_ok=False)
    answer_input = _synthetic_source_input(method=method, tokenizer=tokenizer)
    question_id = str(answer_input["question_id"])
    item_name = contract.item_directory_name(0, question_id)
    item_dir = source_run_dir / "items" / item_name
    item_dir.mkdir(parents=True)
    answer_contract.atomic_json_no_clobber(item_dir / "answer_input.json", answer_input)
    answer_contract.atomic_json_no_clobber(
        item_dir / "checkpoint.json",
        {
            "schema_version": contract.SYNTHETIC_SOURCE_SCHEMA,
            "status": "complete",
            "method": method,
            "dataset_index": 0,
            "question_id": question_id,
            "answer_input_sha256": contract.sha256_file(item_dir / "answer_input.json"),
            "model_calls": 0,
            "network_calls": 0,
        },
    )
    _write_text_no_clobber(
        item_dir / "attempts.jsonl",
        input_contract.canonical_json(
            {
                "schema_version": contract.SYNTHETIC_SOURCE_SCHEMA,
                "status": "complete",
                "method": method,
                "dataset_index": 0,
                "question_id": question_id,
                "model_calls": 0,
                "network_calls": 0,
            }
        )
        + "\n",
    )
    item_record = {
        "dataset_index": 0,
        "question_id": question_id,
        "abstention": False,
        "question_type": "multi-session",
        "item_dir": item_name,
        "answer_input_sha256": contract.sha256_file(item_dir / "answer_input.json"),
        "checkpoint_sha256": contract.sha256_file(item_dir / "checkpoint.json"),
        "ledger_sha256": contract.sha256_file(item_dir / "attempts.jsonl"),
    }
    item_tree_root = contract.canonical_hash(item_record)
    manifest = {
        "schema_version": contract.SYNTHETIC_SOURCE_SCHEMA,
        "status": "inputs_complete",
        "mode": "synthetic_no_network",
        "method": method,
        "completed_items": 1,
        "expected_items": 1,
        "abstention_items": 0,
        "items": [item_record],
        "item_tree_inventory": {
            "entries": 3,
            "root_sha256": item_tree_root,
        },
        "model_calls": 0,
        "network_calls": 0,
    }
    answer_contract.atomic_json_no_clobber(
        source_run_dir / "run_manifest.json", manifest
    )
    audit = {
        "schema_version": contract.SYNTHETIC_SOURCE_SCHEMA,
        "status": "passed",
        "mode": "synthetic_no_network",
        "method": method,
        "items": 1,
        "item_tree_root_sha256": item_tree_root,
        "run_manifest_sha256": contract.sha256_file(
            source_run_dir / "run_manifest.json"
        ),
        "model_calls": 0,
        "network_calls": 0,
    }
    answer_contract.atomic_json_no_clobber(source_run_dir / "audit.json", audit)
    descriptor = {
        "mode": "synthetic_no_network",
        "method": method,
        "source_auditor": contract.synthetic_source_auditor_identity(backend=False),
        "source_plan": None,
        "source_run_dir": str(source_run_dir.resolve()),
        "run_manifest_path": str((source_run_dir / "run_manifest.json").resolve()),
        "run_manifest_sha256": contract.sha256_file(
            source_run_dir / "run_manifest.json"
        ),
        "input_audit_path": str((source_run_dir / "audit.json").resolve()),
        "input_audit_sha256": contract.sha256_file(source_run_dir / "audit.json"),
        "live_audit_content_sha256": contract.canonical_hash(audit),
        "item_tree_root_sha256": item_tree_root,
        "item_tree_entries": 3,
        "dataset_path": None,
        "dataset_sha256": input_contract.EXPECTED_DATASET_SHA256,
        "preregistration_path": str(DEFAULT_PREREGISTRATION.resolve()),
        "preregistration_sha256": contract.sha256_file(DEFAULT_PREREGISTRATION),
        "items": 1,
        "abstention_items": 0,
        "source_item_records_sha256": contract.canonical_hash([item_record]),
    }
    return {"descriptor": descriptor, "manifest": manifest, "items": [item_record]}


def validate_synthetic_source(source_run_dir: Path, method: str) -> dict[str, Any]:
    source_run_dir = contract.require_regular_directory(
        source_run_dir, label="synthetic source run"
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
    contract.require(
        isinstance(items, list) and len(items) == 1, "synthetic source items differ"
    )
    tree = manifest.get("item_tree_inventory")
    contract.require(
        isinstance(tree, Mapping), "synthetic item-tree descriptor differs"
    )
    contract.require(
        isinstance(audit, Mapping)
        and audit.get("schema_version") == contract.SYNTHETIC_SOURCE_SCHEMA
        and audit.get("status") == "passed"
        and audit.get("method") == method
        and audit.get("run_manifest_sha256") == contract.sha256_file(manifest_path)
        and audit.get("item_tree_root_sha256") == tree.get("root_sha256")
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
        "preregistration_path": str(DEFAULT_PREREGISTRATION.resolve()),
        "preregistration_sha256": contract.sha256_file(DEFAULT_PREREGISTRATION),
        "items": 1,
        "abstention_items": 0,
        "source_item_records_sha256": contract.canonical_hash(items),
    }
    return {"descriptor": descriptor, "manifest": manifest, "items": items}


def _manifest_identity(
    *,
    mode: str,
    method: str,
    output_dir: Path,
    source_descriptor: Mapping[str, Any],
    protocol: Mapping[str, Any],
    proxy_log: Path,
    provider_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "benchmark": "LongMemEval-S",
        "milestone": "M1 controlled baselines shared answer",
        "mode": mode,
        "method": method,
        "output_dir": str(output_dir.resolve()),
        "source": copy.deepcopy(dict(source_descriptor)),
        "shared_protocol": copy.deepcopy(dict(protocol)),
        "proxy_log": str(proxy_log.resolve()),
        "transport_contract": (
            "openai_gpt55_flex_exclusive_window"
            if mode == "formal"
            else "deterministic_fake_no_network"
        ),
        "provider_contract": (
            copy.deepcopy(dict(provider_contract))
            if provider_contract is not None
            else None
        ),
        "answer_client": (
            "controlled_exclusive_http"
            if mode == "formal"
            else "deterministic_fake_no_network"
        ),
    }


def _create_or_validate_manifest(
    *,
    output_dir: Path,
    identity: Mapping[str, Any],
    expected_items: int,
) -> dict[str, Any]:
    manifest_path = output_dir / "run_manifest.json"
    hashes = contract.source_hashes()
    run_id = f"lme-shared-answer-{contract.canonical_hash(identity)[:24]}"
    if manifest_path.exists():
        contract.require_regular_file(
            manifest_path, label="existing shared-answer manifest"
        )
        manifest = contract.read_json(manifest_path)
        contract.require(isinstance(manifest, Mapping), "answer manifest is invalid")
        contract.require(
            manifest.get("schema_version") == contract.RUN_SCHEMA
            and manifest.get("run_id") == run_id
            and manifest.get("identity") == dict(identity)
            and manifest.get("source_hashes") == hashes
            and manifest.get("expected_items") == expected_items,
            "existing answer run identity differs",
        )
        return dict(manifest)
    manifest = {
        "schema_version": contract.RUN_SCHEMA,
        "status": "running",
        "run_id": run_id,
        "created_at": input_contract.utc_now(),
        "updated_at": input_contract.utc_now(),
        "identity": copy.deepcopy(dict(identity)),
        "source_hashes": hashes,
        "expected_items": expected_items,
        "completed_items": 0,
        "logical_answer_calls": 0,
        "client_http_attempts": 0,
        "upstream_http_attempts": 0,
        "network_requests": 0,
        "simulated_client_attempts": 0,
    }
    answer_contract.atomic_json_no_clobber(manifest_path, manifest)
    return manifest


def _read_source_input(
    *,
    source_run_dir: Path,
    source_item_record: Mapping[str, Any],
    method: str,
    tokenizer: input_contract.FormalTokenizer,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    return dict(answer_input), validated, snapshot


def _validate_proxy_call(
    *,
    call: shared_answer.AnswerCallResult,
    evidence: Mapping[str, Any],
    question_id: str,
    logical_call_id: str,
) -> None:
    events = evidence.get("events")
    contract.require(isinstance(events, list), "proxy evidence events are invalid")
    contract.require(
        len(events) == call.client_http_attempts
        and evidence.get("client_http_attempts") == call.client_http_attempts
        and evidence.get("upstream_http_attempts") == call.upstream_http_attempts
        and evidence.get("unsupported_parameters") == list(call.unsupported_parameters),
        "proxy attempt totals differ from answer client",
    )
    contract.require(
        [event.get("request_sha256") for event in events] == list(call.request_sha256s)
        and [event.get("response_sha256") for event in events]
        == list(call.response_sha256s)
        and [event.get("event_id") for event in events] == list(call.proxy_event_ids),
        "proxy request/response linkage differs",
    )
    successful = [event for event in events if event.get("status") == "success"]
    contract.require(
        len(successful) == 1, "answer call lacks one successful proxy event"
    )
    success = successful[0]
    contract.require(
        all(
            event.get("question_id") == question_id
            and event.get("logical_call_id") == logical_call_id
            and event.get("requested_model") == contract.EXPECTED_MODEL
            for event in events
        )
        and success.get("response_id") == call.response_id
        and success.get("actual_model") == call.response_model
        and success.get("usage") == call.usage,
        "accepted proxy response identity differs",
    )


def _validate_completed_checkpoint(
    *, item_dir: Path, expected_question_id: str, expected_run_id: str
) -> dict[str, Any]:
    checkpoint_path = item_dir / "checkpoint.json"
    checkpoint = contract.read_json(checkpoint_path)
    contract.require(
        isinstance(checkpoint, Mapping)
        and checkpoint.get("schema_version") == contract.CHECKPOINT_SCHEMA
        and checkpoint.get("status") == "complete"
        and checkpoint.get("question_id") == expected_question_id
        and checkpoint.get("run_id") == expected_run_id,
        "completed answer checkpoint differs",
    )
    for filename, hash_field in (
        ("input_binding.json", "input_binding_sha256"),
        ("answer_ledger.jsonl", "answer_ledger_sha256"),
        ("result.json", "result_sha256"),
    ):
        path = contract.require_regular_file(item_dir / filename, label=filename)
        contract.require(
            checkpoint.get(hash_field) == contract.sha256_file(path),
            f"completed {filename} hash differs",
        )
    return dict(checkpoint)


def answer_one_frozen_item(
    *,
    run_id: str,
    mode: str,
    method: str,
    output_dir: Path,
    source_run_dir: Path,
    source_descriptor: Mapping[str, Any],
    source_item_record: Mapping[str, Any],
    tokenizer: input_contract.FormalTokenizer,
    client: shared_answer.AnswerClient,
    proxy_log: Path,
) -> dict[str, Any]:
    answer_input, validated, snapshot = _read_source_input(
        source_run_dir=source_run_dir,
        source_item_record=source_item_record,
        method=method,
        tokenizer=tokenizer,
    )
    item_name = contract.item_directory_name(
        int(validated["dataset_index"]), str(validated["question_id"])
    )
    item_dir = output_dir / "items" / item_name
    item_dir.mkdir(parents=True, exist_ok=True)
    try:
        input_contract.reject_symlink_components(item_dir)
    except input_contract.ContractError as exc:
        raise contract.SharedAnswerError(str(exc)) from exc
    if item_dir.is_symlink():
        raise contract.SharedAnswerError("answer item directory is a symlink")
    checkpoint_path = item_dir / "checkpoint.json"
    result_path = item_dir / "result.json"
    binding_path = item_dir / "input_binding.json"
    ledger_path = item_dir / "answer_ledger.jsonl"
    logical_call_id = (
        f"{run_id}:{method}:{validated['dataset_index']:04d}:{validated['question_id']}"
    )
    with answer_contract.FileLock(item_dir / ".item.lock"):
        if checkpoint_path.exists():
            _validate_completed_checkpoint(
                item_dir=item_dir,
                expected_question_id=str(validated["question_id"]),
                expected_run_id=run_id,
            )
            result = contract.read_json(result_path)
            contract.require(isinstance(result, Mapping), "completed result is invalid")
            return dict(result)
        if result_path.exists() or result_path.is_symlink():
            raise contract.SharedAnswerError(
                f"{validated['question_id']} has a result without a checkpoint; use a new output root"
            )
        if ledger_path.exists():
            contract.require_regular_file(ledger_path, label="existing answer ledger")
            records = answer_contract.audit_ledger(
                ledger_path, expected_run_id=logical_call_id
            )
            if records:
                raise contract.SharedAnswerError(
                    f"{validated['question_id']} has an incomplete prior answer attempt; refusing repeat"
                )
        binding = contract.input_binding(
            run_id=run_id,
            method=method,
            source_descriptor=source_descriptor,
            source_item_record=source_item_record,
            snapshot=snapshot,
            answer_input=answer_input,
            validated=validated,
        )
        if binding_path.exists():
            contract.require_regular_file(binding_path, label="existing input binding")
            recorded_binding = contract.read_json(binding_path)
            contract.require(
                recorded_binding == binding, "existing input binding differs"
            )
        else:
            answer_contract.atomic_json_no_clobber(binding_path, binding)
        binding_sha = contract.sha256_file(binding_path)
        with answer_contract.DurableLedger(
            ledger_path, run_id=logical_call_id
        ) as ledger:
            ledger.append(
                "question_started",
                {
                    "run_id": run_id,
                    "method": method,
                    "dataset_index": validated["dataset_index"],
                    "question_id": validated["question_id"],
                    "input_binding_sha256": binding_sha,
                    "answer_input_sha256": snapshot["answer_input_sha256"],
                    "rendered_prompt_sha256": validated["rendered_prompt_sha256"],
                    "context_consumed_verbatim": True,
                },
            )
            call = client.complete(
                prompt=str(validated["prompt"]),
                question_id=str(validated["question_id"]),
                logical_call_id=logical_call_id,
                ledger=ledger,
            )
            contract.require(
                call.logical_calls == 1
                and call.response_model == contract.EXPECTED_MODEL
                and isinstance(call.response_id, str)
                and bool(call.response_id),
                "shared answer response identity differs",
            )
            provider_total = call.usage.get("total_tokens")
            contract.require(
                isinstance(provider_total, int)
                and not isinstance(provider_total, bool)
                and 0 <= provider_total <= contract.MODEL_CONTEXT_LIMIT_TOKENS,
                "provider total-token usage is invalid",
            )
            evidence = proxy_evidence(
                proxy_log,
                logical_call_id=logical_call_id,
                formal=True,
            )
            _validate_proxy_call(
                call=call,
                evidence=evidence,
                question_id=str(validated["question_id"]),
                logical_call_id=logical_call_id,
            )
            ledger.append(
                "question_completed",
                {
                    "run_id": run_id,
                    "method": method,
                    "dataset_index": validated["dataset_index"],
                    "question_id": validated["question_id"],
                    "input_binding_sha256": binding_sha,
                    "rendered_prompt_sha256": validated["rendered_prompt_sha256"],
                    "response_id": call.response_id,
                    "response_model": call.response_model,
                    "logical_answer_calls": call.logical_calls,
                    "client_http_attempts": call.client_http_attempts,
                    "upstream_http_attempts": call.upstream_http_attempts,
                    "proxy_log_prefix": evidence["log_prefix"],
                },
            )
        contract.assert_source_unchanged(
            source_run_dir=source_run_dir,
            source_item_record=source_item_record,
            expected_snapshot=snapshot,
        )
        result = {
            "schema_version": contract.RESULT_SCHEMA,
            "status": "complete",
            "mode": mode,
            "run_id": run_id,
            "logical_call_id": logical_call_id,
            "method": method,
            "dataset_index": validated["dataset_index"],
            "question_id": validated["question_id"],
            "question_type": validated["question_type"],
            "abstention": validated["abstention"],
            "input_binding_sha256": binding_sha,
            "source_answer_input_sha256": snapshot["answer_input_sha256"],
            "frozen_context": {
                "sha256": validated["context_sha256"],
                "visible_tokens": validated["context_visible_tokens"],
                "budget_policy": validated["budget_policy"],
                "consumed_verbatim": True,
                "retrieval_ranking_or_truncation_recomputed": False,
            },
            "prompt": {
                "sha256": validated["rendered_prompt_sha256"],
                "local_tokens": validated["rendered_prompt_tokens"],
                "model_context_limit_tokens": contract.MODEL_CONTEXT_LIMIT_TOKENS,
                "answer_completion_reservation_tokens": contract.ANSWER_MAX_TOKENS,
                "local_context_check_passed": True,
                "provider_reported_total_tokens": provider_total,
                "provider_context_check_passed": True,
            },
            "answer": {
                "text": answer_contract.extract_answer(call.raw_output),
                "raw_output": call.raw_output,
                "raw_output_sha256": contract.sha256_text(call.raw_output),
                "requested_model": contract.EXPECTED_MODEL,
                "response_model": call.response_model,
                "response_id": call.response_id,
                "usage": call.usage,
                "logical_answer_calls": call.logical_calls,
                "client_http_attempts": call.client_http_attempts,
                "upstream_http_attempts": call.upstream_http_attempts,
                "exclusive_proxy_event_ids": list(call.proxy_event_ids),
                "request_sha256s": list(call.request_sha256s),
                "response_sha256s": list(call.response_sha256s),
                "unsupported_parameters": list(call.unsupported_parameters),
            },
            "proxy_evidence": evidence,
            "answer_ledger_sha256": contract.sha256_file(ledger_path),
            "model_network_requests": (
                call.client_http_attempts if mode == "formal" else 0
            ),
            "simulated_client_attempts": (
                call.client_http_attempts if mode == "synthetic_no_network" else 0
            ),
        }
        answer_contract.atomic_json_no_clobber(result_path, result)
        checkpoint = {
            "schema_version": contract.CHECKPOINT_SCHEMA,
            "status": "complete",
            "run_id": run_id,
            "method": method,
            "dataset_index": validated["dataset_index"],
            "question_id": validated["question_id"],
            "logical_call_id": logical_call_id,
            "input_binding_sha256": binding_sha,
            "answer_ledger_sha256": contract.sha256_file(ledger_path),
            "result_sha256": contract.sha256_file(result_path),
            "response_id": call.response_id,
        }
        answer_contract.atomic_json_no_clobber(checkpoint_path, checkpoint)
        return result


def _aggregate_results(
    results: Sequence[Mapping[str, Any]], *, mode: str
) -> dict[str, int]:
    return {
        "completed_items": len(results),
        "logical_answer_calls": sum(
            int(result["answer"]["logical_answer_calls"]) for result in results
        ),
        "client_http_attempts": sum(
            int(result["answer"]["client_http_attempts"]) for result in results
        ),
        "upstream_http_attempts": sum(
            int(result["answer"]["upstream_http_attempts"]) for result in results
        ),
        "network_requests": sum(
            int(result["model_network_requests"]) for result in results
        ),
        "simulated_client_attempts": sum(
            int(result["simulated_client_attempts"]) for result in results
        ),
    }


def _without_audit_times(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_audit_times(item)
            for key, item in value.items()
            if key != "audited_at"
        }
    if isinstance(value, list):
        return [_without_audit_times(item) for item in value]
    return value


def _publish_audit_no_clobber(path: Path, report: Mapping[str, Any]) -> None:
    if path.exists():
        contract.require_regular_file(path, label="existing answer audit")
        recorded = contract.read_json(path)
        contract.require(
            _without_audit_times(recorded) == _without_audit_times(report),
            "existing answer audit differs from live independent audit",
        )
        return
    answer_contract.atomic_json_no_clobber(path, report)


def run_answer_inventory(
    *,
    mode: str,
    method: str,
    output_dir: Path,
    source_run_dir: Path,
    source: Mapping[str, Any],
    protocol: Mapping[str, Any],
    client: shared_answer.AnswerClient,
    proxy_log: Path,
    provider_contract: Mapping[str, Any] | None = None,
    publish_audit: bool = True,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().absolute()
    source_run_dir = source_run_dir.expanduser().absolute().resolve()
    proxy_log = contract.require_regular_file(proxy_log, label="exclusive proxy log")
    if proxy_log == source_run_dir or source_run_dir in proxy_log.parents:
        raise contract.SharedAnswerError(
            "exclusive proxy log must not be inside the frozen source run"
        )
    contract.reject_overlapping_paths(
        {"answer_output": output_dir, "source_run": source_run_dir}
    )
    if output_dir.is_symlink():
        raise contract.SharedAnswerError("answer output must not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    items = source.get("items")
    descriptor = source.get("descriptor")
    contract.require(
        isinstance(items, list) and items, "source item inventory is empty"
    )
    contract.require(isinstance(descriptor, Mapping), "source descriptor is missing")
    identity = _manifest_identity(
        mode=mode,
        method=method,
        output_dir=output_dir,
        source_descriptor=descriptor,
        protocol=protocol,
        proxy_log=proxy_log,
        provider_contract=provider_contract,
    )
    with answer_contract.FileLock(output_dir / ".run.lock"):
        manifest = _create_or_validate_manifest(
            output_dir=output_dir,
            identity=identity,
            expected_items=len(items),
        )
        if manifest.get("status") == "complete":
            if not publish_audit:
                return manifest
            from baselines.longmemeval_m1.audit_longmemeval_shared_answers import audit_answer_run  # noqa: PLC0415

            report = audit_answer_run(output_dir)
            _publish_audit_no_clobber(output_dir / "audit.json", report)
            return report
        tokenizer = input_contract.FormalTokenizer.resolve()
        contract.require(
            tokenizer.identity == protocol.get("tokenizer"),
            "answer tokenizer differs from preregistration",
        )
        results = [
            answer_one_frozen_item(
                run_id=str(manifest["run_id"]),
                mode=mode,
                method=method,
                output_dir=output_dir,
                source_run_dir=source_run_dir,
                source_descriptor=descriptor,
                source_item_record=item,
                tokenizer=tokenizer,
                client=client,
                proxy_log=proxy_log,
            )
            for item in items
        ]
        totals = _aggregate_results(results, mode=mode)
        complete_manifest = {
            **manifest,
            "status": "complete",
            "updated_at": input_contract.utc_now(),
            "completed_at": input_contract.utc_now(),
            **totals,
            "question_inventory": contract.question_inventory_summary(results),
        }
        answer_contract.atomic_json_replace(
            output_dir / "run_manifest.json", complete_manifest
        )
    if not publish_audit:
        return complete_manifest
    from baselines.longmemeval_m1.audit_longmemeval_shared_answers import audit_answer_run  # noqa: PLC0415

    report = audit_answer_run(output_dir)
    _publish_audit_no_clobber(output_dir / "audit.json", report)
    return report


def _wait_for_controlled_proxy(
    *, process: subprocess.Popen[Any], ready_path: Path, expected_run_id: str
) -> dict[str, Any]:
    deadline = time.monotonic() + 20
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise contract.SharedAnswerError(
                "controlled Flex proxy exited before readiness"
            )
        if ready_path.is_file():
            try:
                value = contract.read_json(ready_path)
                contract.require(
                    isinstance(value, Mapping)
                    and value.get("run_id") == expected_run_id
                    and value.get("upstream") is not None,
                    "controlled Flex proxy readiness differs",
                )
                base_url = str(value.get("base_url", ""))
                contract.require(
                    base_url.startswith("http://127.0.0.1:")
                    and base_url.endswith("/v1"),
                    "controlled Flex proxy base URL is invalid",
                )
                with opener.open(
                    f"{base_url.removesuffix('/v1')}/healthz", timeout=3
                ) as response:  # noqa: S310
                    health = json.loads(response.read())
                contract.require(
                    response.status == 200
                    and health.get("status") == "ok"
                    and health.get("run_id") == expected_run_id,
                    "controlled Flex proxy health differs",
                )
                return dict(value)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        time.sleep(0.05)
    raise contract.SharedAnswerError(
        f"controlled Flex proxy readiness timed out: {last_error}"
    )


def _read_controlled_proxy_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise contract.SharedAnswerError(
                f"controlled Flex proxy line {line_number} is invalid"
            ) from exc
        contract.require(
            isinstance(value, dict)
            and value.get("status") == "success"
            and value.get("http_status") == 200,
            "formal controlled Flex proxy contains a failed request",
        )
        records.append(value)
    return records


def _stop_controlled_proxy(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def _publish_provider_evidence(
    *,
    output_dir: Path,
    run_id: str,
    ready_path: Path,
    proxy_log: Path,
    window: Mapping[str, Any],
) -> dict[str, Any]:
    directory = ready_path.parent
    records = _read_controlled_proxy_records(proxy_log)
    try:
        report = flex_evidence.audit_window(window, consumer_records=records)
    except flex_evidence.EvidenceError as exc:
        raise contract.SharedAnswerError(str(exc)) from exc
    window_path = directory / "gateway_window.json"
    answer_contract.atomic_json_no_clobber(window_path, dict(window))
    invocation = {
        "schema": "longmemeval-shared-answer-flex-invocation/v1",
        "run_id": run_id,
        "provider_model": report["provider_model"],
        "returned_alias": report["returned_alias"],
        "service_tier": report["service_tier"],
        "billing": report["billing"],
        "requests": report["requests"],
        "gateway_request_ids": [row["gateway_request_id"] for row in records],
        "response_ids": [row["response_id"] for row in records],
        "ready": {
            "path": str(ready_path.relative_to(output_dir)),
            "sha256": contract.sha256_file(ready_path),
        },
        "consumer_log": {
            "path": str(proxy_log.relative_to(output_dir)),
            "sha256": contract.sha256_file(proxy_log),
        },
        "window": {
            "path": str(window_path.relative_to(output_dir)),
            "sha256": contract.sha256_file(window_path),
        },
        "audit": report,
    }
    invocation_path = directory / "invocation.json"
    answer_contract.atomic_json_no_clobber(invocation_path, invocation)
    return {
        "schema": "longmemeval-shared-answer-flex-evidence/v1",
        "invocation": str(invocation_path.relative_to(output_dir)),
        "invocation_sha256": contract.sha256_file(invocation_path),
        "requests": report["requests"],
        "segment_sha256": report["segment_sha256"],
        "committed_cost_nanos": report["committed_cost_nanos"],
    }


def run_formal_answers(
    *,
    method: str,
    source_run_dir: Path,
    output_dir: Path,
    dataset_path: Path,
    preregistration_path: Path,
    gateway_root: Path,
    retries: int,
    allow_model_requests: bool,
) -> dict[str, Any]:
    if not allow_model_requests:
        raise contract.SharedAnswerError(
            "formal answers require the explicit --allow-model-requests flag"
        )
    contract.require(
        method in contract.FORMAL_EXECUTABLE_METHODS, f"{method} is blocked"
    )
    contract.reject_overlapping_paths(
        {
            "answer_output": output_dir.expanduser().absolute(),
            "source_run": source_run_dir.expanduser().absolute(),
        }
    )
    _, protocol = _load_protocol(preregistration_path, dataset_path)
    source = audit_formal_source(
        method=method,
        source_run_dir=source_run_dir,
        dataset_path=dataset_path,
        preregistration_path=preregistration_path,
    )
    output_dir = output_dir.expanduser().absolute()
    existing_manifest = output_dir / "run_manifest.json"
    if existing_manifest.is_file():
        existing = contract.read_json(existing_manifest)
        if isinstance(existing, Mapping) and existing.get("status") == "complete":
            from baselines.longmemeval_m1.audit_longmemeval_shared_answers import audit_answer_run  # noqa: PLC0415

            return audit_answer_run(output_dir)
        raise contract.SharedAnswerError(
            "existing formal answer output is incomplete; use a new output root"
        )
    gateway_root = gateway_root.expanduser().resolve()
    provider_lock = None
    process: subprocess.Popen[Any] | None = None
    try:
        try:
            provider_lock = flex_evidence.acquire_consumer_lock(gateway_root)
            provider_contract = flex_evidence.active_contract(gateway_root)
            start_window = flex_evidence.capture_start(gateway_root)
        except flex_evidence.EvidenceError as exc:
            raise contract.SharedAnswerError(str(exc)) from exc
        evidence_dir = output_dir / "provider_evidence" / "invocation-0001"
        evidence_dir.mkdir(parents=True, exist_ok=False)
        ready_path = evidence_dir / "ready.json"
        proxy_log = evidence_dir / "requests.jsonl"
        run_id = f"lme-shared-{method}-{uuid.uuid4().hex[:20]}"
        process = subprocess.Popen(
            [
                sys.executable,
                str(CONTROLLED_PROXY),
                "--port",
                "0",
                "--upstream",
                str(provider_contract["origin"]),
                "--log",
                str(proxy_log),
                "--ready",
                str(ready_path),
                "--run-id",
                run_id,
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        ready = _wait_for_controlled_proxy(
            process=process, ready_path=ready_path, expected_run_id=run_id
        )
        contract.require(
            ready.get("upstream") == provider_contract["origin"],
            "controlled Flex proxy upstream differs from dynamic gateway",
        )
        client = shared_answer.HttpAnswerClient(
            base_url=str(ready["base_url"]),
            retries=retries,
            answer_max_tokens=contract.ANSWER_MAX_TOKENS,
        )
        manifest = run_answer_inventory(
            mode="formal",
            method=method,
            output_dir=output_dir,
            source_run_dir=source_run_dir,
            source=source,
            protocol=protocol,
            client=client,
            proxy_log=proxy_log,
            provider_contract=provider_contract,
            publish_audit=False,
        )
        _stop_controlled_proxy(process)
        process = None
        try:
            closed_window = flex_evidence.capture_end(start_window)
        except flex_evidence.EvidenceError as exc:
            raise contract.SharedAnswerError(str(exc)) from exc
        provider_evidence = _publish_provider_evidence(
            output_dir=output_dir,
            run_id=run_id,
            ready_path=ready_path,
            proxy_log=proxy_log,
            window=closed_window,
        )
        contract.require(
            provider_evidence["requests"] == manifest.get("network_requests"),
            "shared-answer network count differs from Flex provider requests",
        )
        current = contract.read_json(existing_manifest)
        contract.require(
            isinstance(current, dict) and current.get("status") == "complete",
            "shared-answer manifest changed before provider binding",
        )
        current["provider_evidence"] = provider_evidence
        answer_contract.atomic_json_replace(existing_manifest, current)
    finally:
        if process is not None:
            _stop_controlled_proxy(process)
        if provider_lock is not None:
            provider_lock.close()
    postflight = audit_formal_source(
        method=method,
        source_run_dir=source_run_dir,
        dataset_path=dataset_path,
        preregistration_path=preregistration_path,
    )
    contract.require(
        postflight["descriptor"] == source["descriptor"],
        "formal source descriptor changed between preflight and postflight",
    )
    from baselines.longmemeval_m1.audit_longmemeval_shared_answers import audit_answer_run  # noqa: PLC0415

    report = audit_answer_run(output_dir)
    _publish_audit_no_clobber(output_dir / "audit.json", report)
    return report


def run_synthetic_matrix(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().absolute()
    if output_dir.exists():
        complete = output_dir / "complete.json"
        if not complete.is_file():
            raise contract.SharedAnswerError(
                "existing synthetic output is incomplete; use a new root"
            )
        from baselines.longmemeval_m1.audit_longmemeval_shared_answers import audit_synthetic_matrix  # noqa: PLC0415

        return audit_synthetic_matrix(output_dir)
    output_dir.mkdir(parents=True)
    preregistration, protocol = _load_protocol(DEFAULT_PREREGISTRATION, DEFAULT_DATASET)
    matrix_manifest = {
        "schema_version": contract.SYNTHETIC_MATRIX_SCHEMA,
        "status": "running",
        "mode": "synthetic_no_network",
        "created_at": input_contract.utc_now(),
        "formal_methods": list(contract.FORMAL_METHODS),
        "formal_executable_methods": list(contract.FORMAL_EXECUTABLE_METHODS),
        "backend_source_validation": "isolated_plan_plus_backend_auditor",
        "shared_protocol": protocol,
        "preregistration_content_sha256": preregistration[
            "preregistration_content_sha256"
        ],
        "source_hashes": contract.source_hashes(),
        "model_calls": 0,
        "network_calls": 0,
    }
    answer_contract.atomic_json_no_clobber(
        output_dir / "matrix_manifest.json", matrix_manifest
    )
    tokenizer = input_contract.FormalTokenizer.resolve()
    reports: dict[str, Any] = {}
    plan_isolation: dict[str, Any] = {}
    for method in contract.FORMAL_METHODS:
        source_dir = output_dir / "sources" / method
        if method in contract.BACKEND_INPUT_METHODS:
            source = create_synthetic_backend_source(
                matrix_root=output_dir, method=method
            )
            plan_isolation[method] = source["plan_isolation"]
        else:
            create_synthetic_source(
                source_run_dir=source_dir,
                method=method,
                tokenizer=tokenizer,
            )
            source = validate_synthetic_source(source_dir, method)
        answer_dir = output_dir / "answers" / method
        proxy_log = answer_dir / "synthetic_proxy.jsonl"
        client = shared_answer.FakeAnswerClient(
            proxy_log=proxy_log,
            run_id=f"synthetic-proxy-{method}",
            response_text="<answer>Pixel</answer>",
        )
        reports[method] = run_answer_inventory(
            mode="synthetic_no_network",
            method=method,
            output_dir=answer_dir,
            source_run_dir=source_dir,
            source=source,
            protocol=protocol,
            client=client,
            proxy_log=proxy_log,
        )
    complete = {
        "schema_version": contract.SYNTHETIC_MATRIX_SCHEMA,
        "status": "complete",
        "mode": "synthetic_no_network",
        "completed_at": input_contract.utc_now(),
        "formal_methods": list(contract.FORMAL_METHODS),
        "formal_executable_methods": list(contract.FORMAL_EXECUTABLE_METHODS),
        "method_audit_status": {
            method: reports[method]["status"] for method in contract.FORMAL_METHODS
        },
        "questions": len(contract.FORMAL_METHODS),
        "logical_answer_calls_simulated": len(contract.FORMAL_METHODS),
        "model_calls": 0,
        "network_calls": 0,
        "backend_plan_isolation": plan_isolation,
    }
    answer_contract.atomic_json_no_clobber(output_dir / "complete.json", complete)
    from baselines.longmemeval_m1.audit_longmemeval_shared_answers import audit_synthetic_matrix  # noqa: PLC0415

    report = audit_synthetic_matrix(output_dir)
    _publish_audit_no_clobber(output_dir / "audit.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    preflight.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    preflight.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    preflight.add_argument("--output", type=Path)

    synthetic = subparsers.add_parser("synthetic-sanity")
    synthetic.add_argument("--output-dir", type=Path, required=True)
    synthetic.add_argument("--allow-model-requests", action="store_true")

    answer = subparsers.add_parser("answer")
    answer.add_argument("--method", choices=contract.FORMAL_METHODS, required=True)
    answer.add_argument("--source-run-dir", type=Path, required=True)
    answer.add_argument("--output-dir", type=Path, required=True)
    answer.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    answer.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    answer.add_argument("--gateway-root", type=Path, required=True)
    answer.add_argument("--retries", type=int, default=3)
    answer.add_argument("--allow-model-requests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "preflight":
        report = preflight_matrix(
            root=args.root,
            dataset_path=args.dataset,
            preregistration_path=args.preregistration,
        )
        if args.output:
            answer_contract.atomic_json_no_clobber(args.output, report)
    elif args.command == "synthetic-sanity":
        if args.allow_model_requests:
            raise contract.SharedAnswerError(
                "synthetic sanity forbids --allow-model-requests"
            )
        report = run_synthetic_matrix(args.output_dir)
    else:
        report = run_formal_answers(
            method=args.method,
            source_run_dir=args.source_run_dir,
            output_dir=args.output_dir,
            dataset_path=args.dataset,
            preregistration_path=args.preregistration,
            gateway_root=args.gateway_root,
            retries=args.retries,
            allow_model_requests=args.allow_model_requests,
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
