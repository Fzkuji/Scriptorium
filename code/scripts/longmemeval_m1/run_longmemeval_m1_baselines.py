#!/usr/bin/env python3
"""Prepare no-model LongMemEval-S M1 controlled baseline inputs.

The executable has three operations:

* ``preregister`` freezes the four-row formal matrix before answers exist;
* ``prepare`` creates all 500 full-context or BM25 answer inputs;
* ``plan-backend`` creates a no-execution formal/smoke plan for Mem0 or
  Graphiti OSS.

There is intentionally no answer, judge, HTTP, or provider client in this
module.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.longmemeval_m1 import longmemeval_m1_contract as contract  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "results/gpt55-longmemeval-m1-baselines-20260714"


def _content_hash(payload: Mapping[str, Any], field: str) -> str:
    content = copy.deepcopy(dict(payload))
    content.pop(field, None)
    return contract.canonical_hash(content)


def create_preregistration(
    *, dataset_path: Path,
    output_path: Path,
    model_context_limit_tokens: int,
    answer_reservation_tokens: int,
) -> dict[str, Any]:
    dataset_path = dataset_path.expanduser().resolve()
    output_path = Path(os.path.abspath(output_path.expanduser()))
    if model_context_limit_tokens != contract.MODEL_CONTEXT_LIMIT_TOKENS:
        raise contract.ContractError(
            "M1 preregistration requires model context limit 128000"
        )
    if answer_reservation_tokens != contract.ANSWER_RESERVATION_TOKENS:
        raise contract.ContractError("M1 preregistration requires reservation 4096")
    contract.ensure_distinct_paths(
        {"dataset": dataset_path, "preregistration": output_path}
    )
    data = contract.validate_dataset(dataset_path)
    tokenizer = contract.FormalTokenizer.resolve()
    dependencies = contract.strict_dependency_snapshot()
    rows = [
        {
            "method": method,
            "formal_scope_items": contract.EXPECTED_ITEMS,
            "configuration": contract.method_configuration(method),
        }
        for method in contract.FORMAL_METHODS
    ]
    payload: dict[str, Any] = {
        "schema_version": contract.PREREG_SCHEMA_VERSION,
        "artifact_type": "formal_experiment_matrix_preregistration",
        "status": "frozen",
        "frozen_at": contract.utc_now(),
        "benchmark": "LongMemEval-S",
        "milestone": "M1 controlled baselines",
        "task_scope": "retrieval_inputs_before_answer_generation",
        "dataset": {
            "path": str(dataset_path),
            "sha256": contract.sha256_file(dataset_path),
            "items": len(data),
            "abstention_items": sum(
                str(item["question_id"]).endswith("_abs") for item in data
            ),
            "question_types": dict(contract.EXPECTED_TYPES),
            "independent_history_per_item": True,
            "unique_history_hashes": len(
                {contract.history_content_hash(item) for item in data}
            ),
        },
        "formal_methods": list(contract.FORMAL_METHODS),
        "rows": rows,
        "tokenizer": tokenizer.identity,
        "context_policy": {
            "model_context_limit_tokens": model_context_limit_tokens,
            "answer_completion_reservation_tokens": answer_reservation_tokens,
            "max_rendered_prompt_tokens": (
                model_context_limit_tokens - answer_reservation_tokens
            ),
            "full_context_no_truncation": True,
            "hard_visible_total_methods": ["bm25", "mem0", "graphiti"],
            "hard_visible_total_tokens": contract.VISIBLE_BUDGET_TOKENS,
        },
        "source_mapping": {
            "requirement": "R002",
            "granularity": "session",
            "turn_recall_claimed": False,
            "answer_session_ids_are_audit_only": True,
        },
        "answer_protocol": {
            "status": "not_started",
            "integration": "deferred_to_shared_controlled_gpt55_answer_protocol",
            "requested_model": contract.EXPECTED_MODEL,
            "prompt_template": "scripts.evaluation.prompts.ANSWER_PROMPT",
            "prompt_template_sha256": contract.sha256_text(contract.ANSWER_PROMPT),
            "input_file": "per-item answer_input.json",
            "input_prohibits_dataset_answer_and_reference_session_ids": True,
        },
        "dependencies": dependencies,
        "execution_gates": {
            "formal_answer_generation": "prohibited_until_input_audit_passes",
            "judging": "not_started",
            "mem0_execution": "planned_not_executed",
            "graphiti_execution": "planned_not_executed",
        },
        "model_calls": 0,
        "network_calls": 0,
        "code": {
            "contract": {
                "path": str(Path(contract.__file__).resolve()),
                "sha256": contract.sha256_file(Path(contract.__file__).resolve()),
            },
            "preregister_runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": contract.sha256_file(Path(__file__).resolve()),
            },
        },
    }
    payload["preregistration_content_sha256"] = _content_hash(
        payload, "preregistration_content_sha256"
    )
    contract.atomic_json_no_clobber(output_path, payload)
    return payload


def validate_preregistration(
    *, preregistration_path: Path, dataset_path: Path
) -> dict[str, Any]:
    preregistration_path = preregistration_path.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    contract.ensure_distinct_paths(
        {"dataset": dataset_path, "preregistration": preregistration_path}
    )
    payload = contract.read_json(preregistration_path)
    if not isinstance(payload, dict):
        raise contract.ContractError("preregistration is not an object")
    required = {
        "schema_version": contract.PREREG_SCHEMA_VERSION,
        "artifact_type": "formal_experiment_matrix_preregistration",
        "status": "frozen",
        "benchmark": "LongMemEval-S",
        "task_scope": "retrieval_inputs_before_answer_generation",
        "formal_methods": list(contract.FORMAL_METHODS),
        "tokenizer": contract.FormalTokenizer.resolve().identity,
        "dependencies": contract.strict_dependency_snapshot(),
        "model_calls": 0,
        "network_calls": 0,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise contract.ContractError(f"preregistration field {key} differs")
    if payload.get("preregistration_content_sha256") != _content_hash(
        payload, "preregistration_content_sha256"
    ):
        raise contract.ContractError("preregistration content hash differs")
    dataset = payload.get("dataset")
    if (
        not isinstance(dataset, dict)
        or dataset.get("sha256") != contract.sha256_file(dataset_path)
        or dataset.get("items") != contract.EXPECTED_ITEMS
        or dataset.get("abstention_items") != contract.EXPECTED_ABSTENTION
        or dataset.get("unique_history_hashes") != contract.EXPECTED_ITEMS
    ):
        raise contract.ContractError("preregistration dataset binding differs")
    expected_rows = [
        {
            "method": method,
            "formal_scope_items": contract.EXPECTED_ITEMS,
            "configuration": contract.method_configuration(method),
        }
        for method in contract.FORMAL_METHODS
    ]
    if payload.get("rows") != expected_rows:
        raise contract.ContractError("preregistration matrix differs")
    expected_context = {
        "model_context_limit_tokens": contract.MODEL_CONTEXT_LIMIT_TOKENS,
        "answer_completion_reservation_tokens": contract.ANSWER_RESERVATION_TOKENS,
        "max_rendered_prompt_tokens": contract.MAX_RENDERED_PROMPT_TOKENS,
        "full_context_no_truncation": True,
        "hard_visible_total_methods": ["bm25", "mem0", "graphiti"],
        "hard_visible_total_tokens": contract.VISIBLE_BUDGET_TOKENS,
    }
    if payload.get("context_policy") != expected_context:
        raise contract.ContractError("preregistration context policy differs")
    expected_code = {
        "contract": {
            "path": str(Path(contract.__file__).resolve()),
            "sha256": contract.sha256_file(Path(contract.__file__).resolve()),
        },
        "preregister_runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": contract.sha256_file(Path(__file__).resolve()),
        },
    }
    if payload.get("code") != expected_code:
        raise contract.ContractError("preregistration code binding differs")
    return payload


def _run_identity(
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
            "runner": contract.sha256_file(Path(__file__).resolve()),
            "answer_prompt_source": contract.sha256_file(
                ROOT / "src/evaluation/prompts.py"
            ),
        },
    }


def _new_manifest(identity: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": contract.RUN_MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "created_at": contract.utc_now(),
        "updated_at": contract.utc_now(),
        "run_dir": str(run_dir),
        "identity": copy.deepcopy(dict(identity)),
        "completed_items": 0,
        "expected_items": contract.EXPECTED_ITEMS,
        "abstention_items": 0,
        "model_calls": 0,
        "network_calls": 0,
        "answer_stage": "not_started",
        "items": [],
    }


def _validate_completed_item(
    *,
    method: str,
    item: Mapping[str, Any],
    item_index: int,
    item_dir: Path,
    preregistration_sha256: str,
) -> dict[str, Any]:
    ledger_path = item_dir / "attempts.jsonl"
    answer_path = item_dir / "answer_input.json"
    checkpoint_path = item_dir / "checkpoint.json"
    events = contract.read_jsonl(ledger_path)
    terminal = contract.validate_item_ledger(
        events,
        method=method,
        item_index=item_index,
        question_id=str(item["question_id"]),
        require_complete=True,
    )
    if terminal is None or not answer_path.is_file() or not checkpoint_path.is_file():
        raise contract.ContractError(f"item {item_index} completed artifacts are missing")
    if answer_path.is_symlink() or checkpoint_path.is_symlink():
        raise contract.ContractError(f"item {item_index} contains a symlink")
    answer_input = contract.read_json(answer_path)
    checkpoint = contract.read_json(checkpoint_path)
    if not isinstance(answer_input, dict) or not isinstance(checkpoint, dict):
        raise contract.ContractError(f"item {item_index} artifact is not an object")
    contract.validate_answer_input_allowlist(answer_input)
    if (
        terminal.get("answer_input_sha256") != contract.sha256_file(answer_path)
        or terminal.get("answer_input_content_sha256")
        != contract.canonical_hash(answer_input)
        or checkpoint.get("answer_input_sha256") != contract.sha256_file(answer_path)
        or checkpoint.get("ledger_sha256") != contract.sha256_file(ledger_path)
        or checkpoint.get("preregistration_sha256") != preregistration_sha256
        or checkpoint.get("status") != "complete"
        or checkpoint.get("attempt_id") != terminal.get("attempt_id")
        or checkpoint.get("history_owner_question_id") != item["question_id"]
        or checkpoint.get("history_content_sha256")
        != contract.history_content_hash(item)
    ):
        raise contract.ContractError(f"item {item_index} completed evidence differs")
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
        "visible_tokens": answer_input["context"]["visible_tokens"],
        "rendered_prompt_tokens": answer_input["answer_protocol_interface"][
            "rendered_prompt_tokens"
        ],
        "delivered_source_session_ids": answer_input["context"][
            "source_session_ids"
        ],
        "source_mapping_session_recall": checkpoint["private_audit"][
            "source_mapping"
        ]["session_recall"],
    }


def _process_item(
    *,
    method: str,
    item: Mapping[str, Any],
    item_index: int,
    run_dir: Path,
    preregistration_sha256: str,
    configuration_sha256: str,
    tokenizer: contract.FormalTokenizer,
) -> dict[str, Any]:
    item_dir = run_dir / "items" / contract.item_directory_name(
        item_index, str(item["question_id"])
    )
    contract.reject_symlink_components(item_dir)
    if item_dir.is_symlink():
        raise contract.ContractError(f"item directory is a symlink: {item_dir}")
    item_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = item_dir / "attempts.jsonl"
    answer_path = item_dir / "answer_input.json"
    checkpoint_path = item_dir / "checkpoint.json"
    item_lock_path = item_dir / ".item.lock"
    with contract.advisory_lock(item_lock_path):
        events = contract.read_jsonl(ledger_path)
        if events:
            terminal = contract.validate_item_ledger(
                events,
                method=method,
                item_index=item_index,
                question_id=str(item["question_id"]),
                require_complete=False,
            )
            if terminal is not None:
                return _validate_completed_item(
                    method=method,
                    item=item,
                    item_index=item_index,
                    item_dir=item_dir,
                    preregistration_sha256=preregistration_sha256,
                )
        elif any(path.exists() or path.is_symlink() for path in (answer_path, checkpoint_path)):
            raise contract.ContractError(
                f"item {item_index} has artifacts without an attempt ledger"
            )
        attempt_id = str(uuid.uuid4())
        started = contract.append_jsonl_fsync(
            ledger_path,
            {
                "event": "attempt_started",
                "attempt_id": attempt_id,
                "method": method,
                "dataset_index": item_index,
                "question_id": item["question_id"],
                "history_content_sha256": contract.history_content_hash(item),
                "history_owner_sha256": contract.history_owner_hash(item),
                "preregistration_sha256": preregistration_sha256,
                "configuration_sha256": configuration_sha256,
                "model_calls": 0,
                "network_calls": 0,
            },
        )
        answer_input, private_audit = contract.prepare_item(
            method, item, item_index, tokenizer
        )
        contract.validate_answer_input_allowlist(answer_input)
        contract.atomic_json_no_clobber(answer_path, answer_input)
        terminal = contract.append_jsonl_fsync(
            ledger_path,
            {
                "event": "attempt_completed",
                "attempt_id": attempt_id,
                "method": method,
                "dataset_index": item_index,
                "question_id": item["question_id"],
                "start_event_id": started["event_id"],
                "answer_input_sha256": contract.sha256_file(answer_path),
                "answer_input_content_sha256": contract.canonical_hash(answer_input),
                "private_audit_sha256": contract.canonical_hash(private_audit),
                "visible_tokens": answer_input["context"]["visible_tokens"],
                "rendered_prompt_tokens": answer_input["answer_protocol_interface"][
                    "rendered_prompt_tokens"
                ],
                "model_calls": 0,
                "network_calls": 0,
            },
        )
        checkpoint = {
            "schema_version": contract.CHECKPOINT_SCHEMA_VERSION,
            "status": "complete",
            "method": method,
            "dataset_index": item_index,
            "question_id": item["question_id"],
            "attempt_id": attempt_id,
            "start_event_id": started["event_id"],
            "terminal_event_id": terminal["event_id"],
            "history_owner_question_id": item["question_id"],
            "history_content_sha256": contract.history_content_hash(item),
            "history_owner_sha256": contract.history_owner_hash(item),
            "preregistration_sha256": preregistration_sha256,
            "configuration_sha256": configuration_sha256,
            "answer_input": answer_path.name,
            "answer_input_sha256": contract.sha256_file(answer_path),
            "ledger": ledger_path.name,
            "ledger_sha256": contract.sha256_file(ledger_path),
            "private_audit": private_audit,
            "answer_stage": "not_started",
            "model_calls": 0,
            "network_calls": 0,
            "completed_at": contract.utc_now(),
        }
        contract.atomic_json_no_clobber(checkpoint_path, checkpoint)
        return _validate_completed_item(
            method=method,
            item=item,
            item_index=item_index,
            item_dir=item_dir,
            preregistration_sha256=preregistration_sha256,
        )


def prepare_formal_inputs(
    *,
    method: str,
    dataset_path: Path,
    preregistration_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if method not in contract.PREPARABLE_METHODS:
        raise contract.ContractError(
            "no-model prepare accepts only full_context or bm25"
        )
    dataset_path = dataset_path.expanduser().resolve()
    preregistration_path = preregistration_path.expanduser().resolve()
    output_root = Path(os.path.abspath(output_root.expanduser()))
    run_dir = output_root / method
    contract.ensure_distinct_paths(
        {
            "dataset": dataset_path,
            "preregistration": preregistration_path,
            "output_root": output_root,
            "run_dir": run_dir,
        }
    )
    data = contract.validate_dataset(dataset_path)
    preregistration = validate_preregistration(
        preregistration_path=preregistration_path, dataset_path=dataset_path
    )
    tokenizer = contract.FormalTokenizer.resolve()
    if method == "bm25":
        contract.package_version("rank-bm25", contract.BM25_VERSION)
    output_root.mkdir(parents=True, exist_ok=True)
    contract.reject_symlink_components(output_root)
    if output_root.is_symlink() or run_dir.is_symlink():
        raise contract.ContractError("output directories must not be symlinks")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "items").mkdir(exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    identity = _run_identity(
        method=method,
        dataset_path=dataset_path,
        preregistration_path=preregistration_path,
        preregistration=preregistration,
    )
    configuration_sha256 = contract.canonical_hash(identity["configuration"])
    preregistration_sha256 = contract.sha256_file(preregistration_path)
    with contract.advisory_lock(run_dir / ".run.lock"):
        if manifest_path.exists():
            manifest = contract.read_json(manifest_path)
            if not isinstance(manifest, dict) or manifest.get("identity") != identity:
                raise contract.ContractError("existing run manifest identity differs")
        else:
            manifest = _new_manifest(identity, run_dir)
            contract.atomic_json_replace(manifest_path, manifest)
        completed: list[dict[str, Any]] = []
        for item_index, item in enumerate(data):
            completed.append(
                _process_item(
                    method=method,
                    item=item,
                    item_index=item_index,
                    run_dir=run_dir,
                    preregistration_sha256=preregistration_sha256,
                    configuration_sha256=configuration_sha256,
                    tokenizer=tokenizer,
                )
            )
            manifest["status"] = "running"
            manifest["updated_at"] = contract.utc_now()
            manifest["completed_items"] = len(completed)
            manifest["abstention_items"] = sum(item["abstention"] for item in completed)
            manifest["items"] = completed
            contract.atomic_json_replace(manifest_path, manifest)
        history_hashes = [item["history_content_sha256"] for item in completed]
        owner_hashes = [item["history_owner_sha256"] for item in completed]
        if len(set(history_hashes)) != contract.EXPECTED_ITEMS:
            raise contract.ContractError("completed items reused a history")
        if len(set(owner_hashes)) != contract.EXPECTED_ITEMS:
            raise contract.ContractError("completed items reused a history owner")
        item_inventory = contract.regular_tree_inventory(run_dir / "items")
        manifest["status"] = "inputs_complete"
        manifest["updated_at"] = contract.utc_now()
        manifest["completed_at"] = contract.utc_now()
        manifest["completed_items"] = len(completed)
        manifest["abstention_items"] = sum(item["abstention"] for item in completed)
        manifest["items"] = completed
        manifest["item_tree_inventory"] = {
            "entries": len(item_inventory),
            "bytes": sum(entry["bytes"] for entry in item_inventory),
            "root_sha256": contract.inventory_root(item_inventory),
        }
        manifest["visible_token_summary"] = {
            "min": min(item["visible_tokens"] for item in completed),
            "max": max(item["visible_tokens"] for item in completed),
            "sum": sum(item["visible_tokens"] for item in completed),
        }
        manifest["rendered_prompt_token_summary"] = {
            "min": min(item["rendered_prompt_tokens"] for item in completed),
            "max": max(item["rendered_prompt_tokens"] for item in completed),
            "sum": sum(item["rendered_prompt_tokens"] for item in completed),
        }
        contract.atomic_json_replace(manifest_path, manifest)
        return manifest


def create_backend_plan(
    *,
    method: str,
    scope: str,
    item_index: int | None,
    dataset_path: Path,
    preregistration_path: Path,
    output_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if method not in contract.PLANNED_BACKENDS:
        raise contract.ContractError("backend plan accepts only mem0 or graphiti")
    dataset_path = dataset_path.expanduser().resolve()
    preregistration_path = preregistration_path.expanduser().resolve()
    output_root = Path(os.path.abspath(output_root.expanduser()))
    output_path = Path(os.path.abspath(output_path.expanduser()))
    contract.ensure_distinct_paths(
        {
            "dataset": dataset_path,
            "preregistration": preregistration_path,
            "output_root": output_root,
            "plan": output_path,
        }
    )
    data = contract.validate_dataset(dataset_path)
    preregistration = validate_preregistration(
        preregistration_path=preregistration_path, dataset_path=dataset_path
    )
    dependencies = contract.strict_dependency_snapshot()
    indices = contract.backend_plan_indices(scope, item_index)
    items = []
    workspace_identities: set[str] = set()
    for index in indices:
        item = data[index]
        workspace = contract.backend_workspace_descriptor(
            method=method,
            output_root=output_root,
            item_index=index,
            question_id=item["question_id"],
        )
        identity = contract.path_identity(Path(workspace["workspace"]))
        if identity in workspace_identities:
            raise contract.ContractError("backend plan reuses an item workspace")
        workspace_identities.add(identity)
        items.append(
            {
                "dataset_index": index,
                "question_id": item["question_id"],
                "history_content_sha256": contract.history_content_hash(item),
                "history_owner_sha256": contract.history_owner_hash(item),
                "history_session_ids": item["haystack_session_ids"],
                "workspace": workspace,
                "checkpoint": "checkpoint.json",
                "attempt_ledger": "attempts.jsonl",
                "visible_token_budget": contract.VISIBLE_BUDGET_TOKENS,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": contract.BACKEND_PLAN_SCHEMA_VERSION,
        "status": "planned_not_executed",
        "created_at": contract.utc_now(),
        "benchmark": "LongMemEval-S",
        "method": method,
        "scope": scope,
        "output_root": str(output_root),
        "items": items,
        "item_count": len(items),
        "configuration": contract.method_configuration(method),
        "dependencies": dependencies,
        "dataset_sha256": contract.sha256_file(dataset_path),
        "preregistration_sha256": contract.sha256_file(preregistration_path),
        "preregistration_content_sha256": preregistration[
            "preregistration_content_sha256"
        ],
        "execution_interface": {
            "history_ingest": "one item history into its private workspace",
            "retrieve": "return ordered text plus stable source session IDs",
            "visible_gate": "20K tiktoken o200k_base before answer interface",
            "durability": "per-item checkpoint and append-only attempt ledger",
            "model_binding": "shared controlled GPT-5.5 protocol after input audit",
        },
        "model_calls": 0,
        "network_calls": 0,
    }
    payload["plan_content_sha256"] = _content_hash(payload, "plan_content_sha256")
    contract.atomic_json_no_clobber(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LongMemEval-S M1 controlled baseline input runner (no model calls)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prereg = subparsers.add_parser("preregister")
    prereg.add_argument("--dataset", type=Path, default=contract.DEFAULT_DATASET)
    prereg.add_argument("--output", type=Path, required=True)
    prereg.add_argument(
        "--model-context-limit-tokens",
        type=int,
        default=contract.MODEL_CONTEXT_LIMIT_TOKENS,
    )
    prereg.add_argument(
        "--answer-reservation-tokens",
        type=int,
        default=contract.ANSWER_RESERVATION_TOKENS,
    )

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--method", choices=contract.PREPARABLE_METHODS, required=True)
    prepare.add_argument("--dataset", type=Path, default=contract.DEFAULT_DATASET)
    prepare.add_argument("--preregistration", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    plan = subparsers.add_parser("plan-backend")
    plan.add_argument("--method", choices=contract.PLANNED_BACKENDS, required=True)
    plan.add_argument("--scope", choices=("formal", "smoke"), required=True)
    plan.add_argument("--item-index", type=int)
    plan.add_argument("--dataset", type=Path, default=contract.DEFAULT_DATASET)
    plan.add_argument("--preregistration", type=Path, required=True)
    plan.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    plan.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preregister":
        payload = create_preregistration(
            dataset_path=args.dataset,
            output_path=args.output,
            model_context_limit_tokens=args.model_context_limit_tokens,
            answer_reservation_tokens=args.answer_reservation_tokens,
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "formal_methods": payload["formal_methods"],
                    "output": str(args.output),
                    "model_calls": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "prepare":
        manifest = prepare_formal_inputs(
            method=args.method,
            dataset_path=args.dataset,
            preregistration_path=args.preregistration,
            output_root=args.output_root,
        )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "method": args.method,
                    "completed_items": manifest["completed_items"],
                    "model_calls": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    payload = create_backend_plan(
        method=args.method,
        scope=args.scope,
        item_index=args.item_index,
        dataset_path=args.dataset,
        preregistration_path=args.preregistration,
        output_root=args.output_root,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "method": payload["method"],
                "scope": payload["scope"],
                "item_count": payload["item_count"],
                "model_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
