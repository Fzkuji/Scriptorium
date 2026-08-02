#!/usr/bin/env python3
"""Contract for the LongMemEval-S M1 shared GPT-5.5 answer stage.

The answer stage consumes a frozen ``answer_input.json`` verbatim.  It may
reconstruct the already-recorded prompt for verification, but it must not
retrieve, rank, truncate, or otherwise rebuild the context.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

import controlled_locomo_answer_contract as answer_contract  # noqa: E402
import longmemeval_m1_backend_contract as backend_contract  # noqa: E402
import longmemeval_m1_contract as input_contract  # noqa: E402


RUN_SCHEMA = "longmemeval-m1-shared-answer-run-v1"
RESULT_SCHEMA = "longmemeval-m1-shared-answer-result-v1"
BINDING_SCHEMA = "longmemeval-m1-shared-answer-input-binding-v1"
CHECKPOINT_SCHEMA = "longmemeval-m1-shared-answer-checkpoint-v1"
PREFLIGHT_SCHEMA = "longmemeval-m1-shared-answer-preflight-v1"
SYNTHETIC_MATRIX_SCHEMA = "longmemeval-m1-shared-answer-synthetic-matrix-v1"
SYNTHETIC_SOURCE_SCHEMA = "longmemeval-m1-shared-answer-synthetic-source-v1"

FORMAL_METHODS = input_contract.FORMAL_METHODS
BASE_INPUT_METHODS = input_contract.PREPARABLE_METHODS
BACKEND_INPUT_METHODS = input_contract.PLANNED_BACKENDS
FORMAL_EXECUTABLE_METHODS = FORMAL_METHODS
HARD_CAP_METHODS = ("bm25", "mem0", "graphiti")
EXPECTED_MODEL = input_contract.EXPECTED_MODEL
ANSWER_MAX_TOKENS = input_contract.ANSWER_RESERVATION_TOKENS
MODEL_CONTEXT_LIMIT_TOKENS = input_contract.MODEL_CONTEXT_LIMIT_TOKENS
MAX_RENDERED_PROMPT_TOKENS = input_contract.MAX_RENDERED_PROMPT_TOKENS
VISIBLE_BUDGET_TOKENS = input_contract.VISIBLE_BUDGET_TOKENS
ZERO_HASH = "0" * 64
HEX64 = re.compile(r"[0-9a-f]{64}")


class SharedAnswerError(RuntimeError):
    """Raised when the frozen-input or shared-answer contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SharedAnswerError(message)


def sha256_file(path: Path) -> str:
    return input_contract.sha256_file(path)


def sha256_text(value: str) -> str:
    return input_contract.sha256_text(value)


def canonical_hash(value: Any) -> str:
    return input_contract.canonical_hash(value)


def read_json(path: Path) -> Any:
    try:
        return input_contract.read_json(path)
    except input_contract.ContractError as exc:
        raise SharedAnswerError(str(exc)) from exc


def reject_overlapping_paths(paths: Mapping[str, Path]) -> None:
    """Reject path aliases and ancestor/descendant relationships."""

    try:
        input_contract.ensure_distinct_paths(paths)
    except input_contract.ContractError as exc:
        raise SharedAnswerError(str(exc)) from exc
    resolved = {
        name: path.expanduser().absolute().resolve() for name, path in paths.items()
    }
    items = list(resolved.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise SharedAnswerError(
                    f"path overlap between {left_name} and {right_name}"
                )


def require_regular_file(path: Path, *, label: str, single_link: bool = True) -> Path:
    try:
        input_contract.reject_symlink_components(path)
    except input_contract.ContractError as exc:
        raise SharedAnswerError(str(exc)) from exc
    if path.is_symlink() or not path.is_file():
        raise SharedAnswerError(f"{label} is missing or not a regular file: {path}")
    if single_link and path.stat(follow_symlinks=False).st_nlink != 1:
        raise SharedAnswerError(f"{label} must not be hardlinked: {path}")
    return path.resolve()


def require_regular_directory(path: Path, *, label: str) -> Path:
    try:
        input_contract.reject_symlink_components(path)
    except input_contract.ContractError as exc:
        raise SharedAnswerError(str(exc)) from exc
    if path.is_symlink() or not path.is_dir():
        raise SharedAnswerError(f"{label} is not a regular directory: {path}")
    return path.resolve()


def source_hashes() -> dict[str, str]:
    """Fingerprint every implementation boundary used by the answer stage."""

    paths = {
        "scripts/longmemeval_shared_answer_contract.py": Path(__file__).resolve(),
        "scripts/run_longmemeval_shared_answers.py": (
            ROOT / "scripts/run_longmemeval_shared_answers.py"
        ),
        "scripts/audit_longmemeval_shared_answers.py": (
            ROOT / "scripts/audit_longmemeval_shared_answers.py"
        ),
        "scripts/longmemeval_m1_contract.py": Path(input_contract.__file__).resolve(),
        "scripts/run_longmemeval_m1_baselines.py": (
            ROOT / "scripts/run_longmemeval_m1_baselines.py"
        ),
        "scripts/audit_longmemeval_m1_baselines.py": (
            ROOT / "scripts/audit_longmemeval_m1_baselines.py"
        ),
        "scripts/longmemeval_m1_backend_contract.py": Path(
            backend_contract.__file__
        ).resolve(),
        "scripts/run_longmemeval_m1_backends.py": (
            ROOT / "scripts/run_longmemeval_m1_backends.py"
        ),
        "scripts/audit_longmemeval_m1_backends.py": (
            ROOT / "scripts/audit_longmemeval_m1_backends.py"
        ),
        "scripts/controlled_locomo_answer_contract.py": Path(
            answer_contract.__file__
        ).resolve(),
        "scripts/run_controlled_locomo_answers.py": (
            ROOT / "scripts/run_controlled_locomo_answers.py"
        ),
        "src/evaluation/durable_model_ledger.py": (
            ROOT / "src/evaluation/durable_model_ledger.py"
        ),
        "src/evaluation/visible_token_budget.py": (
            ROOT / "src/evaluation/visible_token_budget.py"
        ),
        "src/evaluation/visible_token_audit.py": (
            ROOT / "src/evaluation/visible_token_audit.py"
        ),
        "src/evaluation/prompts.py": ROOT / "src/evaluation/prompts.py",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SharedAnswerError(f"shared-answer source files are missing: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def source_auditor_identity(method: str) -> dict[str, Any]:
    """Return the exact independent source auditor required for one method."""

    require(method in FORMAL_EXECUTABLE_METHODS, f"unsupported method: {method}")
    if method in BASE_INPUT_METHODS:
        relative = "scripts/audit_longmemeval_m1_baselines.py"
        return {
            "kind": "base_input_auditor",
            "module": "audit_longmemeval_m1_baselines",
            "function": "audit_run",
            "implementation_path": str((ROOT / relative).resolve()),
            "implementation_sha256": source_hashes()[relative],
        }
    relative = "scripts/audit_longmemeval_m1_backends.py"
    return {
        "kind": "backend_input_auditor",
        "module": "audit_longmemeval_m1_backends",
        "function": "audit_backend_run",
        "implementation_path": str((ROOT / relative).resolve()),
        "implementation_sha256": source_hashes()[relative],
    }


def synthetic_source_auditor_identity(*, backend: bool) -> dict[str, Any]:
    """Identify the offline auditor used by a synthetic source fixture."""

    if backend:
        relative = "scripts/audit_longmemeval_m1_backends.py"
        return {
            "kind": "synthetic_backend_input_auditor",
            "module": "audit_longmemeval_m1_backends",
            "function": "audit_backend_run",
            "implementation_path": str((ROOT / relative).resolve()),
            "implementation_sha256": source_hashes()[relative],
        }
    relative = "scripts/audit_longmemeval_shared_answers.py"
    return {
        "kind": "synthetic_source_auditor",
        "module": "audit_longmemeval_shared_answers",
        "function": "_audit_synthetic_source",
        "implementation_path": str((ROOT / relative).resolve()),
        "implementation_sha256": source_hashes()[relative],
    }


def backend_workspace_bindings(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract the ordered plan-owned workspace identity for exact binding."""

    method = plan.get("method")
    require(method in BACKEND_INPUT_METHODS, "backend plan method differs")
    items = plan.get("items")
    require(isinstance(items, list) and items, "backend plan items are missing")
    bindings: list[dict[str, Any]] = []
    identities: set[str] = set()
    for item in items:
        require(isinstance(item, Mapping), "backend plan item is invalid")
        workspace = item.get("workspace")
        require(
            isinstance(workspace, Mapping), "backend workspace descriptor is missing"
        )
        workspace_path = workspace.get("workspace")
        require(
            isinstance(workspace_path, str) and workspace_path,
            "backend workspace path is invalid",
        )
        identity = input_contract.path_identity(Path(workspace_path))
        require(identity not in identities, "backend plan reuses a workspace")
        identities.add(identity)
        bindings.append(
            {
                "dataset_index": item.get("dataset_index"),
                "question_id": item.get("question_id"),
                "history_content_sha256": item.get("history_content_sha256"),
                "history_owner_sha256": item.get("history_owner_sha256"),
                "workspace": dict(workspace),
            }
        )
    return bindings


def validate_backend_plan_isolation(
    *,
    formal_plan: Mapping[str, Any],
    smoke_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject any smoke plan that aliases a formal item workspace."""

    method = formal_plan.get("method")
    require(
        method in BACKEND_INPUT_METHODS
        and smoke_plan.get("method") == method
        and formal_plan.get("scope") == "formal"
        and smoke_plan.get("scope") == "smoke",
        "backend plan isolation identity differs",
    )
    formal_bindings = backend_workspace_bindings(formal_plan)
    smoke_bindings = backend_workspace_bindings(smoke_plan)
    formal_identities = {
        input_contract.path_identity(Path(str(item["workspace"]["workspace"])))
        for item in formal_bindings
    }
    smoke_identities = {
        input_contract.path_identity(Path(str(item["workspace"]["workspace"])))
        for item in smoke_bindings
    }
    collisions = sorted(formal_identities & smoke_identities)
    require(not collisions, "smoke plan workspace overlaps formal plan")
    return {
        "status": "isolated",
        "method": method,
        "formal_workspace_count": len(formal_bindings),
        "smoke_workspace_count": len(smoke_bindings),
        "formal_workspace_bindings_sha256": canonical_hash(formal_bindings),
        "smoke_workspace_bindings_sha256": canonical_hash(smoke_bindings),
    }


def shared_protocol(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the answer-only fields shared by all four rows."""

    require(
        preregistration.get("schema_version") == input_contract.PREREG_SCHEMA_VERSION,
        "preregistration schema differs",
    )
    require(
        preregistration.get("status") == "frozen"
        and preregistration.get("formal_methods") == list(FORMAL_METHODS),
        "preregistered four-method matrix differs",
    )
    answer_protocol = preregistration.get("answer_protocol")
    require(isinstance(answer_protocol, Mapping), "answer protocol is missing")
    require(
        answer_protocol.get("requested_model") == EXPECTED_MODEL
        and answer_protocol.get("prompt_template_sha256")
        == sha256_text(input_contract.ANSWER_PROMPT),
        "preregistered answerer differs",
    )
    context_policy = preregistration.get("context_policy")
    require(
        context_policy
        == {
            "model_context_limit_tokens": MODEL_CONTEXT_LIMIT_TOKENS,
            "answer_completion_reservation_tokens": ANSWER_MAX_TOKENS,
            "max_rendered_prompt_tokens": MAX_RENDERED_PROMPT_TOKENS,
            "full_context_no_truncation": True,
            "hard_visible_total_methods": list(HARD_CAP_METHODS),
            "hard_visible_total_tokens": VISIBLE_BUDGET_TOKENS,
        },
        "preregistered context policy differs",
    )
    rows = preregistration.get("rows")
    require(isinstance(rows, list) and len(rows) == 4, "matrix rows differ")
    row_methods = [row.get("method") for row in rows if isinstance(row, Mapping)]
    require(row_methods == list(FORMAL_METHODS), "matrix row ordering differs")
    for row in rows:
        configuration = row.get("configuration")
        answerer = (
            configuration.get("answerer")
            if isinstance(configuration, Mapping)
            else None
        )
        require(
            isinstance(answerer, Mapping)
            and answerer.get("requested_model") == EXPECTED_MODEL
            and answerer.get("model_context_limit_tokens") == MODEL_CONTEXT_LIMIT_TOKENS
            and answerer.get("answer_completion_reservation_tokens")
            == ANSWER_MAX_TOKENS
            and answerer.get("max_rendered_prompt_tokens")
            == MAX_RENDERED_PROMPT_TOKENS,
            f"{row.get('method')} answerer reservation differs",
        )
    return {
        "formal_methods": list(FORMAL_METHODS),
        "requested_model": EXPECTED_MODEL,
        "prompt_template": "scripts.evaluation.prompts.ANSWER_PROMPT",
        "prompt_template_sha256": sha256_text(input_contract.ANSWER_PROMPT),
        "temperature": 0,
        "answer_max_tokens": ANSWER_MAX_TOKENS,
        "model_context_limit_tokens": MODEL_CONTEXT_LIMIT_TOKENS,
        "max_rendered_prompt_tokens": MAX_RENDERED_PROMPT_TOKENS,
        "tokenizer": preregistration.get("tokenizer"),
        "method_budget_policy": {
            "full_context": {
                "policy": "full_context_unbounded_accounted",
                "visible_budget_tokens": None,
                "truncation_allowed": False,
            },
            **{
                method: {
                    "policy": "hard_visible_total",
                    "visible_budget_tokens": VISIBLE_BUDGET_TOKENS,
                    "truncation_allowed": "already_frozen_upstream_only",
                }
                for method in HARD_CAP_METHODS
            },
        },
        "context_handling": "consume_frozen_context_verbatim_no_second_gate",
    }


def _require_hash(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise SharedAnswerError(f"{label} is not a SHA-256 value")
    return value


def validate_frozen_answer_input(
    payload: Mapping[str, Any],
    *,
    method: str,
    tokenizer: input_contract.FormalTokenizer,
    expected_index: int | None = None,
    expected_question_id: str | None = None,
) -> dict[str, Any]:
    """Validate the final frozen input without rebuilding retrieval output."""

    require(method in FORMAL_METHODS, f"unsupported method: {method}")
    require(isinstance(payload, Mapping), "answer input is not an object")
    try:
        input_contract.validate_answer_input_allowlist(payload)
    except input_contract.ContractError as exc:
        raise SharedAnswerError(str(exc)) from exc
    expected_keys = {
        "schema_version",
        "benchmark",
        "method",
        "dataset_index",
        "question_id",
        "question_type",
        "question",
        "question_date",
        "abstention",
        "history_owner_question_id",
        "history_content_sha256",
        "history_owner_sha256",
        "context",
        "answer_protocol_interface",
    }
    require(set(payload) == expected_keys, "answer input top-level fields differ")
    index = payload.get("dataset_index")
    question_id = payload.get("question_id")
    question = payload.get("question")
    require(
        payload.get("schema_version") == input_contract.SCHEMA_VERSION
        and payload.get("benchmark") == "LongMemEval-S"
        and payload.get("method") == method,
        "answer input identity differs",
    )
    require(
        isinstance(index, int) and not isinstance(index, bool) and index >= 0,
        "dataset index is invalid",
    )
    require(isinstance(question_id, str) and question_id, "question ID is invalid")
    require(isinstance(question, str) and question.strip(), "question is invalid")
    if expected_index is not None:
        require(index == expected_index, "dataset index differs")
    if expected_question_id is not None:
        require(question_id == expected_question_id, "question ID differs")
    require(
        payload.get("history_owner_question_id") == question_id
        and payload.get("abstention") == question_id.endswith("_abs"),
        "history owner or abstention flag differs",
    )
    require(
        payload.get("question_type") in input_contract.EXPECTED_TYPES,
        "question type is invalid",
    )
    require(isinstance(payload.get("question_date"), str), "question date is invalid")
    _require_hash(payload.get("history_content_sha256"), label="history content hash")
    _require_hash(payload.get("history_owner_sha256"), label="history owner hash")

    context = payload.get("context")
    require(isinstance(context, Mapping), "context block is missing")
    context_text = context.get("text")
    require(isinstance(context_text, str), "frozen context text is invalid")
    require(
        context.get("text_sha256") == sha256_text(context_text)
        and context.get("utf8_bytes") == len(context_text.encode("utf-8")),
        "frozen context hash or byte count differs",
    )
    retokenized = tokenizer.count(context_text)
    visible = context.get("visible_tokens")
    require(
        isinstance(visible, int)
        and not isinstance(visible, bool)
        and visible >= 0
        and context.get("retokenized_context_tokens") == retokenized,
        "frozen context token accounting differs",
    )
    source_ids = context.get("source_session_ids")
    events = context.get("events")
    require(
        isinstance(source_ids, list)
        and all(isinstance(value, str) and value for value in source_ids)
        and isinstance(events, list),
        "frozen context source inventory differs",
    )
    budget = context.get("budget")
    require(isinstance(budget, Mapping), "frozen context budget is missing")
    require(
        budget.get("cumulative_visible_tokens") == visible,
        "frozen cumulative visible-token count differs",
    )
    if method == "full_context":
        require(
            budget.get("policy") == "full_context_unbounded_accounted"
            and budget.get("configured_visible_budget_tokens") is None
            and budget.get("truncation_allowed") is False
            and budget.get("truncated_events") == 0
            and visible == retokenized
            and all(
                isinstance(event, Mapping) and event.get("decision") == "delivered_full"
                for event in events
            ),
            "full-context input was capped or truncated",
        )
    else:
        require(
            budget.get("policy") == "hard_visible_total"
            and budget.get("configured_visible_budget_tokens") == VISIBLE_BUDGET_TOKENS
            and visible <= VISIBLE_BUDGET_TOKENS,
            f"{method} frozen 20K budget differs",
        )

    prompt = input_contract.ANSWER_PROMPT.format(
        memories=context_text,
        question=question,
    )
    prompt_tokens = tokenizer.count(prompt)
    interface = payload.get("answer_protocol_interface")
    require(isinstance(interface, Mapping), "answer protocol interface is missing")
    require(
        interface.get("status") == "reserved_not_executed"
        and interface.get("requested_model") == EXPECTED_MODEL
        and interface.get("prompt_template") == "scripts.evaluation.prompts.ANSWER_PROMPT"
        and interface.get("prompt_template_sha256")
        == sha256_text(input_contract.ANSWER_PROMPT)
        and interface.get("rendered_prompt_sha256") == sha256_text(prompt)
        and interface.get("rendered_prompt_tokens") == prompt_tokens
        and interface.get("model_context_limit_tokens") == MODEL_CONTEXT_LIMIT_TOKENS
        and interface.get("answer_completion_reservation_tokens") == ANSWER_MAX_TOKENS
        and interface.get("max_rendered_prompt_tokens") == MAX_RENDERED_PROMPT_TOKENS
        and interface.get("model_calls") == 0
        and prompt_tokens <= MAX_RENDERED_PROMPT_TOKENS,
        "frozen rendered-prompt reservation differs",
    )
    return {
        "dataset_index": index,
        "question_id": question_id,
        "question_type": payload["question_type"],
        "abstention": payload["abstention"],
        "question_sha256": sha256_text(question),
        "context_sha256": sha256_text(context_text),
        "context_visible_tokens": visible,
        "rendered_prompt_sha256": sha256_text(prompt),
        "rendered_prompt_tokens": prompt_tokens,
        "prompt": prompt,
        "budget_policy": budget["policy"],
        "source_session_ids_sha256": canonical_hash(source_ids),
        "context_events_sha256": canonical_hash(events),
    }


def item_directory_name(index: int, question_id: str) -> str:
    return input_contract.item_directory_name(index, question_id)


def source_item_snapshot(
    *,
    source_run_dir: Path,
    source_item_record: Mapping[str, Any],
) -> dict[str, Any]:
    item_name = source_item_record.get("item_dir")
    require(isinstance(item_name, str) and item_name, "source item path is invalid")
    item_dir = require_regular_directory(
        source_run_dir / "items" / item_name,
        label="source item directory",
    )
    answer_path = require_regular_file(
        item_dir / "answer_input.json", label="source answer input"
    )
    checkpoint_path = require_regular_file(
        item_dir / "checkpoint.json", label="source checkpoint"
    )
    ledger_path = require_regular_file(
        item_dir / "attempts.jsonl", label="source preparation ledger"
    )
    observed = {
        "source_item_dir": item_name,
        "answer_input_path": str(answer_path.relative_to(source_run_dir)),
        "answer_input_sha256": sha256_file(answer_path),
        "checkpoint_path": str(checkpoint_path.relative_to(source_run_dir)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "preparation_ledger_path": str(ledger_path.relative_to(source_run_dir)),
        "preparation_ledger_sha256": sha256_file(ledger_path),
    }
    for key in (
        "answer_input_sha256",
        "checkpoint_sha256",
        "ledger_sha256",
    ):
        expected_key = "preparation_ledger_sha256" if key == "ledger_sha256" else key
        require(
            source_item_record.get(key) == observed[expected_key],
            f"source item {item_name} {key} differs from frozen manifest",
        )
    backend_keys = {
        "workspace",
        "workspace_after_retrieval_sha256",
        "model_ledger_sha256",
        "proxy_slice_sha256",
        "source_trace_sha256",
        "retrieval_sha256",
        "budget_trace_sha256",
        "budget_manifest_sha256",
    }
    present_backend_keys = backend_keys & set(source_item_record)
    if present_backend_keys:
        require(
            present_backend_keys == backend_keys,
            f"source item {item_name} backend binding is incomplete",
        )
        workspace_value = source_item_record.get("workspace")
        require(
            isinstance(workspace_value, str) and workspace_value,
            f"source item {item_name} backend workspace is invalid",
        )
        workspace = require_regular_directory(
            Path(workspace_value), label="source backend workspace"
        )
        require(
            workspace == (item_dir / "backend").resolve(),
            f"source item {item_name} backend workspace escaped its item",
        )
        backend_artifacts = {
            "model_ledger": item_dir / "model_evidence/ledger.jsonl",
            "proxy_slice": item_dir / "proxy_slice.json",
            "source_trace": item_dir / "source_trace.json",
            "retrieval": item_dir / "retrieval_raw.json",
            "budget_trace": item_dir / "retrieval_budget.jsonl",
            "budget_manifest": item_dir / "retrieval_budget.manifest.json",
        }
        artifact_hashes = {
            name: sha256_file(require_regular_file(path, label=f"source {name}"))
            for name, path in backend_artifacts.items()
        }
        expected_hashes = {
            "model_ledger": source_item_record["model_ledger_sha256"],
            "proxy_slice": source_item_record["proxy_slice_sha256"],
            "source_trace": source_item_record["source_trace_sha256"],
            "retrieval": source_item_record["retrieval_sha256"],
            "budget_trace": source_item_record["budget_trace_sha256"],
            "budget_manifest": source_item_record["budget_manifest_sha256"],
        }
        require(
            artifact_hashes == expected_hashes,
            f"source item {item_name} backend artifact hash differs",
        )
        observed.update(
            {
                "backend_workspace_path": str(workspace),
                "backend_workspace_after_retrieval_sha256": source_item_record[
                    "workspace_after_retrieval_sha256"
                ],
                "backend_artifact_hashes": artifact_hashes,
            }
        )
    return observed


def input_binding(
    *,
    run_id: str,
    method: str,
    source_descriptor: Mapping[str, Any],
    source_item_record: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    answer_input: Mapping[str, Any],
    validated: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": BINDING_SCHEMA,
        "run_id": run_id,
        "method": method,
        "dataset_index": validated["dataset_index"],
        "question_id": validated["question_id"],
        "source": {
            "run_manifest_sha256": source_descriptor["run_manifest_sha256"],
            "input_audit_sha256": source_descriptor["input_audit_sha256"],
            "item_tree_root_sha256": source_descriptor["item_tree_root_sha256"],
            **dict(snapshot),
        },
        "answer_input_content_sha256": canonical_hash(answer_input),
        "question_sha256": validated["question_sha256"],
        "context_sha256": validated["context_sha256"],
        "context_visible_tokens": validated["context_visible_tokens"],
        "rendered_prompt_sha256": validated["rendered_prompt_sha256"],
        "rendered_prompt_tokens": validated["rendered_prompt_tokens"],
        "source_session_ids_sha256": validated["source_session_ids_sha256"],
        "context_events_sha256": validated["context_events_sha256"],
        "frozen_input_consumption": {
            "retrieval_recomputed": False,
            "ranking_recomputed": False,
            "truncation_recomputed": False,
            "second_visible_token_gate_applied": False,
            "context_consumed_verbatim": True,
        },
    }
    payload["binding_content_sha256"] = canonical_hash(payload)
    return payload


def validate_binding_content(payload: Mapping[str, Any]) -> None:
    require(payload.get("schema_version") == BINDING_SCHEMA, "binding schema differs")
    recorded = payload.get("binding_content_sha256")
    content = dict(payload)
    content.pop("binding_content_sha256", None)
    require(recorded == canonical_hash(content), "binding content hash differs")


def proxy_prefix_matches(path: Path, prefix: Mapping[str, Any]) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    offset = prefix.get("byte_offset")
    expected = prefix.get("prefix_sha256")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        return False
    if not isinstance(expected, str) or HEX64.fullmatch(expected) is None:
        return False
    with path.open("rb") as handle:
        payload = handle.read(offset)
    return len(payload) == offset and input_contract.sha256_bytes(payload) == expected


def assert_source_unchanged(
    *,
    source_run_dir: Path,
    source_item_record: Mapping[str, Any],
    expected_snapshot: Mapping[str, Any],
) -> None:
    observed = source_item_snapshot(
        source_run_dir=source_run_dir,
        source_item_record=source_item_record,
    )
    require(
        observed == dict(expected_snapshot), "frozen source item changed during answer"
    )


def ensure_no_inode_alias(left: Path, right: Path, *, label: str) -> None:
    if left.exists() and right.exists() and os.path.samefile(left, right):
        raise SharedAnswerError(f"inode alias detected for {label}")


def expected_method_policy(method: str) -> dict[str, Any]:
    require(method in FORMAL_METHODS, f"unsupported method: {method}")
    if method == "full_context":
        return {
            "policy": "full_context_unbounded_accounted",
            "visible_budget_tokens": None,
            "frozen_input_truncation_allowed": False,
        }
    return {
        "policy": "hard_visible_total",
        "visible_budget_tokens": VISIBLE_BUDGET_TOKENS,
        "frozen_input_truncation_allowed": True,
    }


def question_inventory_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    types = {name: 0 for name in input_contract.EXPECTED_TYPES}
    abstention = 0
    for record in records:
        question_type = str(record["question_type"])
        if question_type not in types:
            raise SharedAnswerError(f"unknown question type: {question_type}")
        types[question_type] += 1
        abstention += int(bool(record["abstention"]))
    return {
        "items": len(records),
        "abstention_items": abstention,
        "question_types": types,
    }
