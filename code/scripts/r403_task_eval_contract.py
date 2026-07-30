#!/usr/bin/env python3
"""Frozen contract for checkpoint task evaluation downstream of R403.

This module performs no model or network request.  It accepts only a complete
R403 growth artifact that passes the independent build auditor, binds every
checkpoint byte-for-byte, reconstructs the checkpoint-specific question
inventory, and exposes the immutable inputs used by the task runner and its
independent auditor.
"""

from __future__ import annotations

import fcntl
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_r403_incremental_growth as growth_auditor  # noqa: E402
import controlled_locomo_answer_contract as answer_contract  # noqa: E402
import readonly_nativemem_control as readonly_control  # noqa: E402
from src.evaluation.durable_model_ledger import read_ledger  # noqa: E402
from src.evaluation.metrics import f1_locomo_official  # noqa: E402


PREREG_SCHEMA = "nativemem.r403-task-eval-preregistration.v1"
PREFLIGHT_SCHEMA = "nativemem.r403-task-eval-preflight.v1"
SOURCE_BINDING_SCHEMA = "nativemem.r403-task-eval-source-binding.v1"
INPUT_BINDING_SCHEMA = "nativemem.r403-task-eval-input-binding.v1"
M4_TRACE_SCHEMA = "nativemem.r403-task-eval-m4-trace.v1"
METRICS_SCHEMA = "nativemem.r403-task-eval-question-metrics.v1"

METHOD = "r403_checkpoint_task_evaluation"
GENERIC_CONDITION = "dual_source"
MODEL = "gpt-5.5"
CHECKPOINTS = (10, 25, 50, 100)
BUDGET_TOKENS = 20_000
MAX_ROUNDS = 12
MODEL_CONTEXT_LIMIT_TOKENS = 128_000
ANSWER_COMPLETION_RESERVATION_TOKENS = 4_096
ANSWER_MAX_TOKENS = 4_096
ANSWER_RETRIES = 3
PRIMARY_QUESTIONS = 1_540
PRIMARY_SOURCE_RECALL = 1_533
SOURCE_EXCLUSIONS = 7
CATEGORY_COUNTS = {1: 282, 2: 321, 3: 96, 4: 841}

DATA = ROOT / "benchmarks/locomo/data/locomo10.json"
EVIDENCE_DIR = ROOT / "results/paper-experiments-20260714/evidence-mapping/v1"
EVIDENCE_QUESTIONS = EVIDENCE_DIR / "evidence_mapping.v1.questions.jsonl"
PROTOCOL_FREEZE = ROOT / "paper/refine-logs/R403_TASK_EVAL_PROTOCOL_FREEZE.json"
DEFAULT_PREREGISTRATION = (
    ROOT / "paper/refine-logs/R403_TASK_EVAL_PREREGISTRATION.json"
)
DEFAULT_PREFLIGHT = (
    ROOT / "results/paper-experiments-20260714/r403-task-eval-preflight.json"
)
DEFAULT_SOURCE = (
    ROOT / "results/gpt55-flex-benchmarks-20260714/"
    "r403-incremental-growth-gpt55-flex"
)
DEFAULT_FORMAL = (
    ROOT / "results/gpt55-flex-benchmarks-20260714/"
    "r403-checkpoint-task-eval-gpt55-flex"
)
DEFAULT_SYNTHETIC = (
    ROOT / "results/paper-experiments-20260714/"
    "r403-task-eval-synthetic-sanity"
)
BUILD_ACTIVE_LOCK = ROOT / "results/.r403-active-build.lock"

SOURCE_FILES = (
    "paper/refine-logs/R403_TASK_EVAL_PROTOCOL_FREEZE.json",
    "scripts/r403_task_eval_contract.py",
    "scripts/run_r403_task_eval.py",
    "scripts/audit_r403_task_eval.py",
    "scripts/audit_r403_incremental_growth.py",
    "scripts/readonly_nativemem_control.py",
    "scripts/audit_readonly_nativemem_control.py",
    "scripts/controlled_locomo_answer_contract.py",
    "scripts/run_controlled_locomo_answers.py",
    "scripts/controlled_gpt55_run_proxy.py",
    "scripts/gpt55_run_proxy.py",
    "src/openai_gpt55_flex_gateway.py",
    "src/openai_gpt55_flex_gateway_evidence.py",
    "src/evaluation/durable_model_ledger.py",
    "src/evaluation/visible_token_budget.py",
    "src/evaluation/visible_token_audit.py",
    "src/evaluation/metrics.py",
)


class R403TaskEvalError(RuntimeError):
    """The R403 task-evaluation source, protocol, or output is invalid."""


def relative_to_root(path: Path) -> str:
    absolute = path.expanduser().absolute()
    try:
        return absolute.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise R403TaskEvalError(f"path is outside repository: {path}") from exc


def _regular_file(path: Path, *, label: str) -> Path:
    answer_contract.reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise R403TaskEvalError(f"{label} is not a regular file: {path}")
    return path


def _regular_directory(path: Path, *, label: str) -> Path:
    answer_contract.reject_symlink_components(path)
    if path.is_symlink() or not path.is_dir():
        raise R403TaskEvalError(f"{label} is not a regular directory: {path}")
    return path


def _file_binding(path: Path) -> dict[str, Any]:
    checked = _regular_file(path, label="bound file")
    return {
        "path": relative_to_root(checked),
        "bytes": checked.stat().st_size,
        "sha256": answer_contract.sha256_file(checked),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    checked = _regular_file(path, label="JSONL input")
    payload = checked.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise R403TaskEvalError(f"JSONL final line is incomplete: {path}")
    records: list[dict[str, Any]] = []
    for number, raw in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise R403TaskEvalError(
                f"invalid JSONL line {number}: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise R403TaskEvalError(f"JSONL line {number} is not an object")
        records.append(value)
    return records


def _source_hashes() -> dict[str, str]:
    return {
        relative: answer_contract.sha256_file(
            _regular_file(ROOT / relative, label="bound task-eval source")
        )
        for relative in SOURCE_FILES
    }


def _validate_protocol_freeze() -> dict[str, Any]:
    payload = answer_contract.read_json(
        _regular_file(PROTOCOL_FREEZE, label="R403 protocol freeze")
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        != "nativemem.r403-task-eval-protocol-freeze.v1"
        or payload.get("status") != "frozen_before_task_eval_implementation"
        or payload.get("checkpoint_percentages") != list(CHECKPOINTS)
        or "all 1540" not in str(payload.get("denominators", {}).get(
            "task_accuracy_100", ""
        ))
    ):
        raise R403TaskEvalError("R403 denominator protocol freeze differs")
    return payload


def build_preregistration() -> dict[str, Any]:
    """Construct the exact task-evaluation preregistration."""

    tokenizer = answer_contract.formal_token_counter()
    freeze = _validate_protocol_freeze()
    return {
        "schema_version": PREREG_SCHEMA,
        "status": "frozen_before_formal_task_model_requests",
        "experiment_id": "R403-task-evaluation",
        "upstream_experiment": "R403",
        "method": METHOD,
        "scope": {
            "benchmark": "LoCoMo",
            "samples": list(range(10)),
            "checkpoints": list(CHECKPOINTS),
            "primary_categories": [1, 2, 3, 4],
            "primary_questions_at_100_percent": PRIMARY_QUESTIONS,
            "primary_source_recall_at_100_percent": PRIMARY_SOURCE_RECALL,
            "source_exclusions_at_100_percent": SOURCE_EXCLUSIONS,
            "category_counts_at_100_percent": {
                str(key): value for key, value in CATEGORY_COUNTS.items()
            },
        },
        "denominator_protocol": {
            "freeze": _file_binding(PROTOCOL_FREEZE),
            "freeze_content_sha256": answer_contract.canonical_hash(freeze),
            "task_10_25_50": "growth_eligible_true",
            "task_100": "main_qa_100_cohort_true_all_1540",
            "old_fact": "old_fact_cohort_true",
            "update": "update_cohort_true",
            "source_reachability": "source_cohort_eligible_true",
            "labels_must_be_copied_not_inferred": True,
            "primary_correctness": (
                "persist exact answer/gold/category/cohort inputs for the frozen "
                "primary and sensitivity judges"
            ),
            "local_locomo_f1_role": "diagnostic_only",
        },
        "formal_source": {
            "default_root": relative_to_root(DEFAULT_SOURCE),
            "required_schema": growth_auditor.RUN_SCHEMA,
            "required_mode": "formal",
            "required_status": "complete",
            "required_samples": list(range(10)),
            "required_checkpoints": list(CHECKPOINTS),
            "required_model": MODEL,
            "required_live_independent_audit": True,
            "active_build_lock_policy": "reject",
            "binding": (
                "run/sample/checkpoint tree, memory tree, inventory, labels, "
                "task-score binding, and independent-audit SHA-256"
            ),
        },
        "dataset_and_mapping": {
            "dataset": _file_binding(DATA),
            "evidence_questions": _file_binding(EVIDENCE_QUESTIONS),
            "question_text_policy": "R403_inventory_exact_and_raw_dataset_equal",
            "gold_answer_policy": "R403_inventory_exact_and_raw_dataset_equal",
            "gold_labels_model_visible": False,
        },
        "retrieval_and_answer": {
            "retrieval_model": MODEL,
            "answer_model": MODEL,
            "same_retriever_answerer_prompt_all_checkpoints": True,
            "generic_readonly_condition": GENERIC_CONDITION,
            "answer_prompt_sha256": answer_contract.sha256_bytes(
                answer_contract.ANSWER_PROMPT.encode("utf-8")
            ),
            "retrieval_prompt_sha256": answer_contract.sha256_bytes(
                readonly_control.RETRIEVAL_PROMPT.encode("utf-8")
            ),
            "temperature": 0.0,
            "max_retrieval_rounds": MAX_ROUNDS,
            "visible_token_budget_policy": "hard_cap",
            "visible_token_budget_tokens": BUDGET_TOKENS,
            "gate_boundary": "DeliveryResult.delivered_text_only",
            "tokenizer": tokenizer.identity,
            "model_context_limit_tokens": MODEL_CONTEXT_LIMIT_TOKENS,
            "answer_completion_reservation_tokens": (
                ANSWER_COMPLETION_RESERVATION_TOKENS
            ),
            "answer_max_tokens": ANSWER_MAX_TOKENS,
            "answer_retries": ANSWER_RETRIES,
            "retrieval_tools": [
                "list_memory_files",
                "read_memory_file",
                "search_memory",
                "resolve_sources",
            ],
            "source_resolver": (
                "same frozen Dn:m resolver restricted to raw sessions at or "
                "before the checkpoint boundary"
            ),
            "memory_access": "read_only_checkpoint_copy_only",
            "future_checkpoint_access": "forbidden",
        },
        "per_question_outputs": [
            "answer_and_exact_scoring_input",
            "eligibility_and_frozen_cohorts",
            "construction_and_retrieval_source_reachability",
            "old_fact_and_update_correctness_inputs",
            "logical_calls_and_physical_attempts",
            "visible_and_provider_tokens",
            "retrieval_tool_and_provider_latency",
            "memory_before_after_hashes",
            "M4_stage_evidence_and_trace_ids",
            "durable_retrieval_and_answer_ledgers",
            "exclusive_proxy_events_and_prefixes",
        ],
        "durability": {
            "retrieval_hash_chain_ledger": True,
            "answer_hash_chain_ledger": True,
            "request_response_artifacts_fsynced": True,
            "per_question_input_binding": True,
            "per_question_checkpoint": True,
            "no_clobber": True,
            "resume_complete_checkpoints_only_after_independent_audit": True,
            "incomplete_question_policy": "fail_closed_use_new_output_root",
            "orphan_proxy_and_model_call_policy": "reject_or_recover_then_audit",
        },
        "formal_execution_gate": {
            "all_ten_source_required": True,
            "allow_model_requests_flag_required": True,
            "gateway_root_flag_required": True,
            "gateway_origin_derived_from_validated_root": True,
            "arbitrary_upstream_forbidden": True,
            "exclusive_consumer_lock_required": True,
            "exact_provider_window_required": True,
            "provider_model": "gpt-5.5-2026-04-23",
            "service_tier": "flex",
            "scope_limit_flags_available": False,
            "exclusive_controlled_proxy_required": True,
            "new_provider_result_roots_supported": True,
        },
        "source_hashes": _source_hashes(),
    }


def freeze_preregistration(
    path: Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    expected = build_preregistration()
    if path.exists():
        if answer_contract.read_json(path) != expected:
            raise R403TaskEvalError("existing R403 task preregistration differs")
        return expected
    answer_contract.atomic_json_no_clobber(path, expected)
    return expected


def validate_preregistration(
    path: Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    actual = answer_contract.read_json(_regular_file(path, label="preregistration"))
    expected = build_preregistration()
    if actual != expected:
        raise R403TaskEvalError(
            "R403 task preregistration differs from live frozen inputs"
        )
    return actual


def preregistration_binding(path: Path = DEFAULT_PREREGISTRATION) -> dict[str, Any]:
    payload = validate_preregistration(path)
    return {
        "path": relative_to_root(path),
        "sha256": answer_contract.sha256_file(path),
        "content_sha256": answer_contract.canonical_hash(payload),
    }


def _assert_no_active_growth_build() -> None:
    if not BUILD_ACTIVE_LOCK.exists():
        return
    answer_contract.reject_symlink_components(BUILD_ACTIVE_LOCK)
    if BUILD_ACTIVE_LOCK.is_symlink() or not BUILD_ACTIVE_LOCK.is_file():
        raise R403TaskEvalError("R403 build active-lock artifact is unsafe")
    with BUILD_ACTIVE_LOCK.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise R403TaskEvalError("R403 growth source is actively locked") from exc
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


def _checkpoint_binding(
    source_root: Path,
    *,
    sample: int,
    checkpoint: int,
    sample_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_root = (
        source_root / "samples" / f"sample-{sample}" / "checkpoints"
        / f"checkpoint-{checkpoint:03d}"
    )
    _regular_directory(checkpoint_root, label="R403 checkpoint")
    manifest_path = checkpoint_root / "checkpoint_manifest.json"
    inventory_path = checkpoint_root / "question_inventory.jsonl"
    task_binding_path = checkpoint_root / "task_score_binding.json"
    memory = checkpoint_root / "memory"
    checkpoint_manifest = answer_contract.read_json(
        _regular_file(manifest_path, label="checkpoint manifest")
    )
    task_binding = answer_contract.read_json(
        _regular_file(task_binding_path, label="checkpoint task binding")
    )
    inventory = _read_jsonl(inventory_path)
    if (
        not isinstance(checkpoint_manifest, dict)
        or not isinstance(task_binding, dict)
        or checkpoint_manifest.get("sample_index") != sample
        or checkpoint_manifest.get("checkpoint") != checkpoint
        or task_binding.get("sample_index") != sample
        or task_binding.get("checkpoint") != checkpoint
        or checkpoint_manifest.get("question_inventory_sha256")
        != answer_contract.sha256_file(inventory_path)
        or checkpoint_manifest.get("task_score_binding_sha256")
        != answer_contract.sha256_file(task_binding_path)
        or task_binding.get("question_inventory_sha256")
        != answer_contract.sha256_file(inventory_path)
        or task_binding.get("task_results_status") != "not_attached"
        or len(inventory) != len(
            [row for row in inventory if int(row.get("category", 0)) in CATEGORY_COUNTS]
        )
    ):
        raise R403TaskEvalError("R403 checkpoint binding differs")
    expected = sample_manifest.get("result", {}).get("checkpoints", {}).get(
        str(checkpoint)
    )
    tree = growth_auditor.tree_descriptor(checkpoint_root)
    if (
        not isinstance(expected, Mapping)
        or expected.get("checkpoint_tree") != tree
        or expected.get("checkpoint_manifest_sha256")
        != answer_contract.sha256_file(manifest_path)
        or expected.get("inventory_counts")
        != checkpoint_manifest.get("inventory_counts")
    ):
        raise R403TaskEvalError("R403 sample/checkpoint linkage differs")
    readonly_memory = readonly_control.memory_descriptor(memory)
    if (
        readonly_memory.get("file_count")
        != checkpoint_manifest.get("memory_tree", {}).get("file_count")
        or readonly_memory.get("byte_count")
        != checkpoint_manifest.get("memory_tree", {}).get("byte_count")
        or task_binding.get("memory_tree_sha256")
        != checkpoint_manifest.get("memory_tree", {}).get("tree_sha256")
    ):
        raise R403TaskEvalError("R403 checkpoint memory binding differs")
    selected = selected_inventory_rows(inventory, checkpoint=checkpoint)
    return {
        "checkpoint": checkpoint,
        "session_boundary": checkpoint_manifest["session_boundary"],
        "total_sessions": checkpoint_manifest["total_sessions"],
        "checkpoint_root": relative_to_root(checkpoint_root),
        "checkpoint_tree": tree,
        "checkpoint_manifest": _file_binding(manifest_path),
        "memory": {
            "path": relative_to_root(memory),
            "growth_tree": checkpoint_manifest["memory_tree"],
            "readonly_descriptor": readonly_memory,
        },
        "question_inventory": _file_binding(inventory_path),
        "task_score_binding": _file_binding(task_binding_path),
        "frozen_labels_sha256": task_binding["frozen_labels_sha256"],
        "source_mapping_sha256": task_binding["source_mapping_sha256"],
        "cohort_question_ids_sha256": task_binding[
            "cohort_question_ids_sha256"
        ],
        "inventory_counts": checkpoint_manifest["inventory_counts"],
        "selected_question_count": len(selected),
        "selected_question_ids_sha256": answer_contract.canonical_hash(
            [row["question_id"] for row in selected]
        ),
    }


def selected_inventory_rows(
    inventory: Sequence[Mapping[str, Any]], *, checkpoint: int
) -> list[dict[str, Any]]:
    """Apply the preregistered checkpoint denominator without model outputs."""

    if checkpoint not in CHECKPOINTS:
        raise R403TaskEvalError("unknown R403 checkpoint")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in inventory:
        row = dict(raw)
        question_id = str(row.get("question_id", ""))
        if (
            row.get("schema") != growth_auditor.INVENTORY_SCHEMA
            or row.get("checkpoint") != checkpoint
            or not question_id
            or question_id in seen
            or int(row.get("category", 0)) not in CATEGORY_COUNTS
        ):
            raise R403TaskEvalError("R403 checkpoint inventory row differs")
        seen.add(question_id)
        include = (
            bool(row.get("main_qa_100_cohort"))
            if checkpoint == 100
            else bool(row.get("growth_eligible"))
        )
        if include:
            if checkpoint != 100 and not row.get("source_cohort_eligible"):
                raise R403TaskEvalError("partial checkpoint task row lacks source cohort")
            selected.append(row)
    return selected


def _sample_binding(source_root: Path, sample: int) -> dict[str, Any]:
    sample_root = source_root / "samples" / f"sample-{sample}"
    manifest_path = sample_root / "sample_manifest.json"
    manifest = answer_contract.read_json(
        _regular_file(manifest_path, label="R403 sample manifest")
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "complete"
        or manifest.get("sample_index") != sample
    ):
        raise R403TaskEvalError("R403 sample manifest differs")
    checkpoints = [
        _checkpoint_binding(
            source_root,
            sample=sample,
            checkpoint=checkpoint,
            sample_manifest=manifest,
        )
        for checkpoint in CHECKPOINTS
    ]
    return {
        "sample": sample,
        "sample_id": manifest.get("sample_id"),
        "sample_manifest": _file_binding(manifest_path),
        "checkpoints": checkpoints,
        "checkpoints_sha256": answer_contract.canonical_hash(checkpoints),
    }


def audit_growth_source(source_root: Path, *, formal: bool) -> dict[str, Any]:
    """Run the independent R403 auditor twice and bind all task inputs."""

    _assert_no_active_growth_build()
    source_root = source_root.expanduser().absolute()
    _regular_directory(source_root, label="R403 growth source")
    report_first = growth_auditor.audit(source_root)
    if report_first.get("status") != "pass":
        raise R403TaskEvalError("independent R403 growth audit did not pass")
    manifest_path = source_root / "run_manifest.json"
    labels_path = source_root / "frozen_labels.json"
    manifest = answer_contract.read_json(manifest_path)
    expected_mode = "formal" if formal else "synthetic_sanity"
    expected_samples = list(range(10)) if formal else [0]
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "complete"
        or manifest.get("config", {}).get("mode") != expected_mode
        or manifest.get("config", {}).get("samples") != expected_samples
        or manifest.get("config", {}).get("checkpoints") != list(CHECKPOINTS)
        or report_first.get("sample_count") != len(expected_samples)
        or report_first.get("checkpoints") != list(CHECKPOINTS)
    ):
        raise R403TaskEvalError("R403 growth source scope differs")
    if formal and manifest.get("config", {}).get("model") != MODEL:
        raise R403TaskEvalError("formal R403 growth source model differs")
    labels = answer_contract.read_json(
        _regular_file(labels_path, label="R403 frozen labels")
    )
    expected_labels = PRIMARY_QUESTIONS if formal else 4
    expected_exclusions = SOURCE_EXCLUSIONS if formal else 1
    if (
        not isinstance(labels, dict)
        or labels.get("record_count") != expected_labels
        or labels.get("source_exclusion_count") != expected_exclusions
        or manifest.get("frozen_labels_sha256")
        != answer_contract.sha256_file(labels_path)
    ):
        raise R403TaskEvalError("R403 frozen labels scope differs")
    samples = [_sample_binding(source_root, sample) for sample in expected_samples]
    selected_total = sum(
        checkpoint["selected_question_count"]
        for sample in samples
        for checkpoint in sample["checkpoints"]
    )
    selected_at_100 = sum(
        checkpoint["selected_question_count"]
        for sample in samples
        for checkpoint in sample["checkpoints"]
        if checkpoint["checkpoint"] == 100
    )
    if formal and selected_at_100 != PRIMARY_QUESTIONS:
        raise R403TaskEvalError("R403 100-percent main denominator differs")
    if not formal and (selected_total != 8 or selected_at_100 != 4):
        raise R403TaskEvalError("synthetic R403 denominator inventory differs")
    report_second = growth_auditor.audit(source_root)
    if report_second != report_first:
        raise R403TaskEvalError("R403 growth source changed during binding")
    _assert_no_active_growth_build()
    binding: dict[str, Any] = {
        "schema_version": SOURCE_BINDING_SCHEMA,
        "mode": "formal" if formal else "synthetic_no_network",
        "source_root": relative_to_root(source_root),
        "run_manifest": _file_binding(manifest_path),
        "run_fingerprint": manifest.get("run_fingerprint"),
        "frozen_labels": _file_binding(labels_path),
        "frozen_label_records_sha256": labels.get("records_sha256"),
        "independent_auditor": {
            "path": relative_to_root(Path(growth_auditor.__file__)),
            "sha256": answer_contract.sha256_file(Path(growth_auditor.__file__)),
            "status": "pass",
            "report_sha256": answer_contract.canonical_hash(report_first),
            "report": report_first,
        },
        "samples": samples,
        "samples_sha256": answer_contract.canonical_hash(samples),
        "selected_question_checkpoint_artifacts": selected_total,
        "main_question_artifacts_at_100": selected_at_100,
    }
    binding["binding_sha256"] = answer_contract.canonical_hash(binding)
    return binding


def validate_binding_hash(binding: Mapping[str, Any]) -> None:
    payload = dict(binding)
    expected = payload.pop("binding_sha256", None)
    if expected != answer_contract.canonical_hash(payload):
        raise R403TaskEvalError("R403 growth source binding hash differs")


def run_preflight(
    preregistration: Path = DEFAULT_PREREGISTRATION,
    *,
    source_root: Path = DEFAULT_SOURCE,
    write_path: Path | None = DEFAULT_PREFLIGHT,
) -> dict[str, Any]:
    validate_preregistration(preregistration)
    source: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        source = audit_growth_source(source_root, formal=True)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")
    report = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "ready" if source is not None else "blocked",
        "preregistration": preregistration_binding(preregistration),
        "formal_source_root": (
            relative_to_root(source_root)
            if source_root.expanduser().absolute().is_relative_to(ROOT)
            else str(source_root.expanduser().absolute())
        ),
        "formal_source": source,
        "errors": errors,
        "model_requests": 0,
        "network_requests": 0,
    }
    report["report_content_sha256"] = answer_contract.canonical_hash(report)
    if write_path is not None:
        answer_contract.atomic_json_replace(write_path, report)
    return report


def require_ready_binding(preflight: Mapping[str, Any]) -> dict[str, Any]:
    source = preflight.get("formal_source")
    if preflight.get("status") != "ready" or not isinstance(source, dict):
        raise R403TaskEvalError(
            "formal R403 task evaluation requires an independently audited "
            "all-ten formal growth source"
        )
    validate_binding_hash(source)
    return source


def _sample_record(binding: Mapping[str, Any], sample: int) -> dict[str, Any]:
    records = binding.get("samples")
    matches = (
        [row for row in records if row.get("sample") == sample]
        if isinstance(records, list)
        else []
    )
    if len(matches) != 1:
        raise R403TaskEvalError(f"R403 sample binding differs: {sample}")
    return matches[0]


def _checkpoint_record(
    binding: Mapping[str, Any], *, sample: int, checkpoint: int
) -> dict[str, Any]:
    sample_record = _sample_record(binding, sample)
    records = sample_record.get("checkpoints")
    matches = (
        [row for row in records if row.get("checkpoint") == checkpoint]
        if isinstance(records, list)
        else []
    )
    if len(matches) != 1:
        raise R403TaskEvalError("R403 checkpoint binding differs")
    return matches[0]


def _dataset(formal: bool) -> list[dict[str, Any]]:
    if formal:
        payload = answer_contract.read_json(_regular_file(DATA, label="LoCoMo data"))
        if not isinstance(payload, list) or len(payload) != 10:
            raise R403TaskEvalError("LoCoMo dataset differs")
        return payload
    synthetic, _evidence = growth_auditor.synthetic_inputs()
    return synthetic


def conversation_prefix(
    *, sample: int, session_boundary: int, formal: bool
) -> dict[str, Any]:
    dataset = _dataset(formal)
    conversation = dataset[sample].get("conversation")
    if not isinstance(conversation, dict):
        raise R403TaskEvalError("R403 source conversation differs")
    total = growth_auditor.session_count(conversation)
    if not 1 <= session_boundary <= total:
        raise R403TaskEvalError("R403 checkpoint session boundary differs")
    prefix: dict[str, Any] = {
        key: value
        for key, value in conversation.items()
        if key in {"speaker_a", "speaker_b"}
    }
    for session in range(1, session_boundary + 1):
        key = f"session_{session}"
        prefix[key] = conversation[key]
        date_key = f"{key}_date_time"
        if date_key in conversation:
            prefix[date_key] = conversation[date_key]
    return prefix


def load_checkpoint_context(
    source_root: Path,
    source_binding: Mapping[str, Any],
    *,
    sample: int,
    checkpoint: int,
    formal: bool,
) -> dict[str, Any]:
    validate_binding_hash(source_binding)
    record = _checkpoint_record(
        source_binding, sample=sample, checkpoint=checkpoint
    )
    checkpoint_root = source_root / record["checkpoint_root"]
    if source_root == ROOT:
        checkpoint_root = ROOT / record["checkpoint_root"]
    else:
        checkpoint_root = ROOT / record["checkpoint_root"]
    _regular_directory(checkpoint_root, label="bound checkpoint root")
    if growth_auditor.tree_descriptor(checkpoint_root) != record["checkpoint_tree"]:
        raise R403TaskEvalError("R403 checkpoint tree changed after binding")
    memory = checkpoint_root / "memory"
    if readonly_control.memory_descriptor(memory) != record["memory"][
        "readonly_descriptor"
    ]:
        raise R403TaskEvalError("R403 checkpoint memory changed after binding")
    inventory = _read_jsonl(checkpoint_root / "question_inventory.jsonl")
    if (
        answer_contract.sha256_file(checkpoint_root / "question_inventory.jsonl")
        != record["question_inventory"]["sha256"]
    ):
        raise R403TaskEvalError("R403 checkpoint inventory changed after binding")
    selected = selected_inventory_rows(inventory, checkpoint=checkpoint)
    if (
        len(selected) != record["selected_question_count"]
        or answer_contract.canonical_hash([row["question_id"] for row in selected])
        != record["selected_question_ids_sha256"]
    ):
        raise R403TaskEvalError("R403 selected checkpoint inventory differs")
    conversation = conversation_prefix(
        sample=sample,
        session_boundary=int(record["session_boundary"]),
        formal=formal,
    )
    turn_index = readonly_control.build_turn_index(conversation)
    return {
        "binding": record,
        "checkpoint_root": checkpoint_root,
        "memory": memory,
        "inventory": inventory,
        "selected": selected,
        "conversation": conversation,
        "turn_index": turn_index,
    }


def specs_from_binding(
    source_root: Path,
    source_binding: Mapping[str, Any],
    *,
    formal: bool,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    dataset = _dataset(formal)
    expected_samples = list(range(10)) if formal else [0]
    for sample in expected_samples:
        for checkpoint in CHECKPOINTS:
            context = load_checkpoint_context(
                source_root,
                source_binding,
                sample=sample,
                checkpoint=checkpoint,
                formal=formal,
            )
            boundary = int(context["binding"]["session_boundary"])
            for row in context["selected"]:
                question_index = int(row["question_index"])
                qa = dataset[sample]["qa"][question_index]
                if (
                    str(qa.get("question")) != row.get("question")
                    or qa.get("answer") != row.get("gold")
                    or int(qa.get("category", 0)) != int(row.get("category", 0))
                ):
                    raise R403TaskEvalError("R403 inventory/raw QA binding differs")
                sources = [str(value) for value in row["normalized_source_ids"]]
                if any(growth_auditor.dia_session(value) > boundary for value in sources):
                    raise R403TaskEvalError("selected R403 question uses future source")
                if row.get("source_cohort_eligible") and not set(sources).issubset(
                    context["turn_index"]
                ):
                    raise R403TaskEvalError("checkpoint source is absent from prefix")
                specs.append(
                    {
                        "sample": sample,
                        "sample_id": row["sample_id"],
                        "checkpoint": checkpoint,
                        "session_boundary": boundary,
                        "question_index": question_index,
                        "question_id": row["question_id"],
                        "artifact_id": (
                            f"s{sample:02d}-cp{checkpoint:03d}-q{question_index:03d}"
                        ),
                        "question": row["question"],
                        "gold_answer": row["gold"],
                        "gold_answer_sha256": answer_contract.canonical_hash(
                            row["gold"]
                        ),
                        "category": int(row["category"]),
                        "gold_source_ids": sources,
                        "source_recall_eligible": bool(
                            row["source_cohort_eligible"]
                        ),
                        "cohorts": {
                            "task": True,
                            "growth": bool(row["growth_eligible"]),
                            "source": bool(row["source_cohort_eligible"]),
                            "old_fact": bool(row["old_fact_cohort"]),
                            "update": bool(row["update_cohort"]),
                            "main_qa_100": bool(row["main_qa_100_cohort"]),
                        },
                        "source_reachable_at_build": row["source_reachable"],
                        "source_exclusion_reasons": row[
                            "source_exclusion_reasons"
                        ],
                        "upstream_trace_id": row["trace_id"],
                        "upstream_stage_evidence": row["stage_evidence"],
                        "inventory_row_sha256": answer_contract.canonical_hash(row),
                    }
                )
    if formal:
        at_100 = [spec for spec in specs if spec["checkpoint"] == 100]
        if (
            len(at_100) != PRIMARY_QUESTIONS
            or Counter(spec["category"] for spec in at_100) != CATEGORY_COUNTS
            or sum(spec["source_recall_eligible"] for spec in at_100)
            != PRIMARY_SOURCE_RECALL
        ):
            raise R403TaskEvalError("formal R403 task inventory differs")
    elif len(specs) != 8:
        raise R403TaskEvalError("synthetic R403 task inventory differs")
    return specs


def input_binding(
    *,
    preregistration: Path,
    source_binding: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    spec: Mapping[str, Any],
    memory: Path,
    turn_index: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    payload = {
        "schema_version": INPUT_BINDING_SCHEMA,
        "preregistration": preregistration_binding(preregistration),
        "source_binding_sha256": source_binding["binding_sha256"],
        "upstream": {
            "run_manifest_sha256": source_binding["run_manifest"]["sha256"],
            "frozen_labels_sha256": source_binding["frozen_labels"]["sha256"],
            "sample_manifest_sha256": _sample_record(
                source_binding, int(spec["sample"])
            )["sample_manifest"]["sha256"],
            "checkpoint_tree_sha256": checkpoint_binding["checkpoint_tree"][
                "tree_sha256"
            ],
            "checkpoint_manifest_sha256": checkpoint_binding[
                "checkpoint_manifest"
            ]["sha256"],
            "question_inventory_sha256": checkpoint_binding[
                "question_inventory"
            ]["sha256"],
            "task_score_binding_sha256": checkpoint_binding[
                "task_score_binding"
            ]["sha256"],
            "growth_memory_tree_sha256": checkpoint_binding["memory"][
                "growth_tree"
            ]["tree_sha256"],
            "independent_growth_audit_sha256": source_binding[
                "independent_auditor"
            ]["report_sha256"],
        },
        "checkpoint": {
            "sample": spec["sample"],
            "sample_id": spec["sample_id"],
            "percent": spec["checkpoint"],
            "session_boundary": spec["session_boundary"],
            "total_sessions": checkpoint_binding["total_sessions"],
            "future_checkpoint_access": False,
        },
        "question": {
            "question_id": spec["question_id"],
            "artifact_id": spec["artifact_id"],
            "question_index": spec["question_index"],
            "question_sha256": answer_contract.sha256_bytes(
                str(spec["question"]).encode("utf-8")
            ),
            "gold_answer_sha256": spec["gold_answer_sha256"],
            "category": spec["category"],
            "gold_source_ids": spec["gold_source_ids"],
            "source_recall_eligible": spec["source_recall_eligible"],
            "source_exclusion_reasons": spec["source_exclusion_reasons"],
            "cohorts": spec["cohorts"],
            "inventory_row_sha256": spec["inventory_row_sha256"],
            "upstream_trace_id": spec["upstream_trace_id"],
            "upstream_stage_evidence_sha256": answer_contract.canonical_hash(
                spec["upstream_stage_evidence"]
            ),
        },
        "memory": readonly_control.memory_descriptor(memory),
        "turn_index_sha256": answer_contract.canonical_hash(turn_index),
        "turn_index_session_max": spec["session_boundary"],
        "protocol": {
            "method": METHOD,
            "condition": GENERIC_CONDITION,
            "model": MODEL,
            "budget_tokens": BUDGET_TOKENS,
            "max_rounds": MAX_ROUNDS,
            "model_context_limit_tokens": MODEL_CONTEXT_LIMIT_TOKENS,
            "answer_completion_reservation_tokens": (
                ANSWER_COMPLETION_RESERVATION_TOKENS
            ),
            "answer_max_tokens": ANSWER_MAX_TOKENS,
            "answer_retries": ANSWER_RETRIES,
            "answer_prompt_sha256": answer_contract.sha256_bytes(
                answer_contract.ANSWER_PROMPT.encode("utf-8")
            ),
        },
    }
    payload["binding_sha256"] = answer_contract.canonical_hash(payload)
    return payload


def _visible_trace(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def build_m4_trace(
    *,
    attempt: Path,
    spec: Mapping[str, Any],
    turn_index: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    result = answer_contract.read_json(attempt / "result.json")
    trace_name = result.get("artifacts", {}).get("visible_token_trace")
    if not isinstance(trace_name, str) or Path(trace_name).name != trace_name:
        raise R403TaskEvalError("R403 visible-token trace path differs")
    records = _visible_trace(attempt / trace_name)
    gold = list(dict.fromkeys(str(value) for value in spec["gold_source_ids"]))
    retrieval_hits: set[str] = set()
    resolution_hits: set[str] = set()
    trace_ids: dict[str, list[str]] = {
        key: list(spec["upstream_stage_evidence"].get("trace_ids", {}).get(key, []))
        for key in ("mapping", "canonical_entries", "maintenance", "paths")
    }
    trace_ids.update({"retrieval": [], "source_resolution": []})
    for record in records:
        if record.get("record_type") != "delivery":
            continue
        delivered = record.get("delivered")
        if not isinstance(delivered, Mapping):
            continue
        text = str(delivered.get("text", ""))
        ids = readonly_control.source_ids_in_text(text)
        future = [
            value for value in ids
            if growth_auditor.dia_session(value) > int(spec["session_boundary"])
        ]
        if future:
            raise R403TaskEvalError(
                f"checkpoint delivery contains future source IDs: {future}"
            )
        event_trace = f"delivery:{record.get('event_id')}:{delivered.get('sha256')}"
        if record.get("kind") == "source_resolution":
            for value in gold:
                turn = turn_index.get(value)
                if turn is None:
                    continue
                expected = (
                    f"[{value}] ({turn['date']}) {turn['speaker']}: {turn['text']}"
                )
                if value in ids and expected in text:
                    resolution_hits.add(value)
                    if event_trace not in trace_ids["source_resolution"]:
                        trace_ids["source_resolution"].append(event_trace)
        else:
            hits = set(ids) & set(gold)
            if hits:
                retrieval_hits.update(hits)
                if event_trace not in trace_ids["retrieval"]:
                    trace_ids["retrieval"].append(event_trace)
    eligible = bool(spec["source_recall_eligible"])
    upstream = spec["upstream_stage_evidence"].get("m4_fields", {})
    if eligible:
        retrieval_complete: bool | None = set(gold).issubset(retrieval_hits)
        resolution_complete: bool | None = set(gold).issubset(resolution_hits)
        status = "complete"
    else:
        retrieval_complete = None
        resolution_complete = None
        status = "excluded_incomplete_mapping"
    fields = {
        "gold_source_mapping_complete": bool(eligible),
        "gold_source_in_canonical_entries": upstream.get(
            "gold_source_in_canonical_entries"
        ),
        "gold_source_survived_maintenance": upstream.get(
            "gold_source_survived_maintenance"
        ),
        "gold_source_path_valid": upstream.get("gold_source_path_valid"),
        "retrieval_reached_gold_source": retrieval_complete,
        "source_resolution_returned_gold_content": resolution_complete,
    }
    return {
        "schema_version": M4_TRACE_SCHEMA,
        "question_id": spec["question_id"],
        "sample": spec["sample"],
        "checkpoint": spec["checkpoint"],
        "session_boundary": spec["session_boundary"],
        "upstream_trace_id": spec["upstream_trace_id"],
        "upstream_stage_evidence_sha256": answer_contract.canonical_hash(
            spec["upstream_stage_evidence"]
        ),
        "pre_retrieval_stage_evidence": spec["upstream_stage_evidence"],
        "gold_source_ids": gold,
        "source_recall_eligible": eligible,
        "retrieval_hit_source_ids": [value for value in gold if value in retrieval_hits],
        "source_resolution_hit_source_ids": [
            value for value in gold if value in resolution_hits
        ],
        "m4_fields": fields,
        "trace_ids": trace_ids,
        "evidence_status": status,
        "future_source_ids_observed": [],
    }


def official_task_score(*, answer: str, gold_answer: str, category: int) -> float:
    if category not in CATEGORY_COUNTS:
        raise R403TaskEvalError("R403 task score received non-primary category")
    score = float(f1_locomo_official(answer, gold_answer, category))
    if not 0.0 <= score <= 1.0:
        raise R403TaskEvalError("R403 task diagnostic is outside [0,1]")
    return score


def _provider_accounting(
    attempt: Path, result: Mapping[str, Any]
) -> dict[str, Any]:
    retrieval = read_ledger(attempt / "retrieval_model_ledger.jsonl")
    starts = [row for row in retrieval if row.get("event") == "model_call_started"]
    finishes = [row for row in retrieval if row.get("event") == "model_call_finished"]
    if len(starts) != len(finishes):
        raise R403TaskEvalError("R403 retrieval model lifecycle differs")
    answer = result["answer"]
    answer_events = answer.get("proxy_evidence", {}).get("events", [])
    successful_answer_events = [
        row for row in answer_events if row.get("status") == "success"
    ]
    if len(successful_answer_events) != 1:
        raise R403TaskEvalError("R403 answer proxy success count differs")
    retrieval_usage = {
        key: sum(int(row.get("usage", {}).get(key) or 0) for row in finishes)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    answer_usage = {
        key: int(answer.get("usage", {}).get(key) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    total_usage = {
        key: retrieval_usage[key] + answer_usage[key]
        for key in retrieval_usage
    }
    answer_latency = 0.0
    for event in answer_events:
        try:
            start = datetime.fromisoformat(str(event["started_at"]))
            finish = datetime.fromisoformat(str(event["finished_at"]))
        except (KeyError, ValueError, TypeError):
            continue
        answer_latency += max(0.0, (finish - start).total_seconds())
    return {
        "logical_calls": {
            "retrieval": len(starts),
            "answer": int(answer["logical_calls"]),
            "total": len(starts) + int(answer["logical_calls"]),
        },
        "client_http_attempts": {
            "retrieval": sum(
                int(row.get("proxy_evidence", {}).get("client_http_attempts") or 0)
                for row in finishes
            ),
            "answer": int(answer["client_http_attempts"]),
        },
        "physical_upstream_attempts": {
            "retrieval": sum(
                int(row.get("proxy_evidence", {}).get("upstream_http_attempts") or 0)
                for row in finishes
            ),
            "answer": int(answer["upstream_http_attempts"]),
        },
        "local_model_visible_tokens": {
            "retrieval_request_sum": sum(
                int(row.get("local_visible_tokens") or 0) for row in starts
            ),
            "shared_memory_delivered": int(result["budget"]["visible_tokens"]),
            "answer_prompt": int(result["prompt"]["local_tokens"]),
        },
        "provider_tokens": {
            "retrieval": retrieval_usage,
            "answer": answer_usage,
            "total": total_usage,
        },
        "latency_s": {
            "retrieval_model": round(
                sum(float(row.get("latency_s") or 0.0) for row in finishes), 6
            ),
            "retrieval_total": float(result["retrieval"]["latency_s"]),
            "tool_total": float(result["diagnostics"]["tool_latency_s"]),
            "answer_proxy": round(answer_latency, 6),
        },
        "response_ids": [
            *[str(row["response_id"]) for row in finishes],
            str(answer["response_id"]),
        ],
        "proxy_event_ids": [
            *[
                str(event["event_id"])
                for row in finishes
                for event in row.get("proxy_evidence", {}).get("events", [])
            ],
            *[str(event["event_id"]) for event in answer_events],
        ],
    }


def build_question_metrics(
    *,
    attempt: Path,
    spec: Mapping[str, Any],
    result: Mapping[str, Any],
    m4_trace: Mapping[str, Any],
) -> dict[str, Any]:
    accounting = _provider_accounting(attempt, result)
    diagnostics = result["diagnostics"]
    return {
        "schema_version": METRICS_SCHEMA,
        "question_id": spec["question_id"],
        "artifact_id": spec["artifact_id"],
        "sample": spec["sample"],
        "checkpoint": spec["checkpoint"],
        "session_boundary": spec["session_boundary"],
        "category": spec["category"],
        "eligibility": {
            "task_evaluated": True,
            "source_recall_eligible": spec["source_recall_eligible"],
            "source_exclusion_reasons": spec["source_exclusion_reasons"],
            "cohorts": spec["cohorts"],
        },
        "correctness_input": {
            "answer": result["answer"]["text"],
            "answer_sha256": answer_contract.canonical_hash(
                result["answer"]["text"]
            ),
            "gold_answer": spec["gold_answer"],
            "gold_answer_sha256": spec["gold_answer_sha256"],
            "category": spec["category"],
            "primary_judge_status": "not_attached",
            "sensitivity_judge_status": "not_attached",
            "diagnostic_official_locomo_f1": official_task_score(
                answer=str(result["answer"]["text"]),
                gold_answer=str(spec["gold_answer"]),
                category=int(spec["category"]),
            ),
        },
        "source_reachability": {
            "build_checkpoint_reachable": spec["source_reachable_at_build"],
            "retrieval_reached_all_gold": m4_trace["m4_fields"][
                "retrieval_reached_gold_source"
            ],
            "source_resolution_returned_all_gold": m4_trace["m4_fields"][
                "source_resolution_returned_gold_content"
            ],
            "mapped_source_recall": diagnostics["mapped_source_recall"],
            "first_relevant_file": diagnostics["first_relevant_file"],
        },
        "calls_tokens_latency": accounting,
        "navigation": {
            "read_calls": diagnostics["read_calls"],
            "tool_calls": diagnostics["tool_calls"],
            "visible_tokens": result["budget"]["visible_tokens"],
            "source_resolution_tokens": result["budget"][
                "source_resolution_tokens"
            ],
            "configured_hard_cap_tokens": result["budget"][
                "configured_tokens"
            ],
        },
        "memory_tree": {
            "before_sha256": result["memory"]["before"]["sha256"],
            "after_sha256": result["memory"]["after"]["sha256"],
            "unchanged": result["memory"]["unchanged"],
        },
        "m4": {
            "trace_sha256": answer_contract.canonical_hash(m4_trace),
            "fields": m4_trace["m4_fields"],
            "trace_ids": m4_trace["trace_ids"],
            "evidence_status": m4_trace["evidence_status"],
        },
        "durable_artifacts": {
            "retrieval_ledger": {
                "path": result["artifacts"]["retrieval_model_ledger"],
                "sha256": result["artifacts"]["retrieval_model_ledger_sha256"],
            },
            "answer_ledger": {
                "path": result["artifacts"]["answer_ledger"],
                "sha256": result["artifacts"]["answer_ledger_sha256"],
            },
            "visible_token_trace": {
                "path": result["artifacts"]["visible_token_trace"],
                "sha256": result["artifacts"]["visible_token_trace_sha256"],
            },
            "visible_token_manifest": {
                "path": result["artifacts"]["visible_token_manifest"],
                "sha256": result["artifacts"]["visible_token_manifest_sha256"],
            },
        },
    }


def verify_source_unchanged(
    *, source_root: Path, source_binding: Mapping[str, Any], formal: bool
) -> None:
    live = audit_growth_source(source_root, formal=formal)
    if live != source_binding:
        raise R403TaskEvalError("R403 growth source changed after task binding")


__all__ = [
    "ANSWER_COMPLETION_RESERVATION_TOKENS",
    "ANSWER_MAX_TOKENS",
    "ANSWER_RETRIES",
    "BUDGET_TOKENS",
    "CHECKPOINTS",
    "DEFAULT_FORMAL",
    "DEFAULT_PREFLIGHT",
    "DEFAULT_PREREGISTRATION",
    "DEFAULT_SOURCE",
    "DEFAULT_SYNTHETIC",
    "GENERIC_CONDITION",
    "MAX_ROUNDS",
    "METHOD",
    "MODEL",
    "MODEL_CONTEXT_LIMIT_TOKENS",
    "PRIMARY_QUESTIONS",
    "PRIMARY_SOURCE_RECALL",
    "R403TaskEvalError",
    "audit_growth_source",
    "build_m4_trace",
    "build_preregistration",
    "build_question_metrics",
    "conversation_prefix",
    "freeze_preregistration",
    "input_binding",
    "load_checkpoint_context",
    "preregistration_binding",
    "relative_to_root",
    "require_ready_binding",
    "run_preflight",
    "selected_inventory_rows",
    "specs_from_binding",
    "validate_binding_hash",
    "validate_preregistration",
    "verify_source_unchanged",
]
