#!/usr/bin/env python3
"""Frozen contract for R207 task evaluation over path-only memories.

Formal execution is deliberately downstream of the independently audited
all-ten R207 build.  This module reconstructs the primary LoCoMo inventory,
binds both path conditions byte-for-byte, and supplies the exact R203-style
provenance consumed by the shared read-only retrieval/answer implementation.
It does not send model or network requests.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_r207_path_control as r207_auditor  # noqa: E402
import controlled_locomo_answer_contract as answer_contract  # noqa: E402
import r116_formal_contract as r116_contract  # noqa: E402
import readonly_nativemem_control as readonly_control  # noqa: E402
from src.evaluation.durable_model_ledger import read_ledger  # noqa: E402
from src.evaluation.metrics import f1_locomo_official  # noqa: E402


PREREG_SCHEMA = "nativemem.r207-task-eval-preregistration.v1"
PREFLIGHT_SCHEMA = "nativemem.r207-task-eval-preflight.v1"
SOURCE_BINDING_SCHEMA = "nativemem.r207-task-eval-source-binding.v1"
INPUT_BINDING_SCHEMA = "nativemem.r207-task-eval-input-binding.v1"
M4_TRACE_SCHEMA = "nativemem.r207-task-eval-m4-trace.v1"
METRICS_SCHEMA = "nativemem.r207-task-eval-question-metrics.v1"

METHOD = "r207_task_evaluation"
GENERIC_CONDITION = "dual_source"
PATH_CONDITIONS = ("model_directed", "deterministic_permutation")
MODEL = "gpt-5.5"
BUDGET_TOKENS = 20_000
MAX_ROUNDS = 12
MODEL_CONTEXT_LIMIT_TOKENS = 128_000
ANSWER_COMPLETION_RESERVATION_TOKENS = 4_096
ANSWER_MAX_TOKENS = 4_096
ANSWER_RETRIES = 3

PRIMARY_QUESTIONS = 1_540
PRIMARY_SOURCE_RECALL = 1_533
CATEGORY_COUNTS = {1: 282, 2: 321, 3: 96, 4: 841}
CATEGORY_5_SEPARATE = 446
FORMAL_ARTIFACTS = PRIMARY_QUESTIONS * len(PATH_CONDITIONS)

DEFAULT_PREREGISTRATION = (
    ROOT / "paper/refine-logs/R207_TASK_EVAL_PREREGISTRATION.json"
)
DEFAULT_PREFLIGHT = (
    ROOT / "results/paper-experiments-20260714/r207-task-eval-preflight.json"
)
DEFAULT_R207_ROOT = (
    ROOT / "results/paper-experiments-20260714/r207-path-control-gpt55"
)
DEFAULT_SYNTHETIC_SOURCE = (
    ROOT
    / "results/paper-experiments-20260714/r207-task-eval-r207-source-synthetic"
)

SOURCE_FILES = (
    "scripts/r207_task_eval_contract.py",
    "scripts/run_r207_task_eval.py",
    "scripts/audit_r207_task_eval.py",
    "scripts/run_r207_path_control.py",
    "scripts/audit_r207_path_control.py",
    "scripts/r116_formal_contract.py",
    "scripts/run_r116_formal.py",
    "scripts/readonly_nativemem_control.py",
    "scripts/audit_readonly_nativemem_control.py",
    "scripts/controlled_locomo_answer_contract.py",
    "scripts/run_controlled_locomo_answers.py",
    "scripts/controlled_gpt55_run_proxy.py",
    "scripts/gpt55_run_proxy.py",
    "scripts/audit_benchmark_gold_sources.py",
    "src/evaluation/durable_model_ledger.py",
    "src/evaluation/visible_token_budget.py",
    "src/evaluation/visible_token_audit.py",
    "src/evaluation/prompts.py",
    "src/evaluation/metrics.py",
    "src/nativemem.py",
    "src/v8_memory.py",
    "src/adapters/run_nativemem.py",
    "src/openai_gpt55_flex_gateway.py",
    "src/openai_gpt55_flex_gateway_evidence.py",
)


class R207TaskEvalError(RuntimeError):
    """The frozen R207 task-evaluation contract was violated."""


def relative_to_root(path: Path) -> str:
    absolute = path.expanduser().absolute()
    try:
        return absolute.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise R207TaskEvalError(f"path is outside the repository: {path}") from exc


def _regular_file(path: Path, *, label: str) -> Path:
    answer_contract.reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise R207TaskEvalError(f"{label} is not a regular file: {path}")
    return path


def _regular_directory(path: Path, *, label: str) -> Path:
    answer_contract.reject_symlink_components(path)
    if path.is_symlink() or not path.is_dir():
        raise R207TaskEvalError(f"{label} is not a regular directory: {path}")
    return path


def _file_binding(path: Path) -> dict[str, Any]:
    checked = _regular_file(path, label="bound file")
    return {
        "path": relative_to_root(checked),
        "bytes": checked.stat().st_size,
        "sha256": answer_contract.sha256_file(checked),
    }


def _source_hashes() -> dict[str, str]:
    return {
        relative: answer_contract.sha256_file(
            _regular_file(ROOT / relative, label="bound source")
        )
        for relative in SOURCE_FILES
    }


def primary_question_specs() -> list[dict[str, Any]]:
    """Rebuild the exact LoCoMo category 1--4 inventory from R002."""

    all_specs = r116_contract.question_specs(r116_contract.LOCOMO)
    dataset = answer_contract.read_json(r116_contract.LOCOMO_DATA)
    if not isinstance(dataset, list) or len(dataset) != 10:
        raise R207TaskEvalError("LoCoMo dataset inventory differs")
    specs: list[dict[str, Any]] = []
    for original in all_specs:
        category = int(original["category"])
        if category not in CATEGORY_COUNTS:
            continue
        sample = dataset[int(original["dataset_index"])]
        qa = sample["qa"][int(original["question_index"])]
        if (
            qa.get("question") != original["question"]
            or int(qa.get("category", -1)) != category
            or not isinstance(qa.get("answer"), (str, int, float))
            or isinstance(qa.get("answer"), bool)
        ):
            raise R207TaskEvalError(
                f"raw LoCoMo scoring row differs: {original['question_id']}"
            )
        spec = dict(original)
        spec["gold_answer"] = str(qa["answer"])
        spec["gold_answer_sha256"] = answer_contract.sha256_bytes(
            spec["gold_answer"].encode("utf-8")
        )
        spec["primary_scoring_eligible"] = True
        specs.append(spec)
    counts = Counter(int(spec["category"]) for spec in specs)
    if (
        len(specs) != PRIMARY_QUESTIONS
        or dict(sorted(counts.items())) != CATEGORY_COUNTS
        or sum(bool(spec["source_recall_eligible"]) for spec in specs)
        != PRIMARY_SOURCE_RECALL
    ):
        raise R207TaskEvalError("primary LoCoMo scope differs")
    return specs


def specs_by_sample() -> dict[int, list[dict[str, Any]]]:
    grouped = {sample: [] for sample in range(10)}
    for spec in primary_question_specs():
        grouped[int(spec["dataset_index"])].append(spec)
    return grouped


def synthetic_spec() -> dict[str, Any]:
    return {
        "benchmark": "synthetic-r207-task-eval",
        "question_id": "synthetic:r207-task:q000",
        "artifact_id": "s00-q000",
        "dataset_index": 0,
        "question_index": 0,
        "question": "What did A adopt?",
        "category": 2,
        "gold_answer": "a dog",
        "gold_answer_sha256": answer_contract.sha256_bytes(b"a dog"),
        "gold_source_ids": ["D1:1"],
        "source_recall_eligible": True,
        "primary_scoring_eligible": True,
        "evidence_record_sha256": answer_contract.canonical_hash(
            {
                "question_id": "synthetic:r207-task:q000",
                "gold_source_ids": ["D1:1"],
            }
        ),
    }


def synthetic_conversation() -> dict[str, Any]:
    return {
        "speaker_a": "A",
        "speaker_b": "B",
        "session_1": [
            {"dia_id": "D1:1", "speaker": "A", "text": "A adopted a dog"},
            {"dia_id": "D1:2", "speaker": "B", "text": "B moved home"},
        ],
        "session_1_date_time": "2023-05-07",
        "session_2": [
            {"dia_id": "D2:1", "speaker": "A", "text": "A visited a park"}
        ],
        "session_2_date_time": "2023-05-08",
    }


def conversation_for_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if spec.get("benchmark") == r116_contract.LOCOMO:
        return r116_contract.locomo_conversation(spec)
    if spec.get("benchmark") == "synthetic-r207-task-eval":
        return synthetic_conversation()
    raise R207TaskEvalError("unknown question benchmark")


def build_preregistration() -> dict[str, Any]:
    tokenizer = answer_contract.formal_token_counter()
    return {
        "schema_version": PREREG_SCHEMA,
        "status": "frozen_before_formal_model_requests",
        "experiment_id": "R207-task-evaluation",
        "upstream_experiment": "R207",
        "method": METHOD,
        "path_conditions": list(PATH_CONDITIONS),
        "scope": {
            "benchmark": "LoCoMo",
            "samples": list(range(10)),
            "primary_categories": [1, 2, 3, 4],
            "primary_questions_per_condition": PRIMARY_QUESTIONS,
            "primary_question_condition_artifacts": FORMAL_ARTIFACTS,
            "category_counts": {str(k): v for k, v in CATEGORY_COUNTS.items()},
            "source_recall_questions_per_condition": PRIMARY_SOURCE_RECALL,
            "category_5_questions": CATEGORY_5_SEPARATE,
            "category_5_policy": (
                "excluded_from_this_primary_run; any retained category-5 output "
                "must use a separate artifact and score summary"
            ),
        },
        "formal_source": {
            "default_root": relative_to_root(DEFAULT_R207_ROOT),
            "required_schema": r207_auditor.RUN_SCHEMA,
            "required_mode": "locomo",
            "required_samples": list(range(10)),
            "required_live_independent_audit": True,
            "required_audit_status": "pass",
            "required_question_traces": FORMAL_ARTIFACTS,
            "required_conditions": list(PATH_CONDITIONS),
            "memory_binding": "per-condition directory bytes, paths, and SHA-256",
        },
        "dataset_and_mapping": {
            "dataset": _file_binding(r116_contract.LOCOMO_DATA),
            "r002_questions": _file_binding(r116_contract.EVIDENCE_QUESTIONS),
            "r002_manifest": _file_binding(r116_contract.EVIDENCE_MANIFEST),
            "r002_audit": _file_binding(r116_contract.EVIDENCE_AUDIT),
            "question_text_policy": "raw_dataset_question_exact",
            "gold_answer_policy": "str(raw_dataset_answer)_exact",
            "gold_source_policy": "R002 normalized_source_ids",
            "gold_labels_model_visible": False,
        },
        "retrieval_and_answer": {
            "retrieval_model": MODEL,
            "answer_model": MODEL,
            "same_retriever_and_answerer_for_both_conditions": True,
            "generic_readonly_condition": GENERIC_CONDITION,
            "answer_prompt_sha256": answer_contract.sha256_bytes(
                answer_contract.ANSWER_PROMPT.encode("utf-8")
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
            "source_resolver": "same frozen Dn:m turn resolver",
            "memory_access": "read_only",
        },
        "per_question_outputs": [
            "first_relevant_file",
            "mapped_source_recall",
            "navigation_calls",
            "navigation_tokens",
            "memory_tree_before_after_sha256",
            "R004_visible_token_trace",
            "M4_stage_trace",
            "durable_retrieval_and_answer_ledgers",
            "exclusive_proxy_evidence",
            "official_LoCoMo_F1_primary_only",
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
        },
        "formal_execution_gate": {
            "all_ten_source_required": True,
            "allow_model_requests_flag_required": True,
            "audited_R207_source_required": True,
            "gateway_root_flag_required": True,
            "gateway_origin_derived_from_validated_root": True,
            "arbitrary_upstream_forbidden": True,
            "exclusive_consumer_lock_required": True,
            "exact_provider_window_required": True,
            "provider_model": "gpt-5.5-2026-04-23",
            "service_tier": "flex",
            "scope_limit_flags_available": False,
            "exclusive_controlled_proxy_required": True,
        },
        "source_hashes": _source_hashes(),
    }


def freeze_preregistration(
    path: Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    expected = build_preregistration()
    if path.exists():
        if answer_contract.read_json(path) != expected:
            raise R207TaskEvalError("existing R207 task preregistration differs")
        return expected
    answer_contract.atomic_json_no_clobber(path, expected)
    return expected


def validate_preregistration(
    path: Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    actual = answer_contract.read_json(_regular_file(path, label="preregistration"))
    expected = build_preregistration()
    if actual != expected:
        raise R207TaskEvalError("R207 task preregistration differs from live inputs")
    return actual


def preregistration_binding(path: Path = DEFAULT_PREREGISTRATION) -> dict[str, Any]:
    payload = validate_preregistration(path)
    return {
        "path": relative_to_root(path),
        "sha256": answer_contract.sha256_file(path),
        "content_sha256": answer_contract.canonical_hash(payload),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    checked = _regular_file(path, label="JSONL input")
    payload = checked.read_bytes()
    if not payload:
        return []
    if not payload.endswith(b"\n"):
        raise R207TaskEvalError(f"JSONL has an incomplete final line: {path}")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise R207TaskEvalError(f"invalid JSONL line {number}: {path}") from exc
        if not isinstance(value, dict):
            raise R207TaskEvalError(f"JSONL line {number} is not an object")
        records.append(value)
    return records


def _sample_binding(source_root: Path, sample: int) -> dict[str, Any]:
    sample_root = source_root / "samples" / f"sample-{sample}"
    manifest_path = sample_root / "sample_manifest.json"
    manifest = answer_contract.read_json(
        _regular_file(manifest_path, label="R207 sample manifest")
    )
    bank_path = sample_root / "canonical_entry_bank.json"
    traces_path = sample_root / "question_traces.jsonl"
    bank = answer_contract.read_json(_regular_file(bank_path, label="R207 bank"))
    traces = _read_jsonl(traces_path)
    if not isinstance(manifest, dict) or not isinstance(bank, dict):
        raise R207TaskEvalError("R207 sample binding input is invalid")
    condition_memories: dict[str, Any] = {}
    for condition in PATH_CONDITIONS:
        memory = sample_root / "conditions" / condition
        descriptor = readonly_control.memory_descriptor(memory)
        recorded = manifest.get("tree_hashes", {}).get("conditions", {}).get(
            condition
        )
        if not isinstance(recorded, dict):
            raise R207TaskEvalError("R207 condition tree binding is absent")
        condition_memories[condition] = {
            "path": relative_to_root(memory),
            "readonly_descriptor": descriptor,
            "r207_tree_descriptor": recorded,
        }
    return {
        "sample": sample,
        "sample_id": manifest.get("sample_id"),
        "sample_manifest": _file_binding(manifest_path),
        "canonical_bank": _file_binding(bank_path),
        "canonical_entries_sha256": bank.get("entries_sha256"),
        "question_traces": _file_binding(traces_path),
        "question_trace_count": len(traces),
        "condition_memories": condition_memories,
    }


def audit_r207_source(source_root: Path, *, formal: bool) -> dict[str, Any]:
    """Run the independent R207 auditor and bind all task-eval inputs."""

    source_root = source_root.expanduser().absolute()
    _regular_directory(source_root, label="R207 source root")
    report_first = r207_auditor.audit(source_root)
    if report_first.get("status") != "pass":
        raise R207TaskEvalError("independent R207 source audit did not pass")
    manifest_path = source_root / "run_manifest.json"
    manifest = answer_contract.read_json(manifest_path)
    expected_mode = "locomo" if formal else "synthetic_sanity"
    expected_samples = list(range(10)) if formal else [0]
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "complete"
        or manifest.get("config", {}).get("data", {}).get("mode")
        != expected_mode
        or manifest.get("requested_samples") != expected_samples
        or report_first.get("sample_count") != len(expected_samples)
    ):
        raise R207TaskEvalError("R207 source scope differs")
    if formal and (
        report_first.get("question_trace_count") != FORMAL_ARTIFACTS
        or manifest.get("config", {}).get("model_config", {}).get(
            "requested_model"
        )
        != MODEL
    ):
        raise R207TaskEvalError("formal R207 source is not all-ten GPT-5.5")
    if not formal and report_first.get("question_trace_count") != 0:
        raise R207TaskEvalError("synthetic R207 source trace inventory differs")
    samples = [_sample_binding(source_root, sample) for sample in expected_samples]
    report_second = r207_auditor.audit(source_root)
    if report_second != report_first:
        raise R207TaskEvalError("R207 source changed during independent audit")
    binding: dict[str, Any] = {
        "schema_version": SOURCE_BINDING_SCHEMA,
        "mode": "formal" if formal else "synthetic_no_network",
        "source_root": relative_to_root(source_root),
        "run_manifest": _file_binding(manifest_path),
        "run_fingerprint": manifest.get("run_fingerprint"),
        "independent_auditor": {
            "path": relative_to_root(Path(r207_auditor.__file__)),
            "sha256": answer_contract.sha256_file(Path(r207_auditor.__file__)),
            "status": "pass",
            "report_sha256": answer_contract.canonical_hash(report_first),
            "report": report_first,
        },
        "samples": samples,
        "sample_inventory_sha256": answer_contract.canonical_hash(samples),
        "questions_per_condition": PRIMARY_QUESTIONS if formal else 1,
        "question_condition_artifacts": FORMAL_ARTIFACTS if formal else 2,
    }
    binding["binding_sha256"] = answer_contract.canonical_hash(binding)
    return binding


def validate_binding_hash(binding: Mapping[str, Any]) -> None:
    payload = dict(binding)
    expected = payload.pop("binding_sha256", None)
    if expected != answer_contract.canonical_hash(payload):
        raise R207TaskEvalError("R207 source binding content hash differs")


def run_preflight(
    preregistration: Path = DEFAULT_PREREGISTRATION,
    *,
    source_root: Path = DEFAULT_R207_ROOT,
    write_path: Path | None = DEFAULT_PREFLIGHT,
) -> dict[str, Any]:
    prereg = validate_preregistration(preregistration)
    r002 = r116_contract.audit_relevant_evidence_mapping()
    source: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        source = audit_r207_source(source_root, formal=True)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")
    report = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "ready" if source is not None else "blocked",
        "preregistration": preregistration_binding(preregistration),
        "r002_subset_audit": r002,
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
        if write_path.exists():
            actual = answer_contract.read_json(write_path)
            if actual != report:
                answer_contract.atomic_json_replace(write_path, report)
        else:
            answer_contract.atomic_json_no_clobber(write_path, report)
    del prereg
    return report


def require_ready_binding(preflight: Mapping[str, Any]) -> dict[str, Any]:
    source = preflight.get("formal_source")
    if preflight.get("status") != "ready" or not isinstance(source, dict):
        raise R207TaskEvalError(
            "formal R207 task evaluation requires an independently audited "
            "all-ten R207 source"
        )
    validate_binding_hash(source)
    return source


def _sample_record(binding: Mapping[str, Any], sample: int) -> dict[str, Any]:
    records = binding.get("samples")
    if not isinstance(records, list):
        raise R207TaskEvalError("source sample inventory is absent")
    matches = [record for record in records if record.get("sample") == sample]
    if len(matches) != 1:
        raise R207TaskEvalError(f"source sample binding differs: {sample}")
    return matches[0]


def load_sample_context(
    source_root: Path,
    source_binding: Mapping[str, Any],
    *,
    sample: int,
    formal: bool,
) -> dict[str, Any]:
    validate_binding_hash(source_binding)
    record = _sample_record(source_binding, sample)
    sample_root = source_root / "samples" / f"sample-{sample}"
    bank = answer_contract.read_json(sample_root / "canonical_entry_bank.json")
    if (
        not isinstance(bank, dict)
        or answer_contract.sha256_file(sample_root / "canonical_entry_bank.json")
        != record["canonical_bank"]["sha256"]
        or bank.get("entries_sha256") != record["canonical_entries_sha256"]
    ):
        raise R207TaskEvalError("canonical entry bank differs from binding")
    traces: list[dict[str, Any]]
    if formal:
        traces = _read_jsonl(sample_root / "question_traces.jsonl")
        if (
            len(traces) != record["question_trace_count"]
            or answer_contract.sha256_file(sample_root / "question_traces.jsonl")
            != record["question_traces"]["sha256"]
        ):
            raise R207TaskEvalError("R207 question traces differ from binding")
    else:
        traces = []
    memories: dict[str, Path] = {}
    for condition in PATH_CONDITIONS:
        memory = sample_root / "conditions" / condition
        if readonly_control.memory_descriptor(memory) != record[
            "condition_memories"
        ][condition]["readonly_descriptor"]:
            raise R207TaskEvalError("R207 condition memory differs from binding")
        memories[condition] = memory
    return {
        "sample_root": sample_root,
        "bank": bank,
        "traces": traces,
        "memories": memories,
        "binding": record,
    }


def _source_locations(memory: Path) -> dict[str, list[str]]:
    locations: dict[str, list[str]] = {}
    for path in sorted(memory.rglob("*.md")):
        relative = path.relative_to(memory).as_posix()
        for source_id in re.findall(
            r"(?<![\w])D\d+:\d+(?!\w)", path.read_text(encoding="utf-8")
        ):
            locations.setdefault(source_id, []).append(relative)
    return locations


def synthetic_upstream_trace(
    *,
    spec: Mapping[str, Any],
    condition: str,
    bank: Mapping[str, Any],
    memory: Path,
) -> dict[str, Any]:
    gold = list(spec["gold_source_ids"])
    canonical = {
        str(source)
        for entry in bank.get("entries", [])
        for source in entry.get("dia_ids", [])
    }
    locations = _source_locations(memory)
    canonical_ok = all(value in canonical for value in gold)
    paths_ok = all(locations.get(value) for value in gold)
    stage = {
        "mapping_complete": {
            "status": "observed",
            "value": True,
            "artifact": "synthetic_task_mapping",
            "question_id": spec["question_id"],
            "source_ids": gold,
            "source_exclusion_reasons": [],
        },
        "canonical_entry_source_exists": {
            "status": "observed",
            "value": canonical_ok,
            "artifact": "canonical_entry_bank.json",
            "present_source_ids": [value for value in gold if value in canonical],
            "missing_source_ids": [value for value in gold if value not in canonical],
        },
        "maintenance_survival": {
            "status": "observed",
            "value": paths_ok,
            "artifact": "synthetic_path_materialization",
            "present_source_ids": [value for value in gold if locations.get(value)],
            "missing_source_ids": [value for value in gold if not locations.get(value)],
        },
        "path_validity": {
            "status": "observed",
            "value": paths_ok,
            "artifact": f"conditions/{condition}",
            "source_paths": {value: locations.get(value, []) for value in gold},
        },
        "retrieval_reach": {
            "status": "not_observed",
            "value": None,
            "reason": "retrieval_results_not_attached",
        },
        "source_resolution": {
            "status": "not_observed",
            "value": None,
            "reason": "retrieval_results_not_attached",
        },
    }
    stage["trace_ids"] = {
        "mapping": [
            f"gold-mapping:{spec['question_id']}:"
            f"{answer_contract.canonical_hash({'source_ids': gold, 'mapping_complete': True})}"
        ],
        "canonical_entries": [
            f"canonical-sources:{answer_contract.canonical_hash({'question_id': spec['question_id'], 'present_source_ids': gold})}"
        ],
        "maintenance": [
            f"maintenance-sources:{answer_contract.canonical_hash({'question_id': spec['question_id'], 'present_source_ids': gold, 'missing_source_ids': []})}"
        ],
        "paths": [
            f"condition-path:{answer_contract.canonical_hash({'condition': condition, 'source_id': source, 'path': path})}"
            for source in gold
            for path in locations.get(source, [])
        ],
        "retrieval": [],
        "source_resolution": [],
    }
    stage["m4_fields"] = {
        "gold_source_mapping_complete": True,
        "gold_source_in_canonical_entries": canonical_ok,
        "gold_source_survived_maintenance": paths_ok,
        "gold_source_path_valid": paths_ok,
        "retrieval_reached_gold_source": None,
        "source_resolution_returned_gold_content": None,
    }
    stage["evidence_status"] = "partial_pre_retrieval"
    return {
        "schema": "nativemem.r207-question-trace.v1",
        "trace_id": f"r207:{spec['question_id']}:{condition}",
        "sample": 0,
        "sample_id": "synthetic-sample-0",
        "question_index": 0,
        "question_id": spec["question_id"],
        "condition": condition,
        "normalized_source_ids": gold,
        "source_recall_eligible": True,
        "stage_evidence": stage,
    }


def trace_for_question(
    *,
    context: Mapping[str, Any],
    spec: Mapping[str, Any],
    condition: str,
    formal: bool,
) -> dict[str, Any]:
    if condition not in PATH_CONDITIONS:
        raise R207TaskEvalError("unknown R207 path condition")
    if formal:
        matches = [
            trace
            for trace in context["traces"]
            if trace.get("question_id") == spec["question_id"]
            and trace.get("condition") == condition
        ]
        if len(matches) != 1:
            raise R207TaskEvalError("R207 question trace lookup differs")
        trace = matches[0]
    else:
        trace = synthetic_upstream_trace(
            spec=spec,
            condition=condition,
            bank=context["bank"],
            memory=context["memories"][condition],
        )
    if (
        trace.get("normalized_source_ids") != spec["gold_source_ids"]
        or trace.get("source_recall_eligible")
        is not bool(spec["source_recall_eligible"])
        or trace.get("question_index") != int(spec["question_index"])
    ):
        raise R207TaskEvalError("R207 trace and R002 mapping differ")
    return trace


def build_stage_provenance(
    *,
    spec: Mapping[str, Any],
    bank: Mapping[str, Any],
    memory: Path,
    upstream_trace: Mapping[str, Any],
) -> dict[str, Any]:
    gold = list(dict.fromkeys(str(value) for value in spec["gold_source_ids"]))
    mapping_payload = {
        "question_id": spec["question_id"],
        "mapping_complete": bool(spec["source_recall_eligible"]),
        "gold_source_ids": gold,
    }
    mapping_hash = answer_contract.canonical_hash(mapping_payload)
    entries: list[dict[str, Any]] = []
    for raw in bank.get("entries", []):
        entry_id = str(raw.get("entry_id", ""))
        if re.fullmatch(r"[A-Za-z0-9_-]+", entry_id) is None:
            raise R207TaskEvalError("canonical entry ID is unsafe")
        source_ids = list(dict.fromkeys(str(value) for value in raw.get("dia_ids", [])))
        payload = {
            "path": f"topics/canonical-entry-bank/{entry_id}.md",
            "content_sha256": answer_contract.sha256_bytes(
                (str(raw.get("summary_inline", "")) + "\n").encode("utf-8")
            ),
            "source_ids": source_ids,
        }
        entries.append(
            {
                **payload,
                "trace_id": f"memory-entry:{answer_contract.canonical_hash(payload)}",
            }
        )
    canonical_sha = str(bank.get("entries_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", canonical_sha) is None:
        raise R207TaskEvalError("canonical bank hash is invalid")
    output_sha = readonly_control.memory_descriptor(memory)["sha256"]
    return {
        "schema_version": readonly_control.STAGE_PROVENANCE_SCHEMA,
        "question_id": spec["question_id"],
        "mapping": {
            **mapping_payload,
            "record_sha256": mapping_hash,
            "trace_ids": [
                f"gold-mapping:{spec['question_id']}:{mapping_hash}"
            ],
        },
        "canonical_entries_before_maintenance": {
            "memory_sha256": canonical_sha,
            "entries": entries,
            "trace_ids": [
                f"canonical-memory:{canonical_sha}",
                *(entry["trace_id"] for entry in entries),
            ],
        },
        "maintenance": {
            "operation_count": 1,
            "input_memory_sha256": canonical_sha,
            "output_memory_sha256": output_sha,
            "trace_ids": [
                f"r207-materialization:{upstream_trace['trace_id']}:{output_sha}"
            ],
        },
    }


def input_binding(
    *,
    preregistration: Path,
    source_binding: Mapping[str, Any],
    spec: Mapping[str, Any],
    condition: str,
    memory: Path,
    upstream_trace: Mapping[str, Any],
    stage_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": INPUT_BINDING_SCHEMA,
        "preregistration": preregistration_binding(preregistration),
        "source_binding_sha256": source_binding["binding_sha256"],
        "path_condition": condition,
        "generic_readonly_condition": GENERIC_CONDITION,
        "question": {
            "question_id": spec["question_id"],
            "artifact_id": spec["artifact_id"],
            "dataset_index": spec["dataset_index"],
            "question_index": spec["question_index"],
            "question_sha256": answer_contract.sha256_bytes(
                str(spec["question"]).encode("utf-8")
            ),
            "category": spec["category"],
            "gold_answer_sha256": spec["gold_answer_sha256"],
            "gold_source_ids": spec["gold_source_ids"],
            "source_recall_eligible": spec["source_recall_eligible"],
            "primary_scoring_eligible": True,
        },
        "memory": readonly_control.memory_descriptor(memory),
        "upstream_trace_sha256": answer_contract.canonical_hash(upstream_trace),
        "stage_provenance_sha256": answer_contract.canonical_hash(
            stage_provenance
        ),
        "protocol": {
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


def build_m4_trace(
    *,
    condition: str,
    upstream_trace: Mapping[str, Any],
    observed_stage: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        key: observed_stage.get(key)
        for key in (
            "gold_source_mapping_complete",
            "gold_source_in_canonical_entries",
            "gold_source_survived_maintenance",
            "gold_source_path_valid",
            "retrieval_reached_gold_source",
            "source_resolution_returned_gold_content",
        )
    }
    return {
        "schema_version": M4_TRACE_SCHEMA,
        "question_id": upstream_trace["question_id"],
        "path_condition": condition,
        "upstream_trace_id": upstream_trace["trace_id"],
        "upstream_trace_sha256": answer_contract.canonical_hash(upstream_trace),
        "pre_retrieval_stage_evidence": upstream_trace["stage_evidence"],
        "observed_stage_evidence": observed_stage,
        "m4_fields": fields,
        "trace_ids": observed_stage.get("trace_ids", {}),
        "evidence_status": observed_stage.get("evidence_status"),
    }


def official_task_score(*, answer: str, gold_answer: str, category: int) -> float:
    if category not in CATEGORY_COUNTS:
        raise R207TaskEvalError("R207 primary score received a non-primary category")
    score = float(f1_locomo_official(answer, gold_answer, category))
    if not 0.0 <= score <= 1.0:
        raise R207TaskEvalError("official LoCoMo score is outside [0,1]")
    return score


def build_question_metrics(
    *,
    attempt: Path,
    spec: Mapping[str, Any],
    condition: str,
    result: Mapping[str, Any],
    m4_trace: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostics = result["diagnostics"]
    budget = result["budget"]
    memory = result["memory"]
    artifacts = result["artifacts"]
    retrieval_records = read_ledger(attempt / "retrieval_model_ledger.jsonl")
    retrieval_event_ids: list[str] = []
    retrieval_response_ids = []
    for row in retrieval_records:
        if row.get("event") != "model_call_finished":
            continue
        retrieval_response_ids.append(str(row.get("response_id", "")))
        evidence = row.get("proxy_evidence")
        if isinstance(evidence, Mapping):
            for event in evidence.get("events", []):
                if isinstance(event, Mapping):
                    retrieval_event_ids.append(str(event.get("event_id", "")))
    answer = result["answer"]
    score = official_task_score(
        answer=str(answer["text"]),
        gold_answer=str(spec["gold_answer"]),
        category=int(spec["category"]),
    )
    return {
        "schema_version": METRICS_SCHEMA,
        "question_id": spec["question_id"],
        "artifact_id": spec["artifact_id"],
        "path_condition": condition,
        "category": spec["category"],
        "primary_scoring_eligible": True,
        "category_5_mixed_into_primary": False,
        "official_locomo_f1": score,
        "first_relevant_file": diagnostics["first_relevant_file"],
        "mapped_source_recall": diagnostics["mapped_source_recall"],
        "source_recall_eligible": diagnostics["source_recall_eligible"],
        "navigation_calls": {
            "read_calls": diagnostics["read_calls"],
            "tool_calls": diagnostics["tool_calls"],
            "retrieval_model_calls": result["retrieval"]["model_calls"],
        },
        "navigation_tokens": {
            "visible_tokens": budget["visible_tokens"],
            "source_resolution_tokens": budget["source_resolution_tokens"],
            "configured_hard_cap_tokens": budget["configured_tokens"],
        },
        "memory_tree": {
            "before_sha256": memory["before"]["sha256"],
            "after_sha256": memory["after"]["sha256"],
            "unchanged": memory["unchanged"],
        },
        "r004": {
            "visible_token_trace": artifacts["visible_token_trace"],
            "visible_token_trace_sha256": artifacts[
                "visible_token_trace_sha256"
            ],
            "visible_token_manifest": artifacts["visible_token_manifest"],
            "visible_token_manifest_sha256": artifacts[
                "visible_token_manifest_sha256"
            ],
        },
        "m4": {
            "trace_sha256": answer_contract.canonical_hash(m4_trace),
            "fields": m4_trace["m4_fields"],
            "trace_ids": m4_trace["trace_ids"],
            "evidence_status": m4_trace["evidence_status"],
        },
        "durable_ledgers": {
            "retrieval": {
                "path": artifacts["retrieval_model_ledger"],
                "sha256": artifacts["retrieval_model_ledger_sha256"],
                "state": result["retrieval"]["ledger_state"],
            },
            "answer": {
                "path": artifacts["answer_ledger"],
                "sha256": artifacts["answer_ledger_sha256"],
            },
        },
        "exclusive_proxy_evidence": {
            "retrieval_event_ids": retrieval_event_ids,
            "answer_event_ids": answer["exclusive_proxy_event_ids"],
            "retrieval_response_ids": retrieval_response_ids,
            "answer_response_id": answer["response_id"],
            "answer_response_model": answer["response_model"],
        },
    }
def verify_source_unchanged(
    *, source_root: Path, source_binding: Mapping[str, Any], formal: bool
) -> None:
    live = audit_r207_source(source_root, formal=formal)
    if live != source_binding:
        raise R207TaskEvalError("R207 source changed after task-eval binding")


__all__ = [
    "ANSWER_COMPLETION_RESERVATION_TOKENS",
    "ANSWER_MAX_TOKENS",
    "ANSWER_RETRIES",
    "BUDGET_TOKENS",
    "CATEGORY_5_SEPARATE",
    "DEFAULT_PREFLIGHT",
    "DEFAULT_PREREGISTRATION",
    "DEFAULT_R207_ROOT",
    "DEFAULT_SYNTHETIC_SOURCE",
    "FORMAL_ARTIFACTS",
    "GENERIC_CONDITION",
    "INPUT_BINDING_SCHEMA",
    "M4_TRACE_SCHEMA",
    "MAX_ROUNDS",
    "METHOD",
    "METRICS_SCHEMA",
    "MODEL",
    "MODEL_CONTEXT_LIMIT_TOKENS",
    "PATH_CONDITIONS",
    "PRIMARY_QUESTIONS",
    "PRIMARY_SOURCE_RECALL",
    "R207TaskEvalError",
    "audit_r207_source",
    "build_m4_trace",
    "build_preregistration",
    "build_question_metrics",
    "build_stage_provenance",
    "conversation_for_spec",
    "freeze_preregistration",
    "input_binding",
    "load_sample_context",
    "official_task_score",
    "preregistration_binding",
    "primary_question_specs",
    "relative_to_root",
    "require_ready_binding",
    "run_preflight",
    "specs_by_sample",
    "synthetic_spec",
    "trace_for_question",
    "validate_binding_hash",
    "validate_preregistration",
    "verify_source_unchanged",
]
