#!/usr/bin/env python3
"""Frozen local contract for the formal R203 LoCoMo read-only controls.

This module performs no model or network request.  It rebuilds the registered
LoCoMo question inventory from the released dataset and R002 mapping, verifies
the independently audited all-ten NativeMem source, and records the immutable
input bindings consumed by the runner and independent auditor.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import controlled_locomo_answer_contract as answer_contract  # noqa: E402
import r116_formal_contract as r116_contract  # noqa: E402
import readonly_nativemem_control as readonly_control  # noqa: E402


PREREG_SCHEMA = "r203-preregistration-v2"
PREFLIGHT_SCHEMA = "nativemem.r203-formal-preflight.v2"
SOURCE_BINDING_SCHEMA = "nativemem.r203-source-binding.v1"
INPUT_BINDING_SCHEMA = "nativemem.r203-question-input-binding.v1"
METHOD = "r203_readonly_views"
MODEL = "gpt-5.5"
SAMPLES = list(range(10))
CONDITIONS = readonly_control.CONDITIONS
PRIMARY_QUESTIONS = 1_540
TOTAL_ARTIFACTS = 6_160
SOURCE_RECALL_DENOMINATOR = 1_533
CATEGORY_COUNTS = {1: 282, 2: 321, 3: 96, 4: 841}
BUDGET_TOKENS = 20_000
MAX_ROUNDS = 12
MODEL_CONTEXT_LIMIT_TOKENS = 128_000
ANSWER_COMPLETION_RESERVATION_TOKENS = 4_096
ANSWER_MAX_TOKENS = 4_096
ANSWER_RETRIES = 3

DEFAULT_PREREGISTRATION = ROOT / "paper/refine-logs/R203_PREREGISTRATION.json"
DEFAULT_PREFLIGHT = (
    ROOT / "results/paper-experiments-20260714/r203-formal-preflight.json"
)
DEFAULT_SOURCE_ROOT = (
    ROOT / "results/gpt55-benchmarks-20260714/locomo-v88-calendar-gpt55"
)
SUPERSEDED_V1_PREREGISTRATION = (
    ROOT
    / "results/paper-experiments-20260714/code-integrity-superseded"
    / "R203_PREREGISTRATION.v1.json"
)
DATASET = ROOT / "benchmarks/locomo/data/locomo10.json"
EVIDENCE_DIR = ROOT / "results/paper-experiments-20260714/evidence-mapping/v1"
EVIDENCE_MANIFEST = EVIDENCE_DIR / "evidence_mapping.v1.manifest.json"
EVIDENCE_QUESTIONS = EVIDENCE_DIR / "evidence_mapping.v1.questions.jsonl"
EVIDENCE_AUDIT = EVIDENCE_DIR / "evidence_mapping.v1.audit.json"

EXPECTED_DATASET_SHA256 = (
    "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
)
EXPECTED_MAPPING_MANIFEST_SHA256 = (
    "64fa89b05bc2ad075ad046fb514b1cded164cba211f5b1d246719ac67916b735"
)
EXPECTED_MAPPING_QUESTIONS_SHA256 = (
    "3291df579bc20d0678f99b3361ac6d287beee583e4d3dff93b996f4522e56318"
)
V1_PREREGISTRATION_FILE_SHA256 = (
    "fe71bd31beb58c1fed5b82d6990571e223482e8b8855973eb28bdd166a0f7c1d"
)

SOURCE_FILES = (
    "scripts/r203_formal_contract.py",
    "scripts/run_r203_readonly_views.py",
    "scripts/audit_r203_readonly_views.py",
    "scripts/readonly_nativemem_control.py",
    "scripts/audit_readonly_nativemem_control.py",
    "scripts/controlled_locomo_answer_contract.py",
    "scripts/run_controlled_locomo_answers.py",
    "scripts/controlled_gpt55_run_proxy.py",
    "scripts/gpt55_run_proxy.py",
    "scripts/audit_benchmark_gold_sources.py",
    "scripts/r116_formal_contract.py",
    "scripts/run_v88_gpt55_locomo.py",
    "scripts/audit_v88_gpt55_locomo.py",
    "src/openai_gpt55_flex_gateway.py",
    "src/openai_gpt55_flex_gateway_evidence.py",
    "src/evaluation/durable_model_ledger.py",
    "src/evaluation/visible_token_budget.py",
    "src/evaluation/visible_token_audit.py",
    "src/evaluation/prompts.py",
)


class R203FormalError(RuntimeError):
    """A formal R203 input or immutable binding violates the contract."""


def relative_to_root(path: Path) -> str:
    return path.expanduser().absolute().relative_to(ROOT).as_posix()


def _regular_file(path: Path, *, label: str) -> Path:
    answer_contract.reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise R203FormalError(f"{label} is not a regular file: {path}")
    return path


def _regular_directory(path: Path, *, label: str) -> Path:
    answer_contract.reject_symlink_components(path)
    if path.is_symlink() or not path.is_dir():
        raise R203FormalError(f"{label} is not a regular directory: {path}")
    return path


def file_binding(path: Path) -> dict[str, Any]:
    checked = _regular_file(path, label="bound input")
    return {
        "path": relative_to_root(checked),
        "bytes": checked.stat().st_size,
        "sha256": answer_contract.sha256_file(checked),
    }


def source_hashes() -> dict[str, str]:
    return {
        relative: answer_contract.sha256_file(
            _regular_file(ROOT / relative, label="bound source")
        )
        for relative in SOURCE_FILES
    }


def expected_preregistration() -> dict[str, Any]:
    """Return the v2 integrity-only revision of the registered design."""

    return {
        "schema_version": PREREG_SCHEMA,
        "status": "frozen_before_formal_model_requests",
        "frozen_date": "2026-07-14",
        "benchmark": "LoCoMo",
        "scope": {
            "samples": SAMPLES,
            "primary_questions": PRIMARY_QUESTIONS,
            "category_5": "excluded_from_R203_primary_analysis",
            "source_mapping_manifest_sha256": EXPECTED_MAPPING_MANIFEST_SHA256,
            "source_mapping_questions_sha256": EXPECTED_MAPPING_QUESTIONS_SHA256,
        },
        "frozen_method": {
            "memory_source": (
                "completed_and_independently_audited_"
                "NativeMem_v8.8_plus_calendar_all_ten"
            ),
            "condition_initialization": (
                "byte_identical_copy_per_condition_before_questions"
            ),
            "retrieval_model": MODEL,
            "answer_model": MODEL,
            "max_rounds": MAX_ROUNDS,
            "visible_content_budget_tokens": BUDGET_TOKENS,
            "budget_policy": (
                "hard_cap_shared_by_tool_and_source_resolution_content"
            ),
            "tokenizer": "tiktoken==0.12.0:o200k_base_named_fallback",
            "retrieval_read_only": True,
        },
        "conditions": [
            {
                "condition_id": "dual_source",
                "views": ["topics", "timeline"],
                "source_resolver": True,
            },
            {
                "condition_id": "topic_source",
                "views": ["topics"],
                "source_resolver": True,
            },
            {
                "condition_id": "timeline_source",
                "views": ["timeline"],
                "source_resolver": True,
            },
            {
                "condition_id": "dual_no_source",
                "views": ["topics", "timeline"],
                "source_resolver": False,
            },
        ],
        "primary_comparisons": [
            {
                "comparison_id": "locomo_dual_vs_topic",
                "left": "dual_source",
                "right": "topic_source",
            },
            {
                "comparison_id": "locomo_dual_vs_timeline",
                "left": "dual_source",
                "right": "timeline_source",
            },
            {
                "comparison_id": "locomo_dual_source_vs_dual_no_source",
                "left": "dual_source",
                "right": "dual_no_source",
            },
        ],
        "primary_outcome": "primary_gpt4o_mini_judge_correct",
        "diagnostics": [
            "first_relevant_file",
            "mapped_source_recall_on_complete_mappings",
            "navigation_calls",
            "visible_tokens",
            "source_resolution_tokens",
            "latency_seconds",
            "before_after_memory_sha256",
        ],
        "support_rule": (
            "A component supports C2 only when its paired task change is "
            "accompanied by the preregistered source or navigation diagnostic "
            "in the predicted direction."
        ),
        "integrity": {
            "formal_artifact_binding": (
                "bind completed memory audit, dataset, mapping, code, prompt, "
                "proxy, answerer, and judge hashes before the first request"
            ),
            "missingness": (
                "no imputation; missing or failed questions invalidate the "
                "formal comparison"
            ),
            "source_recall_denominator": SOURCE_RECALL_DENOMINATOR,
            "dual_no_source_resolution": (
                "resolver absent from schema; source-resolution outcome remains "
                "an explicit registered mechanism condition"
            ),
            "stage_provenance_interpretation": {
                "canonical_input": (
                    "the independently audited final NativeMem tree is frozen "
                    "as the canonical input at the R203 boundary"
                ),
                "r203_maintenance_operation_count": 0,
                "claim_boundary": (
                    "zero maintenance is asserted only inside each R203 "
                    "read-only condition; it does not establish NativeMem "
                    "builder pre-maintenance provenance"
                ),
            },
        },
        "integrity_revision": {
            "revision_from": "r203-preregistration-v1",
            "previous_file_path": relative_to_root(
                SUPERSEDED_V1_PREREGISTRATION
            ),
            "previous_file_sha256": V1_PREREGISTRATION_FILE_SHA256,
            "changed_fields": [
                "schema_version",
                "integrity.stage_provenance_interpretation",
                "integrity_revision",
            ],
            "experimental_design_changes": [],
            "reason": (
                "clarify the provenance claim boundary before formal requests"
            ),
        },
    }


def validate_preregistration(
    path: Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    actual = answer_contract.read_json(_regular_file(path, label="preregistration"))
    expected = expected_preregistration()
    if actual != expected:
        raise R203FormalError("R203 v2 preregistration differs")
    return actual


def preregistration_binding(
    path: Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    prereg = validate_preregistration(path)
    superseded = file_binding(SUPERSEDED_V1_PREREGISTRATION)
    if superseded["sha256"] != V1_PREREGISTRATION_FILE_SHA256:
        raise R203FormalError("superseded R203 v1 preregistration hash differs")
    return {
        "path": relative_to_root(path),
        "sha256": answer_contract.sha256_file(path),
        "content_sha256": answer_contract.canonical_hash(prereg),
        "superseded_v1": superseded,
    }


def question_specs() -> list[dict[str, Any]]:
    """Rebuild the registered 1,540 category-1--4 questions from R002."""

    specs = [
        dict(spec)
        for spec in r116_contract.question_specs(r116_contract.LOCOMO)
        if int(spec["category"]) in {1, 2, 3, 4}
    ]
    categories = Counter(int(spec["category"]) for spec in specs)
    if len(specs) != PRIMARY_QUESTIONS or dict(sorted(categories.items())) != CATEGORY_COUNTS:
        raise R203FormalError("R203 primary question inventory differs")
    if sum(bool(spec["source_recall_eligible"]) for spec in specs) != SOURCE_RECALL_DENOMINATOR:
        raise R203FormalError("R203 source-recall denominator differs")
    if len({str(spec["artifact_id"]) for spec in specs}) != len(specs):
        raise R203FormalError("R203 artifact identifiers are not unique")
    return specs


def question_inventory_sha256(specs: Sequence[Mapping[str, Any]]) -> str:
    return answer_contract.canonical_hash(
        [
            {
                "question_id": spec["question_id"],
                "artifact_id": spec["artifact_id"],
                "dataset_index": spec["dataset_index"],
                "question_index": spec["question_index"],
                "category": spec["category"],
                "question_sha256": answer_contract.sha256_bytes(
                    str(spec["question"]).encode("utf-8")
                ),
                "evidence_record_sha256": spec["evidence_record_sha256"],
                "source_recall_eligible": spec["source_recall_eligible"],
            }
            for spec in specs
        ]
    )


def conversation_and_turn_index(
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    conversation = r116_contract.locomo_conversation(spec)
    return conversation, readonly_control.build_turn_index(conversation)


def _source_preliminary_blockers(source_root: Path) -> list[str]:
    """List concrete local blockers without invoking a model or network."""

    blockers: list[str] = []
    if source_root.is_symlink() or not source_root.is_dir():
        return [f"source root is absent or unsafe: {source_root}"]
    manifest_path = source_root / "run_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return [f"source run manifest is absent or unsafe: {manifest_path}"]
    try:
        manifest = answer_contract.read_json(manifest_path)
    except Exception as exc:  # noqa: BLE001
        return [f"source run manifest is unreadable: {type(exc).__name__}: {exc}"]
    if not isinstance(manifest, dict):
        return ["source run manifest is not an object"]
    if manifest.get("status") != "complete":
        blockers.append(
            "source run_manifest.status is "
            f"{manifest.get('status')!r}; required 'complete'"
        )
    if manifest.get("benchmark") != "locomo":
        blockers.append("source benchmark is not 'locomo'")
    if manifest.get("method") != "NativeMem-v8.8+calendar":
        blockers.append("source method is not 'NativeMem-v8.8+calendar'")
    if manifest.get("backbone") != MODEL:
        blockers.append(f"source backbone is not {MODEL!r}")
    config = manifest.get("config")
    if not isinstance(config, dict):
        blockers.append("source config is absent")
    else:
        if config.get("provider") != "openai_api_flex_via_exclusive_child_proxy":
            blockers.append(
                "source provider is not the registered OpenAI GPT-5.5 Flex "
                "exclusive-child-proxy path"
            )
        if config.get("gateway_root") in {None, ""}:
            blockers.append("source config has no dynamic Flex gateway_root binding")
        if config.get("samples") != SAMPLES:
            blockers.append("source sample scope is not exactly 0..9")
        if config.get("max_sessions") is not None:
            blockers.append("source max_sessions must be null")
        if config.get("questions_limit") is not None:
            blockers.append("source questions_limit must be null")
    provider = manifest.get("provider_evidence")
    if not isinstance(provider, dict):
        blockers.append("source Flex provider_evidence is absent")
    elif provider.get("active_run_id") is not None:
        blockers.append("source Flex provider_evidence has an active invocation")
    states = manifest.get("samples")
    if not isinstance(states, dict):
        blockers.append("source sample state inventory is absent")
        states = {}
    for sample in SAMPLES:
        state = states.get(str(sample))
        if not isinstance(state, dict) or state.get("status") != "complete":
            blockers.append(f"source sample {sample} is not complete")
        memory = source_root / f"memory_sample{sample}"
        if memory.is_symlink() or not memory.is_dir():
            blockers.append(f"source memory_sample{sample} is absent or unsafe")
        output = source_root / f"sample{sample}_questions.json"
        if output.is_symlink() or not output.is_file():
            blockers.append(f"source sample{sample}_questions.json is absent or unsafe")
    for name in ("audit.json", "questions_all.json"):
        path = source_root / name
        if path.is_symlink() or not path.is_file():
            blockers.append(f"source {name} is absent or unsafe")
    return blockers


def _mapping_binding() -> dict[str, Any]:
    live = r116_contract.audit_relevant_evidence_mapping()
    manifest = file_binding(EVIDENCE_MANIFEST)
    questions = file_binding(EVIDENCE_QUESTIONS)
    audit = file_binding(EVIDENCE_AUDIT)
    if manifest["sha256"] != EXPECTED_MAPPING_MANIFEST_SHA256:
        raise R203FormalError("R002 mapping manifest hash differs")
    if questions["sha256"] != EXPECTED_MAPPING_QUESTIONS_SHA256:
        raise R203FormalError("R002 mapping question hash differs")
    if live.get("locomo_source_recall_questions") != SOURCE_RECALL_DENOMINATOR:
        raise R203FormalError("R002 LoCoMo denominator differs")
    return {
        "manifest": manifest,
        "questions": questions,
        "audit": audit,
        "live_independent_audit": live,
    }


def _verified_source_binding(source_root: Path) -> dict[str, Any]:
    prereg_stub = {
        "benchmarks": {
            r116_contract.LOCOMO: {
                "source_root": relative_to_root(source_root),
            }
        }
    }
    upstream = r116_contract.source_preflight(
        r116_contract.LOCOMO, prereg_stub
    )
    inventory = upstream.get("inventory")
    if not isinstance(inventory, list) or len(inventory) != 10:
        raise R203FormalError("source tree inventory is not all-ten")
    if [entry.get("sample_index") for entry in inventory] != SAMPLES:
        raise R203FormalError("source tree inventory sample order differs")
    binding = {
        "schema_version": SOURCE_BINDING_SCHEMA,
        "source_kind": "NativeMem-v8.8+calendar-all-ten-GPT-5.5-Flex",
        "source_root": relative_to_root(source_root),
        "run_manifest": upstream["run_manifest"],
        "independent_audit": upstream["independent_audit"],
        "source_inventory": inventory,
        "source_inventory_sha256": upstream["inventory_sha256"],
        "raw_dataset": file_binding(DATASET),
        "r002": _mapping_binding(),
        "source_questions": upstream["questions"],
        "r203_primary_questions": PRIMARY_QUESTIONS,
        "r203_condition_artifacts": TOTAL_ARTIFACTS,
        "source_recall_denominator": SOURCE_RECALL_DENOMINATOR,
        "provenance_claim_boundary": {
            "canonical_input": (
                "independently_audited_final_NativeMem_tree_at_R203_entry"
            ),
            "r203_maintenance_operation_count": 0,
            "does_not_claim": "NativeMem_builder_pre_maintenance_provenance",
        },
    }
    if binding["raw_dataset"]["sha256"] != EXPECTED_DATASET_SHA256:
        raise R203FormalError("raw LoCoMo dataset hash differs")
    binding["binding_sha256"] = answer_contract.canonical_hash(binding)
    return binding


def run_preflight(
    preregistration: Path = DEFAULT_PREREGISTRATION,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    write_path: Path | None = DEFAULT_PREFLIGHT,
) -> dict[str, Any]:
    """Run a strictly local R203 readiness audit and persist its exact blockers."""

    validate_preregistration(preregistration)
    specs = question_specs()
    dataset = file_binding(DATASET)
    mapping = _mapping_binding()
    code = source_hashes()
    source_root = source_root.expanduser().absolute()
    blockers = _source_preliminary_blockers(source_root)
    source_record: dict[str, Any]
    if blockers:
        source_record = {
            "status": "blocked",
            "source_root": (
                relative_to_root(source_root)
                if source_root.is_relative_to(ROOT)
                else str(source_root)
            ),
            "blockers": blockers,
        }
    else:
        try:
            binding = _verified_source_binding(source_root)
        except Exception as exc:  # noqa: BLE001
            source_record = {
                "status": "blocked",
                "source_root": (
                    relative_to_root(source_root)
                    if source_root.is_relative_to(ROOT)
                    else str(source_root)
                ),
                "blockers": [
                    f"independent source audit failed: {type(exc).__name__}: {exc}"
                ],
            }
        else:
            source_record = {"status": "ready", "binding": binding}
    report = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "ready" if source_record["status"] == "ready" else "blocked",
        "preregistration": preregistration_binding(preregistration),
        "scope": {
            "samples": SAMPLES,
            "conditions": list(CONDITIONS),
            "primary_questions": len(specs),
            "condition_artifacts": len(specs) * len(CONDITIONS),
            "source_recall_denominator": sum(
                bool(spec["source_recall_eligible"]) for spec in specs
            ),
            "question_inventory_sha256": question_inventory_sha256(specs),
        },
        "raw_dataset": dataset,
        "r002": mapping,
        "source": source_record,
        "code_source_hashes": code,
        "model_requests": 0,
        "network_requests": 0,
    }
    report["preflight_content_sha256"] = answer_contract.canonical_hash(report)
    if write_path is not None:
        answer_contract.atomic_json_replace(write_path, report)
    return report


def validate_content_hash(payload: Mapping[str, Any], field: str) -> None:
    content = dict(payload)
    expected = content.pop(field, None)
    if expected != answer_contract.canonical_hash(content):
        raise R203FormalError(f"{field} differs")


def require_ready_binding(preflight: Mapping[str, Any]) -> dict[str, Any]:
    validate_content_hash(preflight, "preflight_content_sha256")
    source = preflight.get("source")
    if preflight.get("status") != "ready" or not isinstance(source, Mapping):
        blockers = source.get("blockers") if isinstance(source, Mapping) else None
        raise R203FormalError(f"R203 formal source is blocked: {blockers}")
    binding = source.get("binding")
    if not isinstance(binding, dict):
        raise R203FormalError("R203 ready source binding is absent")
    validate_content_hash(binding, "binding_sha256")
    return binding


def source_memory_path(source_binding: Mapping[str, Any], sample: int) -> Path:
    if sample not in SAMPLES:
        raise R203FormalError("R203 sample is outside 0..9")
    return ROOT / str(source_binding["source_root"]) / f"memory_sample{sample}"


def verify_source_unchanged(source_binding: Mapping[str, Any]) -> None:
    inventory = source_binding.get("source_inventory")
    if not isinstance(inventory, list) or len(inventory) != 10:
        raise R203FormalError("source inventory is absent")
    live: list[dict[str, Any]] = []
    source_root = ROOT / str(source_binding["source_root"])
    for sample in SAMPLES:
        live.append(
            {
                "sample_index": sample,
                "memory": readonly_control.memory_descriptor(
                    source_root / f"memory_sample{sample}"
                ),
                "sample_output": file_binding(
                    source_root / f"sample{sample}_questions.json"
                ),
            }
        )
    if (
        live != inventory
        or answer_contract.canonical_hash(live)
        != source_binding.get("source_inventory_sha256")
    ):
        raise R203FormalError("frozen all-ten NativeMem source changed")


def condition_memory_path(output_dir: Path, condition: str, sample: int) -> Path:
    if condition not in CONDITIONS or sample not in SAMPLES:
        raise R203FormalError("condition-memory identity differs")
    return output_dir / "conditions" / condition / f"memory_sample{sample}"


def copy_inventory(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in SAMPLES:
        for condition in CONDITIONS:
            memory = condition_memory_path(output_dir, condition, sample)
            rows.append(
                {
                    "sample_index": sample,
                    "condition": condition,
                    "path": memory.relative_to(output_dir).as_posix(),
                    "descriptor": readonly_control.memory_descriptor(memory),
                }
            )
    return rows


def validate_copy_inventory(
    *, output_dir: Path, source_binding: Mapping[str, Any], inventory: Sequence[Any]
) -> None:
    live = copy_inventory(output_dir)
    if list(inventory) != live:
        raise R203FormalError("R203 condition-copy inventory changed")
    by_sample = {
        int(entry["sample_index"]): entry["memory"]
        for entry in source_binding["source_inventory"]
    }
    for sample in SAMPLES:
        hashes = {
            row["descriptor"]["sha256"]
            for row in live
            if row["sample_index"] == sample
        }
        if hashes != {by_sample[sample]["sha256"]}:
            raise R203FormalError(
                f"sample {sample} condition copies are not byte-identical to source"
            )


def build_stage_provenance(
    *,
    memory_root: Path,
    spec: Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe the audited final tree as input to zero R203 maintenance."""

    provenance = readonly_control.build_zero_maintenance_stage_provenance(
        memory_root=memory_root,
        question_id=str(spec["question_id"]),
        gold_source_ids=list(spec["gold_source_ids"]),
        mapping_complete=bool(spec["source_recall_eligible"]),
    )
    provenance["r203_scope"] = {
        "canonical_input_semantics": (
            "independently_audited_final_NativeMem_tree_frozen_at_R203_entry"
        ),
        "source_binding_sha256": source_binding["binding_sha256"],
        "evidence_record_sha256": spec["evidence_record_sha256"],
        "maintenance_operation_count": 0,
        "claim_boundary": (
            "R203_read_only_zero_maintenance_only;_no_NativeMem_builder_"
            "pre_maintenance_claim"
        ),
    }
    return provenance


def input_binding(
    *,
    preregistration: Path,
    source_binding: Mapping[str, Any],
    copy_inventory_sha256: str,
    condition: str,
    spec: Mapping[str, Any],
    memory_root: Path,
    turn_index: Mapping[str, Mapping[str, str]],
    stage_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    source_entry = source_binding["source_inventory"][int(spec["dataset_index"])]
    binding = {
        "schema_version": INPUT_BINDING_SCHEMA,
        "condition": condition,
        "question_id": spec["question_id"],
        "artifact_id": spec["artifact_id"],
        "dataset_index": spec["dataset_index"],
        "question_index": spec["question_index"],
        "category": spec["category"],
        "question_sha256": answer_contract.sha256_bytes(
            str(spec["question"]).encode("utf-8")
        ),
        "preregistration": preregistration_binding(preregistration),
        "source_binding_sha256": source_binding["binding_sha256"],
        "source_entry": source_entry,
        "raw_dataset": source_binding["raw_dataset"],
        "r002": {
            "manifest_sha256": source_binding["r002"]["manifest"]["sha256"],
            "questions_sha256": source_binding["r002"]["questions"]["sha256"],
            "record": spec["evidence_record"],
            "record_sha256": spec["evidence_record_sha256"],
            "gold_source_ids": spec["gold_source_ids"],
            "source_recall_eligible": spec["source_recall_eligible"],
        },
        "condition_copy_inventory_sha256": copy_inventory_sha256,
        "condition_memory": readonly_control.memory_descriptor(memory_root),
        "turn_index_sha256": answer_contract.canonical_hash(turn_index),
        "stage_provenance_sha256": answer_contract.canonical_hash(
            stage_provenance
        ),
        "provenance_claim_boundary": source_binding["provenance_claim_boundary"],
    }
    binding["binding_sha256"] = answer_contract.canonical_hash(binding)
    return binding


__all__ = [
    "ANSWER_COMPLETION_RESERVATION_TOKENS",
    "ANSWER_MAX_TOKENS",
    "ANSWER_RETRIES",
    "BUDGET_TOKENS",
    "CATEGORY_COUNTS",
    "CONDITIONS",
    "DATASET",
    "DEFAULT_PREFLIGHT",
    "DEFAULT_PREREGISTRATION",
    "DEFAULT_SOURCE_ROOT",
    "INPUT_BINDING_SCHEMA",
    "MAX_ROUNDS",
    "METHOD",
    "MODEL",
    "MODEL_CONTEXT_LIMIT_TOKENS",
    "PREFLIGHT_SCHEMA",
    "PREREG_SCHEMA",
    "PRIMARY_QUESTIONS",
    "R203FormalError",
    "SAMPLES",
    "SOURCE_BINDING_SCHEMA",
    "SOURCE_RECALL_DENOMINATOR",
    "SUPERSEDED_V1_PREREGISTRATION",
    "TOTAL_ARTIFACTS",
    "build_stage_provenance",
    "condition_memory_path",
    "conversation_and_turn_index",
    "copy_inventory",
    "expected_preregistration",
    "file_binding",
    "input_binding",
    "preregistration_binding",
    "question_inventory_sha256",
    "question_specs",
    "relative_to_root",
    "require_ready_binding",
    "run_preflight",
    "source_hashes",
    "source_memory_path",
    "validate_content_hash",
    "validate_copy_inventory",
    "validate_preregistration",
    "verify_source_unchanged",
]
