#!/usr/bin/env python3
"""Frozen input and output contract for formal R116 shared-answer runs.

The contract is deliberately separate from the original R116 synthetic gate.
It consumes only complete, independently audited NativeMem artifacts and never
imports or mutates the active NativeMem implementation.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date as date_type
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_benchmark_gold_sources as mapping_auditor  # noqa: E402
import audit_v88_gpt55_locomo as locomo_auditor  # noqa: E402
import audit_v88_gpt55_longmemeval as lme_auditor  # noqa: E402
import controlled_locomo_answer_contract as answer_contract  # noqa: E402
import readonly_nativemem_control as readonly_control  # noqa: E402


PREREG_SCHEMA = "nativemem.r116-formal-preregistration.v1"
PREFLIGHT_SCHEMA = "nativemem.r116-formal-preflight.v1"
INPUT_BINDING_SCHEMA = "nativemem.r116-question-input-binding.v1"
SOURCE_RECALL_SCHEMA = "nativemem.r116-source-recall.v1"
METHOD = "nativemem_shared_answer"
CONDITION = "dual_source"
MODEL = "gpt-5.5"
BUDGET_TOKENS = 20_000
MAX_ROUNDS = 12
MODEL_CONTEXT_LIMIT_TOKENS = 128_000
ANSWER_COMPLETION_RESERVATION_TOKENS = 4_096
ANSWER_MAX_TOKENS = 4_096
ANSWER_RETRIES = 3

LOCOMO = "locomo"
LONGMEMEVAL = "longmemeval-s"
BENCHMARKS = (LOCOMO, LONGMEMEVAL)

DEFAULT_PREREGISTRATION = (
    ROOT / "paper/refine-logs/R116_SHARED_ANSWER_PREREGISTRATION.json"
)
DEFAULT_PREFLIGHT = (
    ROOT / "results/paper-experiments-20260714/r116-formal-preflight.json"
)
EVIDENCE_DIR = (
    ROOT / "results/paper-experiments-20260714/evidence-mapping/v1"
)
EVIDENCE_MANIFEST = EVIDENCE_DIR / mapping_auditor.MANIFEST_NAME
EVIDENCE_QUESTIONS = EVIDENCE_DIR / mapping_auditor.QUESTIONS_NAME
EVIDENCE_AUDIT = EVIDENCE_DIR / mapping_auditor.AUDIT_NAME
LOCOMO_DATA = ROOT / "benchmarks/locomo/data/locomo10.json"
LME_DATA = ROOT / "benchmarks/longmemeval/data/longmemeval_s_cleaned.json"
LOCOMO_SOURCE = (
    ROOT / "results/gpt55-benchmarks-20260714/locomo-v88-calendar-gpt55"
)
LME_SOURCE = (
    ROOT
    / "results/gpt55-benchmarks-20260714/longmemeval-s-v88-calendar-gpt55"
)

EXPECTED_SCOPE = {
    LOCOMO: {
        "benchmark_label": "LoCoMo",
        "questions": 1_986,
        "primary_cat1_4": 1_540,
        "category_5": 446,
        "category_counts": {"1": 282, "2": 321, "3": 96, "4": 841, "5": 446},
        "source_recall_questions": 1_533,
        "source_granularity": "turn",
        "samples": list(range(10)),
    },
    LONGMEMEVAL: {
        "benchmark_label": "LongMemEval-S",
        "questions": 500,
        "abstention_questions": 30,
        "question_type_counts": {
            "knowledge-update": 78,
            "multi-session": 133,
            "single-session-assistant": 56,
            "single-session-preference": 30,
            "single-session-user": 70,
            "temporal-reasoning": 133,
        },
        "source_recall_questions": 500,
        "source_granularity": "session",
        "items": 500,
    },
}

SOURCE_FILES = (
    "scripts/r116_formal_contract.py",
    "scripts/run_r116_formal.py",
    "scripts/audit_r116_formal.py",
    "scripts/readonly_nativemem_control.py",
    "scripts/audit_readonly_nativemem_control.py",
    "scripts/controlled_locomo_answer_contract.py",
    "scripts/run_controlled_locomo_answers.py",
    "scripts/controlled_gpt55_run_proxy.py",
    "scripts/gpt55_run_proxy.py",
    "scripts/audit_benchmark_gold_sources.py",
    "scripts/audit_v88_gpt55_locomo.py",
    "scripts/audit_v88_gpt55_longmemeval.py",
    "scripts/run_v88_gpt55_locomo.py",
    "scripts/run_v88_gpt55_longmemeval.py",
    "src/nativemem.py",
    "src/v8_memory.py",
    "src/adapters/run_nativemem.py",
    "src/openai_gpt55_flex_gateway.py",
    "src/openai_gpt55_flex_gateway_evidence.py",
    "src/evaluation/durable_model_ledger.py",
    "src/evaluation/visible_token_budget.py",
    "src/evaluation/visible_token_audit.py",
    "src/evaluation/prompts.py",
)


class R116FormalError(RuntimeError):
    """A formal R116 input, binding, or output violates the frozen contract."""


def relative_to_root(path: Path) -> str:
    return path.expanduser().absolute().relative_to(ROOT).as_posix()


def _regular_file(path: Path, *, label: str) -> Path:
    answer_contract.reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise R116FormalError(f"{label} is not a regular file: {path}")
    return path


def _regular_directory(path: Path, *, label: str) -> Path:
    answer_contract.reject_symlink_components(path)
    if path.is_symlink() or not path.is_dir():
        raise R116FormalError(f"{label} is not a regular directory: {path}")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _regular_file(path, label="JSONL input")
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise R116FormalError(f"JSONL input has an incomplete final line: {path}")
    records: list[dict[str, Any]] = []
    for number, raw in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise R116FormalError(f"invalid JSONL line {number}: {path}") from exc
        if not isinstance(value, dict):
            raise R116FormalError(f"JSONL line {number} is not an object: {path}")
        records.append(value)
    return records


def _source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = _regular_file(ROOT / relative, label="bound source")
        result[relative] = answer_contract.sha256_file(path)
    return result


def _file_binding(path: Path) -> dict[str, Any]:
    checked = _regular_file(path, label="bound input")
    return {
        "path": relative_to_root(checked),
        "bytes": checked.stat().st_size,
        "sha256": answer_contract.sha256_file(checked),
    }


def build_preregistration() -> dict[str, Any]:
    """Build the exact reproducible R116 preregistration payload."""

    tokenizer = answer_contract.formal_token_counter()
    return {
        "schema_version": PREREG_SCHEMA,
        "status": "frozen_before_formal_model_requests",
        "experiment_id": "R116",
        "method": METHOD,
        "condition": CONDITION,
        "benchmarks": {
            LOCOMO: {
                "scope": EXPECTED_SCOPE[LOCOMO],
                "dataset": _file_binding(LOCOMO_DATA),
                "source_root": relative_to_root(LOCOMO_SOURCE),
                "required_source_auditor": "scripts/audit_v88_gpt55_locomo.py",
                "required_source_audit": "audit.json",
                "required_source_scoring_input": "questions_all.json",
                "question_text_policy": "raw_dataset_question_exact",
                "source_recall_policy": "delivered_turn_ids_against_R002_turn_ids",
            },
            LONGMEMEVAL: {
                "scope": EXPECTED_SCOPE[LONGMEMEVAL],
                "dataset": _file_binding(LME_DATA),
                "source_root": relative_to_root(LME_SOURCE),
                "required_source_auditor": "scripts/audit_v88_gpt55_longmemeval.py",
                "required_source_audit": "audit.json",
                "required_source_scoring_input": "evaluation_input.json",
                "question_text_policy": "raw_dataset_question_exact",
                "source_recall_policy": (
                    "map_delivered_Dn_m_session_number_to_haystack_session_ids_"
                    "then_compare_R002_session_ids"
                ),
                "turn_level_gold_mapping": False,
            },
        },
        "evidence_mapping": {
            "directory": relative_to_root(EVIDENCE_DIR),
            "manifest": _file_binding(EVIDENCE_MANIFEST),
            "questions": _file_binding(EVIDENCE_QUESTIONS),
            "audit": _file_binding(EVIDENCE_AUDIT),
            "required_live_independent_audit": True,
        },
        "retrieval_and_answer": {
            "retrieval_model": MODEL,
            "answer_model": MODEL,
            "answer_prompt_sha256": answer_contract.sha256_bytes(
                answer_contract.ANSWER_PROMPT.encode("utf-8")
            ),
            "temperature": 0.0,
            "max_retrieval_rounds": MAX_ROUNDS,
            "single_visible_token_gate": True,
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
            "memory_access": "read_only",
            "gold_labels_model_visible": False,
        },
        "durability": {
            "retrieval_hash_chain_ledger": True,
            "answer_hash_chain_ledger": True,
            "request_response_artifacts_fsynced": True,
            "per_question_input_binding": True,
            "per_question_checkpoint": True,
            "no_clobber": True,
            "resume_complete_checkpoints_only": True,
            "incomplete_question_policy": "fail_closed_use_new_output_root",
        },
        "actual_model_evidence": {
            "exclusive_controlled_proxy_required": True,
            "exclusive_proxy_prefix_required_per_call": True,
            "gateway_root_flag_required": True,
            "gateway_origin_derived_from_validated_root": True,
            "arbitrary_upstream_forbidden": True,
            "exclusive_consumer_lock_required": True,
            "exact_provider_window_required": True,
            "provider_model": "gpt-5.5-2026-04-23",
            "service_tier": "flex",
            "response_model_must_equal": MODEL,
            "response_id_required": True,
            "provider_usage_required": True,
            "logical_and_physical_attempts_separate": True,
        },
        "source_hashes": _source_hashes(),
    }


def freeze_preregistration(path: Path = DEFAULT_PREREGISTRATION) -> dict[str, Any]:
    expected = build_preregistration()
    if path.exists():
        actual = answer_contract.read_json(path)
        if actual != expected:
            raise R116FormalError("existing R116 preregistration differs")
        return expected
    answer_contract.atomic_json_no_clobber(path, expected)
    return expected


def validate_preregistration(path: Path = DEFAULT_PREREGISTRATION) -> dict[str, Any]:
    actual = answer_contract.read_json(_regular_file(path, label="preregistration"))
    expected = build_preregistration()
    if actual != expected:
        raise R116FormalError("R116 preregistration differs from live frozen inputs")
    return actual


def preregistration_binding(path: Path = DEFAULT_PREREGISTRATION) -> dict[str, Any]:
    prereg = validate_preregistration(path)
    return {
        "path": relative_to_root(path),
        "sha256": answer_contract.sha256_file(path),
        "content_sha256": answer_contract.canonical_hash(prereg),
    }


def _evidence_records() -> dict[str, dict[str, Any]]:
    records = _read_jsonl(EVIDENCE_QUESTIONS)
    selected = {
        str(record["question_id"]): record
        for record in records
        if record.get("benchmark") in {"LoCoMo", "LongMemEval-S"}
    }
    if len(selected) != EXPECTED_SCOPE[LOCOMO]["questions"] + 500:
        raise R116FormalError("R002 LoCoMo/LongMemEval record count differs")
    return selected


def audit_relevant_evidence_mapping() -> dict[str, Any]:
    """Independently recompute the R116 subset without loading BEAM assets."""

    all_records = _read_jsonl(EVIDENCE_QUESTIONS)
    observed = [
        record
        for record in all_records
        if record.get("benchmark") in {"LoCoMo", "LongMemEval-S"}
    ]
    locomo_records, locomo_summary = mapping_auditor.recompute_locomo(LOCOMO_DATA)
    lme_records, lme_summary = mapping_auditor.recompute_lme(LME_DATA)
    expected = [*locomo_records, *lme_records]
    if observed != expected:
        raise R116FormalError("R002 R116-relevant records differ from recomputation")
    manifest = answer_contract.read_json(EVIDENCE_MANIFEST)
    persisted = answer_contract.read_json(EVIDENCE_AUDIT)
    if (
        not isinstance(manifest, dict)
        or manifest.get("questions_sha256")
        != answer_contract.sha256_file(EVIDENCE_QUESTIONS)
        or manifest.get("benchmarks", {}).get("LoCoMo", {}).get("counts")
        != json.loads(json.dumps(locomo_summary))
        or manifest.get("benchmarks", {}).get("LongMemEval-S", {}).get("counts")
        != json.loads(json.dumps(lme_summary))
        or not isinstance(persisted, dict)
        or persisted.get("status") != "passed"
        or persisted.get("manifest_sha256")
        != answer_contract.sha256_file(EVIDENCE_MANIFEST)
        or persisted.get("questions_sha256")
        != answer_contract.sha256_file(EVIDENCE_QUESTIONS)
        or persisted.get("source_recall_question_denominators", {}).get("LoCoMo")
        != 1_533
        or persisted.get("source_recall_question_denominators", {}).get(
            "LongMemEval-S"
        )
        != 500
    ):
        raise R116FormalError("R002 persisted manifest or audit binding differs")
    return {
        "status": "passed",
        "audit_scope": ["LoCoMo", "LongMemEval-S"],
        "question_count": len(expected),
        "locomo_questions": len(locomo_records),
        "longmemeval_questions": len(lme_records),
        "locomo_source_recall_questions": 1_533,
        "longmemeval_source_recall_questions": 500,
        "records_sha256": answer_contract.canonical_hash(expected),
        "auditor_sha256": answer_contract.sha256_file(Path(__file__)),
        "upstream_mapping_auditor_sha256": answer_contract.sha256_file(
            Path(mapping_auditor.__file__)
        ),
    }


def _load_dataset(path: Path, *, expected: int) -> list[dict[str, Any]]:
    payload = answer_contract.read_json(_regular_file(path, label="dataset"))
    if not isinstance(payload, list) or len(payload) != expected:
        raise R116FormalError(f"dataset inventory differs: {path}")
    if not all(isinstance(item, dict) for item in payload):
        raise R116FormalError(f"dataset contains a non-object item: {path}")
    return payload


def question_specs(benchmark: str) -> list[dict[str, Any]]:
    """Rebuild the complete formal question inventory from raw sources."""

    evidence = _evidence_records()
    specs: list[dict[str, Any]] = []
    if benchmark == LOCOMO:
        dataset = _load_dataset(LOCOMO_DATA, expected=10)
        for sample_index, sample in enumerate(dataset):
            sample_id = str(sample.get("sample_id", ""))
            qa_rows = sample.get("qa")
            if not sample_id or not isinstance(qa_rows, list):
                raise R116FormalError(f"LoCoMo sample {sample_index} is invalid")
            for question_index, qa in enumerate(qa_rows):
                question_id = f"locomo:{sample_id}:q{question_index:03d}"
                mapping = evidence.get(question_id)
                if (
                    not isinstance(qa, dict)
                    or mapping is None
                    or mapping.get("sample_index") != sample_index
                    or mapping.get("question_index") != question_index
                    or mapping.get("question") != qa.get("question")
                    or mapping.get("category") != int(qa.get("category", -1))
                ):
                    raise R116FormalError(f"LoCoMo mapping differs: {question_id}")
                specs.append(
                    {
                        "benchmark": LOCOMO,
                        "question_id": question_id,
                        "artifact_id": f"s{sample_index:02d}-q{question_index:03d}",
                        "dataset_index": sample_index,
                        "question_index": question_index,
                        "question": str(qa["question"]),
                        "category": int(qa["category"]),
                        "gold_source_ids": list(mapping["normalized_source_ids"]),
                        "source_recall_eligible": bool(
                            mapping["source_recall_eligible"]
                        ),
                        "gold_source_granularity": "turn",
                        "evidence_record": mapping,
                        "evidence_record_sha256": answer_contract.canonical_hash(mapping),
                    }
                )
        counts = Counter(spec["category"] for spec in specs)
        if len(specs) != 1_986 or dict(sorted(counts.items())) != {
            1: 282,
            2: 321,
            3: 96,
            4: 841,
            5: 446,
        }:
            raise R116FormalError("LoCoMo formal scope differs")
        if sum(spec["source_recall_eligible"] for spec in specs) != 1_533:
            raise R116FormalError("LoCoMo source-recall denominator differs")
        return specs

    if benchmark == LONGMEMEVAL:
        dataset = _load_dataset(LME_DATA, expected=500)
        for index, item in enumerate(dataset):
            dataset_id = str(item.get("question_id", ""))
            question_id = f"longmemeval-s:{dataset_id}"
            mapping = evidence.get(question_id)
            session_ids = item.get("haystack_session_ids")
            if (
                mapping is None
                or mapping.get("dataset_index") != index
                or mapping.get("dataset_question_id") != dataset_id
                or mapping.get("question") != item.get("question")
                or mapping.get("question_type") != item.get("question_type")
                or mapping.get("normalized_source_ids")
                != [str(value) for value in item.get("answer_session_ids", [])]
                or not isinstance(session_ids, list)
                or not session_ids
            ):
                raise R116FormalError(f"LongMemEval mapping differs: {question_id}")
            specs.append(
                {
                    "benchmark": LONGMEMEVAL,
                    "question_id": question_id,
                    "artifact_id": f"i{index:04d}-{dataset_id}",
                    "dataset_index": index,
                    "question": str(item["question"]),
                    "question_type": str(item["question_type"]),
                    "abstention": dataset_id.endswith("_abs"),
                    "gold_source_ids": list(mapping["normalized_source_ids"]),
                    "source_recall_eligible": bool(mapping["source_recall_eligible"]),
                    "gold_source_granularity": "session",
                    "haystack_session_ids": [str(value) for value in session_ids],
                    "evidence_record": mapping,
                    "evidence_record_sha256": answer_contract.canonical_hash(mapping),
                }
            )
        types = Counter(spec["question_type"] for spec in specs)
        if (
            len(specs) != 500
            or sum(spec["abstention"] for spec in specs) != 30
            or dict(sorted(types.items()))
            != EXPECTED_SCOPE[LONGMEMEVAL]["question_type_counts"]
            or not all(spec["source_recall_eligible"] for spec in specs)
        ):
            raise R116FormalError("LongMemEval formal scope differs")
        return specs
    raise R116FormalError(f"unknown R116 benchmark: {benchmark}")


def locomo_conversation(spec: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _load_dataset(LOCOMO_DATA, expected=10)
    sample = dataset[int(spec["dataset_index"])]
    conversation = sample.get("conversation")
    if not isinstance(conversation, dict):
        raise R116FormalError("LoCoMo conversation is invalid")
    return conversation


def lme_item(spec: Mapping[str, Any]) -> dict[str, Any]:
    return _load_dataset(LME_DATA, expected=500)[int(spec["dataset_index"])]


def lme_conversation(item: Mapping[str, Any]) -> dict[str, Any]:
    """Independently preserve released session/turn order as Dn:m anchors."""

    sessions = item.get("haystack_sessions")
    dates = item.get("haystack_dates")
    session_ids = item.get("haystack_session_ids")
    if (
        not isinstance(sessions, list)
        or not isinstance(dates, list)
        or not isinstance(session_ids, list)
        or not sessions
        or len(sessions) != len(dates)
        or len(sessions) != len(session_ids)
    ):
        raise R116FormalError("LongMemEval session metadata differs")
    conversation: dict[str, Any] = {
        "speaker_a": "user",
        "speaker_b": "assistant",
    }
    for session_number, (session, date) in enumerate(
        zip(sessions, dates), start=1
    ):
        if not isinstance(session, list) or not session:
            raise R116FormalError("LongMemEval session is empty")
        turns = []
        for turn_number, turn in enumerate(session, start=1):
            if (
                not isinstance(turn, dict)
                or not str(turn.get("role", "")).strip()
                or not isinstance(turn.get("content"), str)
            ):
                raise R116FormalError("LongMemEval turn is invalid")
            turns.append(
                {
                    "speaker": str(turn["role"]),
                    "text": turn["content"],
                    "dia_id": f"D{session_number}:{turn_number}",
                }
            )
        conversation[f"session_{session_number}"] = turns
        match = re.search(
            r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", str(date).strip()
        )
        if match is None:
            raise R116FormalError("LongMemEval date is not parseable")
        try:
            normalized_date = date_type(*map(int, match.groups())).isoformat()
        except ValueError as exc:
            raise R116FormalError("LongMemEval date is invalid") from exc
        conversation[f"session_{session_number}_date_time"] = normalized_date
    return conversation


def turn_index_and_session_map(
    spec: Mapping[str, Any],
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    if spec["benchmark"] == LOCOMO:
        index = readonly_control.build_turn_index(locomo_conversation(spec))
        return index, {}
    item = lme_item(spec)
    conversation = lme_conversation(item)
    index = readonly_control.build_turn_index(conversation)
    session_ids = [str(value) for value in item["haystack_session_ids"]]
    mapping = {
        source_id: session_ids[int(source_id.split(":", 1)[0][1:]) - 1]
        for source_id in index
    }
    return index, mapping


def source_memory_path(benchmark: str, source_root: Path, spec: Mapping[str, Any]) -> Path:
    if benchmark == LOCOMO:
        return source_root / f"memory_sample{int(spec['dataset_index'])}"
    pattern = f"{int(spec['dataset_index']):04d}_*/checkpoint.json"
    checkpoints = sorted((source_root / "items").glob(pattern))
    if len(checkpoints) != 1:
        raise R116FormalError(
            f"LongMemEval item {spec['dataset_index']} checkpoint count differs"
        )
    checkpoint = answer_contract.read_json(checkpoints[0])
    if checkpoint.get("question_id") != str(spec["question_id"]).split(":", 1)[1]:
        raise R116FormalError("LongMemEval checkpoint question identity differs")
    return checkpoints[0].parent / "memory"


def _verify_persisted_source_audit(
    *, source_root: Path, benchmark: str, live_report: Mapping[str, Any]
) -> dict[str, Any]:
    persisted_path = _regular_file(source_root / "audit.json", label="source audit")
    persisted = answer_contract.read_json(persisted_path)
    if not isinstance(persisted, dict) or persisted.get("status") != "passed":
        raise R116FormalError("persisted source audit did not pass")
    expected = dict(live_report)
    if benchmark == LOCOMO:
        scoring = _regular_file(
            source_root / "questions_all.json", label="LoCoMo scoring input"
        )
        expected["combined_sha256"] = answer_contract.sha256_file(scoring)
    else:
        scoring = _regular_file(
            source_root / "evaluation_input.json",
            label="LongMemEval scoring input",
        )
        expected["evaluation_input_sha256"] = answer_contract.sha256_file(scoring)
    expected["scoring_input"] = {
        "path": str(scoring),
        "sha256": answer_contract.sha256_file(scoring),
    }
    if persisted != expected:
        raise R116FormalError("persisted source audit differs from live audit")
    return {
        "audit": _file_binding(persisted_path),
        "scoring_input": _file_binding(scoring),
        "report_sha256": answer_contract.canonical_hash(persisted),
    }


def source_preflight(benchmark: str, prereg: Mapping[str, Any]) -> dict[str, Any]:
    """Audit and snapshot one complete canonical NativeMem source artifact."""

    source_root = ROOT / prereg["benchmarks"][benchmark]["source_root"]
    _regular_directory(source_root, label=f"{benchmark} source root")
    if benchmark == LOCOMO:
        _records, first = locomo_auditor.audit(source_root)
    elif benchmark == LONGMEMEVAL:
        _records, first = lme_auditor.audit(source_root)
    else:
        raise R116FormalError(f"unknown benchmark: {benchmark}")
    persisted = _verify_persisted_source_audit(
        source_root=source_root, benchmark=benchmark, live_report=first
    )
    manifest_binding = _file_binding(source_root / "run_manifest.json")
    inventory: list[dict[str, Any]] = []
    if benchmark == LOCOMO:
        for sample in range(10):
            memory = source_root / f"memory_sample{sample}"
            output = source_root / f"sample{sample}_questions.json"
            inventory.append(
                {
                    "sample_index": sample,
                    "memory": readonly_control.memory_descriptor(memory),
                    "sample_output": _file_binding(output),
                }
            )
    else:
        specs = question_specs(LONGMEMEVAL)
        for spec in specs:
            index = int(spec["dataset_index"])
            candidates = sorted(
                (source_root / "items").glob(f"{index:04d}_*/checkpoint.json")
            )
            if len(candidates) != 1:
                raise R116FormalError(f"LongMemEval item {index} is not unique")
            inventory.append(
                {
                    "dataset_index": index,
                    "question_id": spec["question_id"],
                    "checkpoint": _file_binding(candidates[0]),
                    "memory": readonly_control.memory_descriptor(
                        candidates[0].parent / "memory"
                    ),
                }
            )
    # A second independent audit rejects an active or changed source after the
    # potentially long memory inventory pass.
    if benchmark == LOCOMO:
        _records, second = locomo_auditor.audit(source_root)
    else:
        _records, second = lme_auditor.audit(source_root)
    if first != second:
        raise R116FormalError("source audit changed during preflight")
    inventory_sha = answer_contract.canonical_hash(inventory)
    binding = {
        "benchmark": benchmark,
        "source_root": relative_to_root(source_root),
        "run_manifest": manifest_binding,
        "independent_audit": persisted,
        "inventory": inventory,
        "inventory_sha256": inventory_sha,
        "questions": EXPECTED_SCOPE[benchmark]["questions"],
    }
    binding["binding_sha256"] = answer_contract.canonical_hash(binding)
    return binding


def run_preflight(
    preregistration: Path = DEFAULT_PREREGISTRATION,
    *,
    write_path: Path | None = DEFAULT_PREFLIGHT,
) -> dict[str, Any]:
    prereg = validate_preregistration(preregistration)
    mapping_report = audit_relevant_evidence_mapping()
    expected_mapping = prereg["evidence_mapping"]
    live_bindings = {
        "manifest": _file_binding(EVIDENCE_MANIFEST),
        "questions": _file_binding(EVIDENCE_QUESTIONS),
        "audit": _file_binding(EVIDENCE_AUDIT),
    }
    for key, value in live_bindings.items():
        if value != expected_mapping[key]:
            raise R116FormalError(f"R002 evidence {key} binding differs")
    benchmarks: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        try:
            binding = source_preflight(benchmark, prereg)
        except Exception as exc:  # noqa: BLE001
            benchmarks[benchmark] = {
                "status": "blocked",
                "reason": f"{type(exc).__name__}: {exc}",
                "source_root": prereg["benchmarks"][benchmark]["source_root"],
            }
        else:
            benchmarks[benchmark] = {"status": "ready", "binding": binding}
    report = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": (
            "ready" if all(row["status"] == "ready" for row in benchmarks.values())
            else "blocked"
        ),
        "preregistration": preregistration_binding(preregistration),
        "evidence_mapping": {
            "status": "passed",
            "live_audit": mapping_report,
            **live_bindings,
        },
        "benchmarks": benchmarks,
        "model_requests": 0,
        "network_requests": 0,
    }
    report["preflight_content_sha256"] = answer_contract.canonical_hash(report)
    if write_path is not None:
        answer_contract.atomic_json_replace(write_path, report)
    return report


def require_ready_binding(
    preflight: Mapping[str, Any], benchmark: str
) -> dict[str, Any]:
    row = preflight.get("benchmarks", {}).get(benchmark)
    if not isinstance(row, Mapping) or row.get("status") != "ready":
        reason = row.get("reason") if isinstance(row, Mapping) else "missing row"
        raise R116FormalError(f"{benchmark} formal input is blocked: {reason}")
    binding = row.get("binding")
    if not isinstance(binding, dict):
        raise R116FormalError(f"{benchmark} ready binding is absent")
    expected_hash = binding.get("binding_sha256")
    content = dict(binding)
    content.pop("binding_sha256", None)
    if expected_hash != answer_contract.canonical_hash(content):
        raise R116FormalError(f"{benchmark} source binding hash differs")
    return binding


def input_binding(
    *,
    preregistration: Path,
    source_binding: Mapping[str, Any],
    spec: Mapping[str, Any],
    memory_root: Path,
    turn_index: Mapping[str, Mapping[str, str]],
    session_by_turn: Mapping[str, str],
) -> dict[str, Any]:
    benchmark = str(spec["benchmark"])
    source_inventory = source_binding["inventory"]
    source_entry = source_inventory[int(spec["dataset_index"])]
    binding = {
        "schema_version": INPUT_BINDING_SCHEMA,
        "benchmark": benchmark,
        "question_id": spec["question_id"],
        "artifact_id": spec["artifact_id"],
        "question_sha256": answer_contract.sha256_bytes(
            str(spec["question"]).encode("utf-8")
        ),
        "question_text_policy": "raw_dataset_question_exact",
        "preregistration": preregistration_binding(preregistration),
        "source_binding_sha256": source_binding["binding_sha256"],
        "source_entry": source_entry,
        "memory": readonly_control.memory_descriptor(memory_root),
        "turn_index_sha256": answer_contract.canonical_hash(turn_index),
        "session_by_turn_sha256": answer_contract.canonical_hash(session_by_turn),
        "evidence_mapping": {
            "questions_sha256": answer_contract.sha256_file(EVIDENCE_QUESTIONS),
            "record": spec["evidence_record"],
            "record_sha256": spec["evidence_record_sha256"],
            "gold_source_granularity": spec["gold_source_granularity"],
            "gold_source_ids": spec["gold_source_ids"],
            "source_recall_eligible": spec["source_recall_eligible"],
        },
        "generic_control_gold_policy": (
            "R002_turn_ids"
            if benchmark == LOCOMO
            else "empty_not_eligible_session_recall_is_wrapper_audited"
        ),
    }
    binding["binding_sha256"] = answer_contract.canonical_hash(binding)
    return binding


def _trace_delivered_turn_ids(trace_path: Path) -> list[str]:
    turns: list[str] = []
    seen: set[str] = set()
    for record in _read_jsonl(trace_path):
        if record.get("record_type") != "delivery":
            continue
        delivered = record.get("delivered")
        if not isinstance(delivered, Mapping):
            continue
        for source_id in readonly_control.source_ids_in_text(
            str(delivered.get("text", ""))
        ):
            if source_id not in seen:
                seen.add(source_id)
                turns.append(source_id)
    return turns


def compute_source_recall(
    *,
    artifact_dir: Path,
    spec: Mapping[str, Any],
    session_by_turn: Mapping[str, str],
) -> dict[str, Any]:
    """Compute benchmark-granularity recall from gate-delivered evidence only."""

    result = answer_contract.read_json(artifact_dir / "result.json")
    trace_name = result.get("artifacts", {}).get("visible_token_trace")
    if not isinstance(trace_name, str) or Path(trace_name).name != trace_name:
        raise R116FormalError("visible-token trace path differs")
    delivered_turns = _trace_delivered_turn_ids(artifact_dir / trace_name)
    if spec["benchmark"] == LOCOMO:
        retrieved = delivered_turns
    else:
        retrieved = []
        for turn_id in delivered_turns:
            session_id = session_by_turn.get(turn_id)
            if session_id is not None and session_id not in retrieved:
                retrieved.append(session_id)
    gold = list(dict.fromkeys(str(value) for value in spec["gold_source_ids"]))
    hits = [value for value in gold if value in set(retrieved)]
    eligible = bool(spec["source_recall_eligible"])
    first_relevant_file = None
    for access in result.get("retrieval", {}).get("access_log", []):
        observed_turns = [str(value) for value in access.get("source_ids", [])]
        observed = (
            observed_turns
            if spec["benchmark"] == LOCOMO
            else [
                session_by_turn[value]
                for value in observed_turns
                if value in session_by_turn
            ]
        )
        if access.get("path") and set(observed) & set(gold):
            first_relevant_file = access["path"]
            break
    report = {
        "schema_version": SOURCE_RECALL_SCHEMA,
        "benchmark": spec["benchmark"],
        "question_id": spec["question_id"],
        "gold_source_granularity": spec["gold_source_granularity"],
        "evidence_record_sha256": spec["evidence_record_sha256"],
        "delivered_turn_ids": delivered_turns,
        "retrieved_source_ids": retrieved,
        "gold_source_ids": gold,
        "mapped_source_hits": hits,
        "source_recall_eligible": eligible,
        "mapped_source_recall": len(hits) / len(gold) if eligible and gold else None,
        "first_relevant_file": first_relevant_file,
        "derivation": (
            "delivered_Dn_m_turn_ids"
            if spec["benchmark"] == LOCOMO
            else "delivered_Dn_m_to_haystack_session_ids"
        ),
    }
    report["report_sha256"] = answer_contract.canonical_hash(report)
    return report


def validate_source_recall_hash(report: Mapping[str, Any]) -> None:
    expected = report.get("report_sha256")
    content = dict(report)
    content.pop("report_sha256", None)
    if expected != answer_contract.canonical_hash(content):
        raise R116FormalError("source-recall report hash differs")


def verify_source_unchanged(
    *, source_binding: Mapping[str, Any], benchmark: str
) -> None:
    source_root = ROOT / str(source_binding["source_root"])
    inventory = source_binding.get("inventory")
    if not isinstance(inventory, list):
        raise R116FormalError("source inventory is absent")
    live: list[dict[str, Any]] = []
    if benchmark == LOCOMO:
        for entry in inventory:
            sample = int(entry["sample_index"])
            live.append(
                {
                    "sample_index": sample,
                    "memory": readonly_control.memory_descriptor(
                        source_root / f"memory_sample{sample}"
                    ),
                    "sample_output": _file_binding(
                        source_root / f"sample{sample}_questions.json"
                    ),
                }
            )
    else:
        for entry in inventory:
            index = int(entry["dataset_index"])
            candidates = sorted(
                (source_root / "items").glob(f"{index:04d}_*/checkpoint.json")
            )
            if len(candidates) != 1:
                raise R116FormalError(f"LongMemEval item {index} changed")
            live.append(
                {
                    "dataset_index": index,
                    "question_id": entry["question_id"],
                    "checkpoint": _file_binding(candidates[0]),
                    "memory": readonly_control.memory_descriptor(
                        candidates[0].parent / "memory"
                    ),
                }
            )
    if live != inventory or answer_contract.canonical_hash(live) != source_binding.get(
        "inventory_sha256"
    ):
        raise R116FormalError("frozen NativeMem source inventory changed")


def safe_artifact_path(root: Path, relative: str, *, directory: bool) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise R116FormalError("artifact path escapes output root")
    candidate = root / candidate_relative
    answer_contract.reject_symlink_components(candidate)
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise R116FormalError("artifact path escapes output root") from exc
    if candidate.is_symlink() or (not candidate.is_dir() if directory else not candidate.is_file()):
        raise R116FormalError("artifact path is missing or has the wrong type")
    return resolved


__all__ = [
    "ANSWER_COMPLETION_RESERVATION_TOKENS",
    "ANSWER_MAX_TOKENS",
    "ANSWER_RETRIES",
    "BENCHMARKS",
    "BUDGET_TOKENS",
    "CONDITION",
    "DEFAULT_PREFLIGHT",
    "DEFAULT_PREREGISTRATION",
    "EXPECTED_SCOPE",
    "INPUT_BINDING_SCHEMA",
    "LOCOMO",
    "LONGMEMEVAL",
    "MAX_ROUNDS",
    "METHOD",
    "MODEL",
    "MODEL_CONTEXT_LIMIT_TOKENS",
    "PREFLIGHT_SCHEMA",
    "PREREG_SCHEMA",
    "R116FormalError",
    "SOURCE_RECALL_SCHEMA",
    "build_preregistration",
    "compute_source_recall",
    "freeze_preregistration",
    "input_binding",
    "lme_conversation",
    "preregistration_binding",
    "question_specs",
    "require_ready_binding",
    "run_preflight",
    "safe_artifact_path",
    "source_memory_path",
    "turn_index_and_session_map",
    "validate_preregistration",
    "validate_source_recall_hash",
    "verify_source_unchanged",
]
