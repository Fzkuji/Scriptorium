#!/usr/bin/env python3
"""Rubric-nugget semantic judge for the frozen 40-question BEAM screening subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from scripts import evaluate_v88_gpt55_beam as FULL_EVALUATOR
else:
    import evaluate_v88_gpt55_beam as FULL_EVALUATOR


SCHEMA_VERSION = 1
PROTOCOL_CLASS = "screening_subset"
ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_SELECTION = {"100K": [0, 1]}
EXPECTED_QUESTION_COUNT = 40
EXPECTED_RUBRIC_NUGGET_COUNT = 103
MAX_RETRIES = 3
TIERS = ("luna", "terra", "sol")
QUESTION_TYPES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)


class ScreeningJudgeError(RuntimeError):
    """The input or output violates the screening semantic-judge contract."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _validate_source_report(report: object) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        raise ScreeningJudgeError("source report root is not an object")
    content_hash = report.get("report_content_sha256")
    content = dict(report)
    content.pop("report_content_sha256", None)
    if content_hash != _canonical_hash(content):
        raise ScreeningJudgeError("source report content hash differs")
    exact = {
        "schema_version": SCHEMA_VERSION,
        "protocol_class": PROTOCOL_CLASS,
        "benchmark": "BEAM",
        "dataset_split": "100K",
        "formal_scope_verified": False,
        "metric_scope": "deterministic_string_match_diagnostic",
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise ScreeningJudgeError(f"source report has mismatched {key}")
    records = report.get("records")
    if (
        report.get("question_count") != EXPECTED_QUESTION_COUNT
        or not isinstance(records, list)
        or len(records) != EXPECTED_QUESTION_COUNT
    ):
        raise ScreeningJudgeError("screening report must contain exactly 40 questions")
    if any(not isinstance(record, dict) for record in records):
        raise ScreeningJudgeError("screening report contains a non-object record")
    for record in records:
        conversation_index = record.get("conversation_index")
        if isinstance(conversation_index, bool) or not isinstance(
            conversation_index, int
        ):
            raise ScreeningJudgeError("screening conversation index is invalid")
        question_index = record.get("question_index")
        if isinstance(question_index, bool) or not isinstance(question_index, int):
            raise ScreeningJudgeError("screening question index is invalid")
        if not isinstance(record.get("rubric"), list):
            raise ScreeningJudgeError("screening rubric must be a list")
    rubric_count = sum(len(record["rubric"]) for record in records)
    if rubric_count != EXPECTED_RUBRIC_NUGGET_COUNT:
        raise ScreeningJudgeError(
            "screening report must contain exactly 103 rubric nuggets"
        )
    conversation_counts = Counter(
        record.get("conversation_index") for record in records
    )
    if conversation_counts != Counter({0: 20, 1: 20}):
        raise ScreeningJudgeError("screening conversation selection must be 100K 0/1")
    type_counts = Counter(record.get("question_type") for record in records)
    if type_counts != Counter({question_type: 4 for question_type in QUESTION_TYPES}):
        raise ScreeningJudgeError(
            "screening report must contain four questions per type"
        )
    by_conversation: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_conversation[record["conversation_index"]].append(record)
    for conversation_index, conversation_records in by_conversation.items():
        indices = [record.get("question_index") for record in conversation_records]
        if len(set(indices)) != 20 or set(indices) != set(range(20)):
            raise ScreeningJudgeError("screening report has invalid question indices")
        per_type = Counter(
            record.get("question_type") for record in conversation_records
        )
        if per_type != Counter({question_type: 2 for question_type in QUESTION_TYPES}):
            raise ScreeningJudgeError(
                "each screening conversation must contain two questions per type"
            )
        conversation_id = str(conversation_index + 1)
        for record in conversation_records:
            question_index = record["question_index"]
            if record.get("question_id") != f"{conversation_id}-q{question_index}":
                raise ScreeningJudgeError("screening source question IDs differ")
            if record.get("unit_id") != f"100K-conv-{conversation_id}":
                raise ScreeningJudgeError("screening source unit IDs differ")
            if (
                not isinstance(record.get("question"), str)
                or not record["question"].strip()
            ):
                raise ScreeningJudgeError("screening question text is empty")
            if not isinstance(record.get("answer"), str):
                raise ScreeningJudgeError("screening answer is not a string")
            rubric = record.get("rubric")
            if any(not isinstance(item, str) or not item.strip() for item in rubric):
                raise ScreeningJudgeError("screening rubric contains an empty nugget")
    boundary = report.get("official_beam_boundary")
    if boundary != {
        "rubric_judge_executed": False,
        "official_or_formal_score": False,
        "judge_ready_records_included": True,
    }:
        raise ScreeningJudgeError("source report official BEAM boundary differs")
    return records


def load_screening_report(path: Path) -> dict[str, Any]:
    """Convert an offline diagnostic report into tier-independent judge records."""

    source_path = path.expanduser().resolve()
    payload = source_path.read_bytes()
    try:
        report = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScreeningJudgeError("source report is not valid JSON") from exc
    source_records = _validate_source_report(report)
    records: list[dict[str, Any]] = []
    for source in source_records:
        conversation_index = int(source["conversation_index"])
        question_index = int(source["question_index"])
        question_type = str(source["question_type"])
        records.append(
            {
                "question_id": (
                    f"100K_{conversation_index}_q{question_index}_{question_type}"
                ),
                "source_question_id": str(source["question_id"]),
                "chat_size": "100K",
                "conversation_index": conversation_index,
                "question_index": question_index,
                "question_type": question_type,
                "question": str(source["question"]),
                "answer": str(source["answer"]),
                "rubric": list(source["rubric"]),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_class": PROTOCOL_CLASS,
        "benchmark": "BEAM",
        "dataset_split": "100K",
        "formal_scope_verified": False,
        "selected_conversations": EXPECTED_SELECTION,
        "question_count": len(records),
        "rubric_nugget_count": sum(len(record["rubric"]) for record in records),
        "question_type_counts": dict(
            Counter(record["question_type"] for record in records)
        ),
        "source_report_path": str(source_path),
        "source_report_sha256": _sha256_bytes(payload),
        "source_report_content_sha256": report["report_content_sha256"],
        "records": records,
    }


def _paired_screening_inventory_sha256(evaluation_input: dict[str, Any]) -> str:
    fields = (
        "question_id",
        "source_question_id",
        "chat_size",
        "conversation_index",
        "question_index",
        "question_type",
        "question",
        "rubric",
    )
    return FULL_EVALUATOR.stable_hash(
        [{key: record[key] for key in fields} for record in evaluation_input["records"]]
    )


def _judge_procedure_sha256(config: dict[str, Any]) -> str:
    return FULL_EVALUATOR.stable_hash(
        {
            "profile": config["profile"],
            "provider": config["provider"],
            "model": config["model"],
            "expected_response_model": config["expected_response_model"],
            "prompt_source_sha256": config["prompt_source_sha256"],
            "system_prompt_sha256": config["system_prompt_sha256"],
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
            "max_retries": config["max_retries"],
            "scoring": "question_mean_over_rubric_nuggets",
            "valid_scores": [0.0, 0.5, 1.0],
            "transport_contract": config["transport_contract"],
        }
    )


def build_screening_config(
    evaluation_input: dict[str, Any],
    *,
    tier: str,
    base_url: str,
    max_tokens: int,
    openrouter_gateway: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze the primary judge configuration without claiming formal scope."""

    if tier not in TIERS:
        raise ScreeningJudgeError("screening tier is invalid")
    spec = FULL_EVALUATOR.JUDGE_PROFILES["primary"]
    config = {
        "formal": False,
        "protocol_class": PROTOCOL_CLASS,
        "tier": tier,
        "claim_scope": "two-conversation paired screening only",
        "profile": "primary",
        "provider": spec["provider"],
        "model": spec["model"],
        "expected_response_model": spec["expected_response_model"],
        "comparison_label": "primary_openrouter_gpt4o_mini_screening_subset",
        "comparison_scope": "two_conversation_paired_screening_only",
        "judge_procedure_comparable_to_project_unified_protocol": True,
        "comparable_to_project_unified_protocol": False,
        "full_benchmark_comparable": False,
        "comparable_to_published_beam_official": False,
        "comparable_to_mem0": False,
        "official_or_formal_score": False,
        "metric": "BEAM rubric-nugget mean",
        "metric_scope": "rubric_nugget_only",
        "pass_threshold": 0.5,
        "base_url": base_url.rstrip("/"),
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "max_retries": MAX_RETRIES,
        "transport_contract": (
            "marked_openrouter_gateway"
            if openrouter_gateway is not None
            else "injected_test_judge"
        ),
        "openrouter_gateway": openrouter_gateway,
        "openai_flex_gateway": None,
        "proxy_request_log": None,
        "selected_conversations": EXPECTED_SELECTION,
        "question_count": EXPECTED_QUESTION_COUNT,
        "rubric_nugget_count": EXPECTED_RUBRIC_NUGGET_COUNT,
        "question_type_counts": {question_type: 4 for question_type in QUESTION_TYPES},
        "source_report_path": evaluation_input["source_report_path"],
        "source_report_sha256": evaluation_input["source_report_sha256"],
        "source_report_content_sha256": evaluation_input[
            "source_report_content_sha256"
        ],
        "input_records_sha256": FULL_EVALUATOR.stable_hash(evaluation_input["records"]),
        "paired_screening_inventory_sha256": (
            _paired_screening_inventory_sha256(evaluation_input)
        ),
        "evaluator_source": str(SCRIPT_PATH.relative_to(ROOT)),
        "evaluator_source_sha256": FULL_EVALUATOR.sha256_file(SCRIPT_PATH),
        "reused_evaluator_source": str(FULL_EVALUATOR.SCRIPT_PATH.relative_to(ROOT)),
        "reused_evaluator_source_sha256": FULL_EVALUATOR.sha256_file(
            FULL_EVALUATOR.SCRIPT_PATH
        ),
        "prompt_source": str(FULL_EVALUATOR.PROMPTS_PATH.relative_to(ROOT)),
        "prompt_source_sha256": FULL_EVALUATOR.sha256_file(FULL_EVALUATOR.PROMPTS_PATH),
        "system_prompt_sha256": hashlib.sha256(
            FULL_EVALUATOR.BEAM_PROMPTS.BEAM_JUDGE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "event_ordering": {
            "scope": "nugget-only",
            "official_tau_b_times_f1": "not_computed",
        },
    }
    config["judge_procedure_sha256"] = _judge_procedure_sha256(config)
    return config


def validate_screening_config(
    config: dict[str, Any], evaluation_input: dict[str, Any]
) -> None:
    """Reject model, scope, provenance, or evaluator drift."""

    tier = config.get("tier")
    if tier not in TIERS:
        raise ScreeningJudgeError("screening judge config has mismatched tier")
    exact = {
        "formal": False,
        "protocol_class": PROTOCOL_CLASS,
        "tier": tier,
        "claim_scope": "two-conversation paired screening only",
        "profile": "primary",
        "provider": "OpenRouter",
        "model": "openai/gpt-4o-mini",
        "expected_response_model": "openai/gpt-4o-mini",
        "comparison_label": "primary_openrouter_gpt4o_mini_screening_subset",
        "comparison_scope": "two_conversation_paired_screening_only",
        "judge_procedure_comparable_to_project_unified_protocol": True,
        "comparable_to_project_unified_protocol": False,
        "full_benchmark_comparable": False,
        "comparable_to_published_beam_official": False,
        "comparable_to_mem0": False,
        "official_or_formal_score": False,
        "metric": "BEAM rubric-nugget mean",
        "metric_scope": "rubric_nugget_only",
        "pass_threshold": 0.5,
        "temperature": 0.0,
        "max_retries": MAX_RETRIES,
        "openai_flex_gateway": None,
        "proxy_request_log": None,
        "selected_conversations": EXPECTED_SELECTION,
        "question_count": EXPECTED_QUESTION_COUNT,
        "rubric_nugget_count": EXPECTED_RUBRIC_NUGGET_COUNT,
        "question_type_counts": {question_type: 4 for question_type in QUESTION_TYPES},
        "source_report_path": evaluation_input["source_report_path"],
        "source_report_sha256": evaluation_input["source_report_sha256"],
        "source_report_content_sha256": evaluation_input[
            "source_report_content_sha256"
        ],
        "input_records_sha256": FULL_EVALUATOR.stable_hash(evaluation_input["records"]),
        "paired_screening_inventory_sha256": (
            _paired_screening_inventory_sha256(evaluation_input)
        ),
        "evaluator_source": str(SCRIPT_PATH.relative_to(ROOT)),
        "evaluator_source_sha256": FULL_EVALUATOR.sha256_file(SCRIPT_PATH),
        "reused_evaluator_source": str(FULL_EVALUATOR.SCRIPT_PATH.relative_to(ROOT)),
        "reused_evaluator_source_sha256": FULL_EVALUATOR.sha256_file(
            FULL_EVALUATOR.SCRIPT_PATH
        ),
        "prompt_source": str(FULL_EVALUATOR.PROMPTS_PATH.relative_to(ROOT)),
        "prompt_source_sha256": FULL_EVALUATOR.sha256_file(FULL_EVALUATOR.PROMPTS_PATH),
        "system_prompt_sha256": hashlib.sha256(
            FULL_EVALUATOR.BEAM_PROMPTS.BEAM_JUDGE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "event_ordering": {
            "scope": "nugget-only",
            "official_tau_b_times_f1": "not_computed",
        },
        "judge_procedure_sha256": _judge_procedure_sha256(config),
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise ScreeningJudgeError(f"screening judge config has mismatched {key}")
    max_tokens = config.get("max_tokens")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
    ):
        raise ScreeningJudgeError("screening judge config has invalid max_tokens")
    if not str(config.get("base_url", "")).strip():
        raise ScreeningJudgeError("screening judge config has invalid base_url")
    transport = config.get("transport_contract")
    binding = config.get("openrouter_gateway")
    if transport == "injected_test_judge":
        if (
            binding is not None
            or config["base_url"] != FULL_EVALUATOR.TEST_PRIMARY_BASE_URL
        ):
            raise ScreeningJudgeError("injected screening judge binding differs")
    elif transport == "marked_openrouter_gateway":
        if not isinstance(binding, dict):
            raise ScreeningJudgeError("marked OpenRouter gateway binding is absent")
        try:
            FULL_EVALUATOR.openrouter_gateway_evidence.validate_binding(
                binding,
                result_root=Path(str(binding.get("result_root", ""))),
                base_url=config["base_url"],
            )
        except FULL_EVALUATOR.openrouter_gateway_evidence.GatewayEvidenceError as exc:
            raise ScreeningJudgeError(str(exc)) from exc
    else:
        raise ScreeningJudgeError("screening judge transport_contract is invalid")


def _json_payload(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _publish_json_no_clobber_or_validate(path: Path, value: object) -> None:
    payload = _json_payload(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ScreeningJudgeError(f"refusing to clobber existing artifact: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _screening_results(
    evaluation_input: dict[str, Any],
    judgments: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    results = FULL_EVALUATOR.aggregate_results(evaluation_input, judgments, config)
    source_ids = {
        record["question_id"]: record["source_question_id"]
        for record in evaluation_input["records"]
    }
    for evaluation in results["evaluations"]:
        evaluation["source_question_id"] = source_ids[evaluation["question_id"]]
    results.update(
        {
            "protocol_class": PROTOCOL_CLASS,
            "tier": config["tier"],
            "formal_scope_verified": False,
            "rubric_nugget_count": EXPECTED_RUBRIC_NUGGET_COUNT,
            "claim_scope": config["claim_scope"],
            "paired_screening_comparison_allowed": True,
            "paired_screening_inventory_sha256": config[
                "paired_screening_inventory_sha256"
            ],
            "paired_screening_requirement": (
                "compare tiers only when this inventory hash and judge "
                "procedure match"
            ),
            "judge_procedure_sha256": config["judge_procedure_sha256"],
            "judge_procedure_comparable_to_project_unified_protocol": True,
            "comparable_to_project_unified_protocol": False,
            "full_benchmark_comparable": False,
            "comparable_to_published_beam_official": False,
            "comparable_to_mem0": False,
            "official_or_formal_score": False,
            "source_report_path": evaluation_input["source_report_path"],
            "source_report_sha256": evaluation_input["source_report_sha256"],
            "source_report_content_sha256": evaluation_input[
                "source_report_content_sha256"
            ],
            "input_records_sha256": config["input_records_sha256"],
            "judge_config": config,
            "judge_config_hash": FULL_EVALUATOR.stable_hash(config),
        }
    )
    return results


def _exact_job_inventory(
    output_dir: Path, evaluation_input: dict[str, Any]
) -> tuple[set[str], set[str]]:
    identifiers = {
        FULL_EVALUATOR.job_id(record["question_id"], nugget_index)
        for record in evaluation_input["records"]
        for nugget_index, _ in enumerate(record["rubric"])
    }
    expected_judgments = {f"{identifier}.json" for identifier in identifiers}
    expected_ledgers = {f"{identifier}.jsonl" for identifier in identifiers}
    actual_judgments = {path.name for path in (output_dir / "judgments").glob("*.json")}
    actual_ledgers = {
        path.name for path in (output_dir / "attempt_ledgers").glob("*.jsonl")
    }
    if actual_judgments != expected_judgments:
        raise ScreeningJudgeError("screening judgment file inventory differs")
    if actual_ledgers != expected_ledgers:
        raise ScreeningJudgeError("screening attempt-ledger inventory differs")
    return expected_judgments, expected_ledgers


def _build_evaluation_audit(
    *,
    evaluation_input: dict[str, Any],
    output_dir: Path,
    judgments: dict[str, dict[str, Any]],
    results: dict[str, Any],
    config: dict[str, Any],
    gateway_evidence: dict[str, Any],
) -> dict[str, Any]:
    judgment_files, ledger_files = _exact_job_inventory(output_dir, evaluation_input)
    if len(judgments) != EXPECTED_RUBRIC_NUGGET_COUNT or any(
        value.get("status") != "complete" for value in judgments.values()
    ):
        raise ScreeningJudgeError("screening audit requires 103 complete judgments")
    response_ids = [value["response_id"] for value in judgments.values()]
    if len(response_ids) != len(set(response_ids)):
        raise ScreeningJudgeError("screening judge response IDs are duplicated")
    usage = {
        name: sum(int(value["usage"][name]) for value in judgments.values())
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise ScreeningJudgeError("screening aggregate usage differs")
    if results.get("status") != "complete" or results.get("metrics") is None:
        raise ScreeningJudgeError("screening results are incomplete")
    return {
        "schema_version": 1,
        "status": "passed",
        "protocol_class": PROTOCOL_CLASS,
        "benchmark": "BEAM",
        "tier": config["tier"],
        "formal_scope_verified": False,
        "claim_scope": config["claim_scope"],
        "question_count": EXPECTED_QUESTION_COUNT,
        "rubric_nugget_count": EXPECTED_RUBRIC_NUGGET_COUNT,
        "source_report_sha256": evaluation_input["source_report_sha256"],
        "source_report_content_sha256": evaluation_input[
            "source_report_content_sha256"
        ],
        "input_records_sha256": config["input_records_sha256"],
        "paired_screening_inventory_sha256": config[
            "paired_screening_inventory_sha256"
        ],
        "judge_procedure_sha256": config["judge_procedure_sha256"],
        "config_sha256": FULL_EVALUATOR.stable_hash(config),
        "evaluator_source_sha256": config["evaluator_source_sha256"],
        "reused_evaluator_source_sha256": config["reused_evaluator_source_sha256"],
        "prompt_source_sha256": config["prompt_source_sha256"],
        "judgment_files": len(judgment_files),
        "attempt_ledgers": len(ledger_files),
        "judge_response_ids": len(response_ids),
        "judge_usage": usage,
        "gateway_evidence": gateway_evidence,
        "results_sha256": FULL_EVALUATOR.sha256_file(output_dir / "results.json"),
        "metrics_sha256": FULL_EVALUATOR.sha256_file(output_dir / "metrics.json"),
        "full_benchmark_comparable": False,
        "comparable_to_published_beam_official": False,
        "official_or_formal_score": False,
        "event_ordering": {
            "scope": "nugget-only",
            "official_tau_b_times_f1": "not_computed",
        },
    }


def _audit_screening_locked(
    *,
    evaluation_input: dict[str, Any],
    output_dir: Path,
    tier: str,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_path = output_dir / "run_state.json"
    if not state_path.is_file():
        raise ScreeningJudgeError("screening run state is absent")
    state = FULL_EVALUATOR.read_json(state_path)
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != 1
        or state.get("benchmark") != "BEAM"
        or state.get("status") != "complete"
    ):
        raise ScreeningJudgeError("screening run state is not complete")
    config = state.get("config")
    if not isinstance(config, dict):
        raise ScreeningJudgeError("screening run state has no config")
    validate_screening_config(config, evaluation_input)
    if config.get("tier") != tier:
        raise ScreeningJudgeError("screening audit tier differs")
    if expected_config is not None and config != expected_config:
        raise ScreeningJudgeError("stored screening judge config differs")
    config_hash = FULL_EVALUATOR.stable_hash(config)
    exact_state = {
        "input_path": evaluation_input["source_report_path"],
        "input_sha256": evaluation_input["source_report_sha256"],
        "config_hash": config_hash,
        "total_nuggets": EXPECTED_RUBRIC_NUGGET_COUNT,
        "completed_nuggets": EXPECTED_RUBRIC_NUGGET_COUNT,
        "failed_nuggets": 0,
        "missing_nuggets": 0,
    }
    for key, expected in exact_state.items():
        if state.get(key) != expected:
            raise ScreeningJudgeError(f"screening run state has mismatched {key}")
    judgments, missing = FULL_EVALUATOR.load_all_judgments(
        output_dir,
        evaluation_input,
        input_sha256=evaluation_input["source_report_sha256"],
        config_hash=config_hash,
        config=config,
    )
    if missing:
        raise ScreeningJudgeError("screening audit found missing judgments")
    _exact_job_inventory(output_dir, evaluation_input)
    recomputed = _screening_results(evaluation_input, judgments, config)
    results_path = output_dir / "results.json"
    metrics_path = output_dir / "metrics.json"
    audit_path = output_dir / "evaluation_audit.json"
    if (
        not results_path.is_file()
        or FULL_EVALUATOR.read_json(results_path) != recomputed
    ):
        raise ScreeningJudgeError("stored results differ from recomputed judgments")
    if (
        not metrics_path.is_file()
        or FULL_EVALUATOR.read_json(metrics_path) != recomputed["metrics"]
    ):
        raise ScreeningJudgeError("stored metrics differ from recomputed judgments")
    if state.get("results_sha256") != FULL_EVALUATOR.sha256_file(results_path):
        raise ScreeningJudgeError("screening results hash differs")
    if state.get("metrics_sha256") != FULL_EVALUATOR.sha256_file(metrics_path):
        raise ScreeningJudgeError("screening metrics hash differs")
    response_ids = [value["response_id"] for value in judgments.values()]
    if config["transport_contract"] == "marked_openrouter_gateway":
        final_binding = state.get("openrouter_gateway_final")
        if not isinstance(final_binding, dict):
            raise ScreeningJudgeError("screening OpenRouter final binding is absent")
        try:
            gateway_evidence = (
                FULL_EVALUATOR.openrouter_gateway_evidence.verify_response_ids(
                    final_binding,
                    response_ids,
                )
            )
        except FULL_EVALUATOR.openrouter_gateway_evidence.GatewayEvidenceError as exc:
            raise ScreeningJudgeError(str(exc)) from exc
    else:
        gateway_evidence = {
            "status": "synthetic_injected_judge_only",
            "bound_response_ids": 0,
        }
    if state.get("openrouter_gateway_evidence") != gateway_evidence:
        raise ScreeningJudgeError("stored gateway evidence differs")
    recomputed_audit = _build_evaluation_audit(
        evaluation_input=evaluation_input,
        output_dir=output_dir,
        judgments=judgments,
        results=recomputed,
        config=config,
        gateway_evidence=gateway_evidence,
    )
    if (
        not audit_path.is_file()
        or FULL_EVALUATOR.read_json(audit_path) != recomputed_audit
    ):
        raise ScreeningJudgeError("stored evaluation audit differs")
    if state.get("evaluation_audit_sha256") != FULL_EVALUATOR.sha256_file(audit_path):
        raise ScreeningJudgeError("screening evaluation-audit hash differs")
    return recomputed_audit


def audit_screening_run(
    *,
    report_path: Path,
    output_root: Path,
    tier: str,
) -> dict[str, Any]:
    """Recompute one completed tier from source report, ledgers, and responses."""

    if tier not in TIERS:
        raise ScreeningJudgeError("screening tier is invalid")
    evaluation_input = load_screening_report(report_path)
    output_dir = output_root.expanduser().resolve() / tier
    lock_handle = None
    try:
        lock_handle = FULL_EVALUATOR.acquire_output_lock(output_dir)
        return _audit_screening_locked(
            evaluation_input=evaluation_input,
            output_dir=output_dir,
            tier=tier,
        )
    except FULL_EVALUATOR.EvaluationError as exc:
        raise ScreeningJudgeError(str(exc)) from exc
    finally:
        FULL_EVALUATOR.release_output_lock(lock_handle)


def run_screening_judge(
    *,
    report_path: Path,
    output_root: Path,
    tier: str,
    config: dict[str, Any],
    client: Any,
    resume: bool,
) -> dict[str, Any]:
    """Run or resume one tier without invoking the formal input validator."""

    evaluation_input = load_screening_report(report_path)
    if config.get("tier") != tier:
        raise ScreeningJudgeError("screening tier differs from judge config")
    validate_screening_config(config, evaluation_input)
    output_dir = output_root.expanduser().resolve() / tier
    lock_handle = None
    state: dict[str, Any] | None = None
    try:
        lock_handle = FULL_EVALUATOR.acquire_output_lock(output_dir)
        state = FULL_EVALUATOR.load_or_create_state(
            output_dir,
            input_path=report_path.expanduser().resolve(),
            input_sha256=evaluation_input["source_report_sha256"],
            config=config,
            total_jobs=EXPECTED_RUBRIC_NUGGET_COUNT,
            resume=resume,
        )
        if state.get("status") == "complete":
            _audit_screening_locked(
                evaluation_input=evaluation_input,
                output_dir=output_dir,
                tier=tier,
                expected_config=config,
            )
            return FULL_EVALUATOR.read_json(output_dir / "results.json")
        config_hash = state["config_hash"]
        for record in evaluation_input["records"]:
            for nugget_index, nugget in enumerate(record["rubric"]):
                judgment = FULL_EVALUATOR.judge_one_job(
                    client,
                    output_dir=output_dir,
                    record=record,
                    nugget_index=nugget_index,
                    nugget=nugget,
                    input_sha256=evaluation_input["source_report_sha256"],
                    config_hash=config_hash,
                    config=config,
                )
                state.update(
                    {
                        "updated_at": FULL_EVALUATOR.utc_now(),
                        "last_job_id": judgment["job_id"],
                        "last_job_status": judgment["status"],
                    }
                )
                FULL_EVALUATOR.atomic_json(output_dir / "run_state.json", state)
        judgments, missing = FULL_EVALUATOR.load_all_judgments(
            output_dir,
            evaluation_input,
            input_sha256=evaluation_input["source_report_sha256"],
            config_hash=config_hash,
            config=config,
        )
        results = _screening_results(evaluation_input, judgments, config)
        failed = [
            identifier
            for identifier, value in judgments.items()
            if value.get("status") != "complete"
        ]
        if failed or missing or results["status"] != "complete":
            _publish_json_no_clobber_or_validate(output_dir / "results.json", results)
            state.update(
                {
                    "status": "failed",
                    "updated_at": FULL_EVALUATOR.utc_now(),
                    "finished_at": FULL_EVALUATOR.utc_now(),
                    "completed_nuggets": len(judgments) - len(failed),
                    "failed_nuggets": len(failed),
                    "missing_nuggets": len(missing),
                    "results_sha256": FULL_EVALUATOR.sha256_file(
                        output_dir / "results.json"
                    ),
                    "metrics_sha256": None,
                }
            )
            FULL_EVALUATOR.atomic_json(output_dir / "run_state.json", state)
            return results
        gateway_final = None
        if config["transport_contract"] == "marked_openrouter_gateway":
            try:
                gateway_final = (
                    FULL_EVALUATOR.openrouter_gateway_evidence.finalize_binding(
                        config["openrouter_gateway"]
                    )
                )
                gateway_evidence = (
                    FULL_EVALUATOR.openrouter_gateway_evidence.verify_response_ids(
                        gateway_final,
                        [value["response_id"] for value in judgments.values()],
                    )
                )
            except (
                FULL_EVALUATOR.openrouter_gateway_evidence.GatewayEvidenceError
            ) as exc:
                raise ScreeningJudgeError(str(exc)) from exc
        else:
            gateway_evidence = {
                "status": "synthetic_injected_judge_only",
                "bound_response_ids": 0,
            }
        _publish_json_no_clobber_or_validate(output_dir / "results.json", results)
        _publish_json_no_clobber_or_validate(
            output_dir / "metrics.json", results["metrics"]
        )
        audit = _build_evaluation_audit(
            evaluation_input=evaluation_input,
            output_dir=output_dir,
            judgments=judgments,
            results=results,
            config=config,
            gateway_evidence=gateway_evidence,
        )
        _publish_json_no_clobber_or_validate(
            output_dir / "evaluation_audit.json", audit
        )
        state.update(
            {
                "status": "complete",
                "updated_at": FULL_EVALUATOR.utc_now(),
                "finished_at": FULL_EVALUATOR.utc_now(),
                "completed_nuggets": EXPECTED_RUBRIC_NUGGET_COUNT,
                "failed_nuggets": 0,
                "missing_nuggets": 0,
                "results_sha256": FULL_EVALUATOR.sha256_file(
                    output_dir / "results.json"
                ),
                "metrics_sha256": FULL_EVALUATOR.sha256_file(
                    output_dir / "metrics.json"
                ),
                "evaluation_audit_sha256": FULL_EVALUATOR.sha256_file(
                    output_dir / "evaluation_audit.json"
                ),
                "openrouter_gateway_final": gateway_final,
                "openrouter_gateway_evidence": gateway_evidence,
            }
        )
        FULL_EVALUATOR.atomic_json(output_dir / "run_state.json", state)
        return results
    except (FULL_EVALUATOR.EvaluationError, ScreeningJudgeError) as exc:
        if state is not None and state.get("status") != "complete":
            state.update(
                {
                    "status": "failed",
                    "updated_at": FULL_EVALUATOR.utc_now(),
                    "finished_at": FULL_EVALUATOR.utc_now(),
                    "failure": f"{type(exc).__name__}: {exc}"[:2000],
                }
            )
            FULL_EVALUATOR.atomic_json(output_dir / "run_state.json", state)
        if isinstance(exc, ScreeningJudgeError):
            raise
        raise ScreeningJudgeError(str(exc)) from exc
    finally:
        FULL_EVALUATOR.release_output_lock(lock_handle)


def create_openai_client(*, base_url: str) -> Any:
    """Create the zero-retry client only after both request gates pass."""

    from openai import OpenAI

    return OpenAI(
        api_key="local-openrouter-gateway",
        base_url=base_url,
        http_client=FULL_EVALUATOR.httpx.Client(
            trust_env=FULL_EVALUATOR.should_trust_environment_proxy(base_url),
            timeout=180,
        ),
        max_retries=0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--openrouter-gateway-root", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--allow-model-requests", action="store_true")
    parser.add_argument(
        "--confirm-screening-subset-requests",
        action="store_true",
    )
    args = parser.parse_args(argv)
    request_flags = (
        args.allow_model_requests,
        args.confirm_screening_subset_requests,
    )
    if request_flags[0] != request_flags[1]:
        parser.error(
            "real judging requires both --allow-model-requests and "
            "--confirm-screening-subset-requests"
        )
    requests_enabled = all(request_flags)
    if args.audit_only and requests_enabled:
        parser.error("--audit-only cannot be combined with model-request flags")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    try:
        evaluation_input = load_screening_report(args.report)
        if not requests_enabled:
            if args.resume:
                parser.error("--resume is valid only when both request flags are set")
            if args.openrouter_gateway_root is not None or args.base_url is not None:
                parser.error("audit-only mode does not accept live gateway arguments")
            existing_state = (
                args.output_root.expanduser().resolve() / args.tier / "run_state.json"
            )
            if existing_state.is_file():
                audit = audit_screening_run(
                    report_path=args.report,
                    output_root=args.output_root,
                    tier=args.tier,
                )
                print(
                    json.dumps(
                        {
                            "mode": "audit_only",
                            "would_request_model": False,
                            **audit,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                return 0
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "mode": "audit_only",
                        "would_request_model": False,
                        "protocol_class": PROTOCOL_CLASS,
                        "tier": args.tier,
                        "formal_scope_verified": False,
                        "question_count": EXPECTED_QUESTION_COUNT,
                        "rubric_nugget_count": EXPECTED_RUBRIC_NUGGET_COUNT,
                        "source_report_sha256": evaluation_input[
                            "source_report_sha256"
                        ],
                        "claim_scope": "two-conversation paired screening only",
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        if args.openrouter_gateway_root is None or not args.base_url:
            parser.error(
                "real judging requires --openrouter-gateway-root and --base-url"
            )
        if args.resume:
            state_path = (
                args.output_root.expanduser().resolve() / args.tier / "run_state.json"
            )
            if not state_path.is_file():
                raise ScreeningJudgeError("resume state is absent")
            stored_state = FULL_EVALUATOR.read_json(state_path)
            config = stored_state.get("config")
            if not isinstance(config, dict):
                raise ScreeningJudgeError("resume state has no judge config")
            validate_screening_config(config, evaluation_input)
            if config["tier"] != args.tier or config["max_tokens"] != args.max_tokens:
                raise ScreeningJudgeError("resume CLI differs from frozen judge config")
            binding = config.get("openrouter_gateway")
            if not isinstance(binding, dict):
                raise ScreeningJudgeError("resume state has no OpenRouter binding")
            try:
                FULL_EVALUATOR.openrouter_gateway_evidence.validate_binding(
                    binding,
                    result_root=args.openrouter_gateway_root,
                    base_url=args.base_url,
                )
            except (
                FULL_EVALUATOR.openrouter_gateway_evidence.GatewayEvidenceError
            ) as exc:
                raise ScreeningJudgeError(str(exc)) from exc
        else:
            try:
                binding = FULL_EVALUATOR.openrouter_gateway_evidence.capture_binding(
                    args.openrouter_gateway_root,
                    base_url=args.base_url,
                )
            except (
                FULL_EVALUATOR.openrouter_gateway_evidence.GatewayEvidenceError
            ) as exc:
                raise ScreeningJudgeError(str(exc)) from exc
            config = build_screening_config(
                evaluation_input,
                tier=args.tier,
                base_url=str(binding["base_url"]),
                max_tokens=args.max_tokens,
                openrouter_gateway=binding,
            )
        client = create_openai_client(base_url=str(binding["base_url"]))
        results = run_screening_judge(
            report_path=args.report,
            output_root=args.output_root,
            tier=args.tier,
            config=config,
            client=client,
            resume=args.resume,
        )
        print(
            json.dumps(
                {
                    "status": results["status"],
                    "protocol_class": PROTOCOL_CLASS,
                    "tier": args.tier,
                    "question_count": results["question_count"],
                    "rubric_nugget_count": results["rubric_nugget_count"],
                    "output_dir": str(
                        args.output_root.expanduser().resolve() / args.tier
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if results["status"] == "complete" else 1
    except ScreeningJudgeError as exc:
        parser.error(str(exc))


__all__ = [
    "FULL_EVALUATOR",
    "SCRIPT_PATH",
    "ScreeningJudgeError",
    "audit_screening_run",
    "build_screening_config",
    "create_openai_client",
    "load_screening_report",
    "main",
    "run_screening_judge",
    "validate_screening_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
