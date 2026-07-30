#!/usr/bin/env python3
"""Plan and execute isolated QA over completed GPT-5.6 window builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from scripts import gpt56_chunk_curve_qa_contract as contract
from scripts import run_gpt56_chunk_curve as build_runner
from scripts import controlled_locomo_answer_contract as answer_contract
from scripts import readonly_nativemem_control as readonly_control
from scripts.run_v88_gpt55_longmemeval import model_question as lme_model_question
from src.evaluation.durable_model_ledger import read_ledger


ROOT = Path(__file__).resolve().parents[1]
CONTROLLED_QA_PROXY = ROOT / "scripts" / "controlled_subscription_qa_proxy.py"
FORMAL_SCOPE_UNITS = {
    ("locomo", "conv-44"): 123,
    ("locomo", "conv-48"): 191,
    ("longmemeval-s", "2318644b"): 1,
    ("longmemeval-s", "gpt4_6dc9b45b"): 1,
    ("beam-100k", "100K-conv-1"): 20,
    ("beam-100k", "100K-conv-2"): 20,
}
FORMAL_TIERS = ("luna", "terra", "sol")
FORMAL_WINDOWS = (4, 8, 16, 32)
SESSION_GROUP_SIZES = {
    "locomo": {2, 4, 8, 16},
    "longmemeval-s": {2, 4, 8, 16},
    "beam-100k": {1, 2},
}


def formal_run_ids(tier: str, write_turns: int) -> dict[tuple[str, str], str]:
    if tier not in FORMAL_TIERS or write_turns not in FORMAL_WINDOWS:
        raise QARunnerError("formal QA tier or writer window is unsupported")
    return {
        ("locomo", "conv-44"): f"gpt56-locomo-conv-44-{tier}-w{write_turns}",
        ("locomo", "conv-48"): f"gpt56-locomo-conv-48-{tier}-w{write_turns}",
        ("longmemeval-s", "2318644b"): (
            f"gpt56-longmemeval-s-2318644b-{tier}-w{write_turns}"
        ),
        ("longmemeval-s", "gpt4_6dc9b45b"): (
            f"gpt56-longmemeval-s-gpt4-6dc9b45b-{tier}-w{write_turns}"
        ),
        ("beam-100k", "100K-conv-1"): (
            f"gpt56-beam-100k-100K-conv-1-{tier}-w{write_turns}"
        ),
        ("beam-100k", "100K-conv-2"): (
            f"gpt56-beam-100k-100K-conv-2-{tier}-w{write_turns}"
        ),
    }


FORMAL_RUN_IDS_BY_TIER = {
    tier: formal_run_ids(tier, 32) for tier in FORMAL_TIERS
}


class QARunnerError(RuntimeError):
    """The QA plan or execution request is invalid."""


def validate_formal_scope(
    *,
    bindings: Sequence[contract.RunBinding],
    validated_units: Sequence[dict[str, Any]],
) -> None:
    """Require one complete tier-window screening scope."""

    write_scopes = {binding.row.get("write_turns") for binding in bindings}
    if write_scopes == {"session"}:
        if len(bindings) != 2 or len(validated_units) != 2:
            raise QARunnerError("formal session-group QA requires exactly two runs")
        tiers = {str(binding.row.get("tier")) for binding in bindings}
        if len(tiers) != 1 or next(iter(tiers)) not in FORMAL_TIERS:
            raise QARunnerError("formal QA requires exactly one model tier")
        tier = next(iter(tiers))
        benchmarks = {str(binding.row.get("benchmark")) for binding in bindings}
        if len(benchmarks) != 1 or next(iter(benchmarks)) not in SESSION_GROUP_SIZES:
            raise QARunnerError("formal session-group QA requires one benchmark")
        benchmark = next(iter(benchmarks))
        group_sizes = {binding.row.get("session_group_size") for binding in bindings}
        if len(group_sizes) != 1 or next(iter(group_sizes)) not in SESSION_GROUP_SIZES[benchmark]:
            raise QARunnerError("formal QA requires one session group size")
        group_size = next(iter(group_sizes))
        expected_units = {
            unit_id: count
            for (unit_benchmark, unit_id), count in FORMAL_SCOPE_UNITS.items()
            if unit_benchmark == benchmark
        }
        observed_units: set[str] = set()
        for binding, validated in zip(bindings, validated_units, strict=True):
            unit_id = str(binding.row.get("unit_id"))
            questions = validated.get("questions")
            unit = validated.get("unit")
            expected_run_id = f"gpt56-{benchmark}-{unit_id}-{tier}-s{group_size}"
            if (
                unit_id in observed_units
                or unit_id not in expected_units
                or binding.run_id != expected_run_id
                or binding.row.get("model") != f"gpt-5.6-{tier}"
                or binding.row.get("reasoning_effort") != "none"
                or not isinstance(unit, dict)
                or unit.get("run_id") != binding.run_id
                or not isinstance(questions, list)
                or len(questions) != expected_units[unit_id]
            ):
                raise QARunnerError("formal session-group QA unit contract differs")
            observed_units.add(unit_id)
        if observed_units != set(expected_units):
            raise QARunnerError("formal session-group QA unit set differs")
        return

    if len(bindings) != 6 or len(validated_units) != 6:
        raise QARunnerError("formal QA requires exactly six runs")
    tiers = {str(binding.row.get("tier")) for binding in bindings}
    if len(tiers) != 1 or next(iter(tiers)) not in FORMAL_TIERS:
        raise QARunnerError("formal QA requires exactly one model tier")
    tier = next(iter(tiers))
    windows = {binding.row.get("write_turns") for binding in bindings}
    if len(windows) != 1 or next(iter(windows)) not in FORMAL_WINDOWS:
        raise QARunnerError("formal QA requires exactly one writer window")
    write_turns = next(iter(windows))
    expected_by_unit = formal_run_ids(tier, write_turns)
    expected_ids = set(expected_by_unit.values())
    actual_ids = [binding.run_id for binding in bindings]
    if len(set(actual_ids)) != 6 or set(actual_ids) != expected_ids:
        raise QARunnerError("formal QA run identities differ from the six-run scope")
    total_questions = 0
    observed_units: set[tuple[str, str]] = set()
    qualified_questions: set[str] = set()
    for binding, validated in zip(bindings, validated_units, strict=True):
        row = binding.row
        unit_key = (str(row.get("benchmark")), str(row.get("unit_id")))
        questions = validated.get("questions")
        unit = validated.get("unit")
        expected_run_id = expected_by_unit.get(unit_key)
        expected_question_count = FORMAL_SCOPE_UNITS.get(unit_key)
        if (
            expected_run_id != binding.run_id
            or unit_key in observed_units
            or row.get("write_turns") != write_turns
            or row.get("model") != f"gpt-5.6-{tier}"
            or row.get("reasoning_effort") != "none"
            or not isinstance(unit, dict)
            or unit.get("run_id") != binding.run_id
            or not isinstance(questions, list)
        ):
            raise QARunnerError("formal QA unit contract differs")
        if len(questions) != expected_question_count:
            raise QARunnerError("formal QA requires exactly 356 questions")
        observed_units.add(unit_key)
        total_questions += len(questions)
        for question in questions:
            question_id = (
                str(question.get("question_id", ""))
                if isinstance(question, dict)
                else ""
            )
            qualified = f"{unit_key[1]}::{question_id}"
            if not question_id or qualified in qualified_questions:
                raise QARunnerError("formal QA question identities differ")
            qualified_questions.add(qualified)
    if observed_units != set(FORMAL_SCOPE_UNITS):
        raise QARunnerError("formal QA unit set differs")
    if total_questions != 356:
        raise QARunnerError("formal QA requires exactly 356 questions")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=build_runner.DEFAULT_MATRIX)
    builds = parser.add_mutually_exclusive_group(required=True)
    builds.add_argument("--build-root", type=Path)
    builds.add_argument("--completed-runs-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--upstream", default="http://127.0.0.1:8205")
    parser.add_argument("--upstream-code-sha256")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--qa-run-id")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-model-requests", action="store_true")
    return parser.parse_args(argv)


def _select_completed_runs(
    args: argparse.Namespace,
    matrix: Path,
) -> list[contract.RunBinding]:
    return contract.select_completed_runs(
        matrix_path=matrix,
        build_root=(
            args.build_root.expanduser().resolve()
            if args.build_root is not None
            else None
        ),
        completed_runs_manifest=(
            args.completed_runs_manifest.expanduser().resolve()
            if args.completed_runs_manifest is not None
            else None
        ),
        run_ids=args.run_ids,
    )


def prepare_plan(args: argparse.Namespace) -> dict[str, Any]:
    matrix = args.matrix.expanduser().resolve()
    bindings = _select_completed_runs(args, matrix)
    units = [contract.validate_unit_binding(binding) for binding in bindings]
    return contract.build_preregistration(
        bindings=bindings,
        validated_units=units,
        matrix_path=matrix,
        upstream=args.upstream,
        upstream_code_sha256=args.upstream_code_sha256,
    )


def prepare_scope(
    args: argparse.Namespace,
) -> tuple[list[contract.RunBinding], list[dict[str, Any]], dict[str, Any]]:
    """Return the validated bindings, source units, and matching plan."""

    matrix = args.matrix.expanduser().resolve()
    bindings = _select_completed_runs(args, matrix)
    units = [contract.validate_unit_binding(binding) for binding in bindings]
    plan = contract.build_preregistration(
        bindings=bindings,
        validated_units=units,
        matrix_path=matrix,
        upstream=args.upstream,
        upstream_code_sha256=args.upstream_code_sha256,
    )
    return bindings, units, plan


def _default_qa_run_id(plan: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "protocol_id": plan.get("protocol_id"),
            "run_ids": plan.get("run_ids"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"{plan['protocol_id']}-{hashlib.sha256(payload).hexdigest()[:12]}"


def initialize_output(output_dir: Path, plan: dict[str, Any]) -> Path:
    """Create a fresh QA root and publish its immutable preregistration."""

    output = output_dir.expanduser().absolute()
    answer_contract.reject_symlink_components(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "preregistration.json"
    answer_contract.atomic_json_no_clobber(path, plan)
    return path


def _safe_component(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned[:180] or "unknown"


def _model_visible_question(
    *, benchmark: str, question: dict[str, Any]
) -> str:
    raw_question = str(
        question.get("question_text", question.get("question", ""))
    )
    if benchmark == "longmemeval-s":
        try:
            return lme_model_question(
                {
                    "question": raw_question,
                    "question_date": question["question_date"],
                }
            )
        except KeyError as exc:
            raise QARunnerError(
                "LongMemEval question is missing question_date"
            ) from exc
    return raw_question


def _record_from_result(
    *,
    binding: contract.RunBinding,
    question: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    original_question_id = str(question["question_id"])
    question_text = str(question.get("question_text", question.get("question", "")))
    benchmark = str(binding.row["benchmark"])
    record = {
        "run_id": binding.run_id,
        "benchmark": benchmark,
        "tier": binding.row["tier"],
        "unit_id": binding.row["unit_id"],
        "selection": binding.row.get("selection", {}),
        "session_group_size": binding.row.get("session_group_size"),
        "question_id": f"{binding.row['unit_id']}::{original_question_id}",
        "original_question_id": original_question_id,
        "question": question_text,
        "gold": question.get("gold", ""),
        "category": question.get("category"),
        "question_type": question.get("question_type"),
        "gold_field": question.get("gold_field"),
        "rubric": list(question.get("rubric_nuggets", [])),
        "answer": result["answer"]["text"],
        "result": result,
    }
    if benchmark == "longmemeval-s":
        record["question_date"] = str(question["question_date"])
        record["model_question"] = _model_visible_question(
            benchmark=benchmark,
            question=question,
        )
    return record


def _atomic_text_no_clobber(path: Path, text: str) -> None:
    """Publish a text artifact exactly once without replacing existing data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    answer_contract.reject_symlink_components(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _publish_json_or_validate(path: Path, value: Any) -> None:
    if path.is_file() and not path.is_symlink():
        if build_runner.read_json(path) != value:
            raise QARunnerError(f"existing JSON artifact differs: {path}")
        return
    answer_contract.atomic_json_no_clobber(path, value)


def _publish_text_or_validate(path: Path, text: str) -> None:
    if path.is_file() and not path.is_symlink():
        if path.read_text(encoding="utf-8") != text:
            raise QARunnerError(f"existing text artifact differs: {path}")
        return
    _atomic_text_no_clobber(path, text)


def publish_short_answer_evaluator_inputs(
    *,
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    tier: str,
) -> dict[str, Any]:
    """Publish exact evaluator inputs for one tier of LoCoMo and LME QA."""

    tier_records = [record for record in records if record["tier"] == tier]
    locomo_groups: dict[int, list[dict[str, Any]]] = {}
    lme_records: list[dict[str, str]] = []
    for record in tier_records:
        if record["benchmark"] == "locomo":
            sample_index = int(record["selection"]["sample_index"])
            locomo_groups.setdefault(sample_index, []).append(
                {
                    "question_id": record["original_question_id"],
                    "question": record["question"],
                    "gold": record["gold"],
                    "category": record["category"],
                    "answer": record["answer"],
                }
            )
        elif record["benchmark"] == "longmemeval-s":
            lme_records.append(
                {
                    "question_id": record["original_question_id"],
                    "hypothesis": record["answer"],
                }
            )

    evaluator_root = output_dir.expanduser().absolute() / tier / "evaluator_inputs"
    locomo_paths: list[Path] = []
    for sample_index, payload in sorted(locomo_groups.items()):
        path = evaluator_root / "locomo" / f"sample{sample_index}_questions.json"
        _publish_json_or_validate(path, payload)
        locomo_paths.append(path)

    lme_path = evaluator_root / "longmemeval-s" / "hypotheses.jsonl"
    lme_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in lme_records
    )
    _publish_text_or_validate(lme_path, lme_text)
    return {"locomo": locomo_paths, "longmemeval-s": lme_path}


def publish_beam_predictions(
    *,
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    tier: str,
) -> Path:
    """Publish the strict BEAM screening-subset prediction input."""

    predictions = [
        {
            "question_id": str(record["original_question_id"]),
            "answer": str(record["answer"]),
        }
        for record in records
        if record["tier"] == tier and record["benchmark"] == "beam-100k"
    ]
    ids = [record["question_id"] for record in predictions]
    if not predictions or len(ids) != len(set(ids)):
        raise QARunnerError("BEAM predictions must be non-empty with unique IDs")
    path = (
        output_dir.expanduser().absolute()
        / tier
        / "evaluator_inputs"
        / "beam-100k"
        / "predictions.jsonl"
    )
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in predictions
    )
    _publish_text_or_validate(path, text)
    return path


def _execute_readonly_question(
    *,
    binding: contract.RunBinding,
    validated_unit: dict[str, Any],
    question: dict[str, Any],
    output_dir: Path,
    qa_run_id: str,
    completion_resource: Any,
    answer_client: Any,
    tokenizer: Any,
    proxy_log: Path | None,
    formal: bool,
    attempt: int = 1,
) -> dict[str, Any]:
    """Execute one benchmark question against one frozen memory tree."""

    original_question_id = str(question["question_id"])
    question_id = f"{binding.row['unit_id']}::{original_question_id}"
    benchmark = str(binding.row["benchmark"])
    question_text = _model_visible_question(
        benchmark=benchmark,
        question=question,
    )
    if not question_text:
        raise QARunnerError("question text is empty")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise QARunnerError("attempt must be a positive integer")
    evidence = question.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    gold_source_ids = (
        readonly_control.expand_source_specs(evidence)
        if binding.row["benchmark"] == "locomo" and isinstance(evidence, list)
        else []
    )
    prompt_spec = contract.ANSWER_PROMPTS[benchmark]
    prompt_builder = (
        contract.assemble_beam_answer_prompt
        if benchmark == "beam-100k"
        else answer_contract.assemble_answer_prompt
    )
    execution_run_id = f"{qa_run_id}:attempt-{attempt:04d}"
    attempt_manifest = {
        "schema_version": 1,
        "qa_run_id": qa_run_id,
        "execution_run_id": execution_run_id,
        "attempt": attempt,
        "question_id": question_id,
        "benchmark": benchmark,
        "answer_prompt_kind": str(prompt_spec["kind"]),
        "answer_prompt_template_sha256": str(prompt_spec["template_sha256"]),
    }
    artifact_dir = (
        output_dir.expanduser().absolute()
        / str(binding.row["tier"])
        / str(binding.row["benchmark"])
        / _safe_component(binding.row["unit_id"])
        / "questions"
        / _safe_component(question_id)
        / f"attempt-{attempt:04d}"
    )
    result = readonly_control.execute_question(
        artifact_dir=artifact_dir,
        run_id=execution_run_id,
        method=contract.PROTOCOL_ID,
        condition="dual_source",
        memory_root=binding.run_dir / "memory",
        turn_index=readonly_control.build_turn_index(
            validated_unit["conversation"]
        ),
        question_id=question_id,
        question=question_text,
        gold_source_ids=gold_source_ids,
        source_recall_eligible=bool(gold_source_ids),
        completion_resource=completion_resource,
        answer_client=answer_client,
        tokenizer=tokenizer,
        formal=formal,
        proxy_log=proxy_log,
        budget_tokens=20_000,
        max_rounds=12,
        answer_completion_reservation_tokens=contract.ANSWER_MAX_TOKENS,
        answer_prompt_builder=prompt_builder,
        answer_prompt_kind=str(prompt_spec["kind"]),
        answer_prompt_template_sha256=str(prompt_spec["template_sha256"]),
        attempt_manifest=attempt_manifest,
    )
    return _record_from_result(binding=binding, question=question, result=result)


def execute_short_answer_question(**kwargs: Any) -> dict[str, Any]:
    """Execute one LoCoMo/LongMemEval question against frozen memory."""

    binding = kwargs.get("binding")
    if not isinstance(binding, contract.RunBinding) or binding.row["benchmark"] not in {
        "locomo",
        "longmemeval-s",
    }:
        raise QARunnerError("short-answer execution only supports LoCoMo/LME")
    return _execute_readonly_question(**kwargs)


def execute_beam_question(**kwargs: Any) -> dict[str, Any]:
    """Execute one BEAM screening question against frozen memory."""

    binding = kwargs.get("binding")
    if (
        not isinstance(binding, contract.RunBinding)
        or binding.row["benchmark"] != "beam-100k"
    ):
        raise QARunnerError("BEAM execution requires a beam-100k binding")
    return _execute_readonly_question(**kwargs)


def _question_root(
    output_dir: Path,
    binding: contract.RunBinding,
    original_question_id: object,
) -> Path:
    qualified = f"{binding.row['unit_id']}::{original_question_id}"
    return (
        output_dir.expanduser().absolute()
        / str(binding.row["tier"])
        / str(binding.row["benchmark"])
        / _safe_component(binding.row["unit_id"])
        / "questions"
        / _safe_component(qualified)
    )


def _validate_completed_record(
    record: dict[str, Any],
    path: Path,
    *,
    binding: contract.RunBinding,
    original_question_id: str,
    expected_question: dict[str, Any],
    formal: bool,
) -> dict[str, Any]:
    expected = {
        "run_id": binding.run_id,
        "benchmark": binding.row["benchmark"],
        "tier": binding.row["tier"],
        "unit_id": binding.row["unit_id"],
        "original_question_id": original_question_id,
        "question_id": f"{binding.row['unit_id']}::{original_question_id}",
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise QARunnerError(f"completed question record binding differs: {path}")
    expected_payload = {
        "selection": binding.row.get("selection", {}),
        "question": str(
            expected_question.get(
                "question_text", expected_question.get("question", "")
            )
        ),
        "gold": expected_question.get("gold", ""),
        "category": expected_question.get("category"),
        "question_type": expected_question.get("question_type"),
        "gold_field": expected_question.get("gold_field"),
        "rubric": list(expected_question.get("rubric_nuggets", [])),
    }
    if any(record.get(key) != value for key, value in expected_payload.items()):
        raise QARunnerError(f"completed question payload differs: {path}")
    result = record.get("result")
    if (
        not isinstance(result, dict)
        or result.get("status") != "complete"
        or result.get("memory", {}).get("unchanged") is not True
    ):
        raise QARunnerError(f"completed question record is invalid: {path}")
    result_contract = {
        "method": contract.PROTOCOL_ID,
        "condition": "dual_source",
        "question_id": expected["question_id"],
    }
    answer = result.get("answer", {})
    retrieval = result.get("retrieval", {})
    budget = result.get("budget", {})
    prompt = result.get("prompt", {})
    prompt_spec = contract.ANSWER_PROMPTS[str(binding.row["benchmark"])]
    if (
        any(result.get(key) != value for key, value in result_contract.items())
        or answer.get("requested_model") != readonly_control.EXPECTED_MODEL
        or answer.get("response_model") != readonly_control.EXPECTED_MODEL
        or answer.get("text") != record.get("answer")
        or (
            formal
            and answer.get("unsupported_parameters") not in ([], ())
        )
        or retrieval.get("model") != readonly_control.EXPECTED_MODEL
        or budget.get("configured_tokens") != 20_000
        or prompt.get("answer_completion_reservation_tokens")
        not in contract.ACCEPTED_ANSWER_MAX_TOKENS
        or prompt.get("kind") != prompt_spec["kind"]
        or prompt.get("template_sha256") != prompt_spec["template_sha256"]
    ):
        raise QARunnerError(f"completed question result contract differs: {path}")
    attempt_results = [
        candidate
        for candidate in path.parent.glob("attempt-*/result.json")
        if candidate.is_file() and not candidate.is_symlink()
    ]
    matching_attempts = [
        candidate
        for candidate in attempt_results
        if build_runner.read_json(candidate) == result
    ]
    if len(matching_attempts) != 1:
        raise QARunnerError(f"completed question result artifact differs: {path}")
    current = readonly_control.memory_descriptor(binding.run_dir / "memory")
    recorded_after = result["memory"].get("after", {})
    if recorded_after.get("sha256") != current.get("sha256"):
        raise QARunnerError(f"completed question memory binding differs: {path}")
    return record


def _load_completed_record(
    path: Path,
    *,
    binding: contract.RunBinding,
    original_question_id: str,
    expected_question: dict[str, Any],
    formal: bool,
) -> dict[str, Any]:
    record = build_runner.read_json(path)
    if not isinstance(record, dict):
        raise QARunnerError(f"completed question record is not an object: {path}")
    return _validate_completed_record(
        record,
        path,
        binding=binding,
        original_question_id=original_question_id,
        expected_question=expected_question,
        formal=formal,
    )


def _retryable_error_reason(
    *,
    status_code: object,
    error_text: str,
) -> str | None:
    if status_code != 500:
        return None
    lowered = error_text.lower()
    non_transient_markers = (
        "usage_limit_reached",
        "unauthorized",
        "authentication",
        "invalid_api_key",
        "http 401",
        "http 403",
        "http 429",
    )
    if any(marker in lowered for marker in non_transient_markers):
        return None
    if "upstream completed with empty output" in lowered:
        return contract.EMPTY_OUTPUT_RETRY_REASON
    if (
        "upstream stream failed:" in lowered
        and re.search(
            r'"code"\s*:\s*"(?:server_error|server_is_overloaded)"',
            error_text,
            flags=re.IGNORECASE,
        )
        is not None
    ):
        return contract.TRANSIENT_PROVIDER_SERVER_ERROR_RETRY_REASON
    if (
        "upstream failed after retries:" in lowered
        and "proxyerror" in lowered
        and (
            "unable to connect to proxy" in lowered
            or "remote end closed connection" in lowered
            or "connection reset by peer" in lowered
        )
    ):
        return contract.TRANSIENT_PROVIDER_SERVER_ERROR_RETRY_REASON
    if (
        "upstream failed after retries:" in lowered
        and (
            "connectionerror" in lowered
            or "readtimeout" in lowered
        )
        and "read timed out" in lowered
    ):
        return contract.TRANSIENT_PROVIDER_SERVER_ERROR_RETRY_REASON
    return None


def _retryable_question_error_reason(exc: Exception) -> str | None:
    if "failed answer request has unknown provider attempts" in str(exc).lower():
        return contract.TRANSIENT_PROVIDER_SERVER_ERROR_RETRY_REASON
    return _retryable_error_reason(
        status_code=getattr(exc, "status_code", None),
        error_text=str(exc),
    )


def _failed_attempt_evidence(
    attempt_root: Path,
    *,
    expected_reason: str,
    fallback_status_code: int | None = None,
    formal: bool = False,
) -> dict[str, Any]:
    retrieval_path = attempt_root / "retrieval_model_ledger.jsonl"
    if retrieval_path.is_file() and not retrieval_path.is_symlink():
        failed = []
        for record in read_ledger(retrieval_path):
            if record.get("event") != "model_call_failed":
                continue
            error = str(record.get("error", ""))
            match = re.search(r"Error code:\s*(\d+)", error)
            status_code = (
                int(match.group(1))
                if match is not None
                else fallback_status_code
            )
            if (
                _retryable_error_reason(
                    status_code=status_code,
                    error_text=error,
                )
                == expected_reason
            ):
                failed.append(record)
        if len(failed) == 1:
            record = failed[0]
            proxy = record.get("proxy_evidence")
            events = (
                proxy.get("events")
                if isinstance(proxy, dict)
                else None
            )
            event = (
                events[0]
                if isinstance(events, list)
                and len(events) == 1
                and isinstance(events[0], dict)
                else None
            )
            if formal and (
                not isinstance(event, dict)
                or event.get("status") != "error"
                or event.get("http_status") != 500
                or event.get("upstream_http_attempts") is not None
                or event.get("logical_call_id") != record.get("logical_call_id")
                or not isinstance(event.get("event_id"), str)
                or not event.get("event_id")
            ):
                raise QARunnerError("formal retry proxy evidence differs")
            error = str(record.get("error", ""))
            match = re.search(r"Error code:\s*(\d+)", error)
            status_code = (
                int(match.group(1))
                if match is not None
                else event.get("http_status")
                if isinstance(event, dict)
                else fallback_status_code
            )
            return {
                "stage": "retrieval",
                "ledger": retrieval_path.name,
                "ledger_sha256": build_runner.sha256_file(retrieval_path),
                "logical_call_id": record.get("logical_call_id"),
                "proxy_event_id": (
                    event.get("event_id") if isinstance(event, dict) else None
                ),
                "http_status": status_code,
                "upstream_http_attempts": (
                    event.get("upstream_http_attempts")
                    if isinstance(event, dict)
                    else None
                ),
                "error_sha256": answer_contract.sha256_bytes(
                    error.encode("utf-8")
                ),
            }
    answer_path = attempt_root / "answer_ledger.jsonl"
    if answer_path.is_file() and not answer_path.is_symlink():
        raw = [
            json.loads(line)
            for line in answer_path.read_text(encoding="utf-8").splitlines()
        ]
        run_ids = {
            str(record.get("run_id", ""))
            for record in raw
            if isinstance(record, dict)
        }
        if len(run_ids) == 1:
            records = answer_contract.audit_ledger(
                answer_path,
                expected_run_id=next(iter(run_ids)),
            )
            failed = []
            for record in records:
                payload = record.get("payload")
                if (
                    record.get("event") != "physical_http_attempt_finished"
                    or not isinstance(payload, dict)
                ):
                    continue
                error = str(payload.get("error", ""))
                reason = (
                    contract.EMPTY_OUTPUT_RETRY_REASON
                    if payload.get("status") == "retryable_empty_output"
                    and "upstream completed with empty output" in error.lower()
                    else _retryable_error_reason(
                        status_code=payload.get("http_status"),
                        error_text=error,
                    )
                    if payload.get("status") == "unaccountable_error"
                    else None
                )
                if reason == expected_reason:
                    failed.append(record)
            if len(failed) == 1:
                payload = failed[0]["payload"]
                error = str(payload.get("error", ""))
                if (
                    payload.get("http_status") != 500
                    or payload.get("upstream_http_attempts") is not None
                    or (
                        formal
                        and (
                            not isinstance(payload.get("proxy_event_id"), str)
                            or not payload.get("proxy_event_id")
                        )
                    )
                ):
                    detail = (
                        "formal retry proxy evidence differs"
                        if formal or expected_reason != contract.EMPTY_OUTPUT_RETRY_REASON
                        else "retry authorization lacks exact empty-output evidence: "
                        f"{attempt_root}"
                    )
                    raise QARunnerError(
                        detail
                    )
                return {
                    "stage": "answer",
                    "ledger": answer_path.name,
                    "ledger_sha256": build_runner.sha256_file(answer_path),
                    "logical_call_id": next(iter(run_ids)),
                    "proxy_event_id": payload.get("proxy_event_id"),
                    "http_status": payload.get(
                        "http_status", fallback_status_code
                    ),
                    "upstream_http_attempts": payload.get(
                        "upstream_http_attempts"
                    ),
                    "error_sha256": answer_contract.sha256_bytes(
                        error.encode("utf-8")
                    ),
                }
            if (
                expected_reason == contract.EMPTY_OUTPUT_RETRY_REASON
                and any(
                    record.get("event") == "physical_http_attempt_finished"
                    and isinstance(record.get("payload"), dict)
                    and record["payload"].get("status")
                    == "retryable_empty_output"
                    for record in records
                )
            ):
                raise QARunnerError(
                    "retry authorization lacks exact empty-output evidence: "
                    f"{attempt_root}"
                )
    raise QARunnerError(
        f"retry authorization lacks exact supported evidence: {attempt_root}"
    )


def _validate_retry_failure_binding(
    *,
    failure: dict[str, Any],
    manifest: dict[str, Any],
    qa_run_id: str,
    question_id: str,
    attempt: int,
) -> None:
    expected_execution_run_id = f"{qa_run_id}:attempt-{attempt:04d}"
    stage = failure.get("stage")
    logical_call_id = failure.get("logical_call_id")
    expected_prefix = (
        f"{expected_execution_run_id}:retrieval:{question_id}:"
        if stage == "retrieval"
        else f"{expected_execution_run_id}:{question_id}:"
        if stage == "answer"
        else None
    )
    if (
        manifest.get("qa_run_id") != qa_run_id
        or manifest.get("question_id") != question_id
        or manifest.get("attempt") != attempt
        or manifest.get("execution_run_id") != expected_execution_run_id
        or not isinstance(logical_call_id, str)
        or not isinstance(expected_prefix, str)
        or not logical_call_id.startswith(expected_prefix)
    ):
        raise QARunnerError("retry authorization failure binding differs")


def _validate_attempt_manifest_contract(
    *,
    attempt_root: Path,
    qa_run_id: str,
    question_id: str,
    attempt: int,
    benchmark: str,
) -> dict[str, Any]:
    path = attempt_root / "attempt_manifest.json"
    if path.is_symlink() or not path.is_file():
        raise QARunnerError(f"attempt manifest contract differs: {path}")
    value = build_runner.read_json(path)
    prompt_spec = contract.ANSWER_PROMPTS.get(benchmark)
    expected = {
        "schema_version": 1,
        "qa_run_id": qa_run_id,
        "execution_run_id": f"{qa_run_id}:attempt-{attempt:04d}",
        "attempt": attempt,
        "question_id": question_id,
        "benchmark": benchmark,
        "answer_prompt_kind": (
            prompt_spec.get("kind") if isinstance(prompt_spec, dict) else None
        ),
        "answer_prompt_template_sha256": (
            prompt_spec.get("template_sha256")
            if isinstance(prompt_spec, dict)
            else None
        ),
    }
    if value != expected:
        raise QARunnerError(f"attempt manifest contract differs: {path}")
    return value


def _retry_authorization_payload(
    *,
    attempt_root: Path,
    qa_run_id: str,
    question_id: str,
    attempt: int,
    exc: Exception,
    formal: bool,
) -> dict[str, Any]:
    reason = _retryable_question_error_reason(exc)
    if reason is None or attempt >= contract.QUESTION_MAX_ATTEMPTS:
        raise QARunnerError("retry authorization request is not allowed")
    manifest = attempt_root / "attempt_manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise QARunnerError("retry authorization lacks an attempt manifest")
    manifest_value = build_runner.read_json(manifest)
    if not isinstance(manifest_value, dict):
        raise QARunnerError("retry authorization attempt manifest differs")
    failure = _failed_attempt_evidence(
        attempt_root,
        expected_reason=reason,
        fallback_status_code=getattr(exc, "status_code", None),
        formal=formal,
    )
    _validate_retry_failure_binding(
        failure=failure,
        manifest=manifest_value,
        qa_run_id=qa_run_id,
        question_id=question_id,
        attempt=attempt,
    )
    if (
        failure.get("http_status") != 500
        or failure.get("upstream_http_attempts") is not None
    ):
        raise QARunnerError("retry authorization failure evidence differs")
    return {
        "schema_version": 1,
        "qa_run_id": qa_run_id,
        "question_id": question_id,
        "from_attempt": attempt,
        "to_attempt": attempt + 1,
        "reason": reason,
        "attempt_manifest_sha256": build_runner.sha256_file(manifest),
        "failure": failure,
    }


def _validate_retry_authorization(
    *,
    attempt_root: Path,
    qa_run_id: str,
    question_id: str,
    attempt: int,
    formal: bool,
) -> dict[str, Any]:
    path = attempt_root / "retry_authorization.json"
    if path.is_symlink() or not path.is_file():
        raise QARunnerError(
            f"retry authorization is missing: {attempt_root}"
        )
    recorded = build_runner.read_json(path)
    reason = recorded.get("reason") if isinstance(recorded, dict) else None
    if reason not in contract.RETRYABLE_QUESTION_ERRORS:
        raise QARunnerError("retry authorization reason differs")
    recorded_failure = (
        recorded.get("failure") if isinstance(recorded, dict) else None
    )
    fallback_status_code = (
        recorded_failure.get("http_status")
        if isinstance(recorded_failure, dict)
        else None
    )
    manifest = attempt_root / "attempt_manifest.json"
    manifest_value = build_runner.read_json(manifest)
    failure = _failed_attempt_evidence(
        attempt_root,
        expected_reason=str(reason),
        fallback_status_code=fallback_status_code,
        formal=formal,
    )
    if not isinstance(manifest_value, dict):
        raise QARunnerError("retry authorization attempt manifest differs")
    _validate_retry_failure_binding(
        failure=failure,
        manifest=manifest_value,
        qa_run_id=qa_run_id,
        question_id=question_id,
        attempt=attempt,
    )
    if (
        failure.get("http_status") != 500
        or failure.get("upstream_http_attempts") is not None
        or (
            formal
            and (
                not isinstance(failure.get("proxy_event_id"), str)
                or not failure.get("proxy_event_id")
            )
        )
    ):
        raise QARunnerError("retry authorization failure evidence differs")
    expected = {
        "schema_version": 1,
        "qa_run_id": qa_run_id,
        "question_id": question_id,
        "from_attempt": attempt,
        "to_attempt": attempt + 1,
        "reason": reason,
        "attempt_manifest_sha256": build_runner.sha256_file(manifest),
        "failure": failure,
    }
    if recorded != expected:
        raise QARunnerError(
            f"retry authorization differs: {attempt_root}"
        )
    return recorded


def _existing_attempt_numbers(root: Path) -> list[int]:
    attempts: list[int] = []
    if not root.is_dir():
        return attempts
    for path in root.glob("attempt-*"):
        match = re.fullmatch(r"attempt-(\d{4})", path.name)
        if match is None:
            raise QARunnerError(f"question attempt path differs: {path}")
        attempt = int(match.group(1))
        if not 1 <= attempt <= contract.QUESTION_MAX_ATTEMPTS:
            raise QARunnerError(f"question attempt limit differs: {path}")
        attempts.append(attempt)
    if len(attempts) != len(set(attempts)):
        raise QARunnerError(f"question attempt identity is duplicated: {root}")
    return sorted(attempts)


def execute_pending_questions(
    *,
    bindings: Sequence[contract.RunBinding],
    validated_units: Sequence[dict[str, Any]],
    output_dir: Path,
    qa_run_id: str,
    completion_resource: Any,
    answer_client: Any,
    tokenizer: Any,
    proxy_log: Path | None,
    formal: bool,
) -> list[dict[str, Any]]:
    """Execute missing questions and reuse only validated complete records."""

    if len(bindings) != len(validated_units):
        raise QARunnerError("bindings and validated units must align")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for binding, validated in zip(bindings, validated_units, strict=True):
        unit = validated.get("unit")
        questions = validated.get("questions")
        if (
            not isinstance(unit, dict)
            or unit.get("run_id") != binding.run_id
            or not isinstance(questions, list)
        ):
            raise QARunnerError("validated unit does not match its binding")
        for question in questions:
            if not isinstance(question, dict):
                raise QARunnerError("validated question is not an object")
            original_question_id = str(question.get("question_id", ""))
            qualified = f"{binding.row['unit_id']}::{original_question_id}"
            if not original_question_id or qualified in seen:
                raise QARunnerError("question IDs must be non-empty and globally unique")
            seen.add(qualified)
            root = _question_root(output_dir, binding, original_question_id)
            record_path = root / "record.json"
            if record_path.is_file():
                loaded_record = _load_completed_record(
                    record_path,
                    binding=binding,
                    original_question_id=original_question_id,
                    expected_question=question,
                    formal=formal,
                )
                result = loaded_record.get("result")
                execution_run_id = (
                    result.get("run_id") if isinstance(result, dict) else None
                )
                accepted_match = (
                    re.fullmatch(
                        re.escape(qa_run_id) + r":attempt-(\d{4})",
                        execution_run_id,
                    )
                    if isinstance(execution_run_id, str)
                    else None
                )
                if accepted_match is None:
                    raise QARunnerError(
                        f"completed record attempt identity differs: {record_path}"
                    )
                accepted_attempt = int(accepted_match.group(1))
                attempts = _existing_attempt_numbers(root)
                if attempts != list(range(1, accepted_attempt + 1)):
                    raise QARunnerError(
                        f"completed question attempt chain differs: {root}"
                    )
                for manifest_attempt in range(1, accepted_attempt + 1):
                    _validate_attempt_manifest_contract(
                        attempt_root=(
                            root / f"attempt-{manifest_attempt:04d}"
                        ),
                        qa_run_id=qa_run_id,
                        question_id=qualified,
                        attempt=manifest_attempt,
                        benchmark=str(binding.row["benchmark"]),
                    )
                for previous_attempt in range(1, accepted_attempt):
                    _validate_retry_authorization(
                        attempt_root=(
                            root / f"attempt-{previous_attempt:04d}"
                        ),
                        qa_run_id=qa_run_id,
                        question_id=qualified,
                        attempt=previous_attempt,
                        formal=formal,
                    )
                if (
                    root
                    / f"attempt-{accepted_attempt:04d}"
                    / "retry_authorization.json"
                ).exists():
                    raise QARunnerError(
                        f"accepted attempt has retry authorization: {record_path}"
                    )
                records.append(loaded_record)
                continue
            attempts = _existing_attempt_numbers(root)
            completed_attempts: list[tuple[Path, dict[str, Any]]] = []
            if root.is_dir():
                for candidate in sorted(root.glob("attempt-*/result.json")):
                    if not candidate.is_file() or candidate.is_symlink():
                        continue
                    match = re.fullmatch(
                        r"attempt-(\d{4})", candidate.parent.name
                    )
                    candidate_attempt = int(match.group(1)) if match else 0
                    if not 1 <= candidate_attempt <= contract.QUESTION_MAX_ATTEMPTS:
                        raise QARunnerError(
                            f"question attempt limit differs: {candidate.parent}"
                        )
                    value = build_runner.read_json(candidate)
                    if isinstance(value, dict) and value.get("status") == "complete":
                        completed_attempts.append((candidate, value))
            if len(completed_attempts) > 1:
                raise QARunnerError(
                    f"multiple complete attempts exist without a record: {root}"
                )
            if completed_attempts:
                accepted_path = completed_attempts[0][0]
                accepted_match = re.fullmatch(
                    r"attempt-(\d{4})", accepted_path.parent.name
                )
                if accepted_match is None:
                    raise QARunnerError(
                        f"completed question attempt path differs: {accepted_path}"
                    )
                accepted_attempt = int(accepted_match.group(1))
                accepted_result = completed_attempts[0][1]
                if (
                    accepted_result.get("run_id")
                    != f"{qa_run_id}:attempt-{accepted_attempt:04d}"
                ):
                    raise QARunnerError(
                        f"completed result attempt identity differs: {accepted_path}"
                    )
                if attempts != list(range(1, accepted_attempt + 1)):
                    raise QARunnerError(
                        f"completed question attempt chain differs: {root}"
                    )
                for manifest_attempt in range(1, accepted_attempt + 1):
                    _validate_attempt_manifest_contract(
                        attempt_root=(
                            root / f"attempt-{manifest_attempt:04d}"
                        ),
                        qa_run_id=qa_run_id,
                        question_id=qualified,
                        attempt=manifest_attempt,
                        benchmark=str(binding.row["benchmark"]),
                    )
                for previous_attempt in range(1, accepted_attempt):
                    _validate_retry_authorization(
                        attempt_root=(
                            root / f"attempt-{previous_attempt:04d}"
                        ),
                        qa_run_id=qa_run_id,
                        question_id=qualified,
                        attempt=previous_attempt,
                        formal=formal,
                    )
                if (
                    accepted_path.parent / "retry_authorization.json"
                ).exists():
                    raise QARunnerError(
                        f"accepted attempt has retry authorization: {accepted_path}"
                    )
                recovered = _record_from_result(
                    binding=binding,
                    question=question,
                    result=accepted_result,
                )
                _validate_completed_record(
                    recovered,
                    record_path,
                    binding=binding,
                    original_question_id=original_question_id,
                    expected_question=question,
                    formal=formal,
                )
                answer_contract.atomic_json_no_clobber(record_path, recovered)
                records.append(recovered)
                continue
            if attempts:
                if attempts != list(range(1, max(attempts) + 1)):
                    raise QARunnerError(
                        f"question attempt chain differs: {root}"
                    )
                latest = max(attempts)
                authorization: dict[str, Any] | None = None
                for previous_attempt in range(1, latest + 1):
                    authorization = _validate_retry_authorization(
                        attempt_root=(
                            root / f"attempt-{previous_attempt:04d}"
                        ),
                        qa_run_id=qa_run_id,
                        question_id=qualified,
                        attempt=previous_attempt,
                        formal=formal,
                    )
                assert authorization is not None
                for manifest_attempt in range(1, latest + 1):
                    _validate_attempt_manifest_contract(
                        attempt_root=(
                            root / f"attempt-{manifest_attempt:04d}"
                        ),
                        qa_run_id=qa_run_id,
                        question_id=qualified,
                        attempt=manifest_attempt,
                        benchmark=str(binding.row["benchmark"]),
                    )
                attempt = int(authorization["to_attempt"])
            else:
                attempt = 1
            if attempt > contract.QUESTION_MAX_ATTEMPTS:
                raise QARunnerError(
                    f"question attempt limit exhausted: {qualified}"
                )
            executor = (
                execute_beam_question
                if binding.row["benchmark"] == "beam-100k"
                else execute_short_answer_question
            )
            while True:
                try:
                    record = executor(
                        binding=binding,
                        validated_unit=validated,
                        question=question,
                        output_dir=output_dir,
                        qa_run_id=qa_run_id,
                        completion_resource=completion_resource,
                        answer_client=answer_client,
                        tokenizer=tokenizer,
                        proxy_log=proxy_log,
                        formal=formal,
                        attempt=attempt,
                    )
                except Exception as exc:
                    retry_reason = _retryable_question_error_reason(exc)
                    if (
                        retry_reason is None
                        or attempt >= contract.QUESTION_MAX_ATTEMPTS
                    ):
                        raise
                    attempt_root = root / f"attempt-{attempt:04d}"
                    authorization = _retry_authorization_payload(
                        attempt_root=attempt_root,
                        qa_run_id=qa_run_id,
                        question_id=qualified,
                        attempt=attempt,
                        exc=exc,
                        formal=formal,
                    )
                    answer_contract.atomic_json_no_clobber(
                        attempt_root / "retry_authorization.json",
                        authorization,
                    )
                    attempt += 1
                    continue
                break
            answer_contract.atomic_json_no_clobber(record_path, record)
            records.append(record)
    return records


def start_qa_proxy(
    *,
    output_dir: Path,
    python: Path,
    upstream: str,
    upstream_code_sha256: str,
    run_id: str,
) -> tuple[subprocess.Popen[Any], dict[str, Any], Any]:
    """Start one fresh exclusive child proxy and verify its health contract."""

    proxy_root = output_dir.expanduser().absolute() / "proxy"
    proxy_root.mkdir(parents=True, exist_ok=True)
    launches = [
        int(match.group(1))
        for path in proxy_root.glob("invocation-*")
        if (match := re.fullmatch(r"invocation-(\d{4})", path.name))
    ]
    invocation = proxy_root / f"invocation-{max(launches, default=0) + 1:04d}"
    invocation.mkdir(exist_ok=False)
    ready = invocation / "ready.json"
    log = invocation / "requests.jsonl"
    process_log_path = invocation / "process.log"
    process_log = process_log_path.open("x", encoding="utf-8")
    process = subprocess.Popen(
        [
            str(python),
            str(CONTROLLED_QA_PROXY),
            "--port",
            "0",
            "--upstream",
            upstream,
            "--upstream-code-sha256",
            upstream_code_sha256,
            "--log",
            str(log),
            "--ready",
            str(ready),
            "--run-id",
            run_id,
        ],
        cwd=ROOT,
        stdout=process_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + 30
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise QARunnerError("exclusive QA proxy exited before readiness")
            if ready.is_file():
                metadata = build_runner.read_json(ready)
                if (
                    metadata.get("run_id") == run_id
                    and metadata.get("upstream_code_sha256") == upstream_code_sha256
                ):
                    try:
                        health_url = str(metadata["base_url"]).removesuffix("/v1") + "/healthz"
                        health = json.loads(opener.open(health_url, timeout=5).read())
                    except Exception:  # noqa: BLE001
                        health = None
                    if isinstance(health, dict) and health.get("status") == "ok":
                        record = {
                            "run_id": run_id,
                            "pid": process.pid,
                            "base_url": metadata["base_url"],
                            "upstream": metadata["upstream"],
                            "upstream_code_sha256": upstream_code_sha256,
                            "ready_path": str(ready),
                            "log_path": str(log),
                            "process_log_path": str(process_log_path),
                            "started_at": metadata["started_at"],
                            "health": health,
                        }
                        start_path = invocation / "start.json"
                        answer_contract.atomic_json_no_clobber(start_path, record)
                        record["start_path"] = str(start_path)
                        return process, record, process_log
            time.sleep(0.05)
        raise QARunnerError("exclusive QA proxy did not become healthy")
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        process_log.close()
        raise


def recover_stale_qa_proxies(
    *,
    output_dir: Path,
    run_id: str,
) -> list[dict[str, Any]]:
    """Terminate verified orphaned child proxies from interrupted invocations."""

    proxy_root = output_dir.expanduser().absolute() / "proxy"
    recovered: list[dict[str, Any]] = []
    if not proxy_root.is_dir():
        return recovered
    for ready_path in sorted(proxy_root.glob("invocation-*/ready.json")):
        invocation = ready_path.parent
        recovery_path = invocation / "recovery.json"
        if (invocation / "stop.json").is_file() or recovery_path.is_file():
            continue
        ready = build_runner.read_json(ready_path)
        if ready.get("run_id") != run_id:
            raise QARunnerError(f"stale proxy run ID differs: {ready_path}")
        pid = ready.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
            raise QARunnerError(f"stale proxy PID is invalid: {ready_path}")
        command = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = "already_exited"
        if command:
            expected_parts = (
                str(CONTROLLED_QA_PROXY),
                str(ready_path),
                f"--run-id {run_id}",
            )
            if any(part not in command for part in expected_parts):
                raise QARunnerError(
                    f"refusing to terminate PID with a different command: {pid}"
                )
            os.killpg(pid, signal.SIGTERM)
            status = "terminated_stale_proxy"
        record = {
            "run_id": run_id,
            "pid": pid,
            "status": status,
            "ready_path": str(ready_path),
            "ready_sha256": build_runner.sha256_file(ready_path),
            "recovered_at": answer_contract.utc_now(),
        }
        answer_contract.atomic_json_no_clobber(recovery_path, record)
        record["recovery_path"] = str(recovery_path)
        recovered.append(record)
    return recovered


def stop_qa_proxy(
    *,
    process: subprocess.Popen[Any],
    record: dict[str, Any],
    process_log: Any,
) -> dict[str, Any]:
    """Stop an exclusive child proxy and publish its immutable stop record."""

    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    process_log.close()
    stopped = {
        **record,
        "finished_at": answer_contract.utc_now(),
        "returncode": process.returncode,
        "wrapper_sha256": build_runner.sha256_file(CONTROLLED_QA_PROXY),
    }
    stop_path = Path(record["ready_path"]).with_name("stop.json")
    answer_contract.atomic_json_no_clobber(stop_path, stopped)
    stopped["stop_path"] = str(stop_path)
    return stopped


def _initialize_or_validate_output(output_dir: Path, plan: dict[str, Any]) -> Path:
    output = output_dir.expanduser().absolute()
    preregistration = output / "preregistration.json"
    if not output.exists():
        return initialize_output(output, plan)
    answer_contract.reject_symlink_components(output)
    if preregistration.is_symlink() or not preregistration.is_file():
        raise QARunnerError("existing QA root lacks a regular preregistration")
    if build_runner.read_json(preregistration) != plan:
        raise QARunnerError("existing QA preregistration differs")
    return preregistration


def run_execution(
    *,
    plan: dict[str, Any],
    bindings: Sequence[contract.RunBinding],
    validated_units: Sequence[dict[str, Any]],
    output_dir: Path,
    qa_run_id: str,
    completion_resource: Any,
    answer_client: Any,
    tokenizer: Any,
    proxy_log: Path | None,
    formal: bool,
) -> list[dict[str, Any]]:
    """Execute or resume one complete preregistered QA scope."""

    output = output_dir.expanduser().absolute()
    preregistration = _initialize_or_validate_output(output, plan)
    records = execute_pending_questions(
        bindings=bindings,
        validated_units=validated_units,
        output_dir=output,
        qa_run_id=qa_run_id,
        completion_resource=completion_resource,
        answer_client=answer_client,
        tokenizer=tokenizer,
        proxy_log=proxy_log,
        formal=formal,
    )
    expected_count = plan.get("question_count")
    if expected_count != len(records):
        raise QARunnerError(
            f"QA scope incomplete: expected {expected_count}, got {len(records)}"
        )
    completion_path = output / "completion.json"
    if completion_path.is_file():
        completed = build_runner.read_json(completion_path)
        if (
            completed.get("status") != "complete"
            or completed.get("protocol_id") != plan.get("protocol_id")
            or completed.get("question_count") != len(records)
            or completed.get("preregistration_sha256")
            != build_runner.sha256_file(preregistration)
        ):
            raise QARunnerError("existing QA completion artifact differs")
        return records

    outputs: dict[str, Any] = {}
    for tier in sorted({str(record["tier"]) for record in records}):
        tier_records = [record for record in records if record["tier"] == tier]
        tier_outputs: dict[str, Any] = {}
        if any(
            record["benchmark"] in {"locomo", "longmemeval-s"}
            for record in tier_records
        ):
            short = publish_short_answer_evaluator_inputs(
                records=tier_records,
                output_dir=output,
                tier=tier,
            )
            tier_outputs["locomo"] = [str(path) for path in short["locomo"]]
            tier_outputs["longmemeval-s"] = str(short["longmemeval-s"])
        if any(record["benchmark"] == "beam-100k" for record in tier_records):
            tier_outputs["beam-100k"] = str(
                publish_beam_predictions(
                    records=tier_records,
                    output_dir=output,
                    tier=tier,
                )
            )
        outputs[tier] = tier_outputs
    artifact_hashes = {
        tier: {
            benchmark: (
                [build_runner.sha256_file(Path(path)) for path in value]
                if isinstance(value, list)
                else build_runner.sha256_file(Path(value))
            )
            for benchmark, value in tier_outputs.items()
        }
        for tier, tier_outputs in outputs.items()
    }
    completed = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": plan.get("protocol_id"),
        "qa_run_id": qa_run_id,
        "run_ids": plan.get("run_ids"),
        "question_count": len(records),
        "preregistration_sha256": build_runner.sha256_file(preregistration),
        "evaluator_inputs": outputs,
        "evaluator_input_sha256": artifact_hashes,
        "completed_at": answer_contract.utc_now(),
    }
    answer_contract.atomic_json_no_clobber(completion_path, completed)
    return records


def run_preflight(
    *,
    plan: dict[str, Any],
    bindings: Sequence[contract.RunBinding],
    validated_units: Sequence[dict[str, Any]],
    output_dir: Path,
    qa_run_id: str,
    completion_resource: Any,
    answer_client: Any,
    tokenizer: Any,
    proxy_log: Path | None,
    formal: bool,
) -> list[dict[str, Any]]:
    """Execute one question for each tier/benchmark pair without finalizing."""

    if len(bindings) != len(validated_units):
        raise QARunnerError("bindings and validated units must align")
    output = output_dir.expanduser().absolute()
    preregistration = _initialize_or_validate_output(output, plan)
    selected_bindings: list[contract.RunBinding] = []
    selected_units: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for binding, unit in zip(bindings, validated_units, strict=True):
        key = (str(binding.row["tier"]), str(binding.row["benchmark"]))
        if key in seen:
            continue
        questions = unit.get("questions")
        if not isinstance(questions, list) or not questions:
            raise QARunnerError("preflight unit has no questions")
        seen.add(key)
        selected_bindings.append(binding)
        selected_units.append({**unit, "questions": [questions[0]]})
    records = execute_pending_questions(
        bindings=selected_bindings,
        validated_units=selected_units,
        output_dir=output,
        qa_run_id=qa_run_id,
        completion_resource=completion_resource,
        answer_client=answer_client,
        tokenizer=tokenizer,
        proxy_log=proxy_log,
        formal=formal,
    )
    expected_ids = [record["question_id"] for record in records]
    preflight_path = output / "preflight.json"
    if preflight_path.is_file():
        existing = build_runner.read_json(preflight_path)
        if (
            existing.get("status") != "complete"
            or existing.get("question_count") != len(records)
            or existing.get("question_ids") != expected_ids
            or existing.get("preregistration_sha256")
            != build_runner.sha256_file(preregistration)
        ):
            raise QARunnerError("existing preflight artifact differs")
        return records
    payload = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": plan.get("protocol_id"),
        "qa_run_id": qa_run_id,
        "question_count": len(records),
        "question_ids": expected_ids,
        "answer_sha256": {
            record["question_id"]: hashlib.sha256(
                str(record["answer"]).encode("utf-8")
            ).hexdigest()
            for record in records
        },
        "preregistration_sha256": build_runner.sha256_file(preregistration),
        "completed_at": answer_contract.utc_now(),
    }
    answer_contract.atomic_json_no_clobber(preflight_path, payload)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = prepare_plan(args)
    print(
        f"planned QA: {len(plan['run_ids'])} runs / "
        f"{plan['question_count']} questions"
    )
    if not args.execute and not args.allow_model_requests:
        print("model requests sent: 0")
        return 0
    if not (args.execute and args.allow_model_requests):
        raise QARunnerError(
            "real QA requires both --execute and --allow-model-requests"
        )
    if not isinstance(args.upstream_code_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", args.upstream_code_sha256
    ) is None:
        raise QARunnerError("execution requires a frozen upstream code SHA-256")
    bindings, validated_units, execution_plan = prepare_scope(args)
    if execution_plan != plan:
        raise QARunnerError("execution scope differs from the printed plan")
    validate_formal_scope(
        bindings=bindings,
        validated_units=validated_units,
    )
    tokenizer = answer_contract.formal_token_counter()
    if tokenizer.identity != plan.get("qa", {}).get("tokenizer"):
        raise QARunnerError("formal tokenizer identity differs from the QA plan")
    output = args.output_dir.expanduser().absolute()
    _initialize_or_validate_output(output, plan)
    qa_run_id = args.qa_run_id or _default_qa_run_id(plan)
    recover_stale_qa_proxies(output_dir=output, run_id=qa_run_id)
    resume_marker = output / (
        "preflight.json" if args.preflight_only else "completion.json"
    )
    execution_function = run_preflight if args.preflight_only else run_execution
    if resume_marker.is_file():
        records = execution_function(
            plan=plan,
            bindings=bindings,
            validated_units=validated_units,
            output_dir=output,
            qa_run_id=qa_run_id,
            completion_resource=object(),
            answer_client=object(),
            tokenizer=tokenizer,
            proxy_log=None,
            formal=True,
        )
        label = "preflight" if args.preflight_only else "QA"
        print(f"completed {label}: {len(records)} questions")
        return 0

    process = process_log = proxy_record = http_client = None
    try:
        process, proxy_record, process_log = start_qa_proxy(
            output_dir=output,
            python=args.python.expanduser().resolve(),
            upstream=args.upstream,
            upstream_code_sha256=args.upstream_code_sha256,
            run_id=qa_run_id,
        )
        import httpx  # noqa: PLC0415
        from openai import OpenAI  # noqa: PLC0415
        from scripts.run_controlled_locomo_answers import (  # noqa: PLC0415
            HttpAnswerClient,
        )

        http_client = httpx.Client(trust_env=False, timeout=3600.0)
        client = OpenAI(
            api_key="x",
            base_url=proxy_record["base_url"],
            max_retries=0,
            timeout=3600.0,
            http_client=http_client,
        )
        answer_client = HttpAnswerClient(
            base_url=proxy_record["base_url"],
            retries=contract.ANSWER_RETRIES,
            answer_max_tokens=contract.ANSWER_MAX_TOKENS,
            timeout_seconds=3600,
        )
        records = execution_function(
            plan=plan,
            bindings=bindings,
            validated_units=validated_units,
            output_dir=output,
            qa_run_id=qa_run_id,
            completion_resource=client.chat.completions,
            answer_client=answer_client,
            tokenizer=tokenizer,
            proxy_log=Path(proxy_record["log_path"]),
            formal=True,
        )
    finally:
        if http_client is not None:
            http_client.close()
        if process is not None and proxy_record is not None and process_log is not None:
            stop_qa_proxy(
                process=process,
                record=proxy_record,
                process_log=process_log,
            )
    label = "preflight" if args.preflight_only else "QA"
    print(f"completed {label}: {len(records)} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
