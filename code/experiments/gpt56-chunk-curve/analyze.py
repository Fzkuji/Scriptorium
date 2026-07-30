#!/usr/bin/env python3
"""Consolidate and analyze the completed GPT-5.6 writer-window study.

The command is offline. It validates the exact 72 W={4,8,16,32} builds,
measures their final memory artifacts, imports the audited W=32 QA screening,
and writes deterministic tables under this experiment directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_gpt56_w32_builds as build_analysis  # noqa: E402
from scripts import gpt56_chunk_curve_qa_contract as qa_contract  # noqa: E402
from scripts import readonly_nativemem_control as readonly_control  # noqa: E402
from scripts import run_gpt56_chunk_curve as build_runner  # noqa: E402


WINDOWS = (4, 8, 16, 32)
TIERS = ("luna", "terra", "sol")
BENCHMARKS = ("locomo", "longmemeval-s", "beam-100k")
MATRIX = EXPERIMENT_DIR / "run_matrix.jsonl"
QA_SUMMARY = (
    ROOT
    / "results/formal/gpt56-w32-screening-analysis-20260719-r1/summary.json"
)
QA_PAIRED = (
    ROOT
    / "results/formal/gpt56-w32-screening-analysis-20260719-r1/paired_results.json"
)
BUILD_ROOTS = tuple(
    ROOT / value
    for value in (
        "results/formal/gpt56-chunk-curve-subscription-20260718-formal-r2",
        "results/formal/gpt56-chunk-curve-subscription-20260718-reduced-4-8-16-32",
        "results/formal/gpt56-chunk-curve-subscription-20260719-w32-only",
        "results/formal/gpt56-chunk-curve-subscription-20260719-remaining-4-8-16-32-r1",
        "results/formal/gpt56-chunk-curve-subscription-20260720-recovery-r1",
        "results/formal/gpt56-chunk-curve-subscription-20260720-recovery-r2",
        "results/formal/gpt56-chunk-curve-subscription-20260720-recovery-r3",
        "results/formal/gpt56-chunk-curve-subscription-20260720-recovery-r4",
        "results/formal/gpt56-chunk-curve-subscription-20260720-recovery-r5",
        "results/formal/gpt56-chunk-curve-subscription-20260720-recovery-r6",
        "results/formal/gpt56-chunk-curve-subscription-20260720-recovery-r7",
    )
)
QA_ROOTS = (
    ROOT / "results/formal/gpt56-w32-screening-qa-20260719",
    ROOT / "results/formal/gpt56-w32-screening-scores-20260719-r1",
    ROOT / "results/formal/gpt56-w32-screening-analysis-20260719-r1",
    ROOT / "results/formal/gpt56-window-qa-20260721",
    ROOT / "results/formal/gpt56-window-qa-scores-20260721",
)
ARCHIVE_ROOTS = tuple(
    ROOT / value
    for value in (
        "results/formal/gpt56-chunk-curve-frontier-20260718",
        "results/formal/gpt56-chunk-curve-frontier-smoke-20260718",
        "results/formal/gpt56-chunk-curve-subscription-20260718-formal",
        "results/formal/gpt56-chunk-curve-subscription-20260719-recovery-proxy",
        "results/formal/gpt56-chunk-curve-subscription-smoke-20260718",
        "results/formal/gpt56-w32-screening-analysis-20260719-r1-invalid-font-archive",
    )
)
RAW_ROOTS = BUILD_ROOTS + QA_ROOTS + ARCHIVE_ROOTS
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis"
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
RAW_LINK_DIR = EXPERIMENT_DIR / "raw"
CODE_DIR = EXPERIMENT_DIR / "code/current"
CODE_FILES = (
    "experiments/gpt56-chunk-curve/analyze.py",
    "scripts/prepare_gpt56_chunk_curve.py",
    "scripts/run_gpt56_chunk_curve.py",
    "scripts/orchestrate_gpt56_chunk_curve.py",
    "scripts/gpt56_chunk_curve_qa_contract.py",
    "scripts/run_gpt56_chunk_curve_qa.py",
    "scripts/audit_gpt56_chunk_curve_qa.py",
    "scripts/analyze_gpt56_w32_builds.py",
    "scripts/analyze_gpt56_w32_screening.py",
    "scripts/analyze_gpt56_qa_windows.py",
    "scripts/readonly_nativemem_control.py",
    "scripts/controlled_locomo_answer_contract.py",
    "scripts/controlled_subscription_qa_proxy.py",
    "scripts/run_locked_locomo_eval_with_evidence.py",
    "scripts/eval_beam_screening_subset.py",
    "scripts/judge_beam_screening_subset.py",
    "scripts/run_v88_gpt55_longmemeval.py",
    "scripts/run_v88_gpt55_beam.py",
    "scripts/evaluate_v88_gpt55_beam.py",
    "scripts/eval_full.py",
    "src/adapters/run_nativemem.py",
    "src/nativemem.py",
    "src/v10_memory.py",
    "src/v8_memory.py",
    "src/chatgpt_proxy.py",
    "benchmarks/longmemeval/src/evaluation/evaluate_qa.py",
    "tests/test_analyze_gpt56_w32_builds.py",
    "tests/test_analyze_gpt56_w32_screening.py",
    "tests/test_analyze_gpt56_qa_windows.py",
    "tests/test_audit_gpt56_chunk_curve_qa.py",
    "tests/test_chatgpt_proxy.py",
    "tests/test_controlled_subscription_qa_proxy.py",
    "tests/test_eval_beam_screening_subset.py",
    "tests/test_judge_beam_screening_subset.py",
    "tests/test_locked_locomo_eval_evidence.py",
    "tests/test_nativemem_provider_policy.py",
    "tests/test_run_gpt56_chunk_curve.py",
    "tests/test_run_gpt56_chunk_curve_qa.py",
    "tests/test_v10_memory.py",
)
BUILD_HASH_PATHS = {
    "matrix": "experiments/gpt56-chunk-curve/run_matrix.jsonl",
    "runner": "scripts/run_gpt56_chunk_curve.py",
    "adapter": "src/adapters/run_nativemem.py",
    "runtime": "src/nativemem.py",
    "v10_memory": "src/v10_memory.py",
    "v8_memory": "src/v8_memory.py",
    "longmemeval_converter": "scripts/run_v88_gpt55_longmemeval.py",
    "beam_converter": "scripts/run_v88_gpt55_beam.py",
    "locomo_evaluator": "scripts/eval_full.py",
}


class AnalysisError(RuntimeError):
    """The completed campaign is missing, duplicated, or inconsistent."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AnalysisError(f"JSONL row is not an object: {path}:{number}")
        rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def sum_phase(build: Mapping[str, Any], phases: Iterable[str], field: str) -> int:
    usage = build.get("phase_usage")
    if not isinstance(usage, Mapping):
        raise AnalysisError("build phase_usage is absent")
    total = 0
    for phase in phases:
        value = usage.get(phase, {})
        if not isinstance(value, Mapping):
            raise AnalysisError(f"invalid phase usage: {phase}")
        item = value.get(field, 0)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise AnalysisError(f"invalid phase field: {phase}.{field}")
        total += item
    return total


def target_matrix_rows() -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(MATRIX)
        if row.get("tier") in TIERS
        and row.get("benchmark") in BENCHMARKS
        and row.get("write_turns") in WINDOWS
    ]
    run_ids = [str(row.get("run_id")) for row in rows]
    if len(rows) != 72 or len(run_ids) != len(set(run_ids)):
        raise AnalysisError("target matrix must contain exactly 72 unique runs")
    expected_cells = {
        (str(row["tier"]), int(row["write_turns"])) for row in rows
    }
    if expected_cells != {(tier, window) for tier in TIERS for window in WINDOWS}:
        raise AnalysisError("target matrix does not cover all tier-window cells")
    return rows


def locate_builds(rows: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    targets = {str(row["run_id"]) for row in rows}
    candidates: dict[str, list[Path]] = defaultdict(list)
    for root in BUILD_ROOTS:
        if not root.is_dir():
            raise AnalysisError(f"build root is missing: {root}")
        for path in root.rglob("build.json"):
            value = read_json(path)
            run_id = str(value.get("run_id"))
            if run_id in targets:
                candidates[run_id].append(path.parent)
    missing = sorted(targets - set(candidates))
    duplicated = {key: value for key, value in candidates.items() if len(value) != 1}
    if missing or duplicated:
        raise AnalysisError(
            f"invalid build inventory: missing={missing}, "
            f"duplicated={{{', '.join(sorted(duplicated))}}}"
        )
    return {run_id: paths[0] for run_id, paths in candidates.items()}


def source_text_bytes(conversation: Mapping[str, Any]) -> bytes:
    texts: list[str] = []
    session_number = 1
    while f"session_{session_number}" in conversation:
        turns = conversation[f"session_{session_number}"]
        if not isinstance(turns, list):
            raise AnalysisError("conversation session is not a list")
        for turn in turns:
            if not isinstance(turn, Mapping):
                raise AnalysisError("conversation turn is not an object")
            texts.append(str(turn.get("text", "")))
        session_number += 1
    return "\n".join(texts).encode("utf-8")


def analyze_run(row: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    if not build_runner.valid_complete(run_dir, row):
        raise AnalysisError(f"run is not a valid completion: {row['run_id']}")
    binding = qa_contract.RunBinding(
        run_id=str(row["run_id"]), row=row, run_dir=run_dir
    )
    validated = qa_contract.validate_unit_binding(binding)
    build_path = run_dir / "build.json"
    success_path = run_dir / "memory/_SUCCESS.json"
    unit_path = run_dir / "unit.json"
    build = read_json(build_path)
    unit = read_json(unit_path)
    if build.get("status") != "complete":
        raise AnalysisError(f"build is not complete: {row['run_id']}")
    for key in ("run_id", "benchmark", "unit_id", "tier", "reasoning_effort"):
        expected = row.get("model") if key == "builder_model" else row.get(key)
        if key != "builder_model" and build.get(key) != expected:
            raise AnalysisError(f"build identity differs at {key}: {row['run_id']}")
    if build.get("builder_model") != row.get("model"):
        raise AnalysisError(f"builder model differs: {row['run_id']}")
    if int(build.get("write_turns")) != int(row["write_turns"]):
        raise AnalysisError(f"writer window differs: {row['run_id']}")
    if validated.get("unit") != unit:
        raise AnalysisError(f"unit validation differs: {row['run_id']}")

    conversation = validated.get("conversation")
    questions = validated.get("questions")
    if not isinstance(conversation, Mapping) or not isinstance(questions, list):
        raise AnalysisError(f"validated source is absent: {row['run_id']}")
    source_ids = set(readonly_control.build_turn_index(conversation))
    messages = int(build["messages"])
    if len(source_ids) != messages:
        raise AnalysisError(f"source ID count differs: {row['run_id']}")
    memory_dir = run_dir / "memory"
    views = {
        name: build_analysis.analyze_view(memory_dir / name, source_ids=source_ids)
        for name in ("topics", "timeline")
    }
    valid_by_view = {
        name: set(view["references"]["valid_ids"])
        for name, view in views.items()
    }
    combined_valid = valid_by_view["topics"] | valid_by_view["timeline"]
    answer_sessions = (
        build_analysis.longmemeval_answer_session_numbers(binding, validated)
        if row["benchmark"] == "longmemeval-s"
        else None
    )
    gold = build_analysis.gold_coverage(
        benchmark=str(row["benchmark"]),
        questions=questions,
        valid_references=combined_valid,
        answer_session_numbers=answer_sessions,
    )
    mentions = sum(
        int(view["references"]["expanded_mentions"]) for view in views.values()
    )
    valid_mentions = sum(
        int(view["references"]["valid_mentions"]) for view in views.values()
    )
    abstract_bytes = sum(int(view["size"]["bytes"]) for view in views.values())
    abstract_files = sum(int(view["size"]["files"]) for view in views.values())
    abstract_entries = sum(int(view["size"]["entries"]) for view in views.values())
    source_bytes = len(source_text_bytes(conversation))
    source_tokens = int(build["source_tokens"])
    usage = build.get("usage")
    phases = build.get("phase_usage")
    distill = build.get("distill")
    if not all(isinstance(value, Mapping) for value in (usage, phases, distill)):
        raise AnalysisError(f"usage or distill data is absent: {row['run_id']}")
    phase_names = tuple(map(str, phases))
    maintenance_phases = tuple(
        name for name in phase_names if name not in {"v8_distill", "v8_distill_verify"}
    )
    first_unique = int(distill.get("first_pass_unique_dia_ids", 0))
    post_unique = int(distill.get("post_verify_unique_dia_ids", 0))
    event_count = int(build["event_count"])
    calls = int(usage["calls"])
    input_tokens = int(usage["input_tokens"])
    output_tokens = int(usage["output_tokens"])
    wall_time_s = float(build["wall_time_s"])
    return {
        "run_id": str(row["run_id"]),
        "benchmark": str(row["benchmark"]),
        "unit_id": str(row["unit_id"]),
        "tier": str(row["tier"]),
        "model": str(row["model"]),
        "write_turns": int(row["write_turns"]),
        "finished_at": str(build["finished_at"]),
        "run_dir": relative(run_dir),
        "messages": messages,
        "source_tokens": source_tokens,
        "source_bytes": source_bytes,
        "calls": calls,
        "writer_calls": sum_phase(build, ("v8_distill",), "calls"),
        "verify_calls": sum_phase(build, ("v8_distill_verify",), "calls"),
        "maintenance_calls": sum_phase(build, maintenance_phases, "calls"),
        "input_tokens": input_tokens,
        "cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
        "output_tokens": output_tokens,
        "wall_time_s": wall_time_s,
        "writer_chunks": int(distill.get("writer_chunks", 0)),
        "first_pass_events": int(distill.get("first_pass_events", 0)),
        "verify_added_events": int(distill.get("verify_added_events", 0)),
        "post_verify_events": int(distill.get("post_verify_events", event_count)),
        "first_pass_unique_source_ids": first_unique,
        "post_verify_unique_source_ids": post_unique,
        "first_pass_source_coverage": ratio(first_unique, messages),
        "post_verify_source_coverage": ratio(post_unique, messages),
        "verify_source_coverage_gain": ratio(post_unique - first_unique, messages),
        "events_per_message": ratio(event_count, messages),
        "calls_per_100_messages": ratio(calls * 100, messages),
        "input_tokens_per_1000_source_tokens": ratio(
            input_tokens * 1000, source_tokens
        ),
        "output_tokens_per_1000_source_tokens": ratio(
            output_tokens * 1000, source_tokens
        ),
        "wall_minutes_per_100_messages": ratio(
            wall_time_s * 100, messages * 60
        ),
        "final_reference_precision": ratio(valid_mentions, mentions),
        "final_source_coverage": ratio(len(combined_valid), len(source_ids)),
        "topic_source_coverage": ratio(
            len(valid_by_view["topics"]), len(source_ids)
        ),
        "timeline_source_coverage": ratio(
            len(valid_by_view["timeline"]), len(source_ids)
        ),
        "gold_question_any_coverage": gold.get("question_any_coverage"),
        "gold_question_all_coverage": gold.get("question_all_coverage"),
        "gold_unique_target_coverage": gold.get("unique_gold_target_coverage"),
        "gold_level": gold.get("level"),
        "abstract_bytes": abstract_bytes,
        "abstract_files": abstract_files,
        "abstract_entries": abstract_entries,
        "abstract_to_source_byte_ratio": ratio(abstract_bytes, source_bytes),
        "build_sha256": sha256_file(build_path),
        "success_sha256": sha256_file(success_path),
        "unit_sha256": sha256_file(unit_path),
        "memory_sha256": build_analysis.memory_sha256(memory_dir),
    }


AGGREGATE_METRICS = (
    "first_pass_source_coverage",
    "post_verify_source_coverage",
    "verify_source_coverage_gain",
    "final_reference_precision",
    "final_source_coverage",
    "gold_question_any_coverage",
    "gold_question_all_coverage",
    "gold_unique_target_coverage",
    "events_per_message",
    "abstract_to_source_byte_ratio",
)


def mean_present(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def aggregate(rows: Sequence[Mapping[str, Any]], **identity: Any) -> dict[str, Any]:
    messages = sum(int(row["messages"]) for row in rows)
    source_tokens = sum(int(row["source_tokens"]) for row in rows)
    calls = sum(int(row["calls"]) for row in rows)
    input_tokens = sum(int(row["input_tokens"]) for row in rows)
    output_tokens = sum(int(row["output_tokens"]) for row in rows)
    wall_time_s = sum(float(row["wall_time_s"]) for row in rows)
    result = {
        **identity,
        "n": len(rows),
        "messages": messages,
        "source_tokens": source_tokens,
        "calls": calls,
        "writer_calls": sum(int(row["writer_calls"]) for row in rows),
        "verify_calls": sum(int(row["verify_calls"]) for row in rows),
        "maintenance_calls": sum(int(row["maintenance_calls"]) for row in rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "wall_time_hours": wall_time_s / 3600,
        "calls_per_100_messages": ratio(calls * 100, messages),
        "input_tokens_per_1000_source_tokens": ratio(
            input_tokens * 1000, source_tokens
        ),
        "output_tokens_per_1000_source_tokens": ratio(
            output_tokens * 1000, source_tokens
        ),
        "wall_minutes_per_100_messages": ratio(
            wall_time_s * 100, messages * 60
        ),
    }
    result.update({field: mean_present(rows, field) for field in AGGREGATE_METRICS})
    return result


def grouped_aggregates(
    rows: Sequence[dict[str, Any]], fields: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return [
        aggregate(group, **dict(zip(fields, key)))
        for key, group in sorted(groups.items(), key=lambda item: item[0])
    ]


COMPARISON_METRICS = {
    "calls_per_100_messages": "relative_percent",
    "input_tokens_per_1000_source_tokens": "relative_percent",
    "output_tokens_per_1000_source_tokens": "relative_percent",
    "wall_minutes_per_100_messages": "relative_percent",
    "first_pass_source_coverage": "percentage_points",
    "post_verify_source_coverage": "percentage_points",
    "final_source_coverage": "percentage_points",
    "gold_question_all_coverage": "percentage_points",
    "events_per_message": "relative_percent",
    "abstract_to_source_byte_ratio": "relative_percent",
}


def bootstrap_ci(values: Sequence[float], seed: int) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(10000)
    )
    return means[249], means[9749]


def delta_value(base: float, candidate: float, kind: str) -> float:
    if kind == "percentage_points":
        return (candidate - base) * 100
    if kind == "relative_percent":
        if base == 0:
            raise AnalysisError("relative comparison has a zero baseline")
        return (candidate - base) / base * 100
    raise AnalysisError(f"unknown comparison kind: {kind}")


def window_comparisons(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (row["benchmark"], row["unit_id"], row["tier"], row["write_turns"]): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    pair_keys = sorted({key[:3] for key in index})
    pairs = ((4, 8), (8, 16), (16, 32), (4, 16), (4, 32))
    for baseline_window, window in pairs:
        for metric, kind in COMPARISON_METRICS.items():
            deltas: list[float] = []
            for benchmark, unit_id, tier in pair_keys:
                base = index[(benchmark, unit_id, tier, baseline_window)].get(metric)
                candidate = index[(benchmark, unit_id, tier, window)].get(metric)
                if base is None or candidate is None:
                    continue
                deltas.append(delta_value(float(base), float(candidate), kind))
            low, high = bootstrap_ci(deltas, seed=window * 1000 + len(metric))
            output.append(
                {
                    "baseline_window": baseline_window,
                    "candidate_window": window,
                    "metric": metric,
                    "delta_kind": kind,
                    "n_pairs": len(deltas),
                    "mean_delta": statistics.fmean(deltas),
                    "median_delta": statistics.median(deltas),
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "positive_pairs": sum(value > 0 for value in deltas),
                    "zero_pairs": sum(value == 0 for value in deltas),
                    "negative_pairs": sum(value < 0 for value in deltas),
                }
            )
    return output


def tier_comparisons(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (row["benchmark"], row["unit_id"], row["tier"], row["write_turns"]): row
        for row in rows
    }
    pair_keys = sorted({(row["benchmark"], row["unit_id"]) for row in rows})
    comparisons = (("luna", "terra"), ("luna", "sol"), ("terra", "sol"))
    output: list[dict[str, Any]] = []
    for window in WINDOWS:
        for base_tier, candidate_tier in comparisons:
            for metric, kind in COMPARISON_METRICS.items():
                deltas: list[float] = []
                for benchmark, unit_id in pair_keys:
                    base = index[(benchmark, unit_id, base_tier, window)].get(metric)
                    candidate = index[
                        (benchmark, unit_id, candidate_tier, window)
                    ].get(metric)
                    if base is None or candidate is None:
                        continue
                    deltas.append(delta_value(float(base), float(candidate), kind))
                low, high = bootstrap_ci(
                    deltas,
                    seed=window * 10000 + len(metric) + 100 * len(base_tier),
                )
                output.append(
                    {
                        "write_turns": window,
                        "baseline_tier": base_tier,
                        "candidate_tier": candidate_tier,
                        "metric": metric,
                        "delta_kind": kind,
                        "n_pairs": len(deltas),
                        "mean_delta": statistics.fmean(deltas),
                        "median_delta": statistics.median(deltas),
                        "bootstrap_95_low": low,
                        "bootstrap_95_high": high,
                        "positive_pairs": sum(value > 0 for value in deltas),
                        "zero_pairs": sum(value == 0 for value in deltas),
                        "negative_pairs": sum(value < 0 for value in deltas),
                    }
                )
    return output


def qa_rows() -> list[dict[str, Any]]:
    summary = read_json(QA_SUMMARY)
    if summary.get("status") != "complete" or summary.get("scope", {}).get(
        "write_turns"
    ) != 32:
        raise AnalysisError("W32 QA summary is not a completed fixed-W=32 study")
    rows: list[dict[str, Any]] = []
    for tier in ("terra", "sol"):
        quality = summary["quality"][tier]
        rows.extend(
            (
                {
                    "tier": tier,
                    "benchmark": "locomo",
                    "write_turns": 32,
                    "metric": "locked_judge_accuracy",
                    "value": quality["locomo"]["overall_lj"],
                    "n": quality["locomo"]["n"],
                    "scope": "two-conversation screening subset",
                },
                {
                    "tier": tier,
                    "benchmark": "longmemeval-s",
                    "write_turns": 32,
                    "metric": "accuracy",
                    "value": ratio(
                        quality["longmemeval_s"]["correct"],
                        quality["longmemeval_s"]["n"],
                    ),
                    "n": quality["longmemeval_s"]["n"],
                    "scope": "two-question integration screening",
                },
                {
                    "tier": tier,
                    "benchmark": "beam-100k",
                    "write_turns": 32,
                    "metric": "rubric_nugget_mean",
                    "value": quality["beam_100k"]["avg_score"],
                    "n": quality["beam_100k"]["n"],
                    "scope": "two-conversation screening subset",
                },
                {
                    "tier": tier,
                    "benchmark": "beam-100k",
                    "write_turns": 32,
                    "metric": "question_pass_rate",
                    "value": quality["beam_100k"]["pass_rate"],
                    "n": quality["beam_100k"]["n"],
                    "scope": "two-conversation screening subset",
                },
            )
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise AnalysisError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def root_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in RAW_ROOTS:
        if not path.is_dir():
            raise AnalysisError(f"raw artifact root is missing: {path}")
        files = [item for item in path.rglob("*") if item.is_file()]
        rows.append(
            {
                "path": relative(path),
                "files": len(files),
                "bytes": sum(item.stat().st_size for item in files),
                "role": (
                    "construction_target_source"
                    if path in BUILD_ROOTS
                    else (
                        "qa_or_scoring"
                        if path in QA_ROOTS
                        else "excluded_archive"
                    )
                ),
            }
        )
    return rows


def package_raw_links() -> list[dict[str, str]]:
    RAW_LINK_DIR.mkdir(parents=True, exist_ok=True)
    links: list[dict[str, str]] = []
    for source in RAW_ROOTS:
        target = RAW_LINK_DIR / source.name
        relative_target = Path(os.path.relpath(source, target.parent))
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise AnalysisError(f"raw link points to the wrong source: {target}")
        elif target.exists():
            raise AnalysisError(f"raw link target is occupied: {target}")
        else:
            target.symlink_to(relative_target, target_is_directory=True)
        links.append({"link": relative(target), "source": relative(source)})
    return links


def package_code(
    rows: Sequence[Mapping[str, Any]], locations: Mapping[str, Path]
) -> list[dict[str, Any]]:
    recorded: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        build = read_json(locations[str(row["run_id"])] / "build.json")
        source_hashes = build.get("source_hashes")
        if not isinstance(source_hashes, Mapping):
            raise AnalysisError(f"source hashes are absent: {row['run_id']}")
        for alias, value in source_hashes.items():
            if not isinstance(value, str):
                raise AnalysisError(f"invalid source hash: {row['run_id']}:{alias}")
            recorded[str(alias)][value] += 1

    path_aliases: dict[str, list[str]] = defaultdict(list)
    for alias, path in BUILD_HASH_PATHS.items():
        path_aliases[path].append(alias)
    manifest: list[dict[str, Any]] = []
    for path_text in CODE_FILES:
        source = ROOT / path_text
        if not source.is_file():
            raise AnalysisError(f"code source is missing: {source}")
        destination = CODE_DIR / path_text
        destination.parent.mkdir(parents=True, exist_ok=True)
        current_hash = sha256_file(source)
        if destination.is_file() and sha256_file(destination) == current_hash:
            pass
        else:
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            shutil.copy2(source, destination)
        aliases = sorted(path_aliases.get(path_text, []))
        recorded_hashes = {
            value: count
            for alias in aliases
            for value, count in sorted(recorded.get(alias, {}).items())
        }
        manifest.append(
            {
                "path": path_text,
                "snapshot": relative(destination),
                "current_sha256": current_hash,
                "build_hash_aliases": aliases,
                "recorded_build_hash_counts": recorded_hashes,
                "current_hash_recorded_by_builds": (
                    current_hash in recorded_hashes if aliases else None
                ),
            }
        )
    write_json(EXPERIMENT_DIR / "code/manifest.json", manifest)
    return manifest


def main() -> int:
    qa_contract.verify_locomo_evaluator()
    rows = target_matrix_rows()
    locations = locate_builds(rows)
    per_run = [
        analyze_run(row, locations[str(row["run_id"])])
        for row in sorted(rows, key=lambda item: str(item["run_id"]))
    ]
    if len(per_run) != 72:
        raise AnalysisError("analysis did not produce 72 completed runs")
    by_window = grouped_aggregates(per_run, ("write_turns",))
    by_tier_window = grouped_aggregates(per_run, ("tier", "write_turns"))
    by_benchmark_window = grouped_aggregates(
        per_run, ("benchmark", "write_turns")
    )
    by_tier = grouped_aggregates(per_run, ("tier",))
    by_benchmark = grouped_aggregates(per_run, ("benchmark",))
    window_delta = window_comparisons(per_run)
    tier_delta = tier_comparisons(per_run)
    screening = qa_rows()
    raw_roots = root_manifest()
    raw_links = package_raw_links()
    code_manifest = package_code(rows, locations)

    write_jsonl(ARTIFACT_DIR / "completed_runs.jsonl", per_run)
    write_json(ARTIFACT_DIR / "raw_roots.json", raw_roots)
    write_csv(ANALYSIS_DIR / "per_run.csv", per_run)
    write_csv(ANALYSIS_DIR / "aggregate_by_window.csv", by_window)
    write_csv(ANALYSIS_DIR / "aggregate_by_tier_window.csv", by_tier_window)
    write_csv(
        ANALYSIS_DIR / "aggregate_by_benchmark_window.csv", by_benchmark_window
    )
    write_csv(ANALYSIS_DIR / "aggregate_by_tier.csv", by_tier)
    write_csv(ANALYSIS_DIR / "aggregate_by_benchmark.csv", by_benchmark)
    write_csv(ANALYSIS_DIR / "window_comparisons.csv", window_delta)
    write_csv(ANALYSIS_DIR / "tier_comparisons.csv", tier_delta)
    write_csv(ANALYSIS_DIR / "qa_screening_w32.csv", screening)
    write_json(
        ANALYSIS_DIR / "summary.json",
        {
            "schema_version": 1,
            "status": "complete",
            "study": "gpt56-writer-window-4-8-16-32",
            "scope": {
                "builds": len(per_run),
                "windows": list(WINDOWS),
                "tiers": list(TIERS),
                "benchmarks": list(BENCHMARKS),
                "units_per_tier_window": 6,
            },
            "source_integrity": {
                "locked_locomo_evaluator_sha256": qa_contract.verify_locomo_evaluator(),
                "matrix_sha256": sha256_file(MATRIX),
                "qa_summary_sha256": sha256_file(QA_SUMMARY),
                "qa_paired_sha256": sha256_file(QA_PAIRED),
                "analyzer_sha256": sha256_file(Path(__file__)),
            },
            "package": {
                "raw_roots": raw_roots,
                "raw_links": raw_links,
                "code_files": code_manifest,
            },
            "aggregates": {
                "by_window": by_window,
                "by_tier_window": by_tier_window,
                "by_benchmark_window": by_benchmark_window,
                "by_tier": by_tier,
                "by_benchmark": by_benchmark,
            },
            "window_comparisons": window_delta,
            "tier_comparisons": tier_delta,
            "qa_screening_w32": screening,
            "claim_boundary": {
                "construction_curve": (
                    "72 builds: 3 tiers x 4 windows x 6 histories"
                ),
                "downstream_quality": (
                    "W=32 only; Terra and Sol only; screening subsets"
                ),
                "not_established": [
                    "downstream QA differences among W=4,8,16,32",
                    "full-benchmark LoCoMo, LongMemEval, or BEAM performance",
                    "benchmark-generalized statistical significance",
                    "downstream QA performance for Luna",
                ],
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
