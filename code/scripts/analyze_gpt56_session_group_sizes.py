#!/usr/bin/env python3
"""Measure the actual source length of session-group writer calls."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import tiktoken

from scripts import run_gpt56_chunk_curve as build_runner
from scripts import analyze_gpt56_w32_builds as build_analysis
from scripts import readonly_nativemem_control as readonly_control


class SessionGroupAnalysisError(RuntimeError):
    pass


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _distribution(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        raise SessionGroupAnalysisError("cannot summarize an empty distribution")

    def nearest_rank(percentile: float) -> int:
        return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]

    return {
        "min": ordered[0],
        "p25": nearest_rank(0.25),
        "median": statistics.median(ordered),
        "p75": nearest_rank(0.75),
        "max": ordered[-1],
    }


def _construction_metrics(row: Mapping[str, Any], run_dir: Path) -> dict[str, int | float]:
    build = build_runner.read_json(run_dir / "build.json")
    usage = build.get("usage")
    distill = build.get("distill")
    if (
        not isinstance(usage, Mapping)
        or not isinstance(distill, Mapping)
        or build.get("run_id") != row.get("run_id")
    ):
        raise SessionGroupAnalysisError("build metrics are absent or misbound")
    fields = {
        "messages": build.get("messages"),
        "source_tokens": build.get("source_tokens"),
        "calls": usage.get("calls"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "wall_time_s": build.get("wall_time_s"),
        "first_pass_source_ids": distill.get("first_pass_unique_dia_ids"),
        "post_verify_source_ids": distill.get("post_verify_unique_dia_ids"),
    }
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < 0
        for value in fields.values()
    ):
        raise SessionGroupAnalysisError("build metric is invalid")
    return fields


def _evidence_metrics(
    *,
    benchmark: str,
    run_dir: Path,
    conversation: Mapping[str, Any],
    questions: object,
) -> dict[str, Any]:
    if benchmark != "locomo":
        return {"availability": "not_available"}
    if not isinstance(questions, list):
        raise SessionGroupAnalysisError("LoCoMo questions are not a list")
    source_ids = set(readonly_control.build_turn_index(conversation))
    views = [
        build_analysis.analyze_view(run_dir / "memory" / name, source_ids=source_ids)
        for name in ("topics", "timeline")
    ]
    referenced = set().union(
        *(set(view["references"]["valid_ids"]) for view in views)
    )
    targets = [
        set(readonly_control.expand_source_specs(question["evidence"]))
        for question in questions
        if isinstance(question, Mapping)
        and isinstance(question.get("evidence"), list)
        and question["evidence"]
    ]
    gold_ids = set().union(*targets) if targets else set()
    mentions = sum(int(view["references"]["expanded_mentions"]) for view in views)
    valid_mentions = sum(int(view["references"]["valid_mentions"]) for view in views)
    return {
        "availability": "available",
        "source_ids": len(source_ids),
        "referenced_source_ids": len(referenced),
        "reference_mentions": mentions,
        "valid_reference_mentions": valid_mentions,
        "source_coverage": len(referenced) / len(source_ids) if source_ids else None,
        "reference_precision": valid_mentions / mentions if mentions else None,
        "eligible_questions": len(targets),
        "gold_any_hits": sum(bool(target & referenced) for target in targets),
        "gold_all_hits": sum(target <= referenced for target in targets),
        "unique_gold_targets": len(gold_ids),
        "unique_gold_hits": len(gold_ids & referenced),
    }


def _summarize_construction(runs: list[Mapping[str, Any]]) -> dict[str, int | float]:
    totals = {
        field: sum(run["construction"][field] for run in runs)
        for field in (
            "messages",
            "source_tokens",
            "calls",
            "input_tokens",
            "output_tokens",
            "wall_time_s",
            "first_pass_source_ids",
            "post_verify_source_ids",
        )
    }
    messages = totals["messages"]
    source_tokens = totals["source_tokens"]
    if messages <= 0 or source_tokens <= 0:
        raise SessionGroupAnalysisError("construction denominator is empty")
    return {
        "messages": int(messages),
        "source_tokens": int(source_tokens),
        "calls": int(totals["calls"]),
        "input_tokens": int(totals["input_tokens"]),
        "output_tokens": int(totals["output_tokens"]),
        "wall_time_s": float(totals["wall_time_s"]),
        "calls_per_100_messages": totals["calls"] / messages * 100,
        "input_tokens_per_1000_source_tokens": (
            totals["input_tokens"] / source_tokens * 1000
        ),
        "output_tokens_per_1000_source_tokens": (
            totals["output_tokens"] / source_tokens * 1000
        ),
        "wall_minutes_per_100_messages": (
            totals["wall_time_s"] / 60 / messages * 100
        ),
        "first_pass_source_coverage": totals["first_pass_source_ids"] / messages,
        "post_verify_source_coverage": totals["post_verify_source_ids"] / messages,
        "verification_gain": (
            totals["post_verify_source_ids"] - totals["first_pass_source_ids"]
        )
        / messages,
    }


def _summarize_evidence(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    evidence = [run["evidence"] for run in runs]
    if all(row.get("availability") == "not_available" for row in evidence):
        return {"availability": "not_available"}
    if any(row.get("availability") != "available" for row in evidence):
        raise SessionGroupAnalysisError("evidence availability differs within cell")
    totals = {
        field: sum(int(row[field]) for row in evidence)
        for field in (
            "source_ids",
            "referenced_source_ids",
            "reference_mentions",
            "valid_reference_mentions",
            "eligible_questions",
            "gold_any_hits",
            "gold_all_hits",
            "unique_gold_targets",
            "unique_gold_hits",
        )
    }
    return {
        "availability": "available",
        "source_coverage": _ratio(
            totals["referenced_source_ids"], totals["source_ids"]
        ),
        "reference_precision": _ratio(
            totals["valid_reference_mentions"], totals["reference_mentions"]
        ),
        "gold_any_coverage": _ratio(
            totals["gold_any_hits"], totals["eligible_questions"]
        ),
        "gold_all_coverage": _ratio(
            totals["gold_all_hits"], totals["eligible_questions"]
        ),
        "unique_gold_coverage": _ratio(
            totals["unique_gold_hits"], totals["unique_gold_targets"]
        ),
        "source_ids": totals["source_ids"],
        "referenced_source_ids": totals["referenced_source_ids"],
        "eligible_questions": totals["eligible_questions"],
        "unique_gold_targets": totals["unique_gold_targets"],
    }


def summarize_runs(runs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        key = (
            str(run["benchmark"]),
            str(run["tier"]),
            int(run["nominal_session_group_size"]),
        )
        cells[key].append(run)
    summaries: list[dict[str, Any]] = []
    for (benchmark, tier, nominal_size), cell_runs in sorted(cells.items()):
        groups = [group for run in cell_runs for group in run["groups"]]
        summary = {
                "benchmark": benchmark,
                "tier": tier,
                "nominal_session_group_size": nominal_size,
                "runs": len(cell_runs),
                "writer_calls": len(groups),
                "actual_sessions": _distribution(
                    [int(group["actual_sessions"]) for group in groups]
                ),
                "messages": _distribution(
                    [int(group["messages"]) for group in groups]
                ),
                "source_tokens": _distribution(
                    [int(group["source_tokens"]) for group in groups]
                ),
            }
        if all(isinstance(run.get("construction"), Mapping) for run in cell_runs):
            summary["construction"] = _summarize_construction(cell_runs)
        if all(isinstance(run.get("evidence"), Mapping) for run in cell_runs):
            summary["evidence"] = _summarize_evidence(cell_runs)
        summaries.append(summary)
    return summaries


def analyze_campaign(
    *,
    matrix_path: Path,
    build_roots: list[Path],
    allow_partial: bool,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in matrix_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_ids = [str(row.get("run_id", "")) for row in rows]
    if not all(run_ids) or len(run_ids) != len(set(run_ids)):
        raise SessionGroupAnalysisError("matrix run IDs are missing or duplicated")
    runs: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        matches = [
            build_runner.run_dir(root, row)
            for root in build_roots
            if build_runner.valid_complete(build_runner.run_dir(root, row), row)
        ]
        if not matches:
            missing.append(str(row["run_id"]))
            continue
        if len(matches) != 1:
            raise SessionGroupAnalysisError(
                f"run is complete under multiple roots: {row['run_id']}"
            )
        conversation, questions, _metadata = build_runner.load_unit(row)
        run = analyze_run(row, matches[0], conversation)
        run["construction"] = _construction_metrics(row, matches[0])
        run["evidence"] = _evidence_metrics(
            benchmark=str(row["benchmark"]),
            run_dir=matches[0],
            conversation=conversation,
            questions=questions,
        )
        runs.append(run)
    if missing and not allow_partial:
        raise SessionGroupAnalysisError(f"campaign is missing {len(missing)} runs")
    return {
        "status": "partial" if missing else "complete",
        "planned_runs": len(rows),
        "completed_runs": len(runs),
        "missing_run_ids": missing,
        "runs": runs,
        "cells": summarize_runs(runs),
    }


def analyze_run(
    row: Mapping[str, Any],
    run_dir: Path,
    conversation: Mapping[str, Any],
) -> dict[str, Any]:
    source_text: dict[str, str] = {}
    source_session: dict[str, int] = {}
    session_count = 0
    while f"session_{session_count + 1}" in conversation:
        session_count += 1
        turns = conversation[f"session_{session_count}"]
        if not isinstance(turns, list):
            raise SessionGroupAnalysisError("conversation session is not a list")
        for turn in turns:
            dia_id = str(turn.get("dia_id", "")) if isinstance(turn, Mapping) else ""
            if not dia_id or dia_id in source_text:
                raise SessionGroupAnalysisError("source IDs are missing or duplicated")
            source_text[dia_id] = str(turn.get("text", ""))
            source_session[dia_id] = session_count

    trace_path = run_dir / "distill_trace.jsonl"
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    encoding = tiktoken.get_encoding("o200k_base")
    groups: list[dict[str, int]] = []
    covered: list[str] = []
    for record in records:
        ids = [str(value) for value in record.get("input_dia_ids", [])]
        if (
            not ids
            or len(ids) != record.get("turn_count")
            or any(dia_id not in source_text for dia_id in ids)
        ):
            raise SessionGroupAnalysisError("trace source binding differs")
        actual_sessions = len({source_session[dia_id] for dia_id in ids})
        declared_sessions = record.get("session_group_size")
        if declared_sessions is not None and actual_sessions != declared_sessions:
            raise SessionGroupAnalysisError("trace session group size differs")
        groups.append(
            {
                "actual_sessions": actual_sessions,
                "messages": len(ids),
                "source_tokens": len(
                    encoding.encode("\n".join(source_text[dia_id] for dia_id in ids))
                ),
            }
        )
        covered.extend(ids)
    if len(covered) != len(set(covered)) or set(covered) != set(source_text):
        raise SessionGroupAnalysisError("trace does not cover each source message once")
    return {
        "run_id": row.get("run_id"),
        "benchmark": row.get("benchmark"),
        "tier": row.get("tier"),
        "nominal_session_group_size": row.get("session_group_size"),
        "groups": groups,
        "source_coverage": {
            "messages": len(covered),
            "unique_messages": len(set(covered)),
            "sessions": session_count,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    report = analyze_campaign(
        matrix_path=args.matrix,
        build_roots=args.build_root,
        allow_partial=args.allow_partial,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        f"{report['status']}: {report['completed_runs']}/"
        f"{report['planned_runs']} runs -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
