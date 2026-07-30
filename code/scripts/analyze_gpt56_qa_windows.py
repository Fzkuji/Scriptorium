#!/usr/bin/env python3
"""Analyze paired QA quality across GPT-5.6 memory tiers and writer windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import analyze_gpt56_w32_screening as screening


ROOT = Path(__file__).resolve().parents[1]
TIERS = ("luna", "terra", "sol")
WINDOWS = (4, 8, 16, 32)
DEFAULT_COMPLETED_RUNS = (
    ROOT / "experiments/gpt56-chunk-curve/artifacts/completed_runs.jsonl"
)
DEFAULT_NEW_QA_ROOT = ROOT / "results/formal/gpt56-window-qa-20260721"
DEFAULT_NEW_SCORE_ROOT = (
    ROOT / "results/formal/gpt56-window-qa-scores-20260721"
)
DEFAULT_OLD_QA_ROOT = (
    ROOT / "results/formal/gpt56-w32-screening-qa-20260719"
)
DEFAULT_OLD_SCORE_ROOT = (
    ROOT / "results/formal/gpt56-w32-screening-scores-20260719-r1"
)
DEFAULT_OUTPUT = (
    ROOT / "experiments/gpt56-chunk-curve/analysis/qa-window-comparison"
)


class QAWindowAnalysisError(RuntimeError):
    """A QA result cannot support the requested paired comparison."""


def exact_mcnemar_p(candidate_only: int, reference_only: int) -> float:
    """Two-sided exact McNemar p-value for discordant paired binary outcomes."""

    if min(candidate_only, reference_only) < 0:
        raise QAWindowAnalysisError("McNemar counts must be non-negative")
    discordant = candidate_only + reference_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(candidate_only, reference_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def _question_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("benchmark")),
        str(row.get("unit_id")),
        str(row.get("original_question_id")),
    )


def _paired_rows(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    reference_by_key = {_question_key(row): row for row in reference}
    candidate_by_key = {_question_key(row): row for row in candidate}
    if (
        len(reference_by_key) != len(reference)
        or len(candidate_by_key) != len(candidate)
        or set(reference_by_key) != set(candidate_by_key)
    ):
        raise QAWindowAnalysisError("paired question inventory differs")
    pairs = []
    for key in sorted(reference_by_key):
        left = reference_by_key[key]
        right = candidate_by_key[key]
        if left.get("question_sha256") != right.get("question_sha256"):
            raise QAWindowAnalysisError(f"paired question identity differs: {key}")
        pairs.append((left, right))
    return pairs


def _quality_score(row: Mapping[str, Any]) -> float:
    quality = row.get("quality")
    value = quality.get("score") if isinstance(quality, Mapping) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise QAWindowAnalysisError("paired quality score is absent")
    return float(value)


def _quality_bool(row: Mapping[str, Any], field: str) -> bool:
    quality = row.get("quality")
    value = quality.get(field) if isinstance(quality, Mapping) else None
    if not isinstance(value, bool):
        raise QAWindowAnalysisError(f"paired quality {field} is absent")
    return value


def _access_value(row: Mapping[str, Any], field: str) -> float:
    access = row.get("access")
    value = access.get(field) if isinstance(access, Mapping) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise QAWindowAnalysisError(f"paired access metric is absent: {field}")
    return float(value)


def _mean_delta(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    value,
) -> float:
    return statistics.mean(value(candidate) - value(reference) for reference, candidate in pairs)


def compare_cells(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return candidate-minus-reference deltas over the same questions."""

    pairs = _paired_rows(reference, candidate)
    locomo = [pair for pair in pairs if pair[0].get("benchmark") == "locomo"]
    beam = [pair for pair in pairs if pair[0].get("benchmark") == "beam-100k"]
    lme = [
        pair for pair in pairs if pair[0].get("benchmark") == "longmemeval-s"
    ]
    candidate_only = sum(
        _quality_bool(candidate_row, "correct")
        and not _quality_bool(reference_row, "correct")
        for reference_row, candidate_row in locomo
    )
    reference_only = sum(
        _quality_bool(reference_row, "correct")
        and not _quality_bool(candidate_row, "correct")
        for reference_row, candidate_row in locomo
    )
    beam_candidate_only = sum(
        _quality_bool(candidate_row, "pass")
        and not _quality_bool(reference_row, "pass")
        for reference_row, candidate_row in beam
    )
    beam_reference_only = sum(
        _quality_bool(reference_row, "pass")
        and not _quality_bool(candidate_row, "pass")
        for reference_row, candidate_row in beam
    )
    return {
        "questions": len(pairs),
        "locomo_n": len(locomo),
        "locomo_delta": _mean_delta(locomo, _quality_score),
        "locomo_candidate_only": candidate_only,
        "locomo_reference_only": reference_only,
        "locomo_mcnemar_p": exact_mcnemar_p(candidate_only, reference_only),
        "beam_n": len(beam),
        "beam_delta": _mean_delta(beam, _quality_score),
        "beam_candidate_only_pass": beam_candidate_only,
        "beam_reference_only_pass": beam_reference_only,
        "longmemeval_n": len(lme),
        "longmemeval_correct_delta": sum(
            int(_quality_bool(candidate_row, "correct"))
            - int(_quality_bool(reference_row, "correct"))
            for reference_row, candidate_row in lme
        ),
        "visible_tokens_delta": _mean_delta(
            pairs, lambda row: _access_value(row, "visible_tokens")
        ),
        "retrieval_model_calls_delta": _mean_delta(
            pairs, lambda row: _access_value(row, "retrieval_model_calls")
        ),
        "accepted_logical_calls_delta": _mean_delta(
            pairs, lambda row: _access_value(row, "accepted_logical_calls")
        ),
    }


def cell_spec(
    tier: str,
    write_turns: int,
    *,
    new_qa_root: Path,
    new_score_root: Path,
    old_qa_root: Path,
    old_score_root: Path,
) -> dict[str, Any]:
    if tier not in TIERS or write_turns not in WINDOWS:
        raise QAWindowAnalysisError("tier or writer window is unsupported")
    cell = f"{tier}-w{write_turns}"
    if write_turns == 32 and tier in {"terra", "sol"}:
        return {
            "tier": tier,
            "write_turns": write_turns,
            "qa_root": old_qa_root / f"{tier}-r5",
            "score_root": old_score_root,
            "reused_existing_w32": True,
        }
    return {
        "tier": tier,
        "write_turns": write_turns,
        "qa_root": new_qa_root / cell,
        "score_root": new_score_root / cell,
        "reused_existing_w32": False,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise QAWindowAnalysisError(f"JSONL contains a non-object: {path}")
    return rows


def _construction_summary(
    completed_runs: Sequence[Mapping[str, Any]],
    *,
    tier: str,
    write_turns: int,
) -> dict[str, Any]:
    rows = [
        row
        for row in completed_runs
        if row.get("tier") == tier and row.get("write_turns") == write_turns
    ]
    if len(rows) != 6:
        raise QAWindowAnalysisError(f"construction cell is incomplete: {tier}-w{write_turns}")
    messages = sum(int(row["messages"]) for row in rows)
    return {
        "runs": len(rows),
        "messages": messages,
        "calls": sum(int(row["calls"]) for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "wall_time_s": sum(float(row["wall_time_s"]) for row in rows),
        "abstract_bytes": sum(int(row["abstract_bytes"]) for row in rows),
        "final_source_coverage": sum(
            float(row["final_source_coverage"]) * int(row["messages"])
            for row in rows
        )
        / messages,
    }


def _access_mean(access: Mapping[str, Any], metric: str) -> float:
    metrics = access.get("metrics")
    value = metrics.get(metric) if isinstance(metrics, Mapping) else None
    mean = value.get("mean") if isinstance(value, Mapping) else None
    if not isinstance(mean, (int, float)) or isinstance(mean, bool):
        raise QAWindowAnalysisError(f"access mean is absent: {metric}")
    return float(mean)


def load_cell(
    spec: Mapping[str, Any],
    *,
    completed_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tier = str(spec["tier"])
    write_turns = int(spec["write_turns"])
    qa = screening.load_qa_tier(
        Path(spec["qa_root"]),
        tier=tier,
        write_turns=write_turns,
    )
    rows = qa["rows"]
    score_root = Path(spec["score_root"])
    locomo, _ = screening._locomo_score(
        score_root=score_root, tier=tier, rows=rows
    )
    longmemeval, _ = screening._lme_score(
        score_root=score_root, tier=tier, rows=rows
    )
    beam, _, beam_pairing = screening._beam_score(
        score_root=score_root, tier=tier, rows=rows
    )
    if any(row["quality"].get("metric") is None for row in rows):
        raise QAWindowAnalysisError(f"QA cell contains unscored rows: {tier}-w{write_turns}")
    access = screening.aggregate_access(rows)
    construction = _construction_summary(
        completed_runs,
        tier=tier,
        write_turns=write_turns,
    )
    summary = {
        "tier": tier,
        "write_turns": write_turns,
        "questions": len(rows),
        "locomo_score": locomo["overall_lj"],
        "longmemeval_correct": longmemeval["correct"],
        "longmemeval_accuracy": longmemeval["correct"] / longmemeval["n"],
        "beam_score": beam["avg_score"],
        "beam_pass_rate": beam["pass_rate"],
        "visible_tokens_mean": _access_mean(access, "visible_tokens"),
        "retrieval_model_calls_mean": _access_mean(
            access, "retrieval_model_calls"
        ),
        "accepted_logical_calls_mean": _access_mean(
            access, "accepted_logical_calls"
        ),
        "read_calls_mean": _access_mean(access, "read_calls"),
        "retry_total": access["retries"]["total"],
        "budget_hit_rate": access["budget"]["hit_rate"],
        "construction_calls": construction["calls"],
        "construction_input_tokens": construction["input_tokens"],
        "construction_output_tokens": construction["output_tokens"],
        "construction_wall_time_s": construction["wall_time_s"],
        "construction_final_source_coverage": construction[
            "final_source_coverage"
        ],
        "qa_root": str(Path(spec["qa_root"]).resolve()),
        "score_root": str(score_root.resolve()),
        "reused_existing_w32": bool(spec["reused_existing_w32"]),
    }
    return {
        "summary": summary,
        "rows": rows,
        "quality": {
            "locomo": locomo,
            "longmemeval_s": longmemeval,
            "beam_100k": beam,
        },
        "access": access,
        "construction": construction,
        "beam_pairing": beam_pairing,
    }


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _markdown(
    cell_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# GPT-5.6 Memory QA Window Comparison",
        "",
        "所有单元格使用同一组 356 个问题、同一 GPT-5.5 retrieval/answer protocol 和 20K visible-token budget。LoCoMo 与 BEAM 数值为当前 screening subset；LongMemEval-S 仅包含 2 个问题。",
        "",
        "| Builder | Window | LoCoMo | LongMemEval-S | BEAM | Visible tokens | Retrieval calls | Build calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cell_rows:
        lines.append(
            "| {tier} | {write_turns} | {locomo_score:.3f} | "
            "{longmemeval_correct}/2 | {beam_score:.3f} | "
            "{visible_tokens_mean:.0f} | {retrieval_model_calls_mean:.2f} | "
            "{construction_calls} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "Delta 始终为 candidate - reference。McNemar p-value 只用于同一组 LoCoMo 问题的配对二元结果；它不替代多 seed 重复实验。",
            "",
            "| Type | Group | Reference | Candidate | LoCoMo delta | p | BEAM delta | LME correct delta | Visible-token delta |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {comparison_type} | {group} | {reference} | {candidate} | "
            "{locomo_delta:.3f} | {locomo_mcnemar_p:.4f} | "
            "{beam_delta:.3f} | {longmemeval_correct_delta} | "
            "{visible_tokens_delta:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "每个 tier-window 只有一个 memory build，没有独立 seed。问题级配对可以说明当前 356 题 screening scope 内的差异，但不能证明跨构建随机性的稳定性。BEAM 是两段 conversation 的 40 题 rubric-nugget screening，不是完整 BEAM 官方分数。",
            "",
        ]
    )
    return "\n".join(lines)


def latex_tables(cell_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    by_cell = {
        (str(row["tier"]), int(row["write_turns"])): row for row in cell_rows
    }
    if set(by_cell) != {(tier, window) for tier in TIERS for window in WINDOWS}:
        raise QAWindowAnalysisError("LaTeX table cell inventory differs")

    def render(
        *,
        caption: str,
        label: str,
        metric: str,
        value,
    ) -> str:
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Memory builder & $W{=}4$ & $W{=}8$ & $W{=}16$ & $W{=}32$ \\\\",
            "\\midrule",
        ]
        for tier in TIERS:
            values = [value(by_cell[(tier, window)][metric]) for window in WINDOWS]
            lines.append(f"{tier.capitalize()} & " + " & ".join(values) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
        return "\n".join(lines)

    return {
        "qa_locomo_window_table.tex": render(
            caption="LoCoMo screening accuracy across memory builders and writer windows.",
            label="tab:qa-locomo-window",
            metric="locomo_score",
            value=lambda item: f"{float(item) * 100:.1f}",
        ),
        "qa_longmemeval_window_table.tex": render(
            caption="LongMemEval-S screening results across memory builders and writer windows.",
            label="tab:qa-longmemeval-window",
            metric="longmemeval_correct",
            value=lambda item: f"{int(item)}/2",
        ),
        "qa_beam_window_table.tex": render(
            caption="BEAM-100K rubric score on the two-conversation screening subset.",
            label="tab:qa-beam-window",
            metric="beam_score",
            value=lambda item: f"{float(item) * 100:.1f}",
        ),
    }


def analyze(
    *,
    completed_runs_path: Path,
    new_qa_root: Path,
    new_score_root: Path,
    old_qa_root: Path,
    old_score_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    completed_runs = _read_jsonl(completed_runs_path)
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    for tier in TIERS:
        for write_turns in WINDOWS:
            spec = cell_spec(
                tier,
                write_turns,
                new_qa_root=new_qa_root,
                new_score_root=new_score_root,
                old_qa_root=old_qa_root,
                old_score_root=old_score_root,
            )
            cells[(tier, write_turns)] = load_cell(
                spec,
                completed_runs=completed_runs,
            )
    pairing = {cell["beam_pairing"]["judge_procedure_sha256"] for cell in cells.values()}
    inventories = {
        cell["beam_pairing"]["paired_screening_inventory_sha256"]
        for cell in cells.values()
    }
    if len(pairing) != 1 or len(inventories) != 1:
        raise QAWindowAnalysisError("BEAM judge procedure or inventory differs")

    comparisons: list[dict[str, Any]] = []
    for tier in TIERS:
        for candidate_window in (4, 8, 16):
            values = compare_cells(
                cells[(tier, 32)]["rows"],
                cells[(tier, candidate_window)]["rows"],
            )
            comparisons.append(
                {
                    "comparison_type": "window",
                    "group": tier,
                    "reference": f"{tier}-w32",
                    "candidate": f"{tier}-w{candidate_window}",
                    **values,
                }
            )
    for write_turns in WINDOWS:
        for reference_tier, candidate_tier in (
            ("luna", "terra"),
            ("luna", "sol"),
            ("terra", "sol"),
        ):
            values = compare_cells(
                cells[(reference_tier, write_turns)]["rows"],
                cells[(candidate_tier, write_turns)]["rows"],
            )
            comparisons.append(
                {
                    "comparison_type": "builder",
                    "group": f"w{write_turns}",
                    "reference": f"{reference_tier}-w{write_turns}",
                    "candidate": f"{candidate_tier}-w{write_turns}",
                    **values,
                }
            )

    ordered_keys = [(tier, window) for tier in TIERS for window in WINDOWS]
    cell_rows = [cells[key]["summary"] for key in ordered_keys]
    question_rows: list[dict[str, Any]] = []
    for tier, write_turns in ordered_keys:
        cell = cells[(tier, write_turns)]
        for row in cell["rows"]:
            question_rows.append(
                {
                    "tier": tier,
                    "write_turns": write_turns,
                    "benchmark": row["benchmark"],
                    "unit_id": row["unit_id"],
                    "question_id": row["original_question_id"],
                    "category": row.get("category_name"),
                    "question_type": row.get("question_type"),
                    "quality_metric": row["quality"]["metric"],
                    "quality_score": row["quality"]["score"],
                    "correct": row["quality"]["correct"],
                    "pass": row["quality"]["pass"],
                    "visible_tokens": row["access"]["visible_tokens"],
                    "retrieval_model_calls": row["access"][
                        "retrieval_model_calls"
                    ],
                    "accepted_logical_calls": row["access"][
                        "accepted_logical_calls"
                    ],
                    "read_calls": row["access"]["read_calls"],
                    "retry_count": row["access"]["retry_count"],
                }
            )
    report = {
        "schema_version": 1,
        "study": "gpt56-memory-qa-window-comparison",
        "cells": cell_rows,
        "comparisons": comparisons,
        "question_rows": len(question_rows),
        "beam_pairing": {
            "judge_procedure_sha256": next(iter(pairing)),
            "paired_screening_inventory_sha256": next(iter(inventories)),
        },
        "claim_boundary": {
            "build_seeds_per_cell": 1,
            "qa_questions_per_cell": 356,
            "locomo_questions_per_cell": 314,
            "longmemeval_questions_per_cell": 2,
            "beam_questions_per_cell": 40,
            "beam_full_or_official": False,
        },
    }
    _atomic_text(
        output_dir / "qa_analysis.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        output_dir / "qa_cell_summary.csv",
        _csv_text(cell_rows, list(cell_rows[0])),
    )
    _atomic_text(
        output_dir / "qa_paired_deltas.csv",
        _csv_text(comparisons, list(comparisons[0])),
    )
    _atomic_text(
        output_dir / "qa_question_scores.csv",
        _csv_text(question_rows, list(question_rows[0])),
    )
    _atomic_text(output_dir / "QA_RESULTS.md", _markdown(cell_rows, comparisons))
    for name, content in latex_tables(cell_rows).items():
        _atomic_text(output_dir / name, content)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-runs", type=Path, default=DEFAULT_COMPLETED_RUNS)
    parser.add_argument("--new-qa-root", type=Path, default=DEFAULT_NEW_QA_ROOT)
    parser.add_argument("--new-score-root", type=Path, default=DEFAULT_NEW_SCORE_ROOT)
    parser.add_argument("--old-qa-root", type=Path, default=DEFAULT_OLD_QA_ROOT)
    parser.add_argument("--old-score-root", type=Path, default=DEFAULT_OLD_SCORE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze(
        completed_runs_path=args.completed_runs.expanduser().resolve(),
        new_qa_root=args.new_qa_root.expanduser().resolve(),
        new_score_root=args.new_score_root.expanduser().resolve(),
        old_qa_root=args.old_qa_root.expanduser().resolve(),
        old_score_root=args.old_score_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print(
        f"analyzed {len(report['cells'])} cells / "
        f"{report['question_rows']} scored question rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
