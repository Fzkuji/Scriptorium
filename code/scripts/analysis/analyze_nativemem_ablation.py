#!/usr/bin/env python3
"""Summarize a paired NativeMem LoCoMo and BEAM ablation."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from scripts.analyze_gpt56_qa_windows import exact_mcnemar_p


CONDITIONS = ("dual_source", "topic_source", "timeline_source", "dual_no_source")
COMPARISONS = (
    ("timeline_view", "topic_source"),
    ("topical_view", "timeline_source"),
    ("source_resolution", "dual_no_source"),
)


def bootstrap_ci(deltas: list[float], *, seed: int = 20260723) -> list[float]:
    rng = random.Random(seed)
    means = [
        statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(10_000)
    ]
    means.sort()
    return [means[249], means[9749]]


def paired(reference: list[float], candidate: list[float]) -> dict:
    assert len(reference) == len(candidate) and reference
    deltas = [left - right for left, right in zip(reference, candidate)]
    ref_only = sum(left >= 0.5 and right < 0.5 for left, right in zip(reference, candidate))
    cand_only = sum(left < 0.5 and right >= 0.5 for left, right in zip(reference, candidate))
    return {
        "delta": statistics.mean(deltas),
        "bootstrap_95_ci": bootstrap_ci(deltas),
        "reference_only_wins": ref_only,
        "candidate_only_wins": cand_only,
        "exact_mcnemar_p": exact_mcnemar_p(ref_only, cand_only),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()

    locomo: dict[str, dict] = {}
    beam: dict[str, dict] = {}
    for condition in CONDITIONS:
        locomo[condition] = json.loads(
            (root / condition / "locomo/eval_full.json").read_text()
        )
        beam[condition] = json.loads(
            (root / condition / "beam-100k/judge/sol/results.json").read_text()
        )

    reference_questions = [
        (row["question"], row["gold"], row["category"])
        for row in locomo["dual_source"]["records"]
    ]
    for condition in CONDITIONS:
        rows = locomo[condition]["records"]
        assert len(rows) == 314
        assert [
            (row["question"], row["gold"], row["category"]) for row in rows
        ] == reference_questions

    reference_beam_ids = [
        row["source_question_id"] for row in beam["dual_source"]["evaluations"]
    ]
    for condition in CONDITIONS:
        assert beam[condition]["paired_screening_inventory_sha256"] == beam[
            "dual_source"
        ]["paired_screening_inventory_sha256"]
        assert [
            row["source_question_id"] for row in beam[condition]["evaluations"]
        ] == reference_beam_ids

    summary = {"scope": {}, "conditions": {}, "comparisons": {}}
    summary["scope"] = {
        "locomo_questions": 314,
        "locomo_conversations": 2,
        "beam_questions": 40,
        "beam_conversations": 2,
        "beam_is_screening_subset": True,
        "answer_model": "gpt-5.6-sol",
        "locomo_judge": "openai/gpt-4o-mini",
        "beam_judge": "openai/gpt-4o-mini",
    }

    for condition in CONDITIONS:
        locomo_scores = [
            float(row["judge_score"]) for row in locomo[condition]["records"]
        ]
        beam_scores = [
            float(row["score"]) for row in beam[condition]["evaluations"]
        ]
        beam_pass = [float(score >= 0.5) for score in beam_scores]
        result_rows = []
        for unit in ("conv-44", "conv-48"):
            result_rows.extend(
                json.loads(
                    (root / condition / "locomo" / unit / "results.json").read_text()
                )
            )
        summary["conditions"][condition] = {
            "locomo_accuracy": statistics.mean(locomo_scores),
            "locomo_by_category": locomo[condition]["by_category"],
            "beam_rubric_mean": statistics.mean(beam_scores),
            "beam_pass_rate": statistics.mean(beam_pass),
            "retrieval_steps_mean_locomo": statistics.mean(
                row["retrieval_steps"] for row in result_rows
            ),
        }

    for name, candidate in COMPARISONS:
        locomo_reference = [
            float(row["judge_score"]) for row in locomo["dual_source"]["records"]
        ]
        locomo_candidate = [
            float(row["judge_score"]) for row in locomo[candidate]["records"]
        ]
        beam_reference = [
            float(row["score"]) for row in beam["dual_source"]["evaluations"]
        ]
        beam_candidate = [
            float(row["score"]) for row in beam[candidate]["evaluations"]
        ]
        summary["comparisons"][name] = {
            "reference": "dual_source",
            "candidate": candidate,
            "delta_definition": "dual_source - candidate",
            "locomo": paired(locomo_reference, locomo_candidate),
            "beam_rubric": paired(beam_reference, beam_candidate),
            "beam_pass": paired(
                [float(score >= 0.5) for score in beam_reference],
                [float(score >= 0.5) for score in beam_candidate],
            ),
        }

    out = root / "analysis"
    out.mkdir(exist_ok=True)
    (out / "ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )

    lines = [
        "# NativeMem Paired Ablation",
        "",
        "| Condition | LoCoMo J | BEAM rubric | BEAM pass | LoCoMo retrieval steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = summary["conditions"][condition]
        lines.append(
            f"| {condition} | {row['locomo_accuracy'] * 100:.1f} | "
            f"{row['beam_rubric_mean'] * 100:.1f} | "
            f"{row['beam_pass_rate'] * 100:.1f} | "
            f"{row['retrieval_steps_mean_locomo']:.2f} |"
        )
    lines += [
        "",
        "| Component | LoCoMo delta | 95% CI | p | BEAM rubric delta | 95% CI | BEAM pass delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("timeline_view", "topical_view", "source_resolution"):
        row = summary["comparisons"][name]
        l = row["locomo"]
        b = row["beam_rubric"]
        bp = row["beam_pass"]
        lines.append(
            f"| {name} | {l['delta'] * 100:+.1f} | "
            f"[{l['bootstrap_95_ci'][0] * 100:+.1f}, {l['bootstrap_95_ci'][1] * 100:+.1f}] | "
            f"{l['exact_mcnemar_p']:.4f} | {b['delta'] * 100:+.1f} | "
            f"[{b['bootstrap_95_ci'][0] * 100:+.1f}, {b['bootstrap_95_ci'][1] * 100:+.1f}] | "
            f"{bp['delta'] * 100:+.1f} |"
        )
    lines += [
        "",
        "Delta 为 dual_source 减去对应消融条件。LoCoMo 仅覆盖两个 conversation 的 314 题；BEAM 仅覆盖两个 conversation 的 40 题 rubric-nugget screening，不是完整官方 BEAM 分数。每个条件只有一次生成结果，置信区间反映问题级配对差异，不反映构建或回答随机性。",
        "",
    ]
    (out / "ablation_summary.md").write_text("\n".join(lines))
    print(out / "ablation_summary.md")


if __name__ == "__main__":
    main()
