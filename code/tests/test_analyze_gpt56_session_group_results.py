from __future__ import annotations

import json
import hashlib
from pathlib import Path


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_load_quality_reads_all_three_benchmark_metrics(tmp_path: Path) -> None:
    from scripts import analyze_gpt56_session_group_results as analysis

    locomo = tmp_path / "locomo/terra/s2/locomo/eval_full.json"
    records = [{"judge_score": int(index < 300)} for index in range(314)]
    _write_json(locomo, {"n": 314, "overall": 300 / 314, "records": records})

    lme = (
        tmp_path
        / "longmemeval-s/terra/s2/hypotheses.jsonl.eval-results-gpt-4o-mini"
    )
    lme.parent.mkdir(parents=True)
    lme.write_text(
        json.dumps({"question_id": "a", "autoeval_label": {"label": True}})
        + "\n"
        + json.dumps({"question_id": "b", "autoeval_label": {"label": False}})
        + "\n",
        encoding="utf-8",
    )

    beam = tmp_path / "beam-100k/terra/s1/beam-semantic/terra/metrics.json"
    _write_json(
        beam,
        {
            "metric_scope": "rubric_nugget_only",
            "overall": {
                "questions": 40,
                "avg_score": 0.625,
                "pass_rate": 0.75,
            },
        },
    )

    assert analysis.load_quality(tmp_path, "locomo", "terra", 2) == {
        "metric": "locked_locomo_lj",
        "questions": 314,
        "score": 300 / 314,
    }
    assert analysis.load_quality(tmp_path, "longmemeval-s", "terra", 2) == {
        "metric": "official_longmemeval_qa_boolean",
        "questions": 2,
        "correct": 1,
        "score": 0.5,
    }
    assert analysis.load_quality(tmp_path, "beam-100k", "terra", 1) == {
        "metric": "beam_screening_rubric_nugget_mean",
        "questions": 40,
        "score": 0.625,
        "pass_rate": 0.75,
    }


def test_load_quality_marks_beam_luna_semantic_score_not_authorized(
    tmp_path: Path,
) -> None:
    from scripts import analyze_gpt56_session_group_results as analysis

    diagnostic = tmp_path / "beam-100k/luna/s1/beam-diagnostic/report.json"
    _write_json(
        diagnostic,
        {
            "metric_scope": "deterministic_string_match_diagnostic",
            "question_count": 40,
            "formal_scope_verified": False,
            "metrics": {
                "overall": {
                    "questions": 40,
                    "normalized_exact_match": 0.125,
                }
            },
        },
    )

    assert analysis.load_quality(tmp_path, "beam-100k", "luna", 1) == {
        "availability": "not_authorized",
        "reason": "beam_semantic_judge_not_authorized_for_luna",
        "questions": 40,
        "diagnostic": {
            "metric": "normalized_exact_match_diagnostic",
            "score": 0.125,
        },
    }


def test_load_qa_efficiency_requires_verified_audit(tmp_path: Path) -> None:
    from scripts import analyze_gpt56_session_group_results as analysis

    cell = tmp_path / "beam-100k/terra/s1"
    _write_json(
        cell / "completion.json",
        {"status": "complete", "question_count": 40, "run_ids": ["a", "b"]},
    )
    _write_json(
        cell / "audit.json",
        {
            "status": "verified_complete",
            "question_count": 40,
            "run_ids": ["a", "b"],
            "completion_sha256": hashlib.sha256(
                (cell / "completion.json").read_bytes()
            ).hexdigest(),
            "visible_tokens": 360000,
            "retrieval_model_calls": 100,
            "answer_model_calls": 40,
            "proxy": {"attempt_chains": {"authorized_retries": 8}},
        },
    )

    assert analysis.load_qa_efficiency(cell) == {
        "questions": 40,
        "visible_tokens": 360000,
        "visible_tokens_per_question": 9000.0,
        "retrieval_model_calls": 100,
        "retrieval_model_calls_per_question": 2.5,
        "answer_model_calls": 40,
        "authorized_retries": 8,
    }


def test_analyze_campaign_joins_only_audited_scored_cells(tmp_path: Path) -> None:
    from scripts import analyze_gpt56_session_group_results as analysis

    sizes = {
        "planned_runs": 4,
        "completed_runs": 4,
        "cells": [
            {
                "benchmark": "beam-100k",
                "tier": "terra",
                "nominal_session_group_size": 1,
                "runs": 2,
                "source_tokens": {"median": 1000},
            },
            {
                "benchmark": "locomo",
                "tier": "terra",
                "nominal_session_group_size": 2,
                "runs": 2,
                "source_tokens": {"median": 2000},
            },
        ],
    }
    sizes_path = tmp_path / "sizes.json"
    _write_json(sizes_path, sizes)

    qa_cell = tmp_path / "qa/beam-100k/terra/s1"
    _write_json(
        qa_cell / "completion.json",
        {"status": "complete", "question_count": 40, "run_ids": ["a", "b"]},
    )
    _write_json(
        qa_cell / "audit.json",
        {
            "status": "verified_complete",
            "question_count": 40,
            "run_ids": ["a", "b"],
            "completion_sha256": hashlib.sha256(
                (qa_cell / "completion.json").read_bytes()
            ).hexdigest(),
            "visible_tokens": 360000,
            "retrieval_model_calls": 100,
            "answer_model_calls": 40,
            "proxy": {"attempt_chains": {"authorized_retries": 8}},
        },
    )
    _write_json(
        tmp_path / "scores/beam-100k/terra/s1/beam-semantic/terra/metrics.json",
        {
            "metric_scope": "rubric_nugget_only",
            "overall": {"questions": 40, "avg_score": 0.625, "pass_rate": 0.75},
        },
    )

    report = analysis.analyze_campaign(
        sizes_report=sizes_path,
        qa_root=tmp_path / "qa",
        score_root=tmp_path / "scores",
        allow_partial=True,
    )

    assert report["status"] == "partial"
    assert report["planned_cells"] == 2
    assert report["analyzed_cells"] == 1
    assert report["pending_cells"] == ["locomo/terra/s2"]
    assert report["cells"][0]["quality"]["score"] == 0.625
    assert report["cells"][0]["qa_efficiency"]["visible_tokens_per_question"] == 9000


def test_cli_writes_report(monkeypatch, tmp_path: Path) -> None:
    from scripts import analyze_gpt56_session_group_results as analysis

    report = {"status": "partial", "planned_cells": 30, "analyzed_cells": 1}
    monkeypatch.setattr(analysis, "analyze_campaign", lambda **kwargs: report)
    output = tmp_path / "report.json"

    assert analysis.main(
        [
            "--sizes-report",
            str(tmp_path / "sizes.json"),
            "--qa-root",
            str(tmp_path / "qa"),
            "--score-root",
            str(tmp_path / "scores"),
            "--output",
            str(output),
            "--allow-partial",
        ]
    ) == 0
    assert json.loads(output.read_text()) == report
