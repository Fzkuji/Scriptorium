from __future__ import annotations

import json
from pathlib import Path

import tiktoken
import pytest


def test_analyze_run_uses_actual_trace_group_lengths(tmp_path: Path) -> None:
    from scripts import analyze_gpt56_session_group_sizes as analysis

    conversation = {
        "session_1": [
            {"dia_id": "D1:1", "text": "one"},
            {"dia_id": "D1:2", "text": "two"},
        ],
        "session_2": [{"dia_id": "D2:1", "text": "three"}],
        "session_3": [
            {"dia_id": "D3:1", "text": "four"},
            {"dia_id": "D3:2", "text": "five"},
        ],
    }
    records = [
        {
            "input_dia_ids": ["D1:1", "D1:2", "D2:1"],
            "turn_count": 3,
            "session_group_size": 2,
        },
        {
            "input_dia_ids": ["D3:1", "D3:2"],
            "turn_count": 2,
            "session_group_size": 1,
        },
    ]
    (tmp_path / "distill_trace.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    row = {
        "run_id": "run-s2",
        "benchmark": "locomo",
        "tier": "terra",
        "session_group_size": 2,
    }

    result = analysis.analyze_run(row, tmp_path, conversation)

    encoding = tiktoken.get_encoding("o200k_base")
    assert result["groups"] == [
        {
            "actual_sessions": 2,
            "messages": 3,
            "source_tokens": len(encoding.encode("one\ntwo\nthree")),
        },
        {
            "actual_sessions": 1,
            "messages": 2,
            "source_tokens": len(encoding.encode("four\nfive")),
        },
    ]
    assert result["source_coverage"] == {
        "messages": 5,
        "unique_messages": 5,
        "sessions": 3,
    }


def test_analyze_run_infers_sessions_when_trace_omits_group_size(
    tmp_path: Path,
) -> None:
    from scripts import analyze_gpt56_session_group_sizes as analysis

    conversation = {
        "session_1": [{"dia_id": "D1:1", "text": "one"}],
        "session_2": [{"dia_id": "D2:1", "text": "two"}],
    }
    (tmp_path / "distill_trace.jsonl").write_text(
        json.dumps(
            {
                "input_dia_ids": ["D1:1", "D2:1"],
                "turn_count": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = analysis.analyze_run(
        {
            "run_id": "run-s2",
            "benchmark": "locomo",
            "tier": "terra",
            "session_group_size": 2,
        },
        tmp_path,
        conversation,
    )

    assert result["groups"][0]["actual_sessions"] == 2


def test_summarize_runs_groups_matching_cells() -> None:
    from scripts import analyze_gpt56_session_group_sizes as analysis

    base = {
        "benchmark": "locomo",
        "tier": "terra",
        "nominal_session_group_size": 2,
    }
    summary = analysis.summarize_runs(
        [
            {
                **base,
                "run_id": "a",
                "groups": [
                    {"actual_sessions": 2, "messages": 3, "source_tokens": 5},
                    {"actual_sessions": 1, "messages": 2, "source_tokens": 3},
                ],
            },
            {
                **base,
                "run_id": "b",
                "groups": [
                    {"actual_sessions": 2, "messages": 4, "source_tokens": 7}
                ],
            },
        ]
    )

    assert summary == [
        {
            "benchmark": "locomo",
            "tier": "terra",
            "nominal_session_group_size": 2,
            "runs": 2,
            "writer_calls": 3,
            "actual_sessions": {
                "min": 1,
                "p25": 1,
                "median": 2,
                "p75": 2,
                "max": 2,
            },
            "messages": {
                "min": 2,
                "p25": 2,
                "median": 3,
                "p75": 4,
                "max": 4,
            },
            "source_tokens": {
                "min": 3,
                "p25": 3,
                "median": 5,
                "p75": 7,
                "max": 7,
            },
        }
    ]


def test_summarize_runs_adds_weighted_construction_metrics() -> None:
    from scripts import analyze_gpt56_session_group_sizes as analysis

    base = {
        "benchmark": "locomo",
        "tier": "terra",
        "nominal_session_group_size": 2,
        "groups": [{"actual_sessions": 2, "messages": 3, "source_tokens": 5}],
    }
    summary = analysis.summarize_runs(
        [
            {
                **base,
                "run_id": "a",
                "construction": {
                    "messages": 10,
                    "source_tokens": 100,
                    "calls": 5,
                    "input_tokens": 1000,
                    "output_tokens": 100,
                    "wall_time_s": 60,
                    "first_pass_source_ids": 8,
                    "post_verify_source_ids": 9,
                },
            },
            {
                **base,
                "run_id": "b",
                "construction": {
                    "messages": 20,
                    "source_tokens": 300,
                    "calls": 10,
                    "input_tokens": 2000,
                    "output_tokens": 300,
                    "wall_time_s": 180,
                    "first_pass_source_ids": 15,
                    "post_verify_source_ids": 18,
                },
            },
        ]
    )

    assert summary[0]["construction"] == {
        "messages": 30,
        "source_tokens": 400,
        "calls": 15,
        "input_tokens": 3000,
        "output_tokens": 400,
        "wall_time_s": 240.0,
        "calls_per_100_messages": 50.0,
        "input_tokens_per_1000_source_tokens": 7500.0,
        "output_tokens_per_1000_source_tokens": 1000.0,
        "wall_minutes_per_100_messages": 13.333333333333334,
        "first_pass_source_coverage": 23 / 30,
        "post_verify_source_coverage": 0.9,
        "verification_gain": 4 / 30,
    }


def test_evidence_metrics_measure_locomo_gold_references(tmp_path: Path) -> None:
    from scripts import analyze_gpt56_session_group_sizes as analysis

    memory = tmp_path / "memory"
    (memory / "topics").mkdir(parents=True)
    (memory / "timeline").mkdir()
    (memory / "topics" / "a.md").write_text(
        "fact one [D1:1]\nfact two [D2:1]\n", encoding="utf-8"
    )
    (memory / "timeline" / "a.md").write_text(
        "event one [D1:1]\nevent two [D2:1]\n", encoding="utf-8"
    )
    conversation = {
        "session_1": [
            {"dia_id": "D1:1", "text": "one"},
            {"dia_id": "D1:2", "text": "two"},
        ],
        "session_2": [{"dia_id": "D2:1", "text": "three"}],
    }
    questions = [
        {"evidence": ["D1:1"]},
        {"evidence": ["D1:2", "D2:1"]},
    ]

    result = analysis._evidence_metrics(
        benchmark="locomo",
        run_dir=tmp_path,
        conversation=conversation,
        questions=questions,
    )

    assert result == {
        "availability": "available",
        "source_ids": 3,
        "referenced_source_ids": 2,
        "reference_mentions": 4,
        "valid_reference_mentions": 4,
        "source_coverage": 2 / 3,
        "reference_precision": 1.0,
        "eligible_questions": 2,
        "gold_any_hits": 2,
        "gold_all_hits": 1,
        "unique_gold_targets": 3,
        "unique_gold_hits": 2,
    }


def test_summarize_runs_aggregates_available_evidence() -> None:
    from scripts import analyze_gpt56_session_group_sizes as analysis

    base = {
        "benchmark": "locomo",
        "tier": "terra",
        "nominal_session_group_size": 2,
        "groups": [{"actual_sessions": 2, "messages": 3, "source_tokens": 5}],
    }
    summary = analysis.summarize_runs(
        [
            {
                **base,
                "run_id": "a",
                "evidence": {
                    "availability": "available",
                    "source_ids": 10,
                    "referenced_source_ids": 8,
                    "reference_mentions": 20,
                    "valid_reference_mentions": 19,
                    "eligible_questions": 5,
                    "gold_any_hits": 5,
                    "gold_all_hits": 4,
                    "unique_gold_targets": 8,
                    "unique_gold_hits": 7,
                },
            },
            {
                **base,
                "run_id": "b",
                "evidence": {
                    "availability": "available",
                    "source_ids": 20,
                    "referenced_source_ids": 10,
                    "reference_mentions": 30,
                    "valid_reference_mentions": 30,
                    "eligible_questions": 5,
                    "gold_any_hits": 4,
                    "gold_all_hits": 3,
                    "unique_gold_targets": 12,
                    "unique_gold_hits": 9,
                },
            },
        ]
    )

    assert summary[0]["evidence"] == {
        "availability": "available",
        "source_coverage": 0.6,
        "reference_precision": 0.98,
        "gold_any_coverage": 0.9,
        "gold_all_coverage": 0.7,
        "unique_gold_coverage": 0.8,
        "source_ids": 30,
        "referenced_source_ids": 18,
        "eligible_questions": 10,
        "unique_gold_targets": 20,
    }


def test_analyze_campaign_reports_partial_scope(monkeypatch, tmp_path: Path) -> None:
    from scripts import analyze_gpt56_session_group_sizes as analysis

    rows = []
    for name in ("complete", "missing"):
        rows.append(
            {
                "run_id": name,
                "benchmark": "locomo",
                "tier": "luna",
                "model": "gpt-5.6-luna",
                "write_turns": "session",
                "session_group_size": 2,
                "output_dir": f"results/formal/gpt56-chunk-curve/locomo/{name}",
            }
        )
    matrix = tmp_path / "matrix.jsonl"
    matrix.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    build_root = tmp_path / "builds"
    complete = build_root / "locomo" / "complete"
    (complete / "memory" / "topics").mkdir(parents=True)
    (complete / "memory" / "timeline").mkdir()
    (complete / "build.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "run_id": "complete",
                    "builder_model": "gpt-5.6-luna",
                    "write_turns": "session",
                    "session_group_size": 2,
                    "messages": 1,
                    "source_tokens": 1,
                    "wall_time_s": 1,
                    "usage": {
                        "calls": 1,
                        "input_tokens": 1,
                        "output_tokens": 1,
                    },
                    "distill": {
                        "first_pass_unique_dia_ids": 1,
                        "post_verify_unique_dia_ids": 1,
                    },
                }
            ),
        encoding="utf-8",
    )
    (complete / "memory" / "_SUCCESS.json").write_text(
        json.dumps({"run_id": "complete"}), encoding="utf-8"
    )
    (complete / "distill_trace.jsonl").write_text(
        json.dumps(
            {
                "input_dia_ids": ["D1:1"],
                "turn_count": 1,
                "session_group_size": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        analysis.build_runner,
        "load_unit",
        lambda row: ({"session_1": [{"dia_id": "D1:1", "text": "one"}]}, [], {}),
    )

    report = analysis.analyze_campaign(
        matrix_path=matrix,
        build_roots=[build_root],
        allow_partial=True,
    )

    assert report["status"] == "partial"
    assert report["planned_runs"] == 2
    assert report["completed_runs"] == 1
    assert report["missing_run_ids"] == ["missing"]
    with pytest.raises(analysis.SessionGroupAnalysisError, match="missing 1 runs"):
        analysis.analyze_campaign(
            matrix_path=matrix,
            build_roots=[build_root],
            allow_partial=False,
        )


def test_cli_writes_the_analysis_report(monkeypatch, tmp_path: Path) -> None:
    from scripts import analyze_gpt56_session_group_sizes as analysis

    report = {"status": "partial", "planned_runs": 60, "completed_runs": 20}
    monkeypatch.setattr(analysis, "analyze_campaign", lambda **kwargs: report)
    output = tmp_path / "analysis.json"

    assert analysis.main(
        [
            "--matrix",
            str(tmp_path / "matrix.jsonl"),
            "--build-root",
            str(tmp_path / "a"),
            "--build-root",
            str(tmp_path / "b"),
            "--output",
            str(output),
            "--allow-partial",
        ]
    ) == 0

    assert json.loads(output.read_text(encoding="utf-8")) == report
