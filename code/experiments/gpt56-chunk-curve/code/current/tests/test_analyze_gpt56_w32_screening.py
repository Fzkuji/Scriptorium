from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qa_admission_requires_verified_complete_audit() -> None:
    from scripts import analyze_gpt56_w32_screening as analyze

    valid = {
        "status": "verified_complete",
        "question_count": 356,
        "independently_reconstructed_questions": 356,
        "all_memory_unchanged": True,
        "proxy": {"all_successes_accounted": True},
    }
    analyze.validate_qa_audit(valid, tier="terra")

    for mutation in (
        {"status": "complete"},
        {"question_count": 355},
        {"independently_reconstructed_questions": 355},
        {"all_memory_unchanged": False},
        {"proxy": {"all_successes_accounted": False}},
    ):
        candidate = {**valid, **mutation}
        with pytest.raises(analyze.ScreeningAnalysisError):
            analyze.validate_qa_audit(candidate, tier="terra")


def test_completion_requires_the_exact_six_tier_w32_runs() -> None:
    from scripts import analyze_gpt56_w32_screening as analyze

    run_ids = [
        f"gpt56-{benchmark}-{unit.replace('_', '-')}-terra-w32"
        for benchmark, units in analyze.EXPECTED_UNITS.items()
        for unit in units
    ]
    completion = {
        "status": "complete",
        "question_count": 356,
        "run_ids": run_ids,
    }
    audit = {"run_ids": run_ids}
    analyze._validate_completion(completion, audit, tier="terra")

    wrong = {**completion, "run_ids": [*run_ids[:-1], "unrelated-terra-w32"]}
    wrong_audit = {"run_ids": wrong["run_ids"]}
    with pytest.raises(analyze.ScreeningAnalysisError, match="run IDs"):
        analyze._validate_completion(wrong, wrong_audit, tier="terra")


def test_artifact_bindings_cannot_escape_the_accepted_attempt(tmp_path: Path) -> None:
    from scripts import analyze_gpt56_w32_screening as analyze

    accepted = tmp_path / "question" / "attempt-0001"
    outside = tmp_path / "question" / "outside.jsonl"
    _write_jsonl(outside, [{"event": "outside"}])
    artifact_hash = _sha256(outside)
    result = {
        "artifacts": {
            "retrieval_model_ledger": "../outside.jsonl",
            "retrieval_model_ledger_sha256": artifact_hash,
            "answer_ledger": "../outside.jsonl",
            "answer_ledger_sha256": artifact_hash,
            "visible_token_manifest": "../outside.jsonl",
            "visible_token_manifest_sha256": artifact_hash,
            "attempt_manifest": "../outside.jsonl",
            "attempt_manifest_sha256": artifact_hash,
        }
    }
    with pytest.raises(analyze.ScreeningAnalysisError, match="outside accepted attempt"):
        analyze._validate_artifact_hashes(result, accepted)


def test_bundle_publish_preflights_every_conflict_before_writing(tmp_path: Path) -> None:
    from scripts import analyze_gpt56_w32_screening as analyze

    first = tmp_path / "analysis" / "input_manifest.json"
    conflict = tmp_path / "paper" / "gpt56_w32_main_results.tex"
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"existing-different-content")

    with pytest.raises(analyze.ScreeningAnalysisError, match="clobber"):
        analyze._publish_bundle_no_clobber(
            {
                first: b"new-manifest",
                conflict: b"new-table",
            }
        )

    assert not first.exists()
    assert conflict.read_bytes() == b"existing-different-content"


def test_paper_render_failure_publishes_no_partial_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import analyze_gpt56_w32_screening as analyze

    monkeypatch.setattr(analyze, "_main_table", lambda summary: "main")
    monkeypatch.setattr(analyze, "_quality_breakdown_table", lambda summary: "quality")
    monkeypatch.setattr(analyze, "_resource_breakdown_table", lambda summary: "resource")
    monkeypatch.setattr(analyze, "_figure_panels", lambda summary: [])

    def fail_pdf(path: Path, panels: object) -> None:
        raise analyze.ScreeningAnalysisError("forced PDF failure")

    monkeypatch.setattr(analyze, "_render_pdf", fail_pdf)
    paper_dir = tmp_path / "paper"
    with pytest.raises(analyze.ScreeningAnalysisError, match="forced PDF failure"):
        analyze.render_paper_artifacts(
            summary={},
            paired={},
            paper_dir=paper_dir,
        )
    assert not paper_dir.exists()


def test_pdf_plot_uses_only_embedded_truetype_fonts(tmp_path: Path) -> None:
    from scripts import analyze_gpt56_w32_screening as analyze

    output = tmp_path / "plot.pdf"
    panels = [
        {
            "tag": "(a)",
            "ylabel": "Score (%)",
            "pairs": [(80.0, 81.0), (82.0, 83.0)],
            "aggregate": (81.0, 82.0),
        }
    ]

    analyze._render_pdf(output, panels)

    payload = output.read_bytes()
    assert b"/BaseFont /Helvetica" not in payload
    assert b"/FontFile2" in payload


def test_question_extraction_uses_matching_attempt_and_first_opened_file(
    tmp_path: Path,
) -> None:
    from scripts import analyze_gpt56_w32_screening as analyze

    question_root = tmp_path / "questions" / "conv-44_q0"
    accepted = question_root / "attempt-0002"
    abandoned = question_root / "attempt-0001"

    retrieval_events = [
        {
            "event": "model_call_finished",
            "latency_s": 1.25,
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
            },
            "proxy_evidence": {
                "events": [
                    {
                        "status": "success",
                        "client_http_attempts": 1,
                        "upstream_http_attempts": 1,
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 10,
                            "total_tokens": 110,
                            "prompt_tokens_details": {"cached_tokens": 20},
                        },
                    }
                ]
            },
        },
        {
            "event": "model_call_finished",
            "latency_s": 0.75,
            "usage": {
                "prompt_tokens": 80,
                "completion_tokens": 5,
                "total_tokens": 85,
            },
            "proxy_evidence": {
                "events": [
                    {
                        "status": "success",
                        "client_http_attempts": 1,
                        "upstream_http_attempts": 1,
                        "usage": {
                            "prompt_tokens": 80,
                            "completion_tokens": 5,
                            "total_tokens": 85,
                            "prompt_tokens_details": {"cached_tokens": 0},
                        },
                    }
                ]
            },
        },
    ]
    _write_jsonl(accepted / "retrieval_model_ledger.jsonl", retrieval_events)
    _write_jsonl(
        accepted / "answer_ledger.jsonl",
        [
            {"event": "answer_started"},
            {"event": "answer_completed"},
        ],
    )
    _write_json(
        accepted / "visible_tokens.manifest.json",
        {
            "status": "complete",
            "summary": {
                "configured_budget_tokens": 200,
                "cumulative_visible_tokens": 200,
                "cumulative_source_resolution_tokens": 7,
                "exhausted": True,
            },
        },
    )
    _write_json(accepted / "attempt_manifest.json", {"attempt": 2})
    _write_json(abandoned / "attempt_manifest.json", {"attempt": 1})
    _write_json(abandoned / "result.json", {"status": "failed"})

    result = {
        "status": "complete",
        "question_id": "conv-44::q0",
        "retrieval": {
            "model_calls": 2,
            # Operation latency includes tool execution and orchestration overhead;
            # it is therefore not equal to the sum of model-call latencies (2.0s).
            "latency_s": 2.25,
            "access_log": [
                {
                    "path": None,
                    "source_ids": ["D1:2"],
                    "tool_name": "search_memory",
                },
                {
                    "path": "topics/wrong.md",
                    "source_ids": ["D1:9"],
                    "tool_name": "read_memory_file",
                },
                {
                    "path": "topics/right.md",
                    "source_ids": ["D1:1"],
                    "tool_name": "read_memory_file",
                },
            ],
        },
        "diagnostics": {
            "source_recall_eligible": True,
            "gold_source_ids": ["D1:1", "D1:2"],
            "mapped_source_hits": ["D1:1"],
            "mapped_source_recall": 0.5,
            "read_calls": 2,
            "tool_calls": 3,
            "tool_latency_s": 0.1,
            "first_relevant_file": "topics/right.md",
        },
        "budget": {
            "configured_tokens": 200,
            "visible_tokens": 200,
            "source_resolution_tokens": 7,
        },
        "answer": {
            "logical_calls": 1,
            "client_http_attempts": 1,
            "upstream_http_attempts": 1,
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 4,
                "total_tokens": 54,
                "prompt_tokens_details": {"cached_tokens": 5},
            },
        },
        "artifacts": {
            "retrieval_model_ledger": "retrieval_model_ledger.jsonl",
            "retrieval_model_ledger_sha256": _sha256(
                accepted / "retrieval_model_ledger.jsonl"
            ),
            "answer_ledger": "answer_ledger.jsonl",
            "answer_ledger_sha256": _sha256(accepted / "answer_ledger.jsonl"),
            "visible_token_manifest": "visible_tokens.manifest.json",
            "visible_token_manifest_sha256": _sha256(
                accepted / "visible_tokens.manifest.json"
            ),
            "attempt_manifest": "attempt_manifest.json",
            "attempt_manifest_sha256": _sha256(accepted / "attempt_manifest.json"),
        },
    }
    _write_json(accepted / "result.json", result)
    record = {
        "tier": "terra",
        "benchmark": "locomo",
        "unit_id": "conv-44",
        "question_id": "conv-44::q0",
        "original_question_id": "q0",
        "question": "Question?",
        "answer": "Answer.",
        "gold": "Gold.",
        "category": 1,
        "question_type": None,
        "run_id": "gpt56-locomo-conv-44-terra-w32",
        "result": result,
    }
    record_path = question_root / "record.json"
    _write_json(record_path, record)

    row, accepted_files = analyze.extract_question_row(record_path, tier="terra")

    assert row["access"]["first_file_hit"] is False
    assert row["access"]["first_relevant_read_rank"] == 2
    assert row["access"]["mapped_source_recall"] == 0.5
    assert row["access"]["retrieval_prompt_tokens"] == 180
    assert row["access"]["retrieval_cached_input_tokens"] == 20
    assert row["access"]["retrieval_uncached_input_tokens"] == 160
    assert row["access"]["retrieval_completion_tokens"] == 15
    assert row["access"]["retrieval_latency_s"] == 2.25
    assert row["access"]["answer_prompt_tokens"] == 50
    assert row["access"]["answer_cached_input_tokens"] == 5
    assert row["access"]["budget_hit"] is True
    assert row["access"]["retry_count"] == 1
    assert row["provenance"]["attempt_path"].endswith("attempt-0002")
    assert accepted / "result.json" in accepted_files

    non_locomo = {
        **record,
        "benchmark": "beam-100k",
        "unit_id": "100K-conv-1",
        "run_id": "gpt56-beam-100k-100K-conv-1-terra-w32",
        "question_type": "multi_session_reasoning",
        "category": None,
    }
    _write_json(record_path, non_locomo)
    with pytest.raises(analyze.ScreeningAnalysisError, match="LoCoMo"):
        analyze.extract_question_row(record_path, tier="terra")


def test_build_aggregation_is_weighted_and_groups_phases() -> None:
    from scripts import analyze_gpt56_w32_screening as analyze

    runs = []
    for tier, multiplier in (("terra", 1), ("sol", 2)):
        units = [
            ("locomo", "conv-44"),
            ("locomo", "conv-48"),
            ("longmemeval-s", "2318644b"),
            ("longmemeval-s", "gpt4_6dc9b45b"),
            ("beam-100k", "100K-conv-1"),
            ("beam-100k", "100K-conv-2"),
        ]
        for index, (benchmark, unit_id) in enumerate(units):
            source_turns = 10 + index
            source_tokens = 100 + index
            runs.append(
                {
                    "identity": {
                        "tier": tier,
                        "benchmark": benchmark,
                        "unit_id": unit_id,
                        "run_id": f"r-{tier}-{index}",
                        "write_turns": 32,
                    },
                    "source": {
                        "turns": source_turns,
                        "tokens": source_tokens,
                        "bytes": 400 + index,
                        "unique_source_ids": source_turns,
                    },
                    "cost": {
                        "calls": multiplier * source_turns,
                        "input_tokens": multiplier * source_tokens,
                        "cached_input_tokens": 0,
                        "uncached_input_tokens": multiplier * source_tokens,
                        "output_tokens": multiplier * 10,
                        "wall_time_s": multiplier * 60,
                        "phase_usage": {
                            "v8_distill": {
                                "calls": multiplier,
                                "input_tokens": 10,
                                "cached_input_tokens": 0,
                                "uncached_input_tokens": 10,
                                "output_tokens": 2,
                                "reasoning_tokens": 0,
                            },
                            "v8_distill_verify": {
                                "calls": multiplier,
                                "input_tokens": 8,
                                "cached_input_tokens": 0,
                                "uncached_input_tokens": 8,
                                "output_tokens": 1,
                                "reasoning_tokens": 0,
                            },
                            "v8_merge_lines": {
                                "calls": multiplier,
                                "input_tokens": 4,
                                "cached_input_tokens": 0,
                                "uncached_input_tokens": 4,
                                "output_tokens": 1,
                                "reasoning_tokens": 0,
                            },
                            "v8_consolidate": {
                                "calls": multiplier * (source_turns - 3),
                                "input_tokens": multiplier * source_tokens - 22,
                                "cached_input_tokens": 0,
                                "uncached_input_tokens": multiplier
                                * source_tokens
                                - 22,
                                "output_tokens": multiplier * 10 - 4,
                                "reasoning_tokens": 0,
                            },
                            "v8_sections": {
                                "calls": 0,
                                "input_tokens": 0,
                                "cached_input_tokens": 0,
                                "uncached_input_tokens": 0,
                                "output_tokens": 0,
                                "reasoning_tokens": 0,
                            },
                        },
                    },
                    "distill": {
                        "first_pass_unique_dia_ids": source_turns - 2,
                        "post_verify_unique_dia_ids": source_turns - 1,
                    },
                    "combined_references": {
                        "valid_mentions": source_turns,
                        "expanded_mentions": source_turns,
                        "gold_coverage": {
                            "availability": "not_available",
                        },
                    },
                    "memory_size": {
                        "abstract": {"bytes": 100, "files": 2, "entries": 3},
                        "source": {"bytes": 400 + index},
                    },
                }
            )
    report = {
        "status": "complete",
        "study": "gpt56-w32-screening-build-analysis",
        "run_count": 12,
        "runs": runs,
    }

    build = analyze.validate_and_aggregate_build(report)

    assert build["terra"]["normalized"]["calls_per_100_messages"] == 100.0
    assert build["sol"]["normalized"]["calls_per_100_messages"] == 200.0
    assert build["terra"]["phase_groups"]["writer"]["calls"] == 6
    assert build["sol"]["phase_groups"]["verify"]["calls"] == 12
    assert build["sol"]["phase_groups"]["maintenance"]["calls"] == 126
    assert build["terra"]["grounding"]["reference_precision"] == 1.0
    expected = sum(8 + index for index in range(6)) / sum(
        10 + index for index in range(6)
    )
    assert build["terra"]["grounding"]["first_pass_source_coverage"] == expected

    drifted = json.loads(json.dumps(report))
    drifted["runs"][0]["cost"]["calls"] += 1
    with pytest.raises(analyze.ScreeningAnalysisError, match="phase totals"):
        analyze.validate_and_aggregate_build(drifted)


def test_beam_metrics_are_recomputed_and_drift_is_rejected() -> None:
    from scripts import analyze_gpt56_w32_screening as analyze

    evaluations = []
    for type_index, question_type in enumerate(analyze.BEAM_QUESTION_TYPES):
        for within_type in range(4):
            index = type_index * 4 + within_type
            evaluations.append(
                {
                    "question_type": question_type,
                    "score": (0.25, 0.5, 0.75, 1.0)[within_type],
                    "nugget_scores": [
                        {"score": (0.25, 0.5, 0.75, 1.0)[within_type]}
                        for _ in range(3 if index < 23 else 2)
                    ],
                }
            )

    def group(items: list[dict[str, object]]) -> dict[str, object]:
        scores = [float(item["score"]) for item in items]
        passed = sum(score >= 0.5 for score in scores)
        return {
            "questions": len(items),
            "rubric_nuggets": sum(len(item["nugget_scores"]) for item in items),
            "avg_score": sum(scores) / len(scores),
            "pass_threshold": 0.5,
            "passed": passed,
            "pass_rate": passed / len(scores),
            "pass_accuracy_percent": passed / len(scores) * 100,
        }

    metrics = {
        "overall": group(evaluations),
        "by_question_type": {
            question_type: group(
                [
                    item
                    for item in evaluations
                    if item["question_type"] == question_type
                ]
            )
            for question_type in analyze.BEAM_QUESTION_TYPES
        },
    }
    recomputed = analyze._validate_beam_metrics(metrics, evaluations)
    assert recomputed["overall"]["questions"] == 40
    assert recomputed["overall"]["rubric_nuggets"] == 103

    drifted = json.loads(json.dumps(metrics))
    drifted["overall"]["avg_score"] += 0.01
    with pytest.raises(analyze.ScreeningAnalysisError, match="does not recompute"):
        analyze._validate_beam_metrics(drifted, evaluations)

    inconsistent = json.loads(json.dumps(evaluations))
    inconsistent[0]["score"] = 1.0
    with pytest.raises(analyze.ScreeningAnalysisError, match="nugget"):
        analyze._validate_beam_metrics(metrics, inconsistent)


def test_paired_results_keep_fixed_inventory_and_descriptive_claim_boundary() -> None:
    from scripts import analyze_gpt56_w32_screening as analyze

    rows: list[dict[str, object]] = []
    locomo_specs: list[tuple[str, str]] = []
    for category, count in analyze.LOCOMO_CATEGORY_COUNTS.items():
        start = len(locomo_specs)
        locomo_specs.extend(
            (category, f"q{start + offset}") for offset in range(count)
        )

    def add_pair(
        benchmark: str,
        unit_id: str,
        question_id: str,
        *,
        category_name: str | None = None,
        question_type: str | None = None,
        index: int,
    ) -> None:
        for tier in analyze.TIERS:
            correct = tier == "sol" or index % 2 == 0
            rows.append(
                {
                    "tier": tier,
                    "benchmark": benchmark,
                    "unit_id": unit_id,
                    "original_question_id": question_id,
                    "question_sha256": f"question-{benchmark}-{unit_id}-{question_id}",
                    "category_id": (
                        analyze.LOCOMO_CATEGORY_ORDER.index(category_name) + 1
                        if category_name is not None
                        else None
                    ),
                    "category_name": category_name,
                    "question_type": question_type,
                    "quality": {
                        "score": float(correct),
                        "correct": correct,
                        "pass": correct,
                    },
                    "access": {metric: index + (tier == "sol") for metric in analyze.ACCESS_METRICS},
                    "_join": {
                        "question": f"Question {question_id}",
                        "gold": f"Gold {question_id}",
                        "rubric": [f"Rubric {question_id}"],
                        "selection": {"id": question_id},
                    },
                }
            )

    for index, (category, question_id) in enumerate(locomo_specs):
        add_pair(
            "locomo",
            analyze.EXPECTED_UNITS["locomo"][index % 2],
            question_id,
            category_name=category,
            index=index,
        )
    for index in range(40):
        add_pair(
            "beam-100k",
            analyze.EXPECTED_UNITS["beam-100k"][index // 20],
            f"beam-q{index}",
            question_type=analyze.BEAM_QUESTION_TYPES[index // 4],
            index=index,
        )
    for index, question_id in enumerate(analyze.LME_IDS):
        add_pair(
            "longmemeval-s",
            analyze.EXPECTED_UNITS["longmemeval-s"][index],
            question_id,
            index=index,
        )

    build = {
        tier: {
            "normalized": {
                "calls_per_100_messages": 50 + 5 * (tier == "sol"),
                "input_tokens_per_1000_source_tokens": 6000
                + 1000 * (tier == "sol"),
                "output_tokens_per_1000_source_tokens": 700,
                "wall_minutes_per_100_messages": 5,
            }
        }
        for tier in analyze.TIERS
    }
    paired = analyze.build_paired_results(rows, build)

    assert paired["pair_count"] == 356
    assert paired["quality"]["locomo"]["n"] == 314
    assert paired["quality"]["beam_100k"]["n"] == 40
    assert paired["quality"]["longmemeval_s"]["n"] == 2
    assert paired["inference"]["status"] == "descriptive_only"

    mismatched = json.loads(json.dumps(rows))
    mismatched[1]["question_sha256"] = "different-question"
    with pytest.raises(analyze.ScreeningAnalysisError, match="pair identity"):
        analyze.build_paired_results(mismatched, build)


def test_main_table_has_no_overfull_box_in_aaai_submission_template(
    tmp_path: Path,
) -> None:
    from scripts import analyze_gpt56_w32_screening as analyze

    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        pytest.skip("pdflatex is unavailable")
    summary = {
        "quality": {
            tier: {
                "locomo": {"overall_lj": 0.55},
                "beam_100k": {"avg_score": 0.50},
                "longmemeval_s": {"correct": 1},
            }
            for tier in analyze.TIERS
        },
        "build": {
            tier: {
                "normalized": {
                    "calls_per_100_messages": 55.0,
                    "input_tokens_per_1000_source_tokens": 7000.0,
                }
            }
            for tier in analyze.TIERS
        },
    }
    document = "\n".join(
        [
            r"\documentclass[letterpaper]{article}",
            r"\usepackage[submission]{aaai2027}",
            r"\usepackage{booktabs}",
            r"\title{Table check}",
            r"\author{Anonymous Submission}",
            r"\affiliations{}",
            r"\begin{document}",
            r"\maketitle",
            analyze._main_table(summary),
            r"\end{document}",
        ]
    )
    tex_path = tmp_path / "table_check.tex"
    tex_path.write_text(document, encoding="utf-8")
    environment = dict(os.environ)
    environment["TEXINPUTS"] = f"{analyze.ROOT / 'paper'}{os.pathsep}"
    completed = subprocess.run(
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    log = (tmp_path / "table_check.log").read_text(encoding="utf-8")
    assert "Overfull \\hbox" not in log


def test_latex_caption_precedes_tabular_and_figure_has_pdf_png(
    tmp_path: Path,
) -> None:
    from scripts import analyze_gpt56_w32_screening as analyze

    summary = {
        "quality": {
            tier: {
                "locomo": {
                    "n": 314,
                    "overall_lj": 0.5 + 0.1 * (tier == "sol"),
                    "by_category": {
                        name: 0.5 for name in analyze.LOCOMO_CATEGORY_ORDER
                    },
                    "category_counts": {
                        name: count
                        for name, count in analyze.LOCOMO_CATEGORY_COUNTS.items()
                    },
                    "by_unit": {"conv-44": 0.5, "conv-48": 0.6},
                },
                "beam_100k": {
                    "n": 40,
                    "rubric_nuggets": 103,
                    "avg_score": 0.4 + 0.1 * (tier == "sol"),
                    "pass_rate": 0.5,
                    "by_question_type": {
                        name: {"n": 4, "avg_score": 0.5, "pass_rate": 0.5}
                        for name in analyze.BEAM_QUESTION_TYPES
                    },
                    "by_conversation": {
                        "100K-conv-1": {"n": 20, "avg_score": 0.4},
                        "100K-conv-2": {"n": 20, "avg_score": 0.5},
                    },
                },
                "longmemeval_s": {
                    "n": 2,
                    "correct": 1,
                    "items": [
                        {"question_id": "2318644b", "correct": True},
                        {"question_id": "gpt4_6dc9b45b", "correct": False},
                    ],
                },
            }
            for tier in analyze.TIERS
        },
        "build": {
            tier: {
                "normalized": {
                    "calls_per_100_messages": 50 + 5 * (tier == "sol"),
                    "input_tokens_per_1000_source_tokens": 6000
                    + 1000 * (tier == "sol"),
                    "output_tokens_per_1000_source_tokens": 700,
                    "wall_minutes_per_100_messages": 5.0,
                },
                "totals": {"calls": 1, "input_tokens": 2, "output_tokens": 3},
                "grounding": {
                    "first_pass_source_coverage": 0.6,
                    "post_verify_source_coverage": 0.8,
                    "reference_precision": 1.0,
                },
                "representation": {
                    "abstract_to_source_byte_ratio": 1.0,
                    "abstract_files": 10,
                    "abstract_entries": 20,
                },
                "phase_groups": {
                    name: {"calls": 1, "input_tokens": 2, "output_tokens": 3}
                    for name in ("writer", "verify", "maintenance")
                },
                "units": [
                    {
                        "benchmark": "locomo",
                        "unit_id": "conv-44",
                        "calls_per_100_messages": 50,
                        "input_tokens_per_1000_source_tokens": 6000,
                    },
                    {
                        "benchmark": "locomo",
                        "unit_id": "conv-48",
                        "calls_per_100_messages": 55,
                        "input_tokens_per_1000_source_tokens": 6500,
                    },
                ],
            }
            for tier in analyze.TIERS
        },
        "qa_access": {
            tier: {
                "overall": analyze.aggregate_access([]),
                "by_benchmark": {},
            }
            for tier in analyze.TIERS
        },
    }
    paired = {"quality": {"longmemeval_s": {"items": []}}}

    artifacts = analyze.render_paper_artifacts(
        summary=summary,
        paired=paired,
        paper_dir=tmp_path,
    )

    main = (tmp_path / "gpt56_w32_main_results.tex").read_text(encoding="utf-8")
    assert main.index("\\caption{") < main.index("\\begin{tabular}")
    assert "Fixed $W=32$" in main
    assert "x/2" in main
    breakdown = (
        tmp_path / "gpt56_w32_quality_breakdown.tex"
    ).read_text(encoding="utf-8")
    assert breakdown.index("\\caption{") < breakdown.index("\\begin{tabular}")
    resource = (
        tmp_path / "gpt56_w32_resource_breakdown.tex"
    ).read_text(encoding="utf-8")
    assert resource.index("\\caption{") < resource.index("\\begin{tabular}")
    pdf = (tmp_path / "gpt56_w32_quality_cost.pdf").read_bytes()
    assert pdf.startswith(b"%PDF")
    assert b"/FontFile2" in pdf
    assert (tmp_path / "gpt56_w32_quality_cost.png").read_bytes().startswith(
        b"\x89PNG"
    )
    assert len(artifacts) == 5

    assert analyze.render_paper_artifacts(
        summary=summary,
        paired=paired,
        paper_dir=tmp_path,
    ) == artifacts
