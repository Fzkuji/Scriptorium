from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_exact_campaign_scope_accepts_only_the_manifest_run_set() -> None:
    from scripts import analyze_gpt56_w32_builds as analyze

    expected = analyze.EXPECTED_RUN_IDS

    assert analyze.validate_exact_run_ids(tuple(reversed(expected))) == expected
    with pytest.raises(analyze.BuildAnalysisError, match="exact 12-run scope"):
        analyze.validate_exact_run_ids(expected[:-1])
    with pytest.raises(analyze.BuildAnalysisError, match="exact 12-run scope"):
        analyze.validate_exact_run_ids(
            (*expected, "gpt56-locomo-conv-44-luna-w32")
        )


def test_analyze_view_uses_canonical_source_parser_and_grounding_denominators(
    tmp_path: Path,
) -> None:
    from scripts import analyze_gpt56_w32_builds as analyze

    view = tmp_path / "topics"
    view.mkdir()
    payload = (
        "## Pets\n"
        "Adopted Poppy. [D1:1]\n"
        "Poppy grew quickly. [D1:2-3]\n"
        "Unsupported statement. [D9:9]\n"
        "Entry without a source.\n"
    )
    (view / "pets.md").write_text(payload, encoding="utf-8")

    metrics = analyze.analyze_view(
        view,
        source_ids={"D1:1", "D1:2", "D1:3", "D1:4"},
    )

    assert metrics["size"] == {
        "bytes": len(payload.encode("utf-8")),
        "files": 1,
        "entries": 4,
        "nonempty_lines": 5,
    }
    assert metrics["references"]["expanded_mentions"] == 4
    assert metrics["references"]["valid_mentions"] == 3
    assert metrics["references"]["invalid_mentions"] == 1
    assert metrics["references"]["unique_valid_ids"] == 3
    assert metrics["references"]["unique_invalid_id_count"] == 1
    assert metrics["references"]["unique_invalid_ids"] == ["D9:9"]
    assert metrics["references"]["reference_precision"] == 0.75
    assert metrics["grounding"] == {
        "grounded_entries": 2,
        "entry_groundedness": 0.5,
        "grounded_nonempty_lines": 2,
        "line_groundedness": 0.4,
    }


def test_gold_coverage_distinguishes_turn_session_and_unavailable_evidence() -> None:
    from scripts import analyze_gpt56_w32_builds as analyze

    locomo = analyze.gold_coverage(
        benchmark="locomo",
        questions=[
            {"evidence": ["D1:1"]},
            {"evidence": ["D1:2", "D1:3"]},
        ],
        valid_references={"D1:1", "D1:2"},
    )
    assert locomo == {
        "availability": "available",
        "level": "turn",
        "eligible_questions": 2,
        "question_any_coverage": 1.0,
        "question_all_coverage": 0.5,
        "unique_gold_targets": 3,
        "unique_gold_target_coverage": 2 / 3,
    }

    longmemeval = analyze.gold_coverage(
        benchmark="longmemeval-s",
        questions=[{"answer_session_ids": ["s2", "s3"]}],
        valid_references={"D2:1", "D8:1"},
        answer_session_numbers={2, 3},
    )
    assert longmemeval == {
        "availability": "available",
        "level": "session",
        "eligible_questions": 1,
        "question_any_coverage": 1.0,
        "question_all_coverage": 0.0,
        "unique_gold_targets": 2,
        "unique_gold_target_coverage": 0.5,
    }

    beam = analyze.gold_coverage(
        benchmark="beam-100k",
        questions=[{"question_id": "1-q0"}],
        valid_references={"D1:1"},
    )
    assert beam["availability"] == "not_available"
    assert beam["level"] is None
    assert beam["question_any_coverage"] is None
    assert beam["question_all_coverage"] is None
    assert beam["unique_gold_target_coverage"] is None
    assert "does not provide" in beam["reason"]


def test_longmemeval_answer_sessions_are_mapped_through_frozen_dataset_order(
    tmp_path: Path,
) -> None:
    from scripts import analyze_gpt56_w32_builds as analyze
    from scripts.gpt56_chunk_curve_qa_contract import RunBinding

    dataset = tmp_path / "longmemeval.json"
    _write_json(
        dataset,
        [
            {
                "question_id": "qid",
                "haystack_session_ids": ["s1", "s2", "s3"],
                "answer_session_ids": ["s2", "s3"],
            }
        ],
    )
    binding = RunBinding(
        run_id="gpt56-longmemeval-s-qid-terra-w32",
        row={
            "benchmark": "longmemeval-s",
            "selection": {"dataset_index": 0, "question_id": "qid"},
        },
        run_dir=tmp_path / "run",
    )
    validated = {
        "questions": [
            {"question_id": "qid", "answer_session_ids": ["s2", "s3"]}
        ],
        "raw_dataset": {"path": str(dataset)},
    }

    assert analyze.longmemeval_answer_session_numbers(binding, validated) == {
        2,
        3,
    }

    validated["questions"][0]["answer_session_ids"] = ["s1"]
    with pytest.raises(analyze.BuildAnalysisError, match="answer session IDs differ"):
        analyze.longmemeval_answer_session_numbers(binding, validated)


def test_analyze_run_reports_valid_source_and_locomo_gold_coverage(
    tmp_path: Path,
) -> None:
    from scripts import analyze_gpt56_w32_builds as analyze
    from scripts.gpt56_chunk_curve_qa_contract import RunBinding

    run_dir = tmp_path / "locomo" / "conv-44" / "terra" / "w32"
    memory = run_dir / "memory"
    (memory / "topics").mkdir(parents=True)
    (memory / "topics" / "pets.md").write_text(
        "[2026-01-01] Adopted Poppy. [D1:1]\n"
        "[2026-01-02] Poppy likes parks. [D1:3]\n"
        "[2026-01-03] Invalid source. [D9:9]\n",
        encoding="utf-8",
    )
    (memory / "timeline").mkdir()
    (memory / "timeline" / "2026-01-01.md").write_text(
        "[2026-01-01] Earlier event. [D1:2]\n"
        "[2026-01-02] Later event. [D1:4]\n",
        encoding="utf-8",
    )
    build = {
        "status": "complete",
        "run_id": "gpt56-locomo-conv-44-terra-w32",
        "benchmark": "locomo",
        "unit_id": "conv-44",
        "tier": "terra",
        "builder_model": "gpt-5.6-terra",
        "reasoning_effort": "none",
        "write_turns": 32,
        "messages": 4,
        "source_tokens": 100,
        "source_tokenizer": "chars_div_4_fallback",
        "wall_time_s": 20.0,
        "usage": {
            "calls": 10,
            "input_tokens": 1_000,
            "cached_input_tokens": 100,
            "output_tokens": 200,
            "reasoning_tokens": 0,
        },
        "phase_usage": {
            "v8_distill": {
                "calls": 2,
                "input_tokens": 300,
                "cached_input_tokens": 50,
                "output_tokens": 100,
                "reasoning_tokens": 0,
            }
        },
        "distill": {
            "writer_chunks": 2,
            "chunks_triggering_verify": 1,
            "first_pass_unique_dia_ids": 1,
            "post_verify_unique_dia_ids": 2,
        },
    }
    _write_json(run_dir / "build.json", build)
    success = {
        "run_id": build["run_id"],
        "builder_model": "gpt-5.6-terra",
        "write_turns": 32,
    }
    _write_json(memory / "_SUCCESS.json", success)
    unit = {
        "run_id": build["run_id"],
        "benchmark": "locomo",
        "unit_id": "conv-44",
        "selection": {"sample_index": 5, "sample_id": "conv-44"},
        "messages": 4,
        "source_tokens": 100,
        "source_tokenizer": "chars_div_4_fallback",
    }
    _write_json(run_dir / "unit.json", unit)
    dataset = tmp_path / "locomo.json"
    _write_json(dataset, {"frozen": "fixture"})
    binding = RunBinding(
        run_id=str(build["run_id"]),
        row={
            "run_id": build["run_id"],
            "benchmark": "locomo",
            "tier": "terra",
            "model": "gpt-5.6-terra",
            "write_turns": 32,
            "unit_id": "conv-44",
            "reasoning_effort": "none",
            "selection": {"sample_index": 5, "sample_id": "conv-44"},
        },
        run_dir=run_dir,
    )
    validated = {
        "conversation": {
            "session_1": [
                {"dia_id": "D1:1", "text": "a"},
                {"dia_id": "D1:2", "text": "b"},
                {"dia_id": "D1:3", "text": "c"},
                {"dia_id": "D1:4", "text": "d"},
            ]
        },
        "questions": [
            {"question_id": "q0", "evidence": ["D1:1"]},
            {"question_id": "q1", "evidence": ["D1:2", "D1:3"]},
            {"question_id": "q2", "evidence": []},
        ],
        "unit": unit,
        "raw_dataset": {
            "path": str(dataset),
            "sha256": analyze.sha256_file(dataset),
        },
    }

    row = analyze.analyze_run(binding, validated)

    assert row["identity"] == {
        "run_id": build["run_id"],
        "benchmark": "locomo",
        "unit_id": "conv-44",
        "tier": "terra",
        "builder_model": "gpt-5.6-terra",
        "write_turns": 32,
    }
    assert row["source"]["turns"] == 4
    assert row["source"]["bytes"] == len(b"a\nb\nc\nd")
    assert row["source"]["tokens"] == 100
    assert row["views"]["topics"]["references"]["unique_valid_ids"] == 2
    assert row["views"]["topics"]["references"]["unique_invalid_ids"] == [
        "D9:9"
    ]
    assert row["views"]["timeline"]["references"]["unique_valid_ids"] == 2
    assert row["views"]["topics"]["gold_coverage"][
        "question_all_coverage"
    ] == 0.5
    assert row["views"]["timeline"]["gold_coverage"][
        "question_any_coverage"
    ] == 0.5
    assert row["combined_references"]["unique_valid_ids"] == 4
    assert row["combined_references"]["gold_coverage"][
        "question_all_coverage"
    ] == 1.0
    assert row["cross_view_reference_overlap"] == {
        "shared_unique_valid_ids": 0,
        "shared_fraction_of_union": 0.0,
        "shared_fraction_of_smaller_view": 0.0,
    }
    abstract = row["memory_size"]["abstract"]
    assert abstract["files"] == 2
    assert abstract["entries"] == 5
    assert row["memory_size"]["abstract_to_source_byte_ratio"] == (
        abstract["bytes"] / len(b"a\nb\nc\nd")
    )
    assert row["cost"]["uncached_input_tokens"] == 900
    assert row["cost"]["usd"]["value"] is None
    assert row["cost"]["phase_usage"]["v8_distill"][
        "fraction_of_run_input_tokens"
    ] == 0.3
    assert row["normalizations"]["calls_per_100_messages"] == 250.0
    assert row["source_hashes"]["build"] == analyze.sha256_file(
        run_dir / "build.json"
    )
    assert row["source_hashes"]["success"] == analyze.sha256_file(
        memory / "_SUCCESS.json"
    )
    assert row["source_hashes"]["unit"] == analyze.sha256_file(
        run_dir / "unit.json"
    )
    assert row["source_hashes"]["raw_dataset"] == analyze.sha256_file(dataset)
    assert row["source_hashes"]["memory"] == analyze.memory_sha256(memory)
    assert row["validation"]["local_contract_validation"] == "passed"
    assert row["validation"]["proxy_request_audit"] == "not_performed"


def test_cost_summary_aggregates_overall_tier_benchmark_and_phase() -> None:
    from scripts import analyze_gpt56_w32_builds as analyze

    def row(run_id: str, tier: str, benchmark: str, multiplier: int) -> dict:
        return {
            "identity": {
                "run_id": run_id,
                "tier": tier,
                "benchmark": benchmark,
            },
            "cost": {
                "calls": 2 * multiplier,
                "input_tokens": 100 * multiplier,
                "cached_input_tokens": 20 * multiplier,
                "uncached_input_tokens": 80 * multiplier,
                "output_tokens": 30 * multiplier,
                "reasoning_tokens": 0,
                "wall_time_s": 5.0 * multiplier,
                "phase_usage": {
                    "v8_distill": {
                        "calls": multiplier,
                        "input_tokens": 40 * multiplier,
                        "cached_input_tokens": 10 * multiplier,
                        "uncached_input_tokens": 30 * multiplier,
                        "output_tokens": 20 * multiplier,
                        "reasoning_tokens": 0,
                    }
                },
            },
        }

    summary = analyze.summarize_cost(
        [
            row("r1", "terra", "locomo", 1),
            row("r2", "sol", "beam-100k", 2),
        ]
    )

    assert summary["overall"]["run_count"] == 2
    assert summary["overall"]["totals"]["calls"] == 6
    assert summary["overall"]["totals"]["input_tokens"] == 300
    assert summary["overall"]["totals"]["uncached_input_tokens"] == 240
    assert summary["overall"]["totals"]["wall_time_s"] == 15.0
    assert summary["overall"]["mean_per_run"]["input_tokens"] == 150.0
    assert summary["overall"]["phase_usage"]["v8_distill"][
        "fraction_of_group_input_tokens"
    ] == 0.4
    assert summary["by_tier"]["terra"]["totals"]["calls"] == 2
    assert summary["by_tier"]["sol"]["totals"]["calls"] == 4
    assert summary["by_benchmark"]["beam-100k"]["totals"][
        "output_tokens"
    ] == 60
    assert summary["overall"]["usd"]["value"] is None


def test_cli_defaults_and_explicit_run_ids_are_bound_to_current_campaign() -> None:
    from scripts import analyze_gpt56_w32_builds as analyze

    defaults = analyze.parse_args([])
    assert defaults.matrix == analyze.DEFAULT_MATRIX
    assert defaults.build_root == analyze.DEFAULT_BUILD_ROOT
    assert defaults.execution_manifest == analyze.DEFAULT_EXECUTION_MANIFEST
    assert defaults.output == analyze.DEFAULT_OUTPUT
    assert defaults.output.name == "build-quality-cost-v2.json"
    assert defaults.run_ids is None

    argv = ["--matrix", "/tmp/matrix.jsonl", "--build-root", "/tmp/builds"]
    argv.extend(["--output", "/tmp/report.json"])
    for run_id in reversed(analyze.EXPECTED_RUN_IDS):
        argv.extend(["--run-id", run_id])
    explicit = analyze.parse_args(argv)
    assert explicit.matrix == Path("/tmp/matrix.jsonl")
    assert explicit.build_root == Path("/tmp/builds")
    assert explicit.output == Path("/tmp/report.json")
    assert analyze.validate_exact_run_ids(explicit.run_ids) == (
        analyze.EXPECTED_RUN_IDS
    )


def test_write_report_is_deterministic_and_refuses_clobber(tmp_path: Path) -> None:
    from scripts import analyze_gpt56_w32_builds as analyze

    output = tmp_path / "report.json"
    report = {"schema_version": 1, "status": "complete", "runs": []}

    analyze.write_report_no_clobber(output, report)
    analyze.write_report_no_clobber(output, report)
    assert json.loads(output.read_text(encoding="utf-8")) == report

    with pytest.raises(analyze.BuildAnalysisError, match="clobber"):
        analyze.write_report_no_clobber(
            output,
            {"schema_version": 1, "status": "changed", "runs": []},
        )


def test_build_report_binds_exact_scope_sources_cost_and_claim_boundary(
    tmp_path: Path,
) -> None:
    from scripts import analyze_gpt56_w32_builds as analyze

    matrix = tmp_path / "matrix.jsonl"
    matrix.write_text('{"frozen":"matrix"}\n', encoding="utf-8")
    build_root = tmp_path / "builds"
    build_root.mkdir()
    manifest = build_root / "execution_manifest.json"
    _write_json(
        manifest,
        {
            "selected_run_ids": list(analyze.EXPECTED_RUN_IDS),
            "matrix": str(matrix.resolve()),
            "matrix_sha256": analyze.sha256_file(matrix),
            "reasoning_effort": "none",
            "provider": {"name": "subscription-proxy"},
        },
    )
    rows = []
    for run_id in analyze.EXPECTED_RUN_IDS:
        tier = "terra" if "-terra-" in run_id else "sol"
        benchmark = (
            "beam-100k"
            if "-beam-100k-" in run_id
            else "longmemeval-s"
            if "-longmemeval-s-" in run_id
            else "locomo"
        )
        rows.append(
            {
                "identity": {
                    "run_id": run_id,
                    "benchmark": benchmark,
                    "tier": tier,
                },
                "cost": {
                    "calls": 1,
                    "input_tokens": 10,
                    "cached_input_tokens": 2,
                    "uncached_input_tokens": 8,
                    "output_tokens": 3,
                    "reasoning_tokens": 0,
                    "wall_time_s": 1.0,
                    "phase_usage": {
                        "v8_distill": {
                            "calls": 1,
                            "input_tokens": 10,
                            "cached_input_tokens": 2,
                            "uncached_input_tokens": 8,
                            "output_tokens": 3,
                            "reasoning_tokens": 0,
                        }
                    },
                },
                "source_bindings": {"raw_dataset": "/frozen/data.json"},
                "source_hashes": {
                    "matrix_row": "0" * 64,
                    "build": "1" * 64,
                    "success": "2" * 64,
                    "unit": "3" * 64,
                    "raw_dataset": "4" * 64,
                    "conversation": "5" * 64,
                    "questions": "6" * 64,
                    "memory": "7" * 64,
                },
            }
        )

    first = analyze.build_report(
        rows=rows,
        matrix_path=matrix,
        build_root=build_root,
        execution_manifest_path=manifest,
    )
    second = analyze.build_report(
        rows=list(reversed(rows)),
        matrix_path=matrix,
        build_root=build_root,
        execution_manifest_path=manifest,
    )

    assert first == second
    assert first["status"] == "complete"
    assert first["run_count"] == 12
    assert first["run_ids"] == list(analyze.EXPECTED_RUN_IDS)
    assert first["cost_summary"]["overall"]["totals"]["calls"] == 12
    assert first["cost_summary"]["overall"]["usd"]["value"] is None
    assert first["source_hashes"]["matrix"] == analyze.sha256_file(matrix)
    assert first["source_hashes"]["execution_manifest"] == analyze.sha256_file(
        manifest
    )
    assert set(first["source_hashes"]) >= {
        "analyzer",
        "matrix",
        "qa_contract",
        "canonical_source_parser",
        "build_runner",
        "atomic_report_contract",
        "execution_manifest",
    }
    assert first["raw_dataset_bindings"] == [
        {"path": "/frozen/data.json", "sha256": "4" * 64}
    ]
    assert first["external_build_audit_dependency"]["status"] == (
        "required_separate_artifact"
    )
    assert first["claim_boundary"]["scope"] == (
        "exact 12 completed Terra/Sol GPT-5.6 W32 screening builds"
    )
    assert "semantic correctness" in " ".join(
        first["claim_boundary"]["not_established"]
    )
