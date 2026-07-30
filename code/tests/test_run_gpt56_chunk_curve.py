from __future__ import annotations

import json

import pytest

from scripts import run_gpt56_chunk_curve as runner


def test_matrix_selection_is_low_cost_first() -> None:
    rows = runner.load_matrix(runner.DEFAULT_MATRIX)
    args = runner.parse_args([
        "--models", "luna",
        "--max-runs", "6",
    ])
    selected = runner.filter_rows(rows, args)

    assert len(selected) == 6
    assert {row["tier"] for row in selected} == {"luna"}
    assert {str(row["write_turns"]) for row in selected} == {"session"}
    assert [row["benchmark"] for row in selected] == [
        "beam-100k", "beam-100k",
        "longmemeval-s", "longmemeval-s",
        "locomo", "locomo",
    ]


def test_load_beam_unit_ignores_incompatible_huggingface_feature_metadata(
    tmp_path, monkeypatch,
) -> None:
    import pyarrow as pa

    fixture = json.loads(
        (runner.ROOT / "tests/fixtures/beam_hf_fixture.json").read_text(
            encoding="utf-8"
        )
    )["100K"][0]
    table = pa.Table.from_pylist([fixture])
    metadata = dict(table.schema.metadata or {})
    metadata[b"huggingface"] = json.dumps({
        "info": {
            "features": {
                "chat": {
                    "feature": {"_type": "Value", "dtype": "string"},
                    "_type": "List",
                }
            }
        }
    }).encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    arrow_path = tmp_path / "beam.arrow"
    with arrow_path.open("wb") as handle:
        with pa.ipc.new_stream(handle, table.schema) as writer:
            writer.write_table(table)
    monkeypatch.setattr(runner, "BEAM_ARROW", arrow_path)
    row = {
        "benchmark": "beam-100k",
        "selection": {
            "conversation_index": 0,
            "conversation_id": str(fixture["conversation_id"]),
        },
    }

    conversation, questions, metadata = runner.load_unit(row)

    assert conversation["session_1"]
    assert questions
    assert metadata["conversation_id"] == str(fixture["conversation_id"])


def test_trace_summary_keeps_first_pass_and_post_verify_separate(tmp_path) -> None:
    trace = tmp_path / "distill.jsonl"
    records = [
        {
            "first_pass_events": [{"dia_ids": ["D1:1"]}],
            "missing_coverage_points": ["Tokyo"],
            "verify_events": [{"dia_ids": ["D1:2"]}],
            "post_verify_events": [
                {"dia_ids": ["D1:1"]}, {"dia_ids": ["D1:2"]},
            ],
        },
        {
            "first_pass_events": [{"dia_ids": ["D1:3"]}],
            "missing_coverage_points": [],
            "verify_events": [],
            "post_verify_events": [{"dia_ids": ["D1:3"]}],
        },
    ]
    trace.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    assert runner.summarize_trace(trace) == {
        "writer_chunks": 2,
        "chunks_triggering_verify": 1,
        "first_pass_events": 2,
        "verify_added_events": 1,
        "post_verify_events": 3,
        "first_pass_unique_dia_ids": 2,
        "post_verify_unique_dia_ids": 3,
    }


def test_default_cli_is_plan_only(capsys) -> None:
    assert runner.main([
        "--models", "luna",
        "--windows", "session",
        "--max-runs", "1",
    ]) == 0
    output = capsys.readouterr().out
    assert "planned selection: 1 runs" in output
    assert "model requests sent: 0" in output


def test_runtime_hashes_cover_runner_and_memory_runtime() -> None:
    hashes = runner.runtime_source_hashes(runner.DEFAULT_MATRIX)
    assert {
        "matrix",
        "runner",
        "adapter",
        "runtime",
        "v10_memory",
        "v8_memory",
        "longmemeval_converter",
        "beam_converter",
        "locomo_evaluator",
    } == set(hashes)
    assert all(len(value) == 64 for value in hashes.values())


def test_frontier_preflight_requires_frozen_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("FRONTIER_API_KEY", "test-only")
    with pytest.raises(runner.RunError, match="frozen endpoint"):
        runner.provider_preflight(
            "frontier", "https://example.com/v1", {"gpt-5.6-luna"},
        )


def test_frontier_preflight_checks_model_and_hides_key(monkeypatch) -> None:
    monkeypatch.setenv("FRONTIER_API_KEY", "test-only-secret")
    monkeypatch.setattr(
        runner,
        "frontier_models",
        lambda base_url, api_key: {"gpt-5.6-luna"},
    )
    contract = runner.provider_preflight(
        "frontier", runner.FRONTIER_BASE_URL, {"gpt-5.6-luna"},
    )
    assert contract["auth_source"] == "FRONTIER_API_KEY"
    assert "test-only-secret" not in json.dumps(contract)


def test_runtime_environment_never_records_provider_key(monkeypatch) -> None:
    row = runner.load_matrix(runner.DEFAULT_MATRIX)[0]
    monkeypatch.setenv("FRONTIER_API_KEY", "test-only-secret")
    process_env, recorded_env = runner.runtime_environment(
        row, runner.FRONTIER_BASE_URL, "frontier",
    )
    assert process_env["BUILDER_KEY"] == "test-only-secret"
    assert "BUILDER_KEY" not in recorded_env
    assert "test-only-secret" not in json.dumps(recorded_env)
    assert recorded_env["NATIVEMEM_REASONING_EFFORT"] == "none"
    assert recorded_env["NATIVEMEM_OPENAI_MAX_RETRIES"] == "6"


def test_subscription_proxy_retries_transient_requests() -> None:
    row = runner.load_matrix(runner.DEFAULT_MATRIX)[0]
    _, recorded_env = runner.runtime_environment(
        row, "http://127.0.0.1:8220/v1", "subscription-proxy",
    )

    assert recorded_env["NATIVEMEM_OPENAI_MAX_RETRIES"] == "2"


def test_runtime_environment_allows_http_timeout_override(monkeypatch) -> None:
    row = runner.load_matrix(runner.DEFAULT_MATRIX)[0]
    monkeypatch.setenv("FRONTIER_API_KEY", "test-only-secret")
    monkeypatch.setenv("NATIVEMEM_HTTP_TIMEOUT", "900")

    process_env, recorded_env = runner.runtime_environment(
        row, runner.FRONTIER_BASE_URL, "frontier",
    )

    assert process_env["NATIVEMEM_HTTP_TIMEOUT"] == "900"
    assert recorded_env["NATIVEMEM_HTTP_TIMEOUT"] == "900"


def test_valid_complete_requires_build_and_marker(tmp_path) -> None:
    row = runner.load_matrix(runner.DEFAULT_MATRIX)[0]
    target = tmp_path / "run"
    target.mkdir()
    runner.atomic_json(target / "build.json", {
        "status": "complete",
        "run_id": row["run_id"],
        "builder_model": row["model"],
        "write_turns": row["write_turns"],
    })
    assert runner.valid_complete(target, row) is False
    runner.atomic_json(target / "memory" / "_SUCCESS.json", {
        "run_id": row["run_id"],
    })
    assert runner.valid_complete(target, row) is True
