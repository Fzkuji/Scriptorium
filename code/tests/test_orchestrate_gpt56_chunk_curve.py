from __future__ import annotations

import json

from scripts import orchestrate_gpt56_chunk_curve as orchestrator


def test_dry_run_never_executes_models(capsys) -> None:
    assert orchestrator.main([
        "--provider", "frontier",
        "--base-url", "https://api.frontier-intelligence.tech/v1",
    ]) == 0
    output = capsys.readouterr().out
    assert "dry run" in output
    assert "model requests sent: 0" in output


def test_valid_smoke_requires_build_marker_and_zero_reasoning(tmp_path) -> None:
    run_id = "gpt56-locomo-conv-48-luna-w32"
    build = tmp_path / "run" / "build.json"
    build.parent.mkdir(parents=True)
    build.write_text(json.dumps({
        "status": "complete",
        "run_id": run_id,
        "reasoning_effort": "none",
        "usage": {"reasoning_tokens": 0},
    }), encoding="utf-8")
    assert orchestrator.valid_smoke(build, run_id) is False
    marker = build.parent / "memory" / "_SUCCESS.json"
    marker.parent.mkdir()
    marker.write_text("{}", encoding="utf-8")
    assert orchestrator.valid_smoke(build, run_id) is True


def test_valid_smoke_rejects_wrong_run_or_reasoning(tmp_path) -> None:
    build = tmp_path / "run" / "build.json"
    marker = build.parent / "memory" / "_SUCCESS.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    build.write_text(json.dumps({
        "status": "complete",
        "run_id": "different",
        "reasoning_effort": "none",
        "usage": {"reasoning_tokens": 0},
    }), encoding="utf-8")
    assert orchestrator.valid_smoke(build, "expected") is False
