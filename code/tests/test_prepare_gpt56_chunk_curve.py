from __future__ import annotations

import json
import socket

import pytest

from scripts import prepare_gpt56_chunk_curve as prepare


def test_frozen_inputs_and_selected_unit_statistics() -> None:
    observed = prepare.validate_hashes()
    assert observed["scripts/eval_full.py"] == (
        "f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd"
    )

    locomo = {unit["unit_id"]: unit for unit in prepare.locomo_units()}
    assert locomo["conv-44"]["statistics"] == {
        "sessions": 28,
        "messages": 675,
        "max_session_messages": 47,
        "cat1_4_questions": 123,
    }
    assert locomo["conv-48"]["statistics"] == {
        "sessions": 30,
        "messages": 681,
        "max_session_messages": 44,
        "cat1_4_questions": 191,
    }

    longmemeval = {
        unit["unit_id"]: unit for unit in prepare.longmemeval_units()
    }
    assert longmemeval["2318644b"]["statistics"]["messages"] == 523
    assert longmemeval["2318644b"]["statistics"]["max_session_messages"] == 82
    assert longmemeval["gpt4_6dc9b45b"]["statistics"]["messages"] == 565
    assert (
        longmemeval["gpt4_6dc9b45b"]["statistics"]["max_session_messages"]
        == 132
    )

    beam = {unit["unit_id"]: unit for unit in prepare.beam_units()}
    assert beam["100K-conv-1"]["statistics"] == {
        "sessions": 3,
        "messages": 188,
        "session_messages": [60, 56, 72],
        "max_session_messages": 72,
        "questions": 20,
        "source_references": 188,
    }
    assert beam["100K-conv-2"]["statistics"] == {
        "sessions": 3,
        "messages": 200,
        "session_messages": [72, 62, 66],
        "max_session_messages": 72,
        "questions": 20,
        "source_references": 200,
    }


def test_manifest_and_matrix_freeze_one_controlled_variable() -> None:
    manifest = prepare.build_manifest()
    matrix = prepare.build_matrix(manifest)

    assert manifest["status"] == "planned"
    assert manifest["schema_version"] == 2
    assert manifest["model_requests_authorized"] is False
    assert manifest["planned_builds"] == 162
    assert [model["model"] for model in manifest["models"]] == [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    ]
    assert all(
        model["reasoning_effort"] == "none" for model in manifest["models"]
    )
    assert manifest["independent_variable"]["values"] == [
        4, 6, 8, 12, 16, 20, 24, 32, "session",
    ]
    assert manifest["fixed_method_environment"][
        "NATIVEMEM_V10_CONTEXT_MODE"
    ] == "none"
    assert len(matrix) == 162
    assert len({row["run_id"] for row in matrix}) == 162
    assert all(row["model_requests_authorized"] is False for row in matrix)
    assert all(
        row["environment"]["NATIVEMEM_V10_WRITE_TURNS"]
        == str(row["write_turns"])
        for row in matrix
    )
    assert {
        "scripts/prepare_gpt56_chunk_curve.py",
        "scripts/run_gpt56_chunk_curve.py",
        "scripts/run_v88_gpt55_longmemeval.py",
        "scripts/run_v88_gpt55_beam.py",
        "src/nativemem.py",
        "src/chatgpt_proxy.py",
        "src/v10_memory.py",
        "src/v8_memory.py",
        "src/adapters/run_nativemem.py",
    }.issubset(manifest["source_hashes"])


def test_plan_generation_uses_local_files_only(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_network(*_args, **_kwargs):
        raise AssertionError("plan generation attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket.socket, "connect", reject_network)

    output_dir = tmp_path / "plan"
    assert prepare.main(["--output-dir", str(output_dir)]) == 0

    manifest = json.loads(
        (output_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    rows = [
        json.loads(line)
        for line in (output_dir / "run_matrix.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest["planned_builds"] == 162
    assert len(rows) == 162
    assert not any(output_dir.rglob("memory"))
