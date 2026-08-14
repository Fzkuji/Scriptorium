import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_v88_gpt55_locomo.py"
SPEC = importlib.util.spec_from_file_location("run_v88_gpt55_locomo", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_parse_samples_ranges():
    assert MOD.parse_samples("0-2,5,7-6") == [0, 1, 2, 5, 6, 7]


def test_validate_sample_requires_build_stats_answers_and_all_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(MOD, "expected_questions", lambda sample, limit=None: 2)
    path = tmp_path / "sample0_questions.json"
    records = [
        {"question_id": "_build_stats"},
        {"question_id": "s0_q0", "answer": "a"},
        {"question_id": "s0_q1", "answer": "b"},
    ]
    path.write_text(json.dumps(records))
    assert MOD.validate_sample(path, 0) == (True, "complete")
    records[-1]["answer"] = ""
    path.write_text(json.dumps(records))
    ok, reason = MOD.validate_sample(path, 0)
    assert not ok and reason == "missing_answers=1"


def test_validate_sample_checks_builder_model(tmp_path, monkeypatch):
    monkeypatch.setattr(MOD, "expected_questions", lambda sample, limit=None: 1)
    path = tmp_path / "sample0_questions.json"
    path.write_text(json.dumps([
        {"question_id": "_build_stats", "notes": "built fresh: model=gpt-4o-mini"},
        {"question_id": "s0_q0", "answer": "a"},
    ]))
    assert MOD.validate_sample(path, 0, expected_model="gpt-5.5") == (
        False, "builder_model_mismatch")


def test_experiment_env_freezes_v88_calendar_configuration():
    class Args:
        request_concurrency = 3
        model = "gpt-5.5"
        base_url = "http://127.0.0.1:8199/v1"
        api_key = "x"

    env = MOD.experiment_env(Args())
    assert env["NATIVEMEM_PROMPT"] == "v8"
    assert env["NATIVEMEM_CHUNK_TURNS"] == "6"
    assert env["NATIVEMEM_V8_SEGMENT"] == "fixed"
    assert env["NATIVEMEM_V8_SINGLE"] == "1"
    assert env["BUILDER_MODEL"] == "gpt-5.5"
    assert "NATIVEMEM_V9_PIPELINE" not in env


def test_formal_cli_requires_explicit_model_request_authorization(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        MOD.flex_evidence,
        "begin_child_invocation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("child proxy must not start without authorization")
        ),
    )
    output = tmp_path / "out"
    with pytest.raises(SystemExit):
        MOD.main([
            "--gateway-root", str(tmp_path / "gateway"),
            "--output-dir", str(output),
        ])
    assert "allow-model-requests" in capsys.readouterr().err
    assert not output.exists()


def test_force_cannot_resume_a_different_formal_fingerprint(tmp_path, monkeypatch):
    output = tmp_path / "out"
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": "different-formal-fingerprint",
                "provider_evidence": {
                    "schema": "openai-gpt55-flex-invocations/v1",
                    "active_run_id": None,
                    "invocations": [],
                },
            }
        )
    )
    monkeypatch.setattr(
        MOD.flex_evidence,
        "begin_child_invocation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fingerprint mismatch must fail before child proxy startup")
        ),
    )

    with pytest.raises(SystemExit, match="new output directory"):
        MOD.main(
            [
                "--gateway-root",
                str(tmp_path / "gateway"),
                "--output-dir",
                str(output),
                "--resume",
                "--force",
                "--allow-model-requests",
            ]
        )
