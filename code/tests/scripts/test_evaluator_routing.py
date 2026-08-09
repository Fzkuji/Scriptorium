"""Which evaluator scores a run, and with which credentials."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))

from scripts.runners.conversation import evaluation  # noqa: E402


def make_args(benchmark):
    return SimpleNamespace(
        benchmark=benchmark,
        model="deepseek-v4-flash",
        base_url="https://example.invalid",
        api_key="answerer-key",
        judge_api_key="judge-key",
    )


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(evaluation.subprocess, "run", fake_run)
    return calls


def test_locomo_runs_the_locked_evaluator(recorded, tmp_path: Path):
    evaluation.run_evaluator(
        make_args("locomo"), tmp_path, tmp_path / "sample9_questions.json"
    )

    command = recorded[0]["command"]
    assert command[1].endswith("eval_full.py")
    assert command[2] == str(tmp_path)
    # The locked evaluator takes credentials positionally, not by environment.
    assert command[-2:] == ["answerer-key", "judge-key"]
    assert recorded[0]["env"] is None


def test_beam_runs_the_unified_evaluator_with_its_benchmark(
    recorded, tmp_path: Path
):
    questions = tmp_path / "sample0_questions.json"

    evaluation.run_evaluator(make_args("beam"), tmp_path, questions)

    command = recorded[0]["command"]
    assert command[1:3] == ["-m", "scripts.evaluation.evaluate"]
    assert command[command.index("--benchmark") + 1] == "beam"
    assert command[command.index("--input") + 1] == str(questions)
    assert command[command.index("--output") + 1] == str(
        tmp_path / evaluation.RESULT_NAME
    )
    assert command[command.index("--metrics") + 1] == "judge"
    env = recorded[0]["env"]
    assert env["JUDGE_KEY"] == "judge-key"
    assert env["ANSWERER_KEY"] == "answerer-key"
    assert env["ANSWERER_MODEL"] == "deepseek-v4-flash"
    # The judge's model and endpoint are left alone, so a BEAM score is
    # comparable with a LoCoMo one.
    assert env.get("JUDGE_MODEL") == os.environ.get("JUDGE_MODEL")
    assert env.get("JUDGE_BASE") == os.environ.get("JUDGE_BASE")


def test_only_locomo_pays_for_the_hash_check(monkeypatch):
    checked = []
    monkeypatch.setattr(
        evaluation, "sha256_file", lambda path: checked.append(path) or "wrong"
    )

    with pytest.raises(RuntimeError, match="hash mismatch"):
        evaluation.verify_evaluator("locomo")
    evaluation.verify_evaluator("beam")

    assert len(checked) == 1


def test_a_failed_evaluator_names_its_log(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        evaluation.subprocess, "run",
        lambda command, **kwargs: SimpleNamespace(returncode=1),
    )

    with pytest.raises(RuntimeError, match="eval.log"):
        evaluation.run_evaluator(
            make_args("beam"), tmp_path, tmp_path / "sample0_questions.json"
        )
