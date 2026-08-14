import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "monitor_and_evaluate_gpt55_locomo_subscription.py"
)
SPEC = importlib.util.spec_from_file_location("monitor_locomo_subscription", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_valid_checkpoint_answers_counts_only_nonempty(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({
        "answers": {
            "0": {"answer": "yes"},
            "1": {"answer": ""},
            "2": {"question": "missing answer"},
        }
    }))

    assert MOD.valid_checkpoint_answers(path) == (3, 1)


def test_validate_and_combine_rejects_incomplete_manifest(tmp_path):
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "status": "running",
    }))

    with pytest.raises(MOD.MonitorError, match="not complete"):
        MOD.validate_and_combine(tmp_path)


def test_proxy_summary_reports_physical_attempts_and_reasoning(tmp_path):
    path = tmp_path / "proxy.jsonl"
    path.write_text(json.dumps({
        "status": "success",
        "attempts": 1,
        "timestamp": "2026-07-14T00:00:00+00:00",
        "usage": {
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }) + "\n" + json.dumps({
        "status": "error",
        "attempts": 1,
        "timestamp": "2026-07-14T00:01:00+00:00",
    }) + "\n")

    summary = MOD.proxy_summary(path)

    assert summary["successes"] == 1
    assert summary["errors"] == 1
    assert summary["physical_attempts"] == 2
    assert summary["reasoning_tokens"] == 0


def test_supervisor_requires_two_released_launcher_polls(monkeypatch, tmp_path):
    snapshot = {
        "checked_at": "2026-07-14T00:00:00+00:00",
        "phase": "generation",
        "generation_status": "complete",
        "completed_sample_files": 10,
        "saved_question_records": MOD.EXPECTED_QUESTIONS,
        "valid_answers": MOD.EXPECTED_QUESTIONS,
        "expected_questions": MOD.EXPECTED_QUESTIONS,
        "samples": {},
        "proxy": {"errors": 0, "reasoning_tokens": 0},
    }
    checks = []

    def fake_snapshot(_run_dir):
        checks.append(True)
        return json.loads(json.dumps(snapshot))

    monkeypatch.setattr(MOD, "generation_snapshot", fake_snapshot)
    monkeypatch.setattr(MOD, "launcher_is_active", lambda _run_dir: False)
    monkeypatch.setattr(MOD.time, "sleep", lambda _seconds: None)
    args = SimpleNamespace(
        poll_seconds=1,
        monitor_only=True,
    )
    status = tmp_path / "status.json"

    MOD.supervise(args, tmp_path, status)

    assert len(checks) == 2
    result = json.loads(status.read_text())
    assert result["phase"] == "generation_complete"
    assert result["launcher_release_stable_polls"] == 2


def test_locked_eval_script_matches_frozen_hash():
    assert MOD.validate_locked_eval_script() == MOD.LOCOMO_EVAL_SHA256


def test_run_evaluation_uses_only_locked_eval_full(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    commands = []
    combined = tmp_path / "questions_all.json"
    source_audit = tmp_path / "audit.json"
    combined.write_text("[]\n")
    source_audit.write_text("{}\n")

    def fake_run(command, **kwargs):
        del kwargs
        commands.append(command)
        if command[1] != "-c":
            return
        records = [
            {"category": 1, "judge_score": 1}
            for _ in range(MOD.EXPECTED_MAIN_QUESTIONS)
        ]
        (tmp_path / "eval_full.json").write_text(json.dumps({
            "overall": 1.0,
            "n": MOD.EXPECTED_MAIN_QUESTIONS,
            "by_category": {
                "multi-hop": 1.0,
                "temporal": 1.0,
                "open-domain": 1.0,
                "single-hop": 1.0,
            },
            "records": records,
        }))

    monkeypatch.setattr(MOD, "run_logged_command", fake_run)
    monkeypatch.setattr(
        MOD, "validate_source_audit",
        lambda _run_dir: (combined, source_audit, {"status": "passed"}),
    )

    MOD.run_evaluation(
        tmp_path,
        tmp_path / "status.json",
        poll_seconds=1,
    )

    assert len(commands) == 2
    assert commands[1][0:2] == [MOD.sys.executable, "-c"]
    assert str(MOD.LOCOMO_EVAL_SCRIPT) in commands[1]
    assert all("score_v88_gpt55_benchmarks.py" not in part for part in commands[1])
    plan = json.loads((tmp_path / "evaluation_locked_eval_full_plan.json").read_text())
    assert plan["evaluator_sha256"] == MOD.LOCOMO_EVAL_SHA256
    assert plan["questions"] == 1540
    assert plan["category_5"] == "excluded_by_locked_evaluator"
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["phase"] == "complete"
    assert status["evaluator_sha256"] == MOD.LOCOMO_EVAL_SHA256
