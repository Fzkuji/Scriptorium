#!/usr/bin/env python3
"""Monitor the subscription LoCoMo run and invoke the locked legacy evaluator."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
DEFAULT_RUN = (
    ROOT
    / "results"
    / "gpt55-locomo-v88-calendar-thinking-off-fast-20260714"
)
JUDGE_MODEL = "openai/gpt-4o-mini"
JUDGE_PROFILE = "legacy-eval-full-date-tolerant-20260705"
LOCOMO_EVAL_SCRIPT = ROOT / "scripts" / "eval_full.py"
LOCOMO_EVAL_SHA256 = (
    "f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd"
)
EXPECTED_QUESTIONS = 1_986
EXPECTED_MAIN_QUESTIONS = 1_540


class MonitorError(RuntimeError):
    """The generation or evaluation artifact is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_locked_eval_script() -> str:
    """Refuse LoCoMo evaluation if the user-mandated evaluator changed."""

    if not LOCOMO_EVAL_SCRIPT.is_file() or LOCOMO_EVAL_SCRIPT.is_symlink():
        raise MonitorError(
            f"locked LoCoMo evaluator is missing or unsafe: {LOCOMO_EVAL_SCRIPT}"
        )
    actual = sha256_file(LOCOMO_EVAL_SCRIPT)
    if actual != LOCOMO_EVAL_SHA256:
        raise MonitorError(
            "scripts/eval_full.py changed; LoCoMo evaluation is blocked "
            f"(expected {LOCOMO_EVAL_SHA256}, got {actual})"
        )
    return actual


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def exclusive_supervisor_lock(run_dir: Path):
    """Prevent two monitors from writing status or launching evaluators."""

    lock_path = run_dir / ".monitor_and_evaluate.lock"
    if lock_path.is_symlink():
        raise MonitorError(f"supervisor lock is unsafe: {lock_path}")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MonitorError(
                f"another monitor/evaluator is active for {run_dir}"
            ) from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def launcher_is_active(run_dir: Path) -> bool:
    """Check the launcher's advisory lock without relying on process names."""

    lock_path = run_dir / ".launcher.lock"
    if lock_path.is_symlink() or not lock_path.is_file():
        raise MonitorError(f"launcher lock is missing or unsafe: {lock_path}")
    with lock_path.open("r+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False


def expected_questions() -> list[list[dict[str, Any]]]:
    data = read_json(DATA)
    if not isinstance(data, list) or len(data) != 10:
        raise MonitorError("LoCoMo data must contain exactly ten samples")
    result = [sample.get("qa", []) for sample in data]
    if sum(map(len, result)) != EXPECTED_QUESTIONS:
        raise MonitorError("LoCoMo question inventory differs from 1,986")
    return result


def valid_checkpoint_answers(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    state = read_json(path)
    answers = state.get("answers", {}) if isinstance(state, dict) else {}
    if not isinstance(answers, dict):
        return 0, 0
    valid = sum(
        isinstance(record, dict)
        and bool(str(record.get("answer", "")).strip())
        for record in answers.values()
    )
    return len(answers), valid


def proxy_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"records": 0, "successes": 0, "errors": 0}
    rows = []
    for raw in path.read_bytes().splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    successes = [row for row in rows if row.get("status") == "success"]
    return {
        "records": len(rows),
        "successes": len(successes),
        "errors": sum(row.get("status") == "error" for row in rows),
        "physical_attempts": sum(int(row.get("attempts", 0)) for row in rows),
        "reasoning_tokens": sum(
            int(
                ((row.get("usage") or {}).get("completion_tokens_details") or {})
                .get("reasoning_tokens", 0)
                or 0
            )
            for row in successes
        ),
        "last_timestamp": rows[-1].get("timestamp") if rows else None,
    }


def generation_snapshot(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    inventory = expected_questions()
    samples = {}
    saved = 0
    valid = 0
    finals = 0
    for sample, qas in enumerate(inventory):
        final = run_dir / f"sample{sample}_questions.json"
        checkpoint = run_dir / f"sample{sample}_questions.checkpoint.json"
        stored, nonempty = valid_checkpoint_answers(checkpoint)
        saved += stored
        valid += nonempty
        if final.exists():
            finals += 1
        samples[str(sample)] = {
            "expected_questions": len(qas),
            "checkpoint_questions": stored,
            "valid_checkpoint_answers": nonempty,
            "final_exists": final.exists(),
            "manifest_status": (
                (manifest.get("samples") or {}).get(str(sample), {}).get("status")
            ),
        }
    return {
        "checked_at": utc_now(),
        "phase": "generation",
        "generation_status": manifest.get("status", "missing"),
        "completed_sample_files": finals,
        "saved_question_records": saved,
        "valid_answers": valid,
        "expected_questions": EXPECTED_QUESTIONS,
        "samples": samples,
        "proxy": proxy_summary(run_dir / "proxy_requests.jsonl"),
    }


def validate_and_combine(run_dir: Path) -> tuple[list[dict[str, Any]], dict]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise MonitorError("generation manifest is not complete")
    if (
        manifest.get("benchmark") != "locomo"
        or manifest.get("method") != "NativeMem-v8.8+calendar"
        or manifest.get("backbone") != "gpt-5.5"
    ):
        raise MonitorError("generation manifest identity differs")

    inventory = expected_questions()
    builds: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    source_files = []
    seen: set[str] = set()
    for sample, qas in enumerate(inventory):
        state = (manifest.get("samples") or {}).get(str(sample), {})
        if state.get("status") != "complete":
            raise MonitorError(f"sample {sample} is not complete in the manifest")
        path = run_dir / f"sample{sample}_questions.json"
        records = read_json(path)
        if not isinstance(records, list):
            raise MonitorError(f"sample {sample} output is not a list")
        sample_builds = [
            record for record in records
            if isinstance(record, dict)
            and record.get("question_id") == "_build_stats"
        ]
        sample_questions = [
            record for record in records
            if isinstance(record, dict)
            and record.get("question_id") != "_build_stats"
        ]
        if len(sample_builds) != 1 or len(sample_questions) != len(qas):
            raise MonitorError(f"sample {sample} record count differs")
        for index, (record, qa) in enumerate(zip(sample_questions, qas)):
            question_id = f"s{sample}_q{index}"
            expected = {
                "question_id": question_id,
                "question": qa["question"],
                "gold": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                "category": qa.get("category"),
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise MonitorError(f"{question_id} differs from the dataset")
            if not str(record.get("answer", "")).strip():
                raise MonitorError(f"{question_id} has an empty answer")
            if question_id in seen:
                raise MonitorError(f"duplicate question id: {question_id}")
            seen.add(question_id)
        recorded_hash = (state.get("artifact") or {}).get("json_sha256")
        actual_hash = sha256_file(path)
        if recorded_hash != actual_hash:
            raise MonitorError(f"sample {sample} output hash differs")
        builds.append({**sample_builds[0], "sample": sample})
        questions.extend(sample_questions)
        source_files.append({
            "sample": sample,
            "path": str(path),
            "sha256": actual_hash,
            "questions": len(sample_questions),
        })
    if len(questions) != EXPECTED_QUESTIONS:
        raise MonitorError("combined LoCoMo output does not contain 1,986 questions")
    provenance = {
        "schema_version": 1,
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "fingerprint": manifest.get("fingerprint"),
        },
        "source_files": source_files,
        "questions": len(questions),
        "build_records": len(builds),
    }
    return builds + questions, provenance


def evaluation_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"scored": 0, "status": "not_started"}
    payload = read_json(path)
    records = payload.get("results", []) if isinstance(payload, dict) else []
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    return {
        "scored": sum(
            isinstance(record, dict) and "judge_score" in record
            for record in records
        ),
        "expected": meta.get("question_count"),
        "status": meta.get("status"),
    }


def _log_tail(path: Path, limit: int = 2_000) -> str:
    if not path.exists():
        return "log file was not created"
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def run_logged_command(
    command: list[str],
    *,
    log_path: Path,
    status_path: Path,
    phase: str,
    poll_seconds: float,
    progress_path: Path | None = None,
) -> None:
    """Run one deterministic stage and expose durable progress."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        try:
            while process.poll() is None:
                status: dict[str, Any] = {
                    "checked_at": utc_now(),
                    "phase": phase,
                    "process_pid": process.pid,
                    "log": str(log_path),
                }
                if progress_path is not None:
                    status["evaluation"] = evaluation_progress(progress_path)
                atomic_json(status_path, status)
                try:
                    process.wait(timeout=poll_seconds)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            raise
    if process.returncode != 0:
        raise MonitorError(
            f"{phase} exited with code {process.returncode}: "
            f"{_log_tail(log_path)}"
        )


def validate_source_audit(run_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    combined = run_dir / "questions_all.json"
    audit_path = run_dir / "audit.json"
    if not combined.is_file() or not audit_path.is_file():
        raise MonitorError("subscription source audit did not create its artifacts")
    report = read_json(audit_path)
    if not isinstance(report, dict):
        raise MonitorError("subscription source audit is not a JSON object")
    expected = {
        "schema_version": 1,
        "status": "passed",
        "benchmark": "LoCoMo",
        "method": "NativeMem-v8.8+calendar",
        "model": "gpt-5.5",
        "generation_provider": "chatgpt_pro_subscription",
        "formal_flex_result": False,
        "questions": EXPECTED_QUESTIONS,
        "cat1_4": EXPECTED_MAIN_QUESTIONS,
        "empty_answers": 0,
        "source_hashes_match": True,
    }
    mismatches = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    scoring_input = report.get("scoring_input", {})
    proxy = report.get("proxy_window", {})
    accounting = report.get("build_accounting", {})
    if (
        mismatches
        or scoring_input
        != {"path": str(combined), "sha256": sha256_file(combined)}
        or report.get("combined_sha256") != sha256_file(combined)
        or proxy.get("reasoning_tokens") != 0
        or proxy.get("thinking_off_verified") is not True
        or accounting.get("records") != 10
        or accounting.get("unavailable_imported") != 1
        or accounting.get("no_invented_counts") is not True
    ):
        raise MonitorError(
            f"subscription source audit failed validation: {mismatches}"
        )
    return combined, audit_path, report


def validate_legacy_eval_output(path: Path) -> dict[str, Any]:
    """Validate the exact JSON schema written by scripts/eval_full.py."""

    payload = read_json(path)
    records = payload.get("records", []) if isinstance(payload, dict) else []
    by_category = payload.get("by_category", {}) if isinstance(payload, dict) else {}
    if (
        not isinstance(payload, dict)
        or payload.get("n") != EXPECTED_MAIN_QUESTIONS
        or not isinstance(payload.get("overall"), (int, float))
        or not isinstance(records, list)
        or len(records) != EXPECTED_MAIN_QUESTIONS
        or sum(
            isinstance(record, dict)
            and record.get("category") in (1, 2, 3, 4)
            and record.get("judge_score") in (0, 1)
            for record in records
        ) != EXPECTED_MAIN_QUESTIONS
        or set(by_category)
        != {"multi-hop", "temporal", "open-domain", "single-hop"}
    ):
        raise MonitorError("legacy eval_full.json failed validation")
    return payload


def _legacy_eval_command(run_dir: Path) -> list[str]:
    """Execute eval_full.py exactly while keeping the API key out of OS argv."""

    runner = (
        "import os,runpy,sys; "
        "script,run_dir=sys.argv[1:3]; "
        "sys.argv=[script,run_dir,'gpt-5.5','unused','unused',"
        "os.environ['OPENROUTER_API_KEY']]; "
        "runpy.run_path(script,run_name='__main__')"
    )
    return [
        sys.executable,
        "-c",
        runner,
        str(LOCOMO_EVAL_SCRIPT),
        str(run_dir),
    ]


def run_evaluation(
    run_dir: Path, status_path: Path, *, poll_seconds: float,
) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise MonitorError("OPENROUTER_API_KEY is unavailable to the monitor")

    evaluator_hash = validate_locked_eval_script()
    source_audit = run_dir / "audit.json"
    output = run_dir / "eval_full.json"
    plan = {
        "schema_version": 1,
        "created_at": utc_now(),
        "benchmark": "LoCoMo",
        "scope": "category 1-4 only",
        "questions": EXPECTED_MAIN_QUESTIONS,
        "answer_source": "pre_generated_gpt-5.5_answers",
        "skip_answerer": True,
        "evaluator": str(LOCOMO_EVAL_SCRIPT),
        "evaluator_sha256": evaluator_hash,
        "judge_model": JUDGE_MODEL,
        "judge_profile": JUDGE_PROFILE,
        "judge_provider": "OpenRouter",
        "judge_temperature": 0,
        "output": str(output),
        "category_5": "excluded_by_locked_evaluator",
    }
    atomic_json(run_dir / "evaluation_locked_eval_full_plan.json", plan)

    run_logged_command(
        [
            sys.executable,
            "scripts/audit_v88_gpt55_locomo_subscription.py",
            str(run_dir),
        ],
        log_path=run_dir / "subscription_source_audit.log",
        status_path=status_path,
        phase="source_audit",
        poll_seconds=poll_seconds,
    )
    _, _, source_report = validate_source_audit(run_dir)
    if output.exists():
        score = validate_legacy_eval_output(output)
    else:
        run_logged_command(
            _legacy_eval_command(run_dir),
            log_path=run_dir / "evaluation_locked_eval_full.log",
            status_path=status_path,
            phase="score_locomo_eval_full",
            poll_seconds=poll_seconds,
        )
        score = validate_legacy_eval_output(output)

    atomic_json(status_path, {
        "checked_at": utc_now(),
        "phase": "complete",
        "generation_questions": EXPECTED_QUESTIONS,
        "evaluation_questions": EXPECTED_MAIN_QUESTIONS,
        "evaluation_scope": "category 1-4 only",
        "evaluator": str(LOCOMO_EVAL_SCRIPT),
        "evaluator_sha256": evaluator_hash,
        "judge_model": JUDGE_MODEL,
        "judge_profile": JUDGE_PROFILE,
        "source_audit": {
            "path": str(source_audit),
            "sha256": sha256_file(source_audit),
            "status": source_report.get("status"),
        },
        "locomo": {
            "questions": EXPECTED_MAIN_QUESTIONS,
            "output": str(output),
            "sha256": sha256_file(output),
            "overall": score["overall"],
            "by_category": score["by_category"],
        },
        "locomo_cat5": "excluded_by_locked_evaluator",
    })


def supervise(
    args: argparse.Namespace, run_dir: Path, status_path: Path
) -> None:
    plan = {
        "schema_version": 1,
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "monitor_poll_seconds": args.poll_seconds,
        "monitor_only": args.monitor_only,
        "post_generation": {
            "answer_source": "pre_generated_gpt-5.5_answers",
            "skip_answerer": True,
            "evaluator": str(LOCOMO_EVAL_SCRIPT),
            "evaluator_sha256": LOCOMO_EVAL_SHA256,
            "judge_model": JUDGE_MODEL,
            "judge_profile": JUDGE_PROFILE,
            "judge_provider": "OpenRouter",
            "source_audit": "subscription",
            "benchmark": "LoCoMo category 1-4 only",
            "questions": EXPECTED_MAIN_QUESTIONS,
            "category_5": "excluded_by_locked_evaluator",
            "automatic_restart": False,
        },
    }
    atomic_json(run_dir / "monitor_and_evaluate_plan.json", plan)

    last_signature = None
    inactive_running_polls = 0
    released_complete_polls = 0
    try:
        while True:
            snapshot = generation_snapshot(run_dir)
            launcher_active = launcher_is_active(run_dir)
            snapshot["launcher_active"] = launcher_active
            generation_status = snapshot["generation_status"]
            if generation_status == "running":
                released_complete_polls = 0
                inactive_running_polls = (
                    inactive_running_polls + 1 if not launcher_active else 0
                )
                if inactive_running_polls >= 2:
                    raise MonitorError(
                        "launcher lock was released while manifest remained running"
                    )
            elif generation_status == "complete":
                inactive_running_polls = 0
                if snapshot["proxy"].get("reasoning_tokens") != 0:
                    raise MonitorError(
                        "generation completed with nonzero reasoning tokens"
                    )
                if snapshot["completed_sample_files"] != 10:
                    raise MonitorError(
                        "generation manifest completed before all sample files existed"
                    )
                released_complete_polls = (
                    released_complete_polls + 1 if not launcher_active else 0
                )
            else:
                inactive_running_polls = 0
                released_complete_polls = 0
            snapshot["launcher_release_stable_polls"] = released_complete_polls
            atomic_json(status_path, snapshot)
            signature = (
                generation_status,
                snapshot["completed_sample_files"],
                snapshot["valid_answers"],
                snapshot["proxy"].get("errors"),
                launcher_active,
                released_complete_polls,
            )
            if signature != last_signature:
                print(json.dumps({
                    "checked_at": snapshot["checked_at"],
                    "generation_status": generation_status,
                    "completed_sample_files": snapshot["completed_sample_files"],
                    "valid_answers": snapshot["valid_answers"],
                    "expected_questions": snapshot["expected_questions"],
                    "active_samples": {
                        sample: state["valid_checkpoint_answers"]
                        for sample, state in snapshot["samples"].items()
                        if state["manifest_status"] == "running"
                    },
                    "proxy_errors": snapshot["proxy"].get("errors"),
                    "reasoning_tokens": snapshot["proxy"].get(
                        "reasoning_tokens"
                    ),
                    "launcher_active": launcher_active,
                    "launcher_release_stable_polls": released_complete_polls,
                }, ensure_ascii=False), flush=True)
                last_signature = signature
            if generation_status == "complete" and released_complete_polls >= 2:
                break
            if generation_status == "failed":
                raise MonitorError("generation manifest reports failure")
            time.sleep(args.poll_seconds)
        if args.monitor_only:
            atomic_json(status_path, {
                "checked_at": utc_now(),
                "phase": "generation_complete",
                "generation_questions": EXPECTED_QUESTIONS,
                "launcher_release_stable_polls": released_complete_polls,
                "evaluation_started": False,
            })
            return
        run_evaluation(
            run_dir,
            status_path,
            poll_seconds=args.poll_seconds,
        )
    except BaseException as exc:
        atomic_json(status_path, {
            "checked_at": utc_now(),
            "phase": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help="stop after generation completes without starting evaluation",
    )
    args = parser.parse_args(argv)
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be at least 1")
    run_dir = args.run_dir.expanduser().resolve()
    status_path = run_dir / "monitor_and_evaluate_status.json"
    with exclusive_supervisor_lock(run_dir):
        supervise(args, run_dir, status_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
