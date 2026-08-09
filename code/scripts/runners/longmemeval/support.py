"""Dataset conversion and resumable state for the NativeMem LongMemEval runner."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from scripts.runners.common import atomic_json, read_json, sha256_file, utc_now
from src.build import BuildPaused


CODE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = (
    CODE_ROOT / "benchmarks" / "longmemeval" / "data"
    / "longmemeval_s_cleaned.json"
)
EXPECTED_LONGMEMEVAL_SIZE = 500
SCHEMA_VERSION = 1


class DataValidationError(ValueError):
    pass


class ExistingStateError(RuntimeError):
    pass


class EmptyStageError(RuntimeError):
    pass


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=CODE_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


_DATE_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")


def normalize_date(raw: object) -> str:
    match = _DATE_RE.search(str(raw or "").strip())
    if not match:
        raise DataValidationError(f"unparseable LongMemEval date: {raw!r}")
    try:
        return date(*(int(value) for value in match.groups())).isoformat()
    except ValueError as exc:
        raise DataValidationError(f"invalid LongMemEval date: {raw!r}") from exc


def validate_item(item: object, index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise DataValidationError(f"item {index} is not an object")
    required = (
        "question_id", "question_type", "question", "answer", "question_date",
        "answer_session_ids", "haystack_sessions", "haystack_dates",
        "haystack_session_ids",
    )
    missing = [key for key in required if key not in item]
    if missing:
        raise DataValidationError(f"item {index} missing fields: {missing}")
    sessions = item["haystack_sessions"]
    dates = item["haystack_dates"]
    session_ids = item["haystack_session_ids"]
    if not isinstance(sessions, list) or not sessions:
        raise DataValidationError(f"item {index} has no haystack sessions")
    if not isinstance(dates, list) or not isinstance(session_ids, list):
        raise DataValidationError(f"item {index} session metadata is not a list")
    if len(sessions) != len(dates) or len(sessions) != len(session_ids):
        raise DataValidationError(
            f"item {index} session/date/id lengths differ: "
            f"{len(sessions)}/{len(dates)}/{len(session_ids)}"
        )
    if not str(item["question_id"]).strip() or not str(item["question"]).strip():
        raise DataValidationError(f"item {index} has an empty question id/text")
    for raw_date in dates:
        normalize_date(raw_date)
    if item.get("question_date"):
        normalize_date(item["question_date"])
    return item


def to_conversation(item: object, index: int = 0) -> dict[str, Any]:
    sample = validate_item(item, index)
    conversation: dict[str, Any] = {
        "speaker_a": "user",
        "speaker_b": "assistant",
    }
    for session_number, (session, raw_date) in enumerate(
        zip(sample["haystack_sessions"], sample["haystack_dates"]), start=1
    ):
        if not isinstance(session, list) or not session:
            raise DataValidationError(
                f"item {index} session {session_number} is empty or not a list"
            )
        turns = []
        for turn_number, turn in enumerate(session, start=1):
            if not isinstance(turn, dict):
                raise DataValidationError(
                    f"item {index} session {session_number} turn "
                    f"{turn_number} is not an object"
                )
            role = str(turn.get("role", "")).strip()
            content = turn.get("content")
            if not role or not isinstance(content, str):
                raise DataValidationError(
                    f"item {index} session {session_number} turn "
                    f"{turn_number} has an empty role or non-string content"
                )
            turns.append({
                "speaker": role,
                "text": content,
                "dia_id": f"D{session_number}:{turn_number}",
            })
        conversation[f"session_{session_number}"] = turns
        conversation[f"session_{session_number}_date_time"] = normalize_date(raw_date)
    return conversation


def load_dataset(path: Path, expected_count: int | None = None) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise DataValidationError("LongMemEval data root must be a list")
    if expected_count is not None and len(data) != expected_count:
        raise DataValidationError(
            f"expected {expected_count} LongMemEval items, found {len(data)}"
        )
    for index, item in enumerate(data):
        validate_item(item, index)
    return data


def validate_longmemeval_s(data: list[dict[str, Any]]) -> None:
    mean_sessions = sum(len(item["haystack_sessions"]) for item in data) / len(data)
    if mean_sessions < 20:
        raise DataValidationError(
            f"dataset averages only {mean_sessions:.1f} sessions/item; this is "
            "likely LongMemEval-oracle, not LongMemEval-S"
        )


def select_indices(total: int, start: int, limit: int | None) -> list[int]:
    if start < 0 or start >= total:
        raise DataValidationError(f"--start must be in 0..{total - 1}")
    if limit is not None and limit < 1:
        raise DataValidationError("--limit must be positive")
    stop = total if limit is None else min(total, start + limit)
    return list(range(start, stop))


def model_question(item: dict[str, Any]) -> str:
    return f"Current Date: {item['question_date']}\nQuestion: {item['question']}"


def _safe_question_id(question_id: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(question_id)).strip("._")
    return cleaned[:100] or "unknown"


def item_paths(
    output_dir: Path, index: int, question_id: object
) -> dict[str, Path]:
    item_dir = output_dir / "items" / f"{index:04d}_{_safe_question_id(question_id)}"
    return {
        "item_dir": item_dir,
        "memory_dir": item_dir / "memory",
        "checkpoint": item_dir / "checkpoint.json",
        "build_checkpoint": item_dir / "build-checkpoint.json",
    }


def _tracker_reset(backend: Any, phase: str) -> None:
    if tracker := getattr(backend, "tracker", None):
        tracker.reset(phase)


def _tracker_snapshot(backend: Any, phase: str) -> dict[str, Any]:
    if tracker := getattr(backend, "tracker", None):
        return tracker.snapshot(phase)
    return {
        "calls": None,
        "tokens_in": None,
        "tokens_out": None,
        "llm_time_s": None,
    }


def memory_stats(memory_dir: Path) -> dict[str, int]:
    files = [path for path in memory_dir.rglob("*.md") if path.is_file()]
    return {
        "markdown_files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }


def memory_is_valid(memory_dir: Path) -> bool:
    stats = memory_stats(memory_dir) if memory_dir.is_dir() else {}
    return bool(stats.get("markdown_files", 0) and stats.get("bytes", 0))


def _checkpoint_is_complete(
    checkpoint: object,
    item: dict[str, Any],
    index: int,
    memory_dir: Path,
) -> tuple[bool, str]:
    if not isinstance(checkpoint, dict):
        return False, "not_an_object"
    if checkpoint.get("status") != "complete":
        return False, f"status={checkpoint.get('status', 'missing')}"
    if checkpoint.get("dataset_index") != index:
        return False, "dataset_index_mismatch"
    if str(checkpoint.get("question_id")) != str(item["question_id"]):
        return False, "question_id_mismatch"
    if not str(checkpoint.get("answer", {}).get("hypothesis", "")).strip():
        return False, "empty_answer"
    if checkpoint.get("build", {}).get("status") != "complete":
        return False, "build_incomplete"
    if checkpoint.get("retrieval", {}).get("status") != "complete":
        return False, "retrieval_incomplete"
    if not memory_is_valid(memory_dir):
        return False, "memory_missing_or_empty"
    return True, "complete"


def _base_checkpoint(
    item: dict[str, Any],
    index: int,
    paths: dict[str, Path],
    run_meta: dict[str, Any],
) -> dict[str, Any]:
    sessions = item["haystack_sessions"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pending",
        "dataset_index": index,
        "question_id": str(item["question_id"]),
        "question_type": item["question_type"],
        "question": item["question"],
        "gold": item["answer"],
        "question_date": normalize_date(item["question_date"]),
        "question_date_raw": item.get("question_date"),
        "answer_session_ids": item.get("answer_session_ids", []),
        "input": {
            "sessions": len(sessions),
            "turns": sum(len(session) for session in sessions),
            "empty_source_turns": sum(
                not str(turn.get("content", "")).strip()
                for session in sessions
                for turn in session
            ),
            "source_session_ids": item["haystack_session_ids"],
            "source_dates": item["haystack_dates"],
        },
        **run_meta,
        "paths": {key: str(value) for key, value in paths.items()},
        "created_at": utc_now(),
    }


def _record_failure(
    checkpoint: dict[str, Any], checkpoint_path: Path, stage: str, exc: Exception
) -> None:
    checkpoint.update({
        "status": "failed",
        "updated_at": utc_now(),
        "error": {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    })
    atomic_json(checkpoint_path, checkpoint)


def _run_item_unlocked(
    index: int,
    item: dict[str, Any],
    output_dir: Path,
    backend: Any,
    run_meta: dict[str, Any],
    resume: bool,
    build_only: bool = False,
) -> tuple[bool, str, dict[str, Any] | None]:
    paths = item_paths(output_dir, index, item["question_id"])
    old = None
    if paths["checkpoint"].is_file():
        loaded = read_json(paths["checkpoint"])
        old = loaded if isinstance(loaded, dict) else None
    complete, reason = _checkpoint_is_complete(
        old, item, index, paths["memory_dir"]
    )
    if complete:
        return True, "already complete", old
    if old and old.get("status") == "complete":
        raise ExistingStateError(
            f"item {index} has a completed but invalid checkpoint ({reason})"
        )
    if paths["item_dir"].exists() and not resume:
        raise ExistingStateError(
            f"item {index} has existing incomplete state ({reason}); rerun with --resume"
        )
    paths["item_dir"].mkdir(parents=True, exist_ok=True)
    checkpoint = _base_checkpoint(item, index, paths, run_meta)
    if old:
        checkpoint.update(old)
        checkpoint.update({key: run_meta[key] for key in run_meta})
    checkpoint.pop("error", None)

    try:
        conversation = to_conversation(item, index)
    except Exception as exc:  # noqa: BLE001
        _record_failure(checkpoint, paths["checkpoint"], "normalize", exc)
        return False, str(exc), checkpoint

    build_reusable = (
        checkpoint.get("build", {}).get("status") == "complete"
        and memory_is_valid(paths["memory_dir"])
    )
    if not build_reusable:
        resumable_build = bool(
            resume
            and getattr(backend, "supports_batch_checkpoint", False)
            and paths["build_checkpoint"].is_file()
            and paths["memory_dir"].exists()
        )
        if paths["memory_dir"].exists() and not resumable_build:
            shutil.rmtree(paths["memory_dir"])
        checkpoint.update({
            "status": "building",
            "build": {"status": "running", "started_at": utc_now()},
        })
        atomic_json(paths["checkpoint"], checkpoint)
        phase = f"lme_{index}_build"
        _tracker_reset(backend, phase)
        started = time.monotonic()
        try:
            if getattr(backend, "supports_batch_checkpoint", False):
                reported_seconds, events = backend.build_memory(
                    conversation,
                    str(paths["memory_dir"]),
                    checkpoint_path=paths["build_checkpoint"],
                    resume=resumable_build,
                )
            else:
                reported_seconds, events = backend.build_memory(
                    conversation, str(paths["memory_dir"])
                )
            usage = _tracker_snapshot(backend, phase)
            stats = memory_stats(paths["memory_dir"])
            if int(events) <= 0 or not stats["markdown_files"] or not stats["bytes"]:
                raise EmptyStageError("build produced no memory events")
            checkpoint["build"] = {
                "status": "complete",
                "finished_at": utc_now(),
                "reported_time_s": round(float(reported_seconds), 3),
                "wall_time_s": round(time.monotonic() - started, 3),
                "events": int(events),
                "usage": usage,
                **stats,
            }
            atomic_json(paths["checkpoint"], checkpoint)
        except BuildPaused as exc:
            checkpoint["build"].update({
                "status": "paused",
                "usage": _tracker_snapshot(backend, phase),
                "wall_time_s": round(time.monotonic() - started, 3),
                "paused_at": utc_now(),
            })
            checkpoint.update({
                "status": "paused",
                "updated_at": utc_now(),
            })
            checkpoint.pop("error", None)
            atomic_json(paths["checkpoint"], checkpoint)
            return False, str(exc), checkpoint
        except Exception as exc:  # noqa: BLE001
            checkpoint["build"].update({
                "status": "failed",
                "usage": _tracker_snapshot(backend, phase),
                "wall_time_s": round(time.monotonic() - started, 3),
            })
            _record_failure(checkpoint, paths["checkpoint"], "build", exc)
            return False, str(exc), checkpoint

    if build_only:
        checkpoint.update({
            "status": "built",
            "updated_at": utc_now(),
        })
        checkpoint.pop("retrieval", None)
        checkpoint.pop("answer", None)
        atomic_json(paths["checkpoint"], checkpoint)
        return True, "build complete", checkpoint

    checkpoint.update({
        "status": "retrieving",
        "retrieval": {"status": "running", "started_at": utc_now()},
    })
    checkpoint.pop("answer", None)
    atomic_json(paths["checkpoint"], checkpoint)
    phase = f"lme_{index}_retrieve_answer"
    _tracker_reset(backend, phase)
    started = time.monotonic()
    try:
        turn_index = backend.build_turn_index(conversation)
        if len(turn_index) != checkpoint["input"]["turns"]:
            raise EmptyStageError("turn index does not cover every source turn")
        memories, steps, answer, trace = backend.collect_and_answer_longmemeval(
            item, paths["memory_dir"], turn_index
        )
        usage = _tracker_snapshot(backend, phase)
        hypothesis = str(answer or "").strip()
        if not hypothesis:
            raise EmptyStageError("retrieval returned an empty hypothesis")
        if int(steps) <= 0:
            raise EmptyStageError("retrieval completed without a model step")
        checkpoint.update({
            "status": "complete",
            "retrieval": {
                "status": "complete",
                "finished_at": utc_now(),
                "wall_time_s": round(time.monotonic() - started, 3),
                "steps": int(steps),
                "memories_count": len(memories),
                "memories": memories,
                "tool_trace": trace,
                "usage": usage,
                "turn_index_entries": len(turn_index),
                "model_question": model_question(item),
            },
            "answer": {
                "status": "complete",
                "hypothesis": hypothesis,
                "model": run_meta["models"]["answerer"],
                "single_model_context": True,
                "finished_at": utc_now(),
            },
            "completed_at": utc_now(),
            "updated_at": utc_now(),
        })
        atomic_json(paths["checkpoint"], checkpoint)
    except Exception as exc:  # noqa: BLE001
        checkpoint["retrieval"].update({
            "status": "failed",
            "usage": _tracker_snapshot(backend, phase),
            "wall_time_s": round(time.monotonic() - started, 3),
        })
        _record_failure(checkpoint, paths["checkpoint"], "retrieve_answer", exc)
        return False, str(exc), checkpoint
    return True, "complete", checkpoint


def _acquire_item_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise ExistingStateError(
            f"item output is already locked by another writer: {lock_path}"
        ) from exc
    return handle


def _release_item_lock(handle) -> None:
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def run_item(
    index: int,
    item: dict[str, Any],
    output_dir: Path,
    backend: Any,
    run_meta: dict[str, Any],
    resume: bool,
    build_only: bool = False,
) -> tuple[bool, str, dict[str, Any] | None]:
    lock_path = output_dir / ".locks" / f"{index:04d}_{item['question_id']}.lock"
    handle = _acquire_item_lock(lock_path)
    try:
        return _run_item_unlocked(
            index, item, output_dir, backend, run_meta, resume, build_only
        )
    finally:
        _release_item_lock(handle)


def _collect_checkpoints(output_dir: Path) -> list[dict[str, Any]]:
    checkpoints = []
    for path in sorted((output_dir / "items").glob("*/checkpoint.json")):
        value = read_json(path)
        if isinstance(value, dict):
            checkpoints.append(value)
    return sorted(checkpoints, key=lambda item: int(item.get("dataset_index", 10**9)))


def _complete_payload(checkpoint: object) -> bool:
    if not isinstance(checkpoint, dict) or checkpoint.get("status") != "complete":
        return False
    if checkpoint.get("build", {}).get("status") != "complete":
        return False
    if checkpoint.get("retrieval", {}).get("status") != "complete":
        return False
    if not str(checkpoint.get("answer", {}).get("hypothesis", "")).strip():
        return False
    memory_dir = checkpoint.get("paths", {}).get("memory_dir")
    return bool(memory_dir and memory_is_valid(Path(memory_dir)))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            handle.write(value)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def refresh_outputs(output_dir: Path, manifest: dict[str, Any]) -> None:
    checkpoints = _collect_checkpoints(output_dir)
    completed = [item for item in checkpoints if _complete_payload(item)]
    atomic_json(output_dir / "results.json", [{
        "dataset_index": item["dataset_index"],
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "gold": item["gold"],
        "hypothesis": item["answer"]["hypothesis"],
        "build": item["build"],
        "retrieval": item["retrieval"],
        "models": item["models"],
        "config": item["config"],
    } for item in completed])
    _atomic_text(
        output_dir / "hypotheses.jsonl",
        "".join(json.dumps({
            "question_id": item["question_id"],
            "hypothesis": item["answer"]["hypothesis"],
        }, ensure_ascii=False) + "\n" for item in completed),
    )
    manifest.update({
        "checkpoint_counts": dict(sorted(Counter(
            str(item.get("status", "unknown")) for item in checkpoints
        ).items())),
        "completed": len(completed),
        "updated_at": utc_now(),
    })
    atomic_json(output_dir / "run_manifest.json", manifest)


def create_or_resume_manifest(
    output_dir: Path,
    data_path: Path,
    dataset_sha256: str,
    data_count: int,
    run_meta: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    path = output_dir / "run_manifest.json"
    if path.exists():
        old = read_json(path)
        if not isinstance(old, dict):
            raise ExistingStateError("existing run_manifest.json is invalid")
        expected = {
            "dataset_sha256": dataset_sha256,
            "models": run_meta["models"],
            "config": run_meta["config"],
            "code": run_meta["code"],
            "request_audit": run_meta.get("request_audit", {}),
        }
        mismatches = [key for key, value in expected.items() if old.get(key) != value]
        if mismatches:
            raise ExistingStateError(
                f"existing output uses different {', '.join(mismatches)}"
            )
        if not resume and any(
            item.get("status") != "complete" for item in _collect_checkpoints(output_dir)
        ):
            raise ExistingStateError(
                "existing run contains incomplete checkpoints; use --resume"
            )
        return old
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "LongMemEval-S",
        "output_dir": str(output_dir),
        "dataset_path": str(data_path),
        "dataset_sha256": dataset_sha256,
        "dataset_items": data_count,
        **run_meta,
        "created_at": utc_now(),
        "completed": 0,
        "checkpoint_counts": {},
    }


__all__ = [
    "DEFAULT_DATA",
    "EXPECTED_LONGMEMEVAL_SIZE",
    "DataValidationError",
    "EmptyStageError",
    "ExistingStateError",
    "create_or_resume_manifest",
    "git_head",
    "item_paths",
    "load_dataset",
    "memory_is_valid",
    "model_question",
    "refresh_outputs",
    "run_item",
    "select_indices",
    "sha256_file",
    "to_conversation",
    "utc_now",
    "validate_longmemeval_s",
]
