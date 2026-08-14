"""Scriptorium memory-build orchestration."""

import json
import hashlib
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import management as memory
from .conversation import (
    benchmark_source_id,
    build_turn_index,
    normalize_date,
    read_turns,
    session_content,
)
from .management.api import render_writer_input, writer_protocol_sha256
from .runtime.capacity import WriterCapacity, pack_complete_messages
from .runtime.tokenization import TokenCounter


class BuildPaused(RuntimeError):
    """Raised only after a fully committed batch checkpoint requests a pause."""


@dataclass(frozen=True)
class BuildConfig:
    session_batch: int = 1
    calibration_path: str | None = None
    writer_input_token_cap: int | None = None
    local_reorg_every_sessions: int = 5
    verify_writes: bool = True
    verify_every_sessions: int = 1
    final_manage: bool = True
    memory_config: memory.MemoryConfig = field(
        default_factory=memory.MemoryConfig
    )

    def __post_init__(self) -> None:
        if self.session_batch < 1:
            raise ValueError("session_batch must be positive")
        if self.calibration_path is not None and not self.calibration_path.strip():
            raise ValueError("calibration_path must not be empty")
        if self.writer_input_token_cap is not None and self.writer_input_token_cap < 1:
            raise ValueError("writer_input_token_cap must be positive")
        if self.local_reorg_every_sessions < 0:
            raise ValueError("local_reorg_every_sessions must be non-negative")
        if self.verify_every_sessions < 1:
            raise ValueError("verify_every_sessions must be positive")


def _written_blocks(audit):
    return sum(
        int(record.get("count", 0))
        for record in audit
        if record.get("tool") in {"shell", "save_memory", "commit"}
        and record.get("status", "ok") == "ok"
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _batch_plan(
    batches: list[list[dict[str, Any]]],
    *,
    input_tokens: list[int] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    plan = []
    for index, batch in enumerate(batches):
        session_indices = sorted({int(item["session_index"]) for item in batch})
        record = {
            "index": index,
            "source_ids": [ref for session in batch for ref in session["refs"]],
            "observation_dates": [session["observation_date"] for session in batch],
            "session_indices": session_indices,
        }
        if input_tokens is not None:
            record["input_tokens"] = input_tokens[index]
        plan.append(record)
    digest = hashlib.sha256(json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return plan, digest


def build_memory(
    conv: dict[str, Any],
    memory_dir: str | Path,
    max_sessions: int | None = None,
    *,
    agent: Any,
    model: str,
    usage_logger=None,
    config: BuildConfig | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
):
    config = config or BuildConfig()
    started = time.time()
    Path(memory_dir).mkdir(parents=True, exist_ok=True)
    sessions = []
    index = 1
    while f"session_{index}" in conv:
        turns, refs = session_content(conv[f"session_{index}"], index)
        sessions.append({
            "session_index": index,
            "observation_date": normalize_date(
                conv.get(f"session_{index}_date_time", "")
            ),
            "turns": turns,
            "refs": [benchmark_source_id(ref) for ref in refs],
        })
        index += 1
    if max_sessions:
        sessions = sessions[:max_sessions]

    event_count = 0
    session_batch = config.session_batch
    local_every = config.local_reorg_every_sessions
    verify_writes = config.verify_writes
    verify_every = config.verify_every_sessions
    touched_topics: set[str] = set()

    verification_path = Path(memory_dir) / "verification.jsonl"
    writer_token_counter = None
    if config.calibration_path or config.writer_input_token_cap:
        max_input_tokens = config.writer_input_token_cap
        capacity = WriterCapacity.load(
            config.calibration_path,
            model=model,
            writer_protocol_sha256=writer_protocol_sha256(),
        ) if config.calibration_path else None
        if capacity:
            max_input_tokens = min(
                capacity.safe_input_tokens,
                max_input_tokens or capacity.safe_input_tokens,
            )
        writer_token_counter = (
            TokenCounter.from_identity(capacity.tokenizer)
            if capacity else TokenCounter.resolve(requested_model=model)
        )
        batches = pack_complete_messages(
            sessions,
            max_input_tokens=max_input_tokens,
            render_batch=render_writer_input,
            token_counter=writer_token_counter,
        )
    else:
        batches = [
            sessions[start:start + session_batch]
            for start in range(0, len(sessions), session_batch)
        ]

    batch_input_tokens = (
        [writer_token_counter.count(render_writer_input(batch)) for batch in batches]
        if writer_token_counter is not None else None
    )
    plan, plan_hash = _batch_plan(batches, input_tokens=batch_input_tokens)
    progress_path = Path(checkpoint_path) if checkpoint_path else None
    progress: dict[str, Any] = {
        "schema": "scriptorium-build-checkpoint-v1",
        "status": "building",
        "model": model,
        "writer_protocol_sha256": writer_protocol_sha256(),
        "batch_plan": plan,
        "batch_plan_sha256": plan_hash,
        "writer_input_token_cap": config.writer_input_token_cap,
        "writer_tokenizer": (
            writer_token_counter.identity if writer_token_counter is not None else None
        ),
        "total_batches": len(batches),
        "total_sessions": len(sessions),
        "total_source_turns": sum(len(session["turns"]) for session in sessions),
        "completed_batches": 0,
        "completed_sessions": 0,
        "completed_source_turns": 0,
        "event_count": 0,
        "writer_trajectories": [],
        "touched_topics": [],
        "final_management": "pending" if config.final_manage else "disabled",
    }
    if progress_path and resume and progress_path.is_file():
        loaded = json.loads(progress_path.read_text(encoding="utf-8"))
        if loaded.get("schema") != progress["schema"]:
            raise ValueError("unsupported build checkpoint schema")
        if loaded.get("model") != model:
            raise ValueError("build checkpoint model differs")
        if loaded.get("writer_protocol_sha256") != progress["writer_protocol_sha256"]:
            raise ValueError("build checkpoint writer protocol differs")
        if loaded.get("batch_plan_sha256") != plan_hash:
            raise ValueError("build checkpoint batch plan differs")
        completed = int(loaded.get("completed_batches", 0))
        if completed < 0 or completed > len(batches):
            raise ValueError("build checkpoint completed batch count is invalid")
        progress.update(loaded)
        event_count = int(progress.get("event_count", 0))
        touched_topics = set(progress.get("touched_topics", []))
    elif progress_path:
        _atomic_json(progress_path, progress)

    completed_sessions = int(progress.get("completed_sessions", 0))
    completed_source_turns = int(progress.get("completed_source_turns", 0))
    session_by_last_ref = {
        session["refs"][-1]: session for session in sessions if session["refs"]
    }
    completed_batches = int(progress.get("completed_batches", 0))
    for batch_index, batch in enumerate(batches):
        if batch_index < completed_batches:
            continue
        live_audit_path = None
        if progress_path:
            pattern = f"writer-live-batch-{batch_index:03d}-attempt-*.jsonl"
            attempt = len(list(progress_path.parent.glob(pattern))) + 1
            live_audit_path = progress_path.with_name(
                f"writer-live-batch-{batch_index:03d}-attempt-{attempt:03d}.jsonl"
            )
            current_session_indices = sorted({
                int(item["session_index"]) for item in batch
            })
            progress.update({
                "current_batch": batch_index,
                "current_batch_number": batch_index + 1,
                "current_batch_sessions": len(current_session_indices),
                "current_session_start": min(current_session_indices),
                "current_session_end": max(current_session_indices),
                "current_batch_input_tokens": (
                    batch_input_tokens[batch_index]
                    if batch_input_tokens is not None else None
                ),
                "current_batch_source_turns": sum(
                    len(session["turns"]) for session in batch
                ),
                "current_live_audit": str(live_audit_path),
            })
            _atomic_json(progress_path, progress)
        try:
            audit = memory.write_sessions(
                memory_dir,
                agent=agent,
                sessions=batch,
                usage_logger=usage_logger,
                live_audit_path=live_audit_path,
                config=config.memory_config,
            )
        except Exception as exc:
            failure_audit = list(getattr(exc, "scriptorium_audit", []))
            if progress_path:
                attempt_suffix = ""
                if live_audit_path is not None:
                    match = re.search(
                        r"-attempt-(\d+)\.jsonl$", str(live_audit_path)
                    )
                    if match is not None:
                        attempt_suffix = f"-attempt-{int(match.group(1)):03d}"
                failure_path = progress_path.with_name(
                    f"writer-failure-batch-{batch_index:03d}"
                    f"{attempt_suffix}.json"
                )
                _atomic_json(failure_path, {
                    "batch_index": batch_index,
                    "sessions": len(batch),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "tool_calls": len(failure_audit),
                    "audit": failure_audit,
                })
                progress.update({
                    "status": "failed",
                    "failed_batch": batch_index,
                    "failure_audit": str(failure_path),
                    "failure_tool_calls": len(failure_audit),
                })
                _atomic_json(progress_path, progress)
            raise
        event_count += _written_blocks(audit)
        agent_record = next(
            (record for record in reversed(audit) if record.get("tool") == "agent"),
            None,
        )
        writer_trajectories = list(progress.get("writer_trajectories", []))
        trajectory_record = {
            "batch_index": batch_index,
            "sessions": len({int(item["session_index"]) for item in batch}),
            "rounds": int(agent_record.get("rounds", 0)) if agent_record else None,
            "live_audit": str(live_audit_path) if live_audit_path else None,
        }
        if agent_record and agent_record.get("core_repair"):
            trajectory_record["core_repair"] = agent_record["core_repair"]
        if agent_record and agent_record.get("generic_repair"):
            trajectory_record["generic_repair"] = agent_record["generic_repair"]
        writer_trajectories.append(trajectory_record)
        progress["writer_trajectories"] = writer_trajectories
        for record in audit:
            if record.get("status") == "ok":
                touched_topics.update(record.get("topic_paths", []))
        for session in batch:
            complete_session = session_by_last_ref.get(session["refs"][-1])
            if complete_session is None:
                continue
            completed_sessions += 1
            completed_source_turns += len(complete_session["turns"])
            should_verify = verify_writes and (
                completed_sessions % verify_every == 0
                or completed_sessions == len(sessions)
            )
            if should_verify:
                # Verification is an optional self-check over memory that is
                # already committed. A model that cannot hold up its end of it
                # must not take the build down with it.
                try:
                    verification = memory.verify_session(
                        memory_dir,
                        agent=agent,
                        observation_date=complete_session["observation_date"],
                        turns=complete_session["turns"],
                        refs=complete_session["refs"],
                        usage_logger=usage_logger,
                        config=config.memory_config,
                    )
                except Exception as exc:
                    verification = {
                        "skipped": True,
                        "reason": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            else:
                verification = {
                    "skipped": True,
                    "reason": "interval" if verify_writes else "disabled",
                }
            with verification_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(verification, ensure_ascii=False) + "\n")
            repair_trace = verification.get("repair_trace", [])
            event_count += _written_blocks(repair_trace)
            for record in repair_trace:
                if record.get("status") == "ok":
                    touched_topics.update(record.get("topic_paths", []))
            if (
                local_every
                and completed_sessions % local_every == 0
                and completed_sessions < len(sessions)
                and touched_topics
            ):
                memory.organize_topics(
                    memory_dir,
                    agent=agent,
                    touched=touched_topics,
                    usage_logger=usage_logger,
                    config=config.memory_config,
                )
                touched_topics.clear()

        if progress_path:
            progress.update({
                "status": "building",
                "completed_batches": batch_index + 1,
                "completed_sessions": completed_sessions,
                "completed_source_turns": completed_source_turns,
                "last_source_id": batch[-1]["refs"][-1] if batch and batch[-1]["refs"] else None,
                "event_count": event_count,
                "touched_topics": sorted(touched_topics),
                "current_batch": None,
                "current_batch_number": None,
                "current_batch_sessions": None,
                "current_session_start": None,
                "current_session_end": None,
                "current_batch_source_turns": None,
                "current_batch_input_tokens": None,
                "current_live_audit": None,
            })
            _atomic_json(progress_path, progress)
            stop_path = progress_path.parent / ".stop-after-current-batch"
            if stop_path.is_file():
                progress.update({
                    "status": "paused",
                    "pause_reason": "stop-after-current-batch",
                    "paused_after_batch": batch_index + 1,
                })
                _atomic_json(progress_path, progress)
                raise BuildPaused(
                    f"paused after committed batch {batch_index + 1}/{len(batches)}"
                )

    if (
        event_count
        and config.final_manage
        and progress.get("final_management") != "complete"
    ):
        memory.manage_memory(
            memory_dir,
            agent=agent,
            usage_logger=usage_logger,
            config=config.memory_config,
        )
        progress["final_management"] = "complete"
    if progress_path:
        progress.update({
            "status": "complete",
            "completed_batches": len(batches),
            "completed_sessions": completed_sessions,
            "completed_source_turns": completed_source_turns,
            "event_count": event_count,
            "touched_topics": sorted(touched_topics),
        })
        _atomic_json(progress_path, progress)
    return time.time() - started, event_count
