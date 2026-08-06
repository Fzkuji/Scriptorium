"""Scriptorium memory-build orchestration."""

import json
import hashlib
import os
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
        if record.get("tool") in {"shell", "save_memory"}
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


def _batch_plan(batches: list[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], str]:
    plan = [{
        "index": index,
        "source_ids": [ref for session in batch for ref in session["refs"]],
        "observation_dates": [session["observation_date"] for session in batch],
    } for index, batch in enumerate(batches)]
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
        batches = pack_complete_messages(
            sessions,
            max_input_tokens=max_input_tokens,
            render_batch=render_writer_input,
            token_counter=(
                TokenCounter.from_identity(capacity.tokenizer)
                if capacity else TokenCounter.resolve(requested_model=model)
            ),
        )
    else:
        batches = [
            sessions[start:start + session_batch]
            for start in range(0, len(sessions), session_batch)
        ]

    plan, plan_hash = _batch_plan(batches)
    progress_path = Path(checkpoint_path) if checkpoint_path else None
    progress: dict[str, Any] = {
        "schema": "scriptorium-build-checkpoint-v1",
        "status": "building",
        "batch_plan": plan,
        "batch_plan_sha256": plan_hash,
        "completed_batches": 0,
        "completed_sessions": 0,
        "event_count": 0,
        "touched_topics": [],
        "final_management": "pending" if config.final_manage else "disabled",
    }
    if progress_path and resume and progress_path.is_file():
        loaded = json.loads(progress_path.read_text(encoding="utf-8"))
        if loaded.get("schema") != progress["schema"]:
            raise ValueError("unsupported build checkpoint schema")
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
    session_by_last_ref = {
        session["refs"][-1]: session for session in sessions if session["refs"]
    }
    completed_batches = int(progress.get("completed_batches", 0))
    for batch_index, batch in enumerate(batches):
        if batch_index < completed_batches:
            continue
        audit = memory.write_sessions(
            memory_dir,
            agent=agent,
            sessions=batch,
            usage_logger=usage_logger,
            config=config.memory_config,
        )
        event_count += _written_blocks(audit)
        for record in audit:
            if record.get("status") == "ok":
                touched_topics.update(record.get("topic_paths", []))
        for session in batch:
            complete_session = session_by_last_ref.get(session["refs"][-1])
            if complete_session is None:
                continue
            completed_sessions += 1
            should_verify = verify_writes and (
                completed_sessions % verify_every == 0
                or completed_sessions == len(sessions)
            )
            if should_verify:
                verification = memory.verify_session(
                    memory_dir,
                    agent=agent,
                    observation_date=complete_session["observation_date"],
                    turns=complete_session["turns"],
                    refs=complete_session["refs"],
                    usage_logger=usage_logger,
                    config=config.memory_config,
                )
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
                "last_source_id": batch[-1]["refs"][-1] if batch and batch[-1]["refs"] else None,
                "event_count": event_count,
                "touched_topics": sorted(touched_topics),
            })
            _atomic_json(progress_path, progress)

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
            "event_count": event_count,
            "touched_topics": sorted(touched_topics),
        })
        _atomic_json(progress_path, progress)
    return time.time() - started, event_count
