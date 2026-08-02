"""NativeMem memory-build orchestration."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import management as memory
from .conversation import (
    add_benchmark_source_ids,
    benchmark_source_id,
    build_turn_index,
    normalize_date,
    read_turns,
    session_content,
)
from .management.api import render_writer_input, writer_protocol_sha256
from .runtime.capacity import WriterCapacity, pack_complete_sessions
from .runtime.tokenization import TokenCounter


@dataclass(frozen=True)
class BuildConfig:
    session_batch: int = 1
    calibration_path: str | None = None
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


def build_memory(
    conv: dict[str, Any],
    memory_dir: str | Path,
    max_sessions: int | None = None,
    *,
    client: Any,
    model: str,
    usage_logger=None,
    config: BuildConfig | None = None,
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
    if config.calibration_path:
        capacity = WriterCapacity.load(
            config.calibration_path,
            model=model,
            writer_protocol_sha256=writer_protocol_sha256(),
        )
        batches = pack_complete_sessions(
            sessions,
            max_input_tokens=capacity.safe_input_tokens,
            render_batch=render_writer_input,
            token_counter=TokenCounter.from_identity(capacity.tokenizer),
        )
    else:
        batches = [
            sessions[start:start + session_batch]
            for start in range(0, len(sessions), session_batch)
        ]

    completed_sessions = 0
    for batch in batches:
        audit = memory.write_sessions(
            memory_dir,
            client=client,
            model=model,
            sessions=batch,
            usage_logger=usage_logger,
            config=config.memory_config,
        )
        event_count += _written_blocks(audit)
        for record in audit:
            if record.get("status") == "ok":
                touched_topics.update(record.get("topic_paths", []))
        for session in batch:
            completed_sessions += 1
            should_verify = verify_writes and (
                completed_sessions % verify_every == 0
                or completed_sessions == len(sessions)
            )
            if should_verify:
                verification = memory.verify_session(
                    memory_dir,
                    client=client,
                    model=model,
                    observation_date=session["observation_date"],
                    turns=session["turns"],
                    refs=session["refs"],
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
                    client=client,
                    model=model,
                    touched=touched_topics,
                    usage_logger=usage_logger,
                    config=config.memory_config,
                )
                touched_topics.clear()

    if event_count and config.final_manage:
        memory.manage_memory(
            memory_dir,
            client=client,
            model=model,
            usage_logger=usage_logger,
            config=config.memory_config,
        )
    return time.time() - started, event_count
