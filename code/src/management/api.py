"""Public writing and maintenance operations."""

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import MemoryConfig
from .agent import _run_agent, render_conversation
from .prompts import (
    LOCAL_MANAGER_TASK,
    MANAGER_TASK,
    SYSTEM_PROMPT,
    TOOLS,
    WRITER_BATCH_TASK,
    WRITER_TASK,
)


def render_writer_task(sessions: list[dict[str, Any]]) -> str:
    rendered = []
    for number, session in enumerate(sessions, start=1):
        rendered.append(
            f"## Session {number}\nObservation date: "
            f"{session['observation_date']}\n\n"
            f"{render_conversation(session['turns'], session['refs'])}"
        )
    return WRITER_BATCH_TASK.format(sessions="\n\n".join(rendered))


def render_writer_input(sessions: list[dict[str, Any]]) -> str:
    """Render all fixed and session-specific text sent to the Writer."""
    return f"{SYSTEM_PROMPT}\n\n{render_writer_task(sessions)}"


def writer_protocol_sha256() -> str:
    payload = json.dumps(
        {
            "system": SYSTEM_PROMPT,
            "writer_batch": WRITER_BATCH_TASK,
            "tools": TOOLS[:1],
            "runtime": "claude-agent-sdk",
            "contract": "topic-core-v3-runtime-ids",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_session(
    memory_dir: str | Path,
    *,
    agent: Any,
    observation_date: str,
    turns: list[tuple[str, str]],
    refs: list[str],
    usage_logger: Any | None = None,
    config: MemoryConfig | None = None,
) -> list[dict[str, Any]]:
    task = WRITER_TASK.format(
        observation_date=observation_date,
        conversation=render_conversation(turns, refs),
    )
    return _run_agent(memory_dir, agent=agent, task=task,
                      source_sessions=[{
                          "observation_date": observation_date,
                          "turns": turns,
                          "refs": refs,
                      }],
                      usage_logger=usage_logger, config=config)


def write_sessions(
    memory_dir: str | Path,
    *,
    agent: Any,
    sessions: list[dict[str, Any]],
    usage_logger: Any | None = None,
    live_audit_path: str | Path | None = None,
    config: MemoryConfig | None = None,
) -> list[dict[str, Any]]:
    task = render_writer_task(sessions)
    return _run_agent(
        memory_dir,
        agent=agent,
        task=task,
        source_sessions=sessions,
        usage_logger=usage_logger,
        live_audit_path=live_audit_path,
        config=config,
    )


def manage_memory(
    memory_dir: str | Path,
    *,
    agent: Any,
    usage_logger: Any | None = None,
    config: MemoryConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or MemoryConfig()
    return _run_agent(memory_dir, agent=agent, task=MANAGER_TASK,
                      usage_logger=usage_logger,
                      config=config)


def organize_topics(
    memory_dir: str | Path,
    *,
    agent: Any,
    touched: set[str] | None = None,
    final: bool = False,
    usage_logger: Any | None = None,
    config: MemoryConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or MemoryConfig()
    """Compatibility entry point for existing manager callers."""
    if final or touched is None:
        return manage_memory(
            memory_dir, agent=agent, usage_logger=usage_logger,
            config=config,
        )
    normalized = sorted({
        Path(path).as_posix()
        for path in touched
        if Path(path).as_posix().startswith("topics/")
    })
    if not normalized:
        return []
    return _run_agent(
        memory_dir,
        agent=agent,
        task=LOCAL_MANAGER_TASK.format(topic_paths="\n".join(normalized)),
        usage_logger=usage_logger,
        config=config,
    )
