"""Public writing and maintenance operations."""

from pathlib import Path
from typing import Any

from .config import MemoryConfig
from .model import _run_agent, render_conversation
from .prompts import (
    LOCAL_MANAGER_TASK,
    MANAGER_TASK,
    TOOLS,
    WRITER_BATCH_TASK,
    WRITER_TASK,
)


def write_session(
    memory_dir: str | Path,
    *,
    client: Any,
    model: str,
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
    return _run_agent(memory_dir, client=client, model=model, task=task,
                      source_sessions=[{
                          "observation_date": observation_date,
                          "turns": turns,
                          "refs": refs,
                      }],
                      usage_logger=usage_logger,
                      tools=TOOLS[:1], config=config)


def write_sessions(
    memory_dir: str | Path,
    *,
    client: Any,
    model: str,
    sessions: list[dict[str, Any]],
    usage_logger: Any | None = None,
    config: MemoryConfig | None = None,
) -> list[dict[str, Any]]:
    rendered = []
    for number, session in enumerate(sessions, start=1):
        rendered.append(
            f"## Session {number}\nObservation date: "
            f"{session['observation_date']}\n\n"
            f"{render_conversation(session['turns'], session['refs'])}"
        )
    task = WRITER_BATCH_TASK.format(sessions="\n\n".join(rendered))
    return _run_agent(
        memory_dir,
        client=client,
        model=model,
        task=task,
        source_sessions=sessions,
        usage_logger=usage_logger,
        tools=TOOLS[:1],
        config=config,
    )


def manage_memory(
    memory_dir: str | Path,
    *,
    client: Any,
    model: str,
    usage_logger: Any | None = None,
    config: MemoryConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or MemoryConfig()
    return _run_agent(memory_dir, client=client, model=model, task=MANAGER_TASK,
                      usage_logger=usage_logger,
                      max_rounds=config.manager_max_rounds,
                      tools=TOOLS[:1], config=config)


def organize_topics(
    memory_dir: str | Path,
    *,
    client: Any,
    model: str,
    touched: set[str] | None = None,
    final: bool = False,
    usage_logger: Any | None = None,
    config: MemoryConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or MemoryConfig()
    """Compatibility entry point for existing manager callers."""
    if final or touched is None:
        return manage_memory(
            memory_dir, client=client, model=model, usage_logger=usage_logger,
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
        client=client,
        model=model,
        task=LOCAL_MANAGER_TASK.format(topic_paths="\n".join(normalized)),
        usage_logger=usage_logger,
        max_rounds=config.manager_max_rounds,
        tools=TOOLS[:1],
        config=config,
    )
