"""Memory Writer and Manager execution through Claude Code."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..agent_runtime import ClaudeCodeAgent
from .config import MemoryConfig
from ..prompts import FEW_SHOT_INSTRUCTIONS, SYSTEM_PROMPT
from .tools import management_tools
from .workspace import MemoryWorkspace


def render_conversation(
    turns: list[tuple[str, str]], refs: list[str]
) -> str:
    return "\n".join(
        f"[{ref}] {speaker}: {text}"
        for (speaker, text), ref in zip(turns, refs)
    )


def _baseline(workspace: MemoryWorkspace) -> tuple[Any, ...] | None:
    """Snapshot the stage, or None if it cannot be parsed.

    A turn whose edits were rejected leaves the stage rolled back, but a
    malformed file that reached disk another way would make this raise. That
    is a reason to skip committing, never to abort the whole build.
    """
    try:
        return workspace.baseline()
    except Exception:
        workspace._refresh_stage()
        try:
            return workspace.baseline()
        except Exception:
            return None


def _commit_turn(
    workspace: MemoryWorkspace,
    baseline: tuple[Any, ...] | None,
    audit: list[dict[str, Any]],
) -> str | None:
    """Install what the turn staged. Returns the error text if it failed."""
    if baseline is None:
        workspace._refresh_stage()
        return None
    if not workspace.stage_is_dirty():
        return None
    try:
        workspace.commit_edits(*baseline)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        audit.append({
            "tool": "commit", "status": "error", "output": message,
        })
        return message
    audit.append({
        "tool": "commit",
        "status": "ok",
        "count": workspace.last_created_blocks,
        "topic_paths": workspace.last_changed_topics,
    })
    return None


def _run_agent(
    memory_dir: str | Path,
    *,
    agent: ClaudeCodeAgent,
    task: str,
    source_sessions: list[dict[str, Any]] | None = None,
    usage_logger: Any | None = None,
    final_output: list[str] | None = None,
    config: MemoryConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or MemoryConfig()
    workspace = MemoryWorkspace(memory_dir, config=config)
    try:
        if source_sessions:
            workspace.archive_sessions(source_sessions)
            workspace._refresh_stage()
        task = f"{task}\n\nCurrent workspace structure:\n{workspace.structure()}"
        audit: list[dict[str, Any]] = []
        system_prompt = (
            f"{SYSTEM_PROMPT}\n{FEW_SHOT_INSTRUCTIONS}"
            if config.few_shot_instructions
            else SYSTEM_PROMPT
        )
        # Edits made through the built-in file tools never pass through the
        # shell tool, so the transaction runs once the turn is over.
        baseline = _baseline(workspace)
        result = agent.run(
            prompt=task,
            system_prompt=system_prompt,
            cwd=workspace.stage_dir,
            tools=management_tools(workspace, audit),
            max_turns=config.max_turns,
            max_budget_usd=config.max_budget_usd,
        )
        if usage_logger is not None:
            usage_logger(result)
        error = _commit_turn(workspace, baseline, audit)
        if error is not None:
            # A rejected turn was rolled back, so the pre-repair stage is the
            # committed state. Snapshot it now: reading it after the repair
            # edits would parse the model's malformed text and raise.
            repair_baseline = _baseline(workspace)
            repair = agent.run(
                prompt=(
                    "Your edits were rejected and discarded:\n\n"
                    f"{error}\n\n"
                    "The workspace is back to its state before this turn. "
                    "Redo the work, fixing what the message reports.\n\n"
                    f"{task}"
                ),
                system_prompt=system_prompt,
                cwd=workspace.stage_dir,
                tools=management_tools(workspace, audit),
                max_turns=config.max_turns,
                max_budget_usd=config.max_budget_usd,
            )
            if usage_logger is not None:
                usage_logger(repair)
            _commit_turn(workspace, repair_baseline, audit)
        if final_output is not None:
            final_output.append(result.text)
        audit.append({
            "tool": "agent",
            "status": "ok",
            "reason": result.stop_reason or "complete",
            "rounds": result.num_turns,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "anthropic_equivalent_cost_usd": (
                result.anthropic_equivalent_cost_usd
            ),
        })
        return audit
    finally:
        shutil.rmtree(workspace.stage_dir, ignore_errors=True)
