"""Memory Writer and Manager execution through Claude Code."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..agent_runtime import ClaudeCodeAgent
from .config import MemoryConfig
from .model_reconciliation import _make_reconciler
from .prompts import SYSTEM_PROMPT
from .tools import management_tools
from .workspace import MemoryWorkspace


def render_conversation(
    turns: list[tuple[str, str]], refs: list[str]
) -> str:
    return "\n".join(
        f"[{ref}] {speaker}: {text}"
        for (speaker, text), ref in zip(turns, refs)
    )


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
    workspace.reconciler = _make_reconciler(
        agent,
        usage_logger=usage_logger,
        config=config,
        cwd=workspace.stage_dir,
    )
    try:
        if source_sessions:
            workspace.archive_sessions(source_sessions)
            workspace._refresh_stage()
        task = f"{task}\n\nCurrent workspace structure:\n{workspace.structure()}"
        audit: list[dict[str, Any]] = []
        result = agent.run(
            prompt=task,
            system_prompt=SYSTEM_PROMPT,
            cwd=workspace.stage_dir,
            tools=management_tools(workspace, audit),
            max_turns=config.max_turns,
            max_budget_usd=config.max_budget_usd,
        )
        if usage_logger is not None:
            usage_logger(result)
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
