"""Run one agent pass for write-time verification.

Writing has its own pass, in `memory.writing.session`; organizing runs the
shared agent pass itself now, in `memory.organizing.reorganize`. What is
left here is the shape verification needs — build the tool list for a
stage, run the agent, and on a rejected edit, retry once with the reason
attached — plus the "write" stage this same machinery still runs directly
for the interactive ingest CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import MemoryConfig
from ..organizing.tools import _repair_guidance, organizing_tools
from ..workspace import MemoryWorkspace
from ..workspace.agent_pass import run_pass


def _tools_for(agent: Any, stage: str):
    """The tool list this pass hands the model, given the workspace it edits."""
    def build(workspace: MemoryWorkspace, audit: list[dict[str, Any]]):
        return organizing_tools(
            workspace, audit,
            file_tools=not getattr(agent, "has_file_tools", True),
            stage=stage,
        )
    return build


def _run_agent(
    memory_dir: str | Path,
    *,
    agent: Any,
    task: str,
    source_sessions: list[dict[str, Any]] | None = None,
    usage_logger: Any | None = None,
    final_output: list[str] | None = None,
    config: MemoryConfig | None = None,
    history_dir: str | Path | None = None,
    stage: str | None = None,
) -> list[dict[str, Any]]:
    """One verification pass, or a directly-run write/organize turn, with
    organizing's tools and repair rules."""
    resolved = stage or ("write" if source_sessions else "organize")
    return run_pass(
        memory_dir,
        agent=agent,
        task=task,
        source_sessions=source_sessions,
        usage_logger=usage_logger,
        final_output=final_output,
        config=config,
        history_dir=history_dir,
        stage=resolved,
        tools=_tools_for(agent, resolved),
        guidance=_repair_guidance,
    )
