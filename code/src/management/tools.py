"""Claude Code MCP tools for transactional memory edits."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool

from .workspace import MemoryWorkspace
from .block_views import core_token_count


@dataclass
class CoreRepairState:
    """Per-trajectory capacity-check state exposed only to repair agents."""

    limit: int
    target: int
    max_checks: int
    stagnation_limit: int
    checks: list[int] = field(default_factory=list)
    non_decreasing_checks: int = 0

    @property
    def stagnated(self) -> bool:
        return self.non_decreasing_checks >= self.stagnation_limit

    def record(self, token_count: int) -> None:
        if self.checks:
            if token_count >= self.checks[-1]:
                self.non_decreasing_checks += 1
            else:
                self.non_decreasing_checks = 0
        self.checks.append(token_count)


def management_tools(
    workspace: MemoryWorkspace,
    audit: list[dict[str, Any]],
    live_audit_path: str | Path | None = None,
    core_repair_state: CoreRepairState | None = None,
) -> list[Any]:
    @tool(
        "shell",
        (
            "Run one portable POSIX shell command in the memory workspace, "
            "regardless of the host operating system, for things the built-in "
            "file tools cannot do: listing, moving, or removing files. "
            "The argument must be an executable command such as `ls topics`, "
            "never an instruction written in English. To change file contents, "
            "use Read, Edit, or Write instead. Never edit sources/."
        ),
        {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    )
    async def shell(arguments: dict[str, Any]) -> dict[str, Any]:
        command = str(arguments.get("command", ""))
        record: dict[str, Any] = {
            "round": len(audit),
            "tool": "shell",
            "command": command,
        }
        try:
            result = workspace.shell(command)
            output = f"{result.stdout}{result.stderr}"
            record.update({
                "returncode": result.returncode,
                "output": output,
                "count": workspace.last_created_blocks,
                "topic_paths": workspace.last_changed_topics,
                "windows_retries": workspace.last_windows_retries,
                "status": "ok" if result.returncode == 0 else "error",
            })
        except Exception as exc:  # validation failure becomes an MCP tool error
            output = f"{type(exc).__name__}: {exc}"
            record.update({"status": "error", "output": output})
        audit.append(record)
        if live_audit_path is not None:
            path = Path(live_audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return {
            "content": [{"type": "text", "text": output or "(no output)"}],
            "is_error": record["status"] == "error",
        }

    result = [shell]
    if core_repair_state is None:
        return result

    @tool(
        "core_token_count",
        (
            "Count staged core.md with the Runtime's exact tokenizer during "
            "a Core-capacity repair. This tool is read-only. Call it after "
            "each Core edit and keep the final count at or below the target."
        ),
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    async def count_core(_arguments: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "round": len(audit),
            "tool": "core_token_count",
        }
        if len(core_repair_state.checks) >= core_repair_state.max_checks:
            payload = {
                "error": "core token check limit reached",
                "max_checks": core_repair_state.max_checks,
                "checks": list(core_repair_state.checks),
            }
            record.update({"status": "error", "output": payload})
            is_error = True
        else:
            count = core_token_count(workspace.stage_dir / "core.md")
            core_repair_state.record(count)
            payload = {
                "token_count": count,
                "hard_limit": core_repair_state.limit,
                "target": core_repair_state.target,
                "remaining_to_limit": core_repair_state.limit - count,
                "remaining_to_target": core_repair_state.target - count,
                "checks_used": len(core_repair_state.checks),
                "checks_remaining": (
                    core_repair_state.max_checks
                    - len(core_repair_state.checks)
                ),
                "stagnated": core_repair_state.stagnated,
            }
            record.update({"status": "ok", "output": payload})
            is_error = False
        audit.append(record)
        if live_audit_path is not None:
            path = Path(live_audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False),
            }],
            "is_error": is_error,
        }

    result.append(count_core)
    return result
