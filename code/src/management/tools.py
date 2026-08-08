"""Claude Code MCP tools for transactional memory edits."""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import tool

from .workspace import MemoryWorkspace


def management_tools(
    workspace: MemoryWorkspace,
    audit: list[dict[str, Any]],
) -> list[Any]:
    @tool(
        "shell",
        (
            "Run one POSIX shell command in the memory workspace, for things "
            "the file tools cannot do: listing, moving, or removing files. "
            "The argument must be an executable command such as `ls topics`, "
            "never an instruction written in English. To create or change a "
            "file's contents use the Write and Edit tools instead of this one."
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
                "status": "ok" if result.returncode == 0 else "error",
            })
        except Exception as exc:  # validation failure becomes an MCP tool error
            output = f"{type(exc).__name__}: {exc}"
            record.update({"status": "error", "output": output})
        audit.append(record)
        return {
            "content": [{"type": "text", "text": output or "(no output)"}],
            "is_error": record["status"] == "error",
        }

    return [shell]
