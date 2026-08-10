"""Machinery shared by every tool that edits the staged workspace.

Writing's three verbs (`memory.writing.tools`) and the whole-file read,
write and edit tools (`memory.organizing.tools`) both resolve a path inside
the stage, apply one change, and commit it through the same transaction:
accepted, or rolled back with the reason. A
model that cannot see why an edit was refused tries the same edit again, so
`record` also counts failures, once per (tool, path, message), so a repeat
can say that it is one. Two copies of this existed only because both tool
sets used to be defined in the same function; splitting the function without
splitting this would have made that coincidence permanent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .staging import MemoryWorkspace


def staged_path(workspace: MemoryWorkspace, path: str) -> Path:
    """Resolve a workspace-relative path inside the stage, and nowhere else."""
    if not path:
        raise ValueError("path is required")
    target = (workspace.stage_dir / path).resolve()
    root = workspace.stage_dir.resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"path escapes the workspace: {path}")
    return target


def edit_tools(
    workspace: MemoryWorkspace,
    audit: list[dict[str, Any]],
    guidance: Callable[[str], str],
    *,
    commit_each: bool = True,
) -> tuple[
    Callable[[str, Callable[[Path], None]], str],
    Callable[[str, dict[str, Any], Callable[[], str]], dict[str, Any]],
]:
    """Build the (apply, record) pair bound to one pass's workspace and audit.

    One `failed` counter is shared by every tool built from the same call,
    which is what lets `record` recognise a model retrying the exact call
    that was just refused. It resets per pass rather than per tool, since a
    fresh pass deserves a fresh count.
    """
    failed: dict[tuple[str, str], int] = {}

    def current_text(path: str, limit: int = 8000) -> str:
        """The staged file as it is now, for an edit that has to preserve it."""
        try:
            target = staged_path(workspace, path)
        except ValueError:
            return ""
        if not target.is_file():
            return ""
        text = target.read_text(encoding="utf-8")
        return text if len(text) <= limit else text[:limit] + "\n… (truncated)"

    def apply(path: str, change: Callable[[Path], None]) -> str:
        """Run one edit against the stage and commit it the way shell does.

        Same validation, same rollback: a rejected edit leaves the workspace
        as it was, and the reason comes back as the tool result.

        A commit parses every topic file, rebuilds the block index and
        re-checks every source reference in the workspace, so committing per
        edit costs the whole workspace once per edit: twelve facts against a
        fifty-topic memory measured 8.4s, and it grows as the memory does.
        A model editing between turns needs that verdict immediately. A
        caller holding every edit already needs it once, at the end.
        """
        target = staged_path(workspace, path)
        if not commit_each:
            workspace.last_changed_topics = []
            workspace.last_created_blocks = 0
            change(target)
            return f"wrote {path}"
        before_units, before_block_ids, before_topics, before_sources = (
            workspace.baseline()
        )
        workspace.last_changed_topics = []
        workspace.last_created_blocks = 0
        change(target)
        workspace.commit_edits(
            before_units, before_block_ids, before_topics, before_sources
        )
        return f"wrote {path}"

    def record(
        name: str, arguments: dict[str, Any], run: Callable[[], str]
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "round": len(audit), "tool": name, "command": str(arguments)[:400],
        }
        try:
            output = run()
            entry.update({
                "status": "ok", "output": output,
                "count": workspace.last_created_blocks,
                "topic_paths": workspace.last_changed_topics,
            })
        except Exception as exc:  # the model is the one that has to fix it
            output = f"{type(exc).__name__}: {exc}"
            advice = guidance(output)
            if advice:
                output = f"{output}\n\n{advice}"
            # A model that cannot see why a call was refused will make it
            # again, and a turn budget goes on one rejected edit repeated
            # twenty times. Say that it is the same call.
            # Keyed on the file and the kind of refusal, not the exact
            # arguments: a model retrying the same mistake usually rewords
            # it slightly, and an exact-match counter never fires.
            signature = (
                name, str(arguments.get("path", "")), output.split("\n")[0][:60],
            )
            failed[signature] = failed.get(signature, 0) + 1
            if failed[signature] > 1:
                output += (
                    f"\n\nThis is attempt {failed[signature]} at the same "
                    "call, and it fails the same way each time. Repeating it "
                    "will not work. Read the file, then change the one "
                    "passage that needs changing."
                )
            if "block ID must not be removed" in output:
                current = current_text(str(arguments.get("path", "")))
                if current:
                    output += (
                        "\n\nThe file as it stands is below. Every `^id` in "
                        "it has to survive your edit.\n\n" + current
                    )
            entry.update({"status": "error", "output": output})
        audit.append(entry)
        return {
            "content": [{"type": "text", "text": output or "(no output)"}],
            "is_error": entry["status"] == "error",
        }

    return apply, record
