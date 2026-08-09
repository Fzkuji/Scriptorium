"""The tools an organizing pass may call, and what a rejected edit means.

`shell` is for what file tools cannot do: listing, moving, removing. The
whole-file `write_file` / `edit_file` / `read_file` triad is for a runtime
that brings no file tools of its own. `organizing_tools` also hands out
`remember`, `update` and `forget`, sourced from `memory.writing.tools`
rather than redefined here, so the write and organize stages cannot describe
the same verb two different ways.

`_repair_guidance` turns a validation failure into the correction to send
back with it. What a rejection means is this package's business, not the
substrate's: `memory.workspace.tool_support` applies and records an edit for
any pass, but only the pass that offered the verb knows what "block ID must
not be removed" should tell the model to do differently next. Writing
imports `_repair_guidance` too, since `remember`, `update` and `forget`
commit through the same transaction and a rejected one needs the same kind
of correction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..workspace import MemoryWorkspace
from ..workspace.tool_support import edit_tools, staged_path


def _repair_guidance(error: str) -> str:
    """What to change, for the validation failures that recur.

    The raw message names the rule that was broken; a writer that has just
    broken it needs the correction instead.
    """
    if "memory block ID required" in error:
        return (
            "A new paragraph ends with its own `^new-block-<name>` marker, "
            "after the citation: `Dave moved to Pudong.[^new-evidence-move] "
            "^new-block-move`. The Runtime replaces the placeholder with a "
            "stable ID once the edit is accepted."
        )
    if "unused footnote definition" in error:
        return (
            "Every `[^...]:` definition line needs a paragraph citing it. "
            "Either cite it or remove the definition."
        )
    if "appears 0 times" in error:
        return (
            "`old_text` has to be the passage exactly as the file has it. "
            "Read the file first and copy the passage out of it; an "
            "abbreviated or `...`-elided quotation matches nothing."
        )
    if "appears" in error and "times" in error:
        return (
            "`old_text` has to pick out one passage. Include enough "
            "surrounding words that it appears exactly once."
        )
    if "block ID must not be removed" in error:
        return (
            "A paragraph that already ends in `^id` keeps that ID. Copy it "
            "through unchanged rather than dropping or renumbering it."
        )
    if "memory source links required" in error or "undefined footnote" in error:
        return (
            "A citation needs its `[^eN]:` definition line in the same file. "
            "Either add the definition or leave the existing `[^e-...]` "
            "citation exactly as it was."
        )
    if "Source Memory is append-only" in error:
        return (
            "Nothing under sources/ may be edited. Write the fact into a "
            "Topic file under topics/ instead."
        )
    if "Core Memory exceeds" in error:
        return (
            "core.md is full. Leave it alone and put this in a Topic file."
        )
    return ""


def organizing_tools(
    workspace: MemoryWorkspace,
    audit: list[dict[str, Any]],
    *,
    file_tools: bool = False,
    observed: str = "",
    stage: str = "",
) -> list[Any]:
    """Tools for one writing or organizing pass.

    ``file_tools`` supplies read, write and edit for a runtime that brings no
    file tools of its own. Claude Code has its own built-ins and the prompts
    are written against them; an OpenAI-compatible model has only what is
    passed here, and without these it is told to use a Write tool it does not
    have and falls back to shell redirection.
    """
    # Imported here so reading memory does not require the agent SDK: only a
    # run that actually writes needs it.
    from claude_agent_sdk import tool

    def _guidance(command: str, output: str) -> str:
        """What to do about a failure, appended to the raw error.

        A shell error says what went wrong in the shell's terms. Naming the
        tool that does the job turns a retry-the-same-thing loop into one
        corrected call.
        """
        if "No such file or directory" in output and ">" in command:
            return (
                "Redirecting into a path whose directory does not exist fails. "
                "Use the Write tool instead: it creates parent directories and "
                "takes the finished file in one call."
            )
        if "Read-only file system" in output or "Permission denied" in output:
            return (
                "Files under sources/ are the read-only evidence record. The "
                "conversation text is already in your prompt; write the fact "
                "into a Topic file under topics/ instead."
            )
        if "command not found" in output or "syntax error" in output:
            return (
                "This argument has to be an executable command. To create or "
                "change a file's contents, call the Write or Edit tool rather "
                "than describing the change here."
            )
        return ""

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
        if record["status"] == "error":
            advice = _guidance(command, output) or _repair_guidance(output)
            if advice:
                output = f"{output}\n\n{advice}" if output else advice
        return {
            "content": [{"type": "text", "text": output or "(no output)"}],
            "is_error": record["status"] == "error",
        }

    if not file_tools:
        return [shell]

    _apply, _record = edit_tools(workspace, audit, _repair_guidance)

    @tool(
        "write_file",
        (
            "Write a file's whole contents, creating it and any missing "
            "parent directories. Use for a new Topic file or a rewrite."
        ),
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    )
    async def write_file(arguments: dict[str, Any]) -> dict[str, Any]:
        def run() -> str:
            def change(target: Path) -> None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    str(arguments.get("content", "")), encoding="utf-8"
                )
            return _apply(str(arguments.get("path", "")), change)
        return _record("write_file", arguments, run)

    @tool(
        "edit_file",
        (
            "Replace one exact passage in a file. `old_text` must appear "
            "exactly once. Use for revising a paragraph and leaving the rest."
        ),
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
    )
    async def edit_file(arguments: dict[str, Any]) -> dict[str, Any]:
        def run() -> str:
            def change(target: Path) -> None:
                if not target.is_file():
                    raise ValueError(f"no such file: {arguments.get('path')}")
                text = target.read_text(encoding="utf-8")
                old = str(arguments.get("old_text", ""))
                found = text.count(old)
                if found != 1:
                    raise ValueError(
                        f"old_text appears {found} times; it has to appear "
                        "exactly once"
                    )
                target.write_text(
                    text.replace(old, str(arguments.get("new_text", ""))),
                    encoding="utf-8",
                )
            return _apply(str(arguments.get("path", "")), change)
        return _record("edit_file", arguments, run)

    @tool(
        "read_file",
        "Read a file in the memory workspace.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )
    async def read_file(arguments: dict[str, Any]) -> dict[str, Any]:
        def run() -> str:
            target = staged_path(workspace, str(arguments.get("path", "")))
            if not target.is_file():
                return f"no such file: {arguments.get('path')}"
            return target.read_text(encoding="utf-8")
        return _record("read_file", arguments, run)

    # Deferred to break the import cycle: memory.writing.tools imports
    # _repair_guidance from this module, so this module cannot import it back
    # at load time. By the time organizing_tools is actually called, both
    # modules have finished loading and the cycle is not a problem.
    from ..writing.tools import writing_tools
    verbs = writing_tools(workspace, audit, observed=observed)

    if stage == "write":
        # Recording a fact needs one tool. A shell and a whole-file writer are
        # latitude a weak model spends on rejected edits, not on the work.
        return verbs
    return [shell, *verbs, write_file, edit_file, read_file]
