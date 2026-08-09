"""The write pass: turn a batch of conversation into remember / update / forget calls.

Writing is a single pass with a fixed instruction: read the conversation,
and for each fact worth keeping, decide which of the three verbs applies.
Everything else about the format is computed downstream of that choice, in
`subjects` and `render`. This module is what runs the pass — building the
task, choosing the tools, and calling the shared agent-pass machinery with
this stage's rules.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..config import MemoryConfig
from ..organizing.tools import _repair_guidance, organizing_tools
from ..prompts import SYSTEM_PROMPT, WRITE_MEMORY
from ..workspace.agent_pass import run_pass
from . import subjects
from .tools import writing_tools

# The writer protocol hash covers the tool surface as well as the prompts, so
# a change to either invalidates a capacity calibration measured against it.
# This mirrors the `shell` tool's schema: the one tool a runtime with its own
# file tools is given for a write pass, since it edits Topic files with its
# built-in Read, Write and Edit instead of calling the three verbs below.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": (
                "Run one POSIX shell command in the memory workspace, for "
                "things the file tools cannot do: listing, moving, or "
                "removing files."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


def render_conversation(
    turns: list[tuple[str, str]], refs: list[str]
) -> str:
    return "\n".join(
        f"[{ref}] {speaker}: {text}"
        for (speaker, text), ref in zip(turns, refs)
    )


# Said only when the tools it names are the ones actually supplied. The
# shared writing prompt is written against Claude Code's built-ins; a runtime
# given ours has a different set in front of it and has to be told so.
WRITE_PROTOCOL = """
Record each fact with the `remember` tool, once per fact: the subject it is
about, what sort of subject that is, the fact in plain prose, and the `[ref]`
labels it came from. Where it is filed, its block ID and its evidence
footnote are worked out for you, so a fact recorded this way cannot break the
format. There is no other tool in this pass and nothing else to decide: work
through the conversation and record what it says.

A fact belongs to whoever said it. "Absolutely, Caroline! I cherish time with
family" is a fact about the speaker, not about Caroline, however often their
name appears in it — a line addressed to someone is not a line about them.
{subjects}"""


def _write_protocol(memory_dir: str | Path) -> str:
    return WRITE_PROTOCOL.format(subjects=subjects.known_subjects(memory_dir))


def _observed(sessions: list[dict[str, Any]] | None) -> str:
    """The date the last batch was said on, for footnotes written for the model."""
    for session in reversed(sessions or []):
        date = str(session.get("observation_date", "")).strip()
        if date:
            return date
    return ""


def _tools_for(agent: Any, sessions: list[dict[str, Any]] | None):
    """The tool list a write pass hands the model, given the workspace it edits.

    Claude Code brings its own Read, Write and Edit, so a write pass gives it
    only the shell and lets it hand-write Topic files against the prompts
    this pass is run with. A model with no file tools of its own gets the
    three verbs instead: without them it is told by the prompt to call a
    tool it does not have.
    """
    def build(workspace: Any, audit: list[dict[str, Any]]) -> list[Any]:
        if getattr(agent, "has_file_tools", True):
            return organizing_tools(workspace, audit)
        return writing_tools(workspace, audit, observed=_observed(sessions))
    return build


def render_writer_task(sessions: list[dict[str, Any]]) -> str:
    """Render one writer batch.

    A batch is however much source text fits the input budget. Each part
    carries its own observation date, because relative expressions like
    "yesterday" resolve against the date of the text they appear in.
    """
    rendered = []
    for session in sessions:
        rendered.append(
            f"## Observed {session['observation_date']}\n\n"
            f"{render_conversation(session['turns'], session['refs'])}"
        )
    return WRITE_MEMORY.format(sessions="\n\n".join(rendered))


def render_writer_input(sessions: list[dict[str, Any]]) -> str:
    """Render all fixed and session-specific text sent to the Writer."""
    return f"{SYSTEM_PROMPT}\n\n{render_writer_task(sessions)}"


def writer_protocol_sha256() -> str:
    payload = json.dumps(
        {
            "system": SYSTEM_PROMPT,
            "write_memory": WRITE_MEMORY,
            "tools": TOOLS[:1],
            "runtime": "claude-agent-sdk",
            "contract": "topic-core-v3-runtime-ids",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_sessions(
    memory_dir: str | Path,
    *,
    agent: Any,
    sessions: list[dict[str, Any]],
    usage_logger: Any | None = None,
    config: MemoryConfig | None = None,
) -> list[dict[str, Any]]:
    task = render_writer_task(sessions)
    return run_pass(
        memory_dir,
        agent=agent,
        task=task,
        source_sessions=sessions,
        usage_logger=usage_logger,
        config=config,
        stage="write",
        tools=_tools_for(agent, sessions),
        guidance=_repair_guidance,
        protocol=_write_protocol,
    )
