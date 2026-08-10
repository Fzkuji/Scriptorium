"""Write one chunk of conversation through several calls at the same time.

A served Add is one request the caller holds open, and what it waits for is
almost entirely the model writing: measured across 384 production writes,
every extra thousand output tokens costs twenty-one seconds, and the endpoint
generates at fifty-odd tokens a second no matter how the work is arranged.
Turns do not divide that. Concurrency does, and this endpoint has room for it:
one call sustains 74 tokens a second, thirty-two at once sustain 52 each, so
sixteen times the throughput for a tenth off the speed of any one.

Splitting is safe here because a served write only ever records. Across 1905
production passes the model called `remember` 14142 times and `update` or
`forget` not once, and `remember` needs nothing from the workspace: the
subject, the fact and its sources are all the model supplies, while the file,
the block ID and the footnote are worked out on this side. So the messages
divide into groups, each group is read on its own, and every fact they return
is recorded together in the one transaction.

`write_sessions` still drives the single agent pass, and the interactive
memory keeps it: revising what is already written needs to see it.
"""
from __future__ import annotations

import asyncio
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..config import MemoryConfig
from ..workspace import MemoryWorkspace
from ..workspace.agent_pass import _baseline, _commit_turn
from .session import render_conversation
from .tools import writing_tools

# Groups are sized by how much text they carry, not by how many messages.
# Counting messages divided a chunk of twenty short ones into three calls and
# left a chunk of three long ones as a single call, which is the one that
# needed dividing.
CHARACTERS_PER_GROUP = 1_500
MESSAGES_PER_GROUP = 8
MOST_GROUPS = 4

# The writing system prompt teaches the topic file format: block IDs,
# footnotes, where a fact is filed. None of that is this call's business, and
# handing it over cost more than the conversation did — 1360 tokens of input
# against a chunk of 40, and an invitation to elaborate that answered three
# short messages with 3443 tokens of output. This one says only what the call
# is for.
READING_FOR_FACTS = """\
You read a piece of conversation and report what it recorded.

Call `remember` once for each fact the conversation states. Nothing else is
asked of you and there is no file to write.

A fact is something the conversation says: a date, a name, a number, a place,
a decision, a preference, a plan, or how two people are related. Report it in
one plain sentence, in the words the conversation used. Cite the `[ref]`
labels it came from.

Report what is there. A short exchange holds few facts, and reporting three is
a better answer than inventing thirty. Say nothing about pleasantries,
greetings, or your own reading of the conversation.
"""

READ_THIS = """\
This was said on {observed}.

{conversation}
"""


def _groups(turns: list[Any], refs: list[str]) -> list[tuple[list[Any], list[str]]]:
    """Divide one chunk into the parts that will be read at the same time."""
    characters = sum(len(str(text)) for _, text in turns)
    count = max(
        math.ceil(characters / CHARACTERS_PER_GROUP),
        math.ceil(len(turns) / MESSAGES_PER_GROUP),
    )
    count = max(1, min(MOST_GROUPS, count, len(turns)))
    size = math.ceil(len(turns) / count)
    return [
        (turns[at:at + size], refs[at:at + size])
        for at in range(0, len(turns), size)
    ]


def _facts_from(agent: Any, prompt: str, seconds: float | None) -> tuple[list[dict], dict]:
    """One model call, returning the `remember` arguments it asked for.

    The call is made with the writing tool schema so the model answers in the
    shape the recorder already accepts, but no workspace is handed over: this
    reads the conversation and nothing else.
    """
    result = agent.run(
        prompt=prompt,
        system_prompt=READING_FOR_FACTS,
        cwd=Path.cwd(),
        tools=None,
        max_turns=1,
        max_seconds=seconds,
        tool_schemas=_REMEMBER_SCHEMA,
    )
    facts = []
    for call in result.turns or []:
        if call.get("tool") != "remember":
            continue
        try:
            facts.append(json.loads(call.get("arguments") or "{}"))
        except json.JSONDecodeError:
            continue
    return facts, {
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "reason": result.stop_reason,
    }


_REMEMBER_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Record one fact about one subject. Give the subject, what sort of "
            "subject it is, the fact in plain prose, and the `[ref]` labels it "
            "came from."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["person", "project", "relationship", "theme"],
                },
                "fact": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}},
                "date": {"type": "string"},
            },
            "required": ["subject", "kind", "fact", "sources"],
        },
    },
}]


def write_sessions_in_parallel(
    memory_dir: str | Path,
    *,
    agent: Any,
    sessions: list[dict[str, Any]],
    config: MemoryConfig | None = None,
) -> list[dict[str, Any]]:
    """Read one chunk in several groups at once, record what they all found."""
    config = config or MemoryConfig()
    workspace = MemoryWorkspace(memory_dir, config=config)
    try:
        workspace.archive_sessions(sessions)
        workspace._refresh_stage()
        audit: list[dict[str, Any]] = []
        observed = sessions[0]["observation_date"] if sessions else ""

        prompts = []
        for session in sessions:
            for turns, refs in _groups(session["turns"], session["refs"]):
                prompts.append(READ_THIS.format(
                    observed=session["observation_date"],
                    conversation=render_conversation(turns, refs),
                ))

        # Every group gets the whole budget, because they run together and the
        # pass is over when the slowest returns.
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            found = list(pool.map(
                lambda p: _facts_from(agent, p, config.max_seconds), prompts
            ))

        baseline = _baseline(workspace)
        remember = next(
            definition for definition in writing_tools(
                workspace, audit, observed=observed
            ) if definition.name == "remember"
        )
        for facts, _ in found:
            for fact in facts:
                asyncio.run(remember.handler(fact))
        error = _commit_turn(workspace, baseline, audit)

        spent = [usage for _, usage in found]
        audit.append({
            "tool": "agent",
            "status": "rejected" if error else "ok",
            "reason": "rejected" if error else "complete",
            "rounds": len(prompts),
            "turns": [],
            "input_tokens": sum(u["input_tokens"] for u in spent),
            "output_tokens": sum(u["output_tokens"] for u in spent),
            "anthropic_equivalent_cost_usd": None,
        })
        return audit
    finally:
        import shutil

        shutil.rmtree(workspace.stage_dir, ignore_errors=True)
