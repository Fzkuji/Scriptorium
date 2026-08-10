"""remember, update and forget: the tools a writing pass hands the model.

Where a fact lands, how its paragraph and footnote are shaped, and what
block ID it gets are all worked out by `subjects` and `render`; a model that
had to get any of that right itself is a model spending its turns on the
contract instead of on judging what the conversation says.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..markdown.syntax import BLOCK_SUFFIX, SINGLE_CITATION
from ..organizing.tools import _repair_guidance
from ..workspace import MemoryWorkspace
from ..workspace.tool_support import edit_tools
from . import render, subjects


def _find_block(target: Path, fragment: str) -> tuple[int, str]:
    """The paragraph holding `fragment`, and the block ID it carries.

    Pointing at a fact by a phrase from it is the only handle the model
    has: block IDs belong to the Runtime and it never sees them.
    """
    if not target.is_file():
        raise ValueError(
            f"nothing recorded about that subject yet: {target.name}"
        )
    wanted = " ".join(fragment.split()).lower()
    if not wanted:
        raise ValueError("say which recorded fact this is about")
    lines = target.read_text(encoding="utf-8").split("\n")
    hits = [
        index for index, line in enumerate(lines)
        if not line.startswith("[^")
        and wanted in " ".join(line.split()).lower()
    ]
    if not hits:
        raise ValueError(
            f"no recorded fact contains {fragment!r}; read the file and "
            "quote a phrase from the one you mean"
        )
    if len(hits) > 1:
        raise ValueError(
            f"{len(hits)} recorded facts contain {fragment!r}; quote a "
            "longer phrase so it picks out one"
        )
    block = BLOCK_SUFFIX.search(lines[hits[0]])
    if not block:
        raise ValueError("that line is not a recorded fact")
    return hits[0], block.group("id")


def writing_tools(
    workspace: MemoryWorkspace,
    audit: list[dict[str, Any]],
    *,
    observed: str = "",
    commit_each: bool = True,
) -> list[Any]:
    """The three verbs a writing pass may call: remember, update, forget.

    Nothing else is offered in this pass. Recording a fact needs one tool
    call; a shell or a whole-file writer would only be latitude for a weak
    model to spend on a rejected edit instead of on the conversation.

    `commit_each` is for a caller that holds every fact before it records any
    of them: it stages the edits and commits the batch once. A model calling
    these tools between turns keeps the default, because the verdict on one
    edit is what it needs to decide the next.
    """
    # Imported here so reading memory does not require the agent SDK: only a
    # run that actually writes needs it.
    from claude_agent_sdk import tool

    _apply, _record = edit_tools(
        workspace, audit, _repair_guidance, commit_each=commit_each
    )

    @tool(
        "remember",
        (
            "Record one fact about one subject. Give the subject, what sort "
            "of subject it is, the fact in plain prose, and the `[ref]` "
            "labels it came from. Which file it lands in, the block ID and "
            "the evidence footnote are all worked out for you."
        ),
        {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "who or what the fact is about, e.g. Dave",
                },
                "kind": {
                    "type": "string",
                    "enum": ["person", "project", "relationship", "theme"],
                    "description": "what sort of subject that is",
                },
                "fact": {
                    "type": "string",
                    "description": "one sentence, no markup and no citation",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "the [ref] labels the fact came from",
                },
                "heading": {
                    "type": "string",
                    "description": "optional section within the file",
                },
                "date": {
                    "type": "string",
                    "description": "YYYY-MM-DD the fact is about, if it differs "
                                   "from when it was said",
                },
            },
            "required": ["subject", "kind", "fact", "sources"],
            "additionalProperties": False,
        },
    )
    async def remember(arguments: dict[str, Any]) -> dict[str, Any]:
        def run() -> str:
            fact = str(arguments.get("fact", "")).strip()
            if not fact:
                raise ValueError("fact is required")
            sources = render.sources_of(arguments)
            subject = str(arguments.get("subject", "")).strip()
            if not subject:
                raise ValueError("subject is required")
            path = subjects.path_for(
                subject, str(arguments.get("kind", "") or "theme")
            )
            heading = str(arguments.get("heading", "")).strip()
            when = render.when(arguments, observed)

            def change(target: Path) -> None:
                target.parent.mkdir(parents=True, exist_ok=True)
                existing = (
                    target.read_text(encoding="utf-8")
                    if target.is_file() else ""
                )
                title = target.stem.replace("-", " ").title()
                body = existing or f"# {title}\n"
                # A name unique within this edit; the Runtime turns both
                # placeholders into stable IDs once the edit is accepted.
                tag = f"r{len(audit)}"
                paragraph, note = render.fact_block(fact, sources, when, tag)
                body = render.insert(body, heading, paragraph) + f"\n{note}"
                target.write_text(body, encoding="utf-8")

            return _apply(path, change)
        return _record("remember", arguments, run)

    @tool(
        "update",
        (
            "Record a fact that replaces one already in memory — a plan "
            "changed, a value corrected, a preference reversed. Quote a "
            "phrase from the existing fact as `replaces`; the new fact is "
            "recorded and linked to it, and the old one stays as the record "
            "of what was true before."
        ),
        {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["person", "project", "relationship", "theme"],
                },
                "replaces": {
                    "type": "string",
                    "description": "a phrase from the fact being replaced",
                },
                "fact": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}},
                "date": {"type": "string"},
            },
            "required": ["subject", "kind", "replaces", "fact", "sources"],
            "additionalProperties": False,
        },
    )
    async def update(arguments: dict[str, Any]) -> dict[str, Any]:
        def run() -> str:
            fact = str(arguments.get("fact", "")).strip()
            sources = render.sources_of(arguments)
            if not fact:
                raise ValueError("fact is required")
            path = subjects.path_for(
                str(arguments.get("subject", "")).strip(),
                str(arguments.get("kind", "") or "theme"),
            )
            when = render.when(arguments, observed)
            replaces = str(arguments.get("replaces", ""))

            def change(target: Path) -> None:
                _, block_id = _find_block(target, replaces)
                tag = f"u{len(audit)}"
                paragraph, note = render.replacement_block(
                    fact, sources, when, tag, block_id
                )
                body = target.read_text(encoding="utf-8")
                target.write_text(
                    render.insert(body, "", paragraph) + f"\n{note}",
                    encoding="utf-8",
                )

            return _apply(path, change)
        return _record("update", arguments, run)

    @tool(
        "forget",
        (
            "Withdraw a fact that was recorded in error or that the person "
            "asked to have removed. Quote a phrase from it. Its wording is "
            "erased, so nothing can retrieve it again. Use `update` instead "
            "when something merely changed — what was true before is part of "
            "the record."
        ),
        {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["person", "project", "relationship", "theme"],
                },
                "fact": {
                    "type": "string",
                    "description": "a phrase from the fact to remove",
                },
            },
            "required": ["subject", "kind", "fact"],
            "additionalProperties": False,
        },
    )
    async def forget(arguments: dict[str, Any]) -> dict[str, Any]:
        def run() -> str:
            path = subjects.path_for(
                str(arguments.get("subject", "")).strip(),
                str(arguments.get("kind", "") or "theme"),
            )

            def change(target: Path) -> None:
                index, block_id = _find_block(
                    target, str(arguments.get("fact", ""))
                )
                lines = target.read_text(encoding="utf-8").split("\n")
                # The wording goes; the block ID stays. Identity is the
                # Runtime's and every link and derived view reaches through
                # it, so a paragraph that vanished would take working links
                # with it. What is left says a fact was here and was
                # withdrawn, which is itself true and is all that remains.
                citations = "".join(
                    f"[^{name}]" for name in SINGLE_CITATION.findall(lines[index])
                )
                lines[index] = (
                    f"A fact recorded here was withdrawn.{citations} "
                    f"^{block_id}"
                )
                target.write_text("\n".join(lines), encoding="utf-8")

            return _apply(path, change) and f"withdrawn from {path}"
        return _record("forget", arguments, run)

    return [remember, update, forget]
