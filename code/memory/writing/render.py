"""A fact becomes contract-conformant Markdown.

`remember` records a new paragraph; `update` records one that replaces an
earlier paragraph. The two need the same citation and block-marker shape,
differing only in one clause, and used to build it inline in two separate
places that could each drift toward saying it slightly differently. This
module is the one place that renders it, so `remember` and `update` cannot
disagree about what a recorded fact looks like.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def sources_of(arguments: dict[str, Any]) -> list[str]:
    found = [
        str(value).strip()
        for value in (arguments.get("sources") or []) if str(value).strip()
    ]
    if not found:
        raise ValueError(
            "sources is required: name the [ref] labels this fact came from, "
            "as they appear in the conversation"
        )
    return found


def when(arguments: dict[str, Any], observed: str) -> str:
    """A footnote with no time is not one the parser accepts."""
    return (
        str(arguments.get("date", "")).strip()
        or observed
        or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )


def fact_block(
    fact: str, sources: list[str], when: str, tag: str
) -> tuple[str, str]:
    """The paragraph and its footnote definition for one new fact.

    `remember` and `update` both render through this, so a citation or a
    block marker cannot come out shaped differently between the two verbs.
    `tag` names the placeholder pair for this edit; the Runtime replaces both
    with stable IDs once the edit is accepted.
    """
    paragraph = f"{fact}[^new-evidence-{tag}] ^new-block-{tag}\n"
    note = (
        f"[^new-evidence-{tag}]: Time: `{when}`; Sources: {', '.join(sources)}\n"
    )
    return paragraph, note


def replacement_block(
    fact: str, sources: list[str], when: str, tag: str, block_id: str
) -> tuple[str, str]:
    """The same rendering, with the paragraph linked to the fact it replaces.

    The link is what makes the supersession readable: the old paragraph
    keeps its ID and its date, and the new one says what it stands in for.
    """
    return fact_block(
        f"{fact} (replaces [the earlier note](#^{block_id}))",
        sources, when, tag,
    )


def insert(body: str, heading: str, block: str) -> str:
    """Put a paragraph under its heading, or at the end of the prose.

    Footnote definitions collect at the foot of the file, so a new paragraph
    goes above them however the file is sectioned.
    """
    lines = body.rstrip("\n").split("\n")
    notes = next(
        (
            index for index, line in enumerate(lines)
            if line.startswith("[^") and "]:" in line
        ),
        len(lines),
    )
    if heading:
        marked = f"## {heading}"
        if marked not in lines[:notes]:
            lines[notes:notes] = ["", marked]
            notes += 2
        else:
            start = lines.index(marked) + 1
            notes = next(
                (
                    index for index in range(start, notes)
                    if lines[index].startswith("#")
                ),
                notes,
            )
    lines[notes:notes] = ["", block.rstrip("\n")]
    return "\n".join(lines) + "\n"
