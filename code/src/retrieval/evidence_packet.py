"""Deterministic P1 presentation for an unchanged P0 candidate list."""

from __future__ import annotations

import re
from typing import Any


_CANCELLED = re.compile(r"\b(cancelled|canceled|called off|did not|didn't)\b", re.I)
_PLANNED = re.compile(
    r"\b(?:the user|i)\s+(?:plans?|planned|intends?|intended|will|would|"
    r"is considering|was considering|is thinking|was thinking|hopes?|hoped)\b",
    re.I,
)
_COMPLETED = re.compile(
    r"\b(?:the user|i)\s+(?:has\s+|had\s+)?(?:attended|bought|completed|"
    r"downloaded|finished|got|made|married|purchased|raised|spent|tried|"
    r"visited|went|won)\b",
    re.I,
)


def evidence_state(text: str) -> str:
    """Conservatively label explicit event state; otherwise remain unspecified."""
    if _CANCELLED.search(text):
        return "cancelled_or_negated"
    if _PLANNED.search(text):
        return "planned"
    if _COMPLETED.search(text):
        return "completed"
    return "unspecified"


def build_evidence_packet(
    rows: list[dict[str, Any]], contract: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Compile the same candidates, in the same order, into a static packet."""
    entries: list[str] = []
    states: dict[str, int] = {}
    unique_refs: set[str] = set()
    for rank, row in enumerate(rows, start=1):
        state = evidence_state(str(row.get("content", "")))
        states[state] = states.get(state, 0) + 1
        refs = list(row.get("refs") or [])
        unique_refs.update(str(ref) for ref in refs)
        view = str(row.get("view") or (
            "source" if str(row.get("path", "")).startswith("sources/")
            else "topic"
        ))
        entries.append(
            f'<evidence id="E{rank:02d}" rank="{rank}" '
            f'view="{view}" state="{state}" '
            f'date="{row.get("date") or "unknown"}" '
            f'refs="{", ".join(refs) or "none"}">\n'
            f'{str(row.get("content", "")).strip()}\n'
            f'</evidence>'
        )
    shape = str(contract.get("answer_shape") or "direct_fact")
    axis = str(contract.get("coverage_axis") or "best_supported_fact")
    rendered = (
        f'<evidence_packet answer_shape="{shape}" coverage_axis="{axis}" '
        f'candidate_count="{len(rows)}">\n'
        "Use every relevant entry. Treat state labels as conservative hints; "
        "the quoted evidence text controls if a label is uncertain. Count "
        "distinct real-world events/items, not repeated mentions or source copies.\n\n"
        + ("\n\n".join(entries) or "(no evidence)")
        + "\n</evidence_packet>"
    )
    return rendered, {
        "enabled": True,
        "candidate_count": len(rows),
        "candidate_ranks": list(range(1, len(rows) + 1)),
        "unique_source_refs": len(unique_refs),
        "state_counts": states,
    }
