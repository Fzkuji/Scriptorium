"""Claude Code structured reconciliation for free-form Topic edits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import MemoryConfig
from .reconciliation import ReconciliationError, ReconciliationResult


_RECONCILIATION_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "creates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "when": {"type": ["string", "null"]},
                    "source_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["content", "when", "source_refs"],
                "additionalProperties": False,
            },
        },
        "deleted_ids": {"type": "array", "items": {"type": "string"}},
        "organizational_quotes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "matches",
        "creates",
        "deleted_ids",
        "organizational_quotes",
    ],
    "additionalProperties": False,
}


def _make_reconciler(
    agent: Any,
    *,
    usage_logger: Any | None,
    config: MemoryConfig,
    cwd: str | Path,
):
    def reconcile(edited_text, old_units, candidate_sources):
        payload = {
            "edited_topic_text": edited_text,
            "old_memories": [
                {
                    "memory_id": unit.memory_id,
                    "content": unit.content,
                    "when": unit.when,
                    "source_refs": list(unit.source_refs),
                }
                for unit in old_units
            ],
            "candidate_sources": sorted(candidate_sources),
        }
        result = agent.run(
            prompt=json.dumps(payload, ensure_ascii=False),
            system_prompt=(
                "Reconcile edited Topic Markdown with existing memory IDs. "
                "Every returned text value must be an exact quote from "
                "edited_topic_text. Use only candidate_sources."
            ),
            cwd=cwd,
            tools=[],
            max_turns=config.max_turns,
            max_budget_usd=config.max_budget_usd,
            output_schema=_RECONCILIATION_SCHEMA,
        )
        if usage_logger is not None:
            usage_logger(result)
        value = result.structured_output
        if not isinstance(value, dict):
            raise ReconciliationError(
                "Reconciler did not return structured output"
            )
        creates = tuple(
            (
                str(row["content"]),
                None if row.get("when") is None else str(row["when"]),
                tuple(str(ref) for ref in row.get("source_refs", [])),
            )
            for row in value.get("creates", [])
        )
        return ReconciliationResult(
            matches={
                str(key): str(quote)
                for key, quote in value.get("matches", {}).items()
            },
            creates=creates,
            deleted_ids=tuple(
                str(item) for item in value.get("deleted_ids", [])
            ),
            organizational_quotes=tuple(
                str(item)
                for item in value.get("organizational_quotes", [])
            ),
        )

    return reconcile
