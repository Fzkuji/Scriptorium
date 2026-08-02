"""LLM-backed reconciliation for free-form Topic edits."""

import json
import re
from typing import Any

from .reconciliation import ReconciliationError, ReconciliationResult
from .config import MemoryConfig
from .provider import _chat_completion_with_retry, _provider_options


def _make_reconciler(
    client: Any,
    model: str,
    usage_logger: Any | None,
    config: MemoryConfig,
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
        response = _chat_completion_with_retry(
            client.chat.completions.create,
            retry_log=config.retry_log,
            model=model,
            messages=[{
                "role": "system",
                "content": (
                    "Reconcile edited Topic Markdown with existing memory IDs. "
                    "Every returned text value must be an exact quote from "
                    "edited_topic_text. Use only candidate_sources. Output JSON "
                    "with matches, creates, deleted_ids, and "
                    "organizational_quotes. creates contains objects with "
                    "content, when (YYYY-MM-DD or null), and source_refs."
                ),
            }, {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            }],
            temperature=0.0,
            **_provider_options(config),
        )
        if usage_logger is not None:
            usage_logger(response)
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ReconciliationError("Reconciler did not return JSON")
        value = json.loads(match.group(0))
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
                str(value) for value in value.get("deleted_ids", [])
            ),
            organizational_quotes=tuple(
                str(value)
                for value in value.get("organizational_quotes", [])
            ),
        )

    return reconcile
