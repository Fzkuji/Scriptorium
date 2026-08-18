"""Deterministic P2 sufficiency signals over an unchanged P1 packet."""

from __future__ import annotations

import re
from typing import Any


_TEMPORAL_SCOPE = re.compile(
    r"\b(current(?:ly)?|day before|past|this (?:week|month|year)|"
    r"last (?:day|week|month|year)|before|after|latest|most recent)\b",
    re.I,
)
_NUMERIC_COMPARISON = re.compile(
    r"\b(how much|money|most|least|total|cost|spent|raised|price|amount)\b",
    re.I,
)


def build_evidence_gate(
    *,
    question: str,
    rows: list[dict[str, Any]],
    contract: dict[str, Any],
    packet_metrics: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Emit short risk fields; never retrieve, score gold, or decide an answer."""
    shape = str(contract.get("answer_shape") or "direct_fact")
    states = dict(packet_metrics.get("state_counts") or {})
    candidate_count = len(rows)
    unique_refs = int(packet_metrics.get("unique_source_refs") or 0)
    reasons: list[str] = []

    if _TEMPORAL_SCOPE.search(question) and not contract.get("temporal", {}).get("kind"):
        reasons.append("temporal_scope_requires_explicit_date_check")
    if shape in {"count", "list", "ordered_list"} and states.get("completed", 0) < 2:
        reasons.append("enumeration_may_be_incomplete")
    if states.get("planned", 0) or states.get("cancelled_or_negated", 0):
        reasons.append("mixed_event_states_require_filtering")
    if candidate_count and unique_refs / candidate_count < 0.75:
        reasons.append("repeated_sources_require_deduplication")
    if _NUMERIC_COMPARISON.search(question):
        reasons.append("bind_each_number_to_entity_event_and_time_before_comparing")

    status = "review_required" if reasons else "sufficient_no_obvious_risk"
    rendered = (
        f'<evidence_gate status="{status}" '
        f'candidate_count="{candidate_count}" unique_source_refs="{unique_refs}">\n'
        f'risks={";".join(reasons) if reasons else "none"}\n'
        "This gate is diagnostic, not additional evidence. Resolve each listed "
        "risk from Core, Recent, and E01..En. If a required entity/event/value "
        "cannot be bound to supplied evidence, state that the answer is not established.\n"
        "</evidence_gate>"
    )
    return rendered, {
        "enabled": True,
        "status": status,
        "risks": reasons,
        "candidate_count": candidate_count,
        "unique_source_refs": unique_refs,
        "independent_llm_calls": 0,
        "supplemental_retrievals": 0,
    }
