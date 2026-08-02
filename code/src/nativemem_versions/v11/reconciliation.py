"""Classify Topic edits and validate semantic reconciliation output."""

from __future__ import annotations

from dataclasses import dataclass

from .topic_markdown import MemoryUnit, is_valid_temporal_value


class ReconciliationError(ValueError):
    """Raised when reconciliation would break identity or provenance."""


@dataclass(frozen=True)
class TopicDiff:
    kind: str
    changed_ids: tuple[str, ...]
    added_ids: tuple[str, ...]
    removed_ids: tuple[str, ...]

    @property
    def reconciler_required(self) -> bool:
        return self.kind == "semantic"


@dataclass(frozen=True)
class ReconciliationResult:
    matches: dict[str, str]
    creates: tuple[tuple[str, str | None, tuple[str, ...]], ...]
    deleted_ids: tuple[str, ...]
    organizational_quotes: tuple[str, ...] = ()


def classify_topic_diff(before: list[MemoryUnit], after: list[MemoryUnit]) -> TopicDiff:
    old = {unit.memory_id: unit for unit in before}
    new = {unit.memory_id: unit for unit in after}
    added = tuple(sorted(set(new) - set(old)))
    removed = tuple(sorted(set(old) - set(new)))
    changed = tuple(sorted(
        memory_id for memory_id in set(old) & set(new)
        if (
            old[memory_id].content,
            old[memory_id].when,
            old[memory_id].source_refs,
            old[memory_id].source_links,
        ) != (
            new[memory_id].content,
            new[memory_id].when,
            new[memory_id].source_refs,
            new[memory_id].source_links,
        )
    ))
    if not (added or removed or changed):
        kind = "none" if before == after else "structural"
    else:
        kind = "semantic"
    return TopicDiff(kind, changed, added, removed)


def apply_reconciliation(
    edited_text: str,
    old_units: list[MemoryUnit],
    result: ReconciliationResult,
    *,
    candidate_sources: set[str],
    allow_correction: bool = False,
) -> ReconciliationResult:
    """Validate that reconciliation refers only to exact edited text and evidence."""
    old_ids = {unit.memory_id for unit in old_units}
    matched_ids = set(result.matches)
    deleted_ids = set(result.deleted_ids)
    unknown = (matched_ids | deleted_ids) - old_ids
    if unknown:
        raise ReconciliationError(f"unknown old memory_id: {sorted(unknown)[0]}")
    missing = old_ids - matched_ids - deleted_ids
    if missing:
        raise ReconciliationError(f"missing old memory_id: {sorted(missing)[0]}")
    if deleted_ids and not allow_correction:
        raise ReconciliationError("deletion requires explicit correction")
    quotes = (
        list(result.matches.values())
        + [row[0] for row in result.creates]
        + list(result.organizational_quotes)
    )
    for quote in quotes:
        if not quote or quote not in edited_text:
            raise ReconciliationError(f"exact quote not found: {quote!r}")
    for content, when, refs in result.creates:
        if when is not None and not is_valid_temporal_value(when):
            raise ReconciliationError(f"invalid date for {content!r}")
        if not refs:
            raise ReconciliationError(f"new memory requires source: {content!r}")
        invalid = set(refs) - candidate_sources
        if invalid:
            raise ReconciliationError(f"candidate source not available: {sorted(invalid)[0]}")
    return result
