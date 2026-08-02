"""Semantic reconciliation and materialization after free-form Topic edits."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .reconciliation import (
    ReconciliationError,
    apply_reconciliation,
    classify_topic_diff,
)
from ..markdown import parse_topic_tree, render_definition, topic_prose


class TopicReconciliationMixin:
    def _reconcile_topic_edit(
        self,
        before_units: list[Any],
        before_prose: str,
        *,
        allow_correction: bool = False,
    ) -> None:
        after_units = parse_topic_tree(self.stage_dir / "topics", strict=False)
        if any(unit.evidence for unit in after_units):
            return
        diff = classify_topic_diff(before_units, after_units)
        prose_changed = topic_prose(self.stage_dir / "topics") != before_prose
        if (
            not diff.reconciler_required and not prose_changed
        ) or self.reconciler is None:
            return
        affected = set(diff.changed_ids) | set(diff.removed_ids)
        old_units = [
            unit for unit in before_units if unit.memory_id in affected
        ]
        edited_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((self.stage_dir / "topics").rglob("*.md"))
        )
        candidate_sources = {
            ref
            for unit in before_units + after_units
            for ref in unit.source_refs
        }
        for source in (self.stage_dir / "sources").glob("D*.md"):
            for conversation, turn in re.findall(
                r'<a id="d(\d+)-(\d+)"></a>',
                source.read_text(encoding="utf-8"),
            ):
                candidate_sources.add(f"D{conversation}:{turn}")
        for source in (self.stage_dir / "sources").rglob("*.md"):
            candidate_sources.update(re.findall(
                r"<!-- source-id:([^>]+) -->",
                source.read_text(encoding="utf-8"),
            ))
        for attempt in range(2):
            result = self.reconciler(
                edited_text, old_units, candidate_sources
            )
            try:
                apply_reconciliation(
                    edited_text,
                    old_units,
                    result,
                    candidate_sources=candidate_sources,
                    allow_correction=allow_correction,
                )
                break
            except ReconciliationError:
                if attempt == 1:
                    raise
        self._materialize_reconciliation(result, old_units)

    def _materialize_reconciliation(
        self, result: Any, old_units: list[Any]
    ) -> None:
        topics = self.stage_dir / "topics"
        old_by_id = {unit.memory_id: unit for unit in old_units}
        affected = set(result.matches) | set(result.deleted_ids)
        if affected:
            citation = re.compile(
                r"\[\^(" + "|".join(
                    re.escape(value) for value in affected
                ) + r")\]"
            )
            definition = re.compile(
                r"^\[\^(" + "|".join(
                    re.escape(value) for value in affected
                ) + r")\]:"
            )
        else:
            citation = None
            definition = None
        for path in topics.rglob("*.md"):
            rendered = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if definition and definition.match(line):
                    continue
                rendered.append(citation.sub("", line) if citation else line)
            path.write_text(
                "\n".join(rendered).rstrip() + "\n", encoding="utf-8"
            )

        placements: dict[
            str, list[tuple[str, str | None, tuple[str, ...]]]
        ] = {}
        for memory_id, content in result.matches.items():
            unit = old_by_id[memory_id]
            placements.setdefault(content, []).append(
                (memory_id, unit.when, unit.source_refs)
            )
        for content, when, refs in result.creates:
            payload = json.dumps(
                [when, content, list(refs)],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            memory_id = "mem_" + hashlib.sha256(
                payload.encode()
            ).hexdigest()[:16]
            placements.setdefault(content, []).append(
                (memory_id, when, refs)
            )

        definitions: dict[Path, list[str]] = {}
        for content, rows in placements.items():
            candidates = []
            for path in topics.rglob("*.md"):
                body = "\n".join(
                    line
                    for line in path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if not line.startswith("[^mem_")
                )
                if body.count(content) == 1:
                    candidates.append(path)
            if len(candidates) != 1:
                raise ValueError(
                    f"reconciled quote must occur in one Topic: {content!r}"
                )
            path = candidates[0]
            text = path.read_text(encoding="utf-8")
            citations = "".join(
                f"[^{memory_id}]" for memory_id, _when, _refs in rows
            )
            path.write_text(
                text.replace(content, content + citations, 1),
                encoding="utf-8",
            )
            relative_topic = Path("topics") / path.relative_to(topics)
            for memory_id, when, refs in rows:
                links = [
                    self._source_link(relative_topic, ref) for ref in refs
                ]
                definitions.setdefault(path, []).append(
                    render_definition(memory_id, when, links)
                )
        for path, rows in definitions.items():
            text = path.read_text(encoding="utf-8").rstrip()
            path.write_text(
                f"{text}\n\n" + "\n".join(rows) + "\n",
                encoding="utf-8",
            )
