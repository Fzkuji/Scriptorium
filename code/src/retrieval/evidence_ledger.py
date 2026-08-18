"""Deterministic observation ledger over canonical Scriptorium records."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .bm25 import MemoryEvent, parse_source_file, parse_topic_file


_SEARCH_BLOCK = re.compile(
    r"(?ms)^\d+\. (?P<path>[^:\n]+):(?P<line>\d+) "
    r"\[(?P<meta>[^\]]*)\]\n\s+(?P<content>.*?)\n\s+refs: "
    r"(?P<refs>.*?)(?=^\d+\. |\Z)"
)
_PATH_LINE = re.compile(
    r"(?m)^(?P<path>(?:topics|sources)/[^:\n]+):"
    r"(?P<line>\d+):(?P<content>.*)$"
)
_SPACE = re.compile(r"\s+")


def _normalized(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def _component(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else path.split(".", 1)[0]


def _content_hash(text: str) -> str:
    return hashlib.sha256(_normalized(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceObservation:
    method: str
    retrieval_round: int
    arguments: dict[str, Any]
    path: str
    line: int

    def audit(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "retrieval_round": self.retrieval_round,
            "arguments": self.arguments,
            "path": self.path,
            "line": self.line,
        }


@dataclass
class EvidenceEntry:
    evidence_id: str
    content_hash: str
    content: str
    component: str
    source_ref: str
    source_refs: list[str]
    event_time: str | None
    observations: list[EvidenceObservation] = field(default_factory=list)
    verified_by: list[str] = field(default_factory=list)
    retrieval_score: float | None = None
    revision: int = 1

    @property
    def retrieval_methods(self) -> list[str]:
        return list(dict.fromkeys(item.method for item in self.observations))

    @property
    def retrieval_rounds(self) -> list[int]:
        return list(dict.fromkeys(item.retrieval_round for item in self.observations))

    @property
    def verified_from_source(self) -> bool:
        return self.component == "sources" or bool(self.verified_by)

    def audit(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "content_hash": self.content_hash,
            "component": self.component,
            "source_ref": self.source_ref,
            "source_refs": list(self.source_refs),
            "event_time": self.event_time,
            "retrieval_score": self.retrieval_score,
            "verified_from_source": self.verified_from_source,
            "verified_by": list(self.verified_by),
            "observations": [item.audit() for item in self.observations],
        }


class EvidenceLedger:
    """Canonical evidence plus retrieval observations and delta presentation."""

    def __init__(
        self,
        *,
        memory_dir: Path | None = None,
        files: Iterable[Path] | None = None,
        max_entries: int = 24,
        excerpt_chars: int = 240,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if excerpt_chars < 40:
            raise ValueError("excerpt_chars must be at least 40")
        self.memory_dir = Path(memory_dir).resolve() if memory_dir else None
        self.max_entries = max_entries  # presentation limit, not storage
        self.excerpt_chars = excerpt_chars
        self._canonical = self._load_canonical(files or ())
        self._by_location = {
            (event.path, event.line): event for event in self._canonical
        }
        self._entries: dict[str, EvidenceEntry] = {}
        self._pending: list[str] = []
        self.observations = 0
        self.duplicates = 0
        self.rejected_outputs = 0
        self.initial_records = 0

    def _load_canonical(self, files: Iterable[Path]) -> list[MemoryEvent]:
        if self.memory_dir is None:
            return []
        events: list[MemoryEvent] = []
        topics = self.memory_dir / "topics"
        sources = self.memory_dir / "sources"
        for path in sorted(Path(item).resolve() for item in files):
            try:
                relative = path.relative_to(self.memory_dir)
                if relative.parts[0] == "topics" and path.suffix == ".md":
                    events.extend(parse_topic_file(path, topics))
                elif relative.parts[0] == "sources" and path.suffix == ".md":
                    events.extend(parse_source_file(path, sources))
            except (OSError, ValueError):
                continue
        return events

    @property
    def entries(self) -> list[EvidenceEntry]:
        return list(self._entries.values())

    def ingest_initial(self, *, core: str = "", recent: str = "") -> None:
        """Record initial visibility without duplicating giant evidence."""
        self.initial_records = int(bool(core.strip())) + int(bool(recent.strip()))

    def _resolve(
        self, path: str, line: int | None, content: str = ""
    ) -> MemoryEvent | None:
        if line is not None and (path, line) in self._by_location:
            return self._by_location[(path, line)]
        wanted = _normalized(content)
        for event in self._canonical:
            if event.path != path:
                continue
            actual = _normalized(event.content)
            if wanted and (
                wanted == actual or wanted in actual or actual in wanted
            ):
                return event
        return None

    def _observe(
        self,
        event: MemoryEvent,
        *,
        method: str,
        arguments: dict[str, Any],
        retrieval_round: int,
        score: float | None = None,
    ) -> None:
        self.observations += 1
        key = event.event_id
        entry = self._entries.get(key)
        observation = EvidenceObservation(
            method=method,
            retrieval_round=retrieval_round,
            arguments=dict(arguments),
            path=event.path,
            line=event.line,
        )
        if entry is None:
            entry = EvidenceEntry(
                evidence_id=event.event_id,
                content_hash=_content_hash(event.content),
                content=_normalized(event.content),
                component=_component(event.path),
                source_ref=f"{event.path}:{event.line}",
                source_refs=list(event.refs),
                event_time=event.date or None,
                retrieval_score=score,
            )
            self._entries[key] = entry
        else:
            self.duplicates += 1
            entry.revision += 1
            if score is not None and (
                entry.retrieval_score is None or score > entry.retrieval_score
            ):
                entry.retrieval_score = score
        entry.observations.append(observation)
        if key not in self._pending:
            self._pending.append(key)
        self._link_source_verification()

    def _link_source_verification(self) -> None:
        sources = [
            entry for entry in self._entries.values()
            if entry.component == "sources"
        ]
        for entry in self._entries.values():
            if entry.component == "sources":
                continue
            verified = [
                source.evidence_id
                for source in sources
                if set(entry.source_refs).intersection(source.source_refs)
            ]
            if verified != entry.verified_by:
                entry.verified_by = verified
                entry.revision += 1
                if entry.evidence_id not in self._pending:
                    self._pending.append(entry.evidence_id)

    def ingest_tool_output(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        output: str,
        retrieval_round: int,
    ) -> int:
        admitted = 0
        for match in _SEARCH_BLOCK.finditer(output):
            event = self._resolve(
                match.group("path"),
                int(match.group("line")),
                match.group("content"),
            )
            if event is None:
                continue
            score_match = re.search(
                r"(?:score|similarity)=([0-9.]+)", match.group("meta")
            )
            self._observe(
                event,
                method=tool_name,
                arguments=arguments,
                retrieval_round=retrieval_round,
                score=float(score_match.group(1)) if score_match else None,
            )
            admitted += 1

        if not admitted and tool_name == "read_memory_file":
            path = str(arguments.get("path") or "")
            normalized_output = _normalized(output)
            for event in self._canonical:
                if event.path != path:
                    continue
                content = _normalized(event.content)
                undated_content = re.sub(r"^\[[^]]+\]\s*", "", content)
                if content and (
                    content in normalized_output
                    or undated_content in normalized_output
                ):
                    self._observe(
                        event,
                        method=tool_name,
                        arguments=arguments,
                        retrieval_round=retrieval_round,
                    )
                    admitted += 1

        if not admitted and tool_name == "bash":
            for match in _PATH_LINE.finditer(output):
                event = self._resolve(
                    match.group("path"),
                    int(match.group("line")),
                    match.group("content"),
                )
                if event is None:
                    continue
                self._observe(
                    event,
                    method=tool_name,
                    arguments=arguments,
                    retrieval_round=retrieval_round,
                )
                admitted += 1

        if not admitted:
            self.rejected_outputs += 1
        return admitted

    def render_delta(self) -> str:
        keys = self._pending[: self.max_entries]
        self._pending = self._pending[self.max_entries :]
        if not keys:
            return ""
        rows = [
            "<evidence_ledger_delta>",
            "Program-validated canonical records newly observed or changed:",
        ]
        for key in keys:
            entry = self._entries[key]
            metadata = [
                f"id={entry.evidence_id}",
                f"component={entry.component}",
                f"source={entry.source_ref}",
                "rounds=" + ",".join(map(str, entry.retrieval_rounds)),
            ]
            if entry.event_time:
                metadata.append(f"time={entry.event_time}")
            if entry.source_refs:
                metadata.append("refs=" + ",".join(entry.source_refs[:4]))
            if entry.verified_by:
                metadata.append("verified_by=" + ",".join(entry.verified_by))
            excerpt = entry.content[: self.excerpt_chars]
            rows.append("- " + "; ".join(metadata) + f"\n  {excerpt}")
        rows.append("</evidence_ledger_delta>")
        return "\n".join(rows)

    def render(self) -> str:
        """Backward-compatible alias; rendering is intentionally delta-only."""
        return self.render_delta()

    def metrics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "entries": len(self._entries),
            "observations": self.observations,
            "duplicates": self.duplicates,
            "duplicate_ratio": (
                self.duplicates / self.observations if self.observations else 0.0
            ),
            "evictions": 0,
            "rejected_outputs": self.rejected_outputs,
            "initial_records": self.initial_records,
            "source_verified_entries": sum(
                entry.verified_from_source for entry in self._entries.values()
            ),
            "audit": [entry.audit() for entry in self._entries.values()],
        }
