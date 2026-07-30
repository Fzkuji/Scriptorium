"""Local event-level BM25 retrieval for NativeMem.

The index is deliberately lexical: it uses no embedding model or vector store.
Topic files are the canonical indexed view; timeline files are excluded because
they duplicate the same logical events.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Plus

_CACHE_NAME = ".nativemem-bm25.json"
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_REF_RE = re.compile(r"D\d+:\d+(?:-(?:D\d+:)?\d+)?")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_EVENT_RE = re.compile(r"<!--\s*memory-event:(ev_[0-9a-f]+)\s*-->")
_COMMENT_RE = re.compile(r"<!--.*?-->")
_MARKDOWN_LINK_RE = re.compile(r"\[([^]]+)\]\([^)]+\)")


@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    path: str
    line: int
    headings: list[str]
    date: str
    content: str
    refs: list[str]


def tokenize(text: str) -> list[str]:
    """Tokenize lexical text without language-model dependencies."""
    return [token.casefold() for token in _WORD_RE.findall(text)]


def _clean_markdown(text: str) -> str:
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _COMMENT_RE.sub(" ", text)
    return " ".join(text.split())


def _refs(text: str) -> list[str]:
    return list(dict.fromkeys(_REF_RE.findall(text)))


def _date(text: str) -> str:
    match = _DATE_RE.search(text)
    return match.group(0) if match else ""


def _stable_id(path: str, line: int, content: str, refs: list[str]) -> str:
    payload = json.dumps([path, line, content, refs], ensure_ascii=False)
    return "lex_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def parse_topic_file(path: Path, topics_root: Path) -> list[MemoryEvent]:
    """Parse both V8 article/event lines and V11 marker-based topic files."""
    relative = path.relative_to(topics_root).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    headings: list[str] = []
    events: list[MemoryEvent] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            headings = headings[: level - 1] + [heading.group(2).strip()]
            index += 1
            continue

        marker = _EVENT_RE.search(line)
        if marker:
            event_id = marker.group(1)
            block = [line]
            content_line = ""
            content_number = index + 1
            cursor = index + 1
            while cursor < len(lines):
                candidate = lines[cursor]
                if _HEADING_RE.match(candidate) or _EVENT_RE.search(candidate):
                    break
                block.append(candidate)
                if not content_line and _DATE_RE.search(candidate) and not candidate.lstrip().startswith("<!--"):
                    content_line = candidate
                    content_number = cursor + 1
                cursor += 1
            joined = "\n".join(block)
            refs = _refs(joined)
            content = _clean_markdown(content_line)
            if content and refs:
                events.append(MemoryEvent(
                    event_id=event_id,
                    path=f"topics/{relative}",
                    line=content_number,
                    headings=list(headings),
                    date=_date(content_line),
                    content=content,
                    refs=refs,
                ))
            index = cursor
            continue

        refs = _refs(line)
        content = _clean_markdown(line)
        if refs and content and not line.lstrip().startswith("<!--"):
            events.append(MemoryEvent(
                event_id=_stable_id(relative, index + 1, content, refs),
                path=f"topics/{relative}",
                line=index + 1,
                headings=list(headings),
                date=_date(line),
                content=content,
                refs=refs,
            ))
        index += 1

    return events


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MemoryBM25Index:
    """Incrementally parsed local topic index with in-memory BM25 scoring."""

    def __init__(self, memory_dir: str | Path):
        self.memory_dir = Path(memory_dir).resolve()
        self.topics_dir = self.memory_dir / "topics"
        self.cache_path = self.memory_dir / _CACHE_NAME
        self._files: dict[str, dict[str, Any]] = {}
        self.events: list[MemoryEvent] = []
        self._load_cache()
        self.refresh()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if payload.get("version") == 1 and isinstance(payload.get("files"), dict):
                self._files = payload["files"]
        except (OSError, ValueError, TypeError):
            self._files = {}

    def _write_cache(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "files": self._files}
        fd, temporary = tempfile.mkstemp(prefix=".nativemem-bm25-", dir=self.memory_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            os.replace(temporary, self.cache_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def refresh(self) -> None:
        current: dict[str, Path] = {}
        if self.topics_dir.exists():
            current = {
                path.relative_to(self.topics_dir).as_posix(): path
                for path in self.topics_dir.rglob("*.md")
            }

        changed = set(self._files) != set(current)
        refreshed: dict[str, dict[str, Any]] = {}
        for relative, path in sorted(current.items()):
            digest = _file_hash(path)
            cached = self._files.get(relative)
            if cached and cached.get("sha256") == digest:
                refreshed[relative] = cached
                continue
            changed = True
            refreshed[relative] = {
                "sha256": digest,
                "events": [asdict(event) for event in parse_topic_file(path, self.topics_dir)],
            }

        self._files = refreshed
        self.events = [
            MemoryEvent(**row)
            for relative in sorted(self._files)
            for row in self._files[relative].get("events", [])
        ]
        if changed:
            self._write_cache()

    @staticmethod
    def _search_text(event: MemoryEvent) -> str:
        path = event.path.removeprefix("topics/").replace("/", " ").replace("_", " ")
        headings = " ".join(event.headings)
        return f"{event.content} {path} {headings} {event.date}"

    @staticmethod
    def _rule_adjustment(event: MemoryEvent, query: str, tokens: list[str]) -> tuple[float, list[str]]:
        adjustment = 0.0
        reasons: list[str] = []
        content_tokens = set(tokenize(event.content))
        path_tokens = set(tokenize(event.path + " " + " ".join(event.headings)))
        significant = {token for token in tokens if len(token) >= 3}

        entity_hits = significant & content_tokens
        if entity_hits:
            bonus = min(0.18, 0.03 * len(entity_hits))
            adjustment += bonus
            reasons.append("exact_terms=" + ",".join(sorted(entity_hits)[:5]))

        path_hits = significant & path_tokens
        if path_hits:
            bonus = min(0.15, 0.05 * len(path_hits))
            adjustment += bonus
            reasons.append("path_heading=" + ",".join(sorted(path_hits)[:4]))

        query_dates = set(_DATE_RE.findall(query))
        if query_dates and event.date in query_dates:
            adjustment += 0.12
            reasons.append("exact_date")

        if event.refs:
            adjustment += 0.02
            reasons.append("source_grounded")
        return adjustment, reasons

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        path_prefix: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        self.refresh()
        query_tokens = tokenize(query)
        if not query_tokens or not self.events:
            return []

        candidates = []
        for event in self.events:
            if path_prefix:
                normalized = path_prefix.strip().removeprefix("topics/").strip("/")
                event_path = event.path.removeprefix("topics/")
                if not event_path.startswith(normalized):
                    continue
            if date_from and (not event.date or event.date < date_from):
                continue
            if date_to and (not event.date or event.date > date_to):
                continue
            candidates.append(event)
        if not candidates:
            return []

        corpus = [tokenize(self._search_text(event)) for event in candidates]
        scores = BM25Plus(corpus).get_scores(query_tokens)
        max_positive = max((float(score) for score in scores if score > 0), default=0.0)
        results = []
        for event, raw_score in zip(candidates, scores):
            raw = float(raw_score)
            lexical = raw / max_positive if max_positive else 0.0
            adjustment, reasons = self._rule_adjustment(event, query, query_tokens) if rerank else (0.0, [])
            final = lexical + adjustment
            if lexical <= 0 and adjustment <= 0.02:
                continue
            results.append({
                **asdict(event),
                "bm25_score": round(raw, 6),
                "lexical_score": round(lexical, 6),
                "rule_adjustment": round(adjustment, 6),
                "final_score": round(final, 6),
                "rule_features": reasons,
            })

        results.sort(key=lambda row: (-row["final_score"], row["path"], row["line"], row["event_id"]))
        return results[: max(1, min(int(top_k), 50))]


def render_search_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No BM25 matches."
    blocks = []
    for rank, row in enumerate(results, start=1):
        features = ", ".join(row["rule_features"]) or "none"
        blocks.append(
            f"{rank}. {row['path']}:{row['line']} "
            f"[date={row['date'] or 'unknown'}; score={row['final_score']:.4f}; rules={features}]\n"
            f"   {row['content']}\n"
            f"   refs: {', '.join(row['refs'])}"
        )
    return "\n".join(blocks)
