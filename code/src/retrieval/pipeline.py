"""Deterministic P0 retrieval pipeline for frozen-memory experiments."""

from __future__ import annotations

import hashlib
import json
import re
from difflib import get_close_matches
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Plus

from .bm25 import (
    MemoryBM25Index, MemoryEvent, event_matches_time_window, tokenize,
)
from .embedding import MemoryEmbeddingIndex


@dataclass(frozen=True)
class PipelineConfig:
    """Frozen low-level settings for the first P0 implementation."""

    bm25_top_k: int = 10
    embedding_top_k: int = 10
    output_top_k: int = 12
    rrf_k: int = 60
    max_per_path: int = 3
    source_backlinks: int = 4
    answer_max_turns: int = 3
    version: str = "p0-v2"
    multi_query: bool = False
    view_routing: bool = False
    timeline_quota: int = 0
    mmr_lambda: float = 0.7
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        for name in (
            "bm25_top_k", "embedding_top_k", "output_top_k", "rrf_k",
            "max_per_path", "answer_max_turns",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.source_backlinks < 0:
            raise ValueError("source_backlinks must not be negative")
        if self.timeline_quota < 0:
            raise ValueError("timeline_quota must not be negative")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between zero and one")
        if self.version not in {"p0-v2", "p0-v3", "p0b", "p0b-r1", "m3"}:
            raise ValueError("unsupported pipeline version")

    @classmethod
    def for_version(cls, version: str) -> "PipelineConfig":
        if version == "p0-v2":
            return cls()
        if version == "p0-v3":
            return cls(
                bm25_top_k=30,
                embedding_top_k=30,
                output_top_k=16,
                max_per_path=4,
                source_backlinks=6,
                version="p0-v3",
                multi_query=True,
            )
        if version == "p0b":
            return cls(
                bm25_top_k=30,
                embedding_top_k=30,
                output_top_k=16,
                max_per_path=4,
                source_backlinks=6,
                version="p0b",
                multi_query=True,
                view_routing=True,
            )
        if version == "p0b-r1":
            return cls(
                bm25_top_k=30,
                embedding_top_k=30,
                output_top_k=16,
                max_per_path=4,
                source_backlinks=6,
                version="p0b-r1",
                multi_query=True,
                view_routing=True,
                timeline_quota=4,
            )
        if version == "m3":
            return cls(
                bm25_top_k=30,
                embedding_top_k=30,
                output_top_k=16,
                max_per_path=16,
                source_backlinks=0,
                version="m3",
                embedding_model="BAAI/bge-m3",
            )
        raise ValueError(f"unsupported pipeline version: {version}")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _event_key(row: dict[str, Any]) -> str:
    return str(row.get("event_id") or row.get("event") or "")


def _normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "event_id": _event_key(row),
        "path": str(row.get("path", "")),
        "line": int(row.get("line", 0)),
        "date": str(row.get("date", "")),
        "dates": list(row.get("dates") or ([row["date"]] if row.get("date") else [])),
        "content": str(row.get("content", "")).strip(),
        "refs": list(dict.fromkeys(str(ref) for ref in row.get("refs", []))),
    }
    for key in ("view", "structural"):
        if row.get(key):
            normalized[key] = row[key]
    return normalized


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]], *, rrf_k: int = 60
) -> list[dict[str, Any]]:
    """Fuse ranked event lists with stable tie-breaking and provenance."""
    fused: dict[str, dict[str, Any]] = {}
    for list_index, rows in enumerate(ranked_lists):
        for rank, raw in enumerate(rows, start=1):
            row = _normalized_row(raw)
            key = row["event_id"] or f"{row['path']}:{row['line']}"
            entry = fused.setdefault(key, {**row, "rrf_score": 0.0, "ranks": {}})
            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            entry["ranks"][str(list_index)] = rank
    return sorted(
        fused.values(),
        key=lambda row: (
            -row["rrf_score"], row["path"], row["line"], row["event_id"]
        ),
    )


def m3_hybrid_mmr(
    lexical: list[dict[str, Any]],
    semantic: list[dict[str, Any]],
    vectors_by_event: dict[str, Any],
    *,
    limit: int,
    mmr_lambda: float = 0.7,
) -> list[dict[str, Any]]:
    """M3-style 0.7 dense/0.3 BM25 fusion followed by greedy MMR.

    This intentionally reproduces the public retrieval primitive, not M3's
    ingestion, category knobs, temporal expansions, or answer model.
    """
    import numpy as np

    lexical_ranks = {
        _event_key(row): rank for rank, row in enumerate(lexical, start=1)
    }
    semantic_ranks = {
        _event_key(row): rank for rank, row in enumerate(semantic, start=1)
    }
    semantic_by_id = {_event_key(row): row for row in semantic}
    candidates = list(dict.fromkeys([
        *(_event_key(row) for row in semantic),
        *(_event_key(row) for row in lexical),
    ]))
    lexical_size = max(1, len(lexical))
    rows: list[dict[str, Any]] = []
    for event_id in candidates:
        source = semantic_by_id.get(event_id) or next(
            row for row in lexical if _event_key(row) == event_id
        )
        dense = float(source.get("similarity", 0.0))
        rank = lexical_ranks.get(event_id)
        # SQLite FTS5 exposes negative BM25 values. Its documented M3
        # normalization is 1/(1+abs(score)); rank is the stable adapter for
        # our BM25Plus index, whose raw scale is not FTS5-compatible.
        lexical_norm = 0.0 if rank is None else 1.0 / (1.0 + (rank - 1) / lexical_size)
        rows.append({
            **_normalized_row(source),
            "dense_score": dense,
            "lexical_rank": rank,
            "m3_relevance": 0.7 * dense + 0.3 * lexical_norm,
            # Preserve the common audit schema while naming the actual score.
            "rrf_score": 0.7 * dense + 0.3 * lexical_norm,
            "ranks": {
                **({"lexical": rank} if rank is not None else {}),
                **({"semantic": semantic_ranks[event_id]} if event_id in semantic_ranks else {}),
            },
        })
    rows.sort(key=lambda row: (-row["m3_relevance"], row["path"], row["line"]))
    rows = rows[: max(limit, limit * 3)]
    dimension = max(
        (len(vector) for vector in vectors_by_event.values()), default=1
    )
    matrix = np.asarray([
        vectors_by_event.get(row["event_id"], np.zeros(dimension, dtype=float))
        for row in rows
    ], dtype=float)
    if not rows:
        return []
    norms = np.linalg.norm(matrix, axis=1)
    matrix = np.divide(matrix, norms[:, None], out=np.zeros_like(matrix), where=norms[:, None] != 0)
    selected = [0]
    remaining = list(range(1, len(rows)))
    while remaining and len(selected) < limit:
        def score(index: int) -> tuple[float, int]:
            max_similarity = max(float(matrix[index] @ matrix[prior]) for prior in selected)
            value = mmr_lambda * rows[index]["m3_relevance"] - (1.0 - mmr_lambda) * max_similarity
            return value, -index
        best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)
    return [
        {**rows[index], "pipeline_rank": rank, "retrieval_variant": "m3"}
        for rank, index in enumerate(selected, start=1)
    ]


_QUESTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "the",
    "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "with", "would",
}

_TIMELINE_META_WORDS = {
    "ago", "attended", "bake", "baked", "during", "many", "month",
    "months", "past", "something", "time", "times", "week", "weeks",
    "year", "years",
}

_TIMELINE_TOPIC_RE = re.compile(r"\[[^]]+\]\([^)]*topics/[^)#]+\.md#\^([A-Za-z0-9-]+)\)")
_TIMELINE_SOURCE_RE = re.compile(r"\[([^]]+)\]\([^)]*sources/[^)]+\)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^]]+)\]\([^)]+\)")

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
}


def temporal_constraint(question: str, question_date: str | None) -> dict[str, Any]:
    """Normalize common relative windows without domain- or item-specific rules."""
    folded = " ".join(str(question).split()).casefold()
    try:
        anchor = date.fromisoformat(str(question_date or "")[:10])
    except ValueError:
        return {"date_from": None, "date_to": None, "kind": None, "anchor": None}

    number = r"(?P<n>\d+|" + "|".join(_NUMBER_WORDS) + r")"
    match = re.search(rf"\b{number}\s+(?P<unit>day|week|month|year)s?\s+ago\b", folded)
    if match:
        raw = match.group("n")
        amount = int(raw) if raw.isdigit() else _NUMBER_WORDS[raw]
        days = amount * {"day": 1, "week": 7, "month": 30, "year": 365}[match.group("unit")]
        target = anchor - timedelta(days=days)
        return {
            "date_from": (target - timedelta(days=3)).isoformat(),
            "date_to": (target + timedelta(days=3)).isoformat(),
            "kind": "relative_point",
            "anchor": anchor.isoformat(),
        }
    match = re.search(rf"\blast\s+{number}\s+(?P<unit>day|week|month|year)s?\b", folded)
    if match:
        raw = match.group("n")
        amount = int(raw) if raw.isdigit() else _NUMBER_WORDS[raw]
        days = amount * {"day": 1, "week": 7, "month": 30, "year": 365}[match.group("unit")]
        return {
            "date_from": (anchor - timedelta(days=days)).isoformat(),
            "date_to": anchor.isoformat(),
            "kind": "relative_range",
            "anchor": anchor.isoformat(),
        }
    if re.search(r"\blast week\b", folded):
        return {
            "date_from": (anchor - timedelta(days=7)).isoformat(),
            "date_to": anchor.isoformat(),
            "kind": "relative_range",
            "anchor": anchor.isoformat(),
        }
    return {"date_from": None, "date_to": None, "kind": None, "anchor": anchor.isoformat()}


def question_contract(question: str, question_date: str | None = None) -> dict[str, Any]:
    """Build a deterministic retrieval contract, not an answer or tool plan."""
    folded = " ".join(str(question).split()).casefold()
    words = re.findall(r"[^\W_]+", folded)
    terms = list(dict.fromkeys(
        word for word in words if len(word) > 2 and word not in _QUESTION_WORDS
    ))
    if re.search(r"\bhow many\b|\bnumber of\b|\bhow much total\b|\btotal (?:money|amount|cost)\b", folded):
        shape = "count"
    elif re.search(r"\b(list|which|what (?:are|were))\b", folded):
        shape = "list"
    elif re.search(r"\b(before|after|first|last|latest|most recent|order)\b", folded):
        shape = "ordered_list"
    elif re.search(r"\b(prefer|preference|recommend|suggest|tips|advice)\b", folded):
        shape = "preference"
    elif re.search(r"\b(did|does|is|was|were)\b", folded):
        shape = "relation_check"
    else:
        shape = "direct_fact"

    expansions = {
        "count": "completed events distinct items total count amount cost paid hours",
        "list": "all distinct examples history completed",
        "ordered_list": "timeline chronology completed booked planned cancelled latest",
        "preference": "preferences likes dislikes interests plans style favorite",
        "relation_check": "role relationship exact source evidence",
        "direct_fact": "fact exact value current latest",
    }
    compact = " ".join(terms[:12])
    normalized_terms = [
        word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word
        for word in terms[:12]
    ]
    normalized = " ".join(normalized_terms)
    queries = list(dict.fromkeys(filter(None, (
        str(question).strip(), compact, normalized,
        f"{compact} {expansions[shape]}".strip(),
    ))))
    temporal = temporal_constraint(question, question_date)
    return {
        "answer_shape": shape,
        "target_terms": terms[:12],
        "queries": queries,
        "state_terms": (
            ["completed", "booked", "planned", "cancelled"]
            if shape in {"list", "ordered_list", "count"} else []
        ),
        "temporal": temporal,
        "coverage_axis": (
            "distinct_events"
            if shape in {"count", "list", "ordered_list"}
            else "best_supported_fact"
        ),
        "relation_requested": bool(re.search(
            r"\b(relationship|related|role|manager|supervisor|reports? to|team|colleague|friend|family)\b",
            folded,
        )),
    }


def _memory_vocabulary(events: list[Any]) -> set[str]:
    vocabulary: set[str] = set()
    for event in events:
        text = f"{getattr(event, 'path', '')} {' '.join(getattr(event, 'headings', []))}"
        vocabulary.update(word.casefold() for word in re.findall(r"[^\W_]+", text) if len(word) >= 5)
    return vocabulary


def _fuzzy_query(query: str, events: list[Any]) -> str | None:
    """Correct likely typos only against headings/path terms in this memory."""
    vocabulary = _memory_vocabulary(events)
    if not vocabulary:
        return None
    changed = False
    words = re.findall(r"[^\W_]+|[^\w]+", query)
    corrected: list[str] = []
    for word in words:
        folded = word.casefold()
        if len(folded) < 5 or folded in vocabulary or not folded.isalpha():
            corrected.append(word)
            continue
        match = get_close_matches(folded, vocabulary, n=1, cutoff=0.84)
        if match:
            corrected.append(match[0])
            changed = True
        else:
            corrected.append(word)
    return "".join(corrected) if changed else None


def _timeline_date(path: Path, timeline_root: Path) -> str:
    parts = path.relative_to(timeline_root).with_suffix("").parts
    if not parts or not all(part.isdigit() for part in parts):
        return ""
    return "-".join(
        part.zfill(2) if index else part
        for index, part in enumerate(parts)
    )


def parse_timeline_events(memory_dir: Path) -> list[MemoryEvent]:
    """Adapt built Timeline paragraphs to the common evidence record."""
    timeline_root = memory_dir / "timeline"
    if not timeline_root.is_dir():
        return []
    events: list[MemoryEvent] = []
    for path in sorted(timeline_root.rglob("*.md")):
        date_value = _timeline_date(path, timeline_root)
        lines = path.read_text(encoding="utf-8").splitlines()
        start = 0
        while start < len(lines):
            while start < len(lines) and (not lines[start].strip() or lines[start].startswith("#")):
                start += 1
            end = start
            while end < len(lines) and lines[end].strip():
                end += 1
            if start == end:
                continue
            raw = "\n".join(lines[start:end]).strip()
            topic_ids = _TIMELINE_TOPIC_RE.findall(raw)
            refs = list(dict.fromkeys(_TIMELINE_SOURCE_RE.findall(raw)))
            content = " ".join(_MARKDOWN_LINK_RE.sub(r"\1", raw).split())
            if content and (topic_ids or refs):
                stable = topic_ids[0] if topic_ids else hashlib.sha256(raw.encode()).hexdigest()[:12]
                events.append(MemoryEvent(
                    event_id=f"timeline:{stable}",
                    path="timeline/" + path.relative_to(timeline_root).as_posix(),
                    line=start + 1,
                    headings=[date_value] if date_value else [],
                    date=date_value,
                    dates=[date_value] if date_value else [],
                    content=(f"[{date_value}] {content}" if date_value else content),
                    refs=refs,
                ))
            start = end + 1
    return events


def _timeline_queries(question: str, contract: dict[str, Any]) -> list[str]:
    """Create compact event-view lanes without domain-specific vocabulary."""
    terms = [
        term for term in contract["target_terms"]
        if term not in _TIMELINE_META_WORDS
    ]
    focused = " ".join(terms)
    realized = (
        f"{focused} completed made did went attended visited bought finished"
        if focused else "completed made did went attended visited bought finished"
    )
    return list(dict.fromkeys(filter(None, (
        question.strip(), focused, realized,
    ))))


def _shares_evidence(row: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
    refs = set(row.get("refs") or [])
    for prior in selected:
        prior_refs = set(prior.get("refs") or [])
        if refs and prior_refs and refs.intersection(prior_refs):
            return True
        if not refs and not prior_refs and row["event_id"] == prior["event_id"]:
            return True
    return False


def _distinct_evidence(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if _shares_evidence(row, selected):
            continue
        selected.append(row)
        if len(selected) == limit:
            break
    return selected


def _lexical_event_search(
    events: list[MemoryEvent], query: str, *, top_k: int,
    date_from: str | None = None, date_to: str | None = None,
) -> list[dict[str, Any]]:
    candidates = [
        event for event in events
        if event_matches_time_window(event, date_from, date_to)
    ]
    query_tokens = tokenize(query)
    if not candidates or not query_tokens:
        return []
    corpus = [tokenize(f"{event.content} {' '.join(event.headings)}") for event in candidates]
    scores = BM25Plus(corpus).get_scores(query_tokens)
    ranked = sorted(zip(candidates, scores), key=lambda pair: (-pair[1], pair[0].path, pair[0].line))
    return [
        {**_normalized_row(event.__dict__), "view": "timeline", "lexical_score": float(score)}
        for event, score in ranked[:top_k]
    ]


def _relation_neighbors(memory_dir: Path, topic_rows: list[dict[str, Any]], events: list[MemoryEvent]) -> list[dict[str, Any]]:
    path = memory_dir / "relations.json"
    if not path.is_file():
        return []
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    event_map: dict[str, list[MemoryEvent]] = {}
    for event in events:
        if event.path.startswith("topics/"):
            event_map.setdefault(event.event_id.split(":", 1)[0], []).append(event)
    seed_ids = {row["event_id"].split(":", 1)[0] for row in topic_rows[:5]}
    neighbor_ids: list[str] = []
    for seed in seed_ids:
        neighbor_ids.extend(graph.get("outbound", {}).get(seed, []))
        neighbor_ids.extend(graph.get("backlinks", {}).get(seed, []))
    result: list[dict[str, Any]] = []
    for neighbor in dict.fromkeys(neighbor_ids):
        for event in event_map.get(neighbor, []):
            result.append({
                **_normalized_row(event.__dict__),
                "rrf_score": 0.0,
                "ranks": {},
                "view": "relations",
                "structural": "relation_neighbor",
            })
    return result


def _session_balanced(
    rows: list[dict[str, Any]], *, limit: int, max_per_path: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in rows:
        path = row["path"]
        if counts.get(path, 0) >= max_per_path:
            continue
        selected.append(row)
        counts[path] = counts.get(path, 0) + 1
        if len(selected) == limit:
            break
    return selected


def _coverage_balanced(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    max_per_path: int,
    answer_shape: str,
) -> list[dict[str, Any]]:
    """Select evidence from distinct sessions/events before taking duplicates."""
    if answer_shape not in {"count", "list", "ordered_list"}:
        return _session_balanced(rows, limit=limit, max_per_path=max_per_path)
    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    path_counts: dict[str, int] = {}
    seen_clusters: set[str] = set()
    for row in rows:
        refs = row.get("refs") or []
        cluster = str(refs[0]) if refs else f"{row['path']}:{row['date']}"
        if path_counts.get(row["path"], 0) >= min(2, max_per_path):
            deferred.append(row)
            continue
        if cluster in seen_clusters:
            deferred.append(row)
            continue
        selected.append(row)
        seen_clusters.add(cluster)
        path_counts[row["path"]] = path_counts.get(row["path"], 0) + 1
        if len(selected) == limit:
            return selected
    selected_ids = {row["event_id"] for row in selected}
    for row in deferred:
        if len(selected) == limit:
            break
        if row["event_id"] in selected_ids:
            continue
        if path_counts.get(row["path"], 0) >= max_per_path:
            continue
        selected.append(row)
        selected_ids.add(row["event_id"])
        path_counts[row["path"]] = path_counts.get(row["path"], 0) + 1
    return selected


def _structural_reserve(
    fused: list[dict[str, Any]],
    events: list[MemoryEvent],
    contract: dict[str, Any],
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Reserve related event records using the memory's existing hierarchy."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    topic_seeds = [row for row in fused if row["path"].startswith("topics/")][:5]
    temporal = contract["temporal"]
    if temporal["kind"] == "relative_point":
        seed_domains = {
            "/".join(row["path"].split("/")[:2]) for row in topic_seeds
        }
        for event in events:
            domain = "/".join(event.path.split("/")[:2])
            if not event.path.startswith("topics/") or domain not in seed_domains:
                continue
            if not event_matches_time_window(
                event, temporal["date_from"], temporal["date_to"]
            ):
                continue
            row = _normalized_row(event.__dict__)
            if row["event_id"] in seen:
                continue
            selected.append({**row, "rrf_score": 0.0, "ranks": {}, "structural": "dated_domain"})
            seen.add(row["event_id"])
            if len(selected) == limit:
                break
    return selected


def _source_rows(events: list[MemoryEvent]) -> dict[str, list[dict[str, Any]]]:
    by_ref: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if not event.path.startswith("sources/"):
            continue
        row = _normalized_row(event.__dict__)
        for ref in event.refs:
            by_ref.setdefault(ref, []).append(row)
    for rows in by_ref.values():
        rows.sort(key=lambda row: (row["path"], row["line"], row["event_id"]))
    return by_ref


def _expand_source_block(memory_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Replace a one-line source index hit with its complete archived message."""
    if not row["path"].startswith("sources/"):
        return row
    path = (memory_dir / row["path"]).resolve()
    try:
        path.relative_to(memory_dir.resolve())
    except ValueError:
        return row
    if not path.is_file() or path.is_symlink():
        return row
    lines = path.read_text(encoding="utf-8").splitlines()
    start = max(0, row["line"] - 1)
    if start >= len(lines):
        return row
    end = start + 1
    while end < len(lines):
        stripped = lines[end].lstrip()
        if stripped.startswith("<a id=") or stripped.startswith("<!-- source-id:"):
            break
        end += 1
    content = "\n".join(lines[start:end]).strip()
    return {**row, "content": content or row["content"], "source_block_lines": end - start}


def retrieve_pipeline(
    query: str,
    *,
    bm25: Any,
    embedding: Any,
    config: PipelineConfig | None = None,
    question_date: str | None = None,
    memory_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run hybrid recall, RRF, path balancing, and bounded source backlinking."""
    config = config or PipelineConfig()
    contract = question_contract(query, question_date)
    if config.version == "m3":
        lexical = bm25.search(query, top_k=config.bm25_top_k)
        semantic_search = (
            getattr(embedding, "search_broad")
            if hasattr(embedding, "search_broad") else embedding.search
        )
        semantic = semantic_search(query, top_k=config.embedding_top_k)
        union: dict[str, dict[str, Any]] = {}
        for row in [*semantic, *lexical]:
            union.setdefault(_event_key(row), row)
        ordered = list(union.values())
        matrix = embedding.vectors_for_rows(ordered)
        vectors = {
            _event_key(row): vector for row, vector in zip(ordered, matrix)
        }
        return m3_hybrid_mmr(
            lexical,
            semantic,
            vectors,
            limit=config.output_top_k,
            mmr_lambda=config.mmr_lambda,
        )
    queries = contract["queries"] if config.multi_query else [query]
    if config.multi_query:
        corrected = _fuzzy_query(query, list(getattr(bm25, "events", [])))
        if corrected:
            queries = list(dict.fromkeys([*queries, corrected]))
    ranked_lists: list[list[dict[str, Any]]] = []
    for lane in queries:
        ranked_lists.append(bm25.search(lane, top_k=config.bm25_top_k))
        semantic_search = (
            getattr(embedding, "search_broad")
            if config.multi_query and hasattr(embedding, "search_broad")
            else embedding.search
        )
        ranked_lists.append(semantic_search(lane, top_k=config.embedding_top_k))
    temporal = contract["temporal"]
    if config.multi_query and temporal["date_from"]:
        temporal_kwargs = {
            "top_k": config.bm25_top_k,
            "date_from": temporal["date_from"],
            "date_to": temporal["date_to"],
        }
        ranked_lists.append(bm25.search(query, **temporal_kwargs))
        ranked_lists.append(semantic_search(query, **temporal_kwargs))
    timeline_events: list[MemoryEvent] = []
    timeline_lists: list[list[dict[str, Any]]] = []
    timeline_enabled = bool(
        config.view_routing
        and memory_dir is not None
        and (
            contract["temporal"]["kind"]
            or contract["answer_shape"] in {"count", "list", "ordered_list"}
        )
    )
    if timeline_enabled:
        timeline_events = parse_timeline_events(memory_dir)
        timeline_lanes = (
            _timeline_queries(query, contract)
            if config.timeline_quota else queries[:3]
        )
        for lane in timeline_lanes:
            timeline_rows = _lexical_event_search(
                timeline_events,
                lane,
                top_k=min(
                    30 if config.timeline_quota else 12,
                    config.bm25_top_k,
                ),
                date_from=(temporal["date_from"] if temporal["kind"] else None),
                date_to=(temporal["date_to"] if temporal["kind"] else None),
            )
            timeline_lists.append(timeline_rows)
            if not config.timeline_quota:
                ranked_lists.append(timeline_rows)
    fused = reciprocal_rank_fusion(ranked_lists, rrf_k=config.rrf_k)
    base_limit = max(1, config.output_top_k - config.source_backlinks)
    timeline_reserve = (
        _distinct_evidence(
            reciprocal_rank_fusion(timeline_lists, rrf_k=config.rrf_k),
            limit=min(config.timeline_quota, base_limit),
        )
        if config.timeline_quota and timeline_lists else []
    )
    reserve = (
        _structural_reserve(
            fused, list(getattr(bm25, "events", [])), contract,
            limit=max(1, base_limit // 2),
        )
        if config.multi_query else []
    )
    if config.view_routing and memory_dir is not None and contract["relation_requested"]:
        relation_rows = _relation_neighbors(
            memory_dir,
            [row for row in fused if row["path"].startswith("topics/")],
            list(getattr(bm25, "events", [])),
        )
        known = {row["event_id"] for row in reserve}
        reserve.extend(row for row in relation_rows if row["event_id"] not in known)
        reserve = reserve[: max(1, base_limit // 2)]
    reserve = timeline_reserve + [
        row for row in reserve
        if not _shares_evidence(row, timeline_reserve)
    ]
    reserve = reserve[:base_limit]
    reserve_ids = {row["event_id"] for row in reserve}
    selected = reserve + _coverage_balanced(
        [
            row for row in fused
            if row["event_id"] not in reserve_ids
            and not _shares_evidence(row, reserve)
        ],
        limit=base_limit - len(reserve),
        max_per_path=config.max_per_path,
        answer_shape=contract["answer_shape"],
    )

    selected_keys = {row["event_id"] for row in selected}
    backlinks: list[dict[str, Any]] = []
    source_map = _source_rows(list(getattr(bm25, "events", [])))
    for row in selected:
        if len(backlinks) >= config.source_backlinks:
            break
        if row["path"].startswith("sources/") or (
            config.timeline_quota and row.get("view") == "timeline"
        ):
            continue
        for ref in row["refs"]:
            for source in source_map.get(ref, []):
                if source["event_id"] in selected_keys:
                    continue
                source = {**source, "rrf_score": 0.0, "ranks": {}, "backlink_for": row["event_id"]}
                backlinks.append(source)
                selected_keys.add(source["event_id"])
                break
            if len(backlinks) >= config.source_backlinks:
                break

    result = selected + backlinks
    counts: dict[str, int] = {}
    for row in result:
        counts[row["path"]] = counts.get(row["path"], 0) + 1
    for row in fused:
        if len(result) >= config.output_top_k:
            break
        if row["event_id"] in selected_keys or (
            config.timeline_quota and _shares_evidence(row, result)
        ):
            continue
        if counts.get(row["path"], 0) >= config.max_per_path:
            continue
        result.append(row)
        selected_keys.add(row["event_id"])
        counts[row["path"]] = counts.get(row["path"], 0) + 1
    result = result[: config.output_top_k]
    for rank, row in enumerate(result, start=1):
        row["pipeline_rank"] = rank
    return result


def build_pipeline_context(
    runtime: Any,
    *,
    memory_dir: Path,
    files: list[Path],
    query: str,
    question_date: str | None = None,
    config: PipelineConfig | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Build the model-visible P0 context and an auditable metrics record."""
    config = config or PipelineConfig()
    relative_files = tuple(sorted(
        path.relative_to(memory_dir).as_posix()
        for path in files
        if path.suffix == ".md"
        and path.relative_to(memory_dir).parts[0] in {"topics", "sources"}
    ))
    getter = getattr(runtime, "get_retrieval_index", None)

    def cached(kind: str, factory: Any) -> Any:
        return getter((kind, str(memory_dir), relative_files), factory) if callable(getter) else factory()

    bm25 = cached(
        "bm25",
        lambda: MemoryBM25Index(memory_dir, persist=False, files=files),
    )
    embedding = cached(
        f"embedding:{config.embedding_model}",
        lambda: MemoryEmbeddingIndex(
            memory_dir, files=files, model_name=config.embedding_model
        ),
    )
    rows = [
        _expand_source_block(memory_dir, row)
        for row in retrieve_pipeline(
            query,
            bm25=bm25,
            embedding=embedding,
            config=config,
            question_date=question_date,
            memory_dir=memory_dir,
        )
    ]
    blocks = []
    for row in rows:
        refs = ", ".join(row["refs"]) or "none"
        relation = f"; backlink_for={row['backlink_for']}" if row.get("backlink_for") else ""
        blocks.append(
            f"[{row['pipeline_rank']}] {row['path']}:{row['line']} "
            f"(date={row['date'] or 'unknown'}; refs={refs}{relation})\n"
            f"{row['content']}"
        )
    rendered = "<pipeline_context>\n" + ("\n\n".join(blocks) or "(no matches)") + "\n</pipeline_context>"
    metrics = {
        "enabled": True,
        "version": config.version,
        "config_sha256": config.fingerprint,
        "question_contract": question_contract(query, question_date),
        "views_used": sorted(
            ["topics_sources"]
            + (["timeline"] if config.view_routing and any(
                row["path"].startswith("timeline/") for row in rows
            ) else [])
            + (["relations"] if any(
                row.get("view") == "relations" for row in rows
            ) else [])
        ),
        "candidate_count": len(rows),
        "source_backlink_count": sum(bool(row.get("backlink_for")) for row in rows),
        "candidates": [
            {
                "rank": row["pipeline_rank"],
                "event_id": row["event_id"],
                "path": row["path"],
                "line": row["line"],
                "refs": row["refs"],
                "backlink_for": row.get("backlink_for"),
                "rrf_score": row["rrf_score"],
                "ranks": row["ranks"],
                "source_block_lines": row.get("source_block_lines"),
            }
            for row in rows
        ],
    }
    return rendered, rows, metrics


def supplement_pipeline_rows(
    runtime: Any,
    *,
    memory_dir: Path,
    files: list[Path],
    query: str,
    existing_rows: list[dict[str, Any]],
    question_date: str | None = None,
    max_new: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one bounded P3 hybrid lookup and return only new evidence.

    The lookup is deliberately narrower than another agent turn: one
    deterministic query, local BM25 plus the existing local embedding index,
    optional Timeline recall, and at most ``max_new`` appended candidates.
    """
    if max_new < 1:
        raise ValueError("max_new must be positive")
    contract = question_contract(query, question_date)
    focused = " ".join(contract["target_terms"])
    supplemental_query = (
        f"{focused} completed attended made did went bought finished distinct history"
    ).strip()
    relative_files = tuple(sorted(
        path.relative_to(memory_dir).as_posix()
        for path in files
        if path.suffix == ".md"
        and path.relative_to(memory_dir).parts[0] in {"topics", "sources"}
    ))
    getter = getattr(runtime, "get_retrieval_index", None)

    def cached(kind: str, factory: Any) -> Any:
        return getter((kind, str(memory_dir), relative_files), factory) if callable(getter) else factory()

    bm25 = cached("bm25", lambda: MemoryBM25Index(memory_dir, persist=False, files=files))
    embedding = cached("embedding", lambda: MemoryEmbeddingIndex(memory_dir, files=files))
    ranked = [
        bm25.search(supplemental_query, top_k=30),
        embedding.search_broad(supplemental_query, top_k=30),
    ]
    if contract["answer_shape"] in {"count", "list", "ordered_list"}:
        timeline = parse_timeline_events(memory_dir)
        ranked.append(_lexical_event_search(
            timeline, supplemental_query, top_k=30,
            date_from=contract["temporal"]["date_from"],
            date_to=contract["temporal"]["date_to"],
        ))
    fused = reciprocal_rank_fusion(ranked)
    existing_ids = {row["event_id"] for row in existing_rows}
    eligible = [
        row for row in fused
        if row["event_id"] not in existing_ids
        and not _shares_evidence(row, existing_rows)
    ]
    selected = _distinct_evidence(eligible, limit=max_new)
    selected = [_expand_source_block(memory_dir, row) for row in selected]
    start = len(existing_rows) + 1
    for offset, row in enumerate(selected):
        row["pipeline_rank"] = start + offset
        row["supplemental"] = True
    return selected, {
        "triggered": True,
        "lookup_count": 1,
        "query": supplemental_query,
        "max_new": max_new,
        "added_count": len(selected),
        "added_event_ids": [row["event_id"] for row in selected],
    }
