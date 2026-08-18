"""General first-pass recall for adaptive reflective retrieval experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .bm25 import MemoryBM25Index
from .embedding import MemoryEmbeddingIndex
from .pipeline import (
    _expand_source_block,
    _lexical_event_search,
    _relation_neighbors,
    parse_timeline_events,
    reciprocal_rank_fusion,
)


REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "supported_facts": {"type": "array", "items": {"type": "string"}},
        "conflicts": {"type": "array", "items": {"type": "string"}},
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "can_answer": {"type": "boolean"},
        "follow_up_queries": {"type": "array", "items": {"type": "string"}},
        "answer": {"type": ["string", "null"]},
    },
    "required": [
        "supported_facts", "conflicts", "missing_information", "can_answer",
        "follow_up_queries", "answer",
    ],
    "additionalProperties": False,
}


def build_general_first_recall(
    runtime: Any,
    *,
    memory_dir: Path,
    files: list[Path],
    query: str,
    lane_top_k: int = 10,
    output_top_k: int = 16,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Recall broadly from every indexed view without classifying the question.

    The raw question is sent unchanged to each lane.  No answer-shape,
    predicate, temporal, or benchmark-specific routing rule is applied here.
    """
    relative_files = tuple(sorted(
        path.relative_to(memory_dir).as_posix()
        for path in files
        if path.suffix == ".md"
        and path.relative_to(memory_dir).parts[0] in {"topics", "sources"}
    ))
    getter = getattr(runtime, "get_retrieval_index", None)

    def cached(kind: str, factory: Any) -> Any:
        key = (kind, str(memory_dir), relative_files)
        return getter(key, factory) if callable(getter) else factory()

    bm25 = cached(
        "bm25", lambda: MemoryBM25Index(memory_dir, persist=False, files=files)
    )
    embedding = cached(
        "embedding", lambda: MemoryEmbeddingIndex(memory_dir, files=files)
    )
    lexical = bm25.search(query, top_k=lane_top_k)
    semantic = embedding.search_broad(query, top_k=lane_top_k)
    timeline_events = parse_timeline_events(memory_dir)
    timeline = _lexical_event_search(
        timeline_events, query, top_k=lane_top_k
    )
    relation = _relation_neighbors(
        memory_dir, [*lexical, *semantic], list(getattr(bm25, "events", []))
    )[:lane_top_k]
    rows = reciprocal_rank_fusion([lexical, semantic, timeline, relation])
    rows = [
        _expand_source_block(memory_dir, row)
        for row in rows[:output_top_k]
    ]
    blocks: list[str] = []
    for rank, row in enumerate(rows, start=1):
        row["first_recall_rank"] = rank
        blocks.append(
            f"[R{rank}] {row['path']}:{row['line']} "
            f"(date={row['date'] or 'unknown'}; view={row.get('view', 'topics_sources')})\n"
            f"{row['content']}"
        )
    rendered = (
        "<general_first_recall>\n"
        + ("\n\n".join(blocks) or "(no matches)")
        + "\n</general_first_recall>"
    )
    metrics = {
        "enabled": True,
        "version": "h-lite-u-20260817",
        "query_transform": "none_raw_question",
        "routing": "none_all_lanes",
        "lane_counts": {
            "lexical": len(lexical),
            "semantic": len(semantic),
            "timeline": len(timeline),
            "relations": len(relation),
        },
        "candidate_count": len(rows),
        "candidates": [
            {
                "rank": row["first_recall_rank"],
                "event_id": row["event_id"],
                "path": row["path"],
                "line": row["line"],
                "ranks": row["ranks"],
            }
            for row in rows
        ],
    }
    return rendered, rows, metrics


def retrieve_follow_up_queries(
    runtime: Any,
    *,
    memory_dir: Path,
    files: list[Path],
    queries: list[str],
    existing_rows: list[dict[str, Any]],
    lane_top_k: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute LLM-proposed generic queries across the same recall lanes."""
    added: list[dict[str, Any]] = []
    seen = {row["event_id"] for row in existing_rows}
    per_query: list[dict[str, Any]] = []
    for query in dict.fromkeys(q.strip() for q in queries if q.strip()):
        _text, rows, metrics = build_general_first_recall(
            runtime,
            memory_dir=memory_dir,
            files=files,
            query=query,
            lane_top_k=lane_top_k,
            output_top_k=lane_top_k,
        )
        fresh = [row for row in rows if row["event_id"] not in seen]
        for row in fresh:
            seen.add(row["event_id"])
            added.append(row)
        per_query.append({
            "query": query,
            "candidate_count": len(rows),
            "added_count": len(fresh),
            "lane_counts": metrics["lane_counts"],
        })
    return added, {
        "queries": per_query,
        "added_count": len(added),
        "added_event_ids": [row["event_id"] for row in added],
    }


def render_reflective_evidence(rows: list[dict[str, Any]]) -> str:
    blocks = []
    for rank, row in enumerate(rows, start=1):
        blocks.append(
            f"[E{rank}] {row['path']}:{row['line']} "
            f"(date={row.get('date') or 'unknown'})\n{row['content']}"
        )
    return "<retrieved_evidence>\n" + ("\n\n".join(blocks) or "(none)") + "\n</retrieved_evidence>"
