"""BM25 + embedding retrieval merged with reciprocal rank fusion."""

from __future__ import annotations

import re
from typing import Any

from .bm25 import _normalize_path_prefix

_RRF_K = 60
_CANDIDATE_MULTIPLIER = 5
_FILE_COHERENCE_BOOST_FRAC = 0.2
_LITERAL_QUERY_RE = re.compile(
    r"\b\d{4}(?:-\d{2}(?:-\d{2})?)?\b"
    r"|[\"'“‘].+?[\"'”’]"
    r"|(?<=[a-z] )[A-Z][a-z]+"
)


def resolve_alpha(query: str) -> float:
    return 0.3 if _LITERAL_QUERY_RE.search(query) else 0.5


def _row_key(row: dict[str, Any]) -> tuple:
    return (row["path"], row["line"], row.get("event_id") or row.get("event"))


def _rrf(rows: list[dict[str, Any]]) -> dict[tuple, float]:
    return {
        _row_key(row): 1.0 / (_RRF_K + rank)
        for rank, row in enumerate(rows, 1)
    }


def rrf_fuse(
    bm25_rows: list[dict[str, Any]],
    embedding_rows: list[dict[str, Any]],
    *,
    alpha: float = 0.5,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    bm25_rrf = _rrf(bm25_rows)
    embedding_rrf = _rrf(embedding_rows)
    rows_by_key: dict[tuple, dict[str, Any]] = {}
    for row in embedding_rows + bm25_rows:
        rows_by_key.setdefault(_row_key(row), {}).update(row)
    scores = {
        key: alpha * embedding_rrf.get(key, 0.0)
        + (1.0 - alpha) * bm25_rrf.get(key, 0.0)
        for key in rows_by_key
    }
    max_score = max(scores.values(), default=0.0)
    if max_score:
        file_sum: dict[str, float] = {}
        best_key: dict[str, tuple] = {}
        for key, score in scores.items():
            path = key[0]
            file_sum[path] = file_sum.get(path, 0.0) + score
            if path not in best_key or score > scores[best_key[path]]:
                best_key[path] = key
        max_file_sum = max(file_sum.values())
        boost_unit = max_score * _FILE_COHERENCE_BOOST_FRAC
        for path, key in best_key.items():
            scores[key] += boost_unit * file_sum[path] / max_file_sum
    fused = []
    for key in sorted(scores, key=lambda k: (-scores[k], k[0], k[1])):
        row = rows_by_key[key]
        features = []
        if key in bm25_rrf:
            features.append("bm25")
        if key in embedding_rrf:
            features.append("embedding")
        fused.append({
            "path": row["path"],
            "line": row["line"],
            "content": row["content"],
            "refs": row.get("refs", []),
            "dates": row.get("dates")
            or ([row["date"]] if row.get("date") else []),
            "final_score": round(scores[key], 6),
            "rule_features": features,
        })
    return fused[:max(1, int(top_k))]


def fused_search(
    bm25_index: Any,
    embedding_index: Any,
    query: str,
    *,
    top_k: int = 8,
    path_prefix: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    candidate_count = top_k * _CANDIDATE_MULTIPLIER
    bm25_rows = bm25_index.search(
        query, top_k=candidate_count, path_prefix=path_prefix,
        date_from=date_from, date_to=date_to,
    )
    embedding_rows = embedding_index.search(
        query, top_k=candidate_count,
        date_from=date_from, date_to=date_to,
    )
    if path_prefix:
        prefix = _normalize_path_prefix(path_prefix)
        embedding_rows = [
            row for row in embedding_rows if row["path"].startswith(prefix)
        ]
    return rrf_fuse(
        bm25_rows, embedding_rows,
        alpha=resolve_alpha(query), top_k=top_k,
    )
