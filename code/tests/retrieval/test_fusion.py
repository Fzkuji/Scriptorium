"""RRF fusion merges two ranked lists exactly as 1/(k + rank) predicts."""

from memory.retrieval.fusion import resolve_alpha, rrf_fuse

K = 60


def _row(event_id: str, path: str, line: int) -> dict:
    return {
        "event_id": event_id,
        "path": path,
        "line": line,
        "content": f"content {event_id}",
        "refs": ["D1:1"],
        "date": "2026-01-01",
        "dates": ["2026-01-01"],
    }


def test_rrf_fuse_orders_by_blended_rrf_including_single_list_rows():
    # Distinct files so the file-coherence boost applies uniformly (one row
    # per file: every best-row boost is proportional to its own score and
    # ordering is exactly the RRF blend).
    a = _row("a", "topics/a.md", 1)
    b = _row("b", "topics/b.md", 2)
    c = _row("c", "topics/c.md", 3)
    bm25 = [a, b]          # a rank 1, b rank 2
    embedding = [b, c]     # b rank 1, c rank 2; c is embedding-only

    alpha = 0.5
    expected = {
        "a": (1 - alpha) / (K + 1),
        "b": (1 - alpha) / (K + 2) + alpha / (K + 1),
        "c": alpha / (K + 2),
    }
    fused = rrf_fuse(bm25, embedding, alpha=alpha, top_k=8)

    order = [row["content"].split()[-1] for row in fused]
    assert order == sorted(expected, key=lambda k: -expected[k]) == ["b", "a", "c"]
    # embedding-only row survives and is tagged
    assert fused[2]["rule_features"] == ["embedding"]
    assert fused[0]["rule_features"] == ["bm25", "embedding"]


def test_rrf_fuse_empty_backend_keeps_other_ranking():
    rows = [_row("a", "topics/a.md", 1), _row("b", "topics/b.md", 2)]
    fused = rrf_fuse(rows, [], alpha=0.5, top_k=8)
    assert [row["path"] for row in fused] == ["topics/a.md", "topics/b.md"]
    fused = rrf_fuse([], rows, alpha=0.5, top_k=8)
    assert [row["path"] for row in fused] == ["topics/a.md", "topics/b.md"]


def test_file_coherence_boost_promotes_multi_row_file():
    # Both backends rank g first, then f's two rows. Without the boost g
    # stays on top; f's aggregate over two rows lifts f1 above g.
    f1, f2 = _row("f1", "topics/f.md", 1), _row("f2", "topics/f.md", 2)
    g = _row("g", "topics/g.md", 1)
    fused = rrf_fuse([g, f1, f2], [g, f1, f2], alpha=0.5, top_k=8)
    assert fused[0]["path"] == "topics/f.md"


def test_resolve_alpha_leans_bm25_for_literal_queries():
    assert resolve_alpha("what happened on 2026-05-12") == 0.3
    assert resolve_alpha('who said "exact phrase"') == 0.3
    assert resolve_alpha("did they meet Caroline yesterday") == 0.3
    assert resolve_alpha("how does the speaker feel about work") == 0.5
