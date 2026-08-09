"""`nearest` is the deterministic seed every retrieval starts from: one
tool call, no model, nothing when the workspace has nothing to find.
"""

from memory.retrieval import tools as retrieval_tools
from memory.retrieval.embedding import MemoryEmbeddingIndex
from memory.retrieval.search import nearest


class _KeywordEncoder:
    """Stands in for SentenceTransformer so no weights load in a test."""

    def encode(self, texts, **_kwargs):
        return [
            [1.0, 0.0] if "saxophone" in text.lower() else [0.0, 1.0]
            for text in texts
        ]


def _stub_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(
        retrieval_tools, "MemoryEmbeddingIndex",
        lambda root, *, files: MemoryEmbeddingIndex(
            root, files=files, encoder=_KeywordEncoder()
        ),
    )


def test_nearest_returns_a_passage_for_a_matching_query(tmp_path, monkeypatch):
    _stub_embeddings(monkeypatch)
    from memory.retrieval import embedding

    def _forbidden():
        raise AssertionError("nearest() must never load a real encoder")

    # Proves "no model call" even though the fused search runs the embedding
    # backend too: the only encoder it can reach is the stub above.
    monkeypatch.setattr(embedding, "_shared_encoder", _forbidden)

    topic = tmp_path / "topics/music.md"
    topic.parent.mkdir(parents=True)
    topic.write_text(
        "# Music\n\n[2023-05-08] Calvin plays saxophone in a jazz quartet. [D1:1]\n",
        encoding="utf-8",
    )

    result = nearest(tmp_path, "saxophone jazz")

    assert len(result) == 1
    assert "saxophone" in result[0].lower()


def test_nearest_returns_nothing_for_a_workspace_with_no_memory(tmp_path):
    # Empty workspace: both indexes have zero events, so this never even
    # reaches the embedding backend's lazy encoder.
    assert nearest(tmp_path, "saxophone jazz") == []
