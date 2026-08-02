import sys
from pathlib import Path
from types import ModuleType

import pytest

import src.retrieval.embedding as embedding
from src.retrieval.embedding import MemoryEmbeddingIndex
from src.management import MemoryWorkspace


def _write_events(memory_dir: Path, events: list[dict]) -> None:
    workspace = MemoryWorkspace(memory_dir)
    workspace.archive_sessions([{
        "observation_date": event["when"],
        "turns": [("user", event["summary"])],
        "refs": event["dia_ids"],
    } for event in events])
    workspace.save_memory([{
        "when": event["when"],
        "content": event["summary"],
        "refs": event["dia_ids"],
        "topic_path": f"{event['topic']}.md",
        "headings": event["topic"].split("/"),
    } for event in events])


class FakeEncoder:
    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            if text == "visual art":
                vectors.append([1.0, 0.0])
            elif "sunrise" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def test_embedding_search_ranks_topic_events_without_writing_files(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    _write_events(memory_dir, [
        {
            "when": "2023-05-07",
            "summary": "Caroline visited an LGBTQ support group",
            "dia_ids": ["D1:3"],
            "topic": "Caroline/support",
        },
        {
            "when": "2023-05-08",
            "summary": "Melanie painted a sunrise",
            "dia_ids": ["D1:4"],
            "topic": "Melanie/painting",
        },
    ])
    before = {
        path.relative_to(memory_dir).as_posix(): path.read_bytes()
        for path in memory_dir.rglob("*")
        if path.is_file()
    }

    results = MemoryEmbeddingIndex(memory_dir, encoder=FakeEncoder()).search(
        "visual art"
    )

    assert results[0]["refs"] == ["D1:4"]
    assert results[0]["similarity"] == pytest.approx(1.0)
    assert set(results[0]) == {
        "event",
        "path",
        "line",
        "date",
        "content",
        "refs",
        "similarity",
    }
    after = {
        path.relative_to(memory_dir).as_posix(): path.read_bytes()
        for path in memory_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_embedding_search_hard_caps_top_k_at_ten(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    topic = memory_dir / "topics/events.md"
    topic.parent.mkdir(parents=True)
    topic.write_text(
        "# Events\n\n"
        + "\n".join(
            f"[2023-05-{day:02d}] Event {day} [D1:{day}]"
            for day in range(1, 13)
        )
        + "\n",
        encoding="utf-8",
    )

    results = MemoryEmbeddingIndex(memory_dir, encoder=FakeEncoder()).search(
        "events", top_k=999
    )

    assert len(results) == 10


def test_embedding_search_filters_by_partial_temporal_window(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    _write_events(memory_dir, [
        {
            "when": "2023-05-08",
            "summary": "Melanie painted a sunrise in spring",
            "dia_ids": ["D1:4"],
            "topic": "Melanie/painting",
        },
        {
            "when": "2024-01-12",
            "summary": "Melanie painted another sunrise in winter",
            "dia_ids": ["D2:4"],
            "topic": "Melanie/painting",
        },
        {
            "when": "2023-09-20",
            "summary": "Melanie painted a sunrise in autumn",
            "dia_ids": ["D3:4"],
            "topic": "Melanie/painting",
        },
    ])

    results = MemoryEmbeddingIndex(memory_dir, encoder=FakeEncoder()).search(
        "visual art", date_from="2023", date_to="2023"
    )

    assert {tuple(row["refs"]) for row in results} == {("D1:4",), ("D3:4",)}


def test_embedding_index_reuses_document_vectors_for_query_rewrites(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    topic = memory_dir / "topics/art.md"
    topic.parent.mkdir(parents=True)
    topic.write_text(
        "# Art\n\n[2023-05-08] Melanie painted a sunrise [D1:4]\n",
        encoding="utf-8",
    )

    class CountingEncoder(FakeEncoder):
        def __init__(self):
            self.batch_sizes = []

        def encode(self, texts, **kwargs):
            self.batch_sizes.append(len(texts))
            return super().encode(texts, **kwargs)

    encoder = CountingEncoder()
    index = MemoryEmbeddingIndex(memory_dir, encoder=encoder)

    index.search("visual art")
    index.search("painting")

    assert encoder.batch_sizes == [1, 1, 1]


def test_default_encoder_is_loaded_only_when_search_needs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    memory_dir = tmp_path / "memory"
    topic = memory_dir / "topics/art.md"
    topic.parent.mkdir(parents=True)
    topic.write_text(
        "# Art\n\n[2023-05-08] Melanie painted a sunrise [D1:4]\n",
        encoding="utf-8",
    )
    loaded = []

    class StubSentenceTransformer(FakeEncoder):
        def __init__(self, model_name):
            loaded.append(model_name)

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = StubSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    index = MemoryEmbeddingIndex(memory_dir)
    assert loaded == []

    index.search("visual art")

    assert loaded == ["sentence-transformers/all-MiniLM-L6-v2"]


def test_render_search_results_keeps_location_score_and_references():
    rendered = embedding.render_search_results([
        {
            "event": "event-1",
            "path": "topics/art.md",
            "line": 7,
            "date": "2023-05-08",
            "content": "Melanie painted a sunrise.",
            "refs": ["D1:4"],
            "similarity": 0.875,
        }
    ])

    assert "topics/art.md:7" in rendered
    assert "similarity=0.8750" in rendered
    assert "Melanie painted a sunrise." in rendered
    assert "refs: D1:4" in rendered


def test_embedding_searches_source_turns_missing_from_topics(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    source = memory_dir / "sources/locomo/thread_1.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# thread_1\n\n"
        '<a id="source-abc"></a>\n'
        "<!-- source-id:locomo/thread_1/msg_1 -->\n"
        "[2023-10-04] Calvin: I painted a sunrise.\n",
        encoding="utf-8",
    )

    results = MemoryEmbeddingIndex(
        memory_dir, encoder=FakeEncoder()
    ).search("visual art")

    assert results[0]["path"] == "sources/locomo/thread_1.md"
    assert results[0]["refs"] == ["locomo/thread_1/msg_1"]
