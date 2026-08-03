import json
from pathlib import Path

import pytest

from src.retrieval.bm25 import MemoryBM25Index, parse_topic_file, tokenize
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


def test_tokenize_is_lexical_and_unicode_aware():
    assert tokenize("LGBTQ-support 2023 水族箱") == [
        "lgbtq", "support", "2023",
        # The whole run is kept, then made substring-searchable.
        "水族箱", "水", "族", "箱", "水族", "族箱",
    ]


def test_tokenize_leaves_space_delimited_text_alone():
    assert tokenize("cost accounting 2026") == ["cost", "accounting", "2026"]


def test_tokenize_makes_cjk_substrings_searchable():
    """A CJK run is one word-boundary token, so substrings need their own."""
    tokens = tokenize("成本是怎么回事儿")

    assert "成本是怎么回事儿" in tokens
    assert "成本" in tokens
    assert "成" in tokens


def test_cjk_query_matches_a_longer_run(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    topic = memory_dir / "topics/notes.md"
    topic.parent.mkdir(parents=True)
    topic.write_text(
        "# Notes\n\n"
        "用户问成本是怎么回事儿。[^e-cost] ^blockcjk\n\n"
        "[^e-cost]: Time: `2026-08-03`; Sources: [D1:1](../sources/D1.md#d1-1)\n",
        encoding="utf-8",
    )

    result = MemoryBM25Index(memory_dir, persist=False).search("成本")

    assert result and result[0]["path"] == "topics/notes.md"


def test_index_parses_topics_without_timeline_duplicates(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    _write_events(memory_dir, [{
        "when": "2023-05-07",
        "summary": "Caroline visited an LGBTQ support group",
        "dia_ids": ["D1:3"],
        "topic": "Caroline/support",
    }])

    index = MemoryBM25Index(memory_dir)

    topic_events = [event for event in index.events if event.path.startswith("topics/")]
    assert len(topic_events) == 1
    assert topic_events[0].refs == ["D1:3"]
    assert not any(event.path.startswith("timeline/") for event in index.events)


def test_index_parses_native_memory_marker_blocks(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    workspace = MemoryWorkspace(memory_dir)
    workspace.archive_sessions([{
        "observation_date": "2023-05-20",
        "turns": [("user", "I started yoga.")],
        "refs": ["D1:1"],
    }])
    workspace.save_memory([{
        "when": "2023-05-20",
        "content": "The user started yoga.",
        "refs": ["D1:1"],
        "topic_path": "health/yoga.md",
        "headings": ["Health", "Yoga"],
    }])

    events = parse_topic_file(memory_dir / "topics/health/yoga.md", memory_dir / "topics")

    assert len(events) == 1
    assert events[0].content.startswith("[2023-05-20]")
    assert events[0].headings == ["Health", "Yoga"]
    assert events[0].refs == ["D1:1"]


def test_bm25_search_ranks_exact_event_and_returns_trace_fields(tmp_path: Path):
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

    result = MemoryBM25Index(memory_dir).search("Which support group did Caroline visit?")

    assert result[0]["refs"] == ["D1:3"]
    assert result[0]["bm25_score"] > 0
    assert result[0]["final_score"] >= result[0]["lexical_score"]
    assert result[0]["rule_features"]


def test_search_filters_path_and_date(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    _write_events(memory_dir, [
        {
            "when": "2023-05-07",
            "summary": "Caroline joined a support group",
            "dia_ids": ["D1:3"],
            "topic": "Caroline/support",
        },
        {
            "when": "2024-01-01",
            "summary": "Melanie joined a support group",
            "dia_ids": ["D2:3"],
            "topic": "Melanie/support",
        },
    ])

    result = MemoryBM25Index(memory_dir).search(
        "support group",
        path_prefix="Caroline",
        date_from="2023-01-01",
        date_to="2023-12-31",
    )

    assert [row["refs"] for row in result] == [["D1:3"]]


@pytest.mark.parametrize("prefix", ["topics/", "topics", "Caroline", "topics/Caroline"])
def test_path_prefix_accepts_root_and_relative_forms(tmp_path: Path, prefix: str):
    """A prefix naming a root must not be rewritten into topics/topics."""
    memory_dir = tmp_path / "memory"
    _write_events(memory_dir, [
        {
            "when": "2023-05-07",
            "summary": "Caroline joined a support group",
            "dia_ids": ["D1:3"],
            "topic": "Caroline/support",
        },
    ])

    result = MemoryBM25Index(memory_dir).search(
        "support group", path_prefix=prefix
    )

    assert [row["refs"] for row in result] == [["D1:3"]]


def test_path_prefix_sources_root_excludes_topics(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    _write_events(memory_dir, [
        {
            "when": "2023-05-07",
            "summary": "Caroline joined a support group",
            "dia_ids": ["D1:3"],
            "topic": "Caroline/support",
        },
    ])

    index = MemoryBM25Index(memory_dir)

    from_sources = index.search("support group", path_prefix="sources/")
    from_topics = index.search("support group", path_prefix="topics/")

    assert from_sources and from_topics
    assert all(row["path"].startswith("sources/") for row in from_sources)
    assert all(row["path"].startswith("topics/") for row in from_topics)


def test_parse_topic_indexes_each_evidence_supported_claim(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    topic = memory_dir / "topics/personal/residence.md"
    topic.parent.mkdir(parents=True)
    topic.write_text(
        "# Personal\n\n## Residence\n\n"
        "The user lived in Beijing in 2019.[^e-beijing] "
        "The user moved to Shanghai in 2023.[^e-shanghai] ^block123\n\n"
        "[^e-beijing]: Time: `2019`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
        "[^e-shanghai]: Time: `2023`; Sources: [D2:1](../../sources/D2.md#d2-1)\n",
        encoding="utf-8",
    )

    events = parse_topic_file(topic, memory_dir / "topics")

    assert [event.dates for event in events] == [["2019"], ["2023"]]
    assert [event.refs for event in events] == [["D1:1"], ["D2:1"]]
    assert "Beijing" in events[0].content
    assert "Shanghai" in events[1].content


def test_search_uses_discrete_evidence_intervals_not_paragraph_envelope(
    tmp_path: Path,
):
    memory_dir = tmp_path / "memory"
    topic = memory_dir / "topics/work/history.md"
    topic.parent.mkdir(parents=True)
    topic.write_text(
        "# Work\n\n"
        "The user worked on the Atlas project.[^e-2019][^e-2023] ^block456\n\n"
        "[^e-2019]: Time: `2019`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
        "[^e-2023]: Time: `2023`; Sources: [D2:1](../../sources/D2.md#d2-1)\n",
        encoding="utf-8",
    )
    index = MemoryBM25Index(memory_dir, persist=False)

    assert index.search("Atlas project", date_from="2021", date_to="2021") == []
    result = index.search("Atlas project", date_from="2023-05", date_to="2023-05")

    assert len(result) == 1
    assert result[0]["dates"] == ["2019", "2023"]


def test_search_validates_temporal_bounds_and_excludes_undated_claims(
    tmp_path: Path,
):
    memory_dir = tmp_path / "memory"
    topic = memory_dir / "topics/preferences.md"
    topic.parent.mkdir(parents=True)
    topic.write_text(
        "# Preferences\n\n"
        "The user liked analog photography.[^e-undated] ^block789\n\n"
        "[^e-undated]: Time: `undated`; Sources: [D1:1](../sources/D1.md#d1-1)\n",
        encoding="utf-8",
    )
    index = MemoryBM25Index(memory_dir, persist=False)

    assert index.search("analog photography")
    assert index.search("analog photography", date_from="2023") == []
    for invalid in ("23", "2023-1", "2023-13"):
        with pytest.raises(ValueError, match="invalid date_from"):
            index.search("analog photography", date_from=invalid)
    with pytest.raises(ValueError, match="date_from must not be after date_to"):
        index.search("analog photography", date_from="2024", date_to="2023")


def test_incremental_cache_matches_fresh_rebuild(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    _write_events(memory_dir, [{
        "when": "2023-05-07",
        "summary": "Caroline joined a support group",
        "dia_ids": ["D1:3"],
        "topic": "Caroline/support",
    }])
    index = MemoryBM25Index(memory_dir)
    initial = index.search("support group")

    _write_events(memory_dir, [{
        "when": "2023-05-08",
        "summary": "Melanie painted a sunrise",
        "dia_ids": ["D1:4"],
        "topic": "Melanie/painting",
    }])
    incremental = index.search("sunrise painting")
    cache = json.loads((memory_dir / ".nativemem-bm25.json").read_text())
    (memory_dir / ".nativemem-bm25.json").unlink()
    rebuilt = MemoryBM25Index(memory_dir).search("sunrise painting")

    assert initial[0]["refs"] == ["D1:3"]
    assert incremental == rebuilt
    assert set(cache["files"]) == {
        "sources/D1.md",
        "topics/Caroline/support.md",
        "topics/Melanie/painting.md",
    }


def test_bm25_can_search_without_writing_a_query_time_cache(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    _write_events(memory_dir, [{
        "when": "2023-05-07",
        "summary": "Caroline joined a support group",
        "dia_ids": ["D1:3"],
        "topic": "Caroline/support",
    }])

    results = MemoryBM25Index(memory_dir, persist=False).search("support group")

    assert results
    assert not (memory_dir / ".nativemem-bm25.json").exists()


def test_bm25_searches_source_turns_missing_from_topics(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    source = memory_dir / "sources/locomo/thread_1.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# thread_1\n\n"
        '<a id="source-abc"></a>\n'
        "<!-- source-id:locomo/thread_1/msg_1 -->\n"
        "[2023-10-04] Calvin: I bought a drum machine.\n",
        encoding="utf-8",
    )

    results = MemoryBM25Index(memory_dir, persist=False).search("drum machine")

    assert results[0]["path"] == "sources/locomo/thread_1.md"
    assert results[0]["refs"] == ["locomo/thread_1/msg_1"]
    assert "bought a drum machine" in results[0]["content"]
