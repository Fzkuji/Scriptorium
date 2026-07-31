from pathlib import Path

import pytest

from src.nativemem_versions.v11.topic_markdown import (
    MemoryUnit,
    TopicFormatError,
    append_memory_unit,
    parse_topic_tree,
)
from src.v11_memory import MemoryWorkspace


def test_parse_freeform_topic_with_adjacent_memory_ids(tmp_path: Path):
    topics = tmp_path / "topics"
    topic = topics / "personal" / "residence.md"
    topic.parent.mkdir(parents=True)
    topic.write_text(
        "# Personal\n\n## Residence\n\n"
        "The user moved to Shanghai.[^mem_move] "
        "They then started a new job.[^mem_job][^mem_contract]\n\n"
        "[^mem_move]: 2026-07-20 · Sources: [move](../../sources/thread.md#msg-1)\n"
        "[^mem_job]: 2026-08-05 · Sources: [job](../../sources/thread.md#msg-2)\n"
        "[^mem_contract]: undated · Sources: [contract](../../documents/job.pdf#page=2)\n",
        encoding="utf-8",
    )

    units = parse_topic_tree(topics)

    assert [unit.memory_id for unit in units] == [
        "mem_move",
        "mem_job",
        "mem_contract",
    ]
    assert units[0].content == "The user moved to Shanghai."
    assert units[1].content == "They then started a new job."
    assert units[2].content == "They then started a new job."
    assert units[2].when is None
    assert units[0].topic_path == "personal/residence.md"
    assert units[0].headings == ("Personal", "Residence")


def test_parse_topic_rejects_duplicate_or_undefined_ids(tmp_path: Path):
    topics = tmp_path / "topics"
    topics.mkdir()
    (topics / "bad.md").write_text(
        "Fact.[^mem_a] Again.[^mem_a] Missing.[^mem_b]\n\n"
        "[^mem_a]: 2026-01-01 · Sources: [source](../sources/D1.md#d1-1)\n",
        encoding="utf-8",
    )

    with pytest.raises(TopicFormatError, match="duplicate memory_id|undefined footnote"):
        parse_topic_tree(topics)


def test_append_memory_unit_renders_authoritative_footnote(tmp_path: Path):
    topic = tmp_path / "topics" / "people" / "alice.md"
    unit = MemoryUnit(
        memory_id="mem_abc123",
        content="Alice moved to Shanghai.",
        when="2026-07-20",
        source_refs=("D1:2",),
        source_links=("../../sources/D1.md#d1-2",),
        topic_path="people/alice.md",
        headings=("Alice", "Residence"),
        created_order=0,
    )

    append_memory_unit(topic, unit)
    parsed = parse_topic_tree(tmp_path / "topics")

    assert parsed == [unit]
    text = topic.read_text(encoding="utf-8")
    assert "Alice moved to Shanghai.[^mem_abc123]" in text
    assert "[^mem_abc123]: 2026-07-20 · Sources:" in text
    assert "[D1:2](../../sources/D1.md#d1-2)" in text


def test_workspace_save_memory_writes_parseable_footnote_topic(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2026-07-20",
        "turns": [("user", "I moved to Shanghai.")],
        "refs": ["D1:1"],
    }])

    workspace.save_memory([{
        "when": "2026-07-20",
        "content": "The user moved to Shanghai.",
        "refs": ["D1:1"],
        "topic_path": "personal/residence.md",
        "headings": ["Personal", "Residence"],
    }])

    units = parse_topic_tree(tmp_path / "topics")
    assert len(units) == 1
    assert units[0].content == "The user moved to Shanghai."
    assert units[0].source_refs == ("D1:1",)


def test_workspace_rebuilds_timeline_from_topic_without_timeline_catalog(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2026-07-20",
        "turns": [("user", "I moved to Shanghai.")],
        "refs": ["D1:1"],
    }])
    workspace.save_memory([{
        "when": "2026-07-20",
        "content": "The user moved to Shanghai.",
        "refs": ["D1:1"],
        "topic_path": "personal/residence.md",
        "headings": ["Personal", "Residence"],
    }])
    timeline = tmp_path / "timeline"
    for path in sorted(timeline.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    timeline.rmdir()

    organizer = MemoryWorkspace(tmp_path)
    result = organizer.shell(
        "mkdir -p topics/life && mv topics/personal/residence.md topics/life/residence.md"
    )

    assert result.returncode == 0
    assert "topics/life/residence.md" in (tmp_path / "timeline/2026/07/20.md").read_text()


def test_workspace_save_memory_accepts_undated_event(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{"observation_date": "2026-01-01", "turns": [("user", "I like tea")], "refs": ["D1:1"]}])

    workspace.save_memory([{"when": "undated", "content": "The user likes tea.", "refs": ["D1:1"], "topic_path": "preferences.md", "headings": ["Preferences"]}])

    assert (tmp_path / "timeline/undated.md").exists()
    assert parse_topic_tree(tmp_path / "topics")[0].when is None
