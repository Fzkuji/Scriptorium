import re
from pathlib import Path

import pytest

from src.markdown import (
    EvidenceAnnotation,
    MemoryUnit,
    TopicFormatError,
    append_memory_unit,
    parse_topic_tree,
)
from src.management import MemoryWorkspace


def test_parse_obisidian_paragraph_block_with_multiple_evidence_annotations(
    tmp_path: Path,
):
    topics = tmp_path / "topics"
    topic = topics / "personal" / "residence.md"
    topic.parent.mkdir(parents=True)
    topic.write_text(
        "# Personal\n\n## Residence\n\n"
        "The user moved to Shanghai.[^e-move] They later moved to "
        "[Pudong](../places/pudong.md#^place-1234).[^e-pudong] ^8c41d20fa693\n\n"
        "[^e-move]: Time: `2026-07-20`; Sources: "
        "[D1:1, migration discussion](../../sources/D1.md#d1-1)\n"
        "[^e-pudong]: Time: `2026-08-03`; Sources: "
        "[D1:2](../../sources/D1.md#d1-2)\n",
        encoding="utf-8",
    )

    units = parse_topic_tree(topics)

    assert len(units) == 1
    assert units[0].memory_id == "8c41d20fa693"
    assert units[0].content == (
        "The user moved to Shanghai. They later moved to "
        "[Pudong](../places/pudong.md#^place-1234)."
    )
    assert units[0].relation_targets == ("place-1234",)
    assert units[0].evidence == (
        EvidenceAnnotation(
            citation_id="e-move",
            quote="The user moved to Shanghai.",
            when="2026-07-20",
            source_refs=("D1:1",),
            source_links=("../../sources/D1.md#d1-1",),
        ),
        EvidenceAnnotation(
            citation_id="e-pudong",
            quote="They later moved to [Pudong](../places/pudong.md#^place-1234).",
            when="2026-08-03",
            source_refs=("D1:2",),
            source_links=("../../sources/D1.md#d1-2",),
        ),
    )


def test_parse_obisidian_paragraph_rejects_invalid_or_duplicate_block_ids(
    tmp_path: Path,
):
    topics = tmp_path / "topics"
    topics.mkdir()
    (topics / "bad.md").write_text(
        "First.[^e-1] ^bad_id\n\nSecond.[^e-2] ^same-id\n\n"
        "Third.[^e-3] ^same-id\n\n"
        "[^e-1]: Time: `undated`; Sources: [D1:1](../sources/D1.md#d1-1)\n"
        "[^e-2]: Time: `undated`; Sources: [D1:2](../sources/D1.md#d1-2)\n"
        "[^e-3]: Time: `undated`; Sources: [D1:3](../sources/D1.md#d1-3)\n",
        encoding="utf-8",
    )

    with pytest.raises(TopicFormatError, match="invalid block_id|duplicate memory_id"):
        parse_topic_tree(topics)


def test_parse_topic_rejects_untracked_prose_in_block_workspace(tmp_path: Path):
    topics = tmp_path / "topics"
    topics.mkdir()
    (topics / "bad.md").write_text(
        "# Topic\n\nTracked fact.[^e-1] ^abc123\n\n"
        "This paragraph has no block ID or evidence.\n\n"
        "[^e-1]: Time: `undated`; Sources: [D1:1](../sources/D1.md#d1-1)\n",
        encoding="utf-8",
    )

    with pytest.raises(TopicFormatError, match="memory block ID required"):
        parse_topic_tree(topics)


@pytest.mark.parametrize(
    "paragraph, definition, message",
    [
        (
            "Supported.[^e-1] Unsupported tail. ^abc123",
            "[^e-1]: Time: `2026-01-01`; Sources: [D1:1](../sources/D1.md#d1-1)",
            "content after final evidence",
        ),
        (
            "Fact.[^e-1] ^abc123",
            "[^e-1]: Time: `2026-02-30`; Sources: [D1:1](../sources/D1.md#d1-1)",
            "invalid evidence date",
        ),
        (
            "Fact.[^e-1] ^abc123",
            "[^e-1]: Time: `2026-13`; Sources: [D1:1](../sources/D1.md#d1-1)",
            "invalid evidence date",
        ),
    ],
)
def test_parse_topic_rejects_uncovered_tail_or_invalid_calendar_date(
    tmp_path: Path, paragraph: str, definition: str, message: str
):
    topics = tmp_path / "topics"
    topics.mkdir()
    (topics / "bad.md").write_text(
        f"# Topic\n\n{paragraph}\n\n{definition}\n", encoding="utf-8"
    )

    with pytest.raises(TopicFormatError, match=message):
        parse_topic_tree(topics)


def test_parse_freeform_topic_with_adjacent_memory_ids(tmp_path: Path):
    topics = tmp_path / "topics"
    topic = topics / "personal" / "residence.md"
    topic.parent.mkdir(parents=True)
    topic.write_text(
        "# Personal\n\n## Residence\n\n"
        "The user moved to Shanghai.[^mem_move] "
        "They then started a new job.[^mem_job][^mem_contract]\n\n"
        "[^mem_move]: Time: `2026-07-20`; Sources: [move](../../sources/thread.md#msg-1)\n"
        "[^mem_job]: Time: `2026-08-05`; Sources: [job](../../sources/thread.md#msg-2)\n"
        "[^mem_contract]: Time: `undated`; Sources: [contract](../../documents/job.pdf#page=2)\n",
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
        "[^mem_a]: Time: `2026-01-01`; Sources: [source](../sources/D1.md#d1-1)\n",
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
    assert "[^mem_abc123]: Time: `2026-07-20`; Sources:" in text
    assert "·" not in text
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
    assert re.fullmatch(r"[0-9a-f]{8}", units[0].memory_id)
    assert len(units[0].evidence) == 1
    assert units[0].evidence[0].when == "2026-07-20"
    text = (tmp_path / "topics/personal/residence.md").read_text()
    assert f"^{units[0].memory_id}" in text
    assert "<!-- memory-event:" not in text


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

    assert not (tmp_path / "timeline/undated.md").exists()
    assert parse_topic_tree(tmp_path / "topics")[0].when is None


def test_workspace_save_memory_keeps_year_precision_in_timeline(
    tmp_path: Path,
):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2023-05-08",
        "turns": [("user", "I painted that lake sunrise last year.")],
        "refs": ["D1:1"],
    }])

    workspace.save_memory([{
        "when": "2022",
        "content": "The user painted a lake sunrise in 2022.",
        "refs": ["D1:1"],
        "topic_path": "creative/painting.md",
        "headings": ["Creative work", "Painting"],
    }])

    unit = parse_topic_tree(tmp_path / "topics")[0]
    assert unit.when == "2022"
    timeline = (tmp_path / "timeline/2022.md").read_text()
    assert "The user painted a lake sunrise in 2022." in timeline
    assert '"when": "2022"' in (tmp_path / "recent_events.jsonl").read_text()


def test_workspace_normalizes_temporary_ids_and_bare_source_handles(
    tmp_path: Path,
):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2026-08-03",
        "turns": [("user", "I moved to Pudong.")],
        "refs": ["D1:1"],
    }])
    workspace._refresh_stage()

    result = workspace.shell(
        "mkdir -p topics/personal && "
        "printf '%s\\n' '# Personal' '' '## Residence' '' "
        "'On 2026-08-03, the user moved to Pudong.[^new-evidence-1] ^new-block-1' '' "
        "'[^new-evidence-1]: Time: `2026-08-03`; Sources: D1:1' "
        "> topics/personal/residence.md"
    )

    assert result.returncode == 0
    text = (tmp_path / "topics/personal/residence.md").read_text()
    assert "new-evidence-1" not in text
    assert "new-block-1" not in text
    assert "[D1:1](../../sources/D1.md#d1-1)" in text
    unit = parse_topic_tree(tmp_path / "topics")[0]
    assert re.fullmatch(r"[0-9a-f]{8}", unit.memory_id)
    assert re.fullmatch(r"e-[0-9a-f]{10}", unit.evidence[0].citation_id)
