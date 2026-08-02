import json
from pathlib import Path

import pytest

from src.nativemem_versions.v11.derived_views import rebuild_derived_views
from src.nativemem_versions.v11.topic_markdown import EvidenceAnnotation, MemoryUnit


def unit(memory_id, content, when, topic_path, order, ref="D1:1"):
    return MemoryUnit(
        memory_id=memory_id,
        content=content,
        when=when,
        source_refs=(ref,),
        source_links=(f"../../sources/D1.md#d1-{ref.split(':')[1]}",),
        topic_path=topic_path,
        headings=("Topic",),
        created_order=order,
    )


def test_rebuild_views_orders_timeline_and_omits_undated(tmp_path: Path):
    units = [
        unit("mem_b", "second", "2026-07-20", "b.md", 1, "D1:2"),
        unit("mem_a", "first", "2026-07-20", "a.md", 0, "D1:1"),
        unit("mem_c", "unknown", None, "c.md", 2, "D1:3"),
    ]

    rebuild_derived_views(tmp_path, units, recent_limit=2)

    dated = (tmp_path / "timeline/2026/07/20.md").read_text()
    assert dated.index("first") < dated.index("second")
    assert "topics/a.md#mem_a" in dated
    assert "sources/D1.md#d1-1" in dated
    assert not (tmp_path / "timeline/undated.md").exists()
    recent = [json.loads(line) for line in (tmp_path / "recent_events.jsonl").read_text().splitlines()]
    assert [row["memory_id"] for row in recent] == ["mem_b", "mem_c"]


def test_rebuild_timeline_uses_each_dated_evidence_in_one_paragraph_block(
    tmp_path: Path,
):
    unit_with_history = MemoryUnit(
        memory_id="8c41d20fa693",
        content="The user moved to Shanghai and later to Pudong.",
        when=None,
        source_refs=("D1:1", "D1:2", "D1:3"),
        source_links=(
            "../../sources/D1.md#d1-1",
            "../../sources/D1.md#d1-2",
            "../../sources/D1.md#d1-3",
        ),
        topic_path="personal/residence.md",
        headings=("Personal", "Residence"),
        created_order=0,
        evidence=(
            EvidenceAnnotation(
                "e-move", "The user moved to Shanghai.", "2026-07-20",
                ("D1:1",), ("../../sources/D1.md#d1-1",),
            ),
            EvidenceAnnotation(
                "e-pudong", "The user later moved to Pudong.", "2026-08-03",
                ("D1:2",), ("../../sources/D1.md#d1-2",),
            ),
            EvidenceAnnotation(
                "e-preference", "The user prefers short commutes.", None,
                ("D1:3",), ("../../sources/D1.md#d1-3",),
            ),
        ),
    )

    rebuild_derived_views(tmp_path, [unit_with_history])

    july = (tmp_path / "timeline/2026/07/20.md").read_text()
    august = (tmp_path / "timeline/2026/08/03.md").read_text()
    assert "The user moved to Shanghai." in july
    assert "The user later moved to Pudong." in august
    assert "topics/personal/residence.md#^8c41d20fa693" in july
    assert "topics/personal/residence.md#^8c41d20fa693" in august
    assert not (tmp_path / "timeline/undated.md").exists()


@pytest.mark.parametrize(
    ("when", "relative_path"),
    [
        ("2022", "timeline/2022.md"),
        ("2022-05", "timeline/2022/05.md"),
    ],
)
def test_rebuild_timeline_preserves_partial_time_precision(
    tmp_path: Path,
    when: str,
    relative_path: str,
):
    rebuild_derived_views(
        tmp_path,
        [unit("partial-time", f"Event in {when}.", when, "events.md", 0)],
    )

    files = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in (tmp_path / "timeline").rglob("*.md")
    )
    assert files == [relative_path]
    assert f"# {when}" in (tmp_path / relative_path).read_text()


def test_rebuild_views_uses_persisted_creation_order_after_topic_move(tmp_path: Path):
    units = [
        unit("mem_old", "old", "2026-01-01", "new/path.md", 99),
        unit("mem_new", "new", "2026-01-02", "other.md", 0),
    ]

    result = rebuild_derived_views(
        tmp_path,
        units,
        recent_limit=2,
        creation_order={"mem_old": 0, "mem_new": 1},
    )

    recent = [json.loads(line) for line in (tmp_path / "recent_events.jsonl").read_text().splitlines()]
    assert [row["memory_id"] for row in recent] == ["mem_old", "mem_new"]
    assert result.structure_map == "topics/new/path.md\n  # Topic\ntopics/other.md\n  # Topic"


def test_rebuild_views_materializes_relations_and_rejects_dangling_targets(
    tmp_path: Path,
):
    source = unit("source-block", "source", "2026-01-01", "a.md", 0)
    source = MemoryUnit(
        **{
            **source.__dict__,
            "relation_targets": ("target-block",),
        }
    )
    target = unit("target-block", "target", "2026-01-02", "b.md", 1)

    rebuild_derived_views(tmp_path, [source, target])

    relations = json.loads((tmp_path / "relations.json").read_text())
    assert relations == {
        "backlinks": {"target-block": ["source-block"]},
        "outbound": {"source-block": ["target-block"], "target-block": []},
    }

    with pytest.raises(ValueError, match="dangling block link"):
        rebuild_derived_views(tmp_path, [source])
