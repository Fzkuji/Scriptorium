import json
from pathlib import Path

from src.nativemem_versions.v11.derived_views import rebuild_derived_views
from src.nativemem_versions.v11.topic_markdown import MemoryUnit


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


def test_rebuild_views_orders_timeline_and_supports_undated(tmp_path: Path):
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
    assert "unknown" in (tmp_path / "timeline/undated.md").read_text()
    recent = [json.loads(line) for line in (tmp_path / "recent_events.jsonl").read_text().splitlines()]
    assert [row["memory_id"] for row in recent] == ["mem_b", "mem_c"]


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
