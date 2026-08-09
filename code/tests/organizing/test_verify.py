"""unreachable(): a recorded fact search cannot find is invisible.

The "not found" case is not synthetic. `memory/retrieval/bm25.py` only turns
a `^block-id` paragraph into a searchable event; a legacy `[^mem_...]`
citation with no block-id suffix is a fact `memory/markdown/parser.py` still
accepts, but the index silently never sees it. That gap is real today, and
is exactly what this check exists to surface.
"""

from pathlib import Path

from memory.organizing.verify import unreachable
from memory.workspace import MemoryWorkspace


def _seed(memory_dir: Path, refs: list[str]) -> None:
    MemoryWorkspace(memory_dir).archive_sessions([{
        "observation_date": "2026-01-01",
        "turns": [("user", f"about {ref}") for ref in refs],
        "refs": refs,
    }])


def _topic(memory_dir: Path, relative: str, text: str) -> Path:
    path = memory_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_a_recorded_fact_findable_by_its_own_wording_is_reachable(tmp_path):
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1"])
    _topic(memory_dir, "topics/people/dave.md", (
        "# Dave\n\n"
        "Dave moved to Pudong.[^e1] ^blockidaaa\n\n"
        "[^e1]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
    ))

    assert unreachable(memory_dir) == []


def test_a_legacy_citation_fact_with_no_block_id_is_unreachable(tmp_path):
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1"])
    _topic(memory_dir, "topics/people/dave.md", (
        "# Dave\n\n"
        "Dave moved to Pudong.[^mem_test1]\n\n"
        "[^mem_test1]: Time: `2026-01-01`; "
        "Sources: [D1:1](../../sources/D1.md#d1-1)\n"
    ))

    missing = unreachable(memory_dir)

    assert missing == [{
        "path": "topics/people/dave.md",
        "block_id": "mem_test1",
        "fact": "Dave moved to Pudong.",
    }]


def test_a_workspace_with_no_topics_directory_has_nothing_unreachable(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    assert unreachable(memory_dir) == []
