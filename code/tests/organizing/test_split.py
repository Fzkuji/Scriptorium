"""split() against real workspace shapes: it had never been run before.

Fixtures are seeded through `MemoryWorkspace.archive_sessions` for their
source refs, exactly like `test_tidy.py`, and every scenario runs with a
tiny `threshold`/`min_section` override rather than a multi-kilobyte fixture
— the production defaults (16 KB, 500 B) were measured against real
workspaces (see `split.py`'s module and function docstrings), not against
what is convenient to hand-write here.
"""

import importlib
import re
from pathlib import Path

from memory.markdown import parse_topic_tree
from memory.organizing.split import split
from memory.workspace import MemoryWorkspace

# Mirrors test_tidy.py's importlib workaround: memory.organizing's __init__
# does `from .split import split`, rebinding the package attribute to the
# function and shadowing the submodule of the same name.
split_module = importlib.import_module("memory.organizing.split")


def _seed(memory_dir: Path, refs: list[str]) -> None:
    """Archive one source turn per ref, so a footnote citing it is valid."""
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


def test_a_file_past_the_threshold_becomes_a_directory_and_every_block_id_survives(
    tmp_path,
):
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1", "D1:2"])
    _topic(memory_dir, "topics/projects/tracker.md", (
        "## Database\n\n"
        "The schema uses a users table.[^e-fact-a] ^blockdbid1\n\n"
        "## Deployment\n\n"
        "The app deploys to Render.[^e-fact-b] ^blockdeploy1\n\n"
        "[^e-fact-a]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
        "[^e-fact-b]: Time: `2026-01-02`; Sources: [D1:2](../../sources/D1.md#d1-2)\n"
    ))

    report = split(memory_dir, threshold=10, min_section=1)

    assert report == {
        "files": [{
            "from": "topics/projects/tracker.md",
            "to": [
                "topics/projects/tracker/database.md",
                "topics/projects/tracker/deployment.md",
            ],
        }],
    }
    assert not (memory_dir / "topics/projects/tracker.md").exists()
    assert (memory_dir / "topics/projects/tracker/database.md").is_file()
    assert (memory_dir / "topics/projects/tracker/deployment.md").is_file()
    units = parse_topic_tree(memory_dir / "topics")
    assert {unit.memory_id for unit in units} == {"blockdbid1", "blockdeploy1"}
    by_id = {unit.memory_id: unit for unit in units}
    assert by_id["blockdbid1"].topic_path == "projects/tracker/database.md"
    assert by_id["blockdeploy1"].topic_path == "projects/tracker/deployment.md"


def test_a_link_into_a_moved_block_still_resolves_after_the_split(tmp_path):
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1", "D1:2", "D1:3"])
    _topic(memory_dir, "topics/projects/tracker.md", (
        "## Database\n\n"
        "The schema uses a users table.[^e-fact-a] ^blockdbid1\n\n"
        "## Deployment\n\n"
        "The app deploys to Render.[^e-fact-b] ^blockdeploy1\n\n"
        "[^e-fact-a]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
        "[^e-fact-b]: Time: `2026-01-02`; Sources: [D1:2](../../sources/D1.md#d1-2)\n"
    ))
    dave = _topic(memory_dir, "topics/people/dave.md", (
        "# Dave\n\n"
        "Dave designed [the schema](../projects/tracker.md#^blockdbid1) "
        "himself.[^e-fact-c] ^blockdaveid1\n\n"
        "[^e-fact-c]: Time: `2026-01-03`; Sources: [D1:3](../../sources/D1.md#d1-3)\n"
    ))

    split(memory_dir, threshold=10, min_section=1)

    text = dave.read_text(encoding="utf-8")
    match = re.search(r"\[the schema\]\(([^)]+)\)", text)
    assert match is not None
    target, _sep, fragment = match.group(1).partition("#")
    assert fragment == "^blockdbid1"
    resolved = (dave.parent / target).resolve()
    assert resolved == (memory_dir / "topics/projects/tracker/database.md").resolve()
    assert resolved.is_file()


def test_a_one_heading_file_is_untouched(tmp_path):
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1", "D1:2"])
    path = _topic(memory_dir, "topics/projects/weather.md", (
        "## Weather App\n\n"
        "It fetches forecasts from an API.[^e-fact-a] ^blockweather1\n\n"
        "It caches results for an hour.[^e-fact-b] ^blockweather2\n\n"
        "[^e-fact-a]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
        "[^e-fact-b]: Time: `2026-01-02`; Sources: [D1:2](../../sources/D1.md#d1-2)\n"
    ))
    before = path.read_bytes()

    report = split(memory_dir, threshold=10, min_section=1)

    assert report == {"files": []}
    assert path.read_bytes() == before
    assert not (memory_dir / "topics/projects/weather").exists()


def test_a_section_too_small_to_be_worth_its_own_file_leaves_the_whole_file_untouched(
    tmp_path,
):
    """One undersized section among the rest is enough to hold the split
    back entirely — nothing between "all of it" and "none of it" is
    offered, so a real section never has to share a file with a stub."""
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1", "D1:2"])
    path = _topic(memory_dir, "topics/projects/tracker.md", (
        "## Database\n\n"
        "The schema uses a users table with several columns for tracking "
        "user metadata across sessions and requests.[^e-fact-a] ^blockdbid1\n\n"
        "## Deployment\n\n"
        "Shipped.[^e-fact-b] ^blockdeploy1\n\n"
        "[^e-fact-a]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
        "[^e-fact-b]: Time: `2026-01-02`; Sources: [D1:2](../../sources/D1.md#d1-2)\n"
    ))
    before = path.read_bytes()

    # "Deployment" renders to 125 bytes, "Database" to 220: a threshold in
    # between fails only the small one, and that is enough to fail the file.
    report = split(memory_dir, threshold=10, min_section=150)

    assert report == {"files": []}
    assert path.read_bytes() == before
    assert not (memory_dir / "topics/projects/tracker").exists()


def test_a_split_that_would_break_the_contract_leaves_the_workspace_byte_identical(
    tmp_path, monkeypatch
):
    """However `_sections`/`_render_section` might go wrong, breaking the
    contract must not reach disk: this drives that path directly, the way a
    bug that dropped a block ID from a rendered section would."""
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1", "D1:2"])
    path = _topic(memory_dir, "topics/projects/tracker.md", (
        "## Database\n\n"
        "The schema uses a users table.[^e-fact-a] ^blockdbid1\n\n"
        "## Deployment\n\n"
        "The app deploys to Render.[^e-fact-b] ^blockdeploy1\n\n"
        "[^e-fact-a]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
        "[^e-fact-b]: Time: `2026-01-02`; Sources: [D1:2](../../sources/D1.md#d1-2)\n"
    ))
    before = path.read_bytes()

    original = split_module._render_section
    monkeypatch.setattr(
        split_module, "_render_section",
        lambda heading, entries, definitions: re.sub(
            r"\s\^[A-Za-z0-9-]+\n", "\n", original(heading, entries, definitions)
        ),
    )

    report = split(memory_dir, threshold=10, min_section=1)

    assert "error" in report
    assert report["files"] == []
    assert path.read_bytes() == before
    assert not (memory_dir / "topics/projects/tracker").exists()


def test_a_workspace_with_no_topics_directory_is_a_no_op(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    assert split(memory_dir, threshold=10, min_section=1) == {"files": []}
