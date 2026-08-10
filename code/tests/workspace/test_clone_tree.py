"""Staging a workspace by clone must be as separate as staging by copy.

The stage is rebuilt from the workspace on every write and then written to,
so a clone that shared its blocks with the workspace after a write would edit
the committed memory in place. APFS separates them copy-on-write; these pin
that, and pin the fallback for anywhere that cannot clone at all.
"""
from __future__ import annotations

import time
from pathlib import Path

from memory.workspace import staging
from memory.workspace.staging import clone_tree, discard_tree


def _tree(root: Path) -> Path:
    (root / "topics" / "people").mkdir(parents=True)
    (root / "topics" / "people" / "dave.md").write_text(
        "# Dave\n\nDave moved to Berlin. ^abc123\n", encoding="utf-8"
    )
    (root / "topics" / "index.md").write_text("# Index\n", encoding="utf-8")
    return root


def test_a_clone_holds_what_the_original_held(tmp_path: Path) -> None:
    source = _tree(tmp_path / "memory")

    clone_tree(source / "topics", tmp_path / "stage")

    assert (tmp_path / "stage" / "index.md").read_text(encoding="utf-8") == "# Index\n"
    assert "Dave moved to Berlin" in (
        tmp_path / "stage" / "people" / "dave.md"
    ).read_text(encoding="utf-8")


def test_writing_to_the_clone_leaves_the_original_alone(tmp_path: Path) -> None:
    source = _tree(tmp_path / "memory")
    original = source / "topics" / "people" / "dave.md"
    before = original.read_text(encoding="utf-8")

    clone_tree(source / "topics", tmp_path / "stage")
    staged = tmp_path / "stage" / "people" / "dave.md"
    staged.write_text(before + "\nDave took up sailing. ^def456\n", encoding="utf-8")
    (tmp_path / "stage" / "new.md").write_text("# New\n", encoding="utf-8")
    staged.chmod(0o444)

    assert original.read_text(encoding="utf-8") == before, "the commit is untouched"
    assert not (source / "topics" / "new.md").exists()
    assert original.stat().st_mode & 0o200, "and still writable"


def test_a_discarded_tree_is_gone_and_stays_gone(tmp_path: Path) -> None:
    stage = _tree(tmp_path / "stage")
    # Staged sources are read-only. Removing them has to work anyway.
    (stage / "topics" / "index.md").chmod(0o444)

    discard_tree(stage)

    assert not stage.exists(), "the path is free the moment the call returns"
    # Shutting the pool down to wait would close it for every later test, so
    # watch the directory instead: what was renamed aside has to actually go,
    # or a long run leaks a stage per Add.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and list(tmp_path.iterdir()):
        time.sleep(0.02)
    assert not list(tmp_path.iterdir()), "and the files behind it are removed"


def test_discarding_what_is_not_there_is_quiet(tmp_path: Path) -> None:
    discard_tree(tmp_path / "never-staged")


def test_it_copies_where_it_cannot_clone(tmp_path: Path, monkeypatch) -> None:
    # Everything that is not APFS takes this path, and so does a clone that
    # the filesystem refuses.
    monkeypatch.setattr(staging, "_CLONEFILE", None)
    source = _tree(tmp_path / "memory")

    clone_tree(source / "topics", tmp_path / "stage")

    assert "Dave moved to Berlin" in (
        tmp_path / "stage" / "people" / "dave.md"
    ).read_text(encoding="utf-8")
