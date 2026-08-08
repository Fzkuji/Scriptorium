"""Structured write transaction used by the interactive MCP path."""

import subprocess
from pathlib import Path

import pytest

from src.management import MemoryWorkspace
from src.management.transaction import TransactionError, workspace_revision


def make_patch(path: str, lines: list[str], *, create: bool = True) -> str:
    body = "".join(f"+{line}\n" for line in lines)
    old = "/dev/null" if create else f"a/{path}"
    return (
        f"--- {old}\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{body}"
    )


TOPIC_LINES = [
    "# Residence",
    "",
    "The user moved to Pudong.[^new-evidence-move] ^new-block-residence",
    "",
    "[^new-evidence-move]: Time: `2026-08-03`; Sources: new-source-move",
]

SOURCES = [{
    "label": "new-source-move",
    "role": "user",
    "content": "I have moved to Pudong.",
    "observed_at": "2026-08-03T10:30:00+08:00",
}]


def commit_move(workspace: MemoryWorkspace, **kwargs):
    return workspace.update(
        base_revision=workspace.revision(),
        patch=make_patch("topics/personal/residence.md", TOPIC_LINES),
        sources=SOURCES,
        **kwargs,
    )


def test_source_and_topic_commit_atomically(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)

    result = commit_move(workspace)

    topic = (tmp_path / "topics/personal/residence.md").read_text()
    assert "new-source-move" not in topic
    assert result.source_ids["new-source-move"].startswith("claude-code/")
    assert result.block_ids["topics/personal/residence.md#0"]
    # Evidence and the topic citing it must land in the same install.
    archived = list((tmp_path / "sources" / "claude-code").glob("*.md"))
    assert len(archived) == 1
    assert "I have moved to Pudong." in archived[0].read_text()
    assert (tmp_path / "recent_events.jsonl").is_file()
    assert (tmp_path / "relations.json").is_file()


def test_retrying_identical_sources_does_not_duplicate(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    first = commit_move(workspace)

    second = workspace.update(
        base_revision=workspace.revision(),
        patch=make_patch("topics/personal/second.md", [
            "# Second",
            "",
            "The user still lives in Pudong.[^new-evidence-again] ^new-block-again",
            "",
            "[^new-evidence-again]: Time: `2026-08-03`; Sources: new-source-move",
        ]),
        sources=SOURCES,
    )

    assert second.source_ids == first.source_ids
    archived = list((tmp_path / "sources" / "claude-code").glob("*.md"))
    assert len(archived) == 1
    assert archived[0].read_text().count("I have moved to Pudong.") == 1


def test_stale_base_revision_is_rejected(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    stale = workspace.revision()
    commit_move(workspace)

    with pytest.raises(TransactionError) as excinfo:
        workspace.update(
            base_revision=stale,
            patch=make_patch("topics/other.md", [
                "# Other",
                "",
                "Another fact.[^new-evidence-x] ^new-block-x",
                "",
                "[^new-evidence-x]: Time: `2026-08-03`; Sources: new-source-move",
            ]),
            sources=SOURCES,
        )

    assert excinfo.value.code == "CONCURRENT_UPDATE"


def test_undeclared_source_label_is_rejected(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)

    with pytest.raises(TransactionError) as excinfo:
        workspace.update(
            base_revision=workspace.revision(),
            patch=make_patch("topics/personal/residence.md", TOPIC_LINES),
            sources=[],
        )

    assert excinfo.value.code == "MISSING_SOURCE"


@pytest.mark.parametrize("path", [
    "sources/claude-code/thread.md",
    "timeline/2026/08/03.md",
    "relations.json",
    "recent_events.jsonl",
    "../escape.md",
    "/etc/passwd",
])
def test_only_topics_and_core_are_writable(tmp_path: Path, path: str):
    workspace = MemoryWorkspace(tmp_path)

    with pytest.raises(TransactionError) as excinfo:
        workspace.update(
            base_revision=workspace.revision(),
            patch=make_patch(path, ["hello"]),
        )

    assert excinfo.value.code in ("READ_ONLY_PATH", "PATH_OUTSIDE_WORKSPACE")


@pytest.mark.parametrize("header", [
    "rename from topics/a.md",
    "old mode 100644",
    "new file mode 120000",
    "GIT binary patch",
])
def test_unsupported_patch_features_are_rejected(tmp_path: Path, header: str):
    workspace = MemoryWorkspace(tmp_path)

    with pytest.raises(TransactionError) as excinfo:
        workspace.update(
            base_revision=workspace.revision(),
            patch=f"diff --git a/topics/a.md b/topics/a.md\n{header}\n"
                  + make_patch("topics/a.md", ["x"]),
        )

    assert excinfo.value.code == "INVALID_ARGUMENT"


def test_invalid_topic_rolls_back_workspace(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    commit_move(workspace)
    before_revision = workspace.revision()
    before_files = sorted(
        p.relative_to(tmp_path).as_posix()
        for p in tmp_path.rglob("*") if p.is_file()
    )

    with pytest.raises(TransactionError):
        workspace.update(
            base_revision=before_revision,
            # Cites a footnote that is never defined.
            patch=make_patch("topics/broken.md", [
                "# Broken",
                "",
                "Dangling fact.[^new-evidence-missing] ^new-block-broken",
            ]),
        )

    assert workspace.revision() == before_revision
    assert sorted(
        p.relative_to(tmp_path).as_posix()
        for p in tmp_path.rglob("*") if p.is_file()
    ) == before_files
    assert not (tmp_path / "topics/broken.md").exists()


def test_patch_conflict_when_context_does_not_match(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    commit_move(workspace)

    conflicting = (
        "--- a/topics/personal/residence.md\n"
        "+++ b/topics/personal/residence.md\n"
        "@@ -1,1 +1,1 @@\n"
        "-# Something Else\n"
        "+# Renamed\n"
    )
    with pytest.raises(TransactionError) as excinfo:
        workspace.update(
            base_revision=workspace.revision(), patch=conflicting
        )

    assert excinfo.value.code == "PATCH_CONFLICT"


def test_git_commit_on_requires_repository(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)

    with pytest.raises(TransactionError) as excinfo:
        commit_move(workspace, git_commit="on")

    error = excinfo.value
    assert error.code == "GIT_COMMIT_FAILED"
    # The files really were installed; only the commit failed.
    assert error.details["memory_committed"] is True
    assert error.details["git_committed"] is False
    assert (tmp_path / "topics/personal/residence.md").is_file()


def test_git_commit_auto_skips_plain_directory(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)

    result = commit_move(workspace, git_commit="auto")

    assert result.memory_committed is True
    assert result.git_committed is False
    assert result.git_commit is None


def test_git_commit_records_revision_in_repository(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True
    )
    workspace = MemoryWorkspace(tmp_path)

    result = commit_move(workspace, git_commit="on", commit_message="Add residence")

    assert result.git_committed is True
    assert result.git_commit
    log = subprocess.run(
        ["git", "log", "-1", "--pretty=%s"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert log.stdout.strip() == "Add residence"


def test_revision_ignores_retrieval_cache_files(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    commit_move(workspace)
    before = workspace.revision()

    # A read-side BM25 cache must not look like a concurrent write.
    (tmp_path / ".nativemem-bm25-cache.json").write_text("{}")

    assert workspace.revision() == before


def test_commit_message_control_characters_are_stripped(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True
    )
    workspace = MemoryWorkspace(tmp_path)

    commit_move(
        workspace,
        git_commit="on",
        commit_message="Add\nresidence; rm -rf /\x00",
    )

    log = subprocess.run(
        ["git", "log", "-1", "--pretty=%s"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "\n" not in log.stdout.strip()
    assert log.stdout.strip() == "Add residence; rm -rf /"


def test_workspace_revision_changes_with_content(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    before = workspace_revision(tmp_path)
    commit_move(workspace)

    assert workspace_revision(tmp_path) != before
