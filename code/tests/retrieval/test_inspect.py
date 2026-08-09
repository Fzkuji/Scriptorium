"""Read-only inspection functions shared by the CLI and MCP server."""

from pathlib import Path

import pytest

from src.management import MemoryWorkspace
from src.management.transaction import TransactionError
from src.retrieval import inspect

TOPIC_LINES = [
    "# Residence",
    "",
    "## History",
    "",
    "The user moved to Pudong.[^new-evidence-move] ^new-block-residence",
    "",
    "## Notes",
    "",
    "Unrelated note.[^new-evidence-note] ^new-block-note",
    "",
    "[^new-evidence-move]: Time: `2026-08-03`; Sources: new-source-move",
    "[^new-evidence-note]: Time: `2026-08-03`; Sources: new-source-move",
]


@pytest.fixture
def workspace(tmp_path: Path) -> MemoryWorkspace:
    space = MemoryWorkspace(tmp_path)
    body = "".join(f"+{line}\n" for line in TOPIC_LINES)
    space.update(
        base_revision=space.revision(),
        patch=(
            "--- /dev/null\n"
            "+++ b/topics/personal/residence.md\n"
            f"@@ -0,0 +1,{len(TOPIC_LINES)} @@\n{body}"
        ),
        sources=[{
            "label": "new-source-move",
            "role": "user",
            "content": "I have moved to Pudong.",
            "observed_at": "2026-08-03T10:30:00+08:00",
        }],
    )
    return space


def test_status_counts_committed_state(workspace: MemoryWorkspace, tmp_path: Path):
    result = inspect.status(tmp_path)

    assert result["blocks"] == 2
    assert result["topic_files"] == 1
    assert result["source_files"] == 1
    assert result["revision"] == workspace.revision()
    assert result["embedding_available"] is False


def test_list_hides_runtime_directory(workspace: MemoryWorkspace, tmp_path: Path):
    paths = [entry["path"] for entry in inspect.list_files(tmp_path)["files"]]

    assert "topics/personal/residence.md" in paths
    assert not any(path.startswith(".nativemem") for path in paths)


def test_list_can_exclude_derived_views(workspace: MemoryWorkspace, tmp_path: Path):
    entries = inspect.list_files(tmp_path, include_derived=False)["files"]
    paths = [entry["path"] for entry in entries]

    assert "recent_events.jsonl" not in paths
    assert not any(path.startswith("timeline/") for path in paths)
    assert "topics/personal/residence.md" in paths


def test_read_block_includes_cited_footnotes(
    workspace: MemoryWorkspace, tmp_path: Path
):
    topic_hit = next(
        item for item in inspect.search(tmp_path, "Pudong")["results"]
        if item["path"].startswith("topics/")
    )
    block_id = topic_hit["event_id"].split(":")[0]

    result = inspect.read_file(
        tmp_path, "topics/personal/residence.md", block_id=block_id
    )

    assert result["mode"] == "block"
    assert "Pudong" in result["content"]
    # A block cannot be checked against its evidence without the footnote.
    assert "Time: `2026-08-03`" in result["content"]
    assert "Unrelated note" not in result["content"]


def test_read_heading_stops_at_next_heading(
    workspace: MemoryWorkspace, tmp_path: Path
):
    result = inspect.read_file(
        tmp_path, "topics/personal/residence.md", heading="History"
    )

    assert "Pudong" in result["content"]
    assert "Unrelated note" not in result["content"]


def test_read_rejects_multiple_selectors(
    workspace: MemoryWorkspace, tmp_path: Path
):
    with pytest.raises(TransactionError) as excinfo:
        inspect.read_file(
            tmp_path,
            "topics/personal/residence.md",
            heading="History",
            block_id="abcd1234",
        )

    assert excinfo.value.code == "INVALID_ARGUMENT"


def test_read_line_window_is_bounded(workspace: MemoryWorkspace, tmp_path: Path):
    long_file = tmp_path / "topics" / "long.md"
    long_file.write_text("\n".join(f"line {n}" for n in range(5000)))

    result = inspect.read_file(tmp_path, "topics/long.md")

    assert result["content"].count("\n") <= inspect.MAX_READ_LINES


@pytest.mark.parametrize("path", [
    "/etc/passwd",
    "../outside.md",
    "topics/../../outside.md",
])
def test_read_rejects_path_escape(
    workspace: MemoryWorkspace, tmp_path: Path, path: str
):
    with pytest.raises(TransactionError) as excinfo:
        inspect.read_file(tmp_path, path)

    assert excinfo.value.code == "PATH_OUTSIDE_WORKSPACE"


def test_read_rejects_symlink_escape(workspace: MemoryWorkspace, tmp_path: Path):
    secret = tmp_path.parent / "secret.md"
    secret.write_text("classified")
    (tmp_path / "topics" / "link.md").symlink_to(secret)

    with pytest.raises(TransactionError) as excinfo:
        inspect.read_file(tmp_path, "topics/link.md")

    assert excinfo.value.code == "PATH_OUTSIDE_WORKSPACE"


def test_grep_literal_ignores_regex_metacharacters(
    workspace: MemoryWorkspace, tmp_path: Path
):
    (tmp_path / "topics" / "meta.md").write_text("a.c literal\nabc regex\n")

    result = inspect.grep(tmp_path, "a.c", prefix="topics/", literal=True)

    assert [match["text"] for match in result["matches"]] == ["a.c literal"]


def test_grep_regex_mode_compiles_pattern(
    workspace: MemoryWorkspace, tmp_path: Path
):
    result = inspect.grep(
        tmp_path, r"Pud[o]ng", prefix="topics/", literal=False
    )

    assert result["total"] >= 1


def test_grep_rejects_invalid_regex(workspace: MemoryWorkspace, tmp_path: Path):
    with pytest.raises(TransactionError) as excinfo:
        inspect.grep(tmp_path, "unterminated(", literal=False)

    assert excinfo.value.code == "INVALID_ARGUMENT"


def test_grep_reports_truncation(workspace: MemoryWorkspace, tmp_path: Path):
    (tmp_path / "topics" / "many.md").write_text(
        "\n".join("needle" for _ in range(50))
    )

    result = inspect.grep(tmp_path, "needle", limit=5)

    assert len(result["matches"]) == 5
    assert result["total"] >= 50
    assert result["truncated"] is True


def test_search_bm25_finds_topic_and_reports_source(
    workspace: MemoryWorkspace, tmp_path: Path
):
    result = inspect.search(tmp_path, "Pudong", method="bm25")

    assert result["method"] == "bm25"
    assert result["results"]
    assert any("Pudong" in item.get("content", "") for item in result["results"])


def test_search_rejects_unknown_method(workspace: MemoryWorkspace, tmp_path: Path):
    with pytest.raises(TransactionError) as excinfo:
        inspect.search(tmp_path, "Pudong", method="magic")

    assert excinfo.value.code == "INVALID_ARGUMENT"


def test_search_does_not_persist_a_cache(workspace: MemoryWorkspace, tmp_path: Path):
    before = {path.name for path in tmp_path.iterdir()}

    inspect.search(tmp_path, "Pudong")

    assert {path.name for path in tmp_path.iterdir()} == before


def test_embedding_unavailable_never_falls_back(
    workspace: MemoryWorkspace, tmp_path: Path, monkeypatch
):
    def explode(*args, **kwargs):
        raise RuntimeError("sentence_transformers is not installed")

    monkeypatch.setattr(
        "src.retrieval.embedding.MemoryEmbeddingIndex.search", explode
    )

    with pytest.raises(TransactionError) as excinfo:
        inspect.search(tmp_path, "Pudong", method="embedding")

    assert excinfo.value.code == "EMBEDDING_UNAVAILABLE"
