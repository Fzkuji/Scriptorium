"""CLI and MCP adapter for the interactive path."""

import asyncio
import json
from pathlib import Path

import pytest

from scriptorium import cli
from scriptorium.mcp_server import TOOL_NAMES, build_server
from src.management import MemoryWorkspace

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


def patch_text(path: str, lines: list[str]) -> str:
    body = "".join(f"+{line}\n" for line in lines)
    return f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n{body}"


def call(server, name: str, arguments: dict) -> dict:
    _content, payload = asyncio.run(server.call_tool(name, arguments))
    raw = payload["result"] if isinstance(payload, dict) else payload
    return json.loads(raw)


@pytest.fixture
def server(tmp_path: Path):
    cli.main(["init", str(tmp_path)])
    return build_server(tmp_path, git_commit="off")


def test_init_creates_workspace(tmp_path: Path, capsys):
    target = tmp_path / "memory"

    assert cli.main(["init", str(target)]) == 0

    assert (target / "topics").is_dir()
    assert (target / "core.md").is_file()
    assert "initialized workspace" in capsys.readouterr().out


def test_init_does_not_overwrite_existing_memory(tmp_path: Path, capsys):
    cli.main(["init", str(tmp_path)])
    (tmp_path / "core.md").write_text("# Keep me\n")

    assert cli.main(["init", str(tmp_path)]) == 0

    assert (tmp_path / "core.md").read_text() == "# Keep me\n"
    assert "already initialized" in capsys.readouterr().out


def test_validate_reports_status_without_writing(tmp_path: Path, capsys):
    cli.main(["init", str(tmp_path)])
    workspace = MemoryWorkspace(tmp_path)
    workspace.update(
        base_revision=workspace.revision(),
        patch=patch_text("topics/personal/residence.md", TOPIC_LINES),
        sources=SOURCES,
    )
    before = workspace.revision()
    capsys.readouterr()

    assert cli.main(["validate", "--workspace", str(tmp_path)]) == 0

    assert workspace.revision() == before
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_validate_reports_broken_topic(tmp_path: Path, capsys):
    cli.main(["init", str(tmp_path)])
    (tmp_path / "topics" / "broken.md").write_text(
        "# Broken\n\nDangling.[^missing] ^abcd1234\n"
    )

    assert cli.main(["validate", "--workspace", str(tmp_path)]) == 1

    assert json.loads(capsys.readouterr().err)["ok"] is False


def test_server_exposes_exactly_the_documented_tools(server):
    listed = {tool.name for tool in asyncio.run(server.list_tools())}

    assert listed == set(TOOL_NAMES)
    # A user-facing server must never hand out shell access.
    assert not any("shell" in name or "bash" in name for name in listed)


def test_status_returns_revision_used_by_update(server, tmp_path: Path):
    status = call(server, "memory_status", {})

    assert status["ok"] is True
    assert status["data"]["revision"] == status["revision"]


def test_update_commits_source_and_topic(server, tmp_path: Path):
    revision = call(server, "memory_status", {})["revision"]

    result = call(server, "memory_update", {
        "base_revision": revision,
        "patch": patch_text("topics/personal/residence.md", TOPIC_LINES),
        "sources": SOURCES,
    })

    assert result["ok"] is True
    assert result["data"]["block_ids"]["new-block-residence"]
    assert result["data"]["git_committed"] is False
    assert (tmp_path / "topics/personal/residence.md").is_file()
    # Derived views must be rebuilt by the same transaction.
    assert "recent_events.jsonl" in result["data"]["changed_files"]


def test_update_with_stale_revision_returns_error_envelope(server):
    stale = call(server, "memory_status", {})["revision"]
    call(server, "memory_update", {
        "base_revision": stale,
        "patch": patch_text("topics/personal/residence.md", TOPIC_LINES),
        "sources": SOURCES,
    })

    second = call(server, "memory_update", {
        "base_revision": stale,
        "patch": patch_text("topics/other.md", [
            "# Other",
            "",
            "Another fact.[^new-evidence-x] ^new-block-x",
            "",
            "[^new-evidence-x]: Time: `2026-08-03`; Sources: new-source-move",
        ]),
        "sources": SOURCES,
    })

    assert second["ok"] is False
    assert second["error"]["code"] == "CONCURRENT_UPDATE"


def test_write_to_derived_path_is_refused(server):
    revision = call(server, "memory_status", {})["revision"]

    result = call(server, "memory_update", {
        "base_revision": revision,
        "patch": patch_text("relations.json", ["{}"]),
    })

    assert result["ok"] is False
    assert result["error"]["code"] == "READ_ONLY_PATH"


def test_errors_never_leak_stage_paths_or_tracebacks(server):
    result = call(server, "memory_read", {"path": "/etc/passwd"})

    assert result["ok"] is False
    body = json.dumps(result)
    assert "Traceback" not in body
    assert "nativemem-topics-" not in body


def test_read_and_search_round_trip(server, tmp_path: Path):
    revision = call(server, "memory_status", {})["revision"]
    call(server, "memory_update", {
        "base_revision": revision,
        "patch": patch_text("topics/personal/residence.md", TOPIC_LINES),
        "sources": SOURCES,
    })

    found = call(server, "memory_search", {"query": "Pudong"})
    assert found["ok"] is True
    assert found["data"]["results"]

    read = call(server, "memory_read", {
        "path": "topics/personal/residence.md"
    })
    assert "Pudong" in read["data"]["content"]

    grepped = call(server, "memory_grep", {
        "query": "Pudong", "prefix": "topics/"
    })
    assert grepped["data"]["total"] >= 1

    listed = call(server, "memory_list", {"prefix": "topics/"})
    assert any(
        entry["path"] == "topics/personal/residence.md"
        for entry in listed["data"]["files"]
    )


def test_build_server_rejects_missing_workspace(tmp_path: Path):
    with pytest.raises(ValueError, match="scriptorium init"):
        build_server(tmp_path / "absent")
