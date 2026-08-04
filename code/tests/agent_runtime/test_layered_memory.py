"""Layered memory: several workspaces served as one, single layer unchanged."""

import asyncio
import json
from pathlib import Path

import pytest

from scriptorium import cli
from scriptorium.mcp_server import build_server
from src.retrieval import inspect
from src.retrieval.layers import Layer, LayeredMemory

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
def two_layers(tmp_path: Path):
    project = tmp_path / "project"
    global_ = tmp_path / "global"
    cli.main(["init", str(project)])
    cli.main(["init", str(global_)])
    return project, global_


@pytest.fixture
def server(two_layers):
    project, global_ = two_layers
    return build_server(
        [("project", project), ("global", global_)], git_commit="off"
    )


def seed(server, layer: str, topic: str, lines=TOPIC_LINES):
    revision = call(server, "memory_status", {})["revision"]
    result = call(server, "memory_update", {
        "base_revision": revision,
        "patch": patch_text(topic, lines),
        "sources": SOURCES,
        "layer": layer,
    })
    assert result["ok"] is True, result
    return result


# --- single layer must behave exactly as before layering existed ---

def test_single_layer_passthrough_matches_inspect(tmp_path: Path):
    cli.main(["init", str(tmp_path)])
    memory = LayeredMemory([Layer("memory", tmp_path)])

    assert memory.status() == inspect.status(tmp_path)
    assert memory.revision() == inspect.status(tmp_path)["revision"]
    listed = memory.list_files(prefix="")
    assert all("layer" not in entry for entry in listed["files"])
    assert ":" not in json.dumps([e["path"] for e in listed["files"]])


def test_colon_in_path_never_routes_to_a_layer(tmp_path: Path):
    cli.main(["init", str(tmp_path)])
    memory = LayeredMemory([Layer("memory", tmp_path)])
    layer, within = memory.split("topics/a:b.md")
    assert layer.name == "memory"
    assert within == "topics/a:b.md"


# --- layered reads ---

def test_status_combines_revisions(server):
    status = call(server, "memory_status", {})
    assert status["ok"] is True
    assert set(status["data"]["layers"]) == {"project", "global"}
    assert status["data"]["default_layer"] == "project"
    assert "project=" in status["revision"] and "global=" in status["revision"]


def test_reads_span_layers_with_qualified_paths(server):
    seed(server, "project", "topics/api.md")
    seed(server, "global", "topics/person.md")

    listed = call(server, "memory_list", {"prefix": "topics/"})
    paths = {entry["path"] for entry in listed["data"]["files"]}
    assert "project:topics/api.md" in paths
    assert "global:topics/person.md" in paths

    read = call(server, "memory_read", {"path": "global:topics/person.md"})
    assert read["ok"] is True
    assert read["data"]["layer"] == "global"
    assert "Pudong" in read["data"]["content"]

    grepped = call(server, "memory_grep", {"query": "Pudong"})
    layers = {match["layer"] for match in grepped["data"]["matches"]}
    assert layers == {"project", "global"}

    found = call(server, "memory_search", {"query": "Pudong"})
    assert {hit["layer"] for hit in found["data"]["results"]} == {
        "project", "global",
    }


def test_qualified_prefix_narrows_to_one_layer(server):
    seed(server, "project", "topics/api.md")
    seed(server, "global", "topics/person.md")

    listed = call(server, "memory_list", {"prefix": "global:topics/"})
    assert {entry["layer"] for entry in listed["data"]["files"]} == {"global"}


# --- layered writes ---

def test_write_lands_in_named_layer(server, two_layers):
    project, global_ = two_layers
    seed(server, "global", "topics/person.md")
    assert (global_ / "topics/person.md").is_file()
    assert not (project / "topics/person.md").exists()


def test_write_without_layer_lands_in_first_workspace(server, two_layers):
    project, _global = two_layers
    revision = call(server, "memory_status", {})["revision"]
    result = call(server, "memory_update", {
        "base_revision": revision,
        "patch": patch_text("topics/api.md", TOPIC_LINES),
        "sources": SOURCES,
    })
    assert result["ok"] is True
    assert result["data"]["layer"] == "project"
    assert (project / "topics/api.md").is_file()


def test_unknown_layer_is_an_error_envelope(server):
    revision = call(server, "memory_status", {})["revision"]
    result = call(server, "memory_update", {
        "base_revision": revision,
        "patch": patch_text("topics/api.md", TOPIC_LINES),
        "sources": SOURCES,
        "layer": "nope",
    })
    assert result["ok"] is False
    assert "no such layer" in result["error"]["message"]


def test_stale_combined_revision_is_rejected(server):
    stale = call(server, "memory_status", {})["revision"]
    seed(server, "project", "topics/api.md")

    result = call(server, "memory_update", {
        "base_revision": stale,
        "patch": patch_text("topics/other.md", [
            "# Other",
            "",
            "Another fact.[^new-evidence-x] ^new-block-x",
            "",
            "[^new-evidence-x]: Time: `2026-08-03`; Sources: new-source-move",
        ]),
        "sources": SOURCES,
        "layer": "project",
    })
    assert result["ok"] is False
    assert result["error"]["code"] == "CONCURRENT_UPDATE"


def test_write_to_one_layer_does_not_stale_the_other(server):
    """Editing the project must not lock the untouched global layer."""
    revision = call(server, "memory_status", {})["revision"]
    seed(server, "project", "topics/api.md")

    result = call(server, "memory_update", {
        "base_revision": revision,
        "patch": patch_text("topics/person.md", TOPIC_LINES),
        "sources": SOURCES,
        "layer": "global",
    })
    assert result["ok"] is True


# --- resilience ---

def test_broken_layer_costs_its_memory_not_the_session(two_layers):
    project, global_ = two_layers
    memory = LayeredMemory([
        Layer("project", project),
        Layer("global", global_ / "absent"),
    ])
    status = memory.status()
    assert "revision" in status["layers"]["project"]
    assert status["layers"]["global"] == {"error": "unreadable"}
    assert memory.list_files(prefix="topics/")["files"] == []


# --- CLI parsing ---

def test_parse_single_bare_workspace(tmp_path: Path):
    parsed = cli.parse_workspaces([str(tmp_path)])
    assert parsed == [("memory", tmp_path)]


def test_parse_named_layers_keep_order(tmp_path: Path):
    parsed = cli.parse_workspaces([
        f"project={tmp_path}/a", "global=~/memory",
    ])
    assert [name for name, _ in parsed] == ["project", "global"]
    assert parsed[1][1] == Path("~/memory").expanduser()


def test_parse_rejects_unnamed_among_several(tmp_path: Path):
    with pytest.raises(ValueError, match="NAME=PATH"):
        cli.parse_workspaces([f"project={tmp_path}", str(tmp_path)])
