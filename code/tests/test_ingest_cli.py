"""`scriptorium ingest` writes a transcript into memory, once it is worth it."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from memory.agent_runtime import AgentResult
from memory.workspace.transaction import workspace_revision, workspace_write_lock
from scriptorium.cli import main


class RecordingAgent:
    """Runs the tool calls a writer would make, and counts the calls."""

    def __init__(self, tool_calls=()):
        self.tool_calls = list(tool_calls)
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return AgentResult(
            text="done", structured_output=None, num_turns=1,
            input_tokens=10, output_tokens=5,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
            anthropic_equivalent_cost_usd=0.001,
            duration_ms=20, duration_api_ms=15,
            stop_reason="end_turn", session_id="test-session",
        )


@pytest.fixture
def agent(monkeypatch):
    """Stand in for the Claude Code process, and capture how it was built."""
    made = RecordingAgent()
    made.configs = []

    def build(config, **_kwargs):
        made.configs.append(config)
        return made

    monkeypatch.setattr("memory.agent_runtime.ClaudeCodeAgent", build)
    return made


def transcript(path: Path, turns: int = 1, words: int = 3) -> Path:
    # Stamped now, not at a fixed date: writing also fires when the last
    # turn is over an hour old, so a hardcoded timestamp makes every
    # "below the threshold" assertion pass only on the day it was written.
    stamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    rows = [{
        "type": "user",
        "uuid": f"u{index}",
        "sessionId": "s1",
        "timestamp": stamp,
        "origin": {"kind": "human"},
        "message": {"content": " ".join(["shanghai"] * words)},
    } for index in range(turns)]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    return path


def test_a_short_conversation_costs_nothing(tmp_path: Path, agent):
    source = transcript(tmp_path / "s.jsonl")

    code = main([
        "ingest", "--transcript", str(source),
        "--workspace", str(tmp_path / "memory"),
    ])

    assert code == 0
    assert agent.calls == []


def test_crossing_the_threshold_writes(tmp_path: Path, agent):
    source = transcript(tmp_path / "s.jsonl", turns=4, words=50)

    code = main([
        "ingest", "--transcript", str(source),
        "--workspace", str(tmp_path / "memory"),
        "--token-threshold", "10",
    ])

    assert code == 0
    assert len(agent.calls) == 1


def test_a_drained_transcript_is_not_written_again(tmp_path: Path, agent):
    """The cursor is what stops a turn being recorded twice."""
    source = transcript(tmp_path / "s.jsonl", turns=4, words=50)
    argv = [
        "ingest", "--transcript", str(source),
        "--workspace", str(tmp_path / "memory"), "--token-threshold", "10",
    ]

    # Each run takes one batch, so drain before checking it stays drained.
    for _ in range(10):
        main(argv)
    drained = len(agent.calls)
    main(argv)

    assert len(agent.calls) == drained


def test_a_busy_workspace_is_left_alone(tmp_path: Path, agent):
    # Another session holding the lock is ordinary, not an error: this runs
    # again after the next turn.
    workspace = tmp_path / "memory"
    workspace.mkdir()
    source = transcript(tmp_path / "s.jsonl", turns=4, words=50)

    with workspace_write_lock(workspace):
        code = main([
            "ingest", "--transcript", str(source),
            "--workspace", str(workspace), "--token-threshold", "10",
        ])

    assert code == 0
    assert agent.calls == []


def test_writing_runs_as_the_user_not_as_a_tenant(tmp_path: Path, agent):
    source = transcript(tmp_path / "s.jsonl", turns=4, words=50)

    main([
        "ingest", "--transcript", str(source),
        "--workspace", str(tmp_path / "memory"), "--token-threshold", "10",
    ])

    config = agent.configs[0]
    assert config.inherit_auth is True
    assert config.api_key == ""


def test_a_missing_transcript_is_a_usage_error(tmp_path: Path, agent):
    code = main([
        "ingest", "--transcript", str(tmp_path / "absent.jsonl"),
        "--workspace", str(tmp_path / "memory"),
    ])

    assert code == 2
    assert agent.calls == []


def test_a_failed_write_writes_no_topics_and_is_retried(
    tmp_path: Path, monkeypatch
):
    """Evidence is archived before the topics citing it, by design.

    So a writer that dies leaves sources on disk with nothing citing them.
    What must hold is that no half-written topic appears, and that the
    cursor stays put so the same turns are attempted again.
    """
    class Failing:
        def run(self, **_kwargs):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(
        "memory.agent_runtime.ClaudeCodeAgent", lambda *a, **k: Failing()
    )
    workspace = tmp_path / "memory"
    source = transcript(tmp_path / "s.jsonl", turns=4, words=50)
    argv = [
        "ingest", "--transcript", str(source),
        "--workspace", str(workspace), "--token-threshold", "10",
    ]

    with pytest.raises(RuntimeError):
        main(argv)

    assert list((workspace / "topics").rglob("*.md")) == []
    state = json.loads(
        (workspace / ".scriptorium" / "runtime.json").read_text()
    )
    assert not state.get("cursors")

    # Retrying re-archives the same evidence without duplicating it.
    archived = (workspace / "sources").rglob("*.md")
    before = {path: path.read_text() for path in archived}
    with pytest.raises(RuntimeError):
        main(argv)
    assert {path: path.read_text() for path in before} == before


def test_a_long_backlog_is_written_in_bounded_batches(tmp_path: Path, agent):
    """One agent call must not receive a whole day of conversation.

    The threshold says when writing is worth doing, not how much to write.
    A session that ran all day arrives with far more than one call can hold.
    """
    source = transcript(tmp_path / "s.jsonl", turns=40, words=40)
    workspace = str(tmp_path / "memory")
    argv = [
        "ingest", "--transcript", str(source),
        "--workspace", workspace, "--token-threshold", "50",
    ]

    main(argv)
    first = agent.calls[-1]["prompt"]
    main(argv)
    second = agent.calls[-1]["prompt"]

    assert len(agent.calls) == 2
    # Each call carried a slice, and the slices differ: the cursor moved.
    assert first != second
    assert first.count("shanghai") < 40 * 40


def test_enough_accumulated_writing_triggers_reorganisation(
    tmp_path: Path, agent, monkeypatch
):
    """Reorganising is the reason this runs in the background at all.

    Left alone, topic files only grow: one 34 KB file with the timeline cut
    up by subject is what a memory looks like when nobody tidies it.
    """
    fired = []
    monkeypatch.setattr(
        "memory.management.organize_topics",
        lambda memory_dir, **kwargs: fired.append(memory_dir) or [],
    )
    source = transcript(tmp_path / "s.jsonl", turns=200, words=300)
    argv = [
        "ingest", "--transcript", str(source),
        "--workspace", str(tmp_path / "memory"),
    ]

    for _ in range(12):
        main(argv)

    assert fired, "accumulated writes never triggered a reorganisation"
