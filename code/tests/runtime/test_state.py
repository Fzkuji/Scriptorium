import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.runtime.state import (
    RuntimeStateStore,
    SourceRecord,
    should_global_manage,
    should_incremental_write,
    should_local_reorganize,
)
from src.runtime.online import OnlineMemoryRuntime
from src.management import MemoryWorkspace


def test_source_record_preserves_opaque_provider_ids_and_order():
    records = [
        SourceRecord("claude", "thread_x", "msg_z9", 1, "user", "first"),
        SourceRecord("claude", "thread_x", "msg_a1", 2, "assistant", "second"),
    ]

    assert [record.message_id for record in sorted(records, key=lambda row: row.ordinal)] == [
        "msg_z9",
        "msg_a1",
    ]
    assert records[0].source_id == "claude/thread_x/msg_z9"


def test_runtime_state_is_atomic_and_cursor_advances_explicitly(tmp_path: Path):
    store = RuntimeStateStore(tmp_path)
    state = store.load()
    state.advance_cursor("thread_x", "msg_z9", ordinal=1)
    store.save(state)

    saved = json.loads((tmp_path / ".nativemem/runtime.json").read_text())
    assert saved["cursors"]["thread_x"] == {"message_id": "msg_z9", "ordinal": 1}
    assert not (tmp_path / ".nativemem/runtime.json.tmp").exists()


def test_runtime_trigger_decisions_are_deterministic():
    now = datetime(2026, 8, 1, 12, tzinfo=timezone.utc)
    assert should_incremental_write(1000, now, now, token_threshold=1000)
    assert should_incremental_write(1, now - timedelta(hours=1), now, token_threshold=1000)
    assert not should_incremental_write(1, now - timedelta(minutes=59), now, token_threshold=1000)
    assert should_local_reorganize(5, 0, batch_threshold=5, token_threshold=1000)
    assert should_local_reorganize(0, 1000, batch_threshold=5, token_threshold=1000)
    assert should_global_manage(now - timedelta(days=1), now, new_write_commits=1)
    assert not should_global_manage(now - timedelta(days=1), now, new_write_commits=0)
    assert not should_global_manage(now - timedelta(hours=23), now, new_write_commits=1)


def test_runtime_store_can_commit_memory_changes_to_git(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "topics").mkdir()
    (tmp_path / "topics/a.md").write_text("memory\n")
    store = RuntimeStateStore(tmp_path)

    commit = store.git_commit("NativeMem: incremental write")

    assert commit
    assert subprocess.check_output(["git", "log", "-1", "--format=%s"], cwd=tmp_path, text=True).strip() == "NativeMem: incremental write"


def test_workspace_archives_provider_records_with_stable_ids(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    record = SourceRecord("claude", "thread_x", "msg_z9", 1, "user", "hello")

    refs = workspace.archive_source_records([record])

    assert refs == ["claude/thread_x/msg_z9"]
    source = (tmp_path / "sources/claude/thread_x.md").read_text()
    assert "source-id:claude/thread_x/msg_z9" in source
    assert "hello" in source
    workspace.save_memory([{
        "when": "2026-08-01",
        "content": "The user said hello.",
        "refs": refs,
        "topic_path": "conversation.md",
        "headings": ["Conversation"],
    }])
    assert "sources/claude/thread_x.md#source-" in (
        tmp_path / "timeline/2026/08/01.md"
    ).read_text()


def test_workspace_archives_provider_refs_from_writer_sessions(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2026-08-01",
        "turns": [("user", "hello")],
        "refs": ["locomo/thread_a1/msg_b2"],
    }])

    source = tmp_path / "sources/locomo/thread_a1.md"
    assert source.exists()
    assert "source-id:locomo/thread_a1/msg_b2" in source.read_text()


def test_online_runtime_captures_batch_and_leaves_new_messages_for_next_run(tmp_path: Path):
    records = [SourceRecord("claude", "thread_x", "msg_1", 1, "user", "one")]
    batches = []

    def writer(_workspace, batch):
        batches.append([record.message_id for record in batch])
        records.append(SourceRecord("claude", "thread_x", "msg_2", 2, "user", "two"))

    runtime = OnlineMemoryRuntime(tmp_path, token_counter=lambda text: len(text.split()))
    assert runtime.process(records, writer, force=True)
    assert batches == [["msg_1"]]
    assert runtime.pending(records)[0].message_id == "msg_2"


def test_online_runtime_runs_local_and_daily_management_only_after_writes(tmp_path: Path):
    now = datetime(2026, 8, 1, 12, tzinfo=timezone.utc)
    events = []
    runtime = OnlineMemoryRuntime(
        tmp_path,
        token_counter=lambda _text: 10,
        local_batch_threshold=1,
        local_token_threshold=100,
    )
    records = [SourceRecord("claude", "thread_x", "msg_1", 1, "user", "one")]

    runtime.process(
        records,
        lambda _workspace, _batch: events.append("write"),
        local_manager=lambda _workspace: events.append("local"),
        global_manager=lambda _workspace: events.append("global"),
        now=now,
        force=True,
    )
    runtime.process(
        records,
        lambda *_args: events.append("unexpected"),
        local_manager=lambda _workspace: events.append("unexpected-local"),
        global_manager=lambda _workspace: events.append("unexpected-global"),
        now=now + timedelta(days=1),
        force=True,
    )

    assert events == ["write", "local", "global"]


def test_online_runtime_does_not_advance_cursor_when_git_commit_fails(tmp_path: Path, monkeypatch):
    runtime = OnlineMemoryRuntime(tmp_path, token_counter=lambda _text: 1)
    records = [SourceRecord("claude", "thread_x", "msg_1", 1, "user", "one")]
    monkeypatch.setattr(runtime.store, "git_commit", lambda _message: (_ for _ in ()).throw(RuntimeError("commit failed")))

    try:
        runtime.process(records, lambda _workspace, _batch: None, force=True)
    except RuntimeError as exc:
        assert str(exc) == "commit failed"
    else:
        raise AssertionError("commit failure must propagate")

    assert runtime.store.load().cursors == {}
