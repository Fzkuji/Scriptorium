import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.v11_memory import (
    MANAGER_TASK,
    MemoryWorkspace,
    WRITER_BATCH_TASK,
    WRITER_TASK,
    _chat_completion_with_retry,
    _compact_tool_history,
    _run_agent,
    manage_memory,
    verify_session,
)
from src.adapters import run_nativemem
from src.nativemem import execute_tool


def test_memory_workspace_runs_normal_shell_commands_from_memory_dir(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)

    result = workspace.shell(
        "mkdir -p topics/aquarium && "
        "printf '%s\\n' '# Tanks' '###### Acquisition history' 'three tanks' "
        "> topics/aquarium/tanks.md && "
        "find topics -type f -print"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "topics/aquarium/tanks.md"
    assert "###### Acquisition history" in (
        workspace.stage_dir / "topics/aquarium/tanks.md"
    ).read_text()
    assert (tmp_path / "topics/aquarium/tanks.md").exists()


def test_save_memory_normalizes_topic_root_prefix(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2023-05-20",
        "turns": [("user", "I started yoga.")],
        "refs": ["D1:1"],
    }])
    workspace.save_memory([{
        "when": "2023-05-20",
        "content": "The user started yoga.",
        "refs": ["D1:1"],
        "topic_path": "topics/health/yoga.md",
        "headings": ["Health", "Yoga"],
    }])

    assert (workspace.stage_dir / "topics/health/yoga.md").exists()
    assert not (workspace.stage_dir / "topics/topics").exists()


def test_append_event_reuses_existing_heading_prefix(tmp_path: Path):
    path = tmp_path / "topics/projects/budget-tracker.md"
    base = {
        "when": "2024-03-15",
        "refs": ["D1:1"],
        "topic_path": "projects/budget-tracker.md",
    }
    MemoryWorkspace._append_event(path, {
        **base,
        "event_id": "ev_1111111111111111",
        "content": "Initial architecture.",
        "headings": ["Architecture", "Initial architecture"],
    })
    MemoryWorkspace._append_event(path, {
        **base,
        "event_id": "ev_2222222222222222",
        "content": "Application modules.",
        "headings": ["architecture", "Application modules"],
    })

    text = path.read_text()
    assert len(re.findall(r"(?mi)^#\s+architecture\s*$", text)) == 1
    assert text.count("## Initial architecture\n") == 1
    assert text.count("## Application modules\n") == 1


def test_manager_prompt_stays_general():
    assert "Inspect every topic directory and file" not in MANAGER_TASK
    assert "only" not in MANAGER_TASK
    assert "commit" not in MANAGER_TASK


def test_manager_keeps_general_memory_tools(tmp_path):
    captured = {}
    message = SimpleNamespace(content="done", tool_calls=[])

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)
    ))

    manage_memory(tmp_path, client=client, model="test")

    names = [tool["function"]["name"] for tool in captured["tools"]]
    assert names == ["shell", "save_memory"]


def test_writer_prompts_use_supplied_structure_without_reading_restrictions():
    assert "Review the supplied workspace structure" in WRITER_TASK
    assert "Review the supplied workspace structure" in WRITER_BATCH_TASK
    assert "only when needed" not in WRITER_TASK + WRITER_BATCH_TASK
    assert "promptly" not in WRITER_TASK + WRITER_BATCH_TASK


def test_workspace_structure_lists_all_shell_visible_memory_views(tmp_path):
    for relative in ("topics/home.md", "timeline/2026.md", "sources/D1.md"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    (tmp_path / "recent_events.jsonl").write_text("{}\n", encoding="utf-8")

    structure = MemoryWorkspace(tmp_path).structure()

    assert structure.splitlines() == [
        "recent_events.jsonl",
        "sources/D1.md",
        "timeline/2026.md",
        "topics/home.md",
    ]


def test_shell_can_read_all_memory_views(tmp_path):
    for relative in ("topics/home.md", "timeline/2026.md", "sources/D1.md"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    (tmp_path / "recent_events.jsonl").write_text("recent", encoding="utf-8")
    workspace = MemoryWorkspace(tmp_path)

    result = workspace.shell(
        "cat topics/home.md timeline/2026.md sources/D1.md recent_events.jsonl"
    )

    assert result.returncode == 0
    assert result.stdout == (
        "topics/home.mdtimeline/2026.mdsources/D1.mdrecent"
    )


def test_agent_needs_no_final_persistence_action_when_model_finishes(tmp_path):
    message = SimpleNamespace(content="done", tool_calls=[])
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)], usage=None
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_kwargs: response
    )))

    audit = _run_agent(tmp_path, client=client, model="test", task="organize")

    assert audit == []


def test_agent_keeps_successful_tool_changes_when_round_limit_is_reached(tmp_path):
    call = SimpleNamespace(
        id="shell",
        function=SimpleNamespace(name="shell", arguments='{"command":"true"}'),
    )
    message = SimpleNamespace(content=None, tool_calls=[call])
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)], usage=None
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_kwargs: response
    )))

    audit = _run_agent(tmp_path, client=client, model="test", task="organize")

    assert len(audit) == 40
    assert all(record["status"] == "ok" for record in audit)


def test_save_memory_commits_linked_views(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2023-05-23",
        "turns": [("user", "I bought a quarantine tank.")],
        "refs": ["D18:11"],
    }])

    saved = workspace.save_memory([
        {
            "when": "2023-05-23",
            "content": "User bought a quarantine tank.",
            "refs": ["D18:11"],
            "topic_path": "pets/aquarium.md",
            "headings": ["Pets", "Aquarium"],
        }
    ])
    workspace._synchronize()

    topic = (tmp_path / "topics" / "pets" / "aquarium.md").read_text()
    timeline = (tmp_path / "timeline" / "2023" / "05" / "23.md").read_text()
    recent = json.loads((tmp_path / "recent_events.jsonl").read_text())
    source = (tmp_path / "sources" / "D18.md").read_text()

    assert saved == "saved 1 event"
    assert "<!-- memory-event:" in topic
    event_id = recent["event_id"]
    assert event_id in topic and event_id in timeline
    assert "timeline/2023/05/23.md" in topic
    assert "topics/pets/aquarium.md" in timeline
    assert "sources/D18.md#d18-11" in topic
    assert '<a id="d18-11"></a>' in source
    assert recent["topic_path"] == "topics/pets/aquarium.md"
    assert recent["headings"] == ["Pets", "Aquarium"]


def test_save_memory_synchronizes_all_views_immediately(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2023-05-23",
        "turns": [("user", "I bought a quarantine tank.")],
        "refs": ["D18:11"],
    }])

    workspace.save_memory([{
        "when": "2023-05-23",
        "content": "User bought a quarantine tank.",
        "refs": ["D18:11"],
        "topic_path": "pets/aquarium.md",
        "headings": ["Pets", "Aquarium"],
    }])

    assert (tmp_path / "topics/pets/aquarium.md").exists()
    assert (tmp_path / "timeline/2023/05/23.md").exists()
    assert (tmp_path / "recent_events.jsonl").exists()


def test_save_memory_rejects_non_source_refs(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)

    with pytest.raises(ValueError, match="complete source references"):
        workspace.save_memory([{
            "when": "2023-05-23",
            "content": "User bought a tank.",
            "refs": ["Session 1", "D18:1-D18:2"],
            "topic_path": "pets.md",
            "headings": ["Pets"],
        }])


def test_save_memory_expands_compact_source_ranges(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2023-05-23",
        "turns": [("user", "one"), ("assistant", "two")],
        "refs": ["D18:1", "D18:2"],
    }])

    workspace.save_memory([{
        "when": "2023-05-23",
        "content": "A two-turn event.",
        "refs": ["D18:1-D18:2"],
        "topic_path": "event.md",
        "headings": ["Event"],
    }])
    workspace._synchronize()

    recent = json.loads((tmp_path / "recent_events.jsonl").read_text())
    assert recent["refs"] == ["D18:1", "D18:2"]


def test_commit_repairs_links_after_shell_reorganizes_topics(tmp_path: Path):
    first = MemoryWorkspace(tmp_path)
    first.archive_sessions([{
        "observation_date": "2023-05-23",
        "turns": [("user", "I bought a quarantine tank.")],
        "refs": ["D18:11"],
    }])
    first.save_memory([{
        "when": "2023-05-23",
        "content": "User bought a quarantine tank.",
        "refs": ["D18:11"],
        "topic_path": "pets/aquarium.md",
        "headings": ["Pets", "Aquarium"],
    }])
    first._synchronize()

    organizer = MemoryWorkspace(tmp_path)
    result = organizer.shell(
        "mkdir -p topics/life && mv topics/pets/aquarium.md topics/life/pets.md"
    )
    assert result.returncode == 0
    organizer._synchronize()

    timeline = (tmp_path / "timeline/2023/05/23.md").read_text()
    recent = json.loads((tmp_path / "recent_events.jsonl").read_text())
    topic = (tmp_path / "topics/life/pets.md").read_text()
    assert "topics/life/pets.md" in timeline
    assert "topics/pets/aquarium.md" not in timeline
    assert recent["topic_path"] == "topics/life/pets.md"
    assert "../../timeline/2023/05/23.md" in topic


def test_shell_move_synchronizes_paths_and_links_immediately(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2023-05-23",
        "turns": [("user", "I bought a quarantine tank.")],
        "refs": ["D18:11"],
    }])
    workspace.save_memory([{
        "when": "2023-05-23",
        "content": "User bought a quarantine tank.",
        "refs": ["D18:11"],
        "topic_path": "pets/aquarium.md",
        "headings": ["Pets", "Aquarium"],
    }])

    result = workspace.shell(
        "mkdir -p topics/life && mv topics/pets/aquarium.md topics/life/pets.md"
    )

    assert result.returncode == 0
    assert (tmp_path / "topics/life/pets.md").exists()
    recent = json.loads((tmp_path / "recent_events.jsonl").read_text())
    assert recent["topic_path"] == "topics/life/pets.md"


def test_shell_revision_synchronizes_event_content_to_all_views(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2023-05-23",
        "turns": [("user", "I wake up at 6:30 AM.")],
        "refs": ["D1:1"],
    }])
    workspace.save_memory([{
        "when": "2023-05-23",
        "content": "The user wakes up at 7:00 AM.",
        "refs": ["D1:1"],
        "topic_path": "routines/daily.md",
        "headings": ["Daily routine", "Morning"],
    }])

    result = workspace.shell(
        "sed -i '' 's/wakes up at 7:00 AM/wakes up at 6:30 AM/' "
        "topics/routines/daily.md"
    )

    assert result.returncode == 0
    topic = (tmp_path / "topics/routines/daily.md").read_text()
    timeline = (tmp_path / "timeline/2023/05/23.md").read_text()
    recent = json.loads((tmp_path / "recent_events.jsonl").read_text())
    assert "wakes up at 6:30 AM" in topic
    assert "wakes up at 6:30 AM" in timeline
    assert recent["content"] == "The user wakes up at 6:30 AM."


def test_read_only_shell_hides_source_archive(tmp_path: Path):
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources/D1.md").write_text("secret source\n")
    (tmp_path / "topics").mkdir()
    (tmp_path / "topics/memory.md").write_text("public memory\n")

    output = execute_tool("bash", {"command": "find . -type f -print"}, tmp_path, hide_raw=True)

    assert "topics/memory.md" in output
    assert "sources" not in output


def test_chat_completion_retries_rate_limit_and_honors_retry_after(monkeypatch):
    waits = []

    class RateLimited(Exception):
        status_code = 429

        def __init__(self):
            self.response = type(
                "Response", (), {"headers": {"retry-after": "3"}}
            )()

    attempts = iter([RateLimited(), "ok"])

    def create(**_kwargs):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("src.v11_memory.time.sleep", waits.append)

    result = _chat_completion_with_retry(create, model="gpt-5.5")

    assert result == "ok"
    assert waits == [3.0]


def test_chat_completion_keeps_retrying_transient_connection_errors(monkeypatch):
    waits = []
    attempts = 0

    def create(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 10:
            raise ConnectionError("network unavailable")
        return "ok"

    monkeypatch.setattr("src.v11_memory.time.sleep", waits.append)

    assert _chat_completion_with_retry(create, model="gpt-5.5") == "ok"
    assert attempts == 10
    assert waits


def test_chat_completion_does_not_retry_permanent_http_error(monkeypatch):
    attempts = 0

    class BadRequest(Exception):
        status_code = 400

    def create(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise BadRequest("invalid request")

    monkeypatch.setattr(
        "src.v11_memory.time.sleep",
        lambda _delay: pytest.fail("permanent error must not sleep"),
    )

    with pytest.raises(BadRequest):
        _chat_completion_with_retry(create, model="gpt-5.5")
    assert attempts == 1


def test_chat_completion_retries_frontier_temporary_quota_error(monkeypatch):
    waits = []
    attempts = 0

    class TemporaryQuota(Exception):
        status_code = 403

    def create(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TemporaryQuota("insufficient_user_quota: 用户额度不足")
        return "ok"

    monkeypatch.setattr("src.v11_memory.time.sleep", waits.append)

    assert _chat_completion_with_retry(create, model="gpt-5.5") == "ok"
    assert waits == [30]


def test_v11_build_verifies_each_session_before_next_write(
    tmp_path: Path, monkeypatch
):
    calls = []
    conv = {
        "session_1": [("user", "first")],
        "session_1_date_time": "2023-05-01",
        "session_2": [("user", "second")],
        "session_2_date_time": "2023-05-02",
    }

    monkeypatch.setattr(
        run_nativemem,
        "split_into_chunks_structured",
        lambda session, _size: [(session, ["D1:1"])],
    )
    monkeypatch.setattr(
        run_nativemem.v11_memory,
        "write_sessions",
        lambda *args, **kwargs: calls.append(
            ("write", kwargs["sessions"][0]["observation_date"])
        ) or [{"tool": "save_memory", "count": 1}],
    )
    monkeypatch.setattr(
        run_nativemem.v11_memory,
        "verify_session",
        lambda *args, **kwargs: calls.append(
            ("verify", kwargs["observation_date"])
        ) or {"repaired": False},
    )
    monkeypatch.setattr(
        run_nativemem.v11_memory,
        "manage_memory",
        lambda *args, **kwargs: calls.append(("manage", None)) or [],
    )

    _, events = run_nativemem._build_memory_v11(conv, tmp_path)

    assert events == 2
    assert calls == [
        ("write", "2023-05-01"),
        ("verify", "2023-05-01"),
        ("write", "2023-05-02"),
        ("verify", "2023-05-02"),
        ("manage", None),
    ]


def test_compact_tool_history_keeps_latest_outputs_only():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": None},
        {"role": "tool", "tool_call_id": "old", "content": "x" * 5000},
        {"role": "assistant", "content": None},
        {"role": "tool", "tool_call_id": "new", "content": "y" * 5000},
    ]

    _compact_tool_history(messages)

    assert messages[2]["content"] == "[previous tool output omitted]"
    assert messages[4]["content"] == "y" * 5000


def test_compact_tool_history_accepts_sdk_message_objects():
    messages = [
        {"role": "user", "content": "task"},
        SimpleNamespace(role="assistant", content=None),
        {"role": "tool", "tool_call_id": "old", "content": "x" * 5000},
        SimpleNamespace(role="assistant", content=None),
        {"role": "tool", "tool_call_id": "new", "content": "y" * 5000},
    ]

    _compact_tool_history(messages)

    assert messages[2]["content"] == "[previous tool output omitted]"
    assert messages[4]["content"] == "y" * 5000


def test_v11_agent_passes_configured_reasoning_effort(tmp_path, monkeypatch):
    captured = {}
    responses = iter([
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="done", tool_calls=[]
        ))]),
    ])

    def create(**kwargs):
        captured.update(kwargs)
        return next(responses)

    monkeypatch.setenv("NATIVEMEM_REASONING_EFFORT", "none")
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)
    ))

    from src.v11_memory import write_session

    write_session(
        tmp_path,
        client=client,
        model="gpt-5.5",
        observation_date="2023-05-23",
        turns=[("user", "hello")],
        refs=["D1:1"],
    )

    assert captured["reasoning_effort"] == "none"


def test_v11_agent_can_disable_provider_thinking(tmp_path, monkeypatch):
    captured = {}
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="done", tool_calls=[]
    ))], usage=None)

    def create(**kwargs):
        captured.update(kwargs)
        return response

    monkeypatch.setenv("NATIVEMEM_THINKING", "disabled")
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)
    ))

    _run_agent(tmp_path, client=client, model="test", task="organize")

    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_v11_agent_can_correct_invalid_save_event_refs(tmp_path):
    invalid = SimpleNamespace(
        id="invalid",
        function=SimpleNamespace(
            name="save_memory",
            arguments=json.dumps({"events": [{
                "when": "2023-05-23",
                "content": "User bought a tank.",
                "refs": ["Session 1"],
                "topic_path": "pets.md",
                "headings": ["Pets"],
            }]}),
        ),
    )
    valid = SimpleNamespace(
        id="valid",
        function=SimpleNamespace(
            name="save_memory",
            arguments=json.dumps({"events": [{
                "when": "2023-05-23",
                "content": "User bought a tank.",
                "refs": ["D1:1"],
                "topic_path": "pets.md",
                "headings": ["Pets"],
            }]}),
        ),
    )
    responses = iter([
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=None, tool_calls=[invalid]
        ))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=None, tool_calls=[valid]
        ))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="done", tool_calls=[]
        ))]),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_kwargs: next(responses)
    )))

    from src.v11_memory import write_session

    audit = write_session(
        tmp_path,
        client=client,
        model="gpt-5.5",
        observation_date="2023-05-23",
        turns=[("user", "I bought a tank")],
        refs=["D1:1"],
    )

    assert [record["status"] for record in audit] == ["error", "ok"]
    assert audit[-1]["tool"] == "save_memory"
    assert json.loads((tmp_path / "recent_events.jsonl").read_text())["refs"] == ["D1:1"]


def test_v11_agent_recovers_from_malformed_tool_arguments(tmp_path):
    malformed = SimpleNamespace(
        id="bad-json",
        function=SimpleNamespace(name="shell", arguments='{"command":"ls'),
    )
    responses = iter([
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=None, tool_calls=[malformed]
        ))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="done", tool_calls=[]
        ))]),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_kwargs: next(responses)
    )))

    audit = _run_agent(tmp_path, client=client, model="test", task="organize")

    assert audit == [{
        "round": 0,
        "tool": "shell",
        "status": "error",
        "output": "Tool error: invalid JSON arguments",
    }]


def test_verify_session_does_not_repair_when_memory_answers_probe(
    tmp_path, monkeypatch
):
    source_workspace = MemoryWorkspace(tmp_path)
    source_workspace.archive_sessions([{
        "observation_date": "2023-05-23",
        "turns": [("user", "I wake up at 6:30 AM.")],
        "refs": ["D1:1"],
    }])
    calls = []
    responses = iter([
        '{"question":"What time does the user wake up?",'
        '"expected_answer":"6:30 AM","refs":["D1:1"]}',
        "<answer>6:30 AM</answer>",
        '{"supported":true}',
    ])

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=next(responses), tool_calls=[]
            ))],
            usage=None,
        )

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)
    ))
    monkeypatch.setenv("NATIVEMEM_REASONING_EFFORT", "none")
    monkeypatch.setenv("NATIVEMEM_THINKING", "disabled")
    before = list(tmp_path.rglob("*"))

    result = verify_session(
        tmp_path,
        client=client,
        model="test",
        observation_date="2023-05-23",
        turns=[("user", "I wake up at 6:30 AM.")],
        refs=["D1:1"],
    )

    assert result["repaired"] is False
    assert result["initial"]["answer"] == "6:30 AM"
    assert result["post_repair"] is None
    assert list(tmp_path.rglob("*")) == before
    assert "6:30 AM" not in calls[1]["messages"][1]["content"]
    assert all(call["reasoning_effort"] == "none" for call in calls)
    assert all(
        call["extra_body"] == {"thinking": {"type": "disabled"}}
        for call in calls
    )


def test_verify_session_repairs_then_retries_same_question(tmp_path):
    source_workspace = MemoryWorkspace(tmp_path)
    source_workspace.archive_sessions([{
        "observation_date": "2023-05-23",
        "turns": [("user", "I wake up at 6:30 AM.")],
        "refs": ["D1:1"],
    }])
    save = SimpleNamespace(
        id="save",
        function=SimpleNamespace(
            name="save_memory",
            arguments=json.dumps({"events": [{
                "when": "2023-05-23",
                "content": "The user wakes up at 6:30 AM.",
                "refs": ["D1:1"],
                "topic_path": "routines/daily.md",
                "headings": ["Daily routine", "Morning"],
            }]}),
        ),
    )
    responses = iter([
        '{"question":"What time does the user wake up?",'
        '"expected_answer":"6:30 AM","refs":["D1:1"]}',
        "<answer>No information available.</answer>",
        '{"supported":false}',
        SimpleNamespace(content=None, tool_calls=[save]),
        SimpleNamespace(content="done", tool_calls=[]),
        "<answer>6:30 AM</answer>",
        '{"supported":true}',
    ])

    def create(**_kwargs):
        value = next(responses)
        message = (
            value if isinstance(value, SimpleNamespace)
            else SimpleNamespace(content=value, tool_calls=[])
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=None,
        )

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)
    ))

    result = verify_session(
        tmp_path,
        client=client,
        model="test",
        observation_date="2023-05-23",
        turns=[("user", "I wake up at 6:30 AM.")],
        refs=["D1:1"],
    )

    assert result["repaired"] is True
    assert result["initial"]["supported"] is False
    assert result["post_repair"]["supported"] is True
    assert result["post_repair"]["question"] == result["probe"]["question"]
    assert (tmp_path / "topics/routines/daily.md").exists()
    assert (tmp_path / "timeline/2023/05/23.md").exists()
