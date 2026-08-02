import asyncio
import json
import os
import re
from pathlib import Path

import pytest

from src.agent_runtime import AgentResult
from src.management import (
    MemoryWorkspace,
    _run_agent,
    manage_memory,
    organize_topics,
    verify_session,
)
from src import build as adapter
from src import management as memory
from src.markdown import parse_topic_tree


class ScriptedAgent:
    def __init__(self, tool_calls=(), *, text="done", structured_output=None):
        self.tool_calls = list(tool_calls)
        self.text = text
        self.structured_output = structured_output
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        tools = {definition.name: definition for definition in kwargs["tools"]}
        for name, arguments in self.tool_calls:
            asyncio.run(tools[name].handler(arguments))
        return AgentResult(
            text=self.text,
            structured_output=self.structured_output,
            num_turns=max(1, len(self.tool_calls) + 1),
            input_tokens=10,
            output_tokens=5,
            total_cost_usd=0.001,
            duration_ms=20,
            duration_api_ms=15,
            stop_reason="end_turn",
            session_id="test-session",
        )


class QueuedAgent:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        tools = {definition.name: definition for definition in kwargs["tools"]}
        tool_calls = response.get("tool_calls", [])
        for name, arguments in tool_calls:
            asyncio.run(tools[name].handler(arguments))
        return AgentResult(
            text=response.get("text", ""),
            structured_output=response.get("structured_output"),
            num_turns=max(1, len(tool_calls) + 1),
            input_tokens=10,
            output_tokens=5,
            total_cost_usd=0.001,
            duration_ms=20,
            duration_api_ms=15,
            stop_reason="end_turn",
            session_id="test-session",
        )


def test_memory_workspace_runs_normal_shell_commands_from_memory_dir(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2026-01-01",
        "turns": [("user", "three tanks")],
        "refs": ["D1:1"],
    }])

    result = workspace.shell(
        "mkdir -p topics/aquarium && "
        "printf '%s\\n' '# Tanks' '###### Acquisition history' '' "
        "'On 2026-01-01, there were three tanks."
        "[^new-evidence-tanks] ^new-block-tanks' '' "
        "'[^new-evidence-tanks]: Time: `2026-01-01`; Sources: D1:1' "
        "> topics/aquarium/tanks.md && "
        "find topics -type f -print"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "topics/aquarium/tanks.md"
    assert "###### Acquisition history" in (
        workspace.stage_dir / "topics/aquarium/tanks.md"
    ).read_text()
    assert (tmp_path / "topics/aquarium/tanks.md").exists()


def test_writer_delegates_tool_protocol_and_errors_to_agent(tmp_path: Path):
    invalid = (
        "mkdir -p topics && printf '%s\\n' '# Pets' '' "
        "'The user bought a tank.[^new-evidence-tank] ^new-block-tank' '' "
        "'[^new-evidence-tank]: Time: `2023-05-23`; Sources: Session 1' "
        "> topics/pets.md"
    )
    valid = invalid.replace("Session 1", "D1:1")
    agent = ScriptedAgent([
        ("shell", {"command": invalid}),
        ("shell", {"command": valid}),
    ])

    audit = memory.write_session(
        tmp_path,
        agent=agent,
        observation_date="2023-05-23",
        turns=[("user", "I bought a tank")],
        refs=["D1:1"],
    )

    assert len(agent.calls) == 1
    assert [definition.name for definition in agent.calls[0]["tools"]] == [
        "shell"
    ]
    assert [record["status"] for record in audit[:-1]] == ["error", "ok"]
    assert audit[-1]["tool"] == "agent"
    assert audit[-1]["status"] == "ok"
    assert audit[-1]["rounds"] == 3
    assert (tmp_path / "topics/pets.md").is_file()


def test_shell_keeps_time_metadata_out_of_natural_topic_prose(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2023-05-08",
        "turns": [("user", "I painted that lake sunrise last year.")],
        "refs": ["D1:1"],
    }])

    result = workspace.shell(
        "mkdir -p topics/people && "
        "printf '%s\\n' '# Melanie' '' "
        "'Melanie painted a lake sunrise last year.  "
        "[^new-evidence-painting]^new-block-painting' '' "
        "'[^new-evidence-painting]: Time: `2022`; Sources: D1:1' "
        "> topics/people/melanie.md"
    )

    assert result.returncode == 0
    topic = (tmp_path / "topics/people/melanie.md").read_text()
    assert re.search(
        r"Melanie painted a lake sunrise last year\.\[\^e-[0-9a-f]{10}\] "
        r"\^[0-9a-f]{8}$",
        topic,
        re.MULTILINE,
    )
    unit = parse_topic_tree(tmp_path / "topics")[0]
    assert unit.content == "Melanie painted a lake sunrise last year."
    assert unit.evidence[0].when == "2022"


def test_shell_normalizes_an_unquoted_evidence_time(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2026-01-01",
        "turns": [("user", "Remember this fact.")],
        "refs": ["D1:1"],
    }])

    workspace.shell(
        "mkdir -p topics/calibration && "
        "printf '%s\\n' '# Calibration' '' "
        "'Remember this fact.[^new-evidence-fact] ^new-block-fact' '' "
        "'[^new-evidence-fact]: Time: 2026-01-01; Sources: D1:1' "
        "> topics/calibration/facts.md"
    )

    topic = (tmp_path / "topics/calibration/facts.md").read_text()
    assert "Time: `2026-01-01`; Sources:" in topic
    assert parse_topic_tree(tmp_path / "topics")[0].evidence[0].when == "2026-01-01"


def test_shell_rejects_model_invented_block_id(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2023-05-03",
        "turns": [("user", "I work at a local garage.")],
        "refs": ["D1:1"],
    }])

    with pytest.raises(
        ValueError,
        match=r"new memory blocks must use \^new-block-<label>",
    ):
        workspace.shell(
            "mkdir -p topics/people && "
            "printf '%s\\n' '# Dave' '' "
            "'Dave works at a local garage.[^new-evidence-work] "
            "^1478d194b29awork' '' "
            "'[^new-evidence-work]: Time: `2023-05-03`; Sources: D1:1' "
            "> topics/people/dave.md"
        )


def test_shell_rejects_topic_file_link_without_block_target(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2026-01-01",
        "turns": [("user", "Caroline and Melanie are friends.")],
        "refs": ["D1:1"],
    }])

    with pytest.raises(
        ValueError,
        match=r"Topic-to-Topic link must target #\^block-id",
    ):
        workspace.shell(
            "mkdir -p topics/people && "
            "printf '%s\\n' '# Caroline' '' "
            "'On 2026-01-01 Caroline was friends with [Melanie](melanie.md)."
            "[^new-evidence-friend] ^new-block-caroline' '' "
            "'[^new-evidence-friend]: Time: `2026-01-01`; Sources: D1:1' "
            "> topics/people/caroline.md && "
            "printf '%s\\n' '# Melanie' '' "
            "'On 2026-01-01 Melanie was friends with Caroline."
            "[^new-evidence-friend] ^new-block-melanie' '' "
            "'[^new-evidence-friend]: Time: `2026-01-01`; Sources: D1:1' "
            "> topics/people/melanie.md"
        )

    assert not (tmp_path / "topics/people/caroline.md").exists()


def test_shell_resolves_temporary_cross_topic_block_link(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2026-01-01",
        "turns": [
            ("user", "Caroline and Melanie are friends."),
            ("user", "Melanie and Caroline are friends."),
        ],
        "refs": ["D1:1", "D1:2"],
    }])

    workspace.shell(
        "mkdir -p topics/people && "
        "printf '%s\\n' '# Caroline' '' "
        "'On 2026-01-01 Caroline was friends with "
        "[Melanie](melanie.md#^new-block-melanie)."
        "[^new-evidence-caroline] ^new-block-caroline' '' "
        "'[^new-evidence-caroline]: Time: `2026-01-01`; Sources: D1:1' "
        "> topics/people/caroline.md && "
        "printf '%s\\n' '# Melanie' '' "
        "'On 2026-01-01 Melanie was friends with Caroline."
        "[^new-evidence-melanie] ^new-block-melanie' '' "
        "'[^new-evidence-melanie]: Time: `2026-01-01`; Sources: D1:2' "
        "> topics/people/melanie.md"
    )

    units = {unit.topic_path: unit for unit in parse_topic_tree(tmp_path / "topics")}
    source = units["people/caroline.md"]
    target = units["people/melanie.md"]
    assert f"melanie.md#^{target.memory_id}" in source.content
    relations = json.loads((tmp_path / "relations.json").read_text())
    assert relations["outbound"][source.memory_id] == [target.memory_id]
    assert relations["backlinks"][target.memory_id] == [source.memory_id]


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


def test_save_memory_retries_colliding_eight_digit_block_ids(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2026-01-01",
        "turns": [("user", "collision source")],
        "refs": ["D1:1"],
    }])
    events = [
        {
            "when": "2026-01-01",
            "content": f"Collision fact {index}.",
            "refs": ["D1:1"],
            "topic_path": "facts.md",
            "headings": ["Facts"],
        }
        for index in (99938, 168633)
    ]

    workspace.save_memory(events)
    units = parse_topic_tree(tmp_path / "topics")

    assert len(units) == 2
    assert all(re.fullmatch(r"[0-9a-f]{8}", unit.memory_id) for unit in units)
    assert len({unit.memory_id for unit in units}) == 2

    workspace.save_memory(events)
    assert len(parse_topic_tree(tmp_path / "topics")) == 2


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


def test_manager_exposes_only_shell_and_uses_twenty_turn_limit(tmp_path):
    agent = ScriptedAgent()

    audit = manage_memory(tmp_path, agent=agent)

    assert [definition.name for definition in agent.calls[0]["tools"]] == [
        "shell"
    ]
    assert agent.calls[0]["max_turns"] == 20
    assert audit[-1]["status"] == "ok"


def test_local_organizer_receives_only_touched_topic_scope(tmp_path):
    agent = ScriptedAgent()

    organize_topics(
        tmp_path,
        agent=agent,
        touched={"topics/projects/a.md", "topics/projects/b.md"},
    )

    task = agent.calls[0]["prompt"]
    assert "Limit this maintenance pass to these topic files" in task
    assert "topics/projects/a.md" in task
    assert "topics/projects/b.md" in task


def test_writer_may_read_but_must_not_modify_archived_sources():
    from src.management.prompts import SYSTEM_PROMPT, WRITER_BATCH_TASK, WRITER_TASK

    for prompt in (WRITER_TASK, WRITER_BATCH_TASK):
        assert (
            "Inspect whichever existing Topic, Core, or Source files are useful."
            in prompt
        )
        assert "Do not modify files under sources/." in prompt
    assert "[^new-evidence-example] ^new-block-example" in SYSTEM_PROMPT
    assert (
        "Time: `<time>`; Sources: <complete-source-handle-from-input>"
        in SYSTEM_PROMPT
    )


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
        path.write_text(
            "# Home\n" if relative == "topics/home.md" else relative,
            encoding="utf-8",
        )
    (tmp_path / "recent_events.jsonl").write_text("recent", encoding="utf-8")
    workspace = MemoryWorkspace(tmp_path)

    result = workspace.shell(
        "cat topics/home.md timeline/2026.md sources/D1.md recent_events.jsonl"
    )

    assert result.returncode == 0
    assert result.stdout == "# Home\ntimeline/2026.mdsources/D1.mdrecent"


def test_agent_needs_no_final_persistence_action_when_model_finishes(tmp_path):
    audit = _run_agent(
        tmp_path,
        agent=ScriptedAgent(),
        task="organize",
    )

    assert len(audit) == 1
    assert audit[0]["tool"] == "agent"
    assert audit[0]["status"] == "ok"


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
    assert "<!-- memory-event:" not in topic
    memory_id = recent["memory_id"]
    assert f"^{memory_id}" in topic
    assert f"#^{memory_id}" in timeline
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


def test_recent_memory_keeps_only_the_latest_fifty_events_by_default(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    turns = [("user", f"event {index}") for index in range(1, 52)]
    refs = [f"D1:{index}" for index in range(1, 52)]
    workspace.archive_sessions([{
        "observation_date": "2023-05-23",
        "turns": turns,
        "refs": refs,
    }])

    workspace.save_memory([{
        "when": "2023-05-23",
        "content": f"Event {index}.",
        "refs": [f"D1:{index}"],
        "topic_path": "events.md",
        "headings": ["Events"],
    } for index in range(1, 52)])

    recent = [
        json.loads(line)
        for line in (tmp_path / "recent_events.jsonl").read_text().splitlines()
    ]
    assert len(recent) == 50
    assert recent[0]["content"] == "Event 2."
    assert recent[-1]["content"] == "Event 51."


def test_shell_core_memory_changes_persist(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)

    result = workspace.shell("printf '%s\\n' '# Core Memory' 'Stable preference' > core.md")

    assert result.returncode == 0
    assert (tmp_path / "core.md").read_text() == (
        "# Core Memory\nStable preference\n"
    )


def test_shell_normalizes_and_validates_core_source_references(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2026-08-03",
        "turns": [("user", "My stable preference is jasmine tea.")],
        "refs": ["diagnostic/thread-1/msg-1"],
    }])

    workspace.shell(
        "cat > core.md <<'EOF'\n"
        "# Core Memory\n\n"
        "My stable preference is jasmine tea.[^new-evidence-tea] "
        "^new-block-tea\n\n"
        "[^new-evidence-tea]: Time: `2026-08-03`; Sources: "
        "diagnostic/thread-1/msg-1\n"
        "EOF"
    )

    text = (tmp_path / "core.md").read_text(encoding="utf-8")
    assert "new-evidence" not in text
    assert "new-block" not in text
    assert re.search(r"\^[0-9a-f]{8}$", text, re.MULTILINE)
    assert (
        "[diagnostic/thread-1/msg-1]"
        "(sources/diagnostic/thread-1.md#source-54a317afec5ee542)"
    ) in text


def test_shell_rejects_missing_core_source_reference(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)

    with pytest.raises(ValueError, match="missing source reference"):
        workspace.shell(
            "cat > core.md <<'EOF'\n"
            "# Core Memory\n\n"
            "Unsupported claim.[^new-evidence-claim] ^new-block-claim\n\n"
            "[^new-evidence-claim]: Time: `undated`; Sources: "
            "diagnostic/missing/msg-1\n"
            "EOF"
        )

    assert not (tmp_path / "core.md").exists()


def test_core_memory_rejects_content_over_token_limit(tmp_path: Path):
    workspace = MemoryWorkspace(
        tmp_path, config=memory.MemoryConfig(core_max_tokens=1)
    )

    with pytest.raises(ValueError, match="Core Memory exceeds"):
        workspace.shell("printf '%s\\n' 'alpha beta' > core.md")

    assert not (tmp_path / "core.md").exists()


def test_core_memory_restores_previous_content_when_commit_fails(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "core.md").write_text("old core\n", encoding="utf-8")
    workspace = MemoryWorkspace(tmp_path)
    original_replace = os.replace

    def fail_core_install(source, destination):
        if (
            Path(source) == workspace.stage_dir / "core.md"
            and Path(destination) == tmp_path / "core.md"
        ):
            raise OSError("injected core install failure")
        return original_replace(source, destination)

    monkeypatch.setattr(
        "src.management.block_views.os.replace",
        fail_core_install,
    )

    with pytest.raises(OSError, match="injected core install failure"):
        workspace.shell("printf '%s\\n' 'new core' > core.md")

    assert (tmp_path / "core.md").read_text() == "old core\n"


def test_recent_memory_rejects_negative_capacity():
    with pytest.raises(ValueError, match="recent_limit must be non-negative"):
        memory.MemoryConfig(recent_limit=-1)


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
    assert "sources/D18.md#d18-11" in topic


def test_shell_move_rewrites_cross_topic_block_link_and_installs_relations(
    tmp_path: Path,
):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2026-01-01",
        "turns": [("user", "source"), ("user", "target")],
        "refs": ["D1:1", "D1:2"],
    }])
    workspace.save_memory([
        {"when": "2026-01-01", "content": "On 2026-01-01, Source fact.", "refs": ["D1:1"],
         "topic_path": "a.md", "headings": ["A"]},
        {"when": "2026-01-02", "content": "On 2026-01-02, Target fact.", "refs": ["D1:2"],
         "topic_path": "b.md", "headings": ["B"]},
    ])
    units = parse_topic_tree(tmp_path / "topics")
    source_id, target_id = [unit.memory_id for unit in units]
    workspace.shell(
        "sed -i '' 's/Source fact\\./Source fact linked to "
        f"[target](b.md#^{target_id})./' topics/a.md"
    )

    workspace.shell("mkdir -p topics/moved && mv topics/b.md topics/moved/b.md")

    assert f"[target](moved/b.md#^{target_id})" in (
        tmp_path / "topics/a.md"
    ).read_text()
    relations = json.loads((tmp_path / "relations.json").read_text())
    assert relations["outbound"][source_id] == [target_id]
    assert relations["backlinks"][target_id] == [source_id]


def test_deleting_last_topic_block_clears_only_derived_views(tmp_path: Path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2026-01-01",
        "turns": [("user", "fact")],
        "refs": ["D1:1"],
    }])
    workspace.save_memory([{
        "when": "2026-01-01", "content": "Fact.", "refs": ["D1:1"],
        "topic_path": "a.md", "headings": ["A"],
    }])

    workspace.shell("rm topics/a.md")

    assert (tmp_path / "recent_events.jsonl").read_text() == ""
    assert not list((tmp_path / "timeline").rglob("*.md"))
    assert json.loads((tmp_path / "relations.json").read_text()) == {
        "backlinks": {}, "outbound": {}
    }
    assert "fact" in (tmp_path / "sources/D1.md").read_text()


def test_recent_fifo_keeps_full_creation_order_outside_the_window(
    tmp_path: Path,
):
    workspace = MemoryWorkspace(
        tmp_path, config=memory.MemoryConfig(recent_limit=2)
    )
    workspace.archive_sessions([{
        "observation_date": "2026-01-01",
        "turns": [("user", "one"), ("user", "two"), ("user", "three")],
        "refs": ["D1:1", "D1:2", "D1:3"],
    }])
    workspace.save_memory([
        {"when": f"2026-01-0{index}",
         "content": f"On 2026-01-0{index}, Fact {index}.",
         "refs": [f"D1:{index}"], "topic_path": "facts.md", "headings": ["Facts"]}
        for index in range(1, 4)
    ])
    ids = [unit.memory_id for unit in parse_topic_tree(tmp_path / "topics")]

    workspace.shell("sed -i '' 's/Fact 1/Updated fact 1/' topics/facts.md")

    recent = [
        json.loads(line)
        for line in (tmp_path / "recent_events.jsonl").read_text().splitlines()
    ]
    assert [row["memory_id"] for row in recent] == ids[1:]


def test_block_transaction_restores_every_installed_view_on_install_failure(
    tmp_path: Path, monkeypatch
):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{
        "observation_date": "2026-01-01",
        "turns": [("user", "old fact")],
        "refs": ["D1:1"],
    }])
    workspace.save_memory([{
        "when": "2026-01-01", "content": "On 2026-01-01, Old fact.",
        "refs": ["D1:1"],
        "topic_path": "a.md", "headings": ["A"],
    }])
    tracked = [
        "topics/a.md", "timeline/2026/01/01.md", "recent_events.jsonl",
        "relations.json", ".nativemem/runtime.json",
    ]
    before = {name: (tmp_path / name).read_bytes() for name in tracked}
    original_replace = os.replace

    def fail_relations_install(source, destination):
        if (
            Path(source).name == "relations.json"
            and workspace.stage_dir in Path(source).parents
            and Path(destination) == tmp_path / "relations.json"
        ):
            raise OSError("injected relations install failure")
        return original_replace(source, destination)

    monkeypatch.setattr(
        "src.management.block_views.os.replace",
        fail_relations_install,
    )

    with pytest.raises(OSError, match="injected relations install failure"):
        workspace.shell("sed -i '' 's/Old fact/Updated fact/' topics/a.md")

    assert {name: (tmp_path / name).read_bytes() for name in tracked} == before


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
        "content": "On 2023-05-23, the user wakes up at 7:00 AM.",
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
    assert recent["content"] == (
        "On 2023-05-23, the user wakes up at 6:30 AM."
    )


def test_nativemem_build_verifies_each_session_before_next_write(
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
        adapter.memory,
        "write_sessions",
        lambda *args, **kwargs: calls.append(
            ("write", kwargs["sessions"][0]["observation_date"])
        ) or [{"tool": "save_memory", "count": 1}],
    )
    monkeypatch.setattr(
        adapter.memory,
        "verify_session",
        lambda *args, **kwargs: calls.append(
            ("verify", kwargs["observation_date"])
        ) or {"repaired": False},
    )
    monkeypatch.setattr(
        adapter.memory,
        "manage_memory",
        lambda *args, **kwargs: calls.append(("manage", None)) or [],
    )

    _, events = adapter.build_memory(
        conv, tmp_path, agent=object(), model="test"
    )

    assert events == 2
    assert calls == [
        ("write", "2023-05-01"),
        ("verify", "2023-05-01"),
        ("write", "2023-05-02"),
        ("verify", "2023-05-02"),
        ("manage", None),
    ]


def test_nativemem_build_replaces_locomo_sequence_refs_with_opaque_source_ids(
    tmp_path: Path, monkeypatch
):
    captured = []
    conv = {
        "session_1": [{
            "speaker": "Melanie",
            "text": "I painted a lake sunrise last year.",
            "dia_id": "D1:14",
        }],
        "session_1_date_time": "2023-05-08",
    }
    monkeypatch.setattr(
        adapter.memory,
        "write_sessions",
        lambda *args, **kwargs: captured.extend(kwargs["sessions"]) or [],
    )

    adapter.build_memory(
        conv,
        tmp_path,
        agent=object(),
        model="test",
        config=adapter.BuildConfig(verify_writes=False, final_manage=False),
    )

    ref = captured[0]["refs"][0]
    assert re.fullmatch(
        r"locomo/thread_[0-9a-f]{12}/msg_[0-9a-f]{12}", ref
    )
    assert "D1:14" not in ref


def test_nativemem_build_runs_local_reorganization_at_fixed_session_intervals(
    tmp_path: Path, monkeypatch
):
    conv = {}
    for index in range(1, 6):
        conv[f"session_{index}"] = [("user", f"message {index}")]
        conv[f"session_{index}_date_time"] = f"2023-05-0{index}"
    local_calls = []

    monkeypatch.setattr(
        adapter.memory,
        "write_sessions",
        lambda *args, **kwargs: [{
            "tool": "save_memory",
            "status": "ok",
            "count": 1,
            "topic_paths": [
                f"topics/topic-{kwargs['sessions'][0]['observation_date']}.md"
            ],
        }],
    )
    monkeypatch.setattr(
        adapter.memory,
        "verify_session",
        lambda *args, **kwargs: {"repaired": False},
    )
    monkeypatch.setattr(
        adapter.memory,
        "organize_topics",
        lambda *args, **kwargs: local_calls.append(set(kwargs["touched"])) or [],
    )
    monkeypatch.setattr(
        adapter.memory,
        "manage_memory",
        lambda *args, **kwargs: [],
    )

    adapter.build_memory(
        conv,
        tmp_path,
        agent=object(),
        model="test",
        config=adapter.BuildConfig(local_reorg_every_sessions=2),
    )

    assert local_calls == [
        {
            "topics/topic-2023-05-01.md",
            "topics/topic-2023-05-02.md",
        },
        {
            "topics/topic-2023-05-03.md",
            "topics/topic-2023-05-04.md",
        },
    ]


@pytest.mark.parametrize(
    ("audit", "final_manage", "expected_calls"),
    [
        ([], True, 0),
        ([{"tool": "save_memory", "status": "ok", "count": 1}], False, 0),
        ([{"tool": "save_memory", "status": "ok", "count": 1}], True, 1),
    ],
)
def test_nativemem_final_management_requires_new_memory_and_can_be_disabled(
    tmp_path: Path, monkeypatch, audit, final_manage, expected_calls
):
    conv = {
        "session_1": [("user", "message")],
        "session_1_date_time": "2023-05-01",
    }
    calls = []
    monkeypatch.setattr(
        adapter.memory,
        "write_sessions",
        lambda *args, **kwargs: audit,
    )
    monkeypatch.setattr(
        adapter.memory,
        "verify_session",
        lambda *args, **kwargs: {"repaired": False},
    )
    monkeypatch.setattr(
        adapter.memory,
        "manage_memory",
        lambda *args, **kwargs: calls.append("manage") or [],
    )

    adapter.build_memory(
        conv,
        tmp_path,
        agent=object(),
        model="test",
        config=adapter.BuildConfig(final_manage=final_manage),
    )

    assert len(calls) == expected_calls


def test_nativemem_build_can_disable_write_verification(tmp_path: Path, monkeypatch):
    conv = {
        "session_1": [("user", "message")],
        "session_1_date_time": "2023-05-01",
    }
    monkeypatch.setattr(
        adapter.memory,
        "write_sessions",
        lambda *args, **kwargs: [
            {"tool": "save_memory", "status": "ok", "count": 1}
        ],
    )
    monkeypatch.setattr(
        adapter.memory,
        "verify_session",
        lambda *args, **kwargs: pytest.fail("verification should be disabled"),
    )

    _seconds, events = adapter.build_memory(
        conv,
        tmp_path,
        agent=object(),
        model="test",
        config=adapter.BuildConfig(verify_writes=False, final_manage=False),
    )

    assert events == 1
    verification = json.loads((tmp_path / "verification.jsonl").read_text())
    assert verification == {"skipped": True, "reason": "disabled"}


def test_nativemem_build_batches_writes_and_samples_verification(
    tmp_path: Path, monkeypatch
):
    conv = {}
    for index in range(1, 6):
        conv[f"session_{index}"] = [("user", f"message {index}")]
        conv[f"session_{index}_date_time"] = f"2023-05-0{index}"
    writes = []
    verifications = []

    monkeypatch.setattr(
        adapter.memory,
        "write_sessions",
        lambda *args, **kwargs: writes.append([
            session["observation_date"] for session in kwargs["sessions"]
        ]) or [{"tool": "save_memory", "status": "ok", "count": 1}],
    )
    monkeypatch.setattr(
        adapter.memory,
        "verify_session",
        lambda *args, **kwargs: verifications.append(kwargs["observation_date"])
        or {"repaired": False},
    )

    adapter.build_memory(
        conv,
        tmp_path,
        agent=object(),
        model="test",
        config=adapter.BuildConfig(
            session_batch=2,
            verify_every_sessions=2,
            final_manage=False,
        ),
    )

    assert writes == [
        ["2023-05-01", "2023-05-02"],
        ["2023-05-03", "2023-05-04"],
        ["2023-05-05"],
    ]
    assert verifications == ["2023-05-02", "2023-05-04", "2023-05-05"]


def test_verify_session_does_not_repair_when_memory_answers_probe(
    tmp_path, monkeypatch
):
    del monkeypatch
    source_workspace = MemoryWorkspace(tmp_path)
    source_workspace.archive_sessions([{
        "observation_date": "2023-05-23",
        "turns": [("user", "I wake up at 6:30 AM.")],
        "refs": ["D1:1"],
    }])
    agent = QueuedAgent([
        {"structured_output": {
            "question": "What time does the user wake up?",
            "expected_answer": "6:30 AM",
            "refs": ["D1:1"],
        }},
        {"text": "<answer>6:30 AM</answer>"},
        {"structured_output": {"supported": True}},
    ])
    before = list(tmp_path.rglob("*"))

    result = verify_session(
        tmp_path,
        agent=agent,
        observation_date="2023-05-23",
        turns=[("user", "I wake up at 6:30 AM.")],
        refs=["D1:1"],
    )

    assert result["repaired"] is False
    assert result["initial"]["answer"] == "6:30 AM"
    assert result["post_repair"] is None
    assert list(tmp_path.rglob("*")) == before
    assert agent.calls[0]["output_schema"]["required"] == [
        "question", "expected_answer", "refs"
    ]
    assert agent.calls[1].get("output_schema") is None
    assert agent.calls[2]["output_schema"]["required"] == ["supported"]


def test_verify_session_repairs_then_retries_same_question(tmp_path):
    source_workspace = MemoryWorkspace(tmp_path)
    source_workspace.archive_sessions([{
        "observation_date": "2023-05-23",
        "turns": [("user", "I wake up at 6:30 AM.")],
        "refs": ["D1:1"],
    }])
    repair_command = (
        "mkdir -p topics/routines && printf '%s\\n' '# Daily routine' "
        "'## Morning' '' 'On 2023-05-23, the user wakes up at 6:30 AM."
        "[^new-evidence-wake] ^new-block-wake' '' "
        "'[^new-evidence-wake]: Time: `2023-05-23`; Sources: D1:1' "
        "> topics/routines/daily.md"
    )
    agent = QueuedAgent([
        {"structured_output": {
            "question": "What time does the user wake up?",
            "expected_answer": "6:30 AM",
            "refs": ["D1:1"],
        }},
        {"text": "<answer>No information available.</answer>"},
        {"structured_output": {"supported": False}},
        {"tool_calls": [("shell", {"command": repair_command})]},
        {"text": "<answer>6:30 AM</answer>"},
        {"structured_output": {"supported": True}},
    ])

    result = verify_session(
        tmp_path,
        agent=agent,
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
