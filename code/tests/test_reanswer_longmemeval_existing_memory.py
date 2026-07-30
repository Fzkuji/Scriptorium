from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import reanswer_longmemeval_existing_memory as MOD


def test_configure_applies_file_map_and_context(monkeypatch):
    backend = object()
    monkeypatch.setattr(MOD.lme, "load_backend", lambda: backend)
    monkeypatch.setattr(MOD.lme, "verify_backend_contract", lambda value: None)

    result = MOD.configure(
        "http://127.0.0.1:12345/v1",
        12,
        model="gpt-5.5",
        api_key="test-key",
        max_tokens=4000,
        map_mode="files",
        map_inline=128,
        read_context=1,
    )

    assert result is backend
    assert MOD.os.environ["NATIVEMEM_V8_MAP"] == "files"
    assert MOD.os.environ["NATIVEMEM_V8_MAP_INLINE"] == "128"
    assert MOD.os.environ["NATIVEMEM_V8_READ_CONTEXT"] == "1"
    assert MOD.os.environ["NATIVEMEM_V8_CONCURRENCY"] == "12"
    assert MOD.os.environ["NATIVEMEM_V8_MAX_TOKENS"] == "4000"
    assert MOD.os.environ["MODEL"] == "gpt-5.5"
    assert MOD.os.environ["BUILDER_MODEL"] == "gpt-5.5"
    assert MOD.os.environ["BUILDER_KEY"] == "test-key"
    assert MOD.os.environ["ALIYUN_KEY"] == "test-key"


def test_longmemeval_answer_prompt_has_no_length_requirement():
    prompt = MOD.lme.LME_SINGLE_PROMPT.lower()
    assert "fewer than" not in prompt
    assert " words" not in prompt
    assert "concise" not in prompt


def test_longmemeval_prompt_dispatches_to_official_runner(monkeypatch, tmp_path):
    expected = ([{"content": "memory"}], 3, "answer", [{"type": "bash"}])
    calls = []

    def official(backend, item, memory_dir, turn_index):
        calls.append((backend, item, memory_dir, turn_index))
        if len(calls) == 1:
            raise MOD.lme.EmptyStageError("empty answer")
        return expected

    monkeypatch.setattr(MOD.lme, "collect_and_answer_longmemeval", official)
    backend = SimpleNamespace()
    item = {"question": "where?"}
    memory_dir = tmp_path / "memory"
    turn_index = {"D1:1": {}}

    dispatch = getattr(MOD, "collect_answer", None)
    assert callable(dispatch)
    assert dispatch(backend, item, memory_dir, turn_index, "longmemeval") == expected
    assert calls == [
        (backend, item, Path(memory_dir), turn_index),
        (backend, item, Path(memory_dir), turn_index),
    ]


def test_v11_agent_rejects_answer_until_it_reads_memory(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "education.md").write_text(
        "The user graduated in Business Administration. [D1:1]\n"
    )

    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="read_memory_file",
            arguments='{"path":"education.md"}',
        ),
    )
    messages = iter([
        SimpleNamespace(content="<answer>Unknown</answer>", tool_calls=[]),
        SimpleNamespace(content=None, tool_calls=[tool_call]),
        SimpleNamespace(
            content="<answer>Business Administration</answer>", tool_calls=[]
        ),
    ])

    def create(**_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=next(messages), finish_reason="stop")]
        )

    backend = SimpleNamespace(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        ALIYUN_MODEL="gpt-5.5",
        log_usage=lambda *_args, **_kwargs: None,
    )

    memories, steps, answer, trace = MOD.collect_answer(
        backend,
        {"question": "What degree?", "question_date": "2023-05-30"},
        memory_dir,
        {"D1:1": {"speaker": "user", "text": "Business Administration"}},
        "v11-agent",
    )

    assert answer == "Business Administration"
    assert steps == 3
    assert memories and "Business Administration" in memories[0]["text"]
    assert [entry["type"] for entry in trace] == [
        "rejected_no_evidence",
        "read_memory_file",
    ]


def test_v11_agent_can_use_general_read_only_bash(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "education.md").write_text("Business Administration\n")
    tool_call = SimpleNamespace(
        id="call-bash",
        function=SimpleNamespace(
            name="bash",
            arguments='{"command":"rg -n Business ."}',
        ),
    )
    messages = iter([
        SimpleNamespace(content=None, tool_calls=[tool_call]),
        SimpleNamespace(
            content="<answer>Business Administration</answer>", tool_calls=[]
        ),
    ])

    def create(**_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=next(messages), finish_reason="stop")]
        )

    def execute_tool(_name, args, _base_dir, hide_raw):
        assert hide_raw is True
        assert args["command"] == "rg -n Business ."
        return "education.md:1:Business Administration"

    backend = SimpleNamespace(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        ALIYUN_MODEL="gpt-5.5",
        log_usage=lambda *_args, **_kwargs: None,
        execute_tool=execute_tool,
    )

    memories, _, answer, trace = MOD.collect_answer(
        backend,
        {"question": "What degree?", "question_date": "2023-05-30"},
        memory_dir,
        {},
        "v11-agent",
    )

    assert answer == "Business Administration"
    assert memories[0]["text"] == "education.md:1:Business Administration"
    assert trace == [{
        "type": "bash",
        "args": {"command": "rg -n Business ."},
        "nonempty": True,
        "accepted": True,
    }]


def test_v11_inventory_exposes_memory_views_but_hides_sources(tmp_path):
    (tmp_path / "topics").mkdir()
    (tmp_path / "timeline/2023/05").mkdir(parents=True)
    (tmp_path / "sources").mkdir()
    (tmp_path / "topics/education.md").write_text("degree\n")
    (tmp_path / "timeline/2023/05/23.md").write_text("event\n")
    (tmp_path / "sources/D1.md").write_text("raw source\n")
    (tmp_path / "recent_events.jsonl").write_text('{"event_id":"ev_1"}\n')

    relative = [path.relative_to(tmp_path).as_posix() for path in MOD._v11_files(tmp_path)]

    assert relative == [
        "recent_events.jsonl",
        "timeline/2023/05/23.md",
        "topics/education.md",
    ]
    assert "raw source" not in MOD._v11_read(tmp_path, "topics/education.md")
    with pytest.raises(ValueError, match="not an allowed memory file"):
        MOD._v11_read(tmp_path, "sources/D1.md")


def test_v11_ablation_conditions_change_one_view_or_source_tool(tmp_path):
    (tmp_path / "topics").mkdir()
    (tmp_path / "timeline").mkdir()
    (tmp_path / "topics/topic.md").write_text("topic\n")
    (tmp_path / "timeline/day.md").write_text("day\n")
    (tmp_path / "recent_events.jsonl").write_text('{"event_id":"ev_1"}\n')

    def visible(condition):
        return {
            path.relative_to(tmp_path).as_posix()
            for path in MOD._v11_files(tmp_path, condition)
        }

    assert visible("dual_source") == {
        "topics/topic.md", "timeline/day.md", "recent_events.jsonl"
    }
    assert visible("topic_source") == {
        "topics/topic.md", "recent_events.jsonl"
    }
    assert visible("timeline_source") == {
        "timeline/day.md", "recent_events.jsonl"
    }
    assert visible("dual_no_source") == visible("dual_source")
    assert {
        path.relative_to(tmp_path).as_posix()
        for path in MOD._v11_files(
            tmp_path, "dual_source", include_recent=False
        )
    } == {"topics/topic.md", "timeline/day.md"}

    tool_names = lambda condition: {
        tool["function"]["name"] for tool in MOD._v11_tools(condition)
    }
    assert "bash" not in tool_names("dual_source")
    assert "resolve_sources" in tool_names("dual_source")
    assert "resolve_sources" not in tool_names("dual_no_source")


def test_load_completed_results_validates_item_identity(tmp_path):
    dataset = [
        {"question_id": "q0", "question": "first"},
        {"question_id": "q1", "question": "second"},
    ]
    sources = [{"dataset_index": 0}, {"dataset_index": 1}]
    items = tmp_path / "items"
    items.mkdir()
    MOD.atomic_json(items / "0000.json", {
        "dataset_index": 0,
        "question_id": "q0",
        "question": "first",
        "answer": "done",
    })

    assert MOD.load_completed_results(tmp_path, dataset, sources) == {
        0: MOD.read_json(items / "0000.json")
    }

    MOD.atomic_json(items / "0001.json", {
        "dataset_index": 1,
        "question_id": "wrong",
        "question": "second",
        "answer": "done",
    })
    with pytest.raises(ValueError, match="item 1 identity"):
        MOD.load_completed_results(tmp_path, dataset, sources)


def test_write_results_rebuilds_sorted_aggregate(tmp_path):
    completed = {
        2: {"dataset_index": 2, "answer": "third"},
        0: {"dataset_index": 0, "answer": "first"},
    }

    MOD.write_results(tmp_path, completed)

    assert MOD.read_json(tmp_path / "results.json") == [
        completed[0],
        completed[2],
    ]


def test_run_pending_preserves_completion_and_stops_on_interrupt():
    sources = [{"dataset_index": 0}, {"dataset_index": 1}, {"dataset_index": 2}]
    saved = []

    def work(source):
        index = source["dataset_index"]
        if index == 1:
            raise KeyboardInterrupt
        return {"dataset_index": index, "answer": f"answer-{index}"}

    completed, interrupted = MOD.run_pending(
        sources,
        workers=1,
        work=work,
        on_complete=lambda record: saved.append(record["dataset_index"]),
    )

    assert completed == {0: {"dataset_index": 0, "answer": "answer-0"}}
    assert saved == [0]
    assert interrupted is True


def test_stop_signal_becomes_keyboard_interrupt():
    with pytest.raises(KeyboardInterrupt):
        MOD.stop_on_signal(None, None)
