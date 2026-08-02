import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.nativemem import reanswer_longmemeval as MOD
from src.retrieval.embedding import MemoryEmbeddingIndex
from src import retrieval
from src.retrieval import tools as retrieval_tools


def test_configure_uses_explicit_model_client_without_mutating_environment(
    monkeypatch,
):
    client = object()
    monkeypatch.setenv("MODEL", "inherited-model")
    monkeypatch.setenv("BUILDER_KEY", "inherited-key")

    result = retrieval.create_runtime(
        "http://127.0.0.1:12345/v1",
        model="gpt-5.5",
        api_key="test-key",
        client=client,
    )

    assert result.client is client
    assert result.model == "gpt-5.5"
    assert MOD.os.environ["MODEL"] == "inherited-model"
    assert MOD.os.environ["BUILDER_KEY"] == "inherited-key"


def test_reanswer_dispatch_uses_the_budgeted_retrieval_loop(monkeypatch, tmp_path):
    expected = ([{"text": "memory", "date": ""}], 2, "answer", [])
    seen = []

    def budgeted(backend, item, memory_dir, turn_index, **_kwargs):
        seen.append((backend, item, memory_dir, turn_index))
        return expected

    monkeypatch.setattr(retrieval, "collect_answer", budgeted)
    backend = SimpleNamespace()
    item = {"question": "where?", "question_date": "2023-05-30"}
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    turn_index = {"D1:1": {}}

    result = MOD.collect_answer(backend, item, memory_dir, turn_index)

    assert result == expected
    assert seen == [(backend, item, memory_dir, turn_index)]


def test_agent_rejects_answer_until_it_reads_memory(tmp_path):
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
        model="gpt-5.5",
        log_usage=lambda *_args, **_kwargs: None,
    )

    memories, steps, answer, trace = MOD.collect_answer(
        backend,
        {"question": "What degree?", "question_date": "2023-05-30"},
        memory_dir,
        {"D1:1": {"speaker": "user", "text": "Business Administration"}},
    )

    assert answer == "Business Administration"
    assert steps == 3
    assert memories and "Business Administration" in memories[0]["text"]
    assert [entry["type"] for entry in trace] == [
        "rejected_no_evidence",
        "read_memory_file",
        "termination",
    ]
    assert trace[-1]["termination_reason"] == "evidence_sufficient"


def test_agent_can_use_general_read_only_bash(tmp_path):
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
        model="gpt-5.5",
        log_usage=lambda *_args, **_kwargs: None,
        execute_tool=execute_tool,
    )

    memories, _, answer, trace = MOD.collect_answer(
        backend,
        {"question": "What degree?", "question_date": "2023-05-30"},
        memory_dir,
        {},
    )

    assert answer == "Business Administration"
    assert memories[0]["text"] == "education.md:1:Business Administration"
    assert trace[0]["type"] == "bash"
    assert trace[0]["args"] == {"command": "rg -n Business ."}
    assert trace[0]["nonempty"] is True
    assert trace[0]["accepted"] is True
    assert trace[0]["executed"] is True


def test_nativemem_agent_normalizes_the_reported_workspace_root_for_bash(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "core.md").write_text("Business Administration [D1:1]\n")
    absolute = memory_dir.resolve() / "core.md"
    tool_call = SimpleNamespace(
        id="call-absolute",
        function=SimpleNamespace(
            name="bash",
            arguments=json.dumps({"command": f"cat {absolute}"}),
        ),
    )
    responses = iter([
        SimpleNamespace(content=None, tool_calls=[tool_call]),
        SimpleNamespace(
            content="<answer>Business Administration</answer>", tool_calls=[]
        ),
    ])
    executed = []

    def create(**_kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=next(responses))])

    def execute_tool(_name, args, _base_dir, hide_raw):
        executed.append(args["command"])
        return "Business Administration [D1:1]"

    backend = SimpleNamespace(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        model="gpt-5.5",
        log_usage=lambda *_args, **_kwargs: None,
        execute_tool=execute_tool,
    )

    _memories, _steps, answer, trace = retrieval.collect_answer(
        backend, {"question": "What degree?"}, memory_dir, {}
    )

    assert answer == "Business Administration"
    assert executed == ["cat ./core.md"]
    assert trace[1]["accepted"] is True
    assert trace[1]["executed"] is True


def test_nativemem_agent_retries_after_rejected_bash_paths(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "fact.md").write_text("Business Administration [D1:1]\n")
    rejected = [
        SimpleNamespace(
            id=f"call-rejected-{index}",
            function=SimpleNamespace(
                name="bash",
                arguments=json.dumps({"command": f"cat /invented-{index}/fact.md"}),
            ),
        )
        for index in range(2)
    ]
    valid = SimpleNamespace(
        id="call-valid",
        function=SimpleNamespace(
            name="bash", arguments='{"command":"cat fact.md"}'
        ),
    )
    responses = iter([
        SimpleNamespace(content=None, tool_calls=rejected),
        SimpleNamespace(content=None, tool_calls=[valid]),
        SimpleNamespace(
            content="<answer>Business Administration</answer>", tool_calls=[]
        ),
    ])

    def create(**_kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=next(responses))])

    backend = SimpleNamespace(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        model="gpt-5.5",
        log_usage=lambda *_args, **_kwargs: None,
        execute_tool=lambda *_args, **_kwargs: "Business Administration [D1:1]",
    )

    _memories, steps, answer, trace = retrieval.collect_answer(
        backend, {"question": "What degree?"}, memory_dir, {}
    )

    rejected_trace = [entry for entry in trace if entry.get("accepted") is False]
    assert answer == "Business Administration"
    assert steps == 3
    assert len(rejected_trace) == 2
    assert all(entry["executed"] is False for entry in rejected_trace)
    assert all(entry["nonempty"] is False for entry in rejected_trace)
    assert trace[-1]["termination_reason"] == "evidence_sufficient"


def test_nativemem_agent_receives_core_and_recent_before_tool_use(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "core.md").write_text(
        "The user's stable preferred language is Chinese. [D1:1]\n"
    )
    (memory_dir / "recent_events.jsonl").write_text(
        '{"content":"The user started Project Atlas.","refs":["D1:2"]}\n'
    )
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="<answer>Chinese</answer>", tool_calls=[]
        ))])

    backend = SimpleNamespace(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        model="gpt-5.5",
        log_usage=lambda *_args, **_kwargs: None,
    )

    memories, steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "What language does the user prefer?"},
        memory_dir,
        {},
    )

    prompt = requests[0]["messages"][0]["content"]
    assert f"Bash working directory: {memory_dir.resolve()}" in prompt
    assert "Use Inventory paths relative to this directory" in prompt
    assert "stable preferred language is Chinese" in prompt
    assert "started Project Atlas" in prompt
    assert answer == "Chinese"
    assert steps == 1
    assert len(memories) == 2
    assert trace[0]["type"] == "initial_context"


def test_nativemem_agent_stops_after_five_tool_calls_and_finalizes(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    calls = []
    for index in range(6):
        path = memory_dir / f"fact-{index}.md"
        path.write_text(f"unique fact {index} [D1:{index + 1}]\n")
        calls.append(SimpleNamespace(
            id=f"call-{index}",
            function=SimpleNamespace(
                name="read_memory_file",
                arguments=f'{{"path":"fact-{index}.md"}}',
            ),
        ))
    responses = iter([
        SimpleNamespace(content=None, tool_calls=calls),
        SimpleNamespace(
            content="<answer>Insufficient information.</answer>", tool_calls=[]
        ),
    ])
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=next(responses))])

    backend = SimpleNamespace(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        model="gpt-5.5",
        log_usage=lambda *_args, **_kwargs: None,
    )
    memories, steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "unknown?"},
        memory_dir,
        {},
        config=retrieval.QueryConfig(max_tool_calls=5),
    )

    executed = [entry for entry in trace if entry.get("executed") is True]
    assert len(executed) == 5
    assert trace[-1]["termination_reason"] == "tool_call_limit"
    assert "tools" not in requests[-1]
    assert len(memories) == 5
    assert steps == 2
    assert answer == "Insufficient information."


def test_nativemem_default_retrieval_round_limit_is_eight(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    responses = iter([
        *[
            SimpleNamespace(content="<answer>unsupported</answer>", tool_calls=[])
            for _ in range(8)
        ],
        SimpleNamespace(
            content="<answer>Insufficient information.</answer>", tool_calls=[]
        ),
    ])

    def create(**_kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=next(responses))])

    backend = SimpleNamespace(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        model="gpt-5.5",
        log_usage=lambda *_args, **_kwargs: None,
    )
    _memories, steps, answer, trace = retrieval.collect_answer(
        backend, {"question": "unknown?"}, memory_dir, {}
    )

    assert steps == 9
    assert answer == "Insufficient information."
    assert trace[-1]["retrieval_rounds"] == 8
    assert trace[-1]["termination_reason"] == "round_limit"


def test_nativemem_agent_stops_after_two_calls_add_no_new_evidence(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "fact.md").write_text("same fact [D1:1]\n")
    duplicate_calls = [SimpleNamespace(
        id=f"call-{index}",
        function=SimpleNamespace(
            name="read_memory_file",
            arguments='{"path":"fact.md"}',
        ),
    ) for index in range(3)]
    responses = iter([
        SimpleNamespace(content=None, tool_calls=duplicate_calls),
        SimpleNamespace(content="<answer>same fact</answer>", tool_calls=[]),
    ])

    def create(**_kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=next(responses))])

    backend = SimpleNamespace(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        model="gpt-5.5",
        log_usage=lambda *_args, **_kwargs: None,
    )
    _memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "what fact?"},
        memory_dir,
        {},
        config=retrieval.QueryConfig(max_tool_calls=5),
    )

    executed = [entry for entry in trace if entry.get("executed") is True]
    assert len(executed) == 3
    assert trace[-1]["termination_reason"] == "no_new_evidence"
    assert answer == "same fact"


def test_nativemem_visible_memory_budget_truncates_tool_output(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "long.md").write_text(("evidence " * 100) + "[D1:1]\n")
    tool_call = SimpleNamespace(
        id="call-long",
        function=SimpleNamespace(
            name="read_memory_file", arguments='{"path":"long.md"}'
        ),
    )
    responses = iter([
        SimpleNamespace(content=None, tool_calls=[tool_call]),
        SimpleNamespace(
            content="<answer>Insufficient information.</answer>", tool_calls=[]
        ),
    ])

    def create(**_kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=next(responses))])

    backend = SimpleNamespace(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        model="gpt-5.5",
        log_usage=lambda *_args, **_kwargs: None,
    )
    memories, _steps, _answer, trace = retrieval.collect_answer(
        backend,
        {"question": "what evidence?"},
        memory_dir,
        {},
        config=retrieval.QueryConfig(visible_token_limit=30),
    )

    read = next(entry for entry in trace if entry["type"] == "read_memory_file")
    assert read["raw_visible_tokens"] > read["delivered_visible_tokens"]
    assert read["cumulative_visible_tokens"] <= 30
    assert trace[-1]["termination_reason"] == "visible_token_limit"
    assert memories


def test_nativemem_agent_can_choose_read_only_bm25(tmp_path):
    memory_dir = tmp_path / "memory"
    topic = memory_dir / "topics/education.md"
    topic.parent.mkdir(parents=True)
    topic.write_text(
        "# Education\n\n[2023-05-23] The user studied Business Administration [D1:1]\n"
    )
    tool_call = SimpleNamespace(
        id="call-bm25",
        function=SimpleNamespace(
            name="bm25_search",
            arguments='{"query":"Business degree","top_k":10}',
        ),
    )
    responses = iter([
        SimpleNamespace(content=None, tool_calls=[tool_call]),
        SimpleNamespace(
            content="<answer>Business Administration</answer>", tool_calls=[]
        ),
    ])

    def create(**_kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=next(responses))])

    backend = SimpleNamespace(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        model="gpt-5.5",
        log_usage=lambda *_args, **_kwargs: None,
    )

    memories, _steps, answer, trace = retrieval.collect_answer(
        backend, {"question": "What degree?"}, memory_dir, {}
    )

    assert answer == "Business Administration"
    assert "Business Administration" in memories[0]["text"]
    assert trace[0]["type"] == "bm25_search"
    assert not (memory_dir / ".nativemem-bm25.json").exists()


def test_nativemem_agent_forwards_time_window_to_embedding_search(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    topic = memory_dir / "topics/art.md"
    topic.parent.mkdir(parents=True)
    topic.write_text(
        "# Art\n\n"
        "[2023-05-08] Melanie painted a sunrise [D1:4]\n"
        "[2024-01-12] Melanie painted another sunrise [D2:4]\n",
        encoding="utf-8",
    )

    class FakeEncoder:
        def encode(self, texts, **_kwargs):
            return [[1.0, 0.0] for _text in texts]

    monkeypatch.setattr(
        retrieval_tools,
        "MemoryEmbeddingIndex",
        lambda root: MemoryEmbeddingIndex(root, encoder=FakeEncoder()),
    )
    tool_call = SimpleNamespace(
        id="call-embedding",
        function=SimpleNamespace(
            name="embedding_search",
            arguments=json.dumps({
                "query": "sunrise painting",
                "top_k": 10,
                "date_from": "2023",
                "date_to": "2023",
            }),
        ),
    )
    responses = iter([
        SimpleNamespace(content=None, tool_calls=[tool_call]),
        SimpleNamespace(content="<answer>spring sunrise</answer>", tool_calls=[]),
    ])

    def create(**_kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=next(responses))])

    backend = SimpleNamespace(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        model="gpt-5.5",
        log_usage=lambda *_args, **_kwargs: None,
    )

    memories, _steps, answer, trace = retrieval.collect_answer(
        backend, {"question": "What did Melanie paint in 2023?"}, memory_dir, {}
    )

    assert answer == "spring sunrise"
    assert "D1:4" in memories[0]["text"]
    assert "D2:4" not in memories[0]["text"]
    assert trace[0]["args"]["date_from"] == "2023"


def test_nativemem_inventory_and_file_reader_expose_sources(tmp_path):
    (tmp_path / "topics").mkdir()
    (tmp_path / "timeline/2023/05").mkdir(parents=True)
    (tmp_path / "sources").mkdir()
    (tmp_path / "topics/education.md").write_text("degree\n")
    (tmp_path / "timeline/2023/05/23.md").write_text("event\n")
    (tmp_path / "sources/D1.md").write_text("raw source\n")
    (tmp_path / "recent_events.jsonl").write_text('{"event_id":"ev_1"}\n')

    relative = [path.relative_to(tmp_path).as_posix() for path in retrieval.memory_files(tmp_path)]

    assert relative == [
        "recent_events.jsonl",
        "sources/D1.md",
        "timeline/2023/05/23.md",
        "topics/education.md",
    ]
    assert "raw source" not in retrieval.read_memory_file(tmp_path, "topics/education.md")
    assert retrieval.read_memory_file(tmp_path, "sources/D1.md") == "raw source\n"


def test_nativemem_read_only_shell_can_search_sources(tmp_path):
    source = tmp_path / "sources/thread.md"
    source.parent.mkdir()
    source.write_text("Calvin bought a drum machine.\n", encoding="utf-8")

    output = retrieval.execute_workspace_bash(
        "rg -n 'drum machine' sources", tmp_path
    )

    assert "sources/thread.md:1:Calvin bought a drum machine." in output


def test_nativemem_query_normalizes_a_symlinked_memory_root(tmp_path):
    memory_dir = tmp_path / "real-memory"
    memory_dir.mkdir()
    (memory_dir / "core.md").write_text("stable fact [D1:1]\n")
    alias = tmp_path / "memory-alias"
    alias.symlink_to(memory_dir, target_is_directory=True)

    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="<answer>stable fact</answer>", tool_calls=[]
    ))])
    backend = SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )),
        model="gpt-5.5",
        log_usage=lambda *_args, **_kwargs: None,
    )

    _memories, _steps, answer, _trace = retrieval.collect_answer(
        backend, {"question": "what fact?"}, alias, {}
    )

    assert answer == "stable fact"


def test_nativemem_ablation_conditions_change_one_view_or_source_tool(tmp_path):
    (tmp_path / "topics").mkdir()
    (tmp_path / "timeline").mkdir()
    (tmp_path / "sources").mkdir()
    (tmp_path / "topics/topic.md").write_text("topic\n")
    (tmp_path / "timeline/day.md").write_text("day\n")
    (tmp_path / "sources/thread.md").write_text("source\n")
    (tmp_path / "recent_events.jsonl").write_text('{"event_id":"ev_1"}\n')

    def visible(condition):
        return {
            path.relative_to(tmp_path).as_posix()
            for path in retrieval.memory_files(tmp_path, condition)
        }

    assert visible("dual_source") == {
        "topics/topic.md", "timeline/day.md", "sources/thread.md",
        "recent_events.jsonl"
    }
    assert visible("topic_source") == {
        "topics/topic.md", "sources/thread.md", "recent_events.jsonl"
    }
    assert visible("timeline_source") == {
        "timeline/day.md", "sources/thread.md", "recent_events.jsonl"
    }
    assert visible("dual_no_source") == {
        "topics/topic.md", "timeline/day.md", "recent_events.jsonl"
    }
    assert {
        path.relative_to(tmp_path).as_posix()
        for path in retrieval.memory_files(
            tmp_path, "dual_source", include_recent=False
        )
    } == {"topics/topic.md", "timeline/day.md", "sources/thread.md"}

    def tool_names(condition):
        return {tool["function"]["name"] for tool in retrieval.tools_for(condition)}

    def tool_properties(name):
        tool = next(
            item for item in retrieval.TOOL_DEFINITIONS if item["function"]["name"] == name
        )
        return set(tool["function"]["parameters"]["properties"])

    assert "search_memory" not in tool_names("native")
    assert "bash" not in tool_names("dual_source")
    assert {"bm25_search", "embedding_search"} <= tool_names("dual_source")
    assert "bm25_search" not in tool_names("timeline_source")
    assert "embedding_search" not in tool_names("timeline_source")
    assert "resolve_sources" not in tool_names("native")
    assert "resolve_sources" not in tool_names("dual_source")
    assert {"date_from", "date_to"} <= tool_properties("bm25_search")
    assert {"date_from", "date_to"} <= tool_properties("embedding_search")


def test_nativemem_source_verification_can_be_disabled(tmp_path):
    (tmp_path / "core.md").write_text("The stable answer is Shanghai.\n")
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="<answer>Shanghai</answer>", tool_calls=[]
        ))])

    backend = SimpleNamespace(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        model="gpt-5.5",
        log_usage=lambda *_args, **_kwargs: None,
    )

    _memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "Where?"},
        tmp_path,
        {},
        config=retrieval.QueryConfig(verify_sources=False),
    )

    tool_names = {
        tool["function"]["name"] for tool in requests[0]["tools"]
    }
    prompt = requests[0]["messages"][0]["content"]
    assert answer == "Shanghai"
    assert "resolve_sources" not in tool_names
    assert "resolve_sources" not in prompt
    assert "Source files remain directly accessible" in prompt
    assert trace[-1]["source_verification"] is False


def test_nativemem_query_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="max_rounds"):
        retrieval.QueryConfig(max_rounds=-1)


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
