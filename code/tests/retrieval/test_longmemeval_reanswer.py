import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.nativemem import reanswer_longmemeval as MOD
from src import retrieval
from src.agent_runtime import AgentResult
from src.retrieval import tools as retrieval_tools
from src.retrieval.embedding import MemoryEmbeddingIndex


class ScriptedQueryAgent:
    def __init__(self, tool_calls, answer, *, turns=3):
        self.tool_calls = tool_calls
        self.answer = answer
        self.turns = turns
        self.calls = []
        self.tool_results = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        tools = {definition.name: definition for definition in kwargs["tools"]}
        for name, arguments in self.tool_calls:
            self.tool_results.append(
                asyncio.run(tools[name].handler(arguments))
            )
        return AgentResult(
            text=f"<answer>{self.answer}</answer>",
            structured_output=None,
            num_turns=self.turns,
            input_tokens=100,
            output_tokens=20,
            total_cost_usd=0.01,
            duration_ms=50,
            duration_api_ms=40,
            stop_reason="end_turn",
            session_id="test-session",
        )


def scripted_backend(agent):
    logged = []
    backend = SimpleNamespace(
        agent=agent,
        model="test-model",
        log_agent_result=lambda result, phase: logged.append((result, phase)),
        execute_tool=lambda _name, args, root, hide_raw: (
            retrieval.execute_workspace_bash(args["command"], Path(root))
        ),
    )
    return backend, logged


def test_query_agent_uses_framework_tools_and_reports_usage(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "education.md").write_text(
        "The user moved to Shanghai.\n", encoding="utf-8"
    )
    agent = ScriptedQueryAgent(
        [
            ("bash", {"command": "rg -n Shanghai ."}),
            ("bm25_search", {"query": "moved Shanghai", "top_k": 3}),
        ],
        "Shanghai",
    )
    backend, logged = scripted_backend(agent)

    memories, steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "Where did the user move?"},
        memory_dir,
        {},
    )

    assert answer == "Shanghai"
    assert steps == 3
    assert [
        row["type"] for row in trace if row["type"] != "termination"
    ] == ["bash", "bm25_search"]
    assert trace[1]["nonempty"] is False
    assert trace[-1]["tool_calls"] == 2
    assert trace[-1]["termination_reason"] == "end_turn"
    assert len(memories) == 1
    assert memories[0]["text"].endswith("Shanghai.")
    assert logged[0][1] == "memory_qa"


def test_create_runtime_uses_explicit_agent_without_mutating_environment(
    monkeypatch,
):
    agent = object()
    monkeypatch.setenv("MODEL", "inherited-model")
    monkeypatch.setenv("BUILDER_KEY", "inherited-key")

    result = retrieval.create_runtime(
        "http://127.0.0.1:12345",
        model="deepseek-v4",
        api_key="test-key",
        agent=agent,
    )

    assert result.agent is agent
    assert result.model == "deepseek-v4"
    assert os.environ["MODEL"] == "inherited-model"
    assert os.environ["BUILDER_KEY"] == "inherited-key"


def test_reanswer_dispatch_uses_retrieval_agent(monkeypatch, tmp_path):
    expected = ([{"text": "memory", "date": ""}], 2, "answer", [])
    seen = []

    def collect(backend, item, memory_dir, turn_index, **_kwargs):
        seen.append((backend, item, memory_dir, turn_index))
        return expected

    monkeypatch.setattr(retrieval, "collect_answer", collect)
    backend = SimpleNamespace()
    item = {"question": "where?", "question_date": "2023-05-30"}
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    turn_index = {"D1:1": {}}

    result = MOD.collect_answer(backend, item, memory_dir, turn_index)

    assert result == expected
    assert seen == [(backend, item, memory_dir, turn_index)]


def test_query_agent_receives_core_recent_and_framework_limits(tmp_path):
    memory_dir = tmp_path / "real-memory"
    memory_dir.mkdir()
    (memory_dir / "core.md").write_text(
        "The user's stable preferred language is Chinese.\n",
        encoding="utf-8",
    )
    (memory_dir / "recent_events.jsonl").write_text(
        '{"content":"The user started Project Atlas."}\n',
        encoding="utf-8",
    )
    alias = tmp_path / "memory-alias"
    alias.symlink_to(memory_dir, target_is_directory=True)
    agent = ScriptedQueryAgent([], "Chinese", turns=1)
    backend, _logged = scripted_backend(agent)

    memories, steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "What language does the user prefer?"},
        alias,
        {},
        config=retrieval.QueryConfig(max_turns=20, max_budget_usd=0.25),
    )

    request = agent.calls[0]
    assert request["cwd"] == memory_dir.resolve()
    assert request["max_turns"] == 20
    assert request["max_budget_usd"] == 0.25
    assert "stable preferred language is Chinese" in request["prompt"]
    assert "started Project Atlas" in request["prompt"]
    assert answer == "Chinese"
    assert steps == 1
    assert len(memories) == 2
    assert trace[0]["type"] == "initial_context"


def test_query_tool_reports_rejected_bash_and_continues(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "fact.md").write_text(
        "Business Administration\n", encoding="utf-8"
    )
    agent = ScriptedQueryAgent(
        [
            ("bash", {"command": "cat /outside/fact.md"}),
            ("bash", {"command": "cat fact.md"}),
        ],
        "Business Administration",
    )
    backend, _logged = scripted_backend(agent)

    memories, _steps, answer, trace = retrieval.collect_answer(
        backend, {"question": "What degree?"}, memory_dir, {}
    )

    assert agent.tool_results[0]["is_error"] is True
    assert agent.tool_results[1]["is_error"] is False
    assert trace[0]["accepted"] is False
    assert trace[0]["executed"] is False
    assert trace[0]["nonempty"] is False
    assert trace[1]["accepted"] is True
    assert answer == "Business Administration"
    assert len(memories) == 1


def test_query_agent_can_choose_read_only_bm25(tmp_path):
    memory_dir = tmp_path / "memory"
    topic = memory_dir / "topics/education.md"
    topic.parent.mkdir(parents=True)
    topic.write_text(
        "# Education\n\n"
        "[2023-05-23] The user studied Business Administration [D1:1]\n",
        encoding="utf-8",
    )
    agent = ScriptedQueryAgent(
        [("bm25_search", {"query": "Business degree", "top_k": 10})],
        "Business Administration",
    )
    backend, _logged = scripted_backend(agent)

    memories, _steps, answer, trace = retrieval.collect_answer(
        backend, {"question": "What degree?"}, memory_dir, {}
    )

    assert answer == "Business Administration"
    assert "Business Administration" in memories[0]["text"]
    assert trace[0]["type"] == "bm25_search"
    assert not (memory_dir / ".nativemem-bm25.json").exists()


def test_query_agent_forwards_time_window_to_embedding_search(
    tmp_path, monkeypatch
):
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
    agent = ScriptedQueryAgent(
        [(
            "embedding_search",
            {
                "query": "sunrise painting",
                "top_k": 10,
                "date_from": "2023",
                "date_to": "2023",
            },
        )],
        "spring sunrise",
    )
    backend, _logged = scripted_backend(agent)

    memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "What did Melanie paint in 2023?"},
        memory_dir,
        {},
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

    relative = [
        path.relative_to(tmp_path).as_posix()
        for path in retrieval.memory_files(tmp_path)
    ]

    assert relative == [
        "recent_events.jsonl",
        "sources/D1.md",
        "timeline/2023/05/23.md",
        "topics/education.md",
    ]
    assert "raw source" not in retrieval.read_memory_file(
        tmp_path, "topics/education.md"
    )
    assert retrieval.read_memory_file(
        tmp_path, "sources/D1.md"
    ) == "raw source\n"


def test_nativemem_file_reader_exposes_line_window_parameters(tmp_path):
    path = tmp_path / "topics/notes.md"
    path.parent.mkdir()
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    tool = next(
        definition
        for definition in retrieval.TOOL_DEFINITIONS
        if definition["function"]["name"] == "read_memory_file"
    )
    properties = tool["function"]["parameters"]["properties"]

    assert properties["offset"]["minimum"] == 1
    assert properties["limit"]["minimum"] == 1
    assert retrieval.read_memory_file(
        tmp_path, "topics/notes.md", offset=2, limit=1
    ) == "two\n"


def test_nativemem_read_only_shell_can_search_sources(tmp_path):
    source = tmp_path / "sources/thread.md"
    source.parent.mkdir()
    source.write_text(
        "Calvin bought a drum machine.\n", encoding="utf-8"
    )

    output = retrieval.execute_workspace_bash(
        "rg -n 'drum machine' sources", tmp_path
    )

    assert "sources/thread.md:1:Calvin bought a drum machine." in output


def test_nativemem_ablation_conditions_change_views_and_tools(tmp_path):
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
        "topics/topic.md",
        "timeline/day.md",
        "sources/thread.md",
        "recent_events.jsonl",
    }
    assert visible("topic_source") == {
        "topics/topic.md",
        "sources/thread.md",
        "recent_events.jsonl",
    }
    assert visible("timeline_source") == {
        "timeline/day.md",
        "sources/thread.md",
        "recent_events.jsonl",
    }
    assert visible("dual_no_source") == {
        "topics/topic.md",
        "timeline/day.md",
        "recent_events.jsonl",
    }
    assert {
        path.relative_to(tmp_path).as_posix()
        for path in retrieval.memory_files(
            tmp_path, "dual_source", include_recent=False
        )
    } == {"topics/topic.md", "timeline/day.md", "sources/thread.md"}

    def tool_names(condition):
        return {
            tool["function"]["name"]
            for tool in retrieval.tools_for(condition)
        }

    def tool_properties(name):
        tool = next(
            item
            for item in retrieval.TOOL_DEFINITIONS
            if item["function"]["name"] == name
        )
        return set(tool["function"]["parameters"]["properties"])

    assert "search_memory" not in tool_names("native")
    assert "bash" not in tool_names("dual_source")
    assert {"bm25_search", "embedding_search"} <= tool_names("dual_source")
    assert "bm25_search" not in tool_names("timeline_source")
    assert "embedding_search" not in tool_names("timeline_source")
    assert "resolve_sources" not in tool_names("native")
    assert {"date_from", "date_to"} <= tool_properties("bm25_search")
    assert {"date_from", "date_to"} <= tool_properties(
        "embedding_search"
    )


def test_nativemem_source_verification_can_be_disabled(tmp_path):
    (tmp_path / "core.md").write_text(
        "The stable answer is Shanghai.\n", encoding="utf-8"
    )
    agent = ScriptedQueryAgent([], "Shanghai", turns=1)
    backend, _logged = scripted_backend(agent)

    _memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "Where?"},
        tmp_path,
        {},
        config=retrieval.QueryConfig(verify_sources=False),
    )

    request = agent.calls[0]
    tool_names = {tool.name for tool in request["tools"]}
    assert answer == "Shanghai"
    assert "resolve_sources" not in tool_names
    assert "resolve_sources" not in request["prompt"]
    assert "Source files remain directly accessible" in request["prompt"]
    assert trace[-1]["source_verification"] is False


def test_nativemem_query_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="max_turns"):
        retrieval.QueryConfig(max_turns=0)
    with pytest.raises(ValueError, match="max_budget_usd"):
        retrieval.QueryConfig(max_budget_usd=0)


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
    sources = [
        {"dataset_index": 0},
        {"dataset_index": 1},
        {"dataset_index": 2},
    ]
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
