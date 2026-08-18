import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.runners import reanswer_longmemeval as MOD
from src import retrieval
from src.agent_runtime import AgentResult
from src.retrieval import tools as retrieval_tools
from src.retrieval.embedding import MemoryEmbeddingIndex
from src.retrieval.claim_evidence_loop import ClaimEvidenceOrganizer


class ScriptedQueryAgent:
    def __init__(self, tool_calls, answer, *, turns=3, structured_output=None):
        self.tool_calls = tool_calls
        self.answer = answer
        self.turns = turns
        self.structured_output = structured_output
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
            structured_output=self.structured_output,
            num_turns=self.turns,
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            anthropic_equivalent_cost_usd=0.01,
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


def test_reflective_retrieval_records_general_recall_and_sufficiency(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "core.md").write_text("The user prefers tea.\n", encoding="utf-8")
    recall_row = {
        "event_id": "tea:1",
        "path": "core.md",
        "line": 1,
        "date": "",
        "content": "The user prefers tea.",
    }
    monkeypatch.setattr(
        "src.retrieval.agent.build_general_first_recall",
        lambda *args, **kwargs: (
            "<general_first_recall>tea</general_first_recall>",
            [recall_row],
            {
                "enabled": True,
                "version": "h-lite-u-20260817",
                "query_transform": "none_raw_question",
                "routing": "none_all_lanes",
                "lane_counts": {"lexical": 1, "semantic": 1, "timeline": 0, "relations": 0},
                "candidate_count": 1,
                "candidates": [],
            },
        ),
    )
    reflection = {
        "supported_facts": ["The user prefers tea."],
        "conflicts": [],
        "missing_information": [],
        "can_answer": True,
        "follow_up_queries": [],
    }
    agent = ScriptedQueryAgent([], "tea", turns=2, structured_output={
        **reflection, "answer": "tea",
    })
    backend, _logged = scripted_backend(agent)

    memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "What does the user prefer?"},
        memory_dir,
        {},
        config=retrieval.QueryConfig(reflective_retrieval_enabled=True),
    )

    assert answer == "tea"
    assert "retrieved_evidence" in agent.calls[0]["prompt"]
    assert "Do not classify benchmark question types" in agent.calls[0]["system_prompt"]
    assert any(row["type"] == "general_first_recall" for row in trace)
    assert any(row["type"] == "retrieval_reflection" for row in trace)
    assert trace[-1]["reflective_retrieval"]["reflection_count"] == 1
    assert memories[-1]["text"] == "The user prefers tea."


def test_reflective_retrieval_is_independent_from_pipeline_and_ledgers():
    with pytest.raises(ValueError, match="cannot be combined with pipeline"):
        retrieval.QueryConfig(
            pipeline_enabled=True,
            reflective_retrieval_enabled=True,
        )
    with pytest.raises(ValueError, match="independent experiment"):
        retrieval.QueryConfig(
            evidence_ledger_enabled=True,
            reflective_retrieval_enabled=True,
        )


def test_a0r_adaptive_workspace_is_mutable_and_does_not_gate_retrieval(tmp_path):
    memory_dir = tmp_path / "memory"
    topic_dir = memory_dir / "topics"
    topic_dir.mkdir(parents=True)
    (topic_dir / "projects.md").write_text(
        "The user leads project A.\n", encoding="utf-8"
    )
    first = {
        "current_interpretation": "Find projects the user leads.",
        "candidate_answers": [],
        "rejected_candidates": [],
        "open_questions": ["Which named project is led?"],
        "revision_reason": None,
    }
    second = {
        "current_interpretation": "Project A is supported.",
        "candidate_answers": [{
            "candidate_id": "A",
            "value": "project A",
            "status": "supported",
            "evidence": ["topics/projects.md:1"],
        }],
        "rejected_candidates": [],
        "open_questions": [],
        "revision_reason": "Retrieved direct leadership evidence.",
    }
    agent = ScriptedQueryAgent([
        ("update_adaptive_workspace", first),
        ("bm25_search", {"query": "leads project", "top_k": 3}),
        ("update_adaptive_workspace", second),
    ], "1", turns=4)
    backend, _logged = scripted_backend(agent)

    _memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "How many projects does the user lead?"},
        memory_dir,
        {},
        config=retrieval.QueryConfig(adaptive_workspace_enabled=True),
    )

    assert answer == "1"
    assert agent.tool_results[1]["is_error"] is False
    metrics = trace[-1]["adaptive_workspace"]
    assert metrics["update_count"] == 2
    assert metrics["revisions"] == 1
    assert metrics["final_state"]["candidate_answers"][0]["value"] == "project A"


def test_a0r_adaptive_workspace_is_independent_from_pipeline():
    with pytest.raises(ValueError, match="independent A0 extension"):
        retrieval.QueryConfig(
            pipeline_enabled=True,
            adaptive_workspace_enabled=True,
        )


def test_a0c_claim_evidence_state_is_revisable_and_audited(tmp_path):
    memory_dir = tmp_path / "memory"
    topic_dir = memory_dir / "topics"
    topic_dir.mkdir(parents=True)
    (topic_dir / "projects.md").write_text(
        "The user leads project A.\n", encoding="utf-8"
    )
    candidate = {
        "candidate_id": "A",
        "claim": "The user leads project A.",
        "status": "uncertain",
        "evidence_refs": ["topics/projects.md:1"],
        "counterevidence_refs": [],
        "relations": [],
        "reason": "The retrieved summary needs source confirmation.",
        "answer_impact": "Would change the count from 0 to 1.",
        "resolvable": True,
        "next_check": "Verify the source behind topics/projects.md:1.",
    }
    first = {
        "question_target": "Count projects the user leads.",
        "candidates": [candidate],
        "remaining_gaps": ["Confirm project A leadership."],
        "ready_to_answer": False,
        "revision_reason": None,
    }
    supported = dict(candidate)
    supported.update({
        "status": "supported",
        "reason": "Direct leadership evidence is visible.",
        "resolvable": False,
        "next_check": None,
    })
    second = {
        "question_target": "Count projects the user leads.",
        "candidates": [supported],
        "remaining_gaps": [],
        "ready_to_answer": True,
        "revision_reason": "Direct evidence resolved candidate A.",
    }
    agent = ScriptedQueryAgent([
        ("bm25_search", {"query": "leads project", "top_k": 3}),
        ("update_claim_evidence_state", first),
        ("update_claim_evidence_state", second),
    ], "1", turns=4)
    backend, _logged = scripted_backend(agent)

    _memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "How many projects does the user lead?"},
        memory_dir,
        {},
        config=retrieval.QueryConfig(claim_evidence_state_enabled=True),
    )

    assert answer == "1"
    metrics = trace[-1]["claim_evidence_state"]
    assert metrics["update_count"] == 2
    assert metrics["revisions"] == 1
    assert metrics["mechanism_active"] is True
    assert metrics["final_ready"] is True
    assert metrics["final_state"]["candidates"][0]["status"] == "supported"
    assert "you decide which memory views" in agent.calls[0]["system_prompt"]


def test_a0c_claim_evidence_state_is_independent_from_other_experiments():
    with pytest.raises(ValueError, match="independent A0 extension"):
        retrieval.QueryConfig(
            pipeline_enabled=True,
            claim_evidence_state_enabled=True,
        )
    with pytest.raises(ValueError, match="independent A0 extension"):
        retrieval.QueryConfig(
            adaptive_workspace_enabled=True,
            claim_evidence_state_enabled=True,
        )


def test_c4_eager_organizes_each_novel_batch_and_reinjects_latest_state(tmp_path):
    memory_dir = tmp_path / "memory"
    (memory_dir / "topics").mkdir(parents=True)
    (memory_dir / "topics/storage.md").write_text(
        "The user needs more storage and values centralized backups.\n",
        encoding="utf-8",
    )
    patch = {
        "question_target": "Recommend whether to buy storage now.",
        "upsert_candidates": [{
            "candidate_id": "buy-now",
            "claim": "Buying now fits the user's current storage need.",
            "status": "supported",
            "evidence_refs": ["topics/storage.md:1"],
            "counterevidence_refs": [],
            "relations": [],
            "reason": "Current need and backup preference support the option.",
            "answer_impact": "Supports recommending a purchase now.",
            "verification_priority": "low",
        }],
        "remove_candidate_ids": [],
        "remaining_gaps": [],
        "ready_to_answer": True,
        "revision_reason": "New direct preference evidence.",
    }

    class C4Agent(ScriptedQueryAgent):
        def run(self, **kwargs):
            if kwargs.get("output_schema") is not None:
                self.calls.append(kwargs)
                return AgentResult(
                    text="", structured_output=patch, num_turns=1,
                    input_tokens=30, output_tokens=10,
                    cache_creation_input_tokens=0, cache_read_input_tokens=0,
                    anthropic_equivalent_cost_usd=0.002,
                    duration_ms=10, duration_api_ms=8, stop_reason="end_turn",
                    session_id="organizer",
                )
            if not kwargs.get("tools"):
                self.calls.append(kwargs)
                return AgentResult(
                    text="<answer>Buy now.</answer>", structured_output=None,
                    num_turns=1, input_tokens=20, output_tokens=5,
                    cache_creation_input_tokens=0, cache_read_input_tokens=0,
                    anthropic_equivalent_cost_usd=0.001,
                    duration_ms=10, duration_api_ms=8, stop_reason="end_turn",
                    session_id="final",
                )
            return super().run(**kwargs)

    agent = C4Agent([
        ("bash", {"command": "rg -n 'storage|backup' topics"}),
        ("bash", {"command": "rg -n 'storage|backup' topics"}),
        ("bash", {"command": "wc -c topics/storage.md"}),
        ("report_retrieval_event", {
            "event": "evidence_conflict",
            "reason": "The new evidence changes which candidate is supported.",
        }),
    ], "Buy now.", turns=3)
    backend, logged = scripted_backend(agent)

    _memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "Should the user buy storage now?"},
        memory_dir,
        {},
        config=retrieval.QueryConfig(
            claim_evidence_loop_enabled=True,
            claim_evidence_loop_initial_batch_size=1,
        ),
    )

    assert answer == "Buy now."
    metrics = trace[-1]["claim_evidence_loop"]
    assert metrics["organizer_calls"] == 3
    assert metrics["state_versions"] == 3
    assert metrics["mechanism_active"] is True
    assert metrics["final_state"]["candidates"][0]["candidate_id"] == "buy-now"
    tool_rows = [row for row in trace if row["type"] == "bash"]
    assert [row["organizer_triggered"] for row in tool_rows] == [
        True, False, False,
    ]
    event_rows = [
        row for row in trace if row["type"] == "report_retrieval_event"
    ]
    assert len(event_rows) == 1
    assert event_rows[0]["organizer_triggered"] is True
    assert all("current_claim_evidence_state" in result["content"][0]["text"]
               for result in agent.tool_results)
    assert [phase for _result, phase in logged] == [
        "memory_qa_c4_organizer", "memory_qa_c4_organizer",
        "memory_qa", "memory_qa_c4_organizer", "memory_qa_c4_final",
    ]


def test_c4_eager_is_independent_from_a0c_and_pipeline():
    with pytest.raises(ValueError, match="independent A0 extension"):
        retrieval.QueryConfig(
            claim_evidence_loop_enabled=True,
            claim_evidence_state_enabled=True,
        )
    with pytest.raises(ValueError, match="independent A0 extension"):
        retrieval.QueryConfig(
            claim_evidence_loop_enabled=True,
            pipeline_enabled=True,
        )


def test_c4_event_loop_uses_initial_scale_once_then_waits_for_events(tmp_path):
    organizer = ClaimEvidenceOrganizer(
        runtime=SimpleNamespace(), question="q", question_date="",
        memory_dir=tmp_path, initial_batch_size=2,
    )
    assert organizer.queue("first", tool_name="bash") is False
    assert organizer.queue("first", tool_name="bash") is False
    assert organizer.queue("second", tool_name="bm25_search") is True
    organizer.pending_batches.clear()
    organizer.organizer_calls = 2
    assert organizer.queue("third", tool_name="read_memory_file") is False
    assert organizer.queue("fourth", tool_name="bash") is False
    metrics = organizer.metrics()
    assert metrics["mode"] == "c4-event-driven"
    assert metrics["initial_batch_size"] == 2
    assert metrics["pending_unorganized_batches"] == 2


def test_c5a_queues_all_evidence_without_initial_organizer(tmp_path):
    organizer = ClaimEvidenceOrganizer(
        runtime=SimpleNamespace(), question="q", question_date="",
        memory_dir=tmp_path, initial_batch_size=2,
        initial_organizer_enabled=False,
    )
    assert organizer.queue("first", tool_name="bash") is False
    assert organizer.queue("second", tool_name="bm25_search") is False
    assert organizer.queue("third", tool_name="read_memory_file") is False
    metrics = organizer.metrics()
    assert metrics["mode"] == "c5a-final-only"
    assert metrics["initial_organizer_enabled"] is False
    assert metrics["organizer_calls"] == 0
    assert metrics["pending_unorganized_batches"] == 3


def test_c5b_emits_one_nonbinding_reminder_per_evidence_cycle(tmp_path):
    organizer = ClaimEvidenceOrganizer(
        runtime=SimpleNamespace(), question="q", question_date="",
        memory_dir=tmp_path, initial_organizer_enabled=False,
        event_guidance_enabled=True,
    )
    organizer.queue("first", tool_name="bash")
    organizer.queue("second", tool_name="bm25_search")
    assert organizer.take_event_reminder() is False
    organizer.queue("third", tool_name="read_memory_file")
    assert organizer.take_event_reminder() is True
    assert organizer.take_event_reminder() is False
    metrics = organizer.metrics()
    assert metrics["event_guidance_enabled"] is True
    assert metrics["event_reminders"] == 1


def test_c4_event_config_rejects_invalid_initial_batch_size():
    with pytest.raises(ValueError, match="initial batch size must be positive"):
        retrieval.QueryConfig(claim_evidence_loop_initial_batch_size=0)


def test_query_agent_can_enable_programmatic_evidence_ledger(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "core.md").write_text(
        "The user prefers tea.\n", encoding="utf-8"
    )
    (memory_dir / "topics").mkdir()
    (memory_dir / "topics" / "drinks.md").write_text(
        "On 2023-05-01 the user bought tea. D1:1\n", encoding="utf-8"
    )
    agent = ScriptedQueryAgent(
        [("read_memory_file", {"path": "topics/drinks.md"})], "tea", turns=2
    )
    backend, _logged = scripted_backend(agent)

    _memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "What drink?"},
        memory_dir,
        {},
        config=retrieval.QueryConfig(evidence_ledger_enabled=True),
    )

    assert answer == "tea"
    assert "<evidence_ledger" not in agent.calls[0]["prompt"]
    assert "<evidence_ledger_delta>" in agent.tool_results[0]["content"][0]["text"]
    metrics = trace[-1]["evidence_ledger"]
    assert metrics["enabled"] is True
    assert metrics["entries"] == 1
    assert metrics["initial_records"] == 1


def test_query_agent_can_use_grounded_reasoning_workspace(tmp_path):
    memory_dir = tmp_path / "memory"
    topic_dir = memory_dir / "topics"
    topic_dir.mkdir(parents=True)
    topic_path = topic_dir / "roles.md"
    topic_path.write_text(
        "The user is a Senior Engineer. D1:1\n", encoding="utf-8"
    )
    event = retrieval_tools.MemoryBM25Index(
        memory_dir, persist=False
    ).events[0]
    payload = {
        "reasoning_mode": "relation_check",
        "requirements": [{
            "claim": "manager role must be supported",
            "status": "missing",
            "evidence_ids": [],
        }],
        "candidates": [{
            "candidate_id": "senior_engineer",
            "value": "Senior Engineer",
            "predicate": "recorded role",
            "event_time": None,
            "time_precision": "none",
            "evidence_ids": [event.event_id],
            "compatible": False,
            "exclusion_reason": "different role",
        }],
        "conflicts": [],
        "resolution": {
            "policy": "none",
            "selected_candidate_ids": [],
            "rejected_candidate_ids": [],
            "justification_evidence_ids": [],
        },
        "sufficiency": "insufficient",
    }
    agent = ScriptedQueryAgent(
        [
            ("bm25_search", {"query": "Senior Engineer"}),
            ("update_reasoning_ledger", payload),
        ],
        "Insufficient information.",
        turns=3,
    )
    backend, _logged = scripted_backend(agent)

    _memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "How many engineers as manager?"},
        memory_dir,
        {},
        config=retrieval.QueryConfig(
            evidence_ledger_enabled=True,
            reasoning_ledger_enabled=True,
        ),
    )

    assert answer == "Insufficient information."
    assert "Before answering, call update_reasoning_ledger" in (
        agent.calls[0]["system_prompt"]
    )
    assert trace[-1]["reasoning_ledger"]["updates"] == 1
    assert trace[-1]["reasoning_ledger"]["state"]["sufficiency"] == "insufficient"


def test_query_agent_can_plan_before_retrieval(tmp_path):
    memory_dir = tmp_path / "memory"
    topic_dir = memory_dir / "topics"
    topic_dir.mkdir(parents=True)
    (topic_dir / "drinks.md").write_text(
        "On 2023-05-01 the user bought tea. D1:1\n", encoding="utf-8"
    )
    plan = {
        "answer_shape": "direct_fact",
        "target_predicate": "drink the user bought",
        "entities": ["user", "drink"],
        "time_scope": None,
        "inclusion_rules": ["completed purchases"],
        "exclusion_rules": ["planned purchases"],
        "required_evidence": ["drink and purchase relation"],
        "search_queries": ["bought drink"],
        "source_check": "verify a merged relation if encountered",
        "stop_rule": "stop after the purchase relation is supported",
        "revision_reason": None,
    }
    agent = ScriptedQueryAgent(
        [
            ("set_retrieval_plan", plan),
            ("read_memory_file", {"path": "topics/drinks.md"}),
        ],
        "tea",
        turns=3,
    )
    backend, _logged = scripted_backend(agent)

    _memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "What drink?"},
        memory_dir,
        {},
        config=retrieval.QueryConfig(
            evidence_ledger_enabled=True,
            retrieval_plan_enabled=True,
        ),
    )

    assert answer == "tea"
    assert "set_retrieval_plan" in agent.calls[0]["system_prompt"]
    assert trace[-1]["retrieval_plan"]["updates"] == 1
    assert trace[-1]["retrieval_calls"] == 1


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


def test_runtime_keeps_builder_and_query_agents_separate():
    builder = object()
    answerer = object()

    result = retrieval.create_runtime(
        "https://builder.example/v1",
        model="deepseek-builder",
        api_key="builder-key",
        agent=builder,
        query_model="gpt-answerer",
        query_agent=answerer,
    )

    assert result.builder_agent is builder
    assert result.builder_model == "deepseek-builder"
    assert result.agent is answerer
    assert result.model == "gpt-answerer"


def test_runtime_constructs_each_retrieval_index_once_across_threads():
    runtime = retrieval.Runtime(agent=object(), model="test-model")
    created = []

    def factory():
        time.sleep(0.05)
        value = object()
        created.append(value)
        return value

    with ThreadPoolExecutor(max_workers=3) as executor:
        indexes = list(
            executor.map(
                lambda _number: runtime.get_retrieval_index(
                    ("embedding", "/memory", ("topics/a.md",)), factory
                ),
                range(3),
            )
        )

    assert len(created) == 1
    assert indexes == [created[0]] * 3


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


def test_answer_one_uses_portable_sibling_memory_without_rewriting_checkpoint(
    monkeypatch, tmp_path
):
    checkpoint_dir = tmp_path / "portable" / "item"
    memory_dir = checkpoint_dir / "memory"
    memory_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "checkpoint.json"
    checkpoint_path.write_text(
        __import__("json").dumps({
            "status": "built",
            "dataset_index": 0,
            "question_id": "q0",
            "build": {"status": "complete"},
            "paths": {"memory_dir": "/missing/original/memory"},
        }),
        encoding="utf-8",
    )
    source = {"dataset_index": 0, "checkpoint": str(checkpoint_path), "answer": ""}
    dataset = [{
        "question_id": "q0", "question": "where?", "question_type": "test",
        "answer": "gold",
    }]
    backend = SimpleNamespace(build_turn_index=lambda _conversation: {})
    monkeypatch.setattr(MOD.lme, "memory_is_valid", lambda path: path == memory_dir)
    monkeypatch.setattr(MOD.lme, "to_conversation", lambda _item, _index: {})
    monkeypatch.setattr(
        "scripts.runners.longmemeval.execution.collect_answer",
        lambda *_args, **_kwargs: ([{"text": "evidence", "date": ""}], 1, "answer", []),
    )

    record = MOD.answer_one(
        backend,
        dataset,
        source,
        tmp_path / "output",
        model="model",
        query_config=retrieval.QueryConfig(),
    )

    assert record["answer"] == "answer"
    assert __import__("json").loads(checkpoint_path.read_text())["paths"][
        "memory_dir"
    ] == "/missing/original/memory"


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
        lambda root, *, files: MemoryEmbeddingIndex(
            root, files=files, encoder=FakeEncoder()
        ),
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


def test_scriptorium_inventory_and_file_reader_expose_sources(tmp_path):
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


def test_scriptorium_file_reader_exposes_line_window_parameters(tmp_path):
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


def test_scriptorium_read_only_shell_can_search_sources(tmp_path):
    source = tmp_path / "sources/thread.md"
    source.parent.mkdir()
    source.write_text(
        "Calvin bought a drum machine.\n", encoding="utf-8"
    )

    output = retrieval.execute_workspace_bash(
        "rg -n 'drum machine' sources", tmp_path
    )

    assert "sources/thread.md:1:Calvin bought a drum machine." in output


def test_scriptorium_ablation_conditions_change_views_and_tools(tmp_path):
    (tmp_path / "topics").mkdir()
    (tmp_path / "timeline").mkdir()
    (tmp_path / "sources").mkdir()
    (tmp_path / "topics/topic.md").write_text("topic\n")
    (tmp_path / "timeline/day.md").write_text("day\n")
    (tmp_path / "sources/thread.md").write_text("source\n")
    (tmp_path / "core.md").write_text("core\n")
    (tmp_path / "recent_events.jsonl").write_text('{"event_id":"ev_1"}\n')
    (tmp_path / "relations.json").write_text('{"relations":[]}\n')

    def visible(condition):
        return {
            path.relative_to(tmp_path).as_posix()
            for path in retrieval.memory_files(tmp_path, condition)
        }

    assert visible("dual_source") == {
        "core.md",
        "topics/topic.md",
        "timeline/day.md",
        "sources/thread.md",
        "recent_events.jsonl",
    }
    assert visible("topic_source") == {
        "core.md",
        "topics/topic.md",
        "sources/thread.md",
        "recent_events.jsonl",
    }
    assert visible("timeline_source") == {
        "core.md",
        "timeline/day.md",
        "sources/thread.md",
        "recent_events.jsonl",
    }
    assert visible("dual_no_source") == {
        "core.md",
        "topics/topic.md",
        "timeline/day.md",
        "recent_events.jsonl",
    }
    assert {
        path.relative_to(tmp_path).as_posix()
        for path in retrieval.memory_files(
            tmp_path, "dual_source", include_recent=False
        )
    } == {
        "core.md",
        "topics/topic.md",
        "timeline/day.md",
        "sources/thread.md",
    }

    explicit = {
        path.relative_to(tmp_path).as_posix()
        for path in retrieval.memory_files(
            tmp_path, components="topics,relations"
        )
    }
    assert explicit == {"topics/topic.md", "relations.json"}
    assert "core.md" not in explicit
    assert {
        path.relative_to(tmp_path).as_posix()
        for path in retrieval.memory_files(
            tmp_path, components=("core", "recent")
        )
    } == {"core.md", "recent_events.jsonl"}

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
    assert "bash" not in {
        tool["function"]["name"]
        for tool in retrieval.tools_for(
            "native", components=("topics", "core")
        )
    }
    assert "bm25_search" not in {
        tool["function"]["name"]
        for tool in retrieval.tools_for(
            "native", components=("timeline", "relations")
        )
    }
    assert {"date_from", "date_to"} <= tool_properties("bm25_search")
    assert {"date_from", "date_to"} <= tool_properties(
        "embedding_search"
    )


def test_search_tools_cannot_see_files_hidden_by_ablation(
    tmp_path, monkeypatch
):
    topic = tmp_path / "topics/visible.md"
    source = tmp_path / "sources/hidden.md"
    topic.parent.mkdir()
    source.parent.mkdir()
    topic.write_text(
        "# Visible\n\n[2023-01-01] ordinary visible fact [D1:1]\n",
        encoding="utf-8",
    )
    source.write_text(
        "# Hidden\n\n"
        "<!-- source-id:D2:1 -->\n"
        "[2023-01-02] secret-source-only-term\n",
        encoding="utf-8",
    )
    files = retrieval.memory_files(tmp_path, "dual_no_source")

    class ScopeEncoder:
        def encode(self, texts, **_kwargs):
            return [
                [1.0, 0.0]
                if "secret-source-only-term" in text
                else [0.0, 1.0]
                for text in texts
            ]

    monkeypatch.setattr(
        retrieval_tools,
        "MemoryEmbeddingIndex",
        lambda root, *, files: MemoryEmbeddingIndex(
            root, files=files, encoder=ScopeEncoder()
        ),
    )
    backend, _logged = scripted_backend(ScriptedQueryAgent([], ""))

    for tool_name in ("bm25_search", "embedding_search"):
        output, executed, _accepted = retrieval_tools.execute_tool_call(
            backend,
            tool_name,
            {"query": "secret-source-only-term"},
            memory_dir=tmp_path,
            files=files,
            condition="dual_no_source",
            include_recent=True,
            indexes={},
        )
        assert executed is True
        assert "sources/" not in output
        assert "secret-source-only-term" not in output


def test_scriptorium_source_verification_can_be_disabled(tmp_path):
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
    assert "Source memory is masked for this run" in request["prompt"]
    assert trace[-1]["source_verification"] is False


def test_component_mask_hides_sources_from_prompt_and_tools(tmp_path):
    topic = tmp_path / "topics/visible.md"
    source = tmp_path / "sources/hidden.md"
    topic.parent.mkdir()
    source.parent.mkdir()
    topic.write_text("visible topic\n", encoding="utf-8")
    source.write_text("hidden source\n", encoding="utf-8")
    agent = ScriptedQueryAgent([], "visible", turns=1)
    backend, _logged = scripted_backend(agent)

    _memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "What is visible?"},
        tmp_path,
        {},
        config=retrieval.QueryConfig(memory_components="topics"),
    )

    request = agent.calls[0]
    assert answer == "visible"
    assert "Source memory is masked for this run" in request["prompt"]
    assert "sources/hidden.md" not in request["prompt"]
    assert "bash" not in {tool.name for tool in request["tools"]}
    assert trace[-1]["source_verification"] is False
    assert trace[-1]["source_verification_requested"] is True
    assert trace[-1]["memory_components"] == ["topics"]


def test_scriptorium_query_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="max_turns"):
        retrieval.QueryConfig(max_turns=0)
    with pytest.raises(ValueError, match="max_budget_usd"):
        retrieval.QueryConfig(max_budget_usd=0)
    assert retrieval.QueryConfig(
        memory_components="topics,core,topics"
    ).memory_components == ("topics", "core")
    with pytest.raises(ValueError, match="unknown memory components"):
        retrieval.QueryConfig(memory_components="topics,unknown")
    with pytest.raises(ValueError, match="evidence_ledger_max_entries"):
        retrieval.QueryConfig(evidence_ledger_max_entries=0)
    with pytest.raises(ValueError, match="cannot be combined"):
        retrieval.QueryConfig(
            pipeline_enabled=True,
            evidence_ledger_enabled=True,
        )
    with pytest.raises(ValueError, match="requires pipeline"):
        retrieval.QueryConfig(pipeline_evidence_packet_enabled=True)
    with pytest.raises(ValueError, match="p0b or m3"):
        retrieval.QueryConfig(
            pipeline_enabled=True,
            pipeline_version="p0b-r1",
            pipeline_evidence_packet_enabled=True,
        )
    assert retrieval.QueryConfig(
        pipeline_enabled=True,
        pipeline_version="m3",
        pipeline_evidence_packet_enabled=True,
    ).pipeline_version == "m3"
    with pytest.raises(ValueError, match="requires the P1 evidence packet"):
        retrieval.QueryConfig(
            pipeline_enabled=True,
            pipeline_version="p0b",
            pipeline_evidence_gate_enabled=True,
        )
    with pytest.raises(ValueError, match="P3 supplement requires"):
        retrieval.QueryConfig(
            pipeline_enabled=True,
            pipeline_version="p0b",
            pipeline_supplement_enabled=True,
        )
    with pytest.raises(ValueError, match="branch directly from P1"):
        retrieval.QueryConfig(
            pipeline_enabled=True,
            pipeline_version="p0b",
            pipeline_evidence_packet_enabled=True,
            pipeline_evidence_gate_enabled=True,
            pipeline_supplement_enabled=True,
        )


def test_p0_pipeline_answers_without_agent_retrieval_tools(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    agent = ScriptedQueryAgent([], "tea", turns=1)
    backend, _logged = scripted_backend(agent)

    def fake_pipeline(*_args, **_kwargs):
        rows = [{
            "event_id": "ev1",
            "path": "topics/drinks.md",
            "line": 2,
            "date": "2025-01-01",
            "content": "The user prefers tea.",
            "refs": ["D1:1"],
        }]
        return "<pipeline_context>tea</pipeline_context>", rows, {
            "enabled": True,
            "version": "p0-v2",
            "candidate_count": 1,
        }

    monkeypatch.setattr("src.retrieval.agent.build_pipeline_context", fake_pipeline)
    memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "What drink?"},
        memory_dir,
        {},
        config=retrieval.QueryConfig(pipeline_enabled=True),
    )

    assert answer == "tea"
    assert agent.calls[0]["tools"] == []
    assert "<pipeline_context>tea</pipeline_context>" in agent.calls[0]["prompt"]
    assert "No retrieval tools are available" in agent.calls[0]["prompt"]
    assert agent.calls[0]["max_turns"] == 3
    assert memories == [{"text": "The user prefers tea.", "date": "2025-01-01"}]
    assert trace[-1]["tool_calls"] == 0
    assert trace[-1]["pipeline"]["version"] == "p0-v2"


def test_p1_packet_uses_same_pipeline_rows_without_tools(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    agent = ScriptedQueryAgent([], "tea", turns=1)
    backend, _logged = scripted_backend(agent)

    def fake_pipeline(*_args, **_kwargs):
        rows = [{
            "event_id": "ev1", "path": "topics/drinks.md", "line": 2,
            "date": "2025-01-01", "content": "The user prefers tea.",
            "refs": ["D1:1"], "pipeline_rank": 1,
        }]
        return "<pipeline_context>tea</pipeline_context>", rows, {
            "enabled": True, "version": "p0b", "candidate_count": 1,
            "question_contract": {
                "answer_shape": "direct_fact",
                "coverage_axis": "best_supported_fact",
            },
        }

    monkeypatch.setattr("src.retrieval.agent.build_pipeline_context", fake_pipeline)
    memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "What drink?"},
        memory_dir,
        {},
        config=retrieval.QueryConfig(
            pipeline_enabled=True,
            pipeline_version="p0b",
            pipeline_evidence_packet_enabled=True,
        ),
    )

    assert answer == "tea"
    assert agent.calls[0]["tools"] == []
    assert "<evidence_packet" in agent.calls[0]["prompt"]
    assert "The user prefers tea." in agent.calls[0]["prompt"]
    assert memories == [{"text": "The user prefers tea.", "date": "2025-01-01"}]
    assert trace[-1]["pipeline"]["evidence_packet"]["candidate_count"] == 1


def test_p2_gate_preserves_pipeline_rows_and_adds_no_calls(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    agent = ScriptedQueryAgent([], "tea", turns=1)
    backend, _logged = scripted_backend(agent)

    def fake_pipeline(*_args, **_kwargs):
        rows = [{
            "event_id": "ev1", "path": "topics/drinks.md", "line": 2,
            "date": "2025-01-01", "content": "The user prefers tea.",
            "refs": ["D1:1"], "pipeline_rank": 1,
        }]
        return "<pipeline_context>tea</pipeline_context>", rows, {
            "enabled": True, "version": "p0b", "candidate_count": 1,
            "question_contract": {
                "answer_shape": "direct_fact",
                "coverage_axis": "best_supported_fact",
                "temporal": {},
            },
        }

    monkeypatch.setattr("src.retrieval.agent.build_pipeline_context", fake_pipeline)
    memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "What drink?"},
        memory_dir,
        {},
        config=retrieval.QueryConfig(
            pipeline_enabled=True,
            pipeline_version="p0b",
            pipeline_evidence_packet_enabled=True,
            pipeline_evidence_gate_enabled=True,
        ),
    )

    assert answer == "tea"
    assert agent.calls[0]["tools"] == []
    assert "<evidence_packet" in agent.calls[0]["prompt"]
    assert "<evidence_gate" in agent.calls[0]["prompt"]
    assert memories == [{"text": "The user prefers tea.", "date": "2025-01-01"}]
    gate = trace[-1]["pipeline"]["evidence_gate"]
    assert gate["independent_llm_calls"] == 0
    assert gate["supplemental_retrievals"] == 0


def test_p3_runs_at_most_one_supplement_for_detected_gap(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    agent = ScriptedQueryAgent([], "2", turns=1)
    backend, _logged = scripted_backend(agent)
    calls = []

    def fake_pipeline(*_args, **_kwargs):
        rows = [{
            "event_id": "ev1", "path": "topics/events.md", "line": 2,
            "date": "2025-01-01", "content": "The user attended one event.",
            "refs": ["D1:1"], "pipeline_rank": 1,
        }]
        return "old", rows, {
            "enabled": True, "version": "p0b", "candidate_count": 1,
            "question_contract": {
                "answer_shape": "count", "coverage_axis": "distinct_events",
                "temporal": {},
            },
        }

    def fake_supplement(*_args, **_kwargs):
        calls.append(1)
        return [{
            "event_id": "ev2", "path": "topics/events.md", "line": 4,
            "date": "2025-01-02", "content": "The user attended another event.",
            "refs": ["D2:1"], "pipeline_rank": 2, "supplemental": True,
        }], {
            "triggered": True, "lookup_count": 1, "max_new": 4,
            "added_count": 1, "added_event_ids": ["ev2"],
        }

    monkeypatch.setattr("src.retrieval.agent.build_pipeline_context", fake_pipeline)
    monkeypatch.setattr("src.retrieval.agent.supplement_pipeline_rows", fake_supplement)
    memories, _steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": "How many events did the user attend?"},
        memory_dir,
        {},
        config=retrieval.QueryConfig(
            pipeline_enabled=True,
            pipeline_version="p0b",
            pipeline_evidence_packet_enabled=True,
            pipeline_supplement_enabled=True,
        ),
    )

    assert answer == "2"
    assert len(calls) == 1
    assert len(memories) == 2
    assert "The user attended another event." in agent.calls[0]["prompt"]
    assert trace[-1]["pipeline"]["supplement"]["lookup_count"] == 1
    assert trace[-1]["tool_calls"] == 0


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
