import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_v88_gpt55_beam.py"
FIXTURE = Path(__file__).parent / "fixtures" / "beam_hf_fixture.json"
SPEC = importlib.util.spec_from_file_location("run_v88_gpt55_beam", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _args(*, resume=False):
    return SimpleNamespace(
        model="gpt-5.5",
        base_url="http://127.0.0.1:8199/v1",
        api_key="x",
        request_concurrency=2,
        dataset_revision="main",
        resume=resume,
    )


class _Tracker:
    def reset(self, phase):
        self.phase = phase

    class _Bound:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def bind_thread(self, phase):
        return self._Bound()

    def snapshot(self, phase):
        return {
            "calls": 1,
            "tokens_in": 10,
            "tokens_out": 2,
            "llm_time_s": 0.1,
        }


class _Native:
    tracker = _Tracker()

    @staticmethod
    def build_turn_index(conv):
        return {turn["dia_id"]: turn
                for key, turns in conv.items()
                if key.startswith("session_") and not key.endswith("date_time")
                for turn in turns}


def _load_row(split="100K"):
    return json.loads(FIXTURE.read_text())[split][0]


def test_parse_chat_sizes_and_indices():
    assert MOD.parse_chat_sizes("100k,1m,100K") == ["100K", "1M"]
    assert MOD.parse_indices("0-2,5,4-3") == [0, 1, 2, 3, 4, 5]
    assert MOD.parse_indices("all", size=3) == [0, 1, 2]
    with pytest.raises(IndexError):
        MOD.parse_indices("0,999", size=20)
    with pytest.raises(argparse.ArgumentTypeError):
        MOD.parse_chat_sizes("10M")


def test_runtime_freezes_v88_calendar_gpt55(monkeypatch):
    monkeypatch.setenv("NATIVEMEM_V9_PIPELINE", "two_tier")
    monkeypatch.setenv("NATIVEMEM_TRUST_PROXY", "1")
    MOD.configure_native_runtime(_args())
    assert MOD.os.environ["MODEL"] == "gpt-5.5"
    assert MOD.os.environ["NATIVEMEM_PROMPT"] == "v8"
    assert MOD.os.environ["NATIVEMEM_CHUNK_TURNS"] == "6"
    assert MOD.os.environ["NATIVEMEM_V8_TIDY_COMBINED"] == "off"
    assert MOD.os.environ["NATIVEMEM_V8_MAX_ROUNDS"] == "12"
    assert MOD.os.environ["NATIVEMEM_V8_READ_CONTEXT"] == "1"
    assert MOD.os.environ["NATIVEMEM_TRUST_PROXY"] == "0"
    assert "NATIVEMEM_V9_PIPELINE" not in MOD.os.environ


@pytest.mark.parametrize("split", ["100K", "1M"])
def test_hf_fixture_normalizes_sessions_dates_ids_and_questions(split):
    row = _load_row(split)
    conv, stats = MOD.conversation_to_native(row)
    questions = MOD.extract_questions(row)

    assert stats["sessions"] == len(row["chat"])
    assert conv["session_1_date_time"].startswith("2024-")
    assert conv["session_1"][0]["dia_id"] == "D1:1"
    assert conv["session_1"][1]["dia_id"] == "D1:2"
    if split == "100K":
        assert conv["session_2"][0]["dia_id"] == "D2:1"
        assert conv["session_2_date_time"] == "2024-03-18"
        assert [q["question_type"] for q in questions] == [
            "abstention", "knowledge_update"]
        assert questions[1]["rubric_nuggets"] == [
            "The response should state twenty minutes."]
        assert questions[1]["gold_field"] == "answer"
        assert questions[1]["gold"] == "Twenty minutes."
    else:
        assert conv["session_1_date_time"] == "2024-04-01"
        assert questions[0]["question_type"] == "temporal_reasoning"
        assert questions[0]["gold_field"] == "answer"


def test_all_official_question_types_use_explicit_gold_fields():
    grouped = {}
    for question_type, field in MOD.GOLD_FIELD_BY_TYPE.items():
        grouped[question_type] = [{
            "question_text": f"Question for {question_type}?",
            field: f"gold:{question_type}",
            "rubric": ["criterion"],
        }]
    questions = MOD.extract_questions({"probing_questions": grouped})
    assert len(questions) == 10
    assert {
        q["question_type"]: (q["gold_field"], q["gold"])
        for q in questions
    } == {
        question_type: (field, f"gold:{question_type}")
        for question_type, field in MOD.GOLD_FIELD_BY_TYPE.items()
    }


def test_10m_plan_map_schema_is_normalized_but_not_a_cli_split():
    chat = [{
        "plan-2": [{"turns": [[{
            "role": "assistant", "content": "second plan",
            "time_anchor": "May-02-2024",
        }]]}],
        "plan-1": [{"turns": [[{
            "role": "user", "content": "first plan",
            "time_anchor": "May-01-2024",
        }]]}],
    }]
    batches = MOD.parse_beam_chat(chat)
    assert [b[0]["content"] for b in batches] == ["first plan", "second plan"]
    assert "10M" not in MOD.SUPPORTED_CHAT_SIZES


def test_resume_skips_completed_build_and_questions(tmp_path, monkeypatch):
    calls = {"build": 0, "answer": 0}

    def fake_build(native, conv, conv_dir, input_hash, config_hash):
        calls["build"] += 1
        memory = conv_dir / "memory"
        memory.mkdir(parents=True, exist_ok=True)
        (memory / "topic.md").write_text("[D1:1] fixture\n")
        markdown_files, memory_hash = MOD.memory_tree_sha256(memory)
        MOD.atomic_json(memory / MOD.BUILD_MARKER, {
            "schema_version": 1,
            "input_hash": input_hash,
            "config_hash": config_hash,
            "stats": {
                "status": "complete",
                "events": 2,
                "markdown_files": markdown_files,
                "memory_sha256": memory_hash,
                "chunks_total": 1,
                "chunks_with_events": 1,
            },
        })
        return {
            "status": "complete",
            "events": 2,
            "markdown_files": markdown_files,
            "memory_sha256": memory_hash,
            "chunks_total": 1,
            "chunks_with_events": 1,
            "reused_existing_build": False,
        }

    def fake_answer(native, question, memory_dir, turn_index):
        calls["answer"] += 1
        return {
            "answer": "fixture answer",
            "answer_format": "answer_tag",
            "memories": [],
            "steps": 1,
        }

    monkeypatch.setattr(MOD, "build_native_memory", fake_build)
    monkeypatch.setattr(MOD, "answer_question", fake_answer)
    source = {"kind": "fixture", "fingerprint": "fixture"}
    ok, detail = MOD.run_conversation(
        _Native(), _load_row(), "100K", 0, tmp_path, _args(), source)
    assert ok and detail == "complete"
    assert calls == {"build": 1, "answer": 2}

    ok, detail = MOD.run_conversation(
        _Native(), _load_row(), "100K", 0, tmp_path,
        _args(resume=True), source)
    assert ok and detail == "already complete"
    assert calls == {"build": 1, "answer": 2}

    checkpoint = json.loads((tmp_path / "100K" / "conversation_000"
                             / "checkpoint.json").read_text())
    assert checkpoint["status"] == "complete"
    assert all(q["status"] == "complete"
               for q in checkpoint["questions"].values())
    memory = tmp_path / "100K" / "conversation_000" / "memory"
    assert MOD.checkpoint_is_complete(
        checkpoint, MOD.extract_questions(_load_row()), memory)
    (memory / "topic.md").write_text("corrupted\n")
    assert not MOD.checkpoint_is_complete(
        checkpoint, MOD.extract_questions(_load_row()), memory)

    ok, detail = MOD.run_conversation(
        _Native(), _load_row(), "100K", 0, tmp_path,
        _args(resume=True), source)
    assert ok and detail == "complete"
    assert calls == {"build": 2, "answer": 4}
    assert list((memory.parent / "partials").glob(
        "questions-before-memory-rebuild.*.json"))


def test_question_exception_is_recorded_and_not_complete(tmp_path, monkeypatch):
    def fake_build(native, conv, conv_dir, input_hash, config_hash):
        memory = conv_dir / "memory"
        memory.mkdir(parents=True, exist_ok=True)
        (memory / "topic.md").write_text("[D1:1] fixture\n")
        _, memory_hash = MOD.memory_tree_sha256(memory)
        return {
            "events": 2,
            "markdown_files": 1,
            "memory_sha256": memory_hash,
            "reused_existing_build": False,
        }

    def fail_answer(*args, **kwargs):
        raise RuntimeError("proxy unavailable")

    monkeypatch.setattr(MOD, "build_native_memory", fake_build)
    monkeypatch.setattr(MOD, "answer_question", fail_answer)
    ok, detail = MOD.run_conversation(
        _Native(), _load_row(), "100K", 0, tmp_path, _args(),
        {"kind": "fixture", "fingerprint": "fixture"})
    assert not ok and "failed" in detail

    checkpoint = json.loads((tmp_path / "100K" / "conversation_000"
                             / "checkpoint.json").read_text())
    assert checkpoint["status"] == "failed"
    assert checkpoint["errors"]
    assert all(q["status"] == "failed"
               for q in checkpoint["questions"].values())


def test_build_exception_is_recorded_and_not_complete(tmp_path, monkeypatch):
    def fail_build(*args, **kwargs):
        raise RuntimeError("build failed")

    monkeypatch.setattr(MOD, "build_native_memory", fail_build)
    ok, detail = MOD.run_conversation(
        _Native(), _load_row("1M"), "1M", 0, tmp_path, _args(),
        {"kind": "fixture", "fingerprint": "fixture"})
    assert not ok and "build failed" in detail
    checkpoint = json.loads((tmp_path / "1M" / "conversation_000"
                             / "checkpoint.json").read_text())
    assert checkpoint["status"] == "failed"
    assert checkpoint["build"]["status"] == "failed"
    assert not checkpoint["questions"]


def test_exhausted_distill_retries_abort_atomic_build(tmp_path):
    v8 = SimpleNamespace(
        _distill_call=lambda *args, **kwargs: None,
        distill_events=lambda *args, **kwargs: [],
        client=SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: None))),
    )

    def fake_build(conv, memory_dir):
        v8._distill_call([], phase="fixture_distill")
        return 0.1, 1

    native = SimpleNamespace(
        tracker=_Tracker(),
        v8_memory=v8,
        build_memory=fake_build,
    )
    original = v8._distill_call
    with pytest.raises(RuntimeError, match="exhausted retries"):
        MOD.build_native_memory(
            native,
            {"session_1": [{"dia_id": "D1:1"}]},
            tmp_path,
            "input-hash",
            "config-hash",
        )
    assert v8._distill_call is original
    assert not (tmp_path / "memory").exists()
    assert not (tmp_path / ".memory-building").exists()


def test_tool_budget_uses_beam_tools_disabled_finalizer(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_V8_MAX_ROUNDS", "1")
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                function = SimpleNamespace(
                    name="list_memory", arguments='{"path":"."}')
                tool_call = SimpleNamespace(id="tool-1", function=function)
                message = SimpleNamespace(content="", tool_calls=[tool_call])
                finish_reason = "tool_calls"
            else:
                message = SimpleNamespace(
                    content="<answer>twenty minutes</answer>", tool_calls=None)
                finish_reason = "stop"
            return SimpleNamespace(
                id=f"response-{len(calls)}",
                model="gpt-5.5",
                choices=[SimpleNamespace(
                    message=message, finish_reason=finish_reason)],
            )

    native = SimpleNamespace(
        ALIYUN_MODEL="gpt-5.5",
        TOOLS=[],
        _V8_READ_TOOL={},
        _v8_structure_map=lambda path: "topics/",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions())),
        log_usage=lambda *args, **kwargs: None,
        execute_tool=lambda *args, **kwargs: "topics/cache.md",
    )
    result = MOD.answer_question(
        native, "What is the TTL?", tmp_path, {})
    assert result["answer"] == "twenty minutes"
    assert result["response_models"] == ["gpt-5.5"]
    assert len(calls) == 2
    assert "tools" in calls[0] and "tools" not in calls[1]


@pytest.mark.parametrize(
    ("content", "finish_reason", "error"),
    [
        ("<answer>partial", "length", "did not finish normally"),
        ("an unlabeled answer", "stop", "must contain"),
    ],
)
def test_answer_rejects_truncation_and_missing_answer_tag(
        tmp_path, content, finish_reason, error):
    class _Completions:
        @staticmethod
        def create(**kwargs):
            message = SimpleNamespace(content=content, tool_calls=None)
            return SimpleNamespace(
                id="response-1",
                model="gpt-5.5",
                choices=[SimpleNamespace(
                    message=message, finish_reason=finish_reason)],
            )

    native = SimpleNamespace(
        ALIYUN_MODEL="gpt-5.5",
        _V8_READ_TOOL={},
        _v8_structure_map=lambda path: "topics/",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions())),
        log_usage=lambda *args, **kwargs: None,
    )
    with pytest.raises(RuntimeError, match=error):
        MOD.answer_question(native, "Question?", tmp_path, {})


def test_answer_rejects_truncated_tool_response(tmp_path):
    function = SimpleNamespace(
        name="list_memory", arguments='{"path":"."}')
    tool_call = SimpleNamespace(id="tool-1", function=function)

    class _Completions:
        @staticmethod
        def create(**kwargs):
            message = SimpleNamespace(content="", tool_calls=[tool_call])
            return SimpleNamespace(
                id="response-1",
                model="gpt-5.5",
                choices=[SimpleNamespace(
                    message=message, finish_reason="length")],
            )

    native = SimpleNamespace(
        ALIYUN_MODEL="gpt-5.5",
        _V8_READ_TOOL={},
        _v8_structure_map=lambda path: "topics/",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions())),
        log_usage=lambda *args, **kwargs: None,
    )
    with pytest.raises(RuntimeError, match="tool_calls"):
        MOD.answer_question(native, "Question?", tmp_path, {})


def test_memory_tools_are_read_only_and_confined(tmp_path):
    topics = tmp_path / "topics"
    topics.mkdir()
    (topics / "cache.md").write_text("TTL is twenty minutes.\n")
    assert "topics/cache.md" in MOD.execute_memory_tool(
        "list_memory", {"recursive": True}, tmp_path)
    assert "twenty minutes" in MOD.execute_memory_tool(
        "search_memory", {"query": "TWENTY"}, tmp_path)
    assert "TTL" in MOD.execute_memory_tool(
        "read_memory", {"path": "topics/cache.md"}, tmp_path)
    with pytest.raises(ValueError, match="relative"):
        MOD.execute_memory_tool(
            "read_memory", {"path": "/etc/passwd"}, tmp_path)
    with pytest.raises(ValueError, match="leaves"):
        MOD.execute_memory_tool(
            "read_memory", {"path": "../outside.md"}, tmp_path)


def test_missing_model_selected_path_is_returned_as_tool_error(tmp_path):
    output, error = MOD.execute_memory_tool_for_model(
        "read_memory", {"path": "topics/does-not-exist.md"}, tmp_path)
    assert output == error
    assert "FileNotFoundError" in output
    assert "does-not-exist.md" in output


def test_answer_can_recover_after_missing_model_selected_path(
        tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_V8_MAX_ROUNDS", "2")
    observed_messages = []

    class _Completions:
        call_count = 0

        @classmethod
        def create(cls, **kwargs):
            cls.call_count += 1
            observed_messages.append(kwargs["messages"])
            if cls.call_count == 1:
                function = SimpleNamespace(
                    name="read_memory",
                    arguments='{"path":"topics/missing.md"}',
                )
                message = SimpleNamespace(
                    content="",
                    tool_calls=[SimpleNamespace(id="tool-1", function=function)],
                )
                finish_reason = "tool_calls"
            else:
                message = SimpleNamespace(
                    content="<answer>insufficient evidence</answer>",
                    tool_calls=None,
                )
                finish_reason = "stop"
            return SimpleNamespace(
                id=f"response-{cls.call_count}",
                model="gpt-5.5",
                choices=[SimpleNamespace(
                    message=message, finish_reason=finish_reason)],
            )

    native = SimpleNamespace(
        ALIYUN_MODEL="gpt-5.5",
        _V8_READ_TOOL={},
        _v8_structure_map=lambda path: "topics/",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions())),
        log_usage=lambda *args, **kwargs: None,
    )
    result = MOD.answer_question(native, "Question?", tmp_path, {})
    assert result["answer"] == "insufficient evidence"
    assert len(result["tool_input_errors"]) == 1
    tool_message = observed_messages[1][-1]
    assert tool_message["role"] == "tool"
    assert "FileNotFoundError" in tool_message["content"]


def _beam_response(response_id, *, content="", tool_calls=None,
                   finish_reason="stop"):
    return SimpleNamespace(
        id=response_id,
        model="gpt-5.5",
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=content,
                tool_calls=tool_calls,
                refusal="",
            ),
            finish_reason=finish_reason,
        )],
    )


def _read_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "read_original",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_huge_read_memory_is_truncated_and_answer_can_complete(
        tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_V8_MAX_ROUNDS", "2")
    topic = tmp_path / "topics" / "large.md"
    topic.parent.mkdir()
    topic.write_text(
        "".join(f"line {index} " + "evidence " * 30 + "\n"
                for index in range(2000)),
        encoding="utf-8",
    )
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                function = SimpleNamespace(
                    name="read_memory",
                    arguments=json.dumps({"path": "topics/large.md"}),
                )
                tool_call = SimpleNamespace(
                    id="large-read", type="function", function=function)
                return _beam_response(
                    "response-tool",
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                )
            return _beam_response(
                "response-final", content="<answer>bounded answer</answer>")

    native = SimpleNamespace(
        ALIYUN_MODEL="gpt-5.5",
        _V8_READ_TOOL=_read_tool_schema(),
        _v8_structure_map=lambda path: "topics/ (1 file)",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions())),
        log_usage=lambda *args, **kwargs: None,
    )
    result = MOD.answer_question(native, "Question?", tmp_path, {})
    trace = result["tool_trace"][0]
    assert result["answer"] == "bounded answer"
    assert trace["truncated"] is True
    assert trace["raw_tokens"] > MOD.PER_TOOL_CONTENT_TOKEN_LIMIT
    assert trace["delivered_tokens"] <= MOD.PER_TOOL_CONTENT_TOKEN_LIMIT
    assert trace["delivered_text"].endswith(MOD.TOOL_TRUNCATION_MARKER)
    assert result["context_safety"]["delivered_tool_tokens"] == trace[
        "delivered_tokens"
    ]
    assert all(
        observation["bounded_total_tokens"] <= MOD.LOCAL_REQUEST_TOKEN_LIMIT
        for observation in result["context_safety"]["request_observations"]
        if observation["sent"]
    )
    assert len(calls) == 2


def test_tool_delivery_has_per_call_and_total_hard_caps(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_V8_MAX_ROUNDS", "12")
    topic = tmp_path / "topics" / "large.md"
    topic.parent.mkdir()
    topic.write_text(
        "".join(f"line {index} " + "evidence " * 30 + "\n"
                for index in range(2000)),
        encoding="utf-8",
    )
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                tool_calls = []
                for index in range(4):
                    function = SimpleNamespace(
                        name="read_memory",
                        arguments=json.dumps({"path": "topics/large.md"}),
                    )
                    tool_calls.append(SimpleNamespace(
                        id=f"large-read-{index}",
                        type="function",
                        function=function,
                    ))
                return _beam_response(
                    "response-tools",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                )
            return _beam_response(
                "response-final", content="<answer>hard capped</answer>")

    native = SimpleNamespace(
        ALIYUN_MODEL="gpt-5.5",
        _V8_READ_TOOL=_read_tool_schema(),
        _v8_structure_map=lambda path: "topics/ (1 file)",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions())),
        log_usage=lambda *args, **kwargs: None,
    )
    result = MOD.answer_question(native, "Question?", tmp_path, {})
    trace = result["tool_trace"]
    assert trace[0]["applied_token_limit"] == MOD.PER_TOOL_CONTENT_TOKEN_LIMIT
    assert all(
        item["applied_token_limit"]
        == min(
            MOD.PER_TOOL_CONTENT_TOKEN_LIMIT,
            MOD.TOTAL_TOOL_CONTENT_TOKEN_LIMIT
            - (trace[index - 1]["cumulative_delivered_tokens"]
               if index else 0),
        )
        for index, item in enumerate(trace)
    )
    assert all(
        item["delivered_tokens"] <= MOD.PER_TOOL_CONTENT_TOKEN_LIMIT
        for item in trace
    )
    assert trace[-1]["cumulative_delivered_tokens"] == (
        MOD.TOTAL_TOOL_CONTENT_TOKEN_LIMIT
    )
    assert result["context_safety"]["tool_budget_exhausted"] is True
    assert result["context_safety"]["delivered_tool_tokens"] == (
        MOD.TOTAL_TOOL_CONTENT_TOKEN_LIMIT
    )
    assert len(calls) == 2
    assert "tools" not in calls[-1]


def test_request_overflow_uses_compact_finalization(tmp_path, monkeypatch):
    topic = tmp_path / "topics" / "large.md"
    topic.parent.mkdir()
    topic.write_text(
        "".join(f"line {index} " + "evidence " * 10 + "\n"
                for index in range(500)),
        encoding="utf-8",
    )
    tokenizer, _ = MOD._load_context_tokenizer()
    read_tool = _read_tool_schema()
    structure = "topics/ (1 file)"
    question = "Question?"
    prompt = MOD.BEAM_SINGLE_PROMPT.format(
        question=question, structure=structure)
    initial_tokens = MOD._request_local_tokens(
        tokenizer,
        [{"role": "user", "content": prompt}],
        MOD.BEAM_MEMORY_TOOLS + [read_tool],
    )
    monkeypatch.setattr(
        MOD,
        "LOCAL_REQUEST_TOKEN_LIMIT",
        initial_tokens
        + MOD.CONTEXT_OUTPUT_RESERVATION_TOKENS
        + MOD.CONTEXT_SAFETY_MARGIN_TOKENS
        + 10,
    )
    monkeypatch.setattr(MOD, "PER_TOOL_CONTENT_TOKEN_LIMIT", 500)
    monkeypatch.setattr(MOD, "TOTAL_TOOL_CONTENT_TOKEN_LIMIT", 500)
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                function = SimpleNamespace(
                    name="read_memory",
                    arguments=json.dumps({"path": "topics/large.md"}),
                )
                return _beam_response(
                    "response-tool",
                    tool_calls=[SimpleNamespace(
                        id="large-read", type="function", function=function)],
                    finish_reason="tool_calls",
                )
            return _beam_response(
                "response-final", content="<answer>compact answer</answer>")

    native = SimpleNamespace(
        ALIYUN_MODEL="gpt-5.5",
        _V8_READ_TOOL=read_tool,
        _v8_structure_map=lambda path: structure,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions())),
        log_usage=lambda *args, **kwargs: None,
    )
    result = MOD.answer_question(native, question, tmp_path, {})
    context = result["context_safety"]
    assert result["answer"] == "compact answer"
    assert context["context_compaction_used"] is True
    assert context["compact_finalization"]["schema_version"] == (
        "beam-compact-finalization-v1"
    )
    assert [item["phase"] for item in context["request_observations"]] == [
        "retrieve", "finalize_full", "finalize_compact"
    ]
    assert context["request_observations"][1]["sent"] is False
    assert context["request_observations"][2]["sent"] is True
    assert len(calls) == 2
    assert calls[-1]["messages"][0]["content"].startswith(
        "Answer the BEAM question using only"
    )


def test_missing_tiktoken_fails_before_any_client_call(tmp_path, monkeypatch):
    calls = []
    native = SimpleNamespace(
        ALIYUN_MODEL="gpt-5.5",
        _V8_READ_TOOL=_read_tool_schema(),
        _v8_structure_map=lambda path: "(empty memory)",
        client=SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs)))),
        log_usage=lambda *args, **kwargs: None,
    )

    def missing(_name):
        raise MOD.importlib_metadata.PackageNotFoundError("tiktoken")

    monkeypatch.setattr(MOD.importlib_metadata, "version", missing)
    with pytest.raises(RuntimeError, match="requires tiktoken==0.12.0"):
        MOD.answer_question(native, "Question?", tmp_path, {})
    assert calls == []


def test_nonempty_chunk_without_events_aborts_build(tmp_path):
    v8 = SimpleNamespace(
        _distill_call=lambda *args, **kwargs: "{}",
        distill_events=lambda *args, **kwargs: [],
        client=SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: None))),
    )

    def fake_build(conv, memory_dir):
        v8.distill_events(
            [("user", "fact")], "2024-01-01", ["D1:1"])
        return 0.1, 0

    native = SimpleNamespace(
        tracker=_Tracker(),
        v8_memory=v8,
        build_memory=fake_build,
    )
    with pytest.raises(RuntimeError, match="no event"):
        MOD.build_native_memory(
            native,
            {"session_1": [{"dia_id": "D1:1"}]},
            tmp_path,
            "input-hash",
            "config-hash",
        )
    assert not (tmp_path / "memory").exists()


@pytest.mark.parametrize(
    ("response_model", "finish_reason", "refusal", "error"),
    [
        ("not-gpt-5.5", "stop", "", "backend model mismatch"),
        ("gpt-5.5", "length", "", "did not finish normally"),
        ("gpt-5.5", "stop", "policy refusal", "returned a refusal"),
    ],
)
def test_build_rejects_invalid_backend_response(
        tmp_path, response_model, finish_reason, refusal, error):
    class _Completions:
        @staticmethod
        def create(**kwargs):
            message = SimpleNamespace(refusal=refusal)
            return SimpleNamespace(
                model=response_model,
                choices=[SimpleNamespace(
                    finish_reason=finish_reason, message=message)],
            )

    v8 = SimpleNamespace(
        _distill_call=lambda *args, **kwargs: "{}",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions())),
    )

    def fake_distill(turns, obs_date, dia_ids, *args, **kwargs):
        v8.client.chat.completions.create(model="gpt-5.5")
        return [{"summary": "fact", "dia_ids": ["D1:1"]}]

    v8.distill_events = fake_distill

    def fake_build(conv, memory_dir):
        events = v8.distill_events(
            [("user", "fact")], "2024-01-01", ["D1:1"])
        path = Path(memory_dir) / "topics" / "fact.md"
        path.parent.mkdir(parents=True)
        path.write_text("[2024-01-01] fact · [D1:1]\n")
        return 0.1, len(events)

    native = SimpleNamespace(
        ALIYUN_MODEL="gpt-5.5",
        tracker=_Tracker(),
        v8_memory=v8,
        distill_events=fake_distill,
        build_memory=fake_build,
        dia_ids_in=lambda text: ["D1:1"] if "D1:1" in text else [],
    )
    with pytest.raises(RuntimeError, match=error):
        MOD.build_native_memory(
            native,
            {"session_1": [{"dia_id": "D1:1"}]},
            tmp_path,
            "input-hash",
            "config-hash",
        )
    assert not (tmp_path / "memory").exists()


def test_raw_ancestor_does_not_hide_memory_markdown(tmp_path):
    memory = tmp_path / "raw" / "run" / "memory"
    topics = memory / "topics"
    topics.mkdir(parents=True)
    (topics / "fact.md").write_text("[2024-01-01] fact · [D1:1]\n")
    count, _ = MOD.memory_tree_sha256(memory)
    native = SimpleNamespace(
        dia_ids_in=lambda text: ["D1:1"] if "D1:1" in text else [])
    assert count == 1
    assert MOD._stored_dia_ids(native, memory) == {"D1:1"}


def test_main_records_backend_initialization_failure(
        tmp_path, monkeypatch):
    def fail_backend(args):
        raise RuntimeError("backend import failed")

    monkeypatch.setattr(MOD, "load_native", fail_backend)
    output = tmp_path / "out"
    rc = MOD.main([
        "--chat-sizes", "100K",
        "--fixture", str(FIXTURE),
        "--output-dir", str(output),
    ])
    assert rc == 1
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["errors"][-1]["stage"] == "backend_init"


def test_formal_cli_requires_explicit_model_request_authorization(tmp_path, capsys):
    output = tmp_path / "out"
    with pytest.raises(SystemExit):
        MOD.parse_args([
            "--chat-sizes", "100K",
            "--gateway-root", str(tmp_path / "gateway"),
            "--output-dir", str(output),
        ])
    assert "allow-model-requests" in capsys.readouterr().err
    assert not output.exists()


def test_main_rejects_mixed_out_of_range_selection_and_writes_no_qa(
        tmp_path, monkeypatch):
    monkeypatch.setattr(MOD, "load_native", lambda args: _Native())
    output = tmp_path / "out"
    rc = MOD.main([
        "--chat-sizes", "100K",
        "--conversations", "0,999",
        "--fixture", str(FIXTURE),
        "--output-dir", str(output),
    ])
    assert rc == 1
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "100K:dataset" in manifest["failed_conversations"]
    assert not (output / "100K").exists()


def test_main_writes_preflight_failure_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(MOD, "load_native", lambda args: _Native())
    monkeypatch.setattr(
        MOD, "run_conversation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("invalid probing schema")),
    )
    output = tmp_path / "out"
    rc = MOD.main([
        "--chat-sizes", "100K",
        "--conversations", "0",
        "--fixture", str(FIXTURE),
        "--output-dir", str(output),
    ])
    assert rc == 1
    checkpoint = json.loads((
        output / "100K" / "conversation_000" / "checkpoint.json"
    ).read_text())
    assert checkpoint["status"] == "failed"
    assert checkpoint["errors"][-1]["stage"] == "preflight"


def test_resume_keeps_unresolved_historical_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(MOD, "load_native", lambda args: _Native())

    def result_by_split(native, item, chat_size, *args, **kwargs):
        return (False, "fixture failure") if chat_size == "100K" else (
            True, "fixture success")

    monkeypatch.setattr(MOD, "run_conversation", result_by_split)
    output = tmp_path / "out"
    first = MOD.main([
        "--chat-sizes", "100K", "--conversations", "0",
        "--fixture", str(FIXTURE), "--output-dir", str(output),
    ])
    assert first == 1
    second = MOD.main([
        "--chat-sizes", "1M", "--conversations", "0", "--resume",
        "--fixture", str(FIXTURE), "--output-dir", str(output),
    ])
    assert second == 1
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failed_conversations"] == ["100K:0"]
    assert manifest["conversations"]["1M:0"]["status"] == "complete"
