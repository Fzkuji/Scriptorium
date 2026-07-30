import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_v88_gpt55_longmemeval.py"
FIXTURE = ROOT / "tests" / "fixtures" / "longmemeval_s_tiny.json"
SPEC = importlib.util.spec_from_file_location("run_v88_gpt55_longmemeval", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class FakeTracker:
    def __init__(self):
        self.phases = []

    def reset(self, phase):
        self.phases.append(phase)

    def snapshot(self, phase):
        return {"calls": 1, "tokens_in": 10, "tokens_out": 3,
                "llm_time_s": 0.01}


class FakeBackend:
    def __init__(self, answer="Bought a notebook"):
        self.answer = answer
        self.tracker = FakeTracker()
        self.build_calls = 0
        self.answer_calls = 0
        self.v8_memory = SimpleNamespace(
            _V8_DISTILL_PROMPT="日历（日期换算的唯一权威）：\n{calendar}"
        )

    def build_memory(self, conv, memory_dir):
        self.build_calls += 1
        ids = [
            turn["dia_id"]
            for key, session in conv.items()
            if key.startswith("session_") and not key.endswith("_date_time")
            for turn in session
        ]
        assert ids == ["D1:1", "D1:2", "D2:1", "D2:2"]
        path = Path(memory_dir) / "topics" / "items.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[2023-05-19] Bought a notebook. · [D1:1]\n")
        return 0.25, 1

    @staticmethod
    def build_turn_index(conv):
        result = {}
        order = 0
        session_number = 1
        while f"session_{session_number}" in conv:
            current_date = conv[f"session_{session_number}_date_time"]
            for turn in conv[f"session_{session_number}"]:
                result[turn["dia_id"]] = {
                    "speaker": turn["speaker"], "text": turn["text"],
                    "date": current_date, "order": order,
                }
                order += 1
            session_number += 1
        return result

    def _collect_and_answer_v8(self, question, memory_dir, turn_index):
        self.answer_calls += 1
        assert question == (
            "Current Date: 2023/05/22 (Mon) 08:00\n"
            "Question: What did I do first?"
        )
        assert len(turn_index) == 4
        return ([{"text": "Bought a notebook", "date": "2023-05-19"}],
                2, self.answer)

    def collect_and_answer_longmemeval(self, item, memory_dir, turn_index):
        self.answer_calls += 1
        assert MOD.model_question(item) == (
            "Current Date: 2023/05/22 (Mon) 08:00\n"
            "Question: What did I do first?"
        )
        assert len(turn_index) == 4
        return ([{"text": "Bought a notebook", "date": "2023-05-19"}],
                2, self.answer, [{"type": "bash", "accepted": True}])


def run_meta():
    return {
        "method": {"name": "NativeMem", "version": "v8.8+calendar",
                   "calendar": True, "single_model_retrieve_answer": True},
        "models": {"builder": "gpt-5.5", "retriever": "gpt-5.5",
                   "answerer": "gpt-5.5",
                   "base_url": "http://127.0.0.1:8199/v1"},
        "config": MOD.method_config(1),
        "code": {"git_commit": "test", "runner_sha256": "r",
                 "v8_memory_sha256": "a",
                 "adapter_sha256": "b"},
    }


def test_fixture_load_and_conversion_assigns_all_dia_ids():
    data = MOD.load_dataset(FIXTURE, expected_count=2)
    conv = MOD.longmemeval_to_locomo(data[0], 0)
    assert conv["session_1_date_time"] == "2023-05-20"
    assert conv["session_2_date_time"] == "2023-05-21"
    assert [turn["dia_id"] for turn in conv["session_1"]] == ["D1:1", "D1:2"]
    assert [turn["dia_id"] for turn in conv["session_2"]] == ["D2:1", "D2:2"]
    assert conv["session_1"][1]["speaker"] == "assistant"


def test_conversion_rejects_misaligned_session_metadata():
    item = MOD.load_dataset(FIXTURE, expected_count=2)[0]
    item["haystack_dates"] = item["haystack_dates"][:1]
    with pytest.raises(MOD.DataValidationError, match="lengths differ"):
        MOD.longmemeval_to_locomo(item, 0)


def test_conversion_preserves_empty_source_turn_anchor():
    item = MOD.load_dataset(FIXTURE, expected_count=2)[1]
    item["haystack_sessions"][0].insert(1, {"role": "assistant", "content": ""})
    conv = MOD.longmemeval_to_locomo(item, 1)
    assert conv["session_1"][1] == {
        "speaker": "assistant", "text": "", "dia_id": "D1:2"
    }
    assert conv["session_1"][2]["dia_id"] == "D1:3"


def test_start_limit_selection():
    assert MOD.select_indices(500, 4, 3) == [4, 5, 6]
    assert MOD.select_indices(500, 499, 20) == [499]
    with pytest.raises(MOD.DataValidationError):
        MOD.select_indices(500, 500, 1)


def test_run_item_writes_complete_checkpoint_and_never_rebuilds_complete(tmp_path):
    item = MOD.load_dataset(FIXTURE, expected_count=2)[0]
    backend = FakeBackend()
    ok, detail, checkpoint = MOD.run_item(
        0, item, tmp_path, backend, run_meta(), resume=False
    )
    assert ok and detail == "complete"
    assert checkpoint["status"] == "complete"
    assert checkpoint["build"]["events"] == 1
    assert checkpoint["retrieval"]["steps"] == 2
    assert checkpoint["retrieval"]["model_question"].startswith("Current Date:")
    assert checkpoint["retrieval"]["tool_trace"] == [
        {"type": "bash", "accepted": True}
    ]
    assert checkpoint["answer"]["hypothesis"] == "Bought a notebook"
    assert checkpoint["input"]["empty_source_turns"] == 0
    assert checkpoint["models"]["builder"] == "gpt-5.5"
    checkpoint_path = MOD.item_paths(tmp_path, 0, item["question_id"])["checkpoint"]
    assert json.loads(checkpoint_path.read_text())["status"] == "complete"

    second = FakeBackend(answer="must not replace the completed answer")
    ok, detail, resumed = MOD.run_item(
        0, item, tmp_path, second, run_meta(), resume=True
    )
    assert ok and detail == "already complete"
    assert second.build_calls == 0 and second.answer_calls == 0
    assert resumed["answer"]["hypothesis"] == "Bought a notebook"


def test_empty_answer_is_failed_and_resume_reuses_completed_build(tmp_path):
    item = MOD.load_dataset(FIXTURE, expected_count=2)[0]
    blank = FakeBackend(answer="")
    ok, detail, checkpoint = MOD.run_item(
        0, item, tmp_path, blank, run_meta(), resume=False
    )
    assert not ok
    assert "empty hypothesis" in detail
    assert checkpoint["status"] == "failed"
    assert checkpoint["error"]["stage"] == "retrieve_answer"
    assert checkpoint["build"]["status"] == "complete"
    assert checkpoint["retrieval"]["usage"]["calls"] == 1

    good = FakeBackend()
    ok, detail, checkpoint = MOD.run_item(
        0, item, tmp_path, good, run_meta(), resume=True
    )
    assert ok and detail == "complete"
    assert good.build_calls == 0
    assert good.answer_calls == 1
    assert checkpoint["status"] == "complete"


def test_incomplete_state_requires_resume(tmp_path):
    item = MOD.load_dataset(FIXTURE, expected_count=2)[0]
    paths = MOD.item_paths(tmp_path, 0, item["question_id"])
    paths["item_dir"].mkdir(parents=True)
    MOD.atomic_json(paths["checkpoint"], {"status": "building"})
    with pytest.raises(MOD.ExistingStateError, match="--resume"):
        MOD.run_item(0, item, tmp_path, FakeBackend(), run_meta(), resume=False)


def test_refresh_outputs_emits_official_jsonl_without_failed_items(tmp_path):
    data = MOD.load_dataset(FIXTURE, expected_count=2)
    ok, _, _ = MOD.run_item(
        0, data[0], tmp_path, FakeBackend(), run_meta(), resume=False
    )
    assert ok
    manifest = {"models": run_meta()["models"], "config": run_meta()["config"]}
    MOD.refresh_outputs(tmp_path, manifest)
    records = json.loads((tmp_path / "results.json").read_text())
    assert len(records) == 1
    line = json.loads((tmp_path / "hypotheses.jsonl").read_text())
    assert line == {"question_id": "tiny_temporal_0",
                    "hypothesis": "Bought a notebook"}
    assert manifest["checkpoint_counts"] == {"complete": 1}


def test_refresh_outputs_excludes_complete_checkpoint_without_memory(tmp_path):
    item_dir = tmp_path / "items" / "0000_corrupt"
    MOD.atomic_json(item_dir / "checkpoint.json", {
        "status": "complete",
        "dataset_index": 0,
        "question_id": "corrupt",
        "build": {"status": "complete"},
        "retrieval": {"status": "complete"},
        "answer": {"status": "complete", "hypothesis": "x"},
        "paths": {"memory_dir": str(item_dir / "memory")},
    })
    manifest = {}
    MOD.refresh_outputs(tmp_path, manifest)
    assert json.loads((tmp_path / "results.json").read_text()) == []
    assert (tmp_path / "hypotheses.jsonl").read_text() == ""
    assert manifest["completed"] == 0


def test_environment_freezes_v88_calendar_gpt55(monkeypatch):
    monkeypatch.setenv("NATIVEMEM_V9_PIPELINE", "two_tier")
    args = SimpleNamespace(
        request_concurrency=3,
        model="gpt-5.5",
        base_url="http://127.0.0.1:8199/v1",
        api_key="x",
        trust_proxy=False,
    )
    config = MOD.configure_environment(args)
    assert config["NATIVEMEM_PROMPT"] == "v8"
    assert config["NATIVEMEM_V8_SINGLE"] == "1"
    assert config["NATIVEMEM_CHUNK_TURNS"] == "6"
    assert config["NATIVEMEM_V8_CONCURRENCY"] == "3"
    assert MOD.os.environ["MODEL"] == "gpt-5.5"
    assert MOD.os.environ["BUILDER_MODEL"] == "gpt-5.5"
    assert "NATIVEMEM_V9_PIPELINE" not in MOD.os.environ


def test_backend_contract_requires_calendar_prompt():
    backend = FakeBackend()
    MOD.verify_backend_contract(backend)
    backend.v8_memory._V8_DISTILL_PROMPT = "no calendar placeholder"
    with pytest.raises(RuntimeError, match="calendar-enhanced"):
        MOD.verify_backend_contract(backend)


def test_longmemeval_s_guard_rejects_oracle_sized_histories():
    data = MOD.load_dataset(FIXTURE, expected_count=2)
    with pytest.raises(MOD.DataValidationError, match="likely LongMemEval-oracle"):
        MOD.validate_longmemeval_s(data)


def test_formal_cli_requires_explicit_model_request_authorization(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        MOD.flex_evidence,
        "begin_child_invocation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("child proxy must not start without authorization")
        ),
    )
    output = tmp_path / "out"
    with pytest.raises(SystemExit):
        MOD.main([
            "--gateway-root", str(tmp_path / "gateway"),
            "--output-dir", str(output),
        ])
    assert "allow-model-requests" in capsys.readouterr().err
    assert not output.exists()


def test_openrouter_cli_runs_gpt4omini_without_flex_proxy(
    tmp_path, monkeypatch
):
    gateway = tmp_path / "gateway"
    gateway.mkdir()
    (gateway / "openrouter_gpt4o_mini_root.json").write_text(json.dumps({
        "schema": "openrouter-gpt4o-mini-result-root/v1",
        "returned_alias": "openai/gpt-4o-mini",
    }))
    base_url = "http://127.0.0.1:18201/v1"
    (gateway / "gateway_ready.json").write_text(json.dumps({
        "schema": "openrouter-gpt4o-mini-ready/v1",
        "base_url": base_url,
        "result_root": str(gateway),
        "requested_model": "openai/gpt-4o-mini",
    }))
    monkeypatch.setattr(MOD, "EXPECTED_LONGMEMEVAL_SIZE", 2)
    monkeypatch.setattr(MOD, "validate_longmemeval_s", lambda _data: None)
    monkeypatch.setattr(MOD, "load_backend", FakeBackend)
    monkeypatch.setattr(
        MOD.flex_evidence,
        "begin_child_invocation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("OpenRouter mode must not start a Flex child proxy")
        ),
    )

    output = tmp_path / "out"
    assert MOD.main([
        "--provider", "openrouter",
        "--gateway-root", str(gateway),
        "--gateway-base-url", base_url,
        "--data", str(FIXTURE),
        "--output-dir", str(output),
        "--limit", "1",
        "--allow-model-requests",
    ]) == 0
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["models"] == {
        "builder": "openai/gpt-4o-mini",
        "retriever": "openai/gpt-4o-mini",
        "answerer": "openai/gpt-4o-mini",
        "provider": "openrouter_via_cost_gateway",
        "gateway_root": str(gateway.resolve()),
    }
    assert manifest["request_audit"]["mode"] == "exclusive_openrouter_gateway"


@pytest.mark.parametrize("command", [
    "ls -la",
    "grep -rin 'cache' topics | head -20",
    "cat timeline/2023/05/*.md",
    "find . -type f -name '*.md' -print",
    "sed -n '1,120p' topics/items.md",
])
def test_read_only_command_validator_accepts_navigation(command):
    assert MOD.validate_read_only_command(command) == (True, "ok")


@pytest.mark.parametrize("command", [
    "cat /etc/passwd",
    "cat ../secret",
    "rm -rf topics",
    "find . -delete",
    "sed -i 's/a/b/' topics/items.md",
    "cat topics/items.md > copy.md",
    "python3 -c 'print(1)'",
    "cat raw/session.md",
])
def test_read_only_command_validator_rejects_mutation_and_escape(command):
    allowed, _reason = MOD.validate_read_only_command(command)
    assert not allowed


def test_lme_prompt_allows_abstention_and_requires_answer_tag():
    assert "If the history genuinely lacks" in MOD.LME_SINGLE_PROMPT
    assert "<answer>...</answer>" in MOD.LME_SINGLE_PROMPT
    assert "Do not invent" in MOD.LME_SINGLE_PROMPT


def test_longmemeval_optional_review_reconsiders_first_answer(tmp_path):
    calls = []
    responses = iter([
        SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(
                content="<answer>draft</answer>", tool_calls=None,
            ),
        )]),
        SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(
                content="<answer>reviewed</answer>", tool_calls=None,
            ),
        )]),
    ])

    def create(**kwargs):
        calls.append(kwargs)
        return next(responses)

    backend = SimpleNamespace(
        ALIYUN_MODEL="openai/gpt-4o-mini",
        TOOLS=[],
        _V8_READ_TOOL={"type": "function", "function": {"name": "read_original"}},
        memory_structure_map=lambda _path: "topics/example.md [1]",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        log_usage=lambda *_args, **_kwargs: None,
    )
    item = {
        "question": "What happened?",
        "question_date": "2023/05/22 (Mon) 08:00",
    }

    _memories, steps, answer, trace = MOD.collect_and_answer_longmemeval(
        backend,
        item,
        tmp_path,
        {},
        prompt_template=(
            "Library: {structure}\nCurrent Date: {question_date}\n"
            "Question: {question}"
        ),
        review_prompt="Review the draft for missing or conflicting memory.",
    )

    assert steps == 2
    assert answer == "reviewed"
    assert trace == [{"type": "same_model_review"}]
    assert calls[1]["messages"][-2:] == [
        {"role": "assistant", "content": "<answer>draft</answer>"},
        {
            "role": "user",
            "content": "Review the draft for missing or conflicting memory.",
        },
    ]


def test_longmemeval_answer_can_disable_provider_thinking(tmp_path, monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="<answer>yes</answer>", tool_calls=None),
        )])

    backend = SimpleNamespace(
        ALIYUN_MODEL="deepseek-v4-flash",
        TOOLS=[],
        _V8_READ_TOOL={"type": "function", "function": {"name": "read_original"}},
        memory_structure_map=lambda _path: "topics/example.md",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        log_usage=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setenv("NATIVEMEM_THINKING", "disabled")

    MOD.collect_and_answer_longmemeval(
        backend,
        {"question": "What happened?", "question_date": "2023-05-22"},
        tmp_path,
        {},
    )

    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_rate_limit_uses_gateway_cooldown():
    assert MOD._retry_delay(RuntimeError(), 2) == 2
    error = type("RateLimitError", (RuntimeError,), {})()
    assert MOD._retry_delay(error, 2) == 30
