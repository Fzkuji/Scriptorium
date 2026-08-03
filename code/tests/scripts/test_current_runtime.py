import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from src import build as adapter


def test_nativemem_adapter_owns_session_conversion(tmp_path: Path, monkeypatch):
    captured = []
    conversation = {
        "session_1": [{
            "speaker": "Melanie",
            "text": "I painted a lake sunrise last year.",
            "dia_id": "D1:14",
        }],
        "session_1_date_time": "8 May, 2023",
    }
    monkeypatch.setattr(
        adapter.memory,
        "write_sessions",
        lambda *args, **kwargs: captured.extend(kwargs["sessions"]) or [],
    )

    adapter.build_memory(
        conversation,
        tmp_path,
        agent=object(),
        model="test-model",
        config=adapter.BuildConfig(verify_writes=False, final_manage=False),
    )

    assert captured[0]["observation_date"] == "2023-05-08"
    assert captured[0]["turns"] == [
        ("Melanie", "I painted a lake sunrise last year.")
    ]
    assert captured[0]["refs"] == [adapter.benchmark_source_id("D1:14")]


def test_nativemem_adapter_preserves_locomo_image_evidence(
    tmp_path: Path, monkeypatch
):
    captured = []
    conversation = {
        "session_1": [{
            "speaker": "Dave",
            "text": "This is a photo of my shop.",
            "query": "car maintenance shop grand opening",
            "blip_caption": (
                "a photo of a group of people standing in front of a car"
            ),
            "dia_id": "D1:1",
        }],
        "session_1_date_time": "1 May, 2023",
    }
    monkeypatch.setattr(
        adapter.memory,
        "write_sessions",
        lambda *args, **kwargs: captured.extend(kwargs["sessions"]) or [],
    )

    adapter.build_memory(
        conversation,
        tmp_path,
        agent=object(),
        model="test-model",
        config=adapter.BuildConfig(verify_writes=False, final_manage=False),
    )

    expected = (
        "This is a photo of my shop.\n"
        "[Sharing image - query: car maintenance shop grand opening. "
        "The image shows: a photo of a group of people standing in front "
        "of a car]"
    )
    assert captured[0]["turns"] == [("Dave", expected)]
    source_id = adapter.benchmark_source_id("D1:1")
    assert adapter.build_turn_index(conversation)[source_id]["text"] == expected


def test_nativemem_locomo_inventory_selects_stable_sample_id(tmp_path: Path):
    from scripts.nativemem import run_locomo

    dataset = [
        {
            "sample_id": "conv-26",
            "conversation": {"session_1": [{"text": "first"}]},
            "qa": [{"category": 1}],
        },
        {
            "sample_id": "conv-50",
            "conversation": {
                "session_1": [{"text": "one"}, {"text": "two"}],
                "session_1_date_time": "2023-01-01",
                "session_2": [{"text": "three"}],
                "session_2_date_time": "2023-01-02",
            },
            "qa": [{"category": 1}, {"category": 5}],
        },
    ]
    data_path = tmp_path / "locomo.json"
    data_path.write_text(json.dumps(dataset), encoding="utf-8")

    index, sample = run_locomo.load_sample(data_path, "conv-50")
    inventory = run_locomo.sample_inventory(sample)

    assert index == 1
    assert inventory == {
        "sample_id": "conv-50",
        "sessions": 2,
        "messages": 3,
        "questions": 2,
        "primary_questions": 1,
        "adversarial_questions": 1,
    }


def test_nativemem_locomo_usage_summary_reports_phase_and_cost():
    from scripts.nativemem import run_locomo

    records = [
        {
            "phase": "build",
            "calls": 3,
            "prompt_tokens": 1_000_000,
            "completion_tokens": 2_000_000,
            "total_cost_usd": 0.75,
        },
        {
            "phase": "query",
            "calls": 2,
            "prompt_tokens": 500_000,
            "completion_tokens": 250_000,
            "total_cost_usd": 0.25,
        },
    ]

    summary = run_locomo.summarize_usage(
        records,
        input_usd_per_million=1.0,
        output_usd_per_million=2.0,
    )

    assert summary["totals"] == {
        "calls": 5,
        "input_tokens": 1_500_000,
        "output_tokens": 2_250_000,
        "estimated_cost_usd": 6.0,
        "reported_cost_usd": 1.0,
    }
    assert summary["by_phase"]["build"]["calls"] == 3


def test_nativemem_locomo_cli_uses_explicit_credentials(tmp_path: Path):
    from scripts.nativemem import run_locomo

    args = run_locomo.parse_args([
        "--output-dir", str(tmp_path),
        "--base-url", "https://example.test/v1",
        "--api-key", "builder-key",
        "--judge-api-key", "judge-key",
        "--input-usd-per-million", "1.0",
        "--output-usd-per-million", "2.0",
        "--writer-calibration", str(tmp_path / "calibration.json"),
        "--claude-cli", "/opt/claude",
        "--max-turns", "20",
        "--max-budget-usd", "1.5",
        "--build-only",
    ])

    assert args.sample_id == "conv-50"
    assert args.api_key == "builder-key"
    assert args.judge_api_key == "judge-key"
    assert args.claude_cli == "/opt/claude"
    assert args.max_turns == 20
    assert args.max_budget_usd == 1.5
    assert args.build_only is True
    assert args.writer_calibration == tmp_path / "calibration.json"
    assert not hasattr(args, "api_key_env")


def test_nativemem_runtime_exports_the_current_retrieval_api():
    from src import retrieval

    assert callable(retrieval.create_runtime)
    assert callable(retrieval.collect_answer)


def test_longmemeval_cli_uses_one_framework_limit(tmp_path: Path):
    from scripts.nativemem.run_longmemeval import parser

    args = parser().parse_args([
        "--output-dir", str(tmp_path),
        "--start", "0",
        "--limit", "1",
        "--api-key", "test-key",
        "--claude-cli", "/opt/claude",
        "--max-turns", "20",
        "--max-budget-usd", "2.0",
    ])

    assert args.claude_cli == "/opt/claude"
    assert args.max_turns == 20
    assert args.max_budget_usd == 2.0
    assert not hasattr(args, "retrieval_max_tool_calls")


def test_nativemem_management_exports_the_current_writing_api():
    from src import management as memory

    assert hasattr(memory, "__path__")
    assert callable(memory.write_sessions)
    assert callable(memory.verify_session)


def test_locomo_runner_records_build_failure_and_partial_usage(
    tmp_path: Path, monkeypatch
):
    from scripts.nativemem.locomo import runner

    data_path = tmp_path / "locomo.json"
    data_path.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "run"
    args = SimpleNamespace(
        data=data_path,
        output_dir=output_dir,
        sample_id="conv-test",
        core_max_tokens=2_000,
        recent_limit=50,
        max_turns=20,
        max_budget_usd=None,
        session_batch=5,
        writer_calibration=None,
        writer_input_token_cap=None,
        local_reorg_every_sessions=5,
        verify_writes=True,
        verify_every_sessions=5,
        final_manage=False,
        verify_sources=True,
        base_url="https://example.test/v1",
        model="test-model",
        api_key="test-key",
        claude_cli=None,
    )

    class FailingRuntime:
        call_log = [{"phase": "writer", "calls": 2}]

        def build_memory(self, *_args, **_kwargs):
            raise RuntimeError("writer reached its turn limit")

    monkeypatch.setattr(runner, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(runner, "verify_evaluator", lambda: None)
    monkeypatch.setattr(
        runner,
        "load_sample",
        lambda *_args: (0, {"sample_id": "conv-test", "conversation": {}, "qa": []}),
    )
    monkeypatch.setattr(
        runner,
        "sample_inventory",
        lambda _sample: {"sample_id": "conv-test"},
    )
    monkeypatch.setattr(
        runner.retrieval,
        "create_runtime",
        lambda *_args, **_kwargs: FailingRuntime(),
    )

    with pytest.raises(RuntimeError, match="turn limit"):
        runner.main([])

    assert json.loads((output_dir / "call_log.json").read_text()) == [
        {"phase": "writer", "calls": 2}
    ]
    status = json.loads((output_dir / "status.json").read_text())
    assert status["phase"] == "failed"
    assert status["stage"] == "building"
    assert status["error"] == "RuntimeError: writer reached its turn limit"


def test_locomo_build_only_stops_before_querying(tmp_path: Path, monkeypatch):
    from scripts.nativemem.locomo import runner

    data_path = tmp_path / "locomo.json"
    data_path.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "run"
    args = SimpleNamespace(
        data=data_path,
        output_dir=output_dir,
        sample_id="conv-test",
        core_max_tokens=2_000,
        recent_limit=50,
        max_turns=100,
        max_budget_usd=None,
        session_batch=5,
        writer_calibration=None,
        writer_input_token_cap=8_192,
        local_reorg_every_sessions=5,
        verify_writes=True,
        verify_every_sessions=5,
        final_manage=False,
        verify_sources=True,
        base_url="https://example.test/v1",
        model="test-model",
        api_key="test-key",
        claude_cli=None,
        build_only=True,
        workers=1,
        input_usd_per_million=1.0,
        output_usd_per_million=2.0,
        evaluate=False,
    )
    sample = {
        "sample_id": "conv-test",
        "conversation": {},
        "qa": [{"question": "must not be answered", "category": 1}],
    }

    class Runtime:
        call_log = [{
            "phase": "writer",
            "calls": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_cost_usd": 0.0,
        }]

        def build_memory(self, _conversation, memory_dir):
            topic = Path(memory_dir) / "topics" / "fact.md"
            topic.parent.mkdir(parents=True)
            topic.write_text("# Fact\n", encoding="utf-8")
            return memory_dir, 1

        def build_turn_index(self, _conversation):
            return {}

    monkeypatch.setattr(runner, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(runner, "verify_evaluator", lambda: None)
    monkeypatch.setattr(runner, "load_sample", lambda *_args: (0, sample))
    monkeypatch.setattr(
        runner,
        "sample_inventory",
        lambda _sample: {"sample_id": "conv-test", "sessions": 1},
    )
    monkeypatch.setattr(
        runner.retrieval, "create_runtime", lambda *_args, **_kwargs: Runtime()
    )
    monkeypatch.setattr(
        runner,
        "answer_question",
        lambda *_args, **_kwargs: pytest.fail("query phase must not run"),
    )

    assert runner.main([]) == 0
    assert not (output_dir / "sample0_questions.json").exists()
    status = json.loads((output_dir / "status.json").read_text())
    assert status["phase"] == "complete"
    assert status["stage"] == "building"
