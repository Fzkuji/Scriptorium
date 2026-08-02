import json
from pathlib import Path

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
        client=object(),
        model="test-model",
        config=adapter.BuildConfig(verify_writes=False, final_manage=False),
    )

    assert captured[0]["observation_date"] == "2023-05-08"
    assert captured[0]["turns"] == [
        ("Melanie", "I painted a lake sunrise last year.")
    ]
    assert captured[0]["refs"] == [adapter.benchmark_source_id("D1:14")]


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
        {"phase": "build", "prompt_tokens": 1_000_000, "completion_tokens": 2_000_000},
        {"phase": "query", "prompt_tokens": 500_000, "completion_tokens": 250_000},
    ]

    summary = run_locomo.summarize_usage(
        records,
        input_usd_per_million=1.0,
        output_usd_per_million=2.0,
    )

    assert summary["totals"] == {
        "calls": 2,
        "input_tokens": 1_500_000,
        "output_tokens": 2_250_000,
        "estimated_cost_usd": 6.0,
    }
    assert summary["by_phase"]["build"]["calls"] == 1


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
        "--thinking", "disabled",
        "--retry-log",
        "--timeout-seconds", "600",
        "--max-retries", "0",
    ])

    assert args.sample_id == "conv-50"
    assert args.api_key == "builder-key"
    assert args.judge_api_key == "judge-key"
    assert args.thinking == "disabled"
    assert args.retry_log is True
    assert args.timeout_seconds == 600
    assert args.max_retries == 0
    assert args.writer_calibration == tmp_path / "calibration.json"
    assert not hasattr(args, "api_key_env")


def test_nativemem_runtime_exports_the_current_retrieval_api():
    from src import retrieval

    assert callable(retrieval.create_runtime)
    assert callable(retrieval.collect_answer)


def test_nativemem_management_exports_the_current_writing_api():
    from src import management as memory

    assert hasattr(memory, "__path__")
    assert callable(memory.write_sessions)
    assert callable(memory.verify_session)
