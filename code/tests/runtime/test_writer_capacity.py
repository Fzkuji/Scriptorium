import json
from pathlib import Path

import pytest


def test_pack_complete_sessions_never_splits_a_session():
    from src.runtime.capacity import pack_complete_sessions
    from src.runtime.tokenization import TokenCounter

    sessions = [{"text": "aaa"}, {"text": "bbbb"}, {"text": "ccccc"}]
    batches = pack_complete_sessions(
        sessions,
        max_input_tokens=7,
        render_batch=lambda batch: "".join(item["text"] for item in batch),
        token_counter=TokenCounter.utf8_bytes(requested_model="test"),
    )

    assert batches == [sessions[:2], sessions[2:]]


def test_pack_complete_sessions_rejects_one_oversized_session():
    from src.runtime.capacity import SessionTooLargeError, pack_complete_sessions
    from src.runtime.tokenization import TokenCounter

    with pytest.raises(SessionTooLargeError, match="complete session"):
        pack_complete_sessions(
            [{"text": "123456"}],
            max_input_tokens=5,
            render_batch=lambda batch: batch[0]["text"],
            token_counter=TokenCounter.utf8_bytes(requested_model="test"),
        )


def test_writer_capacity_rejects_another_model_or_prompt(tmp_path: Path):
    from src.runtime.capacity import WriterCapacity

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "schema": "nativemem-writer-capacity-v1",
        "model": "model-a",
        "writer_protocol_sha256": "a" * 64,
        "safe_input_tokens": 4096,
        "tokenizer": {
            "implementation": "builtin_utf8_bytes",
            "implementation_version": "1",
            "encoding_name": "utf8_bytes_v1",
            "requested_model": "model-a",
            "resolution": "explicit_byte_fallback",
            "fallback_reason": "test",
            "provider_exact": False,
            "counting_note": "test",
        },
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="model"):
        WriterCapacity.load(path, model="model-b", writer_protocol_sha256="a" * 64)
    with pytest.raises(ValueError, match="protocol"):
        WriterCapacity.load(path, model="model-a", writer_protocol_sha256="b" * 64)


def test_select_safe_capacity_stops_before_first_failed_level():
    from src.runtime.capacity import select_safe_capacity

    probes = [
        {"candidate_tokens": 2048, "probe_id": "a", "passed": True},
        {"candidate_tokens": 2048, "probe_id": "b", "passed": True},
        {"candidate_tokens": 4096, "probe_id": "a", "passed": True},
        {"candidate_tokens": 4096, "probe_id": "b", "passed": True},
        {"candidate_tokens": 8192, "probe_id": "a", "passed": True},
        {"candidate_tokens": 8192, "probe_id": "b", "passed": False},
        {"candidate_tokens": 16384, "probe_id": "a", "passed": True},
        {"candidate_tokens": 16384, "probe_id": "b", "passed": True},
    ]

    assert select_safe_capacity(probes, probe_ids={"a", "b"}) == 4096


def test_select_safe_capacity_ignores_candidates_smaller_than_one_writer_input():
    from src.runtime.capacity import select_safe_capacity

    probes = [
        {
            "candidate_tokens": 1024,
            "probe_id": "a",
            "passed": False,
            "skipped": True,
        },
        {
            "candidate_tokens": 1024,
            "probe_id": "b",
            "passed": False,
            "skipped": True,
        },
        {"candidate_tokens": 2048, "probe_id": "a", "passed": True},
        {"candidate_tokens": 2048, "probe_id": "b", "passed": True},
    ]

    assert select_safe_capacity(probes, probe_ids={"a", "b"}) == 2048


def test_capacity_level_statistics_include_repeated_trial_dispersion():
    from scripts.model_capacity.calibrate_writer import summarize_levels

    records = [
        {
            "candidate_tokens": 4096,
            "probe_id": "a",
            "passed": True,
            "source_coverage": 1.0,
            "session_count": 3,
            "rendered_input_tokens": 4000,
            "calls": 2,
            "provider_first_prompt_tokens": 4100,
            "provider_max_prompt_tokens": 6200,
            "prompt_tokens": 10300,
            "completion_tokens": 500,
            "elapsed_seconds": 10.0,
            "estimated_cost_usd": 0.01,
        },
        {
            "candidate_tokens": 4096,
            "probe_id": "b",
            "passed": False,
            "source_coverage": 0.8,
            "session_count": 3,
            "rendered_input_tokens": 3980,
            "calls": 1,
            "provider_first_prompt_tokens": 4050,
            "provider_max_prompt_tokens": 4050,
            "prompt_tokens": 4050,
            "completion_tokens": 400,
            "elapsed_seconds": 14.0,
            "estimated_cost_usd": 0.02,
        },
        {
            "candidate_tokens": 8192,
            "probe_id": "a",
            "passed": False,
            "skipped": True,
            "calls": 0,
            "provider_first_prompt_tokens": None,
            "provider_max_prompt_tokens": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_seconds": 0.1,
            "estimated_cost_usd": 0.0,
        },
    ]

    assert summarize_levels(records) == [
        {
            "candidate_tokens": 4096,
            "trials": 2,
            "passed_trials": 1,
            "pass_rate": 0.5,
            "skipped_trials": 0,
            "error_trials": 0,
            "tool_errors_total": 0,
            "round_limit_trials": 0,
            "source_coverage_mean": 0.9,
            "source_coverage_min": 0.8,
            "source_coverage_std": 0.1,
            "session_count_mean": 3.0,
            "rendered_input_tokens_mean": 3990.0,
            "provider_first_prompt_tokens_mean": 4075.0,
            "provider_first_prompt_tokens_min": 4050,
            "provider_first_prompt_tokens_max": 4100,
            "provider_max_prompt_tokens_mean": 5125.0,
            "provider_max_prompt_tokens_max": 6200,
            "calls_total": 3,
            "calls_mean": 1.5,
            "prompt_tokens_total": 14350,
            "completion_tokens_total": 900,
            "elapsed_seconds_total": 24.0,
            "elapsed_seconds_mean": 12.0,
            "elapsed_seconds_std": 2.0,
            "estimated_cost_usd_total": 0.03,
            "estimated_cost_usd_mean": 0.015,
        },
        {
            "candidate_tokens": 8192,
            "trials": 0,
            "passed_trials": 0,
            "pass_rate": None,
            "skipped_trials": 1,
            "error_trials": 0,
            "tool_errors_total": 0,
            "round_limit_trials": 0,
            "source_coverage_mean": None,
            "source_coverage_min": None,
            "source_coverage_std": None,
            "session_count_mean": None,
            "rendered_input_tokens_mean": None,
            "provider_first_prompt_tokens_mean": None,
            "provider_first_prompt_tokens_min": None,
            "provider_first_prompt_tokens_max": None,
            "provider_max_prompt_tokens_mean": None,
            "provider_max_prompt_tokens_max": None,
            "calls_total": 0,
            "calls_mean": None,
            "prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "elapsed_seconds_total": 0.1,
            "elapsed_seconds_mean": None,
            "elapsed_seconds_std": None,
            "estimated_cost_usd_total": 0.0,
            "estimated_cost_usd_mean": None,
        },
    ]


def test_probe_sessions_fill_the_candidate_without_splitting():
    from scripts.model_capacity.calibrate_writer import make_probe_sessions
    from src.management.api import render_writer_input
    from src.runtime.tokenization import TokenCounter

    counter = TokenCounter.utf8_bytes(requested_model="test")
    sessions = make_probe_sessions(
        probe_id="facts-a",
        candidate_tokens=20_000,
        token_counter=counter,
    )
    used = counter.count(render_writer_input(sessions))

    assert sessions
    assert 14_000 <= used <= 20_000
    assert len({session["refs"][0] for session in sessions}) == len(sessions)


def test_probe_evaluation_requires_every_source_reference(tmp_path: Path):
    from scripts.model_capacity.calibrate_writer import evaluate_probe

    topic = tmp_path / "topics" / "facts.md"
    topic.parent.mkdir(parents=True)
    topic.write_text("Fact one calibration/probe/msg-1\n", encoding="utf-8")
    audit = [{"tool": "shell", "status": "ok", "count": 1}]

    failed = evaluate_probe(
        tmp_path,
        audit=audit,
        expected_refs={"calibration/probe/msg-1", "calibration/probe/msg-2"},
    )
    topic.write_text(
        "Fact one calibration/probe/msg-1\nFact two calibration/probe/msg-2\n",
        encoding="utf-8",
    )
    passed = evaluate_probe(
        tmp_path,
        audit=audit,
        expected_refs={"calibration/probe/msg-1", "calibration/probe/msg-2"},
    )

    assert failed == {
        "passed": False,
        "source_coverage": 0.5,
        "written_blocks": 1,
        "audit_error_count": 0,
        "round_limit_reached": False,
    }
    assert passed == {
        "passed": True,
        "source_coverage": 1.0,
        "written_blocks": 1,
        "audit_error_count": 0,
        "round_limit_reached": False,
    }


def test_probe_evaluation_accepts_a_corrected_error_when_final_memory_is_complete(
    tmp_path: Path,
):
    from scripts.model_capacity.calibrate_writer import evaluate_probe

    topic = tmp_path / "topics" / "facts.md"
    topic.parent.mkdir(parents=True)
    topic.write_text("Fact calibration/probe/msg-1\n", encoding="utf-8")

    result = evaluate_probe(
        tmp_path,
        audit=[
            {"tool": "shell", "status": "error", "count": 0},
            {"tool": "shell", "status": "ok", "count": 1},
        ],
        expected_refs={"calibration/probe/msg-1"},
    )

    assert result == {
        "passed": True,
        "source_coverage": 1.0,
        "written_blocks": 1,
        "audit_error_count": 1,
        "round_limit_reached": False,
    }


def test_build_uses_calibration_to_pack_complete_sessions(tmp_path: Path, monkeypatch):
    from src import build
    from src.management.api import render_writer_input, writer_protocol_sha256
    from src.runtime.tokenization import TokenCounter

    conversation = {
        "session_1": [{"speaker": "user", "text": "first", "dia_id": "D1:1"}],
        "session_1_date_time": "2026-01-01",
        "session_2": [{"speaker": "user", "text": "second", "dia_id": "D2:1"}],
        "session_2_date_time": "2026-01-02",
    }
    sessions = []
    for index in (1, 2):
        turns, refs = build.session_content(conversation[f"session_{index}"], index)
        sessions.append({
            "observation_date": f"2026-01-0{index}",
            "turns": turns,
            "refs": [build.benchmark_source_id(ref) for ref in refs],
        })
    counter = TokenCounter.utf8_bytes(
        requested_model="test-model", fallback_reason="test"
    )
    one_session_tokens = max(
        counter.count(render_writer_input([session])) for session in sessions
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "schema": "nativemem-writer-capacity-v1",
        "model": "test-model",
        "writer_protocol_sha256": writer_protocol_sha256(),
        "safe_input_tokens": one_session_tokens,
        "tokenizer": counter.identity,
    }), encoding="utf-8")
    batch_sizes = []
    monkeypatch.setattr(
        build.memory,
        "write_sessions",
        lambda *args, **kwargs: batch_sizes.append(len(kwargs["sessions"])) or [],
    )

    build.build_memory(
        conversation,
        tmp_path / "memory",
        client=object(),
        model="test-model",
        config=build.BuildConfig(
            session_batch=99,
            calibration_path=str(calibration),
            verify_writes=False,
            final_manage=False,
        ),
    )

    assert batch_sizes == [1, 1]
