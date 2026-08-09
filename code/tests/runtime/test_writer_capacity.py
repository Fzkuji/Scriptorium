import json
from pathlib import Path

import pytest


def test_pack_complete_messages_splits_a_session_only_between_messages():
    from memory.runtime.capacity import pack_complete_messages
    from memory.runtime.tokenization import TokenCounter

    session = {
        "observation_date": "2026-01-01",
        "turns": [("user", "aaa"), ("assistant", "bbbb"), ("user", "cc")],
        "refs": ["r1", "r2", "r3"],
    }
    batches = pack_complete_messages(
        [session],
        max_input_tokens=7,
        render_batch=lambda batch: "".join(
            text for item in batch for _speaker, text in item["turns"]
        ),
        token_counter=TokenCounter.utf8_bytes(requested_model="test"),
    )

    assert batches == [
        [{**session, "turns": session["turns"][:2], "refs": session["refs"][:2]}],
        [{**session, "turns": session["turns"][2:], "refs": session["refs"][2:]}],
    ]


def test_pack_complete_messages_rejects_one_oversized_message():
    from memory.runtime.capacity import MessageTooLargeError, pack_complete_messages
    from memory.runtime.tokenization import TokenCounter

    with pytest.raises(MessageTooLargeError, match="complete message"):
        pack_complete_messages(
            [{
                "observation_date": "2026-01-01",
                "turns": [("user", "123456")],
                "refs": ["r1"],
            }],
            max_input_tokens=5,
            render_batch=lambda batch: batch[0]["turns"][0][1],
            token_counter=TokenCounter.utf8_bytes(requested_model="test"),
        )


def test_explicit_writer_cap_batches_without_a_calibration_file(
    tmp_path: Path, monkeypatch
):
    from memory import build
    from memory.runtime.tokenization import TokenCounter

    captured = []
    monkeypatch.setattr(
        build.TokenCounter,
        "resolve",
        lambda **_kwargs: TokenCounter.utf8_bytes(requested_model="test-model"),
    )
    monkeypatch.setattr(
        build.memory,
        "write_sessions",
        lambda *_args, **kwargs: captured.append(kwargs["sessions"]) or [],
    )
    conversation = {
        "session_1": [{"speaker": "user", "text": "a" * 3_000, "dia_id": "D1:1"}],
        "session_2": [{"speaker": "user", "text": "b" * 3_000, "dia_id": "D2:1"}],
    }

    # Each session must fit alone and no two may fit together. The rendered
    # batch carries the writer prompt as well, so derive the cap from a real
    # render rather than pinning it to a constant the prompt can outgrow.
    one_session = len(build.render_writer_input([{
        "observation_date": "2026-01-01",
        "turns": [("user", "a" * 3_000)],
        "refs": [build.benchmark_source_id("D1:1")],
    }]).encode("utf-8"))

    build.build_memory(
        conversation,
        tmp_path,
        agent=object(),
        model="test-model",
        config=build.BuildConfig(
            session_batch=10,
            writer_input_token_cap=one_session + 1_000,
            verify_writes=False,
            final_manage=False,
        ),
    )

    assert [[session["refs"] for session in batch] for batch in captured] == [
        [[build.benchmark_source_id("D1:1")]],
        [[build.benchmark_source_id("D2:1")]],
    ]


def test_build_checkpoint_resumes_after_last_committed_batch(
    tmp_path: Path, monkeypatch
):
    from memory import build

    checkpoint = tmp_path / "build-checkpoint.json"
    conversation = {
        f"session_{index}": [{
            "speaker": "user", "text": f"fact-{index}", "dia_id": f"D{index}:1"
        }]
        for index in range(1, 4)
    }
    first_calls = []

    def fail_on_second(_memory_dir, *, sessions, **_kwargs):
        first_calls.append(sessions[0]["refs"][0])
        if len(first_calls) == 2:
            raise RuntimeError("interrupted")
        return []

    monkeypatch.setattr(build.memory, "write_sessions", fail_on_second)
    config = build.BuildConfig(
        session_batch=1, verify_writes=False, final_manage=False
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        build.build_memory(
            conversation,
            tmp_path / "memory",
            agent=object(),
            model="test-model",
            config=config,
            checkpoint_path=checkpoint,
        )

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["completed_batches"] == 1
    resumed_calls = []
    monkeypatch.setattr(
        build.memory,
        "write_sessions",
        lambda _memory_dir, *, sessions, **_kwargs: (
            resumed_calls.append(sessions[0]["refs"][0]) or []
        ),
    )
    build.build_memory(
        conversation,
        tmp_path / "memory",
        agent=object(),
        model="test-model",
        config=config,
        checkpoint_path=checkpoint,
        resume=True,
    )

    assert first_calls[0] not in resumed_calls
    assert len(resumed_calls) == 2
    completed = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert completed["status"] == "complete"
    assert completed["completed_batches"] == 3


def test_capacity_selection_uses_full_workload_cost_not_largest_batch():
    from memory.runtime.capacity import select_writer_capacities

    levels = [
        {
            "candidate_tokens": 2048,
            "max_batch_input_tokens_min": 2000,
            "pass_rate": 1.0,
            "source_coverage_min": 1.0,
            "fact_coverage_min": 1.0,
            "estimated_cost_usd_total": 0.030,
            "estimated_cost_usd_mean": 0.030,
            "elapsed_seconds_total": 80.0,
            "elapsed_seconds_mean": 80.0,
        },
        {
            "candidate_tokens": 4096,
            "max_batch_input_tokens_min": 4000,
            "pass_rate": 1.0,
            "source_coverage_min": 1.0,
            "fact_coverage_min": 1.0,
            "estimated_cost_usd_total": 0.020,
            "estimated_cost_usd_mean": 0.020,
            "elapsed_seconds_total": 50.0,
            "elapsed_seconds_mean": 50.0,
        },
        {
            "candidate_tokens": 8192,
            "max_batch_input_tokens_min": 6000,
            "pass_rate": 1.0,
            "source_coverage_min": 1.0,
            "fact_coverage_min": 1.0,
            "estimated_cost_usd_total": 0.025,
            "estimated_cost_usd_mean": 0.025,
            "elapsed_seconds_total": 40.0,
            "elapsed_seconds_mean": 40.0,
        },
        {
            "candidate_tokens": 16384,
            "max_batch_input_tokens_min": 12_000,
            "pass_rate": 0.5,
            "source_coverage_min": 0.9,
            "fact_coverage_min": 0.8,
            "estimated_cost_usd_total": 0.018,
            "estimated_cost_usd_mean": 0.018,
            "elapsed_seconds_total": 35.0,
            "elapsed_seconds_mean": 35.0,
        },
    ]

    assert select_writer_capacities(levels) == {
        "recommended_input_tokens": 4000,
        "recommended_candidate_tokens": 4096,
        "max_tested_passing_input_tokens": 6000,
        "max_passing_candidate_tokens": 8192,
    }


def test_writer_capacity_rejects_another_model_or_prompt(tmp_path: Path):
    from memory.runtime.capacity import WriterCapacity

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "schema": "nativemem-writer-capacity-v3",
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


@pytest.mark.parametrize(
    "schema", ["nativemem-writer-capacity-v1", "nativemem-writer-capacity-v2"]
)
def test_writer_capacity_rejects_old_calibration(tmp_path: Path, schema: str):
    from memory.runtime.capacity import WriterCapacity

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "schema": schema,
        "model": "model-a",
        "writer_protocol_sha256": "a" * 64,
        "safe_input_tokens": 4096,
        "tokenizer": {},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        WriterCapacity.load(path, model="model-a", writer_protocol_sha256="a" * 64)


def test_capacity_level_statistics_include_repeated_trial_dispersion():
    from scripts.model_capacity.calibrate_writer import summarize_levels

    records = [
        {
            "candidate_tokens": 4096,
            "probe_id": "a",
            "passed": True,
            "source_coverage": 1.0,
            "fact_coverage": 1.0,
            "workload_session_count": 20,
            "workload_message_count": 60,
            "workload_fact_count": 40,
            "workload_input_tokens": 12_000,
            "batch_count": 2,
            "batch_input_tokens": [4000, 3900],
            "batched_input_tokens": 13_000,
            "calls": 2,
            "provider_first_prompt_tokens": 4100,
            "provider_max_prompt_tokens": 6200,
            "prompt_tokens": 10300,
            "completion_tokens": 500,
            "elapsed_seconds": 10.0,
            "estimated_cost_usd": 0.01,
            "round_limit_reached": False,
        },
        {
            "candidate_tokens": 4096,
            "probe_id": "b",
            "passed": False,
            "source_coverage": 0.8,
            "fact_coverage": 0.7,
            "workload_session_count": 20,
            "workload_message_count": 60,
            "workload_fact_count": 40,
            "workload_input_tokens": 12_000,
            "batch_count": 3,
            "batch_input_tokens": [4050, 3950, 3900],
            "batched_input_tokens": 14_000,
            "calls": 1,
            "provider_first_prompt_tokens": 4050,
            "provider_max_prompt_tokens": 4050,
            "prompt_tokens": 4050,
            "completion_tokens": 400,
            "elapsed_seconds": 14.0,
            "estimated_cost_usd": 0.02,
            "round_limit_reached": True,
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

    level = summarize_levels(records)[0]

    assert level["candidate_tokens"] == 4096
    assert level["pass_rate"] == 0.5
    assert level["source_coverage_mean"] == 0.9
    assert level["fact_coverage_mean"] == 0.85
    assert level["workload_session_count"] == 20
    assert level["workload_message_count"] == 60
    assert level["workload_fact_count"] == 40
    assert level["workload_input_tokens"] == 12_000
    assert level["batch_count_mean"] == 2.5
    assert level["batch_count_std"] == 0.5
    assert level["max_batch_input_tokens_min"] == 4000
    assert level["max_batch_input_tokens_max"] == 4050
    assert level["completion_rate"] == 0.5
    assert level["batched_input_tokens_total"] == 27_000
    assert level["calls_total"] == 3
    assert level["elapsed_seconds_total"] == 24.0
    assert level["estimated_cost_usd_total"] == 0.03


def test_probe_workload_has_fixed_sessions_and_complete_messages():
    from scripts.model_capacity.calibrate_writer import make_probe_workload

    sessions, expected_refs, expected_facts = make_probe_workload(
        probe_id="facts-a",
        session_count=2,
        messages_per_session=6,
        facts_per_session=2,
    )

    assert len(sessions) == 2
    assert [len(session["turns"]) for session in sessions] == [6, 6]
    assert [speaker for speaker, _text in sessions[0]["turns"]] == [
        "user", "assistant", "user", "assistant", "user", "assistant",
    ]
    assert len(expected_refs) == len(expected_facts) == 4
    assert sessions[0]["refs"] == [
        "calibration/facts-a-s001/msg-001",
        "calibration/facts-a-s001/msg-002",
        "calibration/facts-a-s001/msg-003",
        "calibration/facts-a-s001/msg-004",
        "calibration/facts-a-s001/msg-005",
        "calibration/facts-a-s001/msg-006",
    ]
    assert expected_refs == {
        "calibration/facts-a-s001/msg-003",
        "calibration/facts-a-s001/msg-005",
        "calibration/facts-a-s002/msg-003",
        "calibration/facts-a-s002/msg-005",
    }
    assert all("S001" not in fact and "M003" not in fact for fact in expected_facts)


def test_probe_workload_rejects_more_facts_than_messages():
    from scripts.model_capacity.calibrate_writer import make_probe_workload

    with pytest.raises(ValueError, match="facts_per_session"):
        make_probe_workload(
            probe_id="facts-a",
            session_count=1,
            messages_per_session=2,
            facts_per_session=3,
        )


def test_probe_workload_defaults_to_one_fact_per_session():
    from scripts.model_capacity.calibrate_writer import make_probe_workload

    _sessions, expected_refs, expected_facts = make_probe_workload(
        probe_id="facts-a",
        session_count=2,
        messages_per_session=6,
    )

    assert len(expected_refs) == len(expected_facts) == 2


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
        expected_facts={"FACT-ONE", "FACT-TWO"},
    )
    topic.write_text(
        "FACT-ONE calibration/probe/msg-1\nFACT-TWO calibration/probe/msg-2\n",
        encoding="utf-8",
    )
    passed = evaluate_probe(
        tmp_path,
        audit=audit,
        expected_refs={"calibration/probe/msg-1", "calibration/probe/msg-2"},
        expected_facts={"FACT-ONE", "FACT-TWO"},
    )

    assert failed.items() >= {
        "passed": False,
        "source_coverage": 0.5,
        "fact_coverage": 0.0,
        "written_blocks": 1,
        "audit_error_count": 0,
        "round_limit_reached": False,
    }.items()
    assert passed.items() >= {
        "passed": True,
        "source_coverage": 1.0,
        "fact_coverage": 1.0,
        "written_blocks": 1,
        "audit_error_count": 0,
        "round_limit_reached": False,
    }.items()


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
        expected_facts={"Fact"},
    )

    assert result.items() >= {
        "passed": True,
        "source_coverage": 1.0,
        "fact_coverage": 1.0,
        "written_blocks": 1,
        "audit_error_count": 1,
        "round_limit_reached": False,
    }.items()


def test_probe_evaluation_reports_an_unfinished_agent_run_separately(tmp_path: Path):
    from scripts.model_capacity.calibrate_writer import evaluate_probe

    topic = tmp_path / "topics" / "facts.md"
    topic.parent.mkdir(parents=True)
    topic.write_text("FACT calibration/probe/msg-1\n", encoding="utf-8")

    result = evaluate_probe(
        tmp_path,
        audit=[
            {"tool": "shell", "status": "ok", "count": 1},
            {"tool": "agent", "status": "stopped", "reason": "round_limit"},
        ],
        expected_refs={"calibration/probe/msg-1"},
        expected_facts={"FACT"},
    )

    assert result["passed"] is True
    assert result["round_limit_reached"] is True


def test_build_caps_calibrated_input_at_explicit_limit(tmp_path: Path, monkeypatch):
    from memory import build
    from memory.management.api import render_writer_input, writer_protocol_sha256
    from memory.runtime.tokenization import TokenCounter

    conversation = {
        "session_1": [
            {"speaker": "user", "text": "first", "dia_id": "D1:1"},
            {"speaker": "assistant", "text": "second", "dia_id": "D1:2"},
        ],
        "session_1_date_time": "2026-01-01",
    }
    sessions = []
    for index in (1,):
        turns, refs = build.session_content(conversation[f"session_{index}"], index)
        sessions.append({
            "observation_date": f"2026-01-0{index}",
            "turns": turns,
            "refs": [build.benchmark_source_id(ref) for ref in refs],
        })
    counter = TokenCounter.utf8_bytes(
        requested_model="test-model", fallback_reason="test"
    )
    one_message_tokens = max(
        counter.count(render_writer_input([{
            **sessions[0],
            "turns": [turn],
            "refs": [ref],
        }]))
        for turn, ref in zip(sessions[0]["turns"], sessions[0]["refs"])
    )
    full_batch_tokens = counter.count(render_writer_input(sessions))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "schema": "nativemem-writer-capacity-v3",
        "model": "test-model",
        "writer_protocol_sha256": writer_protocol_sha256(),
        "safe_input_tokens": full_batch_tokens,
        "tokenizer": counter.identity,
    }), encoding="utf-8")
    batch_message_counts = []
    monkeypatch.setattr(
        build.memory,
        "write_sessions",
        lambda *args, **kwargs: batch_message_counts.append(sum(
            len(session["turns"]) for session in kwargs["sessions"]
        )) or [],
    )

    build.build_memory(
        conversation,
        tmp_path / "memory",
        agent=object(),
        model="test-model",
        config=build.BuildConfig(
            session_batch=99,
            calibration_path=str(calibration),
            writer_input_token_cap=one_message_tokens,
            verify_writes=False,
            final_manage=False,
        ),
    )

    assert batch_message_counts == [1, 1]
