import json
import sys

import pytest

import src.adapters.run_nativemem as runner
import src.v8_memory as v8_memory
import src.v10_memory as v10_memory


def _conversation(turns=7, sessions=1):
    conv = {}
    for session_idx in range(1, sessions + 1):
        conv[f"session_{session_idx}"] = [
            {
                "speaker": "User",
                "dia_id": f"D{session_idx}:{turn_idx}",
                "text": f"fact {session_idx}-{turn_idx}",
            }
            for turn_idx in range(1, turns + 1)
        ]
        conv[f"session_{session_idx}_date_time"] = "2023-05-07"
    return conv


def test_v10_defaults_freeze_selected_v88_policy(monkeypatch):
    for name in (
        "NATIVEMEM_V10_WRITE_TURNS",
        "NATIVEMEM_V10_CONTEXT_MODE",
        "NATIVEMEM_V10_CONTEXT_ITEMS",
        "NATIVEMEM_V10_SUMMARY_MAX_WORDS",
        "NATIVEMEM_V10_TIDY_EVERY_SESSIONS",
        "NATIVEMEM_V10_SESSION_TIDY_PASSES",
        "NATIVEMEM_V10_FINAL_TIDY_PASSES",
    ):
        monkeypatch.delenv(name, raising=False)

    config = v10_memory.V10BuildConfig.from_env()
    assert config.to_dict() == {
        "write_turns": 6,
        "context_mode": "events",
        "context_items": 20,
        "summary_max_words": 180,
        "tidy_every_sessions": 1,
        "session_tidy_passes": 1,
        "final_tidy_passes": 1,
    }


@pytest.mark.parametrize("mode", ["events", "raw", "summary", "none"])
def test_v10_accepts_all_context_modes(monkeypatch, mode):
    monkeypatch.setenv("NATIVEMEM_V10_CONTEXT_MODE", mode)
    assert v10_memory.V10BuildConfig.from_env().context_mode == mode


def test_v10_rejects_unknown_context_mode(monkeypatch):
    monkeypatch.setenv("NATIVEMEM_V10_CONTEXT_MODE", "unknown")
    with pytest.raises(ValueError, match="context_mode"):
        v10_memory.V10BuildConfig.from_env()


def test_v10_fail_fast_propagates_first_api_error(monkeypatch):
    monkeypatch.setenv("NATIVEMEM_FAIL_FAST", "1")
    calls = []

    def fail_request(**_kwargs):
        calls.append(1)
        raise RuntimeError("quota exhausted")

    monkeypatch.setattr(
        v8_memory.client.chat.completions, "create", fail_request
    )

    with pytest.raises(RuntimeError, match="quota exhausted"):
        v8_memory._distill_call([], max_retry=6)

    assert calls == [1]


def test_v10_reuse_requires_matching_source_build_record(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    source = tmp_path / "build.json"
    source.write_text(
        json.dumps(
            [
                {
                    "question_id": "_build_stats",
                    "method": "NativeMem-v10",
                    "sample": 0,
                    "builder_model": "writer-model",
                    "memory_dir": str(memory_dir),
                    "v10_config": {"write_turns": 20},
                }
            ]
        )
    )

    record = runner.load_v10_source_build_record(source, memory_dir, 0)

    assert record["builder_model"] == "writer-model"
    with pytest.raises(ValueError, match="sample"):
        runner.load_v10_source_build_record(source, memory_dir, 1)
    with pytest.raises(ValueError, match="memory_dir"):
        runner.load_v10_source_build_record(source, tmp_path / "other", 0)


def test_v10_accepts_full_session_write_window(monkeypatch):
    monkeypatch.setenv("NATIVEMEM_V10_WRITE_TURNS", "session")
    config = v10_memory.V10BuildConfig.from_env()
    assert config.write_turns == "session"
    assert config.chunk_size(37) == 37


def test_v10_rejects_only_active_v9_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v10")
    monkeypatch.setenv("NATIVEMEM_V9_PIPELINE", "off")
    monkeypatch.setenv("NATIVEMEM_V10_TIDY_EVERY_SESSIONS", "0")
    monkeypatch.setenv("NATIVEMEM_V10_FINAL_TIDY_PASSES", "0")
    monkeypatch.setattr(v8_memory, "distill_events", lambda *_args, **_kwargs: [])

    runner.build_memory(_conversation(turns=1), str(tmp_path / "allowed"))

    monkeypatch.setenv("NATIVEMEM_V9_PIPELINE", "two_tier")
    with pytest.raises(ValueError, match="v10 cannot be combined"):
        runner.build_memory(_conversation(turns=1), str(tmp_path / "rejected"))


def test_v10_default_build_uses_six_turns_and_event_context(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v10")
    monkeypatch.setenv("NATIVEMEM_V10_FINAL_TIDY_PASSES", "0")
    seen = []

    def fake_distill(turns, obs, dia_ids, known_topics=None, recent=None, **_):
        seen.append((len(turns), list(recent or [])))
        index = len(seen)
        return [
            {
                "when": obs,
                "summary": f"event {index}",
                "dia_ids": list(dia_ids),
                "topic": "topic",
            }
        ]

    monkeypatch.setattr(v8_memory, "distill_events", fake_distill)
    monkeypatch.setattr(v8_memory, "dedup_topic_files", lambda *_: 0)
    monkeypatch.setattr(runner, "_v8_tidy", lambda *_args, **_kwargs: None)

    runner.build_memory(_conversation(turns=7), str(tmp_path / "memory"))

    assert seen == [(6, []), (1, ["event 1"])]


def test_v10_context_item_budget_is_independent_of_write_turns(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v10")
    monkeypatch.setenv("NATIVEMEM_V10_WRITE_TURNS", "1")
    monkeypatch.setenv("NATIVEMEM_V10_CONTEXT_ITEMS", "2")
    monkeypatch.setenv("NATIVEMEM_V10_TIDY_EVERY_SESSIONS", "0")
    monkeypatch.setenv("NATIVEMEM_V10_FINAL_TIDY_PASSES", "0")
    seen_recent = []

    def fake_distill(turns, obs, dia_ids, known_topics=None, recent=None, **_):
        seen_recent.append(list(recent or []))
        index = len(seen_recent)
        return [
            {
                "when": obs,
                "summary": f"event {index}",
                "dia_ids": list(dia_ids),
                "topic": "topic",
            }
        ]

    monkeypatch.setattr(v8_memory, "distill_events", fake_distill)
    runner.build_memory(_conversation(turns=4), str(tmp_path / "memory"))

    assert seen_recent == [
        [],
        ["event 1"],
        ["event 1", "event 2"],
        ["event 2", "event 3"],
    ]


def test_v10_session_window_uses_one_writer_call_per_session(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v10")
    monkeypatch.setenv("NATIVEMEM_V10_WRITE_TURNS", "session")
    monkeypatch.setenv("NATIVEMEM_V10_TIDY_EVERY_SESSIONS", "0")
    monkeypatch.setenv("NATIVEMEM_V10_FINAL_TIDY_PASSES", "0")
    seen = []

    def fake_distill(turns, obs, dia_ids, **_):
        seen.append((len(turns), len(dia_ids)))
        return []

    monkeypatch.setattr(v8_memory, "distill_events", fake_distill)
    runner.build_memory(
        _conversation(turns=7, sessions=2), str(tmp_path / "memory")
    )

    assert seen == [(7, 7), (7, 7)]


def test_v10_summary_is_updated_in_the_writer_call(monkeypatch):
    monkeypatch.setenv("NATIVEMEM_V8_VERIFY", "off")
    calls = []

    def fake_call(messages, **kwargs):
        calls.append((messages, kwargs))
        return json.dumps(
            {
                "events": [
                    {
                        "when": "2023-05-07",
                        "summary": "User adopted Poppy [1]",
                        "refs": [1],
                        "topic": "pets",
                    }
                ],
                "context_summary": "User adopted a dog named Poppy.",
            }
        )

    monkeypatch.setattr(v8_memory, "_distill_call", fake_call)
    events, summary = v10_memory.distill_with_context(
        [("User", "I adopted a dog named Poppy")],
        "2023-05-07",
        ["D1:1"],
        prior_context="",
        previous_summary="",
        summary_max_words=40,
    )

    assert len(calls) == 1
    assert events[0]["dia_ids"] == ["D1:1"]
    assert summary == "User adopted a dog named Poppy."


def test_v10_maintenance_frequency_and_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v10")
    monkeypatch.setenv("NATIVEMEM_V10_TIDY_EVERY_SESSIONS", "2")
    monkeypatch.setenv("NATIVEMEM_V10_SESSION_TIDY_PASSES", "2")
    monkeypatch.setenv("NATIVEMEM_V10_FINAL_TIDY_PASSES", "0")

    monkeypatch.setattr(
        v8_memory,
        "distill_events",
        lambda turns, obs, dia_ids, **_: [
            {
                "when": obs,
                "summary": "event",
                "dia_ids": list(dia_ids),
                "topic": "topic",
            }
        ],
    )
    dedup_calls = []
    tidy_calls = []
    monkeypatch.setattr(
        v8_memory, "dedup_topic_files", lambda *_: dedup_calls.append(1) or 0
    )
    monkeypatch.setattr(
        runner,
        "_v8_tidy",
        lambda _memory, touched, *_args, **_kwargs: tidy_calls.append(set(touched)),
    )

    runner.build_memory(
        _conversation(turns=1, sessions=3), str(tmp_path / "memory")
    )

    assert len(dedup_calls) == 2
    assert len(tidy_calls) == 2
    assert tidy_calls == [{"topic"}, {"topic"}]


def test_v10_final_tidy_pass_count(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v10")
    monkeypatch.setenv("NATIVEMEM_V10_TIDY_EVERY_SESSIONS", "0")
    monkeypatch.setenv("NATIVEMEM_V10_FINAL_TIDY_PASSES", "2")
    monkeypatch.setattr(
        v8_memory,
        "distill_events",
        lambda turns, obs, dia_ids, **_: [
            {
                "when": obs,
                "summary": "event",
                "dia_ids": list(dia_ids),
                "topic": "topic",
            }
        ],
    )
    tidy_calls = []
    monkeypatch.setattr(
        runner,
        "_v8_tidy",
        lambda _memory, touched, *_args, **kwargs: tidy_calls.append(
            (set(touched), kwargs.get("final", False))
        ),
    )

    runner.build_memory(_conversation(turns=1), str(tmp_path / "memory"))

    assert tidy_calls == [({"topic"}, True), ({"topic"}, True)]


def test_build_only_writes_stats_without_question_checkpoint(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "locomo.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "conversation": _conversation(turns=1),
                    "qa": [
                        {
                            "question": "What happened?",
                            "answer": "A fact",
                            "category": 1,
                        }
                    ],
                }
            ]
        )
    )
    output = tmp_path / "run" / "build.json"
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v10")
    monkeypatch.setattr(runner, "DATA_PATH", str(dataset))
    monkeypatch.setattr(runner, "build_memory", lambda *_args, **_kwargs: (1.5, 3))
    monkeypatch.setattr(
        runner.nativemem_runtime,
        "CALL_LOG",
        [
            {
                "phase": "v8_distill",
                "prompt_tokens": 100,
                "completion_tokens": 20,
            },
            {
                "phase": "v8_distill",
                "prompt_tokens": 120,
                "completion_tokens": 25,
            },
            {
                "phase": "v8_sections",
                "prompt_tokens": 80,
                "completion_tokens": 10,
            },
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nativemem.py",
            "--sample",
            "0",
            "--output",
            str(output),
            "--build-only",
        ],
    )

    runner.main()

    records = json.loads(output.read_text())
    assert len(records) == 1
    assert records[0]["question_id"] == "_build_stats"
    assert records[0]["method"] == "NativeMem-v10"
    assert records[0]["sample"] == 0
    assert records[0]["max_sessions"] is None
    assert records[0]["builder_model"] == runner.ALIYUN_MODEL
    assert records[0]["memory_dir"].endswith("memory_sample0")
    assert records[0]["v10_config"]["write_turns"] == 6
    assert records[0]["build_phase_usage"] == {
        "v8_distill": {"calls": 2, "tokens_in": 220, "tokens_out": 45},
        "v8_sections": {"calls": 1, "tokens_in": 80, "tokens_out": 10},
    }
    assert not output.with_suffix(".checkpoint.json").exists()
