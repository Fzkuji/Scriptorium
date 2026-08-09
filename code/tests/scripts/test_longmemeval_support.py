import json
from pathlib import Path

import pytest

from scripts.runners.longmemeval import support


FIXTURE = Path(__file__).parents[1] / "fixtures" / "longmemeval_s_tiny.json"


class FakeTracker:
    def reset(self, _phase):
        pass

    def snapshot(self, _phase):
        return {
            "calls": 1,
            "tokens_in": 10,
            "tokens_out": 3,
            "llm_time_s": 0.01,
        }


class FakeBackend:
    def __init__(self, answer="Bought a notebook"):
        self.answer = answer
        self.tracker = FakeTracker()
        self.build_calls = 0
        self.answer_calls = 0

    def build_memory(self, conversation, memory_dir):
        self.build_calls += 1
        ids = [
            turn["dia_id"]
            for key, session in conversation.items()
            if key.startswith("session_") and not key.endswith("_date_time")
            for turn in session
        ]
        assert ids == ["D1:1", "D1:2", "D2:1", "D2:2"]
        path = Path(memory_dir) / "topics" / "items.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Bought a notebook.\n", encoding="utf-8")
        return 0.25, 1

    @staticmethod
    def build_turn_index(conversation):
        result = {}
        order = 0
        session_number = 1
        while f"session_{session_number}" in conversation:
            for turn in conversation[f"session_{session_number}"]:
                result[turn["dia_id"]] = {"order": order}
                order += 1
            session_number += 1
        return result

    def collect_and_answer_longmemeval(self, item, _memory_dir, turn_index):
        self.answer_calls += 1
        assert support.model_question(item) == (
            "Current Date: 2023/05/22 (Mon) 08:00\n"
            "Question: What did I do first?"
        )
        assert len(turn_index) == 4
        return (
            [{"text": "Bought a notebook", "date": "2023-05-19"}],
            2,
            self.answer,
            [{"type": "bash", "accepted": True}],
        )


def run_meta():
    return {
        "method": {
            "name": "NativeMem",
            "implementation": "current",
            "single_model_retrieve_answer": True,
        },
        "models": {
            "builder": "test-model",
            "retriever": "test-model",
            "answerer": "test-model",
            "base_url": "https://example.test/v1",
        },
        "config": {"build": {}, "query": {}},
        "code": {"git_commit": "test", "runner_sha256": "r"},
        "request_audit": {"mode": "test"},
    }


def test_fixture_conversion_assigns_every_turn_id():
    data = support.load_dataset(FIXTURE, expected_count=2)
    conversation = support.to_conversation(data[0], 0)

    assert conversation["session_1_date_time"] == "2023-05-20"
    assert conversation["session_2_date_time"] == "2023-05-21"
    assert [turn["dia_id"] for turn in conversation["session_1"]] == ["D1:1", "D1:2"]
    assert [turn["dia_id"] for turn in conversation["session_2"]] == ["D2:1", "D2:2"]


def test_conversion_rejects_misaligned_session_metadata():
    item = support.load_dataset(FIXTURE, expected_count=2)[0]
    item["haystack_dates"] = item["haystack_dates"][:1]

    with pytest.raises(support.DataValidationError, match="lengths differ"):
        support.to_conversation(item, 0)


def test_conversion_preserves_empty_source_turn_anchor():
    item = support.load_dataset(FIXTURE, expected_count=2)[1]
    item["haystack_sessions"][0].insert(1, {"role": "assistant", "content": ""})

    conversation = support.to_conversation(item, 1)

    assert conversation["session_1"][1] == {
        "speaker": "assistant",
        "text": "",
        "dia_id": "D1:2",
    }
    assert conversation["session_1"][2]["dia_id"] == "D1:3"


def test_run_item_is_resumable_and_does_not_rebuild_complete_state(tmp_path):
    item = support.load_dataset(FIXTURE, expected_count=2)[0]
    backend = FakeBackend()

    ok, detail, checkpoint = support.run_item(
        0, item, tmp_path, backend, run_meta(), resume=False
    )
    second = FakeBackend(answer="must not replace the answer")
    resumed = support.run_item(
        0, item, tmp_path, second, run_meta(), resume=True
    )

    assert ok and detail == "complete"
    assert checkpoint["retrieval"]["tool_trace"] == [
        {"type": "bash", "accepted": True}
    ]
    assert checkpoint["answer"]["hypothesis"] == "Bought a notebook"
    assert resumed[0:2] == (True, "already complete")
    assert second.build_calls == 0 and second.answer_calls == 0


def test_empty_answer_fails_and_resume_reuses_completed_build(tmp_path):
    item = support.load_dataset(FIXTURE, expected_count=2)[0]
    blank = FakeBackend(answer="")
    ok, detail, checkpoint = support.run_item(
        0, item, tmp_path, blank, run_meta(), resume=False
    )
    good = FakeBackend()
    resumed = support.run_item(
        0, item, tmp_path, good, run_meta(), resume=True
    )

    assert not ok and "empty hypothesis" in detail
    assert checkpoint["error"]["stage"] == "retrieve_answer"
    assert resumed[0:2] == (True, "complete")
    assert good.build_calls == 0 and good.answer_calls == 1


def test_build_only_saves_reusable_memory_without_answering(tmp_path):
    item = support.load_dataset(FIXTURE, expected_count=2)[0]
    backend = FakeBackend()

    ok, detail, checkpoint = support.run_item(
        0,
        item,
        tmp_path,
        backend,
        run_meta(),
        resume=False,
        build_only=True,
    )

    assert ok and detail == "build complete"
    assert checkpoint["status"] == "built"
    assert checkpoint["build"]["status"] == "complete"
    assert "answer" not in checkpoint and "retrieval" not in checkpoint
    assert backend.build_calls == 1 and backend.answer_calls == 0


def test_incomplete_state_requires_resume(tmp_path):
    item = support.load_dataset(FIXTURE, expected_count=2)[0]
    paths = support.item_paths(tmp_path, 0, item["question_id"])
    paths["item_dir"].mkdir(parents=True)
    paths["checkpoint"].write_text('{"status":"building"}', encoding="utf-8")

    with pytest.raises(support.ExistingStateError, match="--resume"):
        support.run_item(0, item, tmp_path, FakeBackend(), run_meta(), resume=False)


def test_item_lock_rejects_second_writer_and_releases(tmp_path):
    lock_path = tmp_path / ".locks" / "item.lock"
    first = support._acquire_item_lock(lock_path)
    try:
        with pytest.raises(support.ExistingStateError, match="already locked"):
            support._acquire_item_lock(lock_path)
    finally:
        support._release_item_lock(first)

    second = support._acquire_item_lock(lock_path)
    support._release_item_lock(second)


def test_refresh_outputs_ignores_failed_items(tmp_path):
    data = support.load_dataset(FIXTURE, expected_count=2)
    support.run_item(0, data[0], tmp_path, FakeBackend(), run_meta(), resume=False)
    support.run_item(1, data[1], tmp_path, FakeBackend(answer=""), run_meta(), resume=False)
    manifest = {}

    support.refresh_outputs(tmp_path, manifest)

    assert len(json.loads((tmp_path / "results.json").read_text())) == 1
    assert json.loads((tmp_path / "hypotheses.jsonl").read_text()) == {
        "question_id": "tiny_temporal_0",
        "hypothesis": "Bought a notebook",
    }
    assert manifest["checkpoint_counts"] == {"complete": 1, "failed": 1}


def test_longmemeval_s_guard_rejects_oracle_sized_histories():
    data = support.load_dataset(FIXTURE, expected_count=2)

    with pytest.raises(support.DataValidationError, match="likely LongMemEval-oracle"):
        support.validate_longmemeval_s(data)
