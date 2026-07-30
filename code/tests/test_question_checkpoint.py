import json

import pytest

from src.adapters import question_checkpoint as checkpoint


def qas():
    return [
        {"question": "q0", "answer": "a0", "category": 1},
        {"question": "q1", "answer": "a1", "category": 2},
    ]


def record(index, answer):
    return {
        "question_id": f"s0_q{index}",
        "question": f"q{index}",
        "gold": f"a{index}",
        "category": index + 1,
        "answer": answer,
    }


def test_checkpoint_resumes_only_nonempty_answers(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "topic.md").write_text("memory", encoding="utf-8")
    identity = checkpoint.checkpoint_identity(
        sample=0, qas=qas(), memory_dir=memory
    )
    path = tmp_path / "questions.checkpoint.json"
    state = checkpoint.load_or_create(
        path,
        identity=identity,
        build_record={"question_id": "_build_stats"},
    )
    checkpoint.save_answer(path, state, 0, record(0, "answer 0"))
    checkpoint.save_answer(path, state, 1, record(1, ""))

    restored = checkpoint.load_or_create(
        path,
        identity=identity,
        build_record={"question_id": "_build_stats", "replacement": True},
    )
    completed = checkpoint.completed_answers(
        restored, sample=0, qas=qas(), require_answer=True
    )

    assert list(completed) == [0]
    assert restored["build_record"] == {"question_id": "_build_stats"}
    assert json.loads(path.read_text())["answers"]["1"]["answer"] == ""


def test_checkpoint_rejects_changed_memory_or_question(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    topic = memory / "topic.md"
    topic.write_text("memory", encoding="utf-8")
    original = checkpoint.checkpoint_identity(
        sample=0, qas=qas(), memory_dir=memory
    )
    path = tmp_path / "questions.checkpoint.json"
    state = checkpoint.load_or_create(
        path,
        identity=original,
        build_record={"question_id": "_build_stats"},
    )
    checkpoint.save_answer(path, state, 0, record(0, "answer 0"))

    topic.write_text("changed", encoding="utf-8")
    changed = checkpoint.checkpoint_identity(
        sample=0, qas=qas(), memory_dir=memory
    )
    with pytest.raises(ValueError, match="identity differs"):
        checkpoint.load_or_create(
            path,
            identity=changed,
            build_record={"question_id": "_build_stats"},
        )


def test_checkpoint_rejects_record_that_differs_from_dataset(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    identity = checkpoint.checkpoint_identity(
        sample=0, qas=qas(), memory_dir=memory
    )
    state = checkpoint.load_or_create(
        tmp_path / "absent.json",
        identity=identity,
        build_record={"question_id": "_build_stats"},
    )
    state["answers"]["0"] = {**record(0, "answer 0"), "question": "wrong"}

    with pytest.raises(ValueError, match="differs from the dataset"):
        checkpoint.completed_answers(
            state, sample=0, qas=qas(), require_answer=True
        )
