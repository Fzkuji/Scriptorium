"""Reading a session transcript keeps the conversation and drops the rest."""

import json
from pathlib import Path

import pytest

from memory.ingestion import read_transcript


def write(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    return path


def human(uuid: str, text, **extra) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "sessionId": "s1",
        "timestamp": "2026-08-09T10:00:00Z",
        "origin": {"kind": "human"},
        "message": {"content": text},
        **extra,
    }


def test_tool_results_are_not_conversation(tmp_path: Path):
    # In one real transcript 13685 of 15366 "user" records were tool
    # results. Recording those would bury the conversation in file listings.
    source = write(tmp_path / "s.jsonl", [
        human("u1", "I live in Shanghai"),
        {
            "type": "user", "uuid": "u2", "sessionId": "s1",
            "message": {"content": [
                {"type": "tool_result", "content": "total 48\ndrwxr-xr-x"}
            ]},
        },
    ])

    records = read_transcript(source)

    assert [record.content for record in records] == ["I live in Shanghai"]


def test_assistant_prose_is_kept_and_thinking_is_not(tmp_path: Path):
    source = write(tmp_path / "s.jsonl", [{
        "type": "assistant", "uuid": "a1", "sessionId": "s1",
        "message": {"content": [
            {"type": "thinking", "thinking": "let me consider"},
            {"type": "text", "text": "Noted."},
            {"type": "tool_use", "name": "Read", "input": {}},
        ]},
    }])

    records = read_transcript(source)

    assert [record.content for record in records] == ["Noted."]


def test_a_user_message_may_be_a_string_or_blocks(tmp_path: Path):
    source = write(tmp_path / "s.jsonl", [
        human("u1", "plain string"),
        human("u2", [{"type": "text", "text": "block form"}]),
    ])

    records = read_transcript(source)

    assert [record.content for record in records] == [
        "plain string", "block form"
    ]


def test_a_retracted_message_is_left_out(tmp_path: Path):
    source = write(tmp_path / "s.jsonl", [
        human("u1", "first thought"),
        human("u2", "what I meant", retractedMessageUuids=["u1"]),
    ])

    records = read_transcript(source)

    assert [record.content for record in records] == ["what I meant"]


def test_records_carry_a_stable_source_id(tmp_path: Path):
    source = write(tmp_path / "s.jsonl", [human("u1", "hello")])

    record = read_transcript(source)[0]

    assert record.source_id == "claude-code/s1/u1"
    assert record.role == "user"
    assert record.timestamp == "2026-08-09T10:00:00Z"


def test_a_truncated_line_does_not_lose_the_file(tmp_path: Path):
    # A transcript can be appended to while it is being read.
    source = tmp_path / "s.jsonl"
    source.write_text(
        json.dumps(human("u1", "kept")) + "\n{\"type\": \"user\", \"uu",
        encoding="utf-8",
    )

    assert [record.content for record in read_transcript(source)] == ["kept"]


def test_codex_transcripts_are_recognized(tmp_path: Path):
    source = write(tmp_path / "rollout.jsonl", [
        {"timestamp": "2026-08-09T10:00:00Z",
         "payload": {"role": "user", "id": "m1",
                     "content": [{"type": "text", "text": "hi"}]}},
        {"payload": {"role": "system", "content": "ignored"}},
    ])

    records = read_transcript(source)

    assert [record.provider for record in records] == ["codex"]
    assert [record.content for record in records] == ["hi"]


def test_a_missing_transcript_is_reported(tmp_path: Path):
    with pytest.raises(ValueError, match="not a file"):
        read_transcript(tmp_path / "absent.jsonl")
