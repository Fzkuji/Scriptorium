"""Converting a BEAM conversation into what the runner and judge read."""

import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))

from scripts.runners.beam.convert import (  # noqa: E402
    CATEGORIES, convert_conversation, convert_questions, parse_anchor,
    parse_literal,
)
from scripts.runners.conversation.data import sample_inventory  # noqa: E402
from scripts.runners.conversation.query import CARRIED_FIELDS  # noqa: E402

ROW = {
    "conversation_id": "1",
    "chat": [
        [
            {"role": "user", "content": "Sprint one ends March 29.",
             "id": "0", "time_anchor": "March-15-2024"},
            {"role": "assistant", "content": "Noted.", "id": "1",
             "time_anchor": "March-15-2024"},
        ],
        [
            {"role": "user", "content": "The API is at 250ms now.",
             "id": "2", "time_anchor": "April-05-2024"},
        ],
    ],
    "probing_questions": str({
        "information_extraction": [
            {"question": "When does sprint one end?", "answer": "March 29.",
             "difficulty": "easy",
             "rubric": "['LLM response should state: March 29']"},
        ],
        "abstention": [
            {"question": "What did the reviewers say?",
             "ideal_response": "There is no information about reviewers.",
             "difficulty": "medium", "rubric": "['must decline']"},
        ],
    }),
}


def test_anchor_becomes_an_iso_date_the_runtime_parses():
    assert parse_anchor("March-15-2024") == "2024-03-15"
    assert parse_anchor("nonsense") == ""


def test_literals_stored_as_strings_come_back_as_lists():
    assert parse_literal("['a', 'b']") == ["a", "b"]
    # Plain prose must survive untouched.
    assert parse_literal("just text") == "just text"


def test_sessions_keep_order_and_carry_dates():
    conversation = convert_conversation(ROW, "beam100K-1")

    assert conversation["session_1_date_time"] == "2024-03-15"
    assert conversation["session_2_date_time"] == "2024-04-05"
    assert [turn["dia_id"] for turn in conversation["session_1"]] == [
        "D1:1", "D1:2",
    ]
    assert conversation["session_1"][0]["speaker"] == "User"
    assert conversation["session_1"][1]["speaker"] == "Assistant"
    assert conversation["session_2"][0]["text"] == "The API is at 250ms now."


def test_questions_carry_gold_rubric_and_category():
    questions = convert_questions(ROW)

    extraction = next(q for q in questions if q["beam_category"] ==
                      "information_extraction")
    assert extraction["answer"] == "March 29."
    assert extraction["rubric"] == ["LLM response should state: March 29"]
    assert extraction["category"] == CATEGORIES.index(
        "information_extraction") + 1
    assert extraction["abstention"] is False


def test_abstention_gold_comes_from_ideal_response():
    abstention = next(q for q in convert_questions(ROW)
                      if q["beam_category"] == "abstention")

    assert abstention["answer"].startswith("There is no information")
    assert abstention["abstention"] is True


def test_unknown_category_is_refused_rather_than_dropped():
    row = dict(ROW, probing_questions=str({"made_up_category": []}))
    with pytest.raises(ValueError, match="unknown BEAM categories"):
        convert_questions(row)


def test_judge_fields_survive_into_the_answer_record():
    """convert emits exactly what query.py carries through to the judge."""
    question = convert_questions(ROW)[0]
    for field in ("beam_category", "rubric", "abstention", "difficulty"):
        assert field in question
        assert field in CARRIED_FIELDS
    # The gold answer is not among them: it reaches the record as `gold`.
    assert "answer" not in CARRIED_FIELDS


def test_inventory_counts_abstention_beyond_locomo_categories():
    sample = {
        "sample_id": "beam100K-1",
        "conversation": convert_conversation(ROW, "beam100K-1"),
        "qa": convert_questions(ROW),
    }

    inventory = sample_inventory(sample)

    assert inventory["sessions"] == 2
    assert inventory["messages"] == 3
    assert inventory["questions"] == 2
    assert inventory["primary_questions"] == 1
    assert inventory["adversarial_questions"] == 1
