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


def test_every_category_reference_field_is_read():
    """Each BEAM category names its reference differently; none may be lost."""
    row = dict(ROW, probing_questions=str({
        "contradiction_resolution": [
            {"question": "Have I used Flask?",
             "ideal_answer": "You have said both.", "rubric": "['contradiction']"},
        ],
        "summarization": [
            {"question": "Summarize the project.",
             "ideal_summary": "It began with Flask.", "rubric": "['Flask']"},
        ],
        "instruction_following": [
            {"question": "Show the schema.",
             "expected_compliance": "Response includes a highlighted code block.",
             "rubric": "['code block']"},
        ],
    }))

    golds = {q["beam_category"]: q["answer"] for q in convert_questions(row)}

    assert golds["contradiction_resolution"] == "You have said both."
    assert golds["summarization"] == "It began with Flask."
    assert golds["instruction_following"].startswith("Response includes")


def test_a_question_without_any_reference_is_refused():
    row = dict(ROW, probing_questions=str({
        "summarization": [{"question": "Summarize.", "rubric": "['x']"}],
    }))
    with pytest.raises(ValueError, match="no reference answer"):
        convert_questions(row)


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


def test_benchmark_and_sample_shape_must_agree():
    """A BEAM sample scored as LoCoMo keeps four questions and renames them."""
    from scripts.runners.conversation.data import check_benchmark

    beam = {
        "sample_id": "beam100K-1",
        "conversation": convert_conversation(ROW, "beam100K-1"),
        "qa": convert_questions(ROW),
    }
    locomo = {"sample_id": "conv-50", "qa": [
        {"question": "q", "answer": "a", "category": 1},
        {"question": "q5", "adversarial_answer": "d", "category": 5},
    ]}

    check_benchmark(beam, "beam")
    check_benchmark(locomo, "locomo")
    with pytest.raises(ValueError, match="needs questions carrying beam_category"):
        check_benchmark(locomo, "beam")
    with pytest.raises(ValueError, match="run it with --benchmark beam"):
        check_benchmark(beam, "locomo")


def test_inventory_counts_beam_abstention_not_beam_category_five():
    """BEAM's fifth category is contradiction resolution, not abstention."""
    from scripts.runners.conversation.data import sample_inventory

    row = dict(ROW, probing_questions=str({
        "information_extraction": [
            {"question": "a", "answer": "a", "rubric": "['x']"}],
        "contradiction_resolution": [
            {"question": "b", "ideal_answer": "b", "rubric": "['x']"}],
        "abstention": [
            {"question": "c", "ideal_response": "no record", "rubric": "['x']"}],
    }))
    questions = convert_questions(row)
    assert [q["category"] for q in questions if q["beam_category"] ==
            "contradiction_resolution"] == [5]

    inventory = sample_inventory({
        "sample_id": "beam100K-1",
        "conversation": convert_conversation(row, "beam100K-1"),
        "qa": questions,
    })

    assert inventory["adversarial_questions"] == 1
    assert inventory["primary_questions"] == 2


def test_a_string_rubric_is_not_split_into_characters():
    from scripts.evaluation import judges

    captured = {}

    def fake_chat(messages):
        captured["prompt"] = messages[-1]["content"]
        return '{"label": "CORRECT"}', {"prompt_tokens": 1, "completion_tokens": 1}

    original = judges.judge_chat
    judges.judge_chat = fake_chat
    try:
        judges.judge_beam("q", "gold", "answer", rubric="must state March 29")
    finally:
        judges.judge_chat = original

    assert "- must state March 29" in captured["prompt"]
    assert "- m\n- u" not in captured["prompt"]
