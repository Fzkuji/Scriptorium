from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


QUESTION_TYPES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)

GOLD_FIELD_BY_TYPE = {
    "abstention": "ideal_response",
    "contradiction_resolution": "ideal_answer",
    "event_ordering": "answer",
    "information_extraction": "answer",
    "instruction_following": "expected_compliance",
    "knowledge_update": "answer",
    "multi_session_reasoning": "answer",
    "preference_following": "expected_compliance",
    "summarization": "ideal_summary",
    "temporal_reasoning": "answer",
}


def _module():
    name = "scripts.eval_beam_screening_subset"
    assert importlib.util.find_spec(name) is not None, "offline evaluator module missing"
    return importlib.import_module(name)


def _write_gold_units(tmp_path: Path) -> list[Path]:
    paths: list[Path] = []
    for conversation_index, conversation_id in enumerate(("1", "2")):
        questions: list[dict[str, object]] = []
        question_index = 0
        for question_type in QUESTION_TYPES:
            for _ in range(2):
                question_id = f"{conversation_id}-q{question_index}"
                gold_field = GOLD_FIELD_BY_TYPE[question_type]
                gold = f"Gold {question_id}"
                questions.append(
                    {
                        "question_id": question_id,
                        "question_type": question_type,
                        "question_text": f"Question {question_id}?",
                        "gold_field": gold_field,
                        "gold": gold,
                        gold_field: gold,
                        "rubric_nuggets": [f"Rubric {question_id}"],
                    }
                )
                question_index += 1
        value = {
            "benchmark": "beam-100k",
            "unit_id": f"100K-conv-{conversation_id}",
            "selection": {
                "split": "100K",
                "conversation_index": conversation_index,
                "conversation_id": conversation_id,
            },
            "questions": questions,
        }
        path = tmp_path / f"unit-{conversation_id}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    return paths


def _write_predictions(tmp_path: Path, gold_units: list[Path]) -> Path:
    records: list[dict[str, str]] = []
    for path in gold_units:
        unit = json.loads(path.read_text(encoding="utf-8"))
        for question in unit["questions"]:
            records.append(
                {
                    "question_id": question["question_id"],
                    "answer": question["gold"],
                }
            )
    records[1]["answer"] = "  GOLD   1-q1  "
    records[2]["answer"] = "incorrect"
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_evaluate_reports_screening_normalized_em_and_judge_ready_records(
    tmp_path: Path,
) -> None:
    mod = _module()
    gold_units = _write_gold_units(tmp_path)
    predictions = _write_predictions(tmp_path, gold_units)

    report = mod.evaluate(gold_units=gold_units, predictions_path=predictions)

    assert report["protocol_class"] == "screening_subset"
    assert report["formal_scope_verified"] is False
    assert report["question_count"] == 40
    assert report["coverage"]["ids_exact_match"] is True
    assert report["coverage"]["duplicate_gold_ids"] == []
    assert report["coverage"]["duplicate_prediction_ids"] == []
    assert (
        report["coverage"]["gold_ids_sha256"]
        == report["coverage"]["prediction_ids_sha256"]
    )
    assert report["coverage"]["missing_prediction_ids"] == []
    assert report["coverage"]["extra_prediction_ids"] == []
    overall = report["metrics"]["overall"]
    assert overall["raw_exact_matches"] == 38
    assert overall["raw_exact_match"] == 0.95
    assert overall["normalized_exact_matches"] == 39
    assert overall["normalized_exact_match"] == 0.975
    assert overall["avg_score"] == 0.975
    assert overall["pass_rate"] == 0.975
    assert overall["score_source"] == "normalized_exact_match_diagnostic"
    assert len(report["records"]) == 40
    assert report["records"][0]["question"] == "Question 1-q0?"
    assert report["records"][0]["answer"] == "Gold 1-q0"
    assert report["records"][0]["rubric"] == ["Rubric 1-q0"]
    assert report["records"][1]["raw_exact_match"] is False
    assert report["records"][1]["normalized_exact_match"] is True
    assert report["records"][2]["score"] == 0.0
    assert report["official_beam_boundary"] == {
        "rubric_judge_executed": False,
        "official_or_formal_score": False,
        "judge_ready_records_included": True,
    }
    assert isinstance(report["report_content_sha256"], str)
    assert len(report["report_content_sha256"]) == 64


def test_evaluate_rejects_duplicate_prediction_ids(tmp_path: Path) -> None:
    mod = _module()
    gold_units = _write_gold_units(tmp_path)
    predictions = _write_predictions(tmp_path, gold_units)
    first = predictions.read_text(encoding="utf-8").splitlines()[0]
    predictions.write_text(
        predictions.read_text(encoding="utf-8") + first + "\n",
        encoding="utf-8",
    )

    with pytest.raises(mod.ScreeningEvaluationError, match="duplicate prediction ID"):
        mod.evaluate(gold_units=gold_units, predictions_path=predictions)


def test_evaluate_rejects_missing_and_extra_prediction_ids(tmp_path: Path) -> None:
    mod = _module()
    gold_units = _write_gold_units(tmp_path)
    predictions = _write_predictions(tmp_path, gold_units)
    records = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    missing_id = records.pop()["question_id"]
    records.append({"question_id": "outside-q0", "answer": "extra"})
    predictions.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(mod.ScreeningEvaluationError) as caught:
        mod.evaluate(gold_units=gold_units, predictions_path=predictions)

    assert missing_id in str(caught.value)
    assert "outside-q0" in str(caught.value)


def test_write_report_is_no_clobber_and_content_hash_is_auditable(
    tmp_path: Path,
) -> None:
    mod = _module()
    gold_units = _write_gold_units(tmp_path)
    predictions = _write_predictions(tmp_path, gold_units)
    input_before = {
        path: path.read_bytes() for path in [*gold_units, predictions]
    }
    report = mod.evaluate(gold_units=gold_units, predictions_path=predictions)
    output = tmp_path / "reports" / "beam-screening.json"

    mod.write_report_no_clobber(output, report)

    stored = json.loads(output.read_text(encoding="utf-8"))
    content_hash = stored.pop("report_content_sha256")
    canonical = json.dumps(
        stored,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == content_hash
    assert {path: path.read_bytes() for path in input_before} == input_before
    original_output = output.read_bytes()
    with pytest.raises(FileExistsError):
        mod.write_report_no_clobber(output, report)
    assert output.read_bytes() == original_output


def test_main_writes_one_screening_report_without_model_requests(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _module()
    gold_units = _write_gold_units(tmp_path)
    predictions = _write_predictions(tmp_path, gold_units)
    output = tmp_path / "beam-screening.json"

    returncode = mod.main(
        [
            "--gold-unit",
            str(gold_units[0]),
            "--gold-unit",
            str(gold_units[1]),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ]
    )

    assert returncode == 0
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["protocol_class"] == "screening_subset"
    assert stored["formal_scope_verified"] is False
    assert stored["official_beam_boundary"]["rubric_judge_executed"] is False
    assert capsys.readouterr().out.strip() == (
        "wrote screening_subset normalized-EM diagnostic for 40 questions"
    )
