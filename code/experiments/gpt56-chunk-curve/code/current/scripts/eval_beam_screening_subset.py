#!/usr/bin/env python3
"""Offline deterministic diagnostics for the 40-question BEAM screening subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
PROTOCOL_CLASS = "screening_subset"
SCORE_SOURCE = "normalized_exact_match_diagnostic"
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
EXPECTED_SELECTIONS = {
    "100K-conv-1": {
        "split": "100K",
        "conversation_index": 0,
        "conversation_id": "1",
    },
    "100K-conv-2": {
        "split": "100K",
        "conversation_index": 1,
        "conversation_id": "2",
    },
}


class ScreeningEvaluationError(RuntimeError):
    """The input is outside the frozen two-conversation screening protocol."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _read_regular(path: Path) -> tuple[bytes, str]:
    path = path.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ScreeningEvaluationError(f"input is not a regular file: {path}")
    payload = path.read_bytes()
    return payload, _sha256_bytes(payload)


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _load_gold_units(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(paths) != 2 or len({str(path.expanduser().absolute()) for path in paths}) != 2:
        raise ScreeningEvaluationError("exactly two distinct gold unit files are required")
    units: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for path in paths:
        payload, digest = _read_regular(path)
        try:
            unit = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ScreeningEvaluationError(f"gold unit is not valid JSON: {path}") from exc
        if not isinstance(unit, dict) or unit.get("benchmark") != "beam-100k":
            raise ScreeningEvaluationError("gold unit is not a BEAM-100K object")
        unit_id = unit.get("unit_id")
        expected_selection = EXPECTED_SELECTIONS.get(str(unit_id))
        if expected_selection is None or unit.get("selection") != expected_selection:
            raise ScreeningEvaluationError("gold unit selection is outside the screening subset")
        questions = unit.get("questions")
        if not isinstance(questions, list) or len(questions) != 20:
            raise ScreeningEvaluationError(f"{unit_id} must contain exactly 20 questions")
        units.append(unit)
        provenance.append(
            {
                "path": str(path.expanduser().absolute()),
                "sha256": digest,
                "unit_id": unit_id,
                "selection": expected_selection,
                "question_count": 20,
            }
        )
    units.sort(key=lambda unit: int(unit["selection"]["conversation_index"]))
    provenance.sort(key=lambda item: int(item["selection"]["conversation_index"]))
    if {str(unit["unit_id"]) for unit in units} != set(EXPECTED_SELECTIONS):
        raise ScreeningEvaluationError("gold units do not match both frozen conversations")
    return units, provenance


def _gold_records(units: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for unit in units:
        type_counts: Counter[str] = Counter()
        conversation_id = str(unit["selection"]["conversation_id"])
        for question_index, question in enumerate(unit["questions"]):
            if not isinstance(question, dict):
                raise ScreeningEvaluationError("gold question is not an object")
            question_id = question.get("question_id")
            if question_id != f"{conversation_id}-q{question_index}":
                raise ScreeningEvaluationError("gold question ID or order differs")
            if question_id in seen:
                raise ScreeningEvaluationError(f"duplicate gold question ID: {question_id}")
            seen.add(str(question_id))
            question_type = str(question.get("question_type", ""))
            expected_gold_field = GOLD_FIELD_BY_TYPE.get(question_type)
            if expected_gold_field is None or question.get("gold_field") != expected_gold_field:
                raise ScreeningEvaluationError(f"gold-field mapping differs for {question_id}")
            gold = question.get("gold")
            if not isinstance(gold, str) or question.get(expected_gold_field) != gold:
                raise ScreeningEvaluationError(f"gold answer differs for {question_id}")
            rubric = question.get("rubric_nuggets")
            if (
                not isinstance(rubric, list)
                or not rubric
                or any(not isinstance(item, str) or not item.strip() for item in rubric)
            ):
                raise ScreeningEvaluationError(f"rubric differs for {question_id}")
            question_text = question.get("question_text", question.get("question"))
            if not isinstance(question_text, str) or not question_text.strip():
                raise ScreeningEvaluationError(f"question text is empty for {question_id}")
            type_counts[question_type] += 1
            records.append(
                {
                    "question_id": str(question_id),
                    "unit_id": str(unit["unit_id"]),
                    "conversation_index": unit["selection"]["conversation_index"],
                    "question_index": question_index,
                    "question_type": question_type,
                    "gold_field": expected_gold_field,
                    "question": question_text,
                    "gold": gold,
                    "rubric": list(rubric),
                }
            )
        if type_counts != Counter({question_type: 2 for question_type in QUESTION_TYPES}):
            raise ScreeningEvaluationError(
                f"{unit['unit_id']} does not contain two questions per BEAM type"
            )
    if len(records) != 40:
        raise ScreeningEvaluationError("gold subset must contain exactly 40 questions")
    return records


def _load_predictions(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    payload, digest = _read_regular(path)
    predictions: dict[str, str] = {}
    for line_number, raw_line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ScreeningEvaluationError(
                f"prediction line {line_number} is not valid JSON"
            ) from exc
        if not isinstance(record, dict) or set(record) != {"question_id", "answer"}:
            raise ScreeningEvaluationError(
                f"prediction line {line_number} must contain question_id and answer only"
            )
        question_id = record["question_id"]
        answer = record["answer"]
        if not isinstance(question_id, str) or not question_id:
            raise ScreeningEvaluationError(f"prediction line {line_number} has no ID")
        if not isinstance(answer, str):
            raise ScreeningEvaluationError(
                f"prediction line {line_number} answer is not a string"
            )
        if question_id in predictions:
            raise ScreeningEvaluationError(f"duplicate prediction ID: {question_id}")
        predictions[question_id] = answer
    return predictions, {
        "path": str(path.expanduser().absolute()),
        "sha256": digest,
        "question_count": len(predictions),
    }


def _group_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    raw = sum(bool(record["raw_exact_match"]) for record in records)
    normalized = sum(bool(record["normalized_exact_match"]) for record in records)
    count = len(records)
    normalized_rate = normalized / count if count else None
    return {
        "questions": count,
        "raw_exact_matches": raw,
        "raw_exact_match": raw / count if count else None,
        "raw_exact_match_percent": raw / count * 100 if count else None,
        "normalized_exact_matches": normalized,
        "normalized_exact_match": normalized_rate,
        "normalized_exact_match_percent": (
            normalized_rate * 100 if normalized_rate is not None else None
        ),
        "avg_score": mean(record["score"] for record in records) if records else None,
        "pass_threshold": 0.5,
        "passed": normalized,
        "pass_rate": normalized_rate,
        "pass_accuracy_percent": (
            normalized_rate * 100 if normalized_rate is not None else None
        ),
        "score_source": SCORE_SOURCE,
    }


def evaluate(*, gold_units: Sequence[Path], predictions_path: Path) -> dict[str, Any]:
    """Validate and score the frozen two-conversation subset without model calls."""

    units, gold_provenance = _load_gold_units(gold_units)
    gold_records = _gold_records(units)
    predictions, prediction_provenance = _load_predictions(predictions_path)
    gold_ids = [record["question_id"] for record in gold_records]
    gold_set = set(gold_ids)
    prediction_set = set(predictions)
    missing = sorted(gold_set - prediction_set)
    extra = sorted(prediction_set - gold_set)
    if missing or extra:
        raise ScreeningEvaluationError(
            f"prediction IDs differ: missing={missing}, extra={extra}"
        )

    records: list[dict[str, Any]] = []
    by_type: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for gold_record in gold_records:
        prediction = predictions[gold_record["question_id"]]
        raw_match = prediction == gold_record["gold"]
        normalized_match = _normalized_text(prediction) == _normalized_text(
            gold_record["gold"]
        )
        record = {
            **gold_record,
            "answer": prediction,
            "raw_exact_match": raw_match,
            "normalized_exact_match": normalized_match,
            "score": 1.0 if normalized_match else 0.0,
        }
        records.append(record)
        by_type[record["question_type"]].append(record)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_class": PROTOCOL_CLASS,
        "benchmark": "BEAM",
        "dataset_split": "100K",
        "formal_scope_verified": False,
        "claim_scope": (
            "two-conversation BEAM-100K screening subset; not a semantic, "
            "formal, full, or official BEAM score"
        ),
        "metric_scope": "deterministic_string_match_diagnostic",
        "score_source": SCORE_SOURCE,
        "question_count": len(records),
        "inputs": {
            "gold_units": gold_provenance,
            "predictions": prediction_provenance,
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_bytes(Path(__file__).read_bytes()),
            },
        },
        "coverage": {
            "gold_ids_sha256": _canonical_hash(sorted(gold_ids)),
            "prediction_ids_sha256": _canonical_hash(sorted(predictions)),
            "ids_exact_match": True,
            "duplicate_gold_ids": [],
            "duplicate_prediction_ids": [],
            "missing_prediction_ids": [],
            "extra_prediction_ids": [],
        },
        "metrics": {
            "overall": _group_metrics(records),
            "by_question_type": {
                question_type: _group_metrics(by_type[question_type])
                for question_type in QUESTION_TYPES
            },
        },
        "records": records,
        "official_beam_boundary": {
            "rubric_judge_executed": False,
            "official_or_formal_score": False,
            "judge_ready_records_included": True,
        },
    }
    report["report_content_sha256"] = _canonical_hash(report)
    return report


def write_report_no_clobber(path: Path, report: dict[str, Any]) -> None:
    """Atomically publish one hash-bound report without replacing a path."""

    expected_hash = report.get("report_content_sha256")
    content = dict(report)
    content.pop("report_content_sha256", None)
    if expected_hash != _canonical_hash(content):
        raise ScreeningEvaluationError("report content hash differs")
    output = path.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-unit", action="append", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate(
        gold_units=args.gold_unit,
        predictions_path=args.predictions,
    )
    write_report_no_clobber(args.output, report)
    print(
        "wrote screening_subset normalized-EM diagnostic for "
        f"{report['question_count']} questions"
    )
    return 0


__all__ = [
    "ScreeningEvaluationError",
    "evaluate",
    "main",
    "write_report_no_clobber",
]


if __name__ == "__main__":
    raise SystemExit(main())
