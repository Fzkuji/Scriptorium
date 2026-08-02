"""Ablation matrix selection, checkpoints, and evaluator outputs."""

import json
from pathlib import Path
from typing import Any

from scripts.nativemem.common import atomic_json, read_json
from scripts.nativemem.longmemeval import support as longmemeval


CODE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCOMO_DATA = (
    CODE_ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
)


def selected_rows(
    matrix: Path,
    benchmarks: set[str],
    unit_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in matrix.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chosen = [
        row
        for row in rows
        if row["benchmark"] in benchmarks
        and (not unit_ids or row["unit_id"] in unit_ids)
    ]
    if not chosen:
        raise ValueError("matrix contains no selected benchmark rows")
    return chosen


def build_dir(build_root: Path, row: dict[str, Any]) -> Path:
    if relative := row.get("build_subdir"):
        return build_root / str(relative)
    output = Path(str(row.get("output_dir", row["unit_id"])))
    try:
        output = output.relative_to("results/formal/gpt56-chunk-curve")
    except ValueError:
        output = Path(row["unit_id"])
    return build_root / output


def validate_build(path: Path, row: dict[str, Any]) -> None:
    build_path = path / "build.json"
    if not build_path.is_file() or read_json(build_path).get("status") != "complete":
        raise RuntimeError(f"NativeMem build is incomplete: {path}")
    if not (path / "memory").is_dir():
        raise RuntimeError(f"NativeMem memory is absent: {path}")


def load_unit(
    row: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    selection = row["selection"]
    benchmark = row["benchmark"]
    if benchmark == "locomo":
        data = read_json(Path(row.get("data_path", DEFAULT_LOCOMO_DATA)))
        item = data[int(selection["sample_index"])]
        if item["sample_id"] != selection["sample_id"]:
            raise ValueError("LoCoMo unit identity changed")
        questions = [{
            "question_id": f"q{index}",
            "question": question["question"],
            "gold": str(question.get(
                "answer", question.get("adversarial_answer", "")
            )),
            "category": question.get("category"),
            "evidence": question.get("evidence", []),
        } for index, question in enumerate(item["qa"])
          if question.get("category") in (1, 2, 3, 4)]
        return item["conversation"], questions, {"sample_id": item["sample_id"]}
    if benchmark == "longmemeval-s":
        data_path = Path(row.get("data_path", longmemeval.DEFAULT_DATA))
        index = int(selection["dataset_index"])
        item = longmemeval.load_dataset(data_path)[index]
        if item["question_id"] != selection["question_id"]:
            raise ValueError("LongMemEval unit identity changed")
        question = {
            "question_id": item["question_id"],
            "question": item["question"],
            "question_date": item["question_date"],
            "gold": item["answer"],
            "question_type": item["question_type"],
            "answer_session_ids": item["answer_session_ids"],
        }
        return longmemeval.to_conversation(item, index), [question], {
            "dataset_index": index,
        }
    raise ValueError(f"unsupported benchmark: {benchmark}")


def item_path(
    output_root: Path,
    condition: str,
    row: dict[str, Any],
    question: dict[str, Any],
) -> Path:
    return (
        output_root
        / condition
        / row["benchmark"]
        / row["unit_id"]
        / "items"
        / f"{question['question_id']}.json"
    )


def load_completed(
    output_root: Path,
    condition: str,
    row: dict[str, Any],
    questions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for question in questions:
        path = item_path(output_root, condition, row, question)
        if not path.is_file():
            continue
        record = read_json(path)
        exact = {
            "condition": condition,
            "benchmark": row["benchmark"],
            "unit_id": row["unit_id"],
            "question_id": question["question_id"],
            "question": question.get(
                "question_text", question.get("question", "")
            ),
        }
        if any(record.get(key) != value for key, value in exact.items()):
            raise RuntimeError(f"completed item identity differs: {path}")
        if (
            not str(record.get("answer", "")).strip()
            or record.get("memory_unchanged") is not True
        ):
            raise RuntimeError(f"completed item is invalid: {path}")
        completed[str(question["question_id"])] = record
    return completed


def write_unit_outputs(
    output_root: Path,
    condition: str,
    row: dict[str, Any],
    questions: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
) -> None:
    records = [
        completed[str(question["question_id"])]
        for question in questions
        if str(question["question_id"]) in completed
    ]
    unit_root = output_root / condition / row["benchmark"] / row["unit_id"]
    atomic_json(unit_root / "results.json", records)
    atomic_json(unit_root / "status.json", {
        "phase": "complete" if len(records) == len(questions) else "running",
        "condition": condition,
        "benchmark": row["benchmark"],
        "unit_id": row["unit_id"],
        "completed": len(records),
        "total": len(questions),
    })


def publish_condition_outputs(
    output_root: Path,
    condition: str,
    rows: list[dict[str, Any]],
) -> None:
    condition_root = output_root / condition
    beam_predictions: list[dict[str, str]] = []
    locomo_sample_index = 0
    for row in rows:
        _, questions, _ = load_unit(row)
        records = read_json(
            condition_root
            / row["benchmark"]
            / row["unit_id"]
            / "results.json"
        )
        if len(records) != len(questions):
            raise RuntimeError(f"incomplete QA unit: {row['unit_id']}")
        if row["benchmark"] == "locomo":
            evaluator_records = [{
                key: record[key]
                for key in (
                    "question_id",
                    "question",
                    "gold",
                    "category",
                    "answer",
                    "memories",
                )
            } for record in records]
            atomic_json(
                condition_root
                / "locomo"
                / f"sample{locomo_sample_index}_questions.json",
                evaluator_records,
            )
            locomo_sample_index += 1
        elif row["benchmark"] == "beam-100k":
            beam_predictions.extend({
                "question_id": record["question_id"],
                "answer": record["answer"],
            } for record in records)
    if beam_predictions:
        prediction_path = condition_root / "beam-100k" / "predictions.jsonl"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in beam_predictions
            ),
            encoding="utf-8",
        )
    status_path = condition_root / "status.json"
    status = read_json(status_path) if status_path.is_file() else {}
    status.update({"phase": "complete", "condition": condition})
    if any(row["benchmark"] == "locomo" for row in rows):
        status["locomo_samples"] = locomo_sample_index
    if any(row["benchmark"] == "beam-100k" for row in rows):
        status["beam_questions"] = len(beam_predictions)
    atomic_json(status_path, status)
