"""Input selection and resumable output records for LongMemEval."""

from pathlib import Path
from typing import Any

from scripts.nativemem.common import atomic_json, read_json


def source_records(analysis_path: Path) -> list[dict[str, Any]]:
    value = read_json(analysis_path)
    records = value.get("records") if isinstance(value, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("analysis file has no records")
    indices = [int(record["dataset_index"]) for record in records]
    if len(indices) != len(set(indices)):
        raise ValueError("analysis file has duplicate dataset indices")
    return sorted(records, key=lambda record: int(record["dataset_index"]))


def load_completed_results(
    output_dir: Path,
    dataset: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    expected = {int(source["dataset_index"]) for source in sources}
    completed: dict[int, dict[str, Any]] = {}
    for path in sorted((output_dir / "items").glob("*.json")):
        record = read_json(path)
        if not isinstance(record, dict):
            raise ValueError(f"invalid item record: {path}")
        try:
            index = int(record["dataset_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid item index: {path}") from exc
        if index not in expected or index >= len(dataset):
            raise ValueError(f"item {index} is not part of this run")
        item = dataset[index]
        if (
            str(record.get("question_id")) != str(item["question_id"])
            or record.get("question") != item["question"]
            or not str(record.get("answer", "")).strip()
        ):
            raise ValueError(f"item {index} identity or answer is invalid")
        if index in completed:
            raise ValueError(f"duplicate item {index}")
        completed[index] = record
    return completed


def write_results(
    output_dir: Path, completed: dict[int, dict[str, Any]]
) -> None:
    atomic_json(
        output_dir / "results.json",
        [completed[index] for index in sorted(completed)],
    )
