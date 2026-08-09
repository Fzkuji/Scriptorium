"""Durable per-question checkpoints for benchmark adapters."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def memory_sha256(memory_dir: str | os.PathLike[str]) -> str:
    """Bind a checkpoint to the exact memory tree used to answer questions."""

    root = Path(memory_dir).resolve()
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def checkpoint_path(output: str | os.PathLike[str]) -> Path:
    path = Path(output)
    return path.with_name(f"{path.stem}.checkpoint{path.suffix}")


def checkpoint_identity(
    *, sample: int, qas: list[dict], memory_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    return {
        "sample": sample,
        "questions_sha256": canonical_sha256(qas),
        "memory_sha256": memory_sha256(memory_dir),
    }


def atomic_json(path: str | os.PathLike[str], value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_or_create(
    path: str | os.PathLike[str],
    *,
    identity: dict[str, Any],
    build_record: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "identity": identity,
            "build_record": build_record,
            "answers": {},
            "status": "running",
            "updated_at": utc_now(),
        }

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("question checkpoint must be a JSON object")
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("question checkpoint schema version differs")
    if state.get("identity") != identity:
        raise ValueError("question checkpoint identity differs")
    if not isinstance(state.get("build_record"), dict):
        raise ValueError("question checkpoint lacks build_record")
    answers = state.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("question checkpoint answers must be an object")
    return state


def completed_answers(
    state: dict[str, Any], *, sample: int, qas: list[dict], require_answer: bool
) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    for raw_index, record in state["answers"].items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("question checkpoint has a non-integer index") from exc
        if str(index) != str(raw_index) or not 0 <= index < len(qas):
            raise ValueError("question checkpoint has an unknown question index")
        if not isinstance(record, dict):
            raise ValueError("question checkpoint answer must be an object")
        qa = qas[index]
        expected = {
            "question_id": f"s{sample}_q{index}",
            "question": qa["question"],
            "gold": str(qa.get("answer", qa.get("adversarial_answer", ""))),
            "category": qa.get("category"),
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"question checkpoint record {index} differs from the dataset"
            )
        if require_answer and not str(record.get("answer", "")).strip():
            continue
        completed[index] = record
    return completed


def save_answer(
    path: str | os.PathLike[str], state: dict[str, Any], index: int, record: dict
) -> None:
    state["answers"][str(index)] = record
    state["status"] = "running"
    state["updated_at"] = utc_now()
    atomic_json(path, state)


def mark_complete(path: str | os.PathLike[str], state: dict[str, Any]) -> None:
    state["status"] = "complete"
    state["updated_at"] = utc_now()
    atomic_json(path, state)
