"""Checkpoint and result records for resumable execution."""

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from memory import build as adapter

from memory.workspace_layout import RUNTIME_DIR
from scripts.runners.common import atomic_json, read_json, tree_sha256, utc_now
from .metrics import memory_inventory


def build_record(
    *,
    sample_index: int,
    sample: dict[str, Any],
    memory_dir: Path,
    model: str,
    build_config: adapter.BuildConfig,
    started: float,
    event_count: int,
    call_log: list[dict[str, Any]],
) -> dict[str, Any]:
    build_calls = [record for record in call_log if record["phase"] == "writer"]
    return {
        "status": "complete",
        "sample_index": sample_index,
        "sample_id": sample["sample_id"],
        "model": model,
        "config": asdict(build_config),
        "event_count": event_count,
        "wall_time_s": round(time.monotonic() - started, 3),
        "calls": sum(int(record.get("calls", 1) or 0) for record in build_calls),
        "input_tokens": sum(record["prompt_tokens"] for record in build_calls),
        "output_tokens": sum(
            record["completion_tokens"] for record in build_calls
        ),
        "cache_write_tokens": sum(
            int(record.get("cache_write_tokens", 0) or 0)
            for record in build_calls
        ),
        "cache_read_tokens": sum(
            int(record.get("cache_read_tokens", 0) or 0)
            for record in build_calls
        ),
        "anthropic_equivalent_cost_usd": round(sum(
            float(record.get("anthropic_equivalent_cost_usd", 0) or 0)
            for record in build_calls
        ), 8),
        "memory_dir": str(memory_dir),
        "memory_sha256": tree_sha256(memory_dir),
        # The same tree without the runtime's scratch area. Resuming
        # compares this one, so a cache or cursor written after the
        # build — or a sync agent dropping a conflict copy in there —
        # does not read as the memory having changed.
        "memory_content_sha256": tree_sha256(memory_dir, skip=(RUNTIME_DIR,)),
        "memory": memory_inventory(memory_dir),
        "finished_at": utc_now(),
    }


def load_completed(
    path: Path, sample_index: int
) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: dict[int, dict[str, Any]] = {}
    for record in read_json(path):
        if record.get("question_id") == "_build_stats":
            continue
        if int(record.get("sample_index", -1)) != sample_index:
            raise RuntimeError(f"result sample identity differs: {path}")
        question_index = int(record["question_index"])
        if str(record.get("answer", "")).strip():
            completed[question_index] = record
    return completed


def write_questions(
    path: Path,
    build: dict[str, Any],
    completed: dict[int, dict[str, Any]],
) -> None:
    atomic_json(
        path,
        [{"question_id": "_build_stats", **build}]
        + [completed[index] for index in sorted(completed)],
    )
