#!/usr/bin/env python3
"""Run paired V11 retrieval ablations on frozen LoCoMo and BEAM memories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import tempfile
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

from scripts import reanswer_longmemeval_existing_memory as reanswer  # noqa: E402
from scripts import run_gpt56_chunk_curve as build_runner  # noqa: E402


CONDITIONS = tuple(reanswer.V11_CONDITION_VIEWS)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


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
        row for row in rows
        if row["benchmark"] in benchmarks
        and (not unit_ids or row["unit_id"] in unit_ids)
    ]
    if not chosen:
        raise ValueError("matrix contains no selected benchmark rows")
    return chosen


def build_dir(build_root: Path, row: dict[str, Any]) -> Path:
    return build_runner.run_dir(build_root, row)


def validate_build(path: Path, row: dict[str, Any]) -> None:
    if not build_runner.valid_complete(path, row):
        raise RuntimeError(f"V11 build is incomplete: {path}")
    if not (path / "memory").is_dir():
        raise RuntimeError(f"V11 memory is absent: {path}")


def item_path(output_root: Path, condition: str, row: dict[str, Any], question: dict[str, Any]) -> Path:
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
            "question": question.get("question_text", question.get("question", "")),
        }
        if any(record.get(key) != value for key, value in exact.items()):
            raise RuntimeError(f"completed item identity differs: {path}")
        if not str(record.get("answer", "")).strip() or record.get("memory_unchanged") is not True:
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
    records = [completed[str(question["question_id"])] for question in questions if str(question["question_id"]) in completed]
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
    has_beam = any(row["benchmark"] == "beam-100k" for row in rows)
    has_locomo = any(row["benchmark"] == "locomo" for row in rows)
    locomo_sample_index = 0
    for row in rows:
        _, questions, _ = build_runner.load_unit(row)
        unit_root = condition_root / row["benchmark"] / row["unit_id"]
        records = read_json(unit_root / "results.json")
        if len(records) != len(questions):
            raise RuntimeError(f"incomplete QA unit: {unit_root}")
        if row["benchmark"] == "locomo":
            evaluator_records = [{
                "question_id": record["question_id"],
                "question": record["question"],
                "gold": record["gold"],
                "category": record["category"],
                "answer": record["answer"],
                "memories": record["memories"],
            } for record in records]
            atomic_json(condition_root / "locomo" / f"sample{locomo_sample_index}_questions.json", evaluator_records)
            locomo_sample_index += 1
        elif row["benchmark"] == "beam-100k":
            beam_predictions.extend({
                "question_id": record["question_id"],
                "answer": record["answer"],
            } for record in records)
    if has_beam:
        prediction_path = condition_root / "beam-100k" / "predictions.jsonl"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in beam_predictions),
            encoding="utf-8",
        )
    status_path = condition_root / "status.json"
    status = read_json(status_path) if status_path.is_file() else {}
    status.update({
        "phase": "complete",
        "condition": condition,
    })
    if has_locomo:
        status["locomo_samples"] = locomo_sample_index
    if has_beam:
        status["beam_questions"] = len(beam_predictions)
    atomic_json(status_path, status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--provider-name", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--read-context", type=int, default=8)
    parser.add_argument("--question-retries", type=int, default=3)
    parser.add_argument("--benchmarks", default="locomo,beam-100k")
    parser.add_argument("--unit-ids", default="")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--exclude-recent", action="store_true")
    args = parser.parse_args()
    if (
        args.workers < 1
        or args.max_tokens < 1
        or args.read_context < 0
        or args.question_retries < 1
    ):
        parser.error("workers/max-tokens must be positive and read-context non-negative")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        parser.error(f"environment variable {args.api_key_env} is not set")
    conditions = tuple(filter(None, args.conditions.split(",")))
    if not conditions or any(condition not in CONDITIONS for condition in conditions):
        parser.error(f"conditions must be drawn from {CONDITIONS}")
    benchmarks = set(filter(None, args.benchmarks.split(",")))
    unit_ids = set(filter(None, args.unit_ids.split(",")))
    rows = selected_rows(args.matrix.resolve(), benchmarks, unit_ids)
    build_root = args.build_root.resolve()
    output_root = args.output_root.resolve()
    for row in rows:
        validate_build(build_dir(build_root, row), row)

    backend = reanswer.configure(
        args.base_url,
        args.workers,
        model=args.model,
        api_key=api_key,
        max_tokens=args.max_tokens,
        read_context=args.read_context,
    )
    jobs: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], Path, str]] = []
    per_unit: dict[tuple[str, str], tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]] = {}
    for condition in conditions:
        for row in rows:
            conversation, questions, _ = build_runner.load_unit(row)
            completed = load_completed(output_root, condition, row, questions)
            turn_index = backend.build_turn_index(conversation)
            memory = build_dir(build_root, row) / "memory"
            memory_hash = tree_sha256(memory)
            per_unit[(condition, row["run_id"])] = (questions, completed, row)
            write_unit_outputs(output_root, condition, row, questions, completed)
            for question in questions:
                if str(question["question_id"]) not in completed:
                    jobs.append((condition, row, question, turn_index, memory, memory_hash))

    print(f"resumed={sum(len(value[1]) for value in per_unit.values())} pending={len(jobs)} workers={args.workers}", flush=True)

    def work(job: tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], Path, str]) -> dict[str, Any]:
        condition, row, question, turn_index, memory, memory_hash = job
        visible_question = str(question.get("question_text", question.get("question", "")))
        item = dict(question)
        item["question"] = visible_question
        memories, steps, answer, trace = reanswer.collect_answer_v11(
            backend,
            item,
            memory,
            turn_index,
            condition=condition,
            include_recent=not args.exclude_recent,
        )
        unchanged = tree_sha256(memory) == memory_hash
        if not unchanged:
            raise RuntimeError(f"memory changed during QA: {memory}")
        return {
            "condition": condition,
            "benchmark": row["benchmark"],
            "run_id": row["run_id"],
            "unit_id": row["unit_id"],
            "selection": row["selection"],
            "question_id": question["question_id"],
            "question": visible_question,
            "gold": question.get("gold", ""),
            "category": question.get("category"),
            "question_type": question.get("question_type"),
            "gold_field": question.get("gold_field"),
            "rubric": list(question.get("rubric_nuggets", [])),
            "answer": answer,
            "retrieval_steps": steps,
            "memories": memories,
            "tool_trace": trace,
            "memory_sha256": memory_hash,
            "memory_unchanged": True,
            "recent_enabled": not args.exclude_recent,
            "model": args.model,
            "provider": args.provider_name,
            "base_url": args.base_url,
        }

    interrupted = False
    pool = ThreadPoolExecutor(max_workers=min(args.workers, len(jobs) or 1))
    active: dict[Any, tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], Path, str]] = {}
    pending = deque(jobs)
    attempts: dict[tuple[str, str, str], int] = {}
    try:
        for _ in range(min(args.workers, len(jobs))):
            if pending:
                job = pending.popleft()
                active[pool.submit(work, job)] = job
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                condition, row, question, *_ = job
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001
                    key = (condition, row["run_id"], str(question["question_id"]))
                    count = attempts.get(key, 0) + 1
                    attempts[key] = count
                    if count >= args.question_retries:
                        raise RuntimeError(
                            f"question failed after {count} attempts: {key}"
                        ) from exc
                    pending.append(job)
                    print(
                        f"retry={count}/{args.question_retries} {condition} "
                        f"{row['unit_id']} {question['question_id']} "
                        f"error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                else:
                    atomic_json(item_path(output_root, condition, row, question), record)
                    questions, completed, _ = per_unit[(condition, row["run_id"])]
                    completed[str(question["question_id"])] = record
                    write_unit_outputs(output_root, condition, row, questions, completed)
                    total_done = sum(len(value[1]) for value in per_unit.values())
                    remaining = len(active) + len(pending)
                    print(f"complete={total_done} remaining={remaining} {condition} {row['unit_id']} {question['question_id']}", flush=True)
                if pending:
                    next_job = pending.popleft()
                    active[pool.submit(work, next_job)] = next_job
    except KeyboardInterrupt:
        interrupted = True
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if interrupted:
        return 130
    for condition in conditions:
        publish_condition_outputs(output_root, condition, rows)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, reanswer.stop_on_signal)
    raise SystemExit(main())
