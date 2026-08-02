"""Concurrent execution of paired NativeMem frozen-memory ablations."""

import argparse
import signal
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from scripts.nativemem.common import atomic_json, stop_on_signal, tree_sha256
from src import retrieval

from .outputs import (
    build_dir,
    item_path,
    load_unit,
    load_completed,
    publish_condition_outputs,
    selected_rows,
    validate_build,
    write_unit_outputs,
)


CONDITIONS = tuple(retrieval.CONDITION_VIEWS)
Job = tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], Path, str]


def main() -> int:
    signal.signal(signal.SIGTERM, stop_on_signal)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--provider-name", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--claude-cli")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-budget-usd", type=float)
    parser.add_argument("--benchmarks", default="locomo,beam-100k")
    parser.add_argument("--unit-ids", default="")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--exclude-recent", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.max_turns < 1:
        parser.error("workers and max-turns must be positive")
    if args.max_budget_usd is not None and args.max_budget_usd <= 0:
        parser.error("max-budget-usd must be positive")
    conditions = tuple(filter(None, args.conditions.split(",")))
    if not conditions or any(
        condition not in CONDITIONS for condition in conditions
    ):
        parser.error(f"conditions must be drawn from {CONDITIONS}")
    rows = selected_rows(
        args.matrix.resolve(),
        set(filter(None, args.benchmarks.split(","))),
        set(filter(None, args.unit_ids.split(","))),
    )
    build_root = args.build_root.resolve()
    output_root = args.output_root.resolve()
    for row in rows:
        validate_build(build_dir(build_root, row), row)

    query_config = retrieval.QueryConfig(
        max_turns=args.max_turns,
        max_budget_usd=args.max_budget_usd,
    )
    backend = retrieval.create_runtime(
        args.base_url,
        model=args.model,
        api_key=args.api_key,
        cli_path=args.claude_cli,
        query_config=query_config,
    )
    jobs: list[Job] = []
    per_unit: dict[
        tuple[str, str],
        tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]],
    ] = {}
    for condition in conditions:
        for row in rows:
            conversation, questions, _ = load_unit(row)
            completed = load_completed(
                output_root, condition, row, questions
            )
            turn_index = backend.build_turn_index(conversation)
            memory = build_dir(build_root, row) / "memory"
            memory_hash = tree_sha256(memory)
            per_unit[(condition, row["run_id"])] = (
                questions,
                completed,
                row,
            )
            write_unit_outputs(
                output_root, condition, row, questions, completed
            )
            jobs.extend(
                (condition, row, question, turn_index, memory, memory_hash)
                for question in questions
                if str(question["question_id"]) not in completed
            )
    print(
        f"resumed={sum(len(value[1]) for value in per_unit.values())} "
        f"pending={len(jobs)} workers={args.workers}",
        flush=True,
    )

    def work(job: Job) -> dict[str, Any]:
        condition, row, question, turn_index, memory, memory_hash = job
        visible_question = str(
            question.get("question_text", question.get("question", ""))
        )
        item = dict(question)
        item["question"] = visible_question
        memories, steps, answer, trace = retrieval.collect_answer(
            backend,
            item,
            memory,
            turn_index,
            condition=condition,
            include_recent=not args.exclude_recent,
            config=query_config,
        )
        if tree_sha256(memory) != memory_hash:
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

    pool = ThreadPoolExecutor(max_workers=min(args.workers, len(jobs) or 1))
    active: dict[Any, Job] = {}
    pending = deque(jobs)
    interrupted = False
    try:
        for _ in range(min(args.workers, len(jobs))):
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
                    raise RuntimeError(
                        "question failed: "
                        f"{condition}/{row['run_id']}/"
                        f"{question['question_id']}"
                    ) from exc
                else:
                    atomic_json(
                        item_path(output_root, condition, row, question),
                        record,
                    )
                    questions, completed, _ = per_unit[
                        (condition, row["run_id"])
                    ]
                    completed[str(question["question_id"])] = record
                    write_unit_outputs(
                        output_root, condition, row, questions, completed
                    )
                    total_done = sum(
                        len(value[1]) for value in per_unit.values()
                    )
                    print(
                        f"complete={total_done} "
                        f"remaining={len(active) + len(pending)} "
                        f"{condition} {row['unit_id']} "
                        f"{question['question_id']}",
                        flush=True,
                    )
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
