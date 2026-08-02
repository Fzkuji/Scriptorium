"""Build, query, and evaluate a complete LoCoMo conversation."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any

from src import build as adapter
from src import management as memory
from src import retrieval

from scripts.nativemem.common import atomic_json, read_json, tree_sha256, utc_now
from .config import parse_args
from .data import load_sample, sample_inventory
from .evaluation import run_evaluator, verify_evaluator
from .metrics import latency_summary, memory_inventory, summarize_usage
from .query import answer_question
from .results import build_record, load_completed, write_questions


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    verify_evaluator()
    data_path = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = output_dir / "memory"
    build_path = output_dir / "build.json"
    call_log_path = output_dir / "call_log.json"
    status_path = output_dir / "status.json"

    sample_index, sample = load_sample(data_path, args.sample_id)
    questions_path = output_dir / f"sample{sample_index}_questions.json"
    inventory = sample_inventory(sample)
    memory_config = memory.MemoryConfig(
        core_max_tokens=args.core_max_tokens,
        recent_limit=args.recent_limit,
        agent_max_rounds=args.agent_max_rounds,
        manager_max_rounds=args.manager_max_rounds,
        thinking=args.thinking,
        retry_log=args.retry_log,
    )
    build_config = adapter.BuildConfig(
        session_batch=args.session_batch,
        calibration_path=(
            str(args.writer_calibration.expanduser().resolve())
            if args.writer_calibration else None
        ),
        local_reorg_every_sessions=args.local_reorg_every_sessions,
        verify_writes=args.verify_writes,
        verify_every_sessions=args.verify_every_sessions,
        final_manage=args.final_manage,
        memory_config=memory_config,
    )
    query_config = retrieval.QueryConfig(
        max_rounds=args.retrieval_max_rounds,
        max_tool_calls=args.retrieval_max_tool_calls,
        visible_token_limit=args.memory_visible_tokens,
        max_output_tokens=args.answer_max_tokens,
        verify_sources=args.verify_sources,
    )
    backend = retrieval.create_runtime(
        args.base_url,
        api_format=args.api_format,
        model=args.model,
        api_key=args.api_key,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout_seconds,
        build_config=build_config,
        query_config=query_config,
    )
    if call_log_path.is_file():
        backend.call_log.extend(read_json(call_log_path))

    atomic_json(status_path, {
        "phase": "building",
        "sample": inventory,
        "started_at": utc_now(),
    })
    if build_path.is_file():
        build = read_json(build_path)
        if (
            build.get("status") != "complete"
            or build.get("sample_id") != args.sample_id
            or not memory_dir.is_dir()
            or tree_sha256(memory_dir) != build.get("memory_sha256")
        ):
            raise RuntimeError("existing NativeMem build is incomplete or differs")
    else:
        if memory_dir.exists() and any(memory_dir.iterdir()):
            raise RuntimeError("memory exists without a complete build record")
        started = time.monotonic()
        _, event_count = backend.build_memory(
            sample["conversation"], str(memory_dir)
        )
        build = build_record(
            sample_index=sample_index,
            sample=sample,
            memory_dir=memory_dir,
            model=args.model,
            build_config=build_config,
            started=started,
            event_count=event_count,
            call_log=backend.call_log,
        )
        atomic_json(build_path, build)
        atomic_json(call_log_path, backend.call_log)

    completed = load_completed(questions_path, sample_index)
    write_questions(questions_path, build, completed)
    turn_index = backend.build_turn_index(sample["conversation"])
    memory_hash = tree_sha256(memory_dir)
    query_started = time.monotonic()
    failures: list[dict[str, Any]] = []
    pending = [
        (index, question)
        for index, question in enumerate(sample["qa"])
        if index not in completed
    ]
    atomic_json(status_path, {
        "phase": "querying",
        "sample": inventory,
        "completed": len(completed),
        "pending": len(pending),
        "workers": args.workers,
        "updated_at": utc_now(),
    })
    with ThreadPoolExecutor(
        max_workers=min(args.workers, max(1, len(pending)))
    ) as pool:
        futures = {
            pool.submit(
                answer_question,
                backend,
                query_config,
                memory_dir,
                turn_index,
                sample_index,
                args.sample_id,
                index,
                question,
            ): index
            for index, question in pending
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                record = future.result()
                if not record["answer"]:
                    raise RuntimeError("empty answer")
                completed[index] = record
                write_questions(questions_path, build, completed)
                atomic_json(call_log_path, backend.call_log)
            except Exception as exc:  # noqa: BLE001
                failures.append({
                    "question_index": index,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            atomic_json(status_path, {
                "phase": "querying",
                "sample": inventory,
                "completed": len(completed),
                "pending": len(sample["qa"]) - len(completed),
                "failed": failures,
                "workers": args.workers,
                "updated_at": utc_now(),
            })

    if failures or len(completed) != len(sample["qa"]):
        atomic_json(status_path, {
            "phase": "failed",
            "completed": len(completed),
            "total": len(sample["qa"]),
            "failed": failures,
            "finished_at": utc_now(),
        })
        raise RuntimeError(f"{len(failures)} LoCoMo questions failed")
    if tree_sha256(memory_dir) != memory_hash:
        raise RuntimeError("query phase modified the memory workspace")

    performance = {
        "sample": inventory,
        "model": args.model,
        "build": build,
        "query_wall_time_s": round(time.monotonic() - query_started, 3),
        "query_latency": latency_summary(completed),
        "retrieval_steps": sum(
            record["retrieval"]["steps"] for record in completed.values()
        ),
        "tool_calls": sum(
            record["retrieval"]["tool_calls"] for record in completed.values()
        ),
        "memory_visible_tokens": sum(
            record["retrieval"]["visible_tokens"]
            for record in completed.values()
        ),
        "usage": summarize_usage(
            backend.call_log,
            input_usd_per_million=args.input_usd_per_million,
            output_usd_per_million=args.output_usd_per_million,
        ),
        "memory": memory_inventory(memory_dir),
        "config": {"build": asdict(build_config), "query": asdict(query_config)},
        "finished_at": utc_now(),
    }
    atomic_json(output_dir / "performance.json", performance)
    atomic_json(call_log_path, backend.call_log)
    if args.evaluate:
        atomic_json(status_path, {"phase": "evaluating", "updated_at": utc_now()})
        run_evaluator(args, output_dir)
    atomic_json(status_path, {
        "phase": "complete",
        "completed": len(completed),
        "total": len(sample["qa"]),
        "memory_dir": str(memory_dir),
        "finished_at": utc_now(),
    })
    print(json.dumps({
        "sample": inventory,
        "memory_dir": str(memory_dir),
        "performance": str(output_dir / "performance.json"),
        "evaluation": str(output_dir / "eval_full.json")
        if args.evaluate else None,
    }, ensure_ascii=False))
    return 0
