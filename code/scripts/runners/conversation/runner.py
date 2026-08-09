"""Build, query, and evaluate one complete conversation."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any

from src import build as adapter
from src import management as memory
from src import retrieval

from scripts.runners.common import atomic_json, read_json, tree_sha256, utc_now
from .config import parse_args
from .data import check_benchmark, load_sample, sample_inventory
from .evaluation import run_evaluator, verify_evaluator
from src.runtime.billing import read_spend, spend_delta

from .metrics import latency_summary, memory_inventory, summarize_usage
from .query import answer_question
from .results import build_record, load_completed, write_questions


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # A build-only run never invokes a judge, so an unavailable or drifted
    # evaluator must not block memory construction. Full scored runs still
    # fail before the first provider request, and run_evaluator() verifies the
    # lock again immediately before scoring.
    if getattr(args, "evaluate", True) and not getattr(args, "build_only", False):
        verify_evaluator(args.benchmark)
    data_path = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = output_dir / "memory"
    build_path = output_dir / "build.json"
    checkpoint_path = output_dir / "build-checkpoint.json"
    call_log_path = output_dir / "call_log.json"
    status_path = output_dir / "status.json"

    sample_index, sample = load_sample(data_path, args.sample_id)
    check_benchmark(sample, args.benchmark)
    questions_path = output_dir / f"sample{sample_index}_questions.json"
    inventory = sample_inventory(sample)
    memory_config = memory.MemoryConfig(
        core_max_tokens=args.core_max_tokens,
        core_repair_target_tokens=getattr(args, "core_repair_target_tokens", 2_700),
        core_repair_max_checks=getattr(args, "core_repair_max_checks", 8),
        core_repair_max_trajectories=getattr(
            args, "core_repair_max_trajectories", 2
        ),
        core_repair_stagnation_limit=getattr(
            args, "core_repair_stagnation_limit", 2
        ),
        recent_limit=args.recent_limit,
        max_turns=args.max_turns,
        max_budget_usd=args.max_budget_usd,
        shell_backend=args.shell_backend,
    )
    build_config = adapter.BuildConfig(
        session_batch=args.session_batch,
        calibration_path=(
            str(args.writer_calibration.expanduser().resolve())
            if args.writer_calibration else None
        ),
        writer_input_token_cap=args.writer_input_token_cap,
        local_reorg_every_sessions=args.local_reorg_every_sessions,
        verify_writes=args.verify_writes,
        verify_every_sessions=args.verify_every_sessions,
        final_manage=args.final_manage,
        memory_config=memory_config,
    )
    query_config = retrieval.QueryConfig(
        max_turns=args.max_turns,
        max_budget_usd=args.max_budget_usd,
        verify_sources=args.verify_sources,
    )
    # Read the provider's spend counter before any request, so the run can
    # report what it actually cost rather than only a price-times-tokens guess.
    spend_before = read_spend(args.base_url, args.api_key)
    backend = retrieval.create_runtime(
        args.base_url,
        model=args.model,
        api_key=args.api_key,
        cli_path=args.claude_cli,
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
            raise RuntimeError("existing build is incomplete or differs")
    else:
        resume_build = checkpoint_path.is_file()
        if memory_dir.exists() and any(memory_dir.iterdir()) and not resume_build:
            raise RuntimeError("memory exists without a complete build record")
        started = time.monotonic()
        try:
            _, event_count = backend.build_memory(
                sample["conversation"],
                str(memory_dir),
                checkpoint_path=checkpoint_path,
                resume=resume_build,
            )
        except adapter.BuildPaused as exc:
            atomic_json(call_log_path, backend.call_log)
            atomic_json(status_path, {
                "phase": "paused",
                "stage": "building",
                "sample": inventory,
                "checkpoint": str(checkpoint_path),
                "reason": str(exc),
                "finished_at": utc_now(),
            })
            print(json.dumps({
                "sample": inventory,
                "checkpoint": str(checkpoint_path),
                "status": "paused",
            }, ensure_ascii=False))
            return 0
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for secret in (args.api_key, getattr(args, "judge_api_key", None)):
                if secret:
                    error = error.replace(secret, "[redacted]")
            atomic_json(call_log_path, backend.call_log)
            atomic_json(status_path, {
                "phase": "failed",
                "stage": "building",
                "sample": inventory,
                "error": error,
                "finished_at": utc_now(),
            })
            raise
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
        build["provider_estimated_cost_usd"] = round(
            build["input_tokens"] * args.input_usd_per_million / 1_000_000
            + build["output_tokens"] * args.output_usd_per_million / 1_000_000
            + build["cache_read_tokens"]
            * (
                getattr(args, "cache_read_usd_per_million", None)
                or args.input_usd_per_million
            )
            / 1_000_000
            + build["cache_write_tokens"]
            * (getattr(args, "cache_write_usd_per_million", None) or 0.0)
            / 1_000_000,
            8,
        )
        build["provider_spend"] = spend_delta(
            spend_before, read_spend(args.base_url, args.api_key)
        )
        atomic_json(build_path, build)
        atomic_json(call_log_path, backend.call_log)

    if args.build_only:
        atomic_json(status_path, {
            "phase": "complete",
            "stage": "building",
            "sample": inventory,
            "memory_dir": str(memory_dir),
            "build": str(build_path),
            "finished_at": utc_now(),
        })
        print(json.dumps({
            "sample": inventory,
            "memory_dir": str(memory_dir),
            "build": str(build_path),
        }, ensure_ascii=False))
        return 0

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
        raise RuntimeError(f"{len(failures)} questions failed")
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
            cache_read_usd_per_million=args.cache_read_usd_per_million,
            cache_write_usd_per_million=args.cache_write_usd_per_million,
        ),
        "memory": memory_inventory(memory_dir),
        # What the provider's own counter moved by, which does not depend on
        # the prices passed in or on how this gateway counts tokens.
        "provider_spend": spend_delta(
            spend_before, read_spend(args.base_url, args.api_key)
        ),
        "config": {"build": asdict(build_config), "query": asdict(query_config)},
        "finished_at": utc_now(),
    }
    atomic_json(output_dir / "performance.json", performance)
    atomic_json(call_log_path, backend.call_log)
    if args.evaluate:
        atomic_json(status_path, {"phase": "evaluating", "updated_at": utc_now()})
        try:
            run_evaluator(args, output_dir, questions_path)
        except Exception as exc:
            atomic_json(status_path, {
                "phase": "failed",
                "stage": "evaluating",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": utc_now(),
            })
            raise
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
