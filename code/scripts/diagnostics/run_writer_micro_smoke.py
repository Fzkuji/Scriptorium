#!/usr/bin/env python3
"""Build one complete LongMemEval session through the real Writer path."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.runners.longmemeval import support  # noqa: E402
from src import build, management  # noqa: E402
from src.agent_runtime import ClaudeCodeAgent, ClaudeCodeConfig  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--api-key-file", type=Path, required=True)
    result.add_argument("--base-url", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--item-index", type=int, default=75)
    result.add_argument("--sessions", type=int, default=1)
    result.add_argument("--writer-input-token-cap", type=int)
    result.add_argument("--max-turns", type=int, default=30)
    result.add_argument(
        "--shell-backend",
        choices=("auto", "posix-bash", "native"),
        default="posix-bash",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    if args.sessions < 1:
        raise ValueError("sessions must be positive")
    if (
        args.writer_input_token_cap is not None
        and args.writer_input_token_cap < 1
    ):
        raise ValueError("writer input token cap must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite micro smoke: {output_dir}")
    output_dir.mkdir(parents=True)
    result_path = output_dir / "micro-result.json"
    checkpoint_path = output_dir / "build-checkpoint.json"
    memory_dir = output_dir / "memory"

    key_path = args.api_key_file.expanduser().resolve()
    api_key = key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError("API key file is empty")

    data_path = args.data.expanduser().resolve()
    dataset = support.load_dataset(
        data_path, expected_count=support.EXPECTED_LONGMEMEVAL_SIZE
    )
    support.validate_longmemeval_s(dataset)
    item = dataset[args.item_index]
    conversation = support.to_conversation(item, args.item_index)
    selected_sessions = [
        conversation[f"session_{index}"]
        for index in range(1, args.sessions + 1)
        if f"session_{index}" in conversation
    ]
    if len(selected_sessions) != args.sessions:
        raise ValueError(
            f"item has fewer than {args.sessions} requested sessions"
        )

    agent = ClaudeCodeAgent(ClaudeCodeConfig(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
    ))
    config = build.BuildConfig(
        session_batch=1,
        writer_input_token_cap=args.writer_input_token_cap,
        local_reorg_every_sessions=0,
        verify_writes=False,
        verify_every_sessions=1,
        final_manage=False,
        memory_config=management.MemoryConfig(
            max_turns=args.max_turns,
            shell_backend=args.shell_backend,
        ),
    )
    usage: list[dict[str, object]] = []

    def record_usage(agent_result: object) -> None:
        usage.append({
            "num_turns": int(getattr(agent_result, "num_turns", 0) or 0),
            "input_tokens": int(getattr(agent_result, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(agent_result, "output_tokens", 0) or 0),
            "duration_ms": int(getattr(agent_result, "duration_ms", 0) or 0),
            "stop_reason": getattr(agent_result, "stop_reason", None),
        })

    metadata = {
        "schema": "scriptorium-writer-micro-smoke-v1",
        "status": "running",
        "dataset_index": args.item_index,
        "question_id": str(item["question_id"]),
        "question_type": item["question_type"],
        "sessions": args.sessions,
        "source_turns": sum(len(session) for session in selected_sessions),
        "model": args.model,
        "base_url": args.base_url,
        "shell_backend": args.shell_backend,
        "max_turns": args.max_turns,
        "verification": False,
        "final_management": False,
        "runner_pid": os.getpid(),
        "started_at": support.utc_now(),
    }
    support.atomic_json(result_path, metadata)
    started = time.monotonic()
    try:
        _, events = build.build_memory(
            conversation,
            memory_dir,
            max_sessions=args.sessions,
            agent=agent,
            model=args.model,
            usage_logger=record_usage,
            config=config,
            checkpoint_path=checkpoint_path,
        )
        stats = support.memory_stats(memory_dir)
        if int(events) <= 0:
            raise RuntimeError("Writer completed without creating memory blocks")
        metadata.update({
            "status": "complete",
            "finished_at": support.utc_now(),
            "wall_time_s": round(time.monotonic() - started, 3),
            "events": int(events),
            "usage": usage,
            "memory": stats,
        })
        support.atomic_json(result_path, metadata)
        return 0
    except Exception as exc:
        safe_message = str(exc).replace(api_key, "[redacted]")
        metadata.update({
            "status": "failed",
            "finished_at": support.utc_now(),
            "wall_time_s": round(time.monotonic() - started, 3),
            "usage": usage,
            "error": {"type": type(exc).__name__, "message": safe_message},
        })
        support.atomic_json(result_path, metadata)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
