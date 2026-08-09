#!/usr/bin/env python3
"""Run one resumable NativeMem LongMemEval-S shard."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runners.common import run_config, source_tree_sha256  # noqa: E402
from scripts.runners.longmemeval import support as common  # noqa: E402
from scripts.runners.longmemeval.queue import (  # noqa: E402
    claim_item,
    finish_claim,
    initialize_queue,
    queue_states,
)
from scripts.runners.longmemeval.selection import (  # noqa: E402
    question_type_indices,
    round_robin_indices,
)
from memory import build as adapter  # noqa: E402
from memory import management as memory  # noqa: E402
from memory import retrieval  # noqa: E402


# Config values naming a file resolve against the config file's directory.
_PATH_KEYS = (
    "data", "output_dir", "writer_calibration", "claim_db", "claude_cli",
    "answer_claude_cli", "api_key_file", "answer_api_key_file",
)


def _read_private_key(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"API key file does not exist: {resolved}")
    if resolved.stat().st_mode & 0o077:
        raise ValueError(f"API key file permissions must be 600: {resolved}")
    value = resolved.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"API key file is empty: {resolved}")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    run_config.add_config_flag(result)
    result.add_argument("--data", type=Path, default=common.DEFAULT_DATA)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--start", type=int, required=True)
    result.add_argument("--limit", type=int, required=True)
    result.add_argument("--base-url", required=True)
    result.add_argument("--model", default="gpt-5.5")
    result.add_argument("--provider-name", default="frontier-intelligence")
    builder_key = result.add_mutually_exclusive_group(required=True)
    builder_key.add_argument("--api-key")
    builder_key.add_argument("--api-key-file", type=Path)
    result.add_argument("--claude-cli")
    result.add_argument("--answer-base-url")
    result.add_argument("--answer-model")
    answer_key = result.add_mutually_exclusive_group()
    answer_key.add_argument("--answer-api-key")
    answer_key.add_argument("--answer-api-key-file", type=Path)
    result.add_argument("--answer-provider-name")
    result.add_argument("--answer-claude-cli")
    result.add_argument("--max-turns", type=int, default=20)
    result.add_argument("--max-budget-usd", type=float)
    result.add_argument("--build-only", action="store_true")
    result.add_argument("--resume", action="store_true")
    order = result.add_mutually_exclusive_group()
    order.add_argument("--round-robin-types", action="store_true")
    order.add_argument("--question-type")
    result.add_argument("--reverse", action="store_true")
    result.add_argument("--claim-db", type=Path)
    result.add_argument("--session-batch", type=int, default=5)
    result.add_argument("--writer-calibration", type=Path)
    result.add_argument("--writer-input-token-cap", type=int)
    result.add_argument("--local-reorg-every-sessions", type=int, default=5)
    result.add_argument(
        "--verify-writes", action=argparse.BooleanOptionalAction, default=True
    )
    result.add_argument("--verify-every-sessions", type=int, default=5)
    result.add_argument(
        "--final-manage", action=argparse.BooleanOptionalAction, default=False
    )
    result.add_argument("--recent-limit", type=int, default=50)
    result.add_argument("--core-max-tokens", type=int, default=2_000)
    result.add_argument(
        "--verify-sources", action=argparse.BooleanOptionalAction, default=True
    )
    return result


def main() -> int:
    built = parser()
    try:
        run_config.apply(built, None, path_keys=_PATH_KEYS)
    except run_config.ConfigError as exc:
        built.error(str(exc))
    args = built.parse_args()
    try:
        builder_api_key = (
            _read_private_key(args.api_key_file)
            if args.api_key_file else args.api_key
        )
        answer_api_key = (
            _read_private_key(args.answer_api_key_file)
            if args.answer_api_key_file else args.answer_api_key
        )
    except ValueError as exc:
        built.error(str(exc))
    if args.writer_input_token_cap is not None and args.writer_input_token_cap < 1:
        raise SystemExit("--writer-input-token-cap must be positive")

    data_path = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    data = common.load_dataset(data_path, expected_count=common.EXPECTED_LONGMEMEVAL_SIZE)
    common.validate_longmemeval_s(data)
    if args.question_type:
        indices = question_type_indices(
            data, args.question_type, args.start, args.limit, reverse=args.reverse
        )
    elif args.round_robin_types:
        ordered = round_robin_indices(data)
        positions = common.select_indices(len(ordered), args.start, args.limit)
        indices = [ordered[position] for position in positions]
    else:
        indices = common.select_indices(len(data), args.start, args.limit)
    claim_db = args.claim_db.expanduser().resolve() if args.claim_db else None
    if claim_db:
        initialize_queue(claim_db, list(range(len(data))))

    memory_config = memory.MemoryConfig(
        core_max_tokens=args.core_max_tokens,
        recent_limit=args.recent_limit,
        max_turns=args.max_turns,
        max_budget_usd=args.max_budget_usd,
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
    common.LME_SINGLE_PROMPT = retrieval.ANSWER_PROMPT
    backend = retrieval.create_runtime(
        args.base_url,
        model=args.model,
        api_key=builder_api_key,
        cli_path=args.claude_cli,
        query_base_url=args.answer_base_url,
        query_model=args.answer_model,
        query_api_key=answer_api_key,
        query_cli_path=args.answer_claude_cli,
        build_config=build_config,
        query_config=query_config,
    )
    answer_model = args.answer_model or args.model
    answer_provider = args.answer_provider_name or args.provider_name
    answer_base_url = args.answer_base_url or args.base_url
    config = {
        "build": asdict(build_config),
        "query": asdict(query_config),
    }
    run_meta = {
        "method": {
            "name": "NativeMem",
            "implementation": "current",
            "single_model_retrieve_answer": answer_model == args.model,
        },
        "models": {
            "builder": args.model,
            "retriever": answer_model,
            "answerer": answer_model,
            "builder_provider": args.provider_name,
            "answerer_provider": answer_provider,
            "builder_base_url": args.base_url,
            "answerer_base_url": answer_base_url,
        },
        "config": config,
        "code": {
            "git_commit": common.git_head(),
            "runner_sha256": common.sha256_file(Path(__file__)),
            "management_sha256": source_tree_sha256(ROOT / "src" / "management"),
            "build_sha256": common.sha256_file(ROOT / "src" / "build.py"),
            "retrieval_sha256": source_tree_sha256(ROOT / "src" / "retrieval"),
        },
        "request_audit": {"mode": "claude_agent_sdk"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = common.create_or_resume_manifest(
        output_dir,
        data_path,
        common.sha256_file(data_path),
        len(data),
        run_meta,
        args.resume,
    )
    manifest.setdefault("invocations", []).append({
        "started_at": common.utc_now(),
        "start": args.start,
        "limit": args.limit,
        "resume": args.resume,
        "round_robin_types": args.round_robin_types,
        "question_type": args.question_type,
        "reverse": args.reverse,
        "claim_db": str(claim_db) if claim_db else None,
    })
    common.refresh_outputs(output_dir, manifest)

    failures: list[int] = []
    for position, index in enumerate(indices, start=1):
        item = data[index]
        if claim_db and not claim_item(claim_db, index, str(output_dir)):
            continue
        print(f"[{position}/{len(indices)}] item {index} {item['question_id']}", flush=True)
        try:
            ok, detail, _ = common.run_item(
                index,
                item,
                output_dir,
                backend,
                run_meta,
                args.resume,
                build_only=args.build_only,
            )
        except common.ExistingStateError as exc:
            ok, detail = False, str(exc)
        print(f"  {'complete' if ok else 'failed'}: {detail}", flush=True)
        if not ok:
            failures.append(index)
        if claim_db:
            finish_claim(claim_db, index)
        common.refresh_outputs(output_dir, manifest)

    manifest["last_invocation_finished_at"] = common.utc_now()
    manifest["last_invocation_failures"] = failures
    manifest["status"] = (
        "failed" if failures else "built" if args.build_only else "complete"
    )
    common.refresh_outputs(output_dir, manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
