#!/usr/bin/env python3
"""Run one resumable V11/GPT-5.5 LongMemEval-S shard via Frontier."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from itertools import zip_longest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_v88_gpt55_longmemeval as common  # noqa: E402


V11_ANSWER_PROMPT = """User question:
{question}

Current date:
{question_date}

The user's memory is stored in a read-only workspace.

The shell is already running at this workspace root:
{memory_root}

The paths shown below are relative to that root. You may use normal shell
commands to inspect and manage files inside this workspace. All available
memory, including source files, is contained here.

The workspace may contain:
- `topics/`: information organized by subject. Files and Markdown headings are
  created and organized by the memory agent, so there is no fixed taxonomy.
- `timeline/`: information organized chronologically.
- `recent_events.jsonl`: a bounded list of recent memory events.
- Source references such as `[Dn:m]`: use `read_original` with a complete
  reference when the original conversation is needed.

Available files:
{structure}

Use the read-only tools to look up the information needed to answer the user.
Answer the user directly. Output exactly one <answer>...</answer> block.
"""


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, default=common.DEFAULT_DATA)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--start", type=int, required=True)
    result.add_argument("--limit", type=int, required=True)
    result.add_argument("--base-url", default="https://api.frontier-intelligence.tech/v1")
    result.add_argument("--model", default="gpt-5.5")
    result.add_argument("--provider-name", default="frontier-intelligence")
    result.add_argument(
        "--api-format", choices=("openai", "anthropic"), default="openai"
    )
    result.add_argument("--resume", action="store_true")
    order = result.add_mutually_exclusive_group()
    order.add_argument("--round-robin-types", action="store_true")
    order.add_argument("--question-type")
    result.add_argument("--reverse", action="store_true")
    result.add_argument("--claim-db", type=Path)
    return result


def round_robin_indices(data: list[dict]) -> list[int]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(data):
        groups.setdefault(item["question_type"], []).append(index)
    return [index for row in zip_longest(*groups.values()) for index in row if index is not None]


def question_type_indices(
    data: list[dict], question_type: str, start: int, limit: int, *, reverse: bool
) -> list[int]:
    matching = [
        index for index, item in enumerate(data)
        if item["question_type"] == question_type
    ]
    if not matching:
        raise common.DataValidationError(f"unknown --question-type: {question_type}")
    if reverse:
        matching.reverse()
    positions = common.select_indices(len(matching), start, limit)
    return [matching[position] for position in positions]


def initialize_queue(path: Path, indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS items ("
            "item INTEGER PRIMARY KEY, state INTEGER NOT NULL, owner TEXT)"
        )
        db.executemany(
            "INSERT OR IGNORE INTO items(item, state) VALUES (?, 0)",
            ((index,) for index in indices),
        )


def claim_item(path: Path, index: int, owner: str) -> bool:
    with sqlite3.connect(path, timeout=30, isolation_level=None) as db:
        db.execute("BEGIN IMMEDIATE")
        changed = db.execute(
            "UPDATE items SET state=1, owner=? WHERE item=? AND state=0",
            (owner, index),
        ).rowcount
        db.commit()
    return changed == 1


def finish_claim(path: Path, index: int) -> None:
    with sqlite3.connect(path, timeout=30) as db:
        db.execute("UPDATE items SET state=2 WHERE item=?", (index,))


def queue_states(path: Path) -> dict[int, int]:
    with sqlite3.connect(path, timeout=30) as db:
        return dict(db.execute("SELECT item, state FROM items ORDER BY item"))


def main() -> int:
    args = parser().parse_args()
    api_key = os.environ.get("FRONTIER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("FRONTIER_API_KEY is required")

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

    common.METHOD_ENV = {
        "NATIVEMEM_PROMPT": "v11",
        "NATIVEMEM_STORE_MODE": "oneshot",
        "NATIVEMEM_V8_SINGLE": "1",
        "NATIVEMEM_V8_MAX_ROUNDS": "20",
        "NATIVEMEM_V8_MAX_TOKENS": "16384",
        "NATIVEMEM_V11_SESSION_BATCH": os.environ.get(
            "NATIVEMEM_V11_SESSION_BATCH", "1"
        ),
    }
    common.LME_SINGLE_PROMPT = V11_ANSWER_PROMPT
    args.api_key = api_key
    args.request_concurrency = 1
    args.trust_proxy = False
    config = common.configure_environment(args)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,api.frontier-intelligence.tech"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    backend = common.load_backend()
    if args.api_format == "anthropic":
        from src.anthropic_openai_compat import AnthropicOpenAICompat

        anthropic_client = AnthropicOpenAICompat(args.api_key, args.base_url)
        backend.client = anthropic_client
        backend.nativemem_runtime.client = anthropic_client
        backend.v8_memory.client = anthropic_client
    run_meta = {
        "method": {
            "name": "NativeMem",
            "version": "v11",
            "single_model_retrieve_answer": True,
        },
        "models": {
            "builder": args.model,
            "retriever": args.model,
            "answerer": args.model,
            "provider": args.provider_name,
            "base_url": args.base_url,
        },
        "config": config,
        "code": {
            "git_commit": common.git_head(),
            "runner_sha256": common.sha256_file(Path(__file__)),
            "v11_memory_sha256": common.sha256_file(ROOT / "src" / "v11_memory.py"),
            "adapter_sha256": common.sha256_file(
                ROOT / "src" / "adapters" / "run_nativemem.py"
            ),
        },
        "request_audit": {"mode": "direct_frontier_api"},
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
                index, item, output_dir, backend, run_meta, args.resume
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
    manifest["status"] = "failed" if failures else "complete"
    common.refresh_outputs(output_dir, manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
