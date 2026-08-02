#!/usr/bin/env python3
"""Create and independently audit a minimal R004/G0.2 token-budget trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.visible_token_audit import audit_visible_token_trace  # noqa: E402
from scripts.evaluation.visible_token_budget import (  # noqa: E402
    TokenCounter,
    VisibleTokenBudgetGate,
    copy_snapshot,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: python3 scripts/run_visible_token_budget_sanity.py "
            "--output-dir /private/tmp/nativemem-r004-sanity-001 "
            "--budget-tokens 32 --model gpt-5.5 --allow-byte-fallback"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-tokens", type=int, default=32)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--fallback-encoding", default="o200k_base")
    parser.add_argument(
        "--allow-byte-fallback",
        action="store_true",
        help="Explicitly use deterministic UTF-8 bytes if tiktoken is unavailable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.budget_tokens <= 0:
        raise ValueError("sanity budget must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    tokenizer = TokenCounter.resolve(
        requested_model=args.model,
        fallback_encoding=args.fallback_encoding,
        allow_byte_fallback=args.allow_byte_fallback,
    )

    working_memory = args.output_dir / "memory-working"
    working_memory.mkdir()
    (working_memory / "state.txt").write_text("before\n", encoding="utf-8")
    memory_before_path = args.output_dir / "memory-before"
    memory_before = copy_snapshot(working_memory, memory_before_path)

    trace = args.output_dir / "visible-token-trace.jsonl"
    with VisibleTokenBudgetGate(
        trace_path=trace,
        run_id="r004-sanity",
        configured_budget_tokens=args.budget_tokens,
        tokenizer=tokenizer,
        memory_before=memory_before,
        overflow_policy="truncate",
        metadata={"scope": "sanity_only", "model_requested": args.model},
    ) as gate:
        first = gate.deliver_tool_result(
            event_id="tool-1",
            raw_text="alpha beta",
            tool_name="read_memory",
            tool_call_id="sanity-call-1",
        )
        source = gate.deliver_source_resolution(
            event_id="source-1",
            raw_text="source:s1",
            source_ids=["session-1:turn-2"],
        )
        overflow = gate.deliver_tool_result(
            event_id="tool-overflow",
            raw_text=("0123456789 " * (args.budget_tokens + 2)).strip(),
            tool_name="read_memory",
            tool_call_id="sanity-call-overflow",
        )
        rejected = gate.deliver_tool_result(
            event_id="tool-after-exhaustion",
            raw_text="this text must not cross the model boundary",
            tool_name="read_memory",
            tool_call_id="sanity-call-rejected",
        )

        (working_memory / "state.txt").write_text("after\n", encoding="utf-8")
        memory_after_path = args.output_dir / "memory-after"
        memory_after = copy_snapshot(working_memory, memory_after_path)
        manifest = gate.finalize(
            memory_after=memory_after,
            actual_model_usage={
                "logical_calls": 0,
                "physical_attempts": 0,
                "responses": [],
                "usage_unavailable_reason": "sanity runner performs no model calls",
            },
            metadata={"sanity_events": 4},
        )

    report = audit_visible_token_trace(
        trace,
        memory_before_path=memory_before_path,
        memory_after_path=memory_after_path,
    )
    audit_path = args.output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "audit_status": report["audit_status"],
        "trace": str(trace),
        "manifest": str(trace.with_suffix(trace.suffix + ".manifest.json")),
        "audit": str(audit_path),
        "tokenizer": tokenizer.identity,
        "decisions": [
            first.decision,
            source.decision,
            overflow.decision,
            rejected.decision,
        ],
        "configured_budget_tokens": args.budget_tokens,
        "cumulative_visible_tokens": manifest["summary"]["cumulative_visible_tokens"],
        "source_resolution_tokens": manifest["summary"][
            "cumulative_source_resolution_tokens"
        ],
        "memory_before_sha256": manifest["memory_before_sha256"],
        "memory_after_sha256": manifest["memory_after_sha256"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["audit_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
