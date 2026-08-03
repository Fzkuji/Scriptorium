#!/usr/bin/env python3
"""Calibrate Writer input capacity on one fixed synthetic workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.nativemem.common import atomic_json
from src import management, retrieval
from src.management.api import render_writer_input, writer_protocol_sha256
from src.runtime.capacity import (
    SCHEMA,
    MessageTooLargeError,
    pack_complete_messages,
    select_writer_capacities,
)
from src.runtime.tokenization import TokenCounter


def make_probe_workload(
    *,
    probe_id: str,
    session_count: int,
    messages_per_session: int,
    facts_per_session: int = 1,
    context_sentences: int = 2,
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    if session_count < 1 or messages_per_session < 1 or context_sentences < 1:
        raise ValueError("probe workload sizes must be positive")
    if not 1 <= facts_per_session <= messages_per_session:
        raise ValueError(
            "facts_per_session must be between 1 and messages_per_session"
        )
    sessions = []
    expected_refs: set[str] = set()
    expected_facts: set[str] = set()
    for session_number in range(1, session_count + 1):
        turns = []
        refs = []
        fact_positions = {
            ((index + 1) * messages_per_session) // (facts_per_session + 1)
            for index in range(facts_per_session)
        }
        for message_number in range(1, messages_per_session + 1):
            ref = (
                f"calibration/{probe_id}-s{session_number:03d}/"
                f"msg-{message_number:03d}"
            )
            context = " ".join(
                f"Conversation context {number} for session "
                f"{session_number:03d}, message {message_number:03d} continues "
                "the current discussion without introducing another durable record."
                for number in range(1, context_sentences + 1)
            )
            if message_number - 1 in fact_positions:
                code = "ARCHIVE-" + hashlib.sha256(
                    f"{probe_id}:{session_number}:{message_number}".encode()
                ).hexdigest()[:12].upper()
                text = (
                    f"Durable calibration fact: the assigned archive label is {code}. "
                    f"Retain this fact for later exact recall. {context}"
                )
                expected_refs.add(ref)
                expected_facts.add(code)
            else:
                text = (
                    f"{context} The speaker acknowledges the preceding turn."
                )
            turns.append((
                "user" if message_number % 2 else "assistant",
                text,
            ))
            refs.append(ref)
        sessions.append({
            "observation_date": f"2026-01-{((session_number - 1) % 28) + 1:02d}",
            "turns": turns,
            "refs": refs,
        })
    return sessions, expected_refs, expected_facts


def evaluate_probe(
    memory_dir: Path,
    *,
    audit: list[dict[str, Any]],
    expected_refs: set[str],
    expected_facts: set[str],
) -> dict[str, Any]:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (memory_dir / "topics", memory_dir)
        for path in (
            sorted(root.rglob("*.md"))
            if root.name == "topics"
            else [root / "core.md"]
        )
        if path.is_file()
    )
    covered = sum(ref in text for ref in expected_refs)
    coverage = covered / len(expected_refs) if expected_refs else 0.0
    facts_covered = sum(fact in text for fact in expected_facts)
    fact_coverage = (
        facts_covered / len(expected_facts) if expected_facts else 0.0
    )
    written_blocks = sum(
        int(row.get("count", 0))
        for row in audit
        if row.get("status") == "ok"
    )
    round_limit_reached = any(
        row.get("status") == "stopped" for row in audit
    )
    return {
        "passed": (
            coverage == 1.0
            and fact_coverage == 1.0
            and written_blocks > 0
        ),
        "source_coverage": round(coverage, 6),
        "fact_coverage": round(fact_coverage, 6),
        "written_blocks": written_blocks,
        "audit_error_count": sum(
            row.get("status") == "error" for row in audit
        ),
        "round_limit_reached": round_limit_reached,
    }


def _usage(result: Any) -> tuple[int, int, int, float]:
    return (
        int(getattr(result, "input_tokens", 0) or 0),
        int(getattr(result, "output_tokens", 0) or 0),
        int(getattr(result, "num_turns", 0) or 0),
        float(getattr(result, "anthropic_equivalent_cost_usd", 0) or 0),
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


def _rounded_mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def _rounded_std(values: list[float]) -> float | None:
    return round(statistics.pstdev(values), 6) if values else None


def summarize_levels(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repeated probe outcomes for each candidate input length."""
    summaries = []
    for candidate in sorted({int(row["candidate_tokens"]) for row in records}):
        rows = [
            row for row in records
            if int(row["candidate_tokens"]) == candidate
        ]
        trials = [row for row in rows if not bool(row.get("skipped"))]
        coverage = [
            float(row["source_coverage"])
            for row in trials
            if isinstance(row.get("source_coverage"), (int, float))
        ]
        fact_coverage = [
            float(row["fact_coverage"])
            for row in trials
            if isinstance(row.get("fact_coverage"), (int, float))
        ]
        batch_counts = [
            float(row["batch_count"])
            for row in trials
            if isinstance(row.get("batch_count"), (int, float))
        ]
        max_batch_tokens = [
            max(int(value) for value in row["batch_input_tokens"])
            for row in trials
            if row.get("batch_input_tokens")
        ]
        first_prompt_tokens = [
            int(row["provider_first_prompt_tokens"])
            for row in trials
            if isinstance(row.get("provider_first_prompt_tokens"), int)
        ]
        max_prompt_tokens = [
            int(row["provider_max_prompt_tokens"])
            for row in trials
            if isinstance(row.get("provider_max_prompt_tokens"), int)
        ]
        elapsed = [float(row["elapsed_seconds"]) for row in trials]
        costs = [float(row["estimated_cost_usd"]) for row in trials]
        summaries.append({
            "candidate_tokens": candidate,
            "trials": len(trials),
            "passed_trials": sum(bool(row.get("passed")) for row in trials),
            "pass_rate": (
                round(sum(bool(row.get("passed")) for row in trials) / len(trials), 6)
                if trials else None
            ),
            "skipped_trials": len(rows) - len(trials),
            "error_trials": sum("error" in row for row in trials),
            "tool_errors_total": sum(
                int(row.get("audit_error_count", 0)) for row in trials
            ),
            "round_limit_trials": sum(
                bool(row.get("round_limit_reached")) for row in trials
            ),
            "completion_rate": (
                round(sum(
                    not bool(row.get("round_limit_reached")) for row in trials
                ) / len(trials), 6)
                if trials else None
            ),
            "source_coverage_mean": _rounded_mean(coverage),
            "source_coverage_min": round(min(coverage), 6) if coverage else None,
            "source_coverage_std": _rounded_std(coverage),
            "fact_coverage_mean": _rounded_mean(fact_coverage),
            "fact_coverage_min": (
                round(min(fact_coverage), 6) if fact_coverage else None
            ),
            "fact_coverage_std": _rounded_std(fact_coverage),
            "workload_session_count": next(iter({
                int(row["workload_session_count"])
                for row in trials if "workload_session_count" in row
            }), None),
            "workload_message_count": next(iter({
                int(row["workload_message_count"])
                for row in trials if "workload_message_count" in row
            }), None),
            "workload_fact_count": next(iter({
                int(row["workload_fact_count"])
                for row in trials if "workload_fact_count" in row
            }), None),
            "workload_input_tokens": next(iter({
                int(row["workload_input_tokens"])
                for row in trials if "workload_input_tokens" in row
            }), None),
            "batch_count_mean": _rounded_mean(batch_counts),
            "batch_count_std": _rounded_std(batch_counts),
            "max_batch_input_tokens_min": (
                min(max_batch_tokens) if max_batch_tokens else None
            ),
            "max_batch_input_tokens_max": (
                max(max_batch_tokens) if max_batch_tokens else None
            ),
            "batched_input_tokens_total": sum(
                int(row.get("batched_input_tokens", 0)) for row in rows
            ),
            "provider_first_prompt_tokens_mean": _rounded_mean([
                float(value) for value in first_prompt_tokens
            ]),
            "provider_first_prompt_tokens_min": (
                min(first_prompt_tokens) if first_prompt_tokens else None
            ),
            "provider_first_prompt_tokens_max": (
                max(first_prompt_tokens) if first_prompt_tokens else None
            ),
            "provider_max_prompt_tokens_mean": _rounded_mean([
                float(value) for value in max_prompt_tokens
            ]),
            "provider_max_prompt_tokens_max": (
                max(max_prompt_tokens) if max_prompt_tokens else None
            ),
            "calls_total": sum(int(row["calls"]) for row in rows),
            "calls_mean": _rounded_mean([
                float(row["calls"]) for row in trials
            ]),
            "prompt_tokens_total": sum(int(row["prompt_tokens"]) for row in rows),
            "completion_tokens_total": sum(
                int(row["completion_tokens"]) for row in rows
            ),
            "elapsed_seconds_total": round(sum(
                float(row["elapsed_seconds"]) for row in rows
            ), 6),
            "elapsed_seconds_mean": _rounded_mean(elapsed),
            "elapsed_seconds_std": _rounded_std(elapsed),
            "estimated_cost_usd_total": round(sum(
                float(row["estimated_cost_usd"]) for row in rows
            ), 6),
            "estimated_cost_usd_mean": _rounded_mean(costs),
        })
    return summaries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results/model_capacity"))
    parser.add_argument("--run-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    required = ("provider", "base_url", "model", "api_key_file")
    missing = [key for key in required if not str(config.get(key, "")).strip()]
    if missing:
        raise ValueError(f"calibration config missing: {', '.join(missing)}")
    api_key = Path(config["api_key_file"]).expanduser().read_text().strip()
    if not api_key:
        raise ValueError("api_key_file is empty")
    model = str(config["model"])
    counter = TokenCounter.resolve(requested_model=model)
    candidates = sorted({int(value) for value in config.get(
        "candidate_tokens", [4096, 8192, 12288, 16384, 20480, 24576]
    )})
    probe_ids = tuple(config.get("probe_ids", ["facts-a", "facts-b"]))
    workload_session_count = int(config.get("workload_sessions", 20))
    messages_per_session = int(config.get("messages_per_session", 20))
    facts_per_session = int(config.get("facts_per_session", 1))
    context_sentences = int(config.get("context_sentences", 2))
    if not candidates or any(value < 1 for value in candidates):
        raise ValueError("candidate_tokens must contain positive integers")
    if not probe_ids:
        raise ValueError("probe_ids must not be empty")
    workloads = {
        str(probe_id): make_probe_workload(
            probe_id=str(probe_id),
            session_count=workload_session_count,
            messages_per_session=messages_per_session,
            facts_per_session=facts_per_session,
            context_sentences=context_sentences,
        )
        for probe_id in probe_ids
    }
    runtime = retrieval.create_runtime(
        str(config["base_url"]),
        api_key=api_key,
        model=model,
        cli_path=config.get("claude_cli"),
    )
    memory_config = management.MemoryConfig(
        max_turns=int(config.get("max_turns", 20)),
        max_budget_usd=(
            float(config["max_budget_usd"])
            if config.get("max_budget_usd") is not None
            else None
        ),
    )
    input_price = float(config.get("input_usd_per_million", 0))
    output_price = float(config.get("output_usd_per_million", 0))
    if input_price < 0 or output_price < 0:
        raise ValueError("model prices must be non-negative")
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        for probe_id in probe_ids:
            usage: list[tuple[int, int, int, float]] = []
            started = time.monotonic()
            sessions, expected_refs, expected_facts = workloads[str(probe_id)]
            workload_json = json.dumps(
                sessions,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            record = {
                "candidate_tokens": candidate,
                "probe_id": probe_id,
                "workload_sha256": hashlib.sha256(
                    workload_json.encode("utf-8")
                ).hexdigest(),
                "workload_session_count": len(sessions),
                "workload_message_count": sum(
                    len(session["turns"]) for session in sessions
                ),
                "workload_fact_count": len(expected_facts),
                "workload_input_tokens": counter.count(
                    render_writer_input(sessions)
                ),
            }
            try:
                batches = pack_complete_messages(
                    sessions,
                    max_input_tokens=candidate,
                    render_batch=render_writer_input,
                    token_counter=counter,
                )
                batch_tokens = [
                    counter.count(render_writer_input(batch)) for batch in batches
                ]
                with tempfile.TemporaryDirectory(prefix="nativemem-capacity-") as directory:
                    audit = []
                    for batch in batches:
                        audit.extend(management.write_sessions(
                            directory,
                            agent=runtime.agent,
                            sessions=batch,
                            usage_logger=lambda result: usage.append(
                                _usage(result)
                            ),
                            config=memory_config,
                        ))
                    result = evaluate_probe(
                        Path(directory),
                        audit=audit,
                        expected_refs=expected_refs,
                        expected_facts=expected_facts,
                    )
                record.update({
                    "batch_count": len(batches),
                    "batch_message_counts": [
                        sum(len(session["turns"]) for session in batch)
                        for batch in batches
                    ],
                    "batch_input_tokens": batch_tokens,
                    "batched_input_tokens": sum(batch_tokens),
                    **result,
                })
            except MessageTooLargeError as exc:
                record.update({
                    "passed": False,
                    "skipped": True,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            except Exception as exc:  # noqa: BLE001
                record.update({
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            prompt_tokens = sum(value[0] for value in usage)
            completion_tokens = sum(value[1] for value in usage)
            record.update({
                "calls": sum(value[2] for value in usage),
                "provider_first_prompt_tokens": usage[0][0] if usage else None,
                "provider_max_prompt_tokens": (
                    max(value[0] for value in usage) if usage else None
                ),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "anthropic_equivalent_cost_usd": round(sum(
                    value[3] for value in usage
                ), 6),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "estimated_cost_usd": round(
                    (
                        prompt_tokens * input_price
                        + completion_tokens * output_price
                    ) / 1_000_000,
                    6,
                ),
            })
            records.append(record)
            print(json.dumps({
                key: record.get(key)
                for key in (
                    "candidate_tokens", "probe_id", "passed",
                    "batch_count", "source_coverage", "fact_coverage",
                    "calls", "elapsed_seconds", "estimated_cost_usd",
                )
            }, ensure_ascii=False), flush=True)
    levels = summarize_levels(records)
    try:
        selected = select_writer_capacities(levels)
        safe_input_tokens = selected["recommended_input_tokens"]
        max_tested_passing_input_tokens = selected[
            "max_tested_passing_input_tokens"
        ]
        status = "complete"
    except ValueError:
        safe_input_tokens = None
        max_tested_passing_input_tokens = None
        status = "failed"
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.results_root
        / f"{_safe_name(str(config['provider']))}--{_safe_name(model)}"
        / _safe_name(run_id)
    )
    artifact = {
        "schema": SCHEMA,
        "status": status,
        "provider": config["provider"],
        "base_url": config["base_url"],
        "model": model,
        "writer_protocol_sha256": writer_protocol_sha256(),
        "safe_input_tokens": safe_input_tokens,
        "recommended_input_tokens": safe_input_tokens,
        "max_tested_passing_input_tokens": max_tested_passing_input_tokens,
        "tokenizer": counter.identity,
        "candidate_tokens": candidates,
        "probe_ids": list(probe_ids),
        "workload_sessions": workload_session_count,
        "messages_per_session": messages_per_session,
        "facts_per_session": facts_per_session,
        "context_sentences": context_sentences,
        "levels": levels,
        "probes": records,
        "totals": {
            "calls": sum(int(row["calls"]) for row in records),
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in records),
            "completion_tokens": sum(
                int(row["completion_tokens"]) for row in records
            ),
            "elapsed_seconds": round(
                sum(float(row["elapsed_seconds"]) for row in records), 3
            ),
            "estimated_cost_usd": round(
                sum(float(row["estimated_cost_usd"]) for row in records), 6
            ),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output_dir / "calibration.json", artifact)
    print(output_dir / "calibration.json")
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
