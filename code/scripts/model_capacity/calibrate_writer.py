#!/usr/bin/env python3
"""Calibrate NativeMem Writer input capacity with complete synthetic sessions."""

from __future__ import annotations

import argparse
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
from src.runtime.capacity import SCHEMA, select_safe_capacity
from src.runtime.tokenization import TokenCounter


class CandidateTooSmallError(ValueError):
    """A candidate cannot contain the fixed Writer prompt and one session."""


def _probe_session(probe_id: str, index: int) -> dict[str, Any]:
    code = f"{probe_id.upper()}-{index:03d}"
    filler = " ".join(
        f"Context {number} describes the circumstances without changing the record."
        for number in range(1, 9)
    )
    return {
        "observation_date": f"2026-01-{(index % 28) + 1:02d}",
        "turns": [(
            "user",
            f"Remember calibration record {code}: the assigned archive label is "
            f"{code}. This is a durable fact for later exact recall. {filler}",
        )],
        "refs": [f"calibration/{probe_id}/msg-{index:03d}"],
    }


def make_probe_sessions(
    *,
    probe_id: str,
    candidate_tokens: int,
    token_counter: TokenCounter,
) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    for index in range(1, 1000):
        candidate = [*sessions, _probe_session(probe_id, index)]
        if token_counter.count(render_writer_input(candidate)) > candidate_tokens:
            break
        sessions = candidate
    if not sessions:
        raise CandidateTooSmallError(
            f"candidate {candidate_tokens} is smaller than one complete Writer input"
        )
    return sessions


def evaluate_probe(
    memory_dir: Path,
    *,
    audit: list[dict[str, Any]],
    expected_refs: set[str],
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
    written_blocks = sum(
        int(row.get("count", 0))
        for row in audit
        if row.get("status") == "ok"
    )
    return {
        "passed": coverage == 1.0 and written_blocks > 0,
        "source_coverage": round(coverage, 6),
        "written_blocks": written_blocks,
        "audit_error_count": sum(
            row.get("status") == "error" for row in audit
        ),
        "round_limit_reached": any(
            row.get("status") == "stopped" for row in audit
        ),
    }


def _usage(response: Any) -> tuple[int, int]:
    value = getattr(response, "usage", None)
    return (
        int(getattr(value, "prompt_tokens", 0) or 0),
        int(getattr(value, "completion_tokens", 0) or 0),
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
        session_counts = [
            float(row["session_count"])
            for row in trials
            if isinstance(row.get("session_count"), (int, float))
        ]
        rendered_tokens = [
            float(row["rendered_input_tokens"])
            for row in trials
            if isinstance(row.get("rendered_input_tokens"), (int, float))
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
            "source_coverage_mean": _rounded_mean(coverage),
            "source_coverage_min": round(min(coverage), 6) if coverage else None,
            "source_coverage_std": _rounded_std(coverage),
            "session_count_mean": _rounded_mean(session_counts),
            "rendered_input_tokens_mean": _rounded_mean(rendered_tokens),
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
        "candidate_tokens", [1024, 2048, 4096, 8192, 16384]
    )})
    probe_ids = tuple(config.get(
        "probe_ids", ["facts-a", "facts-b", "facts-c"]
    ))
    runtime = retrieval.create_runtime(
        str(config["base_url"]),
        api_key=api_key,
        api_format=str(config.get("api_format", "openai")),
        model=model,
        max_retries=int(config.get("max_retries", 2)),
        timeout_seconds=float(config.get("timeout_seconds", 180)),
    )
    memory_config = management.MemoryConfig(
        agent_max_rounds=int(config.get("agent_max_rounds", 12)),
        reasoning_effort=config.get("reasoning_effort"),
        thinking=config.get("thinking"),
        retry_log=bool(config.get("retry_log", False)),
    )
    input_price = float(config.get("input_usd_per_million", 0))
    output_price = float(config.get("output_usd_per_million", 0))
    if input_price < 0 or output_price < 0:
        raise ValueError("model prices must be non-negative")
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        for probe_id in probe_ids:
            usage: list[tuple[int, int]] = []
            started = time.monotonic()
            try:
                sessions = make_probe_sessions(
                    probe_id=str(probe_id),
                    candidate_tokens=candidate,
                    token_counter=counter,
                )
                with tempfile.TemporaryDirectory(prefix="nativemem-capacity-") as directory:
                    audit = management.write_sessions(
                        directory,
                        client=runtime.client,
                        model=model,
                        sessions=sessions,
                        usage_logger=lambda response: usage.append(_usage(response)),
                        config=memory_config,
                    )
                    result = evaluate_probe(
                        Path(directory),
                        audit=audit,
                        expected_refs={session["refs"][0] for session in sessions},
                    )
                record = {
                    "candidate_tokens": candidate,
                    "probe_id": probe_id,
                    "session_count": len(sessions),
                    "rendered_input_tokens": counter.count(
                        render_writer_input(sessions)
                    ),
                    **result,
                }
            except CandidateTooSmallError as exc:
                record = {
                    "candidate_tokens": candidate,
                    "probe_id": probe_id,
                    "passed": False,
                    "skipped": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            except Exception as exc:  # noqa: BLE001
                record = {
                    "candidate_tokens": candidate,
                    "probe_id": probe_id,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            prompt_tokens = sum(value[0] for value in usage)
            completion_tokens = sum(value[1] for value in usage)
            record.update({
                "calls": len(usage),
                "provider_first_prompt_tokens": usage[0][0] if usage else None,
                "provider_max_prompt_tokens": (
                    max(value[0] for value in usage) if usage else None
                ),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
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
                    "source_coverage", "calls", "elapsed_seconds",
                )
            }, ensure_ascii=False), flush=True)
    try:
        safe_input_tokens = select_safe_capacity(
            records, probe_ids={str(value) for value in probe_ids}
        )
        status = "complete"
    except ValueError:
        safe_input_tokens = None
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
        "tokenizer": counter.identity,
        "candidate_tokens": candidates,
        "probe_ids": list(probe_ids),
        "levels": summarize_levels(records),
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
