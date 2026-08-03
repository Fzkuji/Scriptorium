"""Usage, latency, and memory-size summaries for LoCoMo runs."""

import statistics
from pathlib import Path
from typing import Any


def summarize_usage(
    records: list[dict[str, Any]],
    *,
    input_usd_per_million: float,
    output_usd_per_million: float,
) -> dict[str, Any]:
    by_phase: dict[str, dict[str, int | float]] = {}
    for record in records:
        phase = str(record.get("phase", "unknown"))
        value = by_phase.setdefault(
            phase,
            {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_write_tokens": 0,
                "cache_read_tokens": 0,
                "anthropic_equivalent_cost_usd": 0.0,
            },
        )
        value["calls"] += int(record.get("calls", 1) or 0)
        value["input_tokens"] += int(record.get("prompt_tokens", 0) or 0)
        value["output_tokens"] += int(record.get("completion_tokens", 0) or 0)
        value["cache_write_tokens"] += int(
            record.get("cache_write_tokens", 0) or 0
        )
        value["cache_read_tokens"] += int(
            record.get("cache_read_tokens", 0) or 0
        )
        value["anthropic_equivalent_cost_usd"] += float(
            record.get("anthropic_equivalent_cost_usd", 0) or 0
        )
    for value in by_phase.values():
        # Cache writes and reads are billed at different rates per provider;
        # without those rates they are reported as volume only, and priced
        # here at the plain input rate.
        billable_input = (
            value["input_tokens"]
            + value["cache_write_tokens"]
            + value["cache_read_tokens"]
        )
        value["estimated_cost_usd"] = round(
            billable_input / 1_000_000 * input_usd_per_million
            + value["output_tokens"] / 1_000_000 * output_usd_per_million,
            8,
        )
        value["anthropic_equivalent_cost_usd"] = round(
            value["anthropic_equivalent_cost_usd"], 8
        )
    return {
        "totals": {
            key: sum(value[key] for value in by_phase.values())
            for key in (
                "calls",
                "input_tokens",
                "output_tokens",
                "cache_write_tokens",
                "cache_read_tokens",
                "estimated_cost_usd",
                "anthropic_equivalent_cost_usd",
            )
        },
        "by_phase": by_phase,
    }


def memory_inventory(root: Path) -> dict[str, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "topic_files": sum(
            path.suffix == ".md" and "topics" in path.relative_to(root).parts
            for path in files
        ),
        "timeline_files": sum(
            path.suffix == ".md" and "timeline" in path.relative_to(root).parts
            for path in files
        ),
        "source_files": sum(
            "sources" in path.relative_to(root).parts for path in files
        ),
    }


def latency_summary(
    completed: dict[int, dict[str, Any]],
) -> dict[str, float]:
    values = sorted(
        float(record["retrieval"]["latency_s"])
        for record in completed.values()
    )
    if not values:
        return {"mean_s": 0.0, "median_s": 0.0, "p95_s": 0.0, "max_s": 0.0}
    p95_index = min(len(values) - 1, int(0.95 * len(values)))
    return {
        "mean_s": round(statistics.fmean(values), 3),
        "median_s": round(statistics.median(values), 3),
        "p95_s": round(values[p95_index], 3),
        "max_s": round(values[-1], 3),
    }
