#!/usr/bin/env python3
"""Aggregate the audited fixed-W=32 GPT-5.6 Terra/Sol screening campaign.

This command is offline.  It reads the completed build, QA, and score artifacts,
validates their frozen inventories, and emits deterministic machine-readable and
paper-facing outputs.  It never sends a model request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_ANALYSIS = (
    ROOT
    / "results/formal/gpt56-chunk-curve-subscription-20260719-w32-only"
    / "analysis/build-quality-cost-v2.json"
)
DEFAULT_QA_BASE = ROOT / "results/formal/gpt56-w32-screening-qa-20260719"
DEFAULT_SCORE_ROOT = (
    ROOT / "results/formal/gpt56-w32-screening-scores-20260719-r1"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "results/formal/gpt56-w32-screening-analysis-20260719-r1"
)
DEFAULT_PAPER_DIR = ROOT / "paper/figures"

SCHEMA_VERSION = 1
STUDY = "gpt56-w32-screening-analysis"
TIERS = ("terra", "sol")
QA_TIERS = ("luna", "terra", "sol")
WRITER_WINDOWS = (4, 8, 16, 32)
BENCHMARKS = ("locomo", "longmemeval-s", "beam-100k")
EXPECTED_COUNTS = {"locomo": 314, "longmemeval-s": 2, "beam-100k": 40}
EXPECTED_QUESTIONS_PER_TIER = sum(EXPECTED_COUNTS.values())
EXPECTED_UNITS = {
    "locomo": ("conv-44", "conv-48"),
    "longmemeval-s": ("2318644b", "gpt4_6dc9b45b"),
    "beam-100k": ("100K-conv-1", "100K-conv-2"),
}
LOCOMO_CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
}
LOCOMO_CATEGORY_ORDER = (
    "multi-hop",
    "temporal",
    "open-domain",
    "single-hop",
)
LOCOMO_CATEGORY_COUNTS = {
    "multi-hop": 51,
    "temporal": 66,
    "open-domain": 17,
    "single-hop": 180,
}
BEAM_QUESTION_TYPES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)
LME_IDS = ("2318644b", "gpt4_6dc9b45b")
LOCKED_LOCOMO_EVALUATOR_SHA256 = (
    "f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd"
)
CLAIM_BOUNDARY = {
    "scope": "fixed W=32 paired screening",
    "not_established": [
        "writer-window curve",
        "W* selection",
        "full-benchmark accuracy",
        "benchmark-level confidence interval",
        "semantic correctness of memory entries",
    ],
}


class ScreeningAnalysisError(RuntimeError):
    """An input is incomplete, unaudited, inconsistent, or out of scope."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _regular_file(path: Path) -> Path:
    resolved = path.expanduser().absolute()
    if resolved.is_symlink() or not resolved.is_file():
        raise ScreeningAnalysisError(f"required regular file is missing: {resolved}")
    return resolved


def _read_json(path: Path) -> Any:
    checked = _regular_file(path)
    try:
        return json.loads(checked.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScreeningAnalysisError(f"invalid JSON input: {checked}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    checked = _regular_file(path)
    payload = checked.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise ScreeningAnalysisError(f"JSONL has an incomplete final line: {checked}")
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ScreeningAnalysisError(
                f"invalid JSONL input: {checked}:{number}"
            ) from exc
        if not isinstance(value, dict):
            raise ScreeningAnalysisError(
                f"JSONL row is not an object: {checked}:{number}"
            )
        rows.append(value)
    return rows


def _display_path(path: Path) -> str:
    absolute = path.expanduser().absolute()
    try:
        return str(absolute.relative_to(ROOT))
    except ValueError:
        return str(absolute)


def _descriptor(path: Path) -> dict[str, str]:
    checked = _regular_file(path)
    return {"path": _display_path(checked), "sha256": sha256_file(checked)}


def _payload_descriptor(path: Path, payload: bytes) -> dict[str, str]:
    return {
        "path": _display_path(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _tree_descriptor(paths: Iterable[Path], *, base: Path) -> dict[str, Any]:
    checked_base = base.expanduser().absolute()
    entries: list[dict[str, str]] = []
    lines: list[str] = []
    for path in sorted({item.expanduser().absolute() for item in paths}):
        checked = _regular_file(path)
        try:
            relative = str(checked.relative_to(checked_base))
        except ValueError as exc:
            raise ScreeningAnalysisError(
                f"tree input is outside its root: {checked}"
            ) from exc
        digest = sha256_file(checked)
        entries.append({"path": relative, "sha256": digest})
        lines.append(f"{relative}\t{digest}\n")
    return {
        "file_count": len(entries),
        "sha256": hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(),
        "entries": entries,
    }


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _write_no_clobber(path: Path, payload: bytes) -> None:
    output = path.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file() or output.read_bytes() != payload:
            raise ScreeningAnalysisError(f"refusing to clobber output: {output}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_bundle_no_clobber(payloads: Mapping[Path, bytes]) -> None:
    """Preflight every target before publishing any file in the bundle."""

    normalized: dict[Path, bytes] = {}
    for path, payload in payloads.items():
        if not isinstance(payload, bytes):
            raise ScreeningAnalysisError("output payload is not bytes")
        target = path.expanduser().absolute()
        previous = normalized.get(target)
        if previous is not None and previous != payload:
            raise ScreeningAnalysisError(f"conflicting output payloads: {target}")
        normalized[target] = payload
    for target, payload in normalized.items():
        if target.exists() or target.is_symlink():
            if (
                target.is_symlink()
                or not target.is_file()
                or target.read_bytes() != payload
            ):
                raise ScreeningAnalysisError(f"refusing to clobber output: {target}")
    for target in normalized:
        target.parent.mkdir(parents=True, exist_ok=True)
    for target in sorted(normalized, key=str):
        _write_no_clobber(target, normalized[target])


def _write_json_no_clobber(path: Path, value: object) -> None:
    _write_no_clobber(path, _json_bytes(value))


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _write_jsonl_no_clobber(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_no_clobber(path, _jsonl_bytes(rows))


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScreeningAnalysisError(f"{label} is not numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ScreeningAnalysisError(f"{label} is not finite")
    return numeric


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ScreeningAnalysisError(f"{label} is not an integer >= {minimum}")
    return value


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _close(left: object, right: object, *, tolerance: float = 1e-10) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _implied_count(rate: object, denominator: int, *, label: str) -> int:
    numeric = _number(rate, label=label)
    implied = numeric * denominator
    rounded = round(implied)
    if not math.isclose(implied, rounded, abs_tol=1e-7):
        raise ScreeningAnalysisError(f"{label} does not imply an integer count")
    return int(rounded)


def validate_qa_audit(audit: object, *, tier: str) -> None:
    """Reject any QA root that lacks the exact completed independent audit."""

    if tier not in QA_TIERS or not isinstance(audit, Mapping):
        raise ScreeningAnalysisError(f"{tier} QA audit is invalid")
    proxy = audit.get("proxy")
    exact = (
        audit.get("status") == "verified_complete",
        audit.get("question_count") == EXPECTED_QUESTIONS_PER_TIER,
        audit.get("independently_reconstructed_questions")
        == EXPECTED_QUESTIONS_PER_TIER,
        audit.get("all_memory_unchanged") is True,
        isinstance(proxy, Mapping) and proxy.get("all_successes_accounted") is True,
    )
    if not all(exact):
        raise ScreeningAnalysisError(f"{tier} QA audit did not pass all gates")


def _empty_usage() -> dict[str, int]:
    return {
        "calls": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "uncached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }


def _add_usage(target: dict[str, int], source: Mapping[str, Any]) -> None:
    for key in target:
        if key == "uncached_input_tokens":
            continue
        target[key] += _integer(source.get(key, 0), label=f"phase {key}")
    target["uncached_input_tokens"] += _integer(
        source.get(
            "uncached_input_tokens",
            _integer(source.get("input_tokens", 0), label="phase input_tokens")
            - _integer(
                source.get("cached_input_tokens", 0),
                label="phase cached_input_tokens",
            ),
        ),
        label="phase uncached_input_tokens",
    )


def _aggregate_phase_usage(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw: dict[str, dict[str, int]] = {}
    for row in runs:
        phases = row.get("cost", {}).get("phase_usage", {})
        if not isinstance(phases, Mapping):
            raise ScreeningAnalysisError("build phase_usage is invalid")
        for name, usage in phases.items():
            if not isinstance(usage, Mapping):
                raise ScreeningAnalysisError("build phase usage is invalid")
            destination = raw.setdefault(str(name), _empty_usage())
            _add_usage(destination, usage)
    expected_phases = {
        "v8_distill",
        "v8_distill_verify",
        "v8_consolidate",
        "v8_merge_lines",
        "v8_sections",
    }
    if set(raw) != expected_phases:
        raise ScreeningAnalysisError("build phase inventory differs")
    phase_totals = _empty_usage()
    for usage in raw.values():
        _add_usage(phase_totals, usage)
    reported_totals = _empty_usage()
    for row in runs:
        cost = row.get("cost")
        if not isinstance(cost, Mapping):
            raise ScreeningAnalysisError("build cost is invalid")
        _add_usage(reported_totals, cost)
    if phase_totals != reported_totals:
        raise ScreeningAnalysisError("build phase totals differ from run costs")
    groups: dict[str, dict[str, int]] = {}
    definitions = {
        "writer": ("v8_distill",),
        "verify": ("v8_distill_verify",),
        "maintenance": ("v8_consolidate", "v8_merge_lines", "v8_sections"),
    }
    for group, names in definitions.items():
        aggregate = _empty_usage()
        for name in names:
            if name in raw:
                _add_usage(aggregate, raw[name])
        groups[group] = aggregate
    return {"raw": raw, "groups": groups}


def _aggregate_construction_gold(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = any_hits = all_hits = 0
    gold_targets = target_hits = 0
    available_runs = 0
    for row in runs:
        if row.get("identity", {}).get("benchmark") != "locomo":
            continue
        coverage = row.get("combined_references", {}).get("gold_coverage", {})
        if not isinstance(coverage, Mapping) or coverage.get("availability") != "available":
            continue
        run_eligible = _integer(
            coverage.get("eligible_questions"), label="gold eligible_questions"
        )
        run_targets = _integer(
            coverage.get("unique_gold_targets"), label="gold unique targets"
        )
        eligible += run_eligible
        gold_targets += run_targets
        any_hits += _implied_count(
            coverage.get("question_any_coverage"),
            run_eligible,
            label="gold question_any_coverage",
        )
        all_hits += _implied_count(
            coverage.get("question_all_coverage"),
            run_eligible,
            label="gold question_all_coverage",
        )
        target_hits += _implied_count(
            coverage.get("unique_gold_target_coverage"),
            run_targets,
            label="gold unique_gold_target_coverage",
        )
        available_runs += 1
    if not available_runs:
        return {
            "availability": "not_available",
            "eligible_questions": 0,
            "question_any_coverage": None,
            "question_all_coverage": None,
            "unique_gold_targets": 0,
            "unique_gold_target_coverage": None,
        }
    return {
        "availability": "available",
        "eligible_questions": eligible,
        "question_any_hits": any_hits,
        "question_any_coverage": _ratio(any_hits, eligible),
        "question_all_hits": all_hits,
        "question_all_coverage": _ratio(all_hits, eligible),
        "unique_gold_targets": gold_targets,
        "unique_gold_target_hits": target_hits,
        "unique_gold_target_coverage": _ratio(target_hits, gold_targets),
    }


def validate_and_aggregate_build(report: object) -> dict[str, Any]:
    """Validate the 12-run W32 build report and recompute weighted summaries."""

    if not isinstance(report, Mapping):
        raise ScreeningAnalysisError("build analysis is not an object")
    if (
        report.get("status") != "complete"
        or report.get("study") != "gpt56-w32-screening-build-analysis"
        or report.get("run_count") != 12
        or not isinstance(report.get("runs"), list)
        or len(report["runs"]) != 12
    ):
        raise ScreeningAnalysisError("build analysis did not pass admission")
    rows = report["runs"]
    results: dict[str, Any] = {}
    for tier in TIERS:
        selected = [
            row
            for row in rows
            if isinstance(row, Mapping)
            and row.get("identity", {}).get("tier") == tier
        ]
        if len(selected) != 6:
            raise ScreeningAnalysisError(f"build tier {tier} does not contain 6 runs")
        inventory = Counter(
            str(row.get("identity", {}).get("benchmark")) for row in selected
        )
        if inventory != Counter({benchmark: 2 for benchmark in BENCHMARKS}):
            raise ScreeningAnalysisError(f"build tier {tier} inventory differs")
        observed_units = {
            (
                str(row.get("identity", {}).get("benchmark")),
                str(row.get("identity", {}).get("unit_id")),
            )
            for row in selected
        }
        expected_units = {
            (benchmark, unit)
            for benchmark, units in EXPECTED_UNITS.items()
            for unit in units
        }
        if observed_units != expected_units or any(
            row.get("identity", {}).get("write_turns") != 32 for row in selected
        ):
            raise ScreeningAnalysisError(f"build tier {tier} unit or W differs")

        totals = {
            "units": 6,
            "messages": 0,
            "source_tokens": 0,
            "source_bytes": 0,
            "calls": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "uncached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "wall_time_s": 0.0,
        }
        first_pass = post_verify = reference_valid = reference_total = 0
        abstract_bytes = abstract_files = abstract_entries = 0
        unit_rows: list[dict[str, Any]] = []
        for row in selected:
            identity = row["identity"]
            source = row.get("source")
            cost = row.get("cost")
            distill = row.get("distill")
            combined = row.get("combined_references")
            memory = row.get("memory_size")
            if not all(
                isinstance(value, Mapping)
                for value in (source, cost, distill, combined, memory)
            ):
                raise ScreeningAnalysisError("build run lacks required metrics")
            messages = _integer(source.get("turns"), label="source turns", minimum=1)
            source_tokens = _integer(
                source.get("tokens"), label="source tokens", minimum=1
            )
            source_bytes = _integer(
                source.get("bytes"), label="source bytes", minimum=1
            )
            unique_sources = _integer(
                source.get("unique_source_ids"),
                label="unique source IDs",
                minimum=1,
            )
            if unique_sources != messages:
                raise ScreeningAnalysisError("source turn denominator differs")
            totals["messages"] += messages
            totals["source_tokens"] += source_tokens
            totals["source_bytes"] += source_bytes
            for name in (
                "calls",
                "input_tokens",
                "cached_input_tokens",
                "uncached_input_tokens",
                "output_tokens",
                "reasoning_tokens",
            ):
                totals[name] += _integer(cost.get(name, 0), label=f"build {name}")
            wall = _number(cost.get("wall_time_s"), label="build wall_time_s")
            totals["wall_time_s"] += wall
            first_pass += _integer(
                distill.get("first_pass_unique_dia_ids"), label="first-pass coverage"
            )
            post_verify += _integer(
                distill.get("post_verify_unique_dia_ids"), label="post-verify coverage"
            )
            reference_valid += _integer(
                combined.get("valid_mentions"), label="valid reference mentions"
            )
            reference_total += _integer(
                combined.get("expanded_mentions"), label="reference mentions"
            )
            abstract = memory.get("abstract")
            if not isinstance(abstract, Mapping):
                raise ScreeningAnalysisError("build abstract memory metrics are absent")
            abstract_bytes += _integer(
                abstract.get("bytes"), label="abstract bytes"
            )
            abstract_files += _integer(
                abstract.get("files"), label="abstract files"
            )
            abstract_entries += _integer(
                abstract.get("entries"), label="abstract entries"
            )
            unit_rows.append(
                {
                    "benchmark": identity["benchmark"],
                    "unit_id": identity["unit_id"],
                    "messages": messages,
                    "source_tokens": source_tokens,
                    "calls_per_100_messages": cost["calls"] / messages * 100,
                    "input_tokens_per_1000_source_tokens": (
                        cost["input_tokens"] / source_tokens * 1000
                    ),
                    "output_tokens_per_1000_source_tokens": (
                        cost["output_tokens"] / source_tokens * 1000
                    ),
                    "wall_minutes_per_100_messages": wall / 60 / messages * 100,
                }
            )
        if totals["input_tokens"] - totals["cached_input_tokens"] != totals[
            "uncached_input_tokens"
        ]:
            raise ScreeningAnalysisError("build cached-token accounting differs")
        phases = _aggregate_phase_usage(selected)
        results[tier] = {
            "totals": totals,
            "normalized": {
                "calls_per_100_messages": totals["calls"] / totals["messages"] * 100,
                "input_tokens_per_1000_source_tokens": (
                    totals["input_tokens"] / totals["source_tokens"] * 1000
                ),
                "output_tokens_per_1000_source_tokens": (
                    totals["output_tokens"] / totals["source_tokens"] * 1000
                ),
                "wall_minutes_per_100_messages": (
                    totals["wall_time_s"] / 60 / totals["messages"] * 100
                ),
            },
            "phase_groups": phases["groups"],
            "raw_phase_usage": phases["raw"],
            "grounding": {
                "first_pass_source_ids": first_pass,
                "post_verify_source_ids": post_verify,
                "source_id_denominator": totals["messages"],
                "first_pass_source_coverage": first_pass / totals["messages"],
                "post_verify_source_coverage": post_verify / totals["messages"],
                "valid_reference_mentions": reference_valid,
                "reference_mentions": reference_total,
                "reference_precision": _ratio(reference_valid, reference_total),
                "locomo_gold_source_coverage": _aggregate_construction_gold(selected),
            },
            "representation": {
                "abstract_bytes": abstract_bytes,
                "source_bytes": totals["source_bytes"],
                "abstract_to_source_byte_ratio": abstract_bytes
                / totals["source_bytes"],
                "abstract_files": abstract_files,
                "abstract_entries": abstract_entries,
            },
            "units": sorted(
                unit_rows, key=lambda row: (row["benchmark"], row["unit_id"])
            ),
        }
    denominators = ("messages", "source_tokens", "source_bytes")
    if any(
        results["terra"]["totals"][key] != results["sol"]["totals"][key]
        for key in denominators
    ):
        raise ScreeningAnalysisError("Terra and Sol build denominators differ")
    return results


def _accepted_attempt(
    question_root: Path, result: Mapping[str, Any]
) -> tuple[Path, list[Path], int]:
    attempts: list[tuple[int, Path]] = []
    matches: list[tuple[int, Path]] = []
    for manifest in sorted(question_root.glob("attempt-*/attempt_manifest.json")):
        root = manifest.parent
        try:
            number = int(root.name.removeprefix("attempt-"))
        except ValueError as exc:
            raise ScreeningAnalysisError(f"invalid attempt directory: {root}") from exc
        attempts.append((number, root))
        result_path = root / "result.json"
        if result_path.is_file() and not result_path.is_symlink():
            candidate = _read_json(result_path)
            if candidate == result:
                matches.append((number, root))
    if len(matches) != 1:
        raise ScreeningAnalysisError(
            f"question does not have one uniquely matching accepted attempt: {question_root}"
        )
    numbers = sorted(number for number, _ in attempts)
    accepted_number, accepted_root = matches[0]
    if numbers != list(range(1, accepted_number + 1)):
        raise ScreeningAnalysisError(f"attempt chain is not contiguous: {question_root}")
    files = [
        path
        for path in sorted(accepted_root.rglob("*"))
        if path.is_file() or path.is_symlink()
    ]
    if any(path.is_symlink() for path in files):
        raise ScreeningAnalysisError(f"accepted attempt contains a symlink: {accepted_root}")
    return accepted_root, files, accepted_number - 1


def _validate_artifact_hashes(
    result: Mapping[str, Any], accepted_root: Path
) -> dict[str, Path]:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ScreeningAnalysisError("question result has no artifact binding")
    names = {
        "retrieval_model_ledger": "retrieval_model_ledger_sha256",
        "answer_ledger": "answer_ledger_sha256",
        "visible_token_manifest": "visible_token_manifest_sha256",
        "visible_token_trace": "visible_token_trace_sha256",
        "attempt_manifest": "attempt_manifest_sha256",
    }
    paths: dict[str, Path] = {}
    for key, hash_key in names.items():
        relative = artifacts.get(key)
        expected_hash = artifacts.get(hash_key)
        if relative is None and key == "visible_token_trace":
            continue
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ScreeningAnalysisError(f"question artifact binding is absent: {key}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ScreeningAnalysisError(
                f"question artifact is outside accepted attempt: {relative}"
            )
        path = accepted_root / relative_path
        if sha256_file(_regular_file(path)) != expected_hash:
            raise ScreeningAnalysisError(f"question artifact hash differs: {path}")
        paths[key] = path
    return paths


def _retrieval_usage(path: Path) -> dict[str, Any]:
    prompt = cached = completion = calls = client_attempts = upstream_attempts = 0
    latency = 0.0
    for event in _read_jsonl(path):
        if event.get("event") != "model_call_finished":
            continue
        usage = event.get("usage")
        evidence = event.get("proxy_evidence")
        if not isinstance(usage, Mapping) or not isinstance(evidence, Mapping):
            raise ScreeningAnalysisError("retrieval ledger lacks usage evidence")
        events = evidence.get("events")
        if not isinstance(events, list):
            raise ScreeningAnalysisError("retrieval proxy evidence is invalid")
        successful = [
            item
            for item in events
            if isinstance(item, Mapping) and item.get("status") == "success"
        ]
        if len(successful) != 1:
            raise ScreeningAnalysisError("retrieval logical call lacks one success")
        proxy_usage = successful[0].get("usage")
        if not isinstance(proxy_usage, Mapping):
            raise ScreeningAnalysisError("retrieval proxy usage is absent")
        ledger_prompt = _integer(
            usage.get("prompt_tokens"), label="retrieval prompt tokens"
        )
        ledger_completion = _integer(
            usage.get("completion_tokens"), label="retrieval completion tokens"
        )
        if (
            proxy_usage.get("prompt_tokens") != ledger_prompt
            or proxy_usage.get("completion_tokens") != ledger_completion
        ):
            raise ScreeningAnalysisError("retrieval ledger and proxy usage differ")
        details = proxy_usage.get("prompt_tokens_details", {})
        if not isinstance(details, Mapping):
            raise ScreeningAnalysisError("retrieval cached-token details are invalid")
        prompt += ledger_prompt
        completion += ledger_completion
        cached += _integer(details.get("cached_tokens", 0), label="cached tokens")
        latency += _number(event.get("latency_s"), label="retrieval latency")
        client_attempts += _integer(
            evidence.get("client_http_attempts", 1),
            label="retrieval client attempts",
        )
        upstream_attempts += _integer(
            evidence.get("upstream_http_attempts", 1),
            label="retrieval upstream attempts",
        )
        calls += 1
    if not calls or cached > prompt:
        raise ScreeningAnalysisError("retrieval usage accounting is invalid")
    return {
        "model_calls": calls,
        "latency_s": latency,
        "prompt_tokens": prompt,
        "cached_input_tokens": cached,
        "uncached_input_tokens": prompt - cached,
        "completion_tokens": completion,
        "client_http_attempts": client_attempts,
        "upstream_http_attempts": upstream_attempts,
    }


def _answer_usage(answer: object) -> dict[str, Any]:
    if not isinstance(answer, Mapping):
        raise ScreeningAnalysisError("answer evidence is absent")
    usage = answer.get("usage")
    if not isinstance(usage, Mapping):
        raise ScreeningAnalysisError("answer usage is absent")
    prompt = _integer(usage.get("prompt_tokens"), label="answer prompt tokens")
    completion = _integer(
        usage.get("completion_tokens"), label="answer completion tokens"
    )
    details = usage.get("prompt_tokens_details", {})
    if not isinstance(details, Mapping):
        raise ScreeningAnalysisError("answer cached-token details are invalid")
    cached = _integer(details.get("cached_tokens", 0), label="answer cached tokens")
    if cached > prompt:
        raise ScreeningAnalysisError("answer cached tokens exceed prompt tokens")
    latency_values: list[float] = []
    evidence = answer.get("proxy_evidence", {})
    events = evidence.get("events", []) if isinstance(evidence, Mapping) else []
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, Mapping) or event.get("status") != "success":
                continue
            start = event.get("started_at")
            finish = event.get("finished_at")
            if isinstance(start, str) and isinstance(finish, str):
                try:
                    latency_values.append(
                        (
                            datetime.fromisoformat(finish)
                            - datetime.fromisoformat(start)
                        ).total_seconds()
                    )
                except ValueError:
                    pass
    return {
        "logical_calls": _integer(
            answer.get("logical_calls"), label="answer logical calls", minimum=1
        ),
        "client_http_attempts": _integer(
            answer.get("client_http_attempts"), label="answer client attempts"
        ),
        "upstream_http_attempts": _integer(
            answer.get("upstream_http_attempts"), label="answer upstream attempts"
        ),
        "prompt_tokens": prompt,
        "cached_input_tokens": cached,
        "uncached_input_tokens": prompt - cached,
        "completion_tokens": completion,
        "latency_s": sum(latency_values) if latency_values else None,
    }


def extract_question_row(
    record_path: Path, *, tier: str, write_turns: int = 32
) -> tuple[dict[str, Any], list[Path]]:
    """Validate one accepted question and return its analysis row and input files."""

    record = _read_json(record_path)
    if not isinstance(record, Mapping) or record.get("tier") != tier:
        raise ScreeningAnalysisError(f"question tier differs: {record_path}")
    benchmark = record.get("benchmark")
    unit_id = record.get("unit_id")
    original_id = record.get("original_question_id")
    result = record.get("result")
    if (
        benchmark not in BENCHMARKS
        or unit_id not in EXPECTED_UNITS[str(benchmark)]
        or not isinstance(original_id, str)
        or not original_id
        or not isinstance(result, Mapping)
        or result.get("status") != "complete"
    ):
        raise ScreeningAnalysisError(f"question identity or result differs: {record_path}")
    if write_turns not in WRITER_WINDOWS:
        raise ScreeningAnalysisError("writer window is outside the analysis set")
    expected_run_id = (
        f"gpt56-{benchmark}-{str(unit_id).replace('_', '-')}-{tier}-w{write_turns}"
    )
    if (
        record.get("run_id") != expected_run_id
        or not isinstance(record.get("question_id"), str)
        or not record.get("question_id")
        or result.get("question_id") != record.get("question_id")
    ):
        raise ScreeningAnalysisError(f"question run or result identity differs: {record_path}")
    accepted_root, accepted_files, retry_count = _accepted_attempt(
        record_path.parent, result
    )
    artifacts = _validate_artifact_hashes(result, accepted_root)
    retrieval = _retrieval_usage(artifacts["retrieval_model_ledger"])
    answer_events = _read_jsonl(artifacts["answer_ledger"])
    if not any(event.get("event") == "answer_completed" for event in answer_events):
        raise ScreeningAnalysisError("accepted answer ledger is incomplete")
    answer = _answer_usage(result.get("answer"))
    result_retrieval = result.get("retrieval")
    diagnostics = result.get("diagnostics")
    budget = result.get("budget")
    if not all(
        isinstance(value, Mapping)
        for value in (result_retrieval, diagnostics, budget)
    ):
        raise ScreeningAnalysisError("question access evidence is incomplete")
    if result_retrieval.get("model_calls") != retrieval["model_calls"]:
        raise ScreeningAnalysisError("retrieval ledger and result summary differ")
    retrieval_latency = _number(
        result_retrieval.get("latency_s"), label="retrieval operation latency"
    )
    if retrieval_latency < 0:
        raise ScreeningAnalysisError("retrieval operation latency is negative")
    visible_manifest = _read_json(artifacts["visible_token_manifest"])
    if (
        not isinstance(visible_manifest, Mapping)
        or visible_manifest.get("status") != "complete"
        or not isinstance(visible_manifest.get("summary"), Mapping)
    ):
        raise ScreeningAnalysisError("visible-token manifest is incomplete")
    visible_summary = visible_manifest["summary"]
    configured = _integer(
        budget.get("configured_tokens"), label="configured visible-token budget"
    )
    visible = _integer(budget.get("visible_tokens"), label="visible tokens")
    source_resolution = _integer(
        budget.get("source_resolution_tokens", 0), label="source-resolution tokens"
    )
    if (
        visible_summary.get("configured_budget_tokens") != configured
        or visible_summary.get("cumulative_visible_tokens") != visible
        or visible_summary.get("cumulative_source_resolution_tokens")
        != source_resolution
    ):
        raise ScreeningAnalysisError("visible-token record and manifest differ")

    eligible = diagnostics.get("source_recall_eligible") is True
    if eligible and benchmark != "locomo":
        raise ScreeningAnalysisError(
            "mapped-source recall eligibility is restricted to LoCoMo"
        )
    gold = diagnostics.get("gold_source_ids")
    hits = diagnostics.get("mapped_source_hits")
    if not isinstance(gold, list) or not isinstance(hits, list):
        raise ScreeningAnalysisError("source-recall diagnostics are invalid")
    gold_ids = {str(value) for value in gold}
    hit_ids = {str(value) for value in hits}
    if eligible and (not gold_ids or not hit_ids <= gold_ids):
        raise ScreeningAnalysisError("eligible source-recall mapping is invalid")
    mapped_recall = len(hit_ids) / len(gold_ids) if eligible else None
    if eligible and not _close(
        diagnostics.get("mapped_source_recall"), mapped_recall
    ):
        raise ScreeningAnalysisError("mapped-source recall differs")
    if not eligible and diagnostics.get("mapped_source_recall") is not None:
        raise ScreeningAnalysisError("ineligible source recall is non-null")

    access_log = result_retrieval.get("access_log")
    if not isinstance(access_log, list):
        raise ScreeningAnalysisError("retrieval access log is invalid")
    reads = [
        item
        for item in access_log
        if isinstance(item, Mapping)
        and isinstance(item.get("path"), str)
        and bool(item.get("path"))
    ]
    first_file_hit: bool | None = None
    first_relevant_rank: int | None = None
    if eligible:
        first_file_hit = bool(
            reads and gold_ids.intersection(map(str, reads[0].get("source_ids", [])))
        )
        for index, read in enumerate(reads, start=1):
            if gold_ids.intersection(map(str, read.get("source_ids", []))):
                first_relevant_rank = index
                break
    category = record.get("category")
    category_name = (
        LOCOMO_CATEGORY_NAMES.get(category)
        if isinstance(category, int) and not isinstance(category, bool)
        else None
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "tier": tier,
        "benchmark": benchmark,
        "unit_id": unit_id,
        "question_id": record.get("question_id"),
        "original_question_id": original_id,
        "category_id": category,
        "category_name": category_name,
        "question_type": record.get("question_type"),
        "question_sha256": sha256_text(str(record.get("question", ""))),
        "answer_sha256": sha256_text(str(record.get("answer", ""))),
        "quality": {"metric": None, "score": None, "correct": None, "pass": None},
        "access": {
            "source_recall_eligible": eligible,
            "gold_source_count": len(gold_ids) if eligible else None,
            "mapped_source_hits": len(hit_ids) if eligible else None,
            "mapped_source_recall": mapped_recall,
            "first_file_hit": first_file_hit,
            "first_relevant_read_rank": first_relevant_rank,
            "read_calls": _integer(
                diagnostics.get("read_calls"), label="read calls"
            ),
            "tool_calls": _integer(
                diagnostics.get("tool_calls"), label="tool calls"
            ),
            "visible_tokens": visible,
            "source_resolution_tokens": source_resolution,
            "configured_tokens": configured,
            "budget_hit": visible >= configured,
            "budget_fraction": _ratio(visible, configured),
            "retrieval_model_calls": retrieval["model_calls"],
            "retrieval_latency_s": retrieval_latency,
            "tool_latency_s": _number(
                diagnostics.get("tool_latency_s"), label="tool latency"
            ),
            "retrieval_prompt_tokens": retrieval["prompt_tokens"],
            "retrieval_cached_input_tokens": retrieval["cached_input_tokens"],
            "retrieval_uncached_input_tokens": retrieval["uncached_input_tokens"],
            "retrieval_completion_tokens": retrieval["completion_tokens"],
            "answer_prompt_tokens": answer["prompt_tokens"],
            "answer_cached_input_tokens": answer["cached_input_tokens"],
            "answer_uncached_input_tokens": answer["uncached_input_tokens"],
            "answer_completion_tokens": answer["completion_tokens"],
            "answer_latency_s": answer["latency_s"],
            "accepted_logical_calls": retrieval["model_calls"]
            + answer["logical_calls"],
            "accepted_client_http_attempts": retrieval["client_http_attempts"]
            + answer["client_http_attempts"],
            "accepted_upstream_http_attempts": retrieval["upstream_http_attempts"]
            + answer["upstream_http_attempts"],
            "retry_count": retry_count,
        },
        "provenance": {
            "record_path": _display_path(record_path),
            "record_sha256": sha256_file(record_path),
            "attempt_path": _display_path(accepted_root),
            "result_sha256": sha256_file(accepted_root / "result.json"),
            "retrieval_ledger_sha256": sha256_file(
                artifacts["retrieval_model_ledger"]
            ),
            "answer_ledger_sha256": sha256_file(artifacts["answer_ledger"]),
        },
        "_join": {
            "question": record.get("question"),
            "answer": record.get("answer"),
            "gold": record.get("gold"),
            "rubric": record.get("rubric"),
            "selection": record.get("selection"),
        },
    }
    return row, accepted_files


def _validate_completion(
    completion: object,
    audit: Mapping[str, Any],
    *,
    tier: str,
    write_turns: int = 32,
) -> None:
    if not isinstance(completion, Mapping):
        raise ScreeningAnalysisError(f"{tier} completion is invalid")
    if (
        completion.get("status") != "complete"
        or completion.get("question_count") != EXPECTED_QUESTIONS_PER_TIER
        or not isinstance(completion.get("run_ids"), list)
        or len(completion["run_ids"]) != 6
    ):
        raise ScreeningAnalysisError(f"{tier} completion did not pass admission")
    if write_turns not in WRITER_WINDOWS:
        raise ScreeningAnalysisError("writer window is outside the analysis set")
    expected_run_ids = [
        f"gpt56-{benchmark}-{unit.replace('_', '-')}-{tier}-w{write_turns}"
        for benchmark, units in EXPECTED_UNITS.items()
        for unit in units
    ]
    if (
        completion.get("run_ids") != expected_run_ids
        or audit.get("run_ids") != expected_run_ids
    ):
        raise ScreeningAnalysisError(f"{tier} audit/completion run IDs differ")


def load_qa_tier(
    root: Path,
    *,
    tier: str,
    write_turns: int = 32,
) -> dict[str, Any]:
    qa_root = root.expanduser().absolute()
    audit_path = qa_root / "audit.json"
    completion_path = qa_root / "completion.json"
    audit = _read_json(audit_path)
    completion = _read_json(completion_path)
    validate_qa_audit(audit, tier=tier)
    _validate_completion(
        completion,
        audit,
        tier=tier,
        write_turns=write_turns,
    )
    if audit.get("completion_sha256") != sha256_file(completion_path):
        raise ScreeningAnalysisError(f"{tier} audit completion hash differs")
    record_paths = sorted(
        qa_root.glob(f"{tier}/*/*/questions/*/record.json")
    )
    if len(record_paths) != EXPECTED_QUESTIONS_PER_TIER:
        raise ScreeningAnalysisError(f"{tier} record inventory is incomplete")
    rows: list[dict[str, Any]] = []
    accepted_files: list[Path] = []
    attempt_resolution_files: list[Path] = []
    seen: set[tuple[str, str, str]] = set()
    for path in record_paths:
        row, files = extract_question_row(
            path,
            tier=tier,
            write_turns=write_turns,
        )
        key = (row["benchmark"], row["unit_id"], row["original_question_id"])
        if key in seen:
            raise ScreeningAnalysisError(f"duplicate QA question: {key}")
        seen.add(key)
        rows.append(row)
        accepted_files.extend(files)
        attempt_resolution_files.extend(
            path.parent.glob("attempt-*/attempt_manifest.json")
        )
        attempt_resolution_files.extend(path.parent.glob("attempt-*/result.json"))
    inventory = Counter(row["benchmark"] for row in rows)
    if inventory != Counter(EXPECTED_COUNTS):
        raise ScreeningAnalysisError(f"{tier} benchmark inventory differs")
    return {
        "root": qa_root,
        "audit_path": audit_path,
        "completion_path": completion_path,
        "rows": rows,
        "record_paths": record_paths,
        "accepted_files": accepted_files,
        "attempt_resolution_files": attempt_resolution_files,
    }


def _assert_score(value: object, expected: float, *, label: str) -> None:
    if not _close(value, expected):
        raise ScreeningAnalysisError(f"{label} does not recompute")


def _locomo_score(
    *, score_root: Path, tier: str, rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[Path]]:
    root = score_root / tier / "locomo"
    eval_path = root / "eval_full.json"
    evaluation = _read_json(eval_path)
    if not isinstance(evaluation, Mapping):
        raise ScreeningAnalysisError("LoCoMo score is invalid")
    qa_by_unit = {
        unit: [row for row in rows if row["benchmark"] == "locomo" and row["unit_id"] == unit]
        for unit in EXPECTED_UNITS["locomo"]
    }
    ordered: list[dict[str, Any]] = []
    inputs: list[Path] = [eval_path]
    for sample, unit in ((5, "conv-44"), (7, "conv-48")):
        path = root / f"sample{sample}_questions.json"
        values = _read_json(path)
        inputs.append(path)
        if not isinstance(values, list):
            raise ScreeningAnalysisError("LoCoMo score input is not a list")
        qa_map = {row["original_question_id"]: row for row in qa_by_unit[unit]}
        if len(values) != len(qa_map):
            raise ScreeningAnalysisError("LoCoMo score-input inventory differs")
        for value in values:
            if not isinstance(value, Mapping):
                raise ScreeningAnalysisError("LoCoMo score input row is invalid")
            row = qa_map.get(str(value.get("question_id")))
            if row is None:
                raise ScreeningAnalysisError("LoCoMo score input ID differs")
            join = row["_join"]
            exact = {
                "question_id": row["original_question_id"],
                "question": join["question"],
                "gold": join["gold"],
                "category": row["category_id"],
                "answer": join["answer"],
            }
            if any(value.get(key) != expected for key, expected in exact.items()):
                raise ScreeningAnalysisError("LoCoMo score input and QA record differ")
            ordered.append(row)
    scored = evaluation.get("records")
    if evaluation.get("n") != 314 or not isinstance(scored, list) or len(scored) != 314:
        raise ScreeningAnalysisError("LoCoMo score inventory differs")
    by_category: defaultdict[str, list[int]] = defaultdict(list)
    by_unit: defaultdict[str, list[int]] = defaultdict(list)
    for row, score_row in zip(ordered, scored, strict=True):
        if not isinstance(score_row, Mapping):
            raise ScreeningAnalysisError("LoCoMo scored row is invalid")
        join = row["_join"]
        exact = {
            "question_id": row["original_question_id"],
            "question": join["question"],
            "gold": join["gold"],
            "category": row["category_id"],
            "answer": join["answer"],
        }
        if any(score_row.get(key) != expected for key, expected in exact.items()):
            raise ScreeningAnalysisError("LoCoMo score order or content differs")
        score = score_row.get("judge_score")
        if score not in (0, 1):
            raise ScreeningAnalysisError("LoCoMo judge score is not binary")
        row["quality"] = {
            "metric": "locked_locomo_lj",
            "score": score,
            "correct": bool(score),
            "pass": bool(score),
        }
        name = str(row["category_name"])
        by_category[name].append(score)
        by_unit[str(row["unit_id"])].append(score)
    if {name: len(values) for name, values in by_category.items()} != LOCOMO_CATEGORY_COUNTS:
        raise ScreeningAnalysisError("LoCoMo category inventory differs")
    overall = statistics.mean(row["quality"]["score"] for row in ordered)
    _assert_score(evaluation.get("overall"), overall, label="LoCoMo overall")
    stored_categories = evaluation.get("by_category")
    if not isinstance(stored_categories, Mapping) or set(stored_categories) != set(
        LOCOMO_CATEGORY_ORDER
    ):
        raise ScreeningAnalysisError("LoCoMo category metrics differ")
    category_means = {
        name: statistics.mean(by_category[name]) for name in LOCOMO_CATEGORY_ORDER
    }
    for name, value in category_means.items():
        _assert_score(stored_categories.get(name), value, label=f"LoCoMo {name}")
    return (
        {
            "n": 314,
            "overall_lj": overall,
            "by_category": category_means,
            "category_counts": dict(LOCOMO_CATEGORY_COUNTS),
            "by_unit": {
                unit: statistics.mean(by_unit[unit]) for unit in EXPECTED_UNITS["locomo"]
            },
        },
        inputs,
    )


def _lme_score(
    *, score_root: Path, tier: str, rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[Path]]:
    path = (
        score_root
        / tier
        / "longmemeval-s/hypotheses.jsonl.eval-results-gpt-4o-mini"
    )
    values = _read_jsonl(path)
    if len(values) != 2 or {str(value.get("question_id")) for value in values} != set(
        LME_IDS
    ):
        raise ScreeningAnalysisError("LongMemEval score inventory differs")
    qa = {
        row["original_question_id"]: row
        for row in rows
        if row["benchmark"] == "longmemeval-s"
    }
    items: list[dict[str, Any]] = []
    for value in values:
        question_id = str(value["question_id"])
        row = qa.get(question_id)
        label = value.get("autoeval_label")
        correct = label.get("label") if isinstance(label, Mapping) else None
        if (
            row is None
            or value.get("hypothesis") != row["_join"]["answer"]
            or not isinstance(correct, bool)
        ):
            raise ScreeningAnalysisError("LongMemEval score and QA record differ")
        model = label.get("model")
        if model not in {"gpt-4o-mini-2024-07-18", "gpt-4o-mini"}:
            raise ScreeningAnalysisError("LongMemEval judge model differs")
        row["quality"] = {
            "metric": "official_longmemeval_qa_boolean",
            "score": int(correct),
            "correct": correct,
            "pass": correct,
        }
        items.append({"question_id": question_id, "correct": correct})
    items.sort(key=lambda item: LME_IDS.index(item["question_id"]))
    return {
        "n": 2,
        "correct": sum(item["correct"] for item in items),
        "items": items,
    }, [path]


def _beam_group(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores: list[float] = []
    nuggets = 0
    for item in evaluations:
        stored_score = _number(item.get("score"), label="BEAM score")
        nugget_rows = item.get("nugget_scores")
        if not isinstance(nugget_rows, list) or not nugget_rows:
            raise ScreeningAnalysisError("BEAM nugget scores are invalid")
        nugget_scores: list[float] = []
        for nugget in nugget_rows:
            if not isinstance(nugget, Mapping):
                raise ScreeningAnalysisError("BEAM nugget score row is invalid")
            score = _number(nugget.get("score"), label="BEAM nugget score")
            if not 0 <= score <= 1:
                raise ScreeningAnalysisError("BEAM nugget score is outside [0,1]")
            nugget_scores.append(score)
        recomputed_score = statistics.mean(nugget_scores)
        if not _close(stored_score, recomputed_score):
            raise ScreeningAnalysisError("BEAM score and nugget scores differ")
        scores.append(recomputed_score)
        nuggets += len(nugget_scores)
    passed = sum(score >= 0.5 for score in scores)
    return {
        "questions": len(scores),
        "rubric_nuggets": nuggets,
        "avg_score": statistics.mean(scores) if scores else None,
        "pass_threshold": 0.5,
        "passed": passed,
        "pass_rate": passed / len(scores) if scores else None,
        "pass_accuracy_percent": passed / len(scores) * 100 if scores else None,
    }


def _validate_beam_metrics(
    metrics: object, evaluations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        raise ScreeningAnalysisError("BEAM metrics are invalid")
    by_type: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in evaluations:
        by_type[str(item.get("question_type"))].append(item)
    if {name: len(values) for name, values in by_type.items()} != {
        name: 4 for name in BEAM_QUESTION_TYPES
    }:
        raise ScreeningAnalysisError("BEAM question-type inventory differs")
    recomputed = {
        "overall": _beam_group(evaluations),
        "by_question_type": {
            name: _beam_group(by_type[name]) for name in BEAM_QUESTION_TYPES
        },
    }
    if recomputed["overall"]["rubric_nuggets"] != 103:
        raise ScreeningAnalysisError("BEAM rubric-nugget inventory differs")
    stored_overall = metrics.get("overall")
    stored_types = metrics.get("by_question_type")
    if not isinstance(stored_overall, Mapping) or not isinstance(stored_types, Mapping):
        raise ScreeningAnalysisError("BEAM stored metrics are incomplete")
    for group, expected in (
        (stored_overall, recomputed["overall"]),
        *(
            (stored_types.get(name), recomputed["by_question_type"][name])
            for name in BEAM_QUESTION_TYPES
        ),
    ):
        if not isinstance(group, Mapping):
            raise ScreeningAnalysisError("BEAM metric group is absent")
        for key, value in expected.items():
            observed = group.get(key)
            if isinstance(value, float):
                _assert_score(observed, value, label=f"BEAM {key}")
            elif observed != value:
                raise ScreeningAnalysisError(f"BEAM {key} does not recompute")
    return recomputed


def _beam_score(
    *, score_root: Path, tier: str, rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[Path], dict[str, str]]:
    root = score_root / "beam-semantic" / tier
    metrics_path = root / "metrics.json"
    results_path = root / "results.json"
    audit_path = root / "evaluation_audit.json"
    metrics = _read_json(metrics_path)
    results = _read_json(results_path)
    audit = _read_json(audit_path)
    if not isinstance(results, Mapping) or not isinstance(audit, Mapping):
        raise ScreeningAnalysisError("BEAM result or audit is invalid")
    if (
        audit.get("status") != "passed"
        or audit.get("tier") != tier
        or audit.get("question_count") != 40
        or audit.get("rubric_nugget_count") != 103
        or audit.get("results_sha256") != sha256_file(results_path)
        or audit.get("metrics_sha256") != sha256_file(metrics_path)
        or results.get("status") != "complete"
        or results.get("question_count") != 40
        or results.get("metrics") != metrics
    ):
        raise ScreeningAnalysisError(f"{tier} BEAM audit did not pass all gates")
    evaluations = results.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 40:
        raise ScreeningAnalysisError("BEAM evaluation inventory differs")
    typed = [item for item in evaluations if isinstance(item, Mapping)]
    if len(typed) != 40:
        raise ScreeningAnalysisError("BEAM evaluation row is invalid")
    recomputed = _validate_beam_metrics(metrics, typed)
    qa = {
        row["original_question_id"]: row
        for row in rows
        if row["benchmark"] == "beam-100k"
    }
    by_unit: defaultdict[str, list[float]] = defaultdict(list)
    seen_source_ids: set[str] = set()
    for value in typed:
        source_id = value.get("source_question_id")
        row = qa.get(str(source_id))
        if (
            row is None
            or value.get("question") != row["_join"]["question"]
            or value.get("question_type") != row["question_type"]
            or value.get("answer") != row["_join"]["answer"]
            or value.get("status") != "complete"
        ):
            raise ScreeningAnalysisError("BEAM score and QA record differ")
        if str(source_id) in seen_source_ids:
            raise ScreeningAnalysisError("BEAM source question is duplicated")
        seen_source_ids.add(str(source_id))
        score = _number(value.get("score"), label="BEAM semantic score")
        if not 0 <= score <= 1:
            raise ScreeningAnalysisError("BEAM semantic score is outside [0,1]")
        passed = score >= 0.5
        row["quality"] = {
            "metric": "beam_rubric_nugget_mean",
            "score": score,
            "correct": None,
            "pass": passed,
        }
        by_unit[str(row["unit_id"])].append(score)
    if seen_source_ids != set(qa):
        raise ScreeningAnalysisError("BEAM source-question inventory differs")
    by_type = {
        name: {
            "n": recomputed["by_question_type"][name]["questions"],
            "rubric_nuggets": recomputed["by_question_type"][name][
                "rubric_nuggets"
            ],
            "avg_score": recomputed["by_question_type"][name]["avg_score"],
            "pass_rate": recomputed["by_question_type"][name]["pass_rate"],
        }
        for name in BEAM_QUESTION_TYPES
    }
    summary = {
        "n": 40,
        "rubric_nuggets": 103,
        "avg_score": recomputed["overall"]["avg_score"],
        "pass_rate": recomputed["overall"]["pass_rate"],
        "by_question_type": by_type,
        "by_conversation": {
            unit: {
                "n": len(by_unit[unit]),
                "avg_score": statistics.mean(by_unit[unit]),
            }
            for unit in EXPECTED_UNITS["beam-100k"]
        },
    }
    comparable = {
        "paired_screening_inventory_sha256": str(
            audit.get("paired_screening_inventory_sha256")
        ),
        "judge_procedure_sha256": str(audit.get("judge_procedure_sha256")),
    }
    if any(value in {"", "None"} for value in comparable.values()):
        raise ScreeningAnalysisError("BEAM pairing hashes are absent")
    return summary, [metrics_path, results_path, audit_path], comparable


def load_scores(
    *, score_root: Path, qa: Mapping[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    quality: dict[str, Any] = {}
    inputs: dict[str, list[Path]] = {}
    beam_comparability: dict[str, dict[str, str]] = {}
    for tier in TIERS:
        rows = qa[tier]["rows"]
        locomo, locomo_inputs = _locomo_score(
            score_root=score_root, tier=tier, rows=rows
        )
        lme, lme_inputs = _lme_score(score_root=score_root, tier=tier, rows=rows)
        beam, beam_inputs, comparable = _beam_score(
            score_root=score_root, tier=tier, rows=rows
        )
        if any(row["quality"]["metric"] is None for row in rows):
            raise ScreeningAnalysisError(f"{tier} contains unscored QA records")
        quality[tier] = {
            "locomo": locomo,
            "beam_100k": beam,
            "longmemeval_s": lme,
        }
        inputs[tier] = [*locomo_inputs, *lme_inputs, *beam_inputs]
        beam_comparability[tier] = comparable
    if beam_comparability["terra"] != beam_comparability["sol"]:
        raise ScreeningAnalysisError("BEAM tier judge procedures are not paired")
    return quality, inputs


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _stats(values: Iterable[object]) -> dict[str, Any]:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    return {
        "n": len(numeric),
        "total": sum(numeric),
        "mean": statistics.mean(numeric) if numeric else None,
        "median": statistics.median(numeric) if numeric else None,
        "p95": _percentile(numeric, 0.95),
    }


ACCESS_METRICS = (
    "read_calls",
    "tool_calls",
    "visible_tokens",
    "source_resolution_tokens",
    "retrieval_model_calls",
    "retrieval_latency_s",
    "tool_latency_s",
    "retrieval_prompt_tokens",
    "retrieval_cached_input_tokens",
    "retrieval_uncached_input_tokens",
    "retrieval_completion_tokens",
    "answer_prompt_tokens",
    "answer_cached_input_tokens",
    "answer_uncached_input_tokens",
    "answer_completion_tokens",
    "answer_latency_s",
    "accepted_logical_calls",
    "accepted_client_http_attempts",
    "accepted_upstream_http_attempts",
    "retry_count",
)


def aggregate_access(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {
        name: _stats(
            row.get("access", {}).get(name)
            for row in rows
            if isinstance(row.get("access"), Mapping)
        )
        for name in ACCESS_METRICS
    }
    eligible = [
        row
        for row in rows
        if isinstance(row.get("access"), Mapping)
        and row["access"].get("source_recall_eligible") is True
    ]
    gold_total = sum(int(row["access"]["gold_source_count"]) for row in eligible)
    hit_total = sum(int(row["access"]["mapped_source_hits"]) for row in eligible)
    recalls = [float(row["access"]["mapped_source_recall"]) for row in eligible]
    first_hits = [bool(row["access"]["first_file_hit"]) for row in eligible]
    budget_hits = [bool(row.get("access", {}).get("budget_hit")) for row in rows]
    return {
        "questions": len(rows),
        "metrics": metrics,
        "mapped_source_recall": {
            "availability": "available" if eligible else "not_available",
            "eligible_questions": len(eligible),
            "gold_source_ids": gold_total,
            "mapped_source_hits": hit_total,
            "micro": _ratio(hit_total, gold_total),
            "macro": statistics.mean(recalls) if recalls else None,
        },
        "first_opened_file": {
            "availability": "available" if eligible else "not_available",
            "eligible_questions": len(eligible),
            "hits": sum(first_hits),
            "hit_rate": statistics.mean(first_hits) if first_hits else None,
            "first_relevant_read_rank": _stats(
                row["access"].get("first_relevant_read_rank") for row in eligible
            ),
        },
        "budget": {
            "hits": sum(budget_hits),
            "questions": len(rows),
            "hit_rate": statistics.mean(budget_hits) if budget_hits else None,
        },
        "retries": {
            "total": sum(int(row.get("access", {}).get("retry_count", 0)) for row in rows),
            "questions_with_retry": sum(
                int(row.get("access", {}).get("retry_count", 0)) > 0 for row in rows
            ),
        },
    }


def _quality_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    terra = [float(row["terra"]["quality"]["score"]) for row in rows]
    sol = [float(row["sol"]["quality"]["score"]) for row in rows]
    return {
        "n": len(rows),
        "terra_mean": statistics.mean(terra),
        "sol_mean": statistics.mean(sol),
        "delta": statistics.mean(sol) - statistics.mean(terra),
    }


def _contingency(
    rows: Sequence[Mapping[str, Any]], *, field: str
) -> dict[str, int]:
    result = {"both": 0, "neither": 0, "sol_only": 0, "terra_only": 0}
    for row in rows:
        terra = bool(row["terra"]["quality"][field])
        sol = bool(row["sol"]["quality"][field])
        key = (
            "both"
            if terra and sol
            else "sol_only"
            if sol
            else "terra_only"
            if terra
            else "neither"
        )
        result[key] += 1
    return result


def _paired_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    triples: list[tuple[float, float, float]] = []
    for row in rows:
        terra = row["terra"]["access"].get(metric)
        sol = row["sol"]["access"].get(metric)
        if (
            isinstance(terra, (int, float))
            and not isinstance(terra, bool)
            and isinstance(sol, (int, float))
            and not isinstance(sol, bool)
        ):
            triples.append((float(terra), float(sol), float(sol) - float(terra)))
    return {
        "n": len(triples),
        "terra": _stats(value[0] for value in triples),
        "sol": _stats(value[1] for value in triples),
        "delta": _stats(value[2] for value in triples),
    }


def build_paired_results(
    rows: Sequence[Mapping[str, Any]], build: Mapping[str, Any]
) -> dict[str, Any]:
    by_key: defaultdict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["benchmark"]), str(row["unit_id"]), str(row["original_question_id"]))
        tier = str(row["tier"])
        if tier in by_key[key]:
            raise ScreeningAnalysisError(f"duplicate paired tier row: {key}/{tier}")
        by_key[key][tier] = row
    if len(by_key) != EXPECTED_QUESTIONS_PER_TIER or any(
        set(value) != set(TIERS) for value in by_key.values()
    ):
        raise ScreeningAnalysisError("Terra/Sol question pairing is incomplete")
    identity_fields = (
        "question_id",
        "question_sha256",
        "category_id",
        "category_name",
        "question_type",
    )
    join_fields = ("question", "gold", "rubric", "selection")
    for key, tiers in by_key.items():
        terra = tiers["terra"]
        sol = tiers["sol"]
        terra_join = terra.get("_join")
        sol_join = sol.get("_join")
        if (
            any(terra.get(field) != sol.get(field) for field in identity_fields)
            or not isinstance(terra_join, Mapping)
            or not isinstance(sol_join, Mapping)
            or any(terra_join.get(field) != sol_join.get(field) for field in join_fields)
        ):
            raise ScreeningAnalysisError(f"Terra/Sol pair identity differs: {key}")
    pairs = [
        {"key": key, "terra": value["terra"], "sol": value["sol"]}
        for key, value in sorted(by_key.items())
    ]
    locomo = [row for row in pairs if row["key"][0] == "locomo"]
    beam = [row for row in pairs if row["key"][0] == "beam-100k"]
    lme = [row for row in pairs if row["key"][0] == "longmemeval-s"]

    locomo_summary = {
        **_quality_group(locomo),
        **{
            "both_correct": _contingency(locomo, field="correct")["both"],
            "both_wrong": _contingency(locomo, field="correct")["neither"],
            "sol_only": _contingency(locomo, field="correct")["sol_only"],
            "terra_only": _contingency(locomo, field="correct")["terra_only"],
        },
        "by_category": {
            name: _quality_group(
                [row for row in locomo if row["terra"]["category_name"] == name]
            )
            for name in LOCOMO_CATEGORY_ORDER
        },
        "by_unit": {
            unit: _quality_group(
                [row for row in locomo if row["key"][1] == unit]
            )
            for unit in EXPECTED_UNITS["locomo"]
        },
    }
    beam_summary = {
        "n": len(beam),
        "terra_mean_score": _quality_group(beam)["terra_mean"],
        "sol_mean_score": _quality_group(beam)["sol_mean"],
        "delta": _quality_group(beam)["delta"],
        "terra_pass_rate": statistics.mean(
            bool(row["terra"]["quality"]["pass"]) for row in beam
        ),
        "sol_pass_rate": statistics.mean(
            bool(row["sol"]["quality"]["pass"]) for row in beam
        ),
        "pass_contingency": _contingency(beam, field="pass"),
        "by_question_type": {
            name: _quality_group(
                [row for row in beam if row["terra"]["question_type"] == name]
            )
            for name in BEAM_QUESTION_TYPES
        },
        "by_unit": {
            unit: _quality_group([row for row in beam if row["key"][1] == unit])
            for unit in EXPECTED_UNITS["beam-100k"]
        },
        "items": [
            {
                "unit_id": row["key"][1],
                "question_id": row["key"][2],
                "terra_score": row["terra"]["quality"]["score"],
                "sol_score": row["sol"]["quality"]["score"],
                "delta": row["sol"]["quality"]["score"]
                - row["terra"]["quality"]["score"],
            }
            for row in beam
        ],
    }
    lme_items = [
        {
            "question_id": row["key"][2],
            "terra_correct": bool(row["terra"]["quality"]["correct"]),
            "sol_correct": bool(row["sol"]["quality"]["correct"]),
        }
        for row in lme
    ]
    lme_summary = {
        "n": len(lme),
        "terra_correct": sum(item["terra_correct"] for item in lme_items),
        "sol_correct": sum(item["sol_correct"] for item in lme_items),
        "delta": sum(item["sol_correct"] for item in lme_items)
        - sum(item["terra_correct"] for item in lme_items),
        "items": lme_items,
    }
    access_groups = {
        "all": pairs,
        **{
            benchmark.replace("-", "_"): [
                row for row in pairs if row["key"][0] == benchmark
            ]
            for benchmark in BENCHMARKS
        },
    }
    access = {
        name: {metric: _paired_metric(group, metric) for metric in ACCESS_METRICS}
        for name, group in access_groups.items()
    }
    build_pairs: dict[str, Any] = {}
    build_metrics = (
        "calls_per_100_messages",
        "input_tokens_per_1000_source_tokens",
        "output_tokens_per_1000_source_tokens",
        "wall_minutes_per_100_messages",
    )
    for metric in build_metrics:
        terra = float(build["terra"]["normalized"][metric])
        sol = float(build["sol"]["normalized"][metric])
        build_pairs[metric] = {
            "terra": terra,
            "sol": sol,
            "delta": sol - terra,
            "relative_delta": (sol - terra) / terra if terra else None,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "tier_order": list(TIERS),
        "delta_definition": "sol_minus_terra",
        "pairing_key": ["benchmark", "unit_id", "original_question_id"],
        "pair_count": len(pairs),
        "quality": {
            "locomo": locomo_summary,
            "beam_100k": beam_summary,
            "longmemeval_s": lme_summary,
        },
        "access": access,
        "build": build_pairs,
        "inference": {
            "status": "descriptive_only",
            "reason": (
                "screening contains only two conversations per QA benchmark; "
                "no benchmark-generalized clustered confidence interval"
            ),
        },
    }


def _clean_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "_join"}
        for row in sorted(
            rows,
            key=lambda row: (
                str(row["benchmark"]),
                str(row["unit_id"]),
                str(row["original_question_id"]),
                TIERS.index(str(row["tier"])),
            ),
        )
    ]


def _format_percent(value: object, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.{digits}f}"


def _format_number(value: object, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
    }
    return "".join(replacements.get(character, character) for character in value)


def _best(value: str, selected: bool) -> str:
    return f"\\textbf{{{value}}}" if selected else value


def _main_table(summary: Mapping[str, Any]) -> str:
    values: dict[str, dict[str, float]] = {}
    for tier in TIERS:
        quality = summary["quality"][tier]
        build = summary["build"][tier]["normalized"]
        values[tier] = {
            "locomo": quality["locomo"]["overall_lj"] * 100,
            "beam": quality["beam_100k"]["avg_score"] * 100,
            "lme": quality["longmemeval_s"]["correct"],
            "calls": build["calls_per_100_messages"],
            "input": build["input_tokens_per_1000_source_tokens"],
        }
    lines = []
    for tier in TIERS:
        current = values[tier]
        other = values["sol" if tier == "terra" else "terra"]
        cells = [tier.capitalize()]
        for key, digits, higher in (
            ("locomo", 1, True),
            ("beam", 1, True),
            ("lme", 0, True),
            ("calls", 2, False),
            ("input", 1, False),
        ):
            formatted = f"{current[key]:.{digits}f}"
            is_best = current[key] >= other[key] if higher else current[key] <= other[key]
            cells.append(_best(formatted, is_best))
        cells[3] = cells[3] + "/2"
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            (
                "\\caption{Fixed $W=32$ writer-capability screening. LoCoMo and "
                "BEAM each contain only two conversations, and LongMemEval-S (LME-S) "
                "contains two items; these are not full-benchmark estimates.}"
            ),
            "\\label{tab:gpt56-w32-main}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            (
                "Writer & \\shortstack{LoCoMo LJ\\\\(\\%)} & "
                "\\shortstack{BEAM rubric\\\\(\\%)} & "
                "\\shortstack{LME-S\\\\correct (x/2)} & "
                "\\shortstack{Build calls\\\\/100 msg} & "
                "\\shortstack{Build input tok\\\\/1K source} \\\\"
            ),
            "\\midrule",
            *lines,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )


def _quality_breakdown_table(summary: Mapping[str, Any]) -> str:
    rows: list[str] = []

    def percent_row(label: str, count: int, terra: float, sol: float) -> None:
        rows.append(
            f"{_latex_escape(label)} & {count} & {terra * 100:.1f} & "
            f"{sol * 100:.1f} & {(sol - terra) * 100:+.1f} \\\\"
        )

    percent_row(
        "LoCoMo overall",
        314,
        summary["quality"]["terra"]["locomo"]["overall_lj"],
        summary["quality"]["sol"]["locomo"]["overall_lj"],
    )
    for name in LOCOMO_CATEGORY_ORDER:
        percent_row(
            f"LoCoMo: {name}",
            LOCOMO_CATEGORY_COUNTS[name],
            summary["quality"]["terra"]["locomo"]["by_category"][name],
            summary["quality"]["sol"]["locomo"]["by_category"][name],
        )
    rows.append("\\midrule")
    percent_row(
        "BEAM overall",
        40,
        summary["quality"]["terra"]["beam_100k"]["avg_score"],
        summary["quality"]["sol"]["beam_100k"]["avg_score"],
    )
    for name in BEAM_QUESTION_TYPES:
        percent_row(
            f"BEAM: {name}",
            4,
            summary["quality"]["terra"]["beam_100k"]["by_question_type"][name][
                "avg_score"
            ],
            summary["quality"]["sol"]["beam_100k"]["by_question_type"][name][
                "avg_score"
            ],
        )
    rows.append("\\midrule")
    terra_items = {
        item["question_id"]: item["correct"]
        for item in summary["quality"]["terra"]["longmemeval_s"]["items"]
    }
    sol_items = {
        item["question_id"]: item["correct"]
        for item in summary["quality"]["sol"]["longmemeval_s"]["items"]
    }
    for question_id in LME_IDS:
        terra = int(terra_items[question_id])
        sol = int(sol_items[question_id])
        rows.append(
            f"LME-S: {_latex_escape(question_id)} & 1 & {terra} & {sol} & {sol - terra:+d} \\\\"
        )
    terra_total = summary["quality"]["terra"]["longmemeval_s"]["correct"]
    sol_total = summary["quality"]["sol"]["longmemeval_s"]["correct"]
    rows.append(
        f"LME-S total & 2 & {terra_total}/2 & {sol_total}/2 & {sol_total - terra_total:+d} \\\\"
    )
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            (
                "\\caption{Quality breakdown for the fixed $W=32$ paired screening. "
                "LoCoMo reports the locked LJ correctness rate, whereas BEAM reports "
                "the rubric-nugget mean; the two metrics are not averaged across "
                "benchmarks. LME-S reports binary outcomes for two items without a "
                "confidence interval.}"
            ),
            "\\label{tab:gpt56-w32-quality-breakdown}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Scope & $n$ & Terra & Sol & Sol$-$Terra \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )


def _access_value(summary: Mapping[str, Any], tier: str, path: Sequence[str]) -> Any:
    value: Any = summary["qa_access"][tier]["overall"]
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _resource_breakdown_table(summary: Mapping[str, Any]) -> str:
    rows = ["\\multicolumn{3}{l}{\\textit{Panel A: QA access}} \\\\ "]

    def access_row(label: str, path: Sequence[str], digits: int = 2) -> None:
        terra = _access_value(summary, "terra", path)
        sol = _access_value(summary, "sol", path)
        rows.append(
            f"{label} & {_format_number(terra, digits)} & {_format_number(sol, digits)} \\\\"
        )

    access_row("Mapped-source recall, micro (LoCoMo)", ("mapped_source_recall", "micro"), 3)
    access_row("Mapped-source recall, macro (LoCoMo)", ("mapped_source_recall", "macro"), 3)
    access_row("First-opened-file hit rate (LoCoMo)", ("first_opened_file", "hit_rate"), 3)
    access_row("Retrieval model calls/question", ("metrics", "retrieval_model_calls", "mean"))
    access_row("Tool calls/question", ("metrics", "tool_calls", "mean"))
    access_row("Read calls/question", ("metrics", "read_calls", "mean"))
    access_row("Visible tokens/question", ("metrics", "visible_tokens", "mean"), 1)
    access_row("Retrieval prompt tokens/question", ("metrics", "retrieval_prompt_tokens", "mean"), 1)
    access_row("Answer prompt tokens/question", ("metrics", "answer_prompt_tokens", "mean"), 1)
    access_row("Retrieval completion tokens/question", ("metrics", "retrieval_completion_tokens", "mean"), 1)
    access_row("Answer completion tokens/question", ("metrics", "answer_completion_tokens", "mean"), 1)
    access_row("Retrieval latency/question (s)", ("metrics", "retrieval_latency_s", "mean"))
    access_row("Budget hits", ("budget", "hits"), 0)
    access_row("Retry count", ("retries", "total"), 0)
    rows.extend(["\\midrule", "\\multicolumn{3}{l}{\\textit{Panel B: memory construction}} \\\\ "])

    build_metrics = (
        ("Calls/100 messages", "normalized", "calls_per_100_messages", 2),
        ("Input tokens/1K source tokens", "normalized", "input_tokens_per_1000_source_tokens", 1),
        ("Output tokens/1K source tokens", "normalized", "output_tokens_per_1000_source_tokens", 1),
        ("Minutes/100 messages", "normalized", "wall_minutes_per_100_messages", 2),
        ("First-pass source coverage", "grounding", "first_pass_source_coverage", 3),
        ("Post-verify source coverage", "grounding", "post_verify_source_coverage", 3),
        ("Reference precision", "grounding", "reference_precision", 3),
        ("Abstract/source byte ratio", "representation", "abstract_to_source_byte_ratio", 3),
        ("Abstract files", "representation", "abstract_files", 0),
        ("Abstract entries", "representation", "abstract_entries", 0),
    )
    for label, group, key, digits in build_metrics:
        terra = summary["build"]["terra"][group].get(key)
        sol = summary["build"]["sol"][group].get(key)
        rows.append(
            f"{label} & {_format_number(terra, digits)} & {_format_number(sol, digits)} \\\\"
        )
    for group in ("writer", "verify", "maintenance"):
        for field, label in (
            ("calls", "calls"),
            ("input_tokens", "input tokens"),
            ("output_tokens", "output tokens"),
        ):
            terra = summary["build"]["terra"]["phase_groups"][group][field]
            sol = summary["build"]["sol"]["phase_groups"][group][field]
            rows.append(
                f"{group.capitalize()} {label} & {_format_number(terra, 0)} & {_format_number(sol, 0)} \\\\"
            )
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            (
                "\\caption{Resource breakdown for the fixed $W=32$ screening. "
                "Mapped-source and first-file metrics are defined only for eligible "
                "LoCoMo questions; structurally unavailable values are shown as N/A. "
                "All QA resource values use the accepted execution, while retry "
                "counts report separate question-level overhead.}"
            ),
            "\\label{tab:gpt56-w32-resources}",
            "\\begin{tabular}{lrr}",
            "\\toprule",
            "Metric & Terra & Sol \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )


def _unit_build_values(summary: Mapping[str, Any], metric: str) -> list[tuple[float, float]]:
    terra = {
        (row["benchmark"], row["unit_id"]): row[metric]
        for row in summary["build"]["terra"]["units"]
    }
    sol = {
        (row["benchmark"], row["unit_id"]): row[metric]
        for row in summary["build"]["sol"]["units"]
    }
    if set(terra) != set(sol):
        raise ScreeningAnalysisError("build unit pairing differs")
    return [(float(terra[key]), float(sol[key])) for key in sorted(terra)]


def _quality_unit_values(
    summary: Mapping[str, Any], benchmark: str
) -> list[tuple[float, float]]:
    key = "locomo" if benchmark == "locomo" else "beam_100k"
    field = "by_unit" if key == "locomo" else "by_conversation"
    metric = None if key == "locomo" else "avg_score"
    terra = summary["quality"]["terra"][key][field]
    sol = summary["quality"]["sol"][key][field]
    pairs: list[tuple[float, float]] = []
    for unit in sorted(terra):
        left = terra[unit] if metric is None else terra[unit][metric]
        right = sol[unit] if metric is None else sol[unit][metric]
        pairs.append((float(left) * 100, float(right) * 100))
    return pairs


def _figure_panels(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "tag": "(a)",
            "ylabel": "LoCoMo LJ (%)",
            "pairs": _quality_unit_values(summary, "locomo"),
            "aggregate": (
                summary["quality"]["terra"]["locomo"]["overall_lj"] * 100,
                summary["quality"]["sol"]["locomo"]["overall_lj"] * 100,
            ),
        },
        {
            "tag": "(b)",
            "ylabel": "BEAM rubric score (%)",
            "pairs": _quality_unit_values(summary, "beam-100k"),
            "aggregate": (
                summary["quality"]["terra"]["beam_100k"]["avg_score"] * 100,
                summary["quality"]["sol"]["beam_100k"]["avg_score"] * 100,
            ),
        },
        {
            "tag": "(c)",
            "ylabel": "Build calls / 100 messages",
            "pairs": _unit_build_values(summary, "calls_per_100_messages"),
            "aggregate": tuple(
                summary["build"][tier]["normalized"]["calls_per_100_messages"]
                for tier in TIERS
            ),
        },
        {
            "tag": "(d)",
            "ylabel": "Build input tokens / 1K source tokens",
            "pairs": _unit_build_values(
                summary, "input_tokens_per_1000_source_tokens"
            ),
            "aggregate": tuple(
                summary["build"][tier]["normalized"][
                    "input_tokens_per_1000_source_tokens"
                ]
                for tier in TIERS
            ),
        },
    ]


def _render_pdf(path: Path, panels: Sequence[Mapping[str, Any]]) -> None:
    try:
        import reportlab
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise ScreeningAnalysisError("reportlab is required for PDF output") from exc
    regular_candidates = (
        Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
        Path(reportlab.__file__).resolve().parent / "fonts/Vera.ttf",
    )
    bold_candidates = (
        Path("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"),
        Path(reportlab.__file__).resolve().parent / "fonts/VeraBd.ttf",
    )
    regular_path = next((item for item in regular_candidates if item.is_file()), None)
    bold_path = next((item for item in bold_candidates if item.is_file()), None)
    if regular_path is None or bold_path is None:
        raise ScreeningAnalysisError("embedded TrueType plotting fonts are unavailable")
    pdfmetrics.registerFont(TTFont("W32PlotSerif", str(regular_path)))
    pdfmetrics.registerFont(TTFont("W32PlotSerifBold", str(bold_path)))
    width, height = 540.0, 360.0
    output = path.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".pdf", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        drawing = canvas.Canvas(
            str(temporary),
            pagesize=(width, height),
            invariant=1,
            initialFontName="W32PlotSerif",
            initialFontSize=8,
            initialLeading=9.6,
        )
        drawing.setTitle("")
        _draw_reportlab_panels(
            drawing,
            panels,
            width=width,
            height=height,
            regular_font="W32PlotSerif",
            bold_font="W32PlotSerifBold",
        )
        drawing.save()
        _write_no_clobber(output, temporary.read_bytes())
    finally:
        temporary.unlink(missing_ok=True)


def _axis_range(panel: Mapping[str, Any]) -> tuple[float, float]:
    values = [float(value) for pair in panel["pairs"] for value in pair]
    values.extend(float(value) for value in panel["aggregate"])
    low, high = min(values), max(values)
    if panel["tag"] in {"(a)", "(b)"}:
        low = max(0.0, math.floor((low - 5) / 10) * 10)
        high = min(100.0, math.ceil((high + 5) / 10) * 10)
    else:
        padding = max((high - low) * 0.2, high * 0.05, 1.0)
        low = max(0.0, low - padding)
        high += padding
    if high <= low:
        high = low + 1.0
    return low, high


def _draw_reportlab_panels(
    drawing: Any,
    panels: Sequence[Mapping[str, Any]],
    *,
    width: float,
    height: float,
    regular_font: str,
    bold_font: str,
) -> None:
    blue = (0.0, 0.447, 0.698)
    orange = (0.902, 0.624, 0.0)
    for index, panel in enumerate(panels):
        column, row = index % 2, index // 2
        left = 55 + column * 265
        bottom = height - 165 - row * 170
        plot_width, plot_height = 185, 115
        low, high = _axis_range(panel)

        def y(value: float) -> float:
            return bottom + (value - low) / (high - low) * plot_height

        drawing.setStrokeColorRGB(0.2, 0.2, 0.2)
        drawing.setLineWidth(0.7)
        drawing.line(left, bottom, left, bottom + plot_height)
        drawing.line(left, bottom, left + plot_width, bottom)
        for tick_index in range(5):
            value = low + (high - low) * tick_index / 4
            position = y(value)
            drawing.setStrokeColorRGB(0.88, 0.88, 0.88)
            drawing.line(left, position, left + plot_width, position)
            drawing.setFillColorRGB(0.2, 0.2, 0.2)
            drawing.setFont(regular_font, 7)
            label = f"{value:.0f}" if high >= 20 else f"{value:.1f}"
            drawing.drawRightString(left - 5, position - 2, label)
        x_terra, x_sol = left + 55, left + 135
        drawing.setStrokeColorRGB(0.72, 0.72, 0.72)
        drawing.setLineWidth(0.45)
        for terra, sol in panel["pairs"]:
            drawing.line(x_terra, y(float(terra)), x_sol, y(float(sol)))
        drawing.setFillColorRGB(*blue)
        drawing.circle(x_terra, y(float(panel["aggregate"][0])), 4.0, fill=1, stroke=0)
        drawing.setFillColorRGB(*orange)
        sol_y = y(float(panel["aggregate"][1]))
        drawing.rect(x_sol - 4, sol_y - 4, 8, 8, fill=1, stroke=0)
        drawing.setFillColorRGB(0.1, 0.1, 0.1)
        drawing.setFont(regular_font, 8)
        drawing.drawCentredString(x_terra, bottom - 13, "Terra")
        drawing.drawCentredString(x_sol, bottom - 13, "Sol")
        drawing.setFont(bold_font, 9)
        drawing.drawString(left, bottom + plot_height + 7, str(panel["tag"]))
        drawing.saveState()
        drawing.translate(left - 37, bottom + plot_height / 2)
        drawing.rotate(90)
        drawing.setFont(regular_font, 8)
        drawing.drawCentredString(0, 0, str(panel["ylabel"]))
        drawing.restoreState()


def _render_png(path: Path, panels: Sequence[Mapping[str, Any]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ScreeningAnalysisError("Pillow is required for PNG output") from exc
    scale = 4
    width, height = 540 * scale, 360 * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_candidates = (
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    )

    def font(size: int, *, bold: bool = False) -> Any:
        candidates = (
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        ) if bold else font_candidates
        for candidate in candidates:
            if Path(candidate).is_file():
                return ImageFont.truetype(candidate, size * scale)
        return ImageFont.load_default()

    normal = font(8)
    bold_font = font(9, bold=True)
    blue, orange, gray, dark = "#0072B2", "#E69F00", "#B5B5B5", "#222222"
    for index, panel in enumerate(panels):
        column, row = index % 2, index // 2
        left = (55 + column * 265) * scale
        bottom = (360 - 165 - row * 170) * scale
        plot_width, plot_height = 185 * scale, 115 * scale
        low, high = _axis_range(panel)

        def y(value: float) -> int:
            return round(bottom + plot_height - (value - low) / (high - low) * plot_height)

        top = bottom
        base = bottom + plot_height
        for tick_index in range(5):
            value = low + (high - low) * tick_index / 4
            position = y(value)
            draw.line((left, position, left + plot_width, position), fill="#E0E0E0", width=1 * scale)
            label = f"{value:.0f}" if high >= 20 else f"{value:.1f}"
            box = draw.textbbox((0, 0), label, font=normal)
            draw.text((left - (box[2] - box[0]) - 5 * scale, position - 4 * scale), label, fill=dark, font=normal)
        draw.line((left, top, left, base), fill=dark, width=1 * scale)
        draw.line((left, base, left + plot_width, base), fill=dark, width=1 * scale)
        x_terra, x_sol = left + 55 * scale, left + 135 * scale
        for terra, sol in panel["pairs"]:
            draw.line((x_terra, y(float(terra)), x_sol, y(float(sol))), fill=gray, width=1 * scale)
        terra_y = y(float(panel["aggregate"][0]))
        sol_y = y(float(panel["aggregate"][1]))
        radius = 4 * scale
        draw.ellipse((x_terra - radius, terra_y - radius, x_terra + radius, terra_y + radius), fill=blue)
        draw.rectangle((x_sol - radius, sol_y - radius, x_sol + radius, sol_y + radius), fill=orange)
        for x, label in ((x_terra, "Terra"), (x_sol, "Sol")):
            box = draw.textbbox((0, 0), label, font=normal)
            draw.text((x - (box[2] - box[0]) / 2, base + 5 * scale), label, fill=dark, font=normal)
        draw.text((left, top - 14 * scale), str(panel["tag"]), fill=dark, font=bold_font)
        label = str(panel["ylabel"])
        label_box = draw.textbbox((0, 0), label, font=normal)
        label_image = Image.new(
            "RGBA",
            (label_box[2] - label_box[0] + 4 * scale, label_box[3] - label_box[1] + 4 * scale),
            (255, 255, 255, 0),
        )
        label_draw = ImageDraw.Draw(label_image)
        label_draw.text((2 * scale, 0), label, fill=dark, font=normal)
        rotated = label_image.rotate(90, expand=True)
        image.paste(
            rotated,
            (
                left - 43 * scale,
                round(top + plot_height / 2 - rotated.height / 2),
            ),
            rotated,
        )
    output = path.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".png", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="PNG", dpi=(300, 300), optimize=True)
        _write_no_clobber(output, temporary.read_bytes())
    finally:
        temporary.unlink(missing_ok=True)


def _paper_artifact_payloads(summary: Mapping[str, Any]) -> dict[str, bytes]:
    outputs: dict[str, bytes] = {
        "gpt56_w32_main_results.tex": _main_table(summary).encode("utf-8"),
        "gpt56_w32_quality_breakdown.tex": _quality_breakdown_table(summary).encode(
            "utf-8"
        ),
        "gpt56_w32_resource_breakdown.tex": _resource_breakdown_table(summary).encode(
            "utf-8"
        ),
    }
    panels = _figure_panels(summary)
    with tempfile.TemporaryDirectory(prefix="gpt56-w32-paper-") as temporary:
        staging = Path(temporary)
        pdf_path = staging / "gpt56_w32_quality_cost.pdf"
        png_path = staging / "gpt56_w32_quality_cost.png"
        _render_pdf(pdf_path, panels)
        _render_png(png_path, panels)
        outputs[pdf_path.name] = pdf_path.read_bytes()
        outputs[png_path.name] = png_path.read_bytes()
    return outputs


def render_paper_artifacts(
    *, summary: Mapping[str, Any], paired: Mapping[str, Any], paper_dir: Path
) -> list[Path]:
    del paired  # The tables are sourced from the already paired/validated summary.
    directory = paper_dir.expanduser().absolute()
    payloads = _paper_artifact_payloads(summary)
    targets = {directory / name: payload for name, payload in payloads.items()}
    _publish_bundle_no_clobber(targets)
    return [directory / name for name in payloads]


def _score_manifest(paths: Sequence[Path]) -> dict[str, Any]:
    by_name = {_display_path(path): _descriptor(path) for path in paths}
    return {"files": [by_name[name] for name in sorted(by_name)]}


def analyze_campaign(
    *,
    build_analysis_path: Path,
    qa_base: Path,
    score_root: Path,
    output_root: Path,
    paper_dir: Path,
) -> dict[str, Any]:
    build_report = _read_json(build_analysis_path)
    build = validate_and_aggregate_build(build_report)
    qa = {
        tier: load_qa_tier(qa_base / f"{tier}-r5", tier=tier) for tier in TIERS
    }
    quality, score_inputs = load_scores(score_root=score_root, qa=qa)
    rows = [row for tier in TIERS for row in qa[tier]["rows"]]
    paired = build_paired_results(rows, build)
    clean_rows = _clean_rows(rows)
    qa_access = {
        tier: {
            "overall": aggregate_access(
                [row for row in clean_rows if row["tier"] == tier]
            ),
            "by_benchmark": {
                benchmark: aggregate_access(
                    [
                        row
                        for row in clean_rows
                        if row["tier"] == tier and row["benchmark"] == benchmark
                    ]
                )
                for benchmark in BENCHMARKS
            },
        }
        for tier in TIERS
    }

    evaluator = ROOT / "scripts/eval_full.py"
    if sha256_file(_regular_file(evaluator)) != LOCKED_LOCOMO_EVALUATOR_SHA256:
        raise ScreeningAnalysisError("locked LoCoMo evaluator hash differs")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "study": STUDY,
        "parameters": {
            "write_turns": 32,
            "tiers": list(TIERS),
            "benchmarks": list(BENCHMARKS),
        },
        "analysis_script": _descriptor(Path(__file__)),
        "inputs": {
            "build_analysis": _descriptor(build_analysis_path),
            "evaluators": {"locomo": _descriptor(evaluator)},
            "qa": {},
            "scores": {},
        },
        "admission": {
            "all_passed": True,
            "checks": [
                "12 complete W32 build runs: 6 Terra and 6 Sol",
                "356 independently audited QA records per tier",
                "accepted attempt and bound ledgers verified per question",
                "314 locked LoCoMo scores per tier recomputed",
                "2 official LongMemEval boolean judgments per tier",
                "40-question, 103-nugget BEAM semantic audit passed per tier",
                "BEAM paired inventory and judge procedure hashes match",
            ],
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for tier in TIERS:
        manifest["inputs"]["qa"][tier] = {
            "root": _display_path(qa[tier]["root"]),
            "audit": _descriptor(qa[tier]["audit_path"]),
            "completion": _descriptor(qa[tier]["completion_path"]),
            "record_count": len(qa[tier]["record_paths"]),
            "records_tree": _tree_descriptor(
                qa[tier]["record_paths"], base=qa[tier]["root"]
            ),
            "accepted_attempt_artifacts_tree": _tree_descriptor(
                qa[tier]["accepted_files"], base=qa[tier]["root"]
            ),
            "attempt_resolution_tree": _tree_descriptor(
                qa[tier]["attempt_resolution_files"], base=qa[tier]["root"]
            ),
        }
        manifest["inputs"]["scores"][tier] = _score_manifest(score_inputs[tier])

    output = output_root.expanduser().absolute()
    manifest_path = output / "input_manifest.json"
    rows_path = output / "per_question.jsonl"
    paired_path = output / "paired_results.json"
    summary_path = output / "summary.json"
    manifest_payload = _json_bytes(manifest)
    rows_payload = _jsonl_bytes(clean_rows)
    paired_payload = _json_bytes(paired)

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "study": STUDY,
        "input_manifest": _payload_descriptor(manifest_path, manifest_payload),
        "claim_boundary": CLAIM_BOUNDARY,
        "scope": {
            "write_turns": 32,
            "tiers": list(TIERS),
            "histories": 6,
            "questions_per_tier": EXPECTED_QUESTIONS_PER_TIER,
            "paired_questions": EXPECTED_QUESTIONS_PER_TIER,
        },
        "quality": quality,
        "qa_access": qa_access,
        "build": build,
        "comparisons": _payload_descriptor(paired_path, paired_payload),
    }
    paper_directory = paper_dir.expanduser().absolute()
    paper_payloads = _paper_artifact_payloads(summary)
    summary["generated_artifacts"] = {
        "per_question": _payload_descriptor(rows_path, rows_payload),
        "paired_results": _payload_descriptor(paired_path, paired_payload),
        "paper": [
            _payload_descriptor(paper_directory / name, payload)
            for name, payload in paper_payloads.items()
        ],
    }
    summary_payload = _json_bytes(summary)
    bundle = {
        manifest_path: manifest_payload,
        rows_path: rows_payload,
        paired_path: paired_payload,
        summary_path: summary_payload,
        **{
            paper_directory / name: payload
            for name, payload in paper_payloads.items()
        },
    }
    _publish_bundle_no_clobber(bundle)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-analysis", type=Path, default=DEFAULT_BUILD_ANALYSIS
    )
    parser.add_argument("--qa-base", type=Path, default=DEFAULT_QA_BASE)
    parser.add_argument("--score-root", type=Path, default=DEFAULT_SCORE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = analyze_campaign(
        build_analysis_path=args.build_analysis,
        qa_base=args.qa_base,
        score_root=args.score_root,
        output_root=args.output_root,
        paper_dir=args.paper_dir,
    )
    print(
        "validated and wrote fixed-W=32 screening analysis for "
        f"{summary['scope']['paired_questions']} paired questions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
