#!/usr/bin/env python3
"""Run preregistered paired M4 statistics from a frozen comparison artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.m4_reliability import (  # noqa: E402
    ReliabilityError,
    analyze_comparison,
    holm_adjust,
)


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json_no_clobber(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ReliabilityError(
                f"refusing to overwrite existing output: {path}"
            ) from exc
        os.unlink(temporary)
        temporary = ""
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def validate_input(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ReliabilityError("comparison input has unsupported schema_version")
    if payload.get("status") != "frozen":
        raise ReliabilityError("comparison input status must be frozen")
    analysis_id = payload.get("analysis_id")
    comparisons = payload.get("comparisons")
    if not isinstance(analysis_id, str) or not analysis_id:
        raise ReliabilityError("comparison input has invalid analysis_id")
    if not isinstance(comparisons, list) or not comparisons:
        raise ReliabilityError("comparison input has no comparisons")
    seen: set[str] = set()
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ReliabilityError("comparison is not an object")
        comparison_id = comparison.get("comparison_id")
        family = comparison.get("family")
        if (
            not isinstance(comparison_id, str)
            or not comparison_id
            or comparison_id in seen
            or not isinstance(family, str)
            or not family
        ):
            raise ReliabilityError("comparison identity or family is invalid")
        if comparison.get("kind") not in {"binary", "continuous"}:
            raise ReliabilityError(f"{comparison_id} has invalid kind")
        records = comparison.get("records")
        if not isinstance(records, list) or not records:
            raise ReliabilityError(f"{comparison_id} records are empty")
        seen.add(comparison_id)
    return payload


def derive_seed(base_seed: int, analysis_id: str, comparison_id: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}|{analysis_id}|{comparison_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def calculate(
    source: Path, *, repetitions: int, base_seed: int
) -> dict[str, Any]:
    payload = validate_input(read_json(source))
    results: list[dict[str, Any]] = []
    by_family: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for comparison in payload["comparisons"]:
        comparison_id = comparison["comparison_id"]
        seed = derive_seed(base_seed, payload["analysis_id"], comparison_id)
        statistics = analyze_comparison(
            comparison["records"],
            kind=comparison["kind"],
            repetitions=repetitions,
            seed=seed,
        )
        result = {
            "comparison_id": comparison_id,
            "family": comparison["family"],
            "left_method": comparison.get("left_method"),
            "right_method": comparison.get("right_method"),
            "metric": comparison.get("metric"),
            "statistics": statistics,
        }
        if comparison["kind"] == "binary":
            by_family[comparison["family"]].append(
                (comparison_id, statistics["exact_mcnemar"]["p_value"])
            )
        results.append(result)

    adjusted: dict[str, dict[str, float]] = {
        family: holm_adjust(values) for family, values in sorted(by_family.items())
    }
    for result in results:
        family_adjusted = adjusted.get(result["family"], {})
        if result["comparison_id"] in family_adjusted:
            result["statistics"]["exact_mcnemar"]["holm_adjusted_p_value"] = (
                family_adjusted[result["comparison_id"]]
            )
            result["statistics"]["exact_mcnemar"]["holm_family"] = result["family"]

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "analysis_id": payload["analysis_id"],
        "created_at": utc_now(),
        "source": {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
        },
        "protocol": {
            "paired_difference": "left_minus_right",
            "cluster_bootstrap_repetitions": repetitions,
            "base_seed": base_seed,
            "confidence": 0.95,
            "binary_test": "two_sided_exact_mcnemar",
            "multiple_testing": "Holm within each declared binary family",
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    result = calculate(
        args.input.expanduser().resolve(),
        repetitions=args.repetitions,
        base_seed=args.seed,
    )
    atomic_json_no_clobber(args.output.expanduser().resolve(), result)
    print(json.dumps({"status": "complete", "comparisons": len(result["results"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
