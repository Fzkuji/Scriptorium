#!/usr/bin/env python3
"""Freeze paired M4 records from complete, hash-bound method artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_m4_statistics import (  # noqa: E402
    atomic_json_no_clobber,
    read_json,
    sha256_file,
)
from src.evaluation.m4_reliability import ReliabilityError  # noqa: E402


SCHEMA_VERSION = 1
LOCOMO_ID = re.compile(r"^s([0-9]+)_q[0-9]+$")


def dotted_get(record: dict[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ReliabilityError(f"missing dotted field {path!r}")
        current = current[part]
    return current


def _finite(value: object, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReliabilityError(f"{label} is not numeric")
    if not math.isfinite(float(value)):
        raise ReliabilityError(f"{label} is not finite")
    return value


def _resolve_artifact(spec_path: Path, text: str) -> Path:
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = spec_path.parent / candidate
    return candidate.resolve()


def _cluster_id(method: dict[str, Any], record: dict[str, Any], question_id: str) -> str:
    if isinstance(method.get("cluster_field"), str):
        cluster = dotted_get(record, method["cluster_field"])
        if not isinstance(cluster, (str, int)) or isinstance(cluster, bool):
            raise ReliabilityError(f"{question_id} has invalid cluster field")
        return str(cluster)
    if method.get("cluster_rule") == "locomo_question_id":
        match = LOCOMO_ID.fullmatch(question_id)
        if not match:
            raise ReliabilityError(f"cannot derive LoCoMo cluster from {question_id}")
        return f"s{match.group(1)}"
    raise ReliabilityError("method must declare cluster_field or cluster_rule")


def load_method(
    method: dict[str, Any], *, spec_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    method_id = method.get("method_id")
    artifact_text = method.get("artifact")
    expected_sha = method.get("sha256")
    expected_questions = method.get("expected_questions")
    metrics = method.get("metrics")
    if (
        not isinstance(method_id, str)
        or not method_id
        or not isinstance(artifact_text, str)
        or not artifact_text
        or not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or isinstance(expected_questions, bool)
        or not isinstance(expected_questions, int)
        or expected_questions < 1
        or not isinstance(metrics, dict)
        or not metrics
    ):
        raise ReliabilityError("method specification is incomplete")
    artifact = _resolve_artifact(spec_path, artifact_text)
    if not artifact.is_file() or artifact.is_symlink():
        raise ReliabilityError(f"method artifact is unsafe or absent: {artifact}")
    actual_sha = sha256_file(artifact)
    if actual_sha != expected_sha:
        raise ReliabilityError(f"method artifact hash changed: {method_id}")
    payload = read_json(artifact)
    if not isinstance(payload, dict):
        raise ReliabilityError(f"method artifact is not an object: {method_id}")
    required_meta = method.get("required_meta", {})
    if not isinstance(required_meta, dict):
        raise ReliabilityError(f"{method_id} required_meta is not an object")
    meta = payload.get(method.get("meta_key", "meta"))
    if not isinstance(meta, dict):
        raise ReliabilityError(f"{method_id} has no metadata object")
    for key, expected in required_meta.items():
        if dotted_get(meta, key) != expected:
            raise ReliabilityError(f"{method_id} metadata differs at {key}")
    if meta.get("status") != "complete":
        raise ReliabilityError(f"{method_id} artifact status is not complete")
    records = payload.get(method.get("records_key", "results"))
    if not isinstance(records, list):
        raise ReliabilityError(f"{method_id} records are not a list")
    question_id_field = method.get("question_id_field", "question_id")
    if not isinstance(question_id_field, str) or not question_id_field:
        raise ReliabilityError(f"{method_id} question_id_field is invalid")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ReliabilityError(f"{method_id} contains a non-object record")
        question_id = dotted_get(record, question_id_field)
        if question_id == "_build_stats":
            continue
        if not isinstance(question_id, str) or not question_id or question_id in indexed:
            raise ReliabilityError(f"{method_id} has invalid question identities")
        metric_values: dict[str, int | float] = {}
        for metric, field in metrics.items():
            if not isinstance(metric, str) or not metric or not isinstance(field, str):
                raise ReliabilityError(f"{method_id} metric specification is invalid")
            metric_values[metric] = _finite(
                dotted_get(record, field), f"{method_id}.{question_id}.{metric}"
            )
        indexed[question_id] = {
            "cluster_id": _cluster_id(method, record, question_id),
            "metrics": metric_values,
        }
    if len(indexed) != expected_questions:
        raise ReliabilityError(
            f"{method_id} has {len(indexed)} questions, expected {expected_questions}"
        )
    source = {
        "method_id": method_id,
        "path": str(artifact),
        "sha256": actual_sha,
        "questions": len(indexed),
        "required_meta": copy.deepcopy(required_meta),
        "metrics": copy.deepcopy(metrics),
    }
    return source, indexed


def calculate(spec_path: Path) -> dict[str, Any]:
    spec = read_json(spec_path)
    if not isinstance(spec, dict) or spec.get("schema_version") != SCHEMA_VERSION:
        raise ReliabilityError("M4 pairing spec has invalid schema")
    if spec.get("status") != "frozen":
        raise ReliabilityError("M4 pairing spec status must be frozen")
    analysis_id = spec.get("analysis_id")
    methods = spec.get("methods")
    comparisons = spec.get("comparisons")
    if (
        not isinstance(analysis_id, str)
        or not analysis_id
        or not isinstance(methods, list)
        or not methods
        or not isinstance(comparisons, list)
        or not comparisons
    ):
        raise ReliabilityError("M4 pairing spec is incomplete")

    sources: list[dict[str, Any]] = []
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for method in methods:
        if not isinstance(method, dict):
            raise ReliabilityError("M4 method specification is not an object")
        source, records = load_method(method, spec_path=spec_path)
        method_id = source["method_id"]
        if method_id in indexed:
            raise ReliabilityError(f"duplicate M4 method_id: {method_id}")
        sources.append(source)
        indexed[method_id] = records

    frozen_comparisons: list[dict[str, Any]] = []
    seen_comparisons: set[str] = set()
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ReliabilityError("M4 comparison specification is not an object")
        comparison_id = comparison.get("comparison_id")
        family = comparison.get("family")
        left = comparison.get("left_method")
        right = comparison.get("right_method")
        metric = comparison.get("metric")
        kind = comparison.get("kind")
        if (
            not isinstance(comparison_id, str)
            or not comparison_id
            or comparison_id in seen_comparisons
            or not isinstance(family, str)
            or not family
            or left not in indexed
            or right not in indexed
            or not isinstance(metric, str)
            or not metric
            or kind not in {"binary", "continuous"}
        ):
            raise ReliabilityError("M4 comparison specification is invalid")
        left_ids = set(indexed[left])
        right_ids = set(indexed[right])
        if left_ids != right_ids:
            raise ReliabilityError(f"{comparison_id} question sets differ")
        records: list[dict[str, Any]] = []
        for question_id in sorted(left_ids):
            left_record = indexed[left][question_id]
            right_record = indexed[right][question_id]
            if left_record["cluster_id"] != right_record["cluster_id"]:
                raise ReliabilityError(f"{comparison_id} cluster differs for {question_id}")
            if metric not in left_record["metrics"] or metric not in right_record["metrics"]:
                raise ReliabilityError(f"{comparison_id} metric is absent for {question_id}")
            left_value = left_record["metrics"][metric]
            right_value = right_record["metrics"][metric]
            if kind == "binary" and (left_value not in (0, 1) or right_value not in (0, 1)):
                raise ReliabilityError(f"{comparison_id} has a non-binary outcome")
            records.append(
                {
                    "question_id": question_id,
                    "cluster_id": left_record["cluster_id"],
                    "left": left_value,
                    "right": right_value,
                }
            )
        frozen_comparisons.append(
            {
                "comparison_id": comparison_id,
                "family": family,
                "kind": kind,
                "left_method": left,
                "right_method": right,
                "metric": metric,
                "records": records,
            }
        )
        seen_comparisons.add(comparison_id)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "analysis_id": analysis_id,
        "spec": {"path": str(spec_path.resolve()), "sha256": sha256_file(spec_path)},
        "source_artifacts": sources,
        "comparisons": frozen_comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calculate(args.spec.expanduser().resolve())
    atomic_json_no_clobber(args.output.expanduser().resolve(), result)
    print(json.dumps({"status": "frozen", "comparisons": len(result["comparisons"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
