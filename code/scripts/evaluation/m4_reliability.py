"""Deterministic reliability statistics and failure attribution for M4.

The functions in this module operate on already-audited, paired question
records.  They make no model or network calls and intentionally require
callers to supply explicit cluster and stage evidence instead of inferring
missing provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable


FAILURE_LABELS = (
    "extraction_omission",
    "wrong_path",
    "maintenance_error",
    "navigation_miss",
    "source_resolution_error",
    "answer_error",
    "judge_ambiguity",
)


class ReliabilityError(ValueError):
    """Raised when M4 inputs are incomplete or internally inconsistent."""


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReliabilityError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ReliabilityError(f"{label} must be a finite number")
    return result


def _validate_paired_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ReliabilityError("paired records are empty")
    identities: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ReliabilityError(f"record {index} is not an object")
        identity = record.get("question_id")
        cluster = record.get("cluster_id")
        if not isinstance(identity, str) or not identity:
            raise ReliabilityError(f"record {index} has invalid question_id")
        if identity in identities:
            raise ReliabilityError(f"duplicate question_id: {identity}")
        if not isinstance(cluster, str) or not cluster:
            raise ReliabilityError(f"{identity} has invalid cluster_id")
        _finite_number(record.get("left"), f"{identity}.left")
        _finite_number(record.get("right"), f"{identity}.right")
        identities.add(identity)


def percentile(values: list[float], probability: float) -> float:
    """Return a deterministic type-7 sample percentile."""
    if not values:
        raise ReliabilityError("cannot take a percentile of an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise ReliabilityError("percentile probability must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def clustered_bootstrap_difference(
    records: list[dict[str, Any]],
    *,
    repetitions: int = 10_000,
    seed: int = 20260714,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Question-weighted paired difference with conversation-cluster resampling."""
    _validate_paired_records(records)
    if isinstance(repetitions, bool) or repetitions < 1:
        raise ReliabilityError("repetitions must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ReliabilityError("confidence must be between zero and one")

    clusters: dict[str, list[float]] = defaultdict(list)
    all_differences: list[float] = []
    for record in records:
        difference = float(record["left"]) - float(record["right"])
        clusters[str(record["cluster_id"])].append(difference)
        all_differences.append(difference)
    cluster_ids = sorted(clusters)
    if len(cluster_ids) < 2:
        raise ReliabilityError("clustered bootstrap requires at least two clusters")

    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(repetitions):
        total = 0.0
        count = 0
        for _ in cluster_ids:
            cluster_id = cluster_ids[rng.randrange(len(cluster_ids))]
            values = clusters[cluster_id]
            total += sum(values)
            count += len(values)
        replicates.append(total / count)

    alpha = 1.0 - confidence
    return {
        "estimand": "question_weighted_mean_left_minus_right",
        "estimate": sum(all_differences) / len(all_differences),
        "confidence": confidence,
        "ci_low": percentile(replicates, alpha / 2.0),
        "ci_high": percentile(replicates, 1.0 - alpha / 2.0),
        "repetitions": repetitions,
        "seed": seed,
        "questions": len(records),
        "clusters": len(cluster_ids),
        "cluster_sizes": {key: len(clusters[key]) for key in cluster_ids},
        "resampling_unit": "cluster_id_with_all_questions",
    }


def exact_mcnemar(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the two-sided exact McNemar test for paired binary outcomes."""
    _validate_paired_records(records)
    cells: Counter[tuple[int, int]] = Counter()
    for record in records:
        left = record["left"]
        right = record["right"]
        if left not in (0, 1) or right not in (0, 1):
            raise ReliabilityError("McNemar inputs must be binary 0/1 values")
        cells[(int(left), int(right))] += 1
    left_only = cells[(1, 0)]
    right_only = cells[(0, 1)]
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "test": "two_sided_exact_mcnemar",
        "both_correct": cells[(1, 1)],
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "both_incorrect": cells[(0, 0)],
        "discordant": discordant,
        "p_value": p_value,
    }


def holm_adjust(p_values: Iterable[tuple[str, float]]) -> dict[str, float]:
    """Return Holm-adjusted p-values keyed by unique comparison identifier."""
    values = list(p_values)
    if not values:
        return {}
    names = [name for name, _ in values]
    if len(set(names)) != len(names):
        raise ReliabilityError("Holm comparison identifiers must be unique")
    normalized: list[tuple[str, float]] = []
    for name, p_value in values:
        numeric = _finite_number(p_value, f"p_value[{name}]")
        if not 0.0 <= numeric <= 1.0:
            raise ReliabilityError(f"p_value[{name}] must be in [0, 1]")
        normalized.append((name, numeric))
    ordered = sorted(normalized, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * p_value))
        adjusted[name] = running
    return adjusted


def analyze_comparison(
    records: list[dict[str, Any]],
    *,
    kind: str,
    repetitions: int = 10_000,
    seed: int = 20260714,
) -> dict[str, Any]:
    _validate_paired_records(records)
    if kind not in {"binary", "continuous"}:
        raise ReliabilityError("comparison kind must be binary or continuous")
    left = [float(record["left"]) for record in records]
    right = [float(record["right"]) for record in records]
    result: dict[str, Any] = {
        "kind": kind,
        "left_mean": sum(left) / len(left),
        "right_mean": sum(right) / len(right),
        "paired_clustered_bootstrap": clustered_bootstrap_difference(
            records, repetitions=repetitions, seed=seed
        ),
        "paired_records_sha256": sha256_json(records),
    }
    if kind == "binary":
        result["exact_mcnemar"] = exact_mcnemar(records)
    return result


def _required_bool(record: dict[str, Any], key: str) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise ReliabilityError(f"{record.get('question_id', '<unknown>')}.{key} is not bool")
    return value


def attribute_failure(record: dict[str, Any]) -> dict[str, Any]:
    """Assign one earliest primary failure stage from explicit trace evidence.

    Questions without complete gold-source mapping are marked out of scope for
    stage attribution.  No negative source claim is inferred from a missing
    field.
    """
    question_id = record.get("question_id")
    if not isinstance(question_id, str) or not question_id:
        raise ReliabilityError("failure record has invalid question_id")
    trace_ids = record.get("trace_ids")
    if not isinstance(trace_ids, dict):
        raise ReliabilityError(f"{question_id}.trace_ids is not an object")
    for key, value in trace_ids.items():
        if not isinstance(key, str) or not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise ReliabilityError(f"{question_id}.trace_ids[{key!r}] is invalid")

    mapping_complete = _required_bool(record, "gold_source_mapping_complete")
    score_correct = _required_bool(record, "score_correct")
    judge_disagreement = _required_bool(record, "primary_sensitivity_disagree")
    if not mapping_complete:
        return {
            "question_id": question_id,
            "eligible": False,
            "exclusion_reason": "incomplete_gold_source_mapping",
            "primary_label": None,
            "secondary_tags": ["judge_ambiguity"] if judge_disagreement else [],
            "trace_ids": trace_ids,
        }

    in_entries = _required_bool(record, "gold_source_in_canonical_entries")
    survived = _required_bool(record, "gold_source_survived_maintenance")
    path_valid = _required_bool(record, "gold_source_path_valid")
    reached = _required_bool(record, "retrieval_reached_gold_source")
    resolved = _required_bool(record, "source_resolution_returned_gold_content")

    primary: str | None = None
    if not in_entries:
        primary = "extraction_omission"
    elif not path_valid:
        primary = "wrong_path"
    elif not survived:
        primary = "maintenance_error"
    elif not reached:
        primary = "navigation_miss"
    elif not resolved:
        primary = "source_resolution_error"
    elif not score_correct and judge_disagreement:
        primary = "judge_ambiguity"
    elif not score_correct:
        primary = "answer_error"

    secondary: list[str] = []
    if primary != "judge_ambiguity" and judge_disagreement:
        secondary.append("judge_ambiguity")
    downstream = (
        ("maintenance_error", not survived),
        ("wrong_path", not path_valid),
        ("navigation_miss", not reached),
        ("source_resolution_error", not resolved),
        ("answer_error", not score_correct),
    )
    for label, present in downstream:
        if present and label != primary and label not in secondary:
            secondary.append(label)
    return {
        "question_id": question_id,
        "eligible": True,
        "exclusion_reason": None,
        "primary_label": primary,
        "secondary_tags": secondary,
        "trace_ids": trace_ids,
    }


def select_qualitative_cases(
    labels: list[dict[str, Any]], *, per_label: int = 2, seed: str = "m4-v1"
) -> list[dict[str, Any]]:
    """Select cases by a frozen hash order within each primary failure label."""
    if isinstance(per_label, bool) or per_label < 1:
        raise ReliabilityError("per_label must be a positive integer")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for record in labels:
        question_id = record.get("question_id")
        label = record.get("primary_label")
        if not isinstance(question_id, str) or not question_id or question_id in seen:
            raise ReliabilityError("qualitative labels have invalid question identity")
        seen.add(question_id)
        if label is None:
            continue
        if label not in FAILURE_LABELS:
            raise ReliabilityError(f"unknown failure label: {label}")
        grouped[label].append(record)
    selected: list[dict[str, Any]] = []
    for label in FAILURE_LABELS:
        ordered = sorted(
            grouped.get(label, []),
            key=lambda record: hashlib.sha256(
                f"{seed}|{label}|{record['question_id']}".encode()
            ).hexdigest(),
        )
        for rank, record in enumerate(ordered[:per_label], start=1):
            selected.append(
                {
                    "failure_label": label,
                    "selection_rank": rank,
                    "question_id": record["question_id"],
                    "trace_ids": record.get("trace_ids", {}),
                }
            )
    return selected
