#!/usr/bin/env python3
"""Validate two independent R501 label files and compute agreement statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.human_agreement.build_locomo_human_packet import VERDICTS, read_json, sha256_file  # noqa: E402
from baselines.m4_statistics.run_m4_statistics import atomic_json_no_clobber  # noqa: E402
from scripts.evaluation.m4_reliability import ReliabilityError  # noqa: E402


EXPECTED_FIELDS = ["packet_id", "verdict", "confidence", "rationale"]


def load_labels(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_FIELDS:
            raise ReliabilityError(f"{path} has invalid CSV columns")
        labels: dict[str, dict[str, Any]] = {}
        for row_number, row in enumerate(reader, start=2):
            packet_id = row.get("packet_id", "")
            verdict = row.get("verdict", "").strip().casefold()
            confidence_text = row.get("confidence", "").strip()
            if packet_id not in expected_ids or packet_id in labels:
                raise ReliabilityError(f"{path}:{row_number} has invalid packet_id")
            if verdict not in VERDICTS:
                raise ReliabilityError(f"{path}:{row_number} has invalid verdict")
            try:
                confidence = int(confidence_text)
            except ValueError as exc:
                raise ReliabilityError(
                    f"{path}:{row_number} confidence must be an integer"
                ) from exc
            if not 1 <= confidence <= 5:
                raise ReliabilityError(f"{path}:{row_number} confidence must be 1..5")
            labels[packet_id] = {
                "verdict": verdict,
                "confidence": confidence,
                "rationale": row.get("rationale", "").strip(),
            }
    if set(labels) != expected_ids:
        missing = sorted(expected_ids - set(labels))
        raise ReliabilityError(f"{path} is missing {len(missing)} packet IDs")
    return labels


def cohen_kappa(pairs: list[tuple[str, str]], classes: tuple[str, ...]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "raw_agreement": None, "kappa": None}
    left = Counter(item[0] for item in pairs)
    right = Counter(item[1] for item in pairs)
    observed = sum(a == b for a, b in pairs) / len(pairs)
    expected = sum(left[label] * right[label] for label in classes) / (len(pairs) ** 2)
    denominator = 1.0 - expected
    kappa = (observed - expected) / denominator if denominator else (1.0 if observed == 1 else None)
    return {
        "n": len(pairs),
        "classes": list(classes),
        "raw_agreement": observed,
        "chance_agreement": expected,
        "kappa": kappa,
        "left_marginals": dict(left),
        "right_marginals": dict(right),
    }


def judge_agreement(
    labels: dict[str, dict[str, Any]], key_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    eligible = [
        packet_id
        for packet_id, label in labels.items()
        if label["verdict"] != "uncertain"
    ]
    matches = 0
    confusion: Counter[tuple[int, int]] = Counter()
    for packet_id in eligible:
        human = int(labels[packet_id]["verdict"] == "correct")
        judge = int(key_by_id[packet_id]["primary_judge_score"])
        matches += human == judge
        confusion[(human, judge)] += 1
    return {
        "n_binary_human_labels": len(eligible),
        "agreement": matches / len(eligible) if eligible else None,
        "human_correct_judge_correct": confusion[(1, 1)],
        "human_correct_judge_incorrect": confusion[(1, 0)],
        "human_incorrect_judge_correct": confusion[(0, 1)],
        "human_incorrect_judge_incorrect": confusion[(0, 0)],
    }


def calculate(
    private_key_path: Path,
    labels_a_path: Path,
    labels_b_path: Path,
    *,
    annotator_a: str,
    annotator_b: str,
) -> dict[str, Any]:
    if not annotator_a.strip() or not annotator_b.strip() or annotator_a == annotator_b:
        raise ReliabilityError("two distinct non-empty annotator IDs are required")
    key = read_json(private_key_path)
    if not isinstance(key, dict) or key.get("schema_version") != 1:
        raise ReliabilityError("private key has invalid schema")
    key_items = key.get("items")
    if not isinstance(key_items, list) or len(key_items) != 100:
        raise ReliabilityError("private key must contain exactly 100 items")
    key_by_id = {item.get("packet_id"): item for item in key_items if isinstance(item, dict)}
    if len(key_by_id) != 100 or any(not isinstance(item, str) for item in key_by_id):
        raise ReliabilityError("private key has invalid packet identities")
    expected_ids = set(key_by_id)
    labels_a = load_labels(labels_a_path, expected_ids)
    labels_b = load_labels(labels_b_path, expected_ids)
    order = [item["packet_id"] for item in key_items]
    all_pairs = [
        (labels_a[packet_id]["verdict"], labels_b[packet_id]["verdict"])
        for packet_id in order
    ]
    binary_pairs = [
        pair for pair in all_pairs if "uncertain" not in pair
    ]
    consensus_ids = [
        packet_id
        for packet_id in order
        if labels_a[packet_id]["verdict"] == labels_b[packet_id]["verdict"]
        and labels_a[packet_id]["verdict"] != "uncertain"
    ]
    consensus_matches = sum(
        int(labels_a[packet_id]["verdict"] == "correct")
        == int(key_by_id[packet_id]["primary_judge_score"])
        for packet_id in consensus_ids
    )
    confidence_pairs = [
        (labels_a[packet_id]["confidence"], labels_b[packet_id]["confidence"])
        for packet_id in order
    ]
    return {
        "schema_version": 1,
        "status": "complete",
        "packet_id": key.get("packet_id"),
        "annotators": [annotator_a, annotator_b],
        "source_hashes": {
            "private_key": sha256_file(private_key_path),
            "labels_a": sha256_file(labels_a_path),
            "labels_b": sha256_file(labels_b_path),
        },
        "three_class_agreement": cohen_kappa(all_pairs, VERDICTS),
        "binary_agreement_excluding_any_uncertain": cohen_kappa(
            binary_pairs, ("correct", "incorrect")
        ),
        "annotator_a_vs_primary_judge": judge_agreement(labels_a, key_by_id),
        "annotator_b_vs_primary_judge": judge_agreement(labels_b, key_by_id),
        "consensus_binary_vs_primary_judge": {
            "n": len(consensus_ids),
            "agreement": (
                consensus_matches / len(consensus_ids) if consensus_ids else None
            ),
        },
        "mean_confidence": {
            "annotator_a": sum(pair[0] for pair in confidence_pairs) / 100,
            "annotator_b": sum(pair[1] for pair in confidence_pairs) / 100,
        },
        "limitations": [
            "uncertain labels are retained in the three-class statistic",
            "binary agreement excludes any pair containing uncertain",
            "agreement with the automated judge is not human accuracy against an independent fact source",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--labels-a", type=Path, required=True)
    parser.add_argument("--labels-b", type=Path, required=True)
    parser.add_argument("--annotator-a", required=True)
    parser.add_argument("--annotator-b", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calculate(
        args.private_key.expanduser().resolve(),
        args.labels_a.expanduser().resolve(),
        args.labels_b.expanduser().resolve(),
        annotator_a=args.annotator_a,
        annotator_b=args.annotator_b,
    )
    if not all(
        math.isfinite(value)
        for value in result["mean_confidence"].values()
    ):
        raise ReliabilityError("mean confidence is not finite")
    atomic_json_no_clobber(args.output.expanduser().resolve(), result)
    print(json.dumps({"status": "complete", "packet_id": result["packet_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
