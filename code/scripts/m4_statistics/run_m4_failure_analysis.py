#!/usr/bin/env python3
"""Apply the frozen earliest-stage failure rule and select qualitative cases."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m4_statistics.run_m4_statistics import (  # noqa: E402
    atomic_json_no_clobber,
    read_json,
    sha256_file,
)
from scripts.evaluation.m4_reliability import (  # noqa: E402
    FAILURE_LABELS,
    ReliabilityError,
    attribute_failure,
    select_qualitative_cases,
)


SCHEMA_VERSION = 1
RULE_VERSION = "m4-earliest-stage-v1"
SELECTION_SEED = "m4-qualitative-v1"


def calculate(source: Path, *, per_label: int = 2) -> dict[str, Any]:
    payload = read_json(source)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ReliabilityError("failure input has invalid schema")
    if payload.get("status") != "frozen" or payload.get("rule_version") != RULE_VERSION:
        raise ReliabilityError("failure input is not frozen to the expected rule")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ReliabilityError("failure input records are empty")
    labels = [attribute_failure(record) for record in records]
    if len({label["question_id"] for label in labels}) != len(labels):
        raise ReliabilityError("failure input has duplicate question identities")
    counts = Counter(
        label["primary_label"]
        for label in labels
        if label["eligible"] and label["primary_label"] is not None
    )
    selected = select_qualitative_cases(
        labels, per_label=per_label, seed=SELECTION_SEED
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "rule_version": RULE_VERSION,
        "source": {"path": str(source.resolve()), "sha256": sha256_file(source)},
        "rule_order": [
            "extraction_omission",
            "wrong_path",
            "maintenance_error",
            "navigation_miss",
            "source_resolution_error",
            "judge_ambiguity_when_incorrect_and_judges_disagree",
            "answer_error",
        ],
        "records": len(labels),
        "eligible_records": sum(label["eligible"] for label in labels),
        "excluded_incomplete_mapping": sum(not label["eligible"] for label in labels),
        "correct_without_primary_failure": sum(
            label["eligible"] and label["primary_label"] is None for label in labels
        ),
        "primary_failure_counts": {label: counts[label] for label in FAILURE_LABELS},
        "labels": labels,
        "qualitative_selection": {
            "seed": SELECTION_SEED,
            "per_label": per_label,
            "rule": "first SHA-256 ordered cases within each frozen primary label",
            "cases": selected,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=2)
    args = parser.parse_args()
    result = calculate(args.input.expanduser().resolve(), per_label=args.per_label)
    atomic_json_no_clobber(args.output.expanduser().resolve(), result)
    print(json.dumps({"status": "complete", "records": result["records"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
