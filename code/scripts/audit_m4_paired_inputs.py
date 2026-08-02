#!/usr/bin/env python3
"""Regenerate and audit a frozen M4 paired-record input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_m4_paired_inputs import calculate  # noqa: E402
from scripts.run_m4_statistics import (  # noqa: E402
    atomic_json_no_clobber,
    read_json,
    sha256_file,
)
from scripts.evaluation.m4_reliability import ReliabilityError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = args.spec.expanduser().resolve()
    frozen = args.input.expanduser().resolve()
    existing = read_json(frozen)
    regenerated = calculate(spec)
    if existing != regenerated:
        raise ReliabilityError("paired input differs from independent regeneration")
    audit = {
        "schema_version": 1,
        "status": "pass",
        "spec_sha256": sha256_file(spec),
        "input_sha256": sha256_file(frozen),
        "source_artifacts": len(existing.get("source_artifacts", [])),
        "comparisons": len(existing.get("comparisons", [])),
        "checks": [
            "source_artifact_hashes",
            "complete_status_and_required_metadata",
            "exact_question_set_pairing",
            "cluster_identity",
            "metric_type",
        ],
    }
    atomic_json_no_clobber(args.output.expanduser().resolve(), audit)
    print(json.dumps(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
