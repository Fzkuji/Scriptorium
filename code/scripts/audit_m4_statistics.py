#!/usr/bin/env python3
"""Independently regenerate an M4 statistics artifact and compare its payload."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_m4_statistics import calculate, read_json, sha256_file  # noqa: E402
from scripts.evaluation.m4_reliability import ReliabilityError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    result_path = args.result.expanduser().resolve()
    audit_path = args.output.expanduser().resolve()
    existing = read_json(result_path)
    if not isinstance(existing, dict) or existing.get("status") != "complete":
        raise ReliabilityError("statistics result is not complete")
    protocol = existing.get("protocol")
    if not isinstance(protocol, dict):
        raise ReliabilityError("statistics result has no protocol")
    regenerated = calculate(
        source,
        repetitions=protocol.get("cluster_bootstrap_repetitions"),
        base_seed=protocol.get("base_seed"),
    )
    left = copy.deepcopy(existing)
    right = copy.deepcopy(regenerated)
    left.pop("created_at", None)
    right.pop("created_at", None)
    if left != right:
        raise ReliabilityError("statistics result differs from independent regeneration")
    audit = {
        "schema_version": 1,
        "status": "pass",
        "input_sha256": sha256_file(source),
        "result_sha256": sha256_file(result_path),
        "comparisons": len(existing.get("results", [])),
        "checks": [
            "frozen_input_hash",
            "paired_record_identity",
            "clustered_bootstrap_exact_regeneration",
            "exact_mcnemar",
            "holm_within_family",
        ],
    }
    from scripts.run_m4_statistics import atomic_json_no_clobber

    atomic_json_no_clobber(audit_path, audit)
    print(json.dumps(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
