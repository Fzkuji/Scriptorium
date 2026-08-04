#!/usr/bin/env python3
"""Independently regenerate and audit the R601/R602 failure artifact."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m4_statistics.run_m4_failure_analysis import calculate  # noqa: E402
from scripts.m4_statistics.run_m4_statistics import (  # noqa: E402
    atomic_json_no_clobber,
    read_json,
    sha256_file,
)
from scripts.evaluation.m4_reliability import ReliabilityError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    result_path = args.result.expanduser().resolve()
    existing = read_json(result_path)
    if not isinstance(existing, dict) or existing.get("status") != "complete":
        raise ReliabilityError("failure result is not complete")
    selection = existing.get("qualitative_selection")
    if not isinstance(selection, dict):
        raise ReliabilityError("failure result lacks qualitative selection")
    regenerated = calculate(source, per_label=selection.get("per_label"))
    if copy.deepcopy(existing) != regenerated:
        raise ReliabilityError("failure result differs from independent regeneration")
    audit = {
        "schema_version": 1,
        "status": "pass",
        "input_sha256": sha256_file(source),
        "result_sha256": sha256_file(result_path),
        "records": existing["records"],
        "checks": [
            "complete_source_mapping_exclusions",
            "earliest_primary_stage",
            "optional_secondary_tags",
            "trace_identifier_preservation",
            "deterministic_stratified_case_selection",
        ],
    }
    atomic_json_no_clobber(args.output.expanduser().resolve(), audit)
    print(json.dumps(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
