#!/usr/bin/env python3
"""Analyze completed build-only runs from the v10 context-history ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analyze_v10_write_interval import analyze_build
from scripts.run_v10_context_history_ablation import VARIANTS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    rows = []
    for build_path in sorted((results_dir / "runs").glob("*/*/*/build.json")):
        row = analyze_build(build_path, dataset)
        relative = build_path.relative_to(results_dir / "runs")
        row["writer_id"] = relative.parts[0]
        row["variant"] = relative.parts[1]
        variant = VARIANTS.get(row["variant"])
        row["variant_display_name"] = (
            variant.display_name if variant is not None else row["variant"]
        )
        rows.append(row)

    output = {
        "schema_version": 1,
        "study": "NativeMem-v10-context-history",
        "results_dir": str(results_dir),
        "completed_builds": len(rows),
        "runs": rows,
    }
    output_path = args.output or (results_dir / "analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} analyzed builds -> {output_path}")


if __name__ == "__main__":
    main()
