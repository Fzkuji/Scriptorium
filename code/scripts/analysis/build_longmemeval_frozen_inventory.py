#!/usr/bin/env python3
"""Create a deterministic inventory of atomically built LongMemEval items."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: dict[int, dict[str, object]] = {}
    for path in sorted(args.formal_root.glob("*/items/*/checkpoint.json")):
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("status") != "built":
            continue
        if checkpoint.get("build", {}).get("status") != "complete":
            continue
        index = int(checkpoint["dataset_index"])
        record = {
            "dataset_index": index,
            "question_id": checkpoint["question_id"],
            "checkpoint": str(path.resolve()),
            "answer": "",
        }
        previous = records.get(index)
        if previous is not None and previous["checkpoint"] != record["checkpoint"]:
            raise RuntimeError(f"duplicate built checkpoint for item {index}")
        records[index] = record

    payload = {
        "schema_version": 1,
        "selection": "status=built and build.status=complete",
        "count": len(records),
        "records": [records[index] for index in sorted(records)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
