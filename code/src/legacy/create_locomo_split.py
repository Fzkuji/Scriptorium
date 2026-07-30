"""Create fixed LoCoMo JSONL splits for NativeMem experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_balanced_rows(
    data: list[dict[str, Any]],
    source_path: Path,
    split_id: str,
    per_category: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_sha256 = sha256_file(source_path)

    for category in [1, 2, 3, 4, 5]:
        selected_for_category = 0
        cursors = [0 for _ in data]

        while selected_for_category < per_category:
            progressed = False
            for sample_idx, sample in enumerate(data):
                qa_list = sample.get("qa", [])
                while cursors[sample_idx] < len(qa_list):
                    qa_idx = cursors[sample_idx]
                    cursors[sample_idx] += 1
                    qa = qa_list[qa_idx]
                    if qa.get("category") != category:
                        continue

                    question_id = f"conv{sample_idx}_q{qa_idx}"
                    rows.append({
                        "split_id": split_id,
                        "dataset_id": "D1",
                        "source_dataset": str(source_path),
                        "source_sha256": source_sha256,
                        "conversation_id": f"conv{sample_idx}",
                        "sample_idx": sample_idx,
                        "source_sample_id": sample.get("sample_id"),
                        "question_id": question_id,
                        "qa_index": qa_idx,
                        "category": category,
                        "category_name": CATEGORY_NAMES.get(category, f"cat{category}"),
                        "question": qa["question"],
                    })
                    selected_for_category += 1
                    progressed = True
                    break
                if selected_for_category >= per_category:
                    break
            if not progressed:
                raise ValueError(
                    f"Not enough questions for category {category}: "
                    f"requested {per_category}, found {selected_for_category}"
                )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="code/locomo/data/locomo10.json")
    parser.add_argument("--split-id", default="locomo_balanced_100_v1")
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--out", default="artifacts/splits/locomo_balanced_100_v1.jsonl")
    args = parser.parse_args()

    data_path = Path(args.data)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    rows = build_balanced_rows(
        data=data,
        source_path=data_path,
        split_id=args.split_id,
        per_category=args.per_category,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category_name"]] = counts.get(row["category_name"], 0) + 1

    print(f"wrote {len(rows)} rows to {out_path}")
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
