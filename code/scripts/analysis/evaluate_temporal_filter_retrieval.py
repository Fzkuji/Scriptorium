#!/usr/bin/env python3
"""Compare retrieval with no dates against an oracle evidence-date window."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.retrieval.bm25 import (  # noqa: E402
    MemoryBM25Index,
    event_matches_time_window,
    temporal_bounds,
)
from src.retrieval.embedding import MemoryEmbeddingIndex  # noqa: E402

DEFAULT_DATASET = ROOT / "benchmarks/longmemeval/data/longmemeval_s_cleaned.json"
KS = (1, 3, 5, 10)
REF_RE = re.compile(r"^D(\d+):")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_items(
    run_root: Path, dataset_path: Path, limit: int | None
) -> list[dict[str, Any]]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    by_question = {row["question_id"]: (index, row) for index, row in enumerate(dataset)}
    selected: dict[str, dict[str, Any]] = {}
    for checkpoint_path in sorted(run_root.rglob("checkpoint.json")):
        item_dir = checkpoint_path.parent
        topics = item_dir / "memory/topics"
        if not topics.is_dir() or not any(topics.rglob("*.md")):
            continue
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        question_id = checkpoint.get("question_id")
        dataset_entry = by_question.get(question_id)
        if (
            checkpoint.get("status") != "complete"
            or checkpoint.get("question_type") != "temporal-reasoning"
            or dataset_entry is None
            or question_id in selected
        ):
            continue
        dataset_index, row = dataset_entry
        selected[question_id] = {
            "dataset_index": dataset_index,
            "dataset": row,
            "checkpoint": checkpoint,
            "item_dir": item_dir,
        }
    items = sorted(selected.values(), key=lambda row: row["dataset_index"])
    return items[:limit] if limit else items


def source_ids_for_refs(refs: list[str], source_session_ids: list[str]) -> set[str]:
    sessions = set()
    for ref in refs:
        match = REF_RE.match(ref)
        if match and 0 < int(match.group(1)) <= len(source_session_ids):
            sessions.add(source_session_ids[int(match.group(1)) - 1])
    return sessions


def source_ids_for_results(
    results: list[dict[str, Any]], source_session_ids: list[str]
) -> list[set[str]]:
    return [source_ids_for_refs(result["refs"], source_session_ids) for result in results]


def retrieval_metrics(ranked: list[set[str]], gold: set[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    first_rank = next(
        (rank for rank, sessions in enumerate(ranked, start=1) if sessions & gold),
        None,
    )
    metrics["mrr"] = 0.0 if first_rank is None else 1.0 / first_rank
    for k in KS:
        retrieved = set().union(*ranked[:k]) if ranked[:k] else set()
        matched = retrieved & gold
        metrics[f"hit@{k}"] = float(bool(matched))
        metrics[f"recall@{k}"] = len(matched) / len(gold)
        metrics[f"all@{k}"] = float(matched == gold)
    return metrics


def aggregate(records: list[dict[str, Any]], condition: str) -> dict[str, float]:
    keys = ["mrr"] + [f"{name}@{k}" for k in KS for name in ("hit", "recall", "all")]
    return {key: mean(row[condition]["metrics"][key] for row in records) for key in keys}


def evaluate(
    method: str,
    limit: int | None,
    *,
    run_root: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    items = select_items(run_root, dataset_path, limit)
    encoder = None
    if method == "embedding":
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="mps",
            local_files_only=True,
        )

    records = []
    started = time.perf_counter()
    for number, item in enumerate(items, start=1):
        row = item["dataset"]
        checkpoint = item["checkpoint"]
        source_ids = checkpoint["input"]["source_session_ids"]
        gold = set(row["answer_session_ids"])
        memory_dir = item["item_dir"] / "memory"

        if method == "bm25":
            index = MemoryBM25Index(memory_dir, persist=False)
            events = index.events
        else:
            index = MemoryEmbeddingIndex(memory_dir, encoder=encoder)
            events = index._events()

        memory_sessions = set()
        dated_gold_sessions = set()
        semantic_intervals = []
        for event in events:
            event_sessions = source_ids_for_refs(event.refs, source_ids)
            memory_sessions.update(event_sessions)
            matched_gold = event_sessions & gold
            if not matched_gold or not event.dates:
                continue
            dated_gold_sessions.update(matched_gold)
            semantic_intervals.extend(temporal_bounds(value) for value in event.dates)
        retrievable_gold = memory_sessions & gold
        filter_eligible = bool(semantic_intervals) and dated_gold_sessions >= retrievable_gold
        if filter_eligible:
            date_from = min(start for start, _ in semantic_intervals).isoformat()
            date_to = (max(end for _, end in semantic_intervals) - timedelta(days=1)).isoformat()
        else:
            date_from = date_to = None

        unfiltered_results = index.search(row["question"], top_k=max(KS))
        filtered_results = index.search(
            row["question"],
            top_k=max(KS),
            date_from=date_from,
            date_to=date_to,
        )
        all_ranked = source_ids_for_results(unfiltered_results, source_ids)
        filtered_ranked = source_ids_for_results(filtered_results, source_ids)
        filtered_candidates = sum(
            event_matches_time_window(event, date_from, date_to) for event in events
        )
        records.append(
            {
                "dataset_index": item["dataset_index"],
                "question_id": row["question_id"],
                "question": row["question"],
                "answer_session_ids": sorted(gold),
                "filter_eligible": filter_eligible,
                "oracle_date_from": date_from,
                "oracle_date_to": date_to,
                "events": len(events),
                "filtered_candidates": filtered_candidates,
                "memory_evidence_recall": len(memory_sessions & gold) / len(gold),
                "unfiltered": {
                    "metrics": retrieval_metrics(all_ranked, gold),
                    "result_events": [result.get("event_id", result.get("event")) for result in unfiltered_results],
                },
                "oracle_filtered": {
                    "metrics": retrieval_metrics(filtered_ranked, gold),
                    "result_events": [result.get("event_id", result.get("event")) for result in filtered_results],
                },
            }
        )
        if number % 10 == 0 or number == len(items):
            print(f"{method}: {number}/{len(items)}", flush=True)

    paired = {}
    for k in KS:
        before = f"recall@{k}"
        deltas = [
            row["oracle_filtered"]["metrics"][before]
            - row["unfiltered"]["metrics"][before]
            for row in records
        ]
        paired[before] = {
            "improved": sum(delta > 0 for delta in deltas),
            "unchanged": sum(delta == 0 for delta in deltas),
            "degraded": sum(delta < 0 for delta in deltas),
            "mean_delta": mean(deltas),
        }
    return {
        "method": method,
        "protocol": "same query/index/top_k; oracle bounds span semantic dates on memory events citing gold evidence sessions",
        "oracle_warning": "The filtered condition is a retrieval ceiling, not an end-to-end Agent result.",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "memory_run": str(run_root),
        "sample_count": len(records),
        "available_temporal_questions": len(records),
        "filter_eligible_questions": sum(row["filter_eligible"] for row in records),
        "elapsed_s": time.perf_counter() - started,
        "mean_events": mean(row["events"] for row in records),
        "mean_filtered_candidates": mean(row["filtered_candidates"] for row in records),
        "mean_memory_evidence_recall": mean(row["memory_evidence_recall"] for row in records),
        "unfiltered": aggregate(records, "unfiltered"),
        "oracle_filtered": aggregate(records, "oracle_filtered"),
        "paired": paired,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("bm25", "embedding"), required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = evaluate(
        args.method,
        args.limit,
        run_root=args.run_root,
        dataset_path=args.data,
    )
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "method", "sample_count", "elapsed_s", "mean_events",
        "mean_filtered_candidates", "mean_memory_evidence_recall",
        "unfiltered", "oracle_filtered", "paired",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
