#!/usr/bin/env python3
"""Print a no-network Writer token-batch plan for one LongMemEval item."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.runners.longmemeval import support  # noqa: E402
from src.conversation import (  # noqa: E402
    benchmark_source_id,
    normalize_date,
    session_content,
)
from src.management.api import render_writer_input  # noqa: E402
from src.runtime.capacity import pack_complete_messages  # noqa: E402
from src.runtime.tokenization import TokenCounter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--item-index", type=int, default=75)
    parser.add_argument("--model", required=True)
    parser.add_argument("--token-cap", type=int, required=True)
    args = parser.parse_args()
    if args.token_cap < 1:
        parser.error("--token-cap must be positive")

    dataset = support.load_dataset(
        args.data.expanduser().resolve(),
        expected_count=support.EXPECTED_LONGMEMEVAL_SIZE,
    )
    support.validate_longmemeval_s(dataset)
    item = dataset[args.item_index]
    conversation = support.to_conversation(item, args.item_index)
    sessions = []
    index = 1
    while f"session_{index}" in conversation:
        turns, refs = session_content(conversation[f"session_{index}"], index)
        sessions.append({
            "session_index": index,
            "observation_date": normalize_date(
                conversation.get(f"session_{index}_date_time", "")
            ),
            "turns": turns,
            "refs": [benchmark_source_id(ref) for ref in refs],
        })
        index += 1

    counter = TokenCounter.resolve(requested_model=args.model)
    batches = pack_complete_messages(
        sessions,
        max_input_tokens=args.token_cap,
        render_batch=render_writer_input,
        token_counter=counter,
    )
    payload = {
        "dataset_index": args.item_index,
        "question_id": str(item["question_id"]),
        "model": args.model,
        "token_cap": args.token_cap,
        "tokenizer": counter.identity,
        "total_sessions": len(sessions),
        "total_source_turns": sum(len(session["turns"]) for session in sessions),
        "total_batches": len(batches),
        "batches": [{
            "index": batch_index,
            "input_tokens": counter.count(render_writer_input(batch)),
            "session_indices": sorted({
                int(fragment["session_index"]) for fragment in batch
            }),
            "source_turns": sum(len(fragment["turns"]) for fragment in batch),
        } for batch_index, batch in enumerate(batches)],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
