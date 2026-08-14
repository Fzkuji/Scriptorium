"""Build wiki + evaluate on LongMemEval dataset.

Usage:
    # Build wiki for sample 0 then evaluate
    python3 longmemeval_runner.py --sample 0 --build --eval --max-qa 10

    # Only evaluate (wiki already built)
    python3 longmemeval_runner.py --sample 0 --eval --max-qa 10
"""
import json
import argparse
import os
import sys
import time
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import memory_builder
from memory_builder import process_conversation_turn, verify_storage, llm_judge_score

sys.stdout.reconfigure(line_buffering=True)

CHUNK_SIZE = 10  # turns per chunk


def build_wiki(data, sample_idx, outdir, model):
    """Build wiki from LongMemEval sessions."""
    sample = data[sample_idx]
    sessions = sample["haystack_sessions"]
    dates = sample.get("haystack_dates", [""] * len(sessions))
    session_ids = sample.get("haystack_session_ids", [f"session_{i}" for i in range(len(sessions))])

    memory_dir = os.path.join(outdir, f"lme_sample{sample_idx}_{model}")
    if os.path.exists(memory_dir):
        shutil.rmtree(memory_dir)
    os.makedirs(memory_dir)

    memory_builder.ALIYUN_MODEL = model
    memory_builder.TOTAL_CALLS = 0
    memory_builder.TOTAL_TOKENS = 0
    memory_builder.CALL_LOG = []

    print(f"Sample {sample_idx}: {len(sessions)} sessions")
    print(f"Memory dir: {memory_dir}")
    print(f"Model: {model}")

    t0 = time.time()
    total_chunks = 0

    for si, (session, date, sid) in enumerate(zip(sessions, dates, session_ids)):
        # Convert session turns to text
        turns_text = ""
        for turn in session:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            turns_text += f"{role}: {content}\n\n"

        if not turns_text.strip():
            continue

        # Split into chunks
        turns = session
        for chunk_start in range(0, len(turns), CHUNK_SIZE):
            chunk = turns[chunk_start:chunk_start + CHUNK_SIZE]
            chunk_text = ""
            for t in chunk:
                chunk_text += f"{t.get('role', 'user')}: {t.get('content', '')}\n\n"

            if not chunk_text.strip():
                continue

            total_chunks += 1
            if total_chunks % 10 == 0:
                print(f"  Chunk {total_chunks} (session {si+1}/{len(sessions)})")

            process_conversation_turn(chunk_text, date, memory_dir)

    build_time = time.time() - t0
    file_count = len(list(Path(memory_dir).rglob("*.md")))
    total_size = sum(f.stat().st_size for f in Path(memory_dir).rglob("*.md"))

    print(f"\nBuild complete: {total_chunks} chunks, {build_time:.0f}s")
    print(f"Wiki: {file_count} files, {total_size // 1024}KB")
    print(f"LLM calls: {memory_builder.TOTAL_CALLS}, tokens: {memory_builder.TOTAL_TOKENS}")

    # Save stats
    stats = {
        "sample": sample_idx,
        "model": model,
        "sessions": len(sessions),
        "chunks": total_chunks,
        "build_time": build_time,
        "llm_calls": memory_builder.TOTAL_CALLS,
        "total_tokens": memory_builder.TOTAL_TOKENS,
        "file_count": file_count,
        "total_size": total_size,
    }
    with open(os.path.join(outdir, f"lme_stats_sample{sample_idx}.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return memory_dir


def evaluate(data, sample_idx, memory_dir, model, max_qa):
    """Evaluate wiki on LongMemEval questions."""
    sample = data[sample_idx]
    question = sample["question"]
    answer = sample["answer"]
    q_type = sample["question_type"]

    memory_builder.ALIYUN_MODEL = model
    memory_builder.TOTAL_CALLS = 0
    memory_builder.TOTAL_TOKENS = 0
    memory_builder.CALL_LOG = []

    # LongMemEval: each sample has ONE question
    qa_list = [{"question": question, "answer": answer, "category": q_type}]

    print(f"\nEvaluating sample {sample_idx}")
    print(f"  Type: {q_type}")
    print(f"  Q: {question}")
    print(f"  Expected: {answer}")

    correct, tested = verify_storage(memory_dir, qa_list, n_questions=1)

    # Read verify json for details
    verify_path = os.path.join(os.path.dirname(memory_dir), f"verify_{os.path.basename(memory_dir)}.json")
    if os.path.exists(verify_path):
        vdata = json.load(open(verify_path))
        results = vdata.get("results", [])
        if results:
            r = results[0]
            print(f"  LJ: {r.get('judge_score', '?')}")
            print(f"  Response: {r.get('response', '')[:100]}")
            print(f"  Files: {r.get('files_visited', [])}")

    return correct, tested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="longmemeval/data/longmemeval_s_cleaned.json")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--outdir", default="longmemeval_test")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--wiki-model", default=None)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--max-qa", type=int, default=1)
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    print(f"LongMemEval: {len(data)} samples")

    os.makedirs(args.outdir, exist_ok=True)
    wiki_model = args.wiki_model or args.model
    memory_dir = os.path.join(args.outdir, f"lme_sample{args.sample}_{wiki_model}")

    if args.build:
        memory_dir = build_wiki(data, args.sample, args.outdir, args.model)

    if args.eval:
        if not os.path.exists(memory_dir):
            print(f"Wiki not built: {memory_dir}")
            return
        evaluate(data, args.sample, memory_dir, args.model, args.max_qa)


if __name__ == "__main__":
    main()
