"""
Pilot experiment: Wiki Memory vs Flat Memory vs Full Context
Using LongMemEval oracle data (only evidence sessions, ~3 sessions per question)

Usage:
    python3 run_pilot.py --n 10 --method wiki
    python3 run_pilot.py --n 10 --method flat
    python3 run_pilot.py --n 10 --method fullctx
"""

import json
import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))
from wiki_memory import WikiMemory, SimpleMemory, call_llm


def load_data(data_path: str, n: int = 10) -> list:
    with open(data_path) as f:
        data = json.load(f)
    # Take first n questions, skip abstention questions
    filtered = [d for d in data if not d["question_id"].endswith("_abs")]
    return filtered[:n]


def sessions_to_memories(sessions: list, dates: list) -> list[dict]:
    """Extract memory-worthy items from chat sessions."""
    memories = []
    for i, (session, date) in enumerate(zip(sessions, dates)):
        for turn in session:
            if turn["role"] == "user":
                # User messages often contain facts worth remembering
                content = turn["content"].strip()
                if len(content) > 20:  # Skip very short messages
                    memories.append({
                        "text": content,
                        "timestamp": date,
                        "session_id": i,
                        "has_answer": turn.get("has_answer", False)
                    })
            elif turn["role"] == "assistant" and turn.get("has_answer"):
                # Assistant messages with answers are also useful
                content = turn["content"].strip()
                if len(content) > 20:
                    memories.append({
                        "text": content,
                        "timestamp": date,
                        "session_id": i,
                        "has_answer": True
                    })
    return memories


def run_wiki_memory(questions: list, output_path: str):
    """Run our Wiki Memory system."""
    results = []

    for idx, q in enumerate(questions):
        print(f"\n[{idx+1}/{len(questions)}] Question: {q['question'][:80]}...")

        # Create fresh wiki for each question
        wiki_dir = f"/tmp/wiki_memory_pilot_{idx}"
        if os.path.exists(wiki_dir):
            import shutil
            shutil.rmtree(wiki_dir)

        wiki = WikiMemory(wiki_dir)

        # Extract memories from sessions
        memories = sessions_to_memories(
            q["haystack_sessions"], q["haystack_dates"]
        )

        print(f"  Storing {len(memories)} memories...")
        t0 = time.time()

        # Store memories one by one (simulating streaming)
        for mem in memories:
            wiki.store(mem["text"], mem["timestamp"])

        store_time = time.time() - t0
        print(f"  Storage done in {store_time:.1f}s ({wiki.stats['llm_calls']} LLM calls)")

        # Retrieve and answer
        print(f"  Retrieving...")
        t0 = time.time()
        result = wiki.retrieve(q["question"])
        retrieve_time = time.time() - t0

        print(f"  Answer: {result['answer'][:100]}...")
        print(f"  Files visited: {result['num_hops']}")
        print(f"  Expected: {q['answer']}")

        results.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "question_type": q["question_type"],
            "hypothesis": result["answer"],
            "reference": q["answer"],
            "files_visited": result.get("files_visited", []),
            "num_hops": result["num_hops"],
            "num_memories": len(memories),
            "store_time": store_time,
            "retrieve_time": retrieve_time,
            "llm_calls": wiki.stats["llm_calls"],
        })

    # Save results
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResults saved to {output_path}")
    return results


def run_flat_memory(questions: list, output_path: str):
    """Run baseline: flat memory list (all memories in one prompt)."""
    results = []

    for idx, q in enumerate(questions):
        print(f"\n[{idx+1}/{len(questions)}] Question: {q['question'][:80]}...")

        storage_dir = f"/tmp/flat_memory_pilot_{idx}"
        if os.path.exists(storage_dir):
            import shutil
            shutil.rmtree(storage_dir)

        flat = SimpleMemory(storage_dir)

        memories = sessions_to_memories(
            q["haystack_sessions"], q["haystack_dates"]
        )

        print(f"  Storing {len(memories)} memories...")
        t0 = time.time()
        for mem in memories:
            flat.store(mem["text"], mem["timestamp"])
        store_time = time.time() - t0

        print(f"  Retrieving...")
        t0 = time.time()
        result = flat.retrieve(q["question"])
        retrieve_time = time.time() - t0

        print(f"  Answer: {result['answer'][:100]}...")
        print(f"  Expected: {q['answer']}")

        results.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "question_type": q["question_type"],
            "hypothesis": result["answer"],
            "reference": q["answer"],
            "num_hops": 1,
            "num_memories": len(memories),
            "store_time": store_time,
            "retrieve_time": retrieve_time,
            "llm_calls": flat.stats["llm_calls"],
        })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResults saved to {output_path}")
    return results


def run_full_context(questions: list, output_path: str):
    """Run upper bound: dump all sessions into context."""
    results = []

    for idx, q in enumerate(questions):
        print(f"\n[{idx+1}/{len(questions)}] Question: {q['question'][:80]}...")

        # Build full context from all sessions
        context_parts = []
        for session, date in zip(q["haystack_sessions"], q["haystack_dates"]):
            context_parts.append(f"\n--- Session ({date}) ---")
            for turn in session:
                context_parts.append(f"{turn['role']}: {turn['content']}")
        full_context = "\n".join(context_parts)

        prompt = f"""Here is the complete chat history with a user:

{full_context}

Based on this history, answer the following question:
"{q['question']}"

Give a concise answer. If the information is not in the history, say "I don't have this information."
Respond with ONLY the answer."""

        t0 = time.time()
        answer = call_llm(prompt)
        retrieve_time = time.time() - t0

        print(f"  Answer: {answer[:100]}...")
        print(f"  Expected: {q['answer']}")

        results.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "question_type": q["question_type"],
            "hypothesis": answer,
            "reference": q["answer"],
            "num_hops": 0,
            "retrieve_time": retrieve_time,
            "llm_calls": 1,
        })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResults saved to {output_path}")
    return results


def evaluate_results(results: list) -> dict:
    """Simple evaluation: check if reference answer appears in hypothesis."""
    correct = 0
    total = len(results)
    by_type = {}

    for r in results:
        ref = r["reference"].lower().strip()
        hyp = r["hypothesis"].lower().strip()
        # Simple containment check
        is_correct = ref in hyp or hyp in ref
        if is_correct:
            correct += 1

        qtype = r["question_type"]
        if qtype not in by_type:
            by_type[qtype] = {"correct": 0, "total": 0}
        by_type[qtype]["total"] += 1
        if is_correct:
            by_type[qtype]["correct"] += 1

    metrics = {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "by_type": {
            k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0}
            for k, v in by_type.items()
        }
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["wiki", "flat", "fullctx", "all"], default="all")
    parser.add_argument("--n", type=int, default=5, help="Number of questions to test")
    parser.add_argument("--data", default="longmemeval/data/longmemeval_oracle.json")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    questions = load_data(args.data, args.n)
    print(f"Loaded {len(questions)} questions")

    methods = [args.method] if args.method != "all" else ["fullctx", "flat", "wiki"]
    all_metrics = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running method: {method}")
        print(f"{'='*60}")

        output_path = os.path.join(args.outdir, f"pilot_{method}.jsonl")

        if method == "wiki":
            results = run_wiki_memory(questions, output_path)
        elif method == "flat":
            results = run_flat_memory(questions, output_path)
        elif method == "fullctx":
            results = run_full_context(questions, output_path)

        metrics = evaluate_results(results)
        all_metrics[method] = metrics

        print(f"\n--- {method} Results ---")
        print(f"Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
        for qtype, m in metrics["by_type"].items():
            print(f"  {qtype}: {m['accuracy']:.1%} ({m['correct']}/{m['total']})")

    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        for method, metrics in all_metrics.items():
            print(f"  {method:10s}: {metrics['accuracy']:.1%}")

    # Save summary
    summary_path = os.path.join(args.outdir, "pilot_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
