"""
LoCoMo Benchmark Experiment
Run wiki memory vs baselines on LoCoMo dataset.

Strategy: For each QA pair, give codex the conversation history + question,
let it build wiki and answer in one call (much faster than per-memory calls).

Usage:
    python3 run_locomo.py --n_samples 2 --n_qa 10 --method wiki
    python3 run_locomo.py --n_samples 2 --n_qa 10 --method all
"""

import json
import argparse
import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)

REPO_DIR = str(Path(__file__).resolve().parents[3])


def call_codex(prompt: str, timeout: int = 180) -> str:
    """Call codex exec with prompt via stdin file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_file = f.name
    try:
        result = subprocess.run(
            ["codex", "exec", "-"],
            capture_output=True, text=True, timeout=timeout,
            cwd=REPO_DIR,
            stdin=open(prompt_file, "r")
        )
    finally:
        os.unlink(prompt_file)
    output = result.stdout.strip()
    if "tokens used" in output:
        parts = output.split("tokens used")
        after = parts[-1].strip()
        lines = after.split("\n", 1)
        if len(lines) > 1:
            return lines[1].strip()
        before = parts[0].strip()
        if "\ncodex\n" in before:
            return before.split("\ncodex\n")[-1].strip()
    return output


def format_conversation(conversation: dict, max_sessions: int = 20) -> str:
    """Format LoCoMo conversation dict into text."""
    parts = []
    session_keys = sorted(
        [k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1])
    )
    selected = session_keys[-max_sessions:] if len(session_keys) > max_sessions else session_keys
    for key in selected:
        date_key = key + "_date_time"
        date = conversation.get(date_key, "")
        parts.append(f"\n--- {key} ({date}) ---")
        session = conversation[key]
        for turn in session:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            if len(text) > 500:
                text = text[:500] + "..."
            parts.append(f"{speaker}: {text}")
    return "\n".join(parts)


def run_wiki_qa(conversation_text: str, question: str) -> str:
    """Use wiki-based memory approach: organize then retrieve."""
    prompt = f"""You are a personal assistant with a memory wiki system.

TASK: Read the conversation history below, organize key information into a structured wiki
(mentally create files by topic/person/event), then answer the question.

When organizing, think about WHERE you would look for each piece of information later:
- Use concrete nouns as file names (people's names, project names, dates)
- Group related memories together
- Note temporal relationships

CONVERSATION HISTORY:
{conversation_text}

QUESTION: {question}

First, briefly describe how you would organize these memories into wiki files (2-3 lines).
Then answer the question concisely.

Format your response as:
WIKI STRUCTURE: <brief description of files you'd create>
ANSWER: <your concise answer>"""

    response = call_codex(prompt, timeout=120)
    # Extract just the answer
    if "ANSWER:" in response:
        return response.split("ANSWER:")[-1].strip()
    return response


def run_flat_qa(conversation_text: str, question: str) -> str:
    """Baseline: dump all conversation and ask."""
    prompt = f"""Here is a conversation history with a user:

{conversation_text}

Based on this history, answer the following question concisely:
"{question}"

If the answer is not in the conversation, say "I don't know."
Give ONLY the answer, nothing else."""

    return call_codex(prompt, timeout=120)


def run_summary_qa(conversation_text: str, question: str) -> str:
    """Baseline: summarize then answer (similar to MemoryBank approach)."""
    prompt = f"""You are a personal assistant. First, read the conversation history and extract
the key facts and events as a bullet-point summary. Then answer the question.

CONVERSATION HISTORY:
{conversation_text}

QUESTION: {question}

Format:
KEY FACTS:
- fact 1
- fact 2
...
ANSWER: <concise answer>"""

    response = call_codex(prompt, timeout=120)
    if "ANSWER:" in response:
        return response.split("ANSWER:")[-1].strip()
    return response


def evaluate_single(question: str, reference: str, hypothesis: str) -> bool:
    """Use codex to judge if answer is correct (like official eval)."""
    prompt = f"""I will give you a question, a correct answer, and a model's response.
Answer "yes" if the response contains the correct answer (even if worded differently).
Answer "no" if it does not.

Question: {question}
Correct Answer: {reference}
Model Response: {hypothesis}

Is the model response correct? Answer yes or no only."""

    result = call_codex(prompt, timeout=120)
    return result.strip().lower().startswith("yes")


def load_locomo(data_path: str, n_samples: int, n_qa: int) -> list:
    """Load LoCoMo data and prepare QA instances."""
    with open(data_path) as f:
        data = json.load(f)

    instances = []
    for sample in data[:n_samples]:
        conversation = sample["conversation"]
        conv_text = format_conversation(conversation)
        qa_list = sample["qa"][:n_qa]
        for qa in qa_list:
            instances.append({
                "sample_id": sample["sample_id"],
                "conversation_text": conv_text,
                "question": qa["question"],
                "answer": str(qa["answer"]),
                "category": qa.get("category", "unknown"),
                "evidence": qa.get("evidence", []),
            })
    return instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["wiki", "flat", "summary", "all"], default="all")
    parser.add_argument("--n_samples", type=int, default=2, help="Number of LoCoMo samples")
    parser.add_argument("--n_qa", type=int, default=10, help="QA pairs per sample")
    parser.add_argument("--data", default="locomo/data/locomo10.json")
    parser.add_argument("--outdir", default="results_locomo")
    parser.add_argument("--skip_eval", action="store_true", help="Skip LLM-based evaluation")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    instances = load_locomo(args.data, args.n_samples, args.n_qa)
    print(f"Loaded {len(instances)} QA instances from {args.n_samples} samples")

    methods = [args.method] if args.method != "all" else ["flat", "summary", "wiki"]
    all_metrics = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method} ({len(instances)} questions)")
        print(f"{'='*60}")

        results = []
        for idx, inst in enumerate(instances):
            print(f"  [{idx+1}/{len(instances)}] Q: {inst['question'][:60]}...")
            t0 = time.time()

            if method == "wiki":
                hyp = run_wiki_qa(inst["conversation_text"], inst["question"])
            elif method == "flat":
                hyp = run_flat_qa(inst["conversation_text"], inst["question"])
            elif method == "summary":
                hyp = run_summary_qa(inst["conversation_text"], inst["question"])

            elapsed = time.time() - t0
            print(f"       A: {hyp[:80]}  ({elapsed:.1f}s)")
            print(f"       Expected: {str(inst['answer'])[:80]}")

            results.append({
                "sample_id": inst["sample_id"],
                "question": inst["question"],
                "hypothesis": hyp,
                "reference": inst["answer"],
                "category": inst["category"],
                "time": elapsed,
            })

        # Save results first (before eval, in case eval crashes)
        out_path = os.path.join(args.outdir, f"locomo_{method}.jsonl")
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Answers saved to {out_path}")

        # Evaluate with LLM judge
        if not args.skip_eval:
            print(f"\n  Evaluating with LLM judge...")
            correct = 0
            by_cat = {}
            for r in results:
                is_correct = evaluate_single(r["question"], r["reference"], r["hypothesis"])
                r["correct"] = is_correct
                if is_correct:
                    correct += 1
                cat = str(r["category"])
                if cat not in by_cat:
                    by_cat[cat] = {"correct": 0, "total": 0}
                by_cat[cat]["total"] += 1
                if is_correct:
                    by_cat[cat]["correct"] += 1

            accuracy = correct / len(results) if results else 0
            metrics = {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(results),
                "by_category": {
                    k: {**v, "accuracy": v["correct"]/v["total"] if v["total"] else 0}
                    for k, v in by_cat.items()
                },
                "avg_time": sum(r["time"] for r in results) / len(results) if results else 0,
            }
        else:
            # Simple string match as fallback
            correct = sum(1 for r in results
                         if r["reference"].lower() in r["hypothesis"].lower()
                         or r["hypothesis"].lower() in r["reference"].lower())
            accuracy = correct / len(results) if results else 0
            metrics = {"accuracy": accuracy, "correct": correct, "total": len(results)}

        all_metrics[method] = metrics

        print(f"\n--- {method} Results ---")
        print(f"  Accuracy: {accuracy:.1%} ({metrics['correct']}/{metrics['total']})")
        if "by_category" in metrics:
            for cat, m in sorted(metrics["by_category"].items()):
                print(f"    Cat {cat}: {m['accuracy']:.1%} ({m['correct']}/{m['total']})")
        print(f"  Avg time/question: {metrics.get('avg_time', 0):.1f}s")

        # Re-save results with eval labels
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Comparison
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Method':<12} {'Accuracy':>10} {'Correct':>8} {'Total':>6} {'Avg Time':>10}")
        for method, m in all_metrics.items():
            print(f"{method:<12} {m['accuracy']:>9.1%} {m['correct']:>8}/{m['total']:<6} {m.get('avg_time',0):>9.1f}s")

    # Known baselines for reference
    print(f"\n--- Known Baselines (from papers) ---")
    print(f"  Mem0 (LoCoMo):        66.88%")
    print(f"  Memobase (LoCoMo):    ~80% (temporal 85.1%)")
    print(f"  Backboard (LoCoMo):   90.1% (commercial, unverified)")

    summary_path = os.path.join(args.outdir, "locomo_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
