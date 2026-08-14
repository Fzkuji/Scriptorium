"""
Run LLM judge evaluation on collected LoCoMo answers.
Uses codex to judge if each answer is semantically correct.

Usage:
    python3 run_eval.py results_locomo/locomo_flat.jsonl
    python3 run_eval.py results_locomo/locomo_summary.jsonl
    python3 run_eval.py results_locomo/locomo_wiki.jsonl
    python3 run_eval.py --all
"""

import json
import sys
import os
import subprocess
import tempfile
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)

REPO_DIR = str(Path(__file__).resolve().parents[3])


def call_codex(prompt: str, timeout: int = 120) -> str:
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


def judge(question: str, reference: str, hypothesis: str) -> bool:
    prompt = f"""I will give you a question, a correct answer, and a model's response.
Answer "yes" if the response contains or is equivalent to the correct answer, even if worded differently.
Answer "no" if it does not contain the correct information.
Do not penalize minor formatting differences (dates, capitalization, etc).
For temporal questions, do not penalize off-by-one day errors.

Question: {question}
Correct Answer: {reference}
Model Response: {hypothesis}

Is the model response correct? Answer yes or no only."""

    try:
        result = call_codex(prompt, timeout=120)
        return result.strip().lower().startswith("yes")
    except Exception as e:
        print(f"    Judge error: {e}")
        return False


def evaluate_file(filepath: str) -> dict:
    results = []
    with open(filepath) as f:
        for line in f:
            results.append(json.loads(line))

    method = Path(filepath).stem.replace("locomo_", "")
    print(f"\n{'='*50}")
    print(f"Evaluating: {method} ({len(results)} questions)")
    print(f"{'='*50}")

    correct = 0
    by_cat = {}
    for i, r in enumerate(results):
        is_correct = judge(r["question"], r["reference"], r["hypothesis"])
        r["llm_correct"] = is_correct
        if is_correct:
            correct += 1
        cat = str(r.get("category", "unknown"))
        if cat not in by_cat:
            by_cat[cat] = {"correct": 0, "total": 0}
        by_cat[cat]["total"] += 1
        if is_correct:
            by_cat[cat]["correct"] += 1

        status = "OK" if is_correct else "MISS"
        print(f"  [{i+1}/{len(results)}] [{status}] {r['question'][:50]}...")
        if not is_correct:
            print(f"         Got: {r['hypothesis'][:60]}")
            print(f"         Exp: {r['reference'][:60]}")

    accuracy = correct / len(results) if results else 0
    metrics = {
        "method": method,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "by_category": {
            k: {**v, "accuracy": v["correct"]/v["total"] if v["total"] else 0}
            for k, v in by_cat.items()
        }
    }

    print(f"\n--- {method} (LLM Judge) ---")
    print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(results)})")
    for cat, m in sorted(metrics["by_category"].items()):
        print(f"    Cat {cat}: {m['accuracy']:.1%} ({m['correct']}/{m['total']})")

    # Save evaluated results
    eval_path = filepath.replace(".jsonl", "_eval.jsonl")
    with open(eval_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return metrics


def main():
    if "--all" in sys.argv:
        files = sorted(Path("results_locomo").glob("locomo_*.jsonl"))
        files = [f for f in files if "_eval" not in f.name]
    else:
        files = [Path(a) for a in sys.argv[1:] if a.endswith(".jsonl")]

    if not files:
        print("Usage: python3 run_eval.py --all")
        print("   or: python3 run_eval.py results_locomo/locomo_flat.jsonl")
        return

    all_metrics = {}
    for f in files:
        m = evaluate_file(str(f))
        all_metrics[m["method"]] = m

    if len(all_metrics) > 1:
        print(f"\n{'='*50}")
        print("FINAL COMPARISON (LLM Judge)")
        print(f"{'='*50}")
        print(f"{'Method':<12} {'Accuracy':>10} {'Correct':>8}")
        for method, m in all_metrics.items():
            print(f"{method:<12} {m['accuracy']:>9.1%} {m['correct']:>4}/{m['total']}")

        print(f"\n--- Known Baselines ---")
        print(f"  Mem0 (LoCoMo):      66.88%")
        print(f"  Memobase (LoCoMo):  ~80%")

    summary_path = "results_locomo/eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
