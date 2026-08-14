"""
Unified judge scoring for all methods.

For each method's results, use the same LLM to:
1. Generate an answer from retrieved context (for Mem0)
2. Score the answer against gold with LJ judge

Usage:
    python3 run_judge.py --nativemem results_v2_full/eval_50q.json --mem0 results_mem0_full/mem0_sample0.json
"""
import json
import argparse
import os
import sys
import time
import re
from openai import OpenAI

ALIYUN_KEY = os.environ.get("ALIYUN_KEY",
    "sk-sp-D.LIEPX.T6GA.MEYCIQCmomYgcDaqBtAWGXYyua8ve/6RPImY/+0W2BuPqNXzZQIhAJwxunRW89LTfdar0B2A59wjemXSIAHYOPGBgIluBUCh")
ALIYUN_BASE = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-v4-flash")
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "deepseek-v4-flash")

client = OpenAI(api_key=ALIYUN_KEY, base_url=ALIYUN_BASE)


def generate_answer(question, context):
    """Generate answer from retrieved context using shared answer model."""
    prompt = f"""Answer the question using ONLY the provided context. If the context does not contain the answer, say "NOT FOUND".

Context:
{context}

Question: {question}

Answer concisely."""
    try:
        resp = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, temperature=0,
        )
        answer = resp.choices[0].message.content.strip()
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        return answer
    except Exception as e:
        return f"ERROR: {e}"


def judge_score(question, gold, answer):
    """Score answer 0-100."""
    prompt = f"""Score 0-100 based on factual accuracy and completeness.

Question: {question}
Ground Truth: {gold}
System Answer: {answer}

0: completely wrong or NOT FOUND when answer exists.
25: related topic but key facts wrong.
50: core facts correct but important details missing.
75: mostly correct with minor omissions.
100: all factual details correct.

Output ONLY a number 0-100."""
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10, temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        nums = re.findall(r'\d+', content)
        return int(nums[0]) if nums else 0
    except:
        return 0


def score_nativemem(results_file):
    """Score NativeMem results (already has answers)."""
    with open(results_file) as f:
        data = json.load(f)

    results = data.get("results", data) if isinstance(data, dict) else data
    scored = []
    for r in results:
        question = r["question"]
        gold = r["gold"]
        answer = r.get("answer", "NOT FOUND")
        score = judge_score(question, gold, answer)
        scored.append({
            "question": question, "gold": gold,
            "answer": answer[:200], "score": score,
            "category": r.get("category", -1),
            "steps": r.get("steps", 0),
            "time": r.get("time", 0),
        })
    return scored


def score_mem0(results_file):
    """Score Mem0 results (need to generate answers from context first)."""
    with open(results_file) as f:
        data = json.load(f)

    results = data.get("results", [])
    scored = []
    for r in results:
        question = r["question"]
        gold = r["gold"]
        context = r.get("context", "")

        # Generate answer from context
        if context and context != "No relevant memories found.":
            answer = generate_answer(question, context)
        else:
            answer = "NOT FOUND"

        score = judge_score(question, gold, answer)
        scored.append({
            "question": question, "gold": gold,
            "answer": answer[:200], "score": score,
            "category": r.get("category", -1),
            "context_len": len(context),
            "num_memories": r.get("num_memories", 0),
            "retrieve_time": r.get("retrieve_time", 0),
        })
    return scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nativemem", default=None)
    parser.add_argument("--mem0", default=None)
    parser.add_argument("--outdir", default="judge_results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    all_results = {}

    if args.nativemem:
        print("=== Scoring NativeMem v2 ===")
        scored = score_nativemem(args.nativemem)
        avg = sum(s["score"] for s in scored) / len(scored)
        print(f"  LJ Score: {avg:.1f} ({len(scored)} questions)")

        # Per category
        from collections import defaultdict
        cats = defaultdict(list)
        for s in scored:
            cats[s["category"]].append(s["score"])
        for c in sorted(cats):
            cat_avg = sum(cats[c]) / len(cats[c])
            print(f"  Category {c}: {cat_avg:.1f} (n={len(cats[c])})")

        all_results["nativemem"] = {"avg_score": avg, "results": scored}
        with open(os.path.join(args.outdir, "nativemem_scored.json"), "w") as f:
            json.dump({"avg_score": avg, "results": scored}, f, indent=2)

    if args.mem0:
        print("\n=== Scoring Mem0 ===")
        scored = score_mem0(args.mem0)
        avg = sum(s["score"] for s in scored) / len(scored)
        print(f"  LJ Score: {avg:.1f} ({len(scored)} questions)")

        from collections import defaultdict
        cats = defaultdict(list)
        for s in scored:
            cats[s["category"]].append(s["score"])
        for c in sorted(cats):
            cat_avg = sum(cats[c]) / len(cats[c])
            print(f"  Category {c}: {cat_avg:.1f} (n={len(cats[c])})")

        all_results["mem0"] = {"avg_score": avg, "results": scored}
        with open(os.path.join(args.outdir, "mem0_scored.json"), "w") as f:
            json.dump({"avg_score": avg, "results": scored}, f, indent=2)

    # Summary
    if all_results:
        print("\n=== SUMMARY ===")
        for method, data in all_results.items():
            print(f"  {method}: LJ = {data['avg_score']:.1f}")


if __name__ == "__main__":
    main()
