"""
Standard evaluation: F1 + BLEU-1 + LLM-as-Judge (0/1 binary).
Aligns with LoCoMo benchmark conventions used by Mem0, Mnemis, MemMachine, etc.

Judge uses GPT-5.5 via ChatGPT proxy (localhost:8199).
F1/BLEU-1 are computed locally (no API needed).

Usage:
    python3 src/eval_standard.py --results results/nativemem-v3-locomo-s0/eval_standard_f1.json --label NativeMem  # 从项目根目录运行
    python3 src/eval_standard.py --results results/mem0-locomo-s0/mem0_scored.json --label Mem0
    python3 src/eval_standard.py --results results/amem-locomo-s0/results_199qa.json --label A-Mem --format amem
"""

import json
import argparse
import sys
import re
import os
import time
import numpy as np
from collections import defaultdict
from openai import OpenAI

sys.path.insert(0, "benchmarks/locomo/task_eval")
from evaluation import f1_score, f1 as multi_f1

# GPT-5.5 via ChatGPT proxy for judging
JUDGE_BASE = os.environ.get("JUDGE_BASE", "http://localhost:8199/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.5")

judge_client = OpenAI(api_key="not-needed", base_url=JUDGE_BASE)

# Mem0-style judge prompt (0/1 binary, 14-day date tolerance, partial credit)
JUDGE_SYSTEM = "You are evaluating conversational AI memory recall. Return JSON only."

JUDGE_PROMPT = """Label the generated answer as CORRECT or WRONG.

## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. Judge semantic meaning, not exact wording.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT.

5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea. Different wording should not result in WRONG if the underlying concept matches.

6. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer
- The answer addresses a completely different topic

## Question
Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return JSON: {{"reasoning": "one sentence", "label": "CORRECT" or "WRONG"}}"""


def compute_f1(prediction, gold, category):
    prediction = str(prediction)
    gold = str(gold)
    if category == 5:
        # Adversarial: keyword matching
        lower = prediction.lower()
        if "no information available" in lower or "not mentioned" in lower or "no information" in lower:
            return 1.0
        return 0.0
    elif category == 1:
        return multi_f1(prediction, gold)
    elif category == 3:
        gold = gold.split(";")[0].strip()
        return f1_score(prediction, gold)
    else:
        return f1_score(prediction, gold)


def compute_bleu1(prediction, gold):
    from collections import Counter
    # Strip markdown formatting
    prediction = re.sub(r'\*\*|__', '', str(prediction))
    gold = re.sub(r'\*\*|__', '', str(gold))
    pred_tokens = prediction.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    clipped = sum(min(pred_counts[w], gold_counts[w]) for w in pred_counts)
    precision = clipped / len(pred_tokens) if pred_tokens else 0
    bp = min(1.0, len(pred_tokens) / len(gold_tokens)) if gold_tokens else 0
    return bp * precision


def judge_binary(question, gold, response):
    prompt = JUDGE_PROMPT.format(question=question, answer=gold, response=response)
    try:
        resp = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200, temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        # Extract JSON
        if '"label"' in content:
            match = re.search(r'"label"\s*:\s*"(CORRECT|WRONG)"', content, re.IGNORECASE)
            if match:
                return 1.0 if match.group(1).upper() == "CORRECT" else 0.0
        if "CORRECT" in content.upper() and "WRONG" not in content.upper():
            return 1.0
        return 0.0
    except Exception as e:
        print(f"  Judge error: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--label", default="Method")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--format", default="auto", choices=["auto", "amem", "nativemem", "mem0"])
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    # Handle different result formats
    if isinstance(data, dict) and "individual_results" in data:
        # A-Mem format
        items = data["individual_results"]
        args.format = "amem"
    elif isinstance(data, dict) and "results" in data:
        items = data["results"]
    elif isinstance(data, list):
        items = data
    else:
        print("Unknown format")
        return

    # Normalize field names: need '_pred', '_gold', 'category' for each item
    for item in items:
        if args.format == "amem":
            item["_pred"] = str(item.get("prediction", ""))
            item["_gold"] = str(item.get("reference", ""))
            item["category"] = item.get("category", -1)
        elif "prediction" in item and "answer" in item and "gold" not in item:
            # NativeMem format: prediction=model answer, answer=gold
            item["_pred"] = str(item["prediction"])
            item["_gold"] = str(item["answer"])
        elif "gold" in item:
            # Mem0 format: answer=model answer, gold=gold answer
            item["_pred"] = str(item.get("answer", ""))
            item["_gold"] = str(item["gold"])
        else:
            item["_pred"] = str(item.get("answer", ""))
            item["_gold"] = str(item.get("ground_truth_answer", ""))

    cat_names = {1: "Multi-hop", 2: "Temporal", 3: "Session/Open-domain",
                 4: "Single-hop/Open-domain", 5: "Adversarial"}

    # Compute F1 and BLEU-1
    all_f1 = []
    all_bleu = []
    all_judge = []
    cats = defaultdict(lambda: {"f1": [], "bleu": [], "judge": []})

    for i, item in enumerate(items):
        q = item.get("question", "")
        gold = item["_gold"]
        pred = item["_pred"]
        cat = item.get("category", -1)

        f1_val = compute_f1(pred, gold, cat)
        bleu_val = compute_bleu1(pred, gold)
        all_f1.append(f1_val)
        all_bleu.append(bleu_val)
        cats[cat]["f1"].append(f1_val)
        cats[cat]["bleu"].append(bleu_val)

        if not args.skip_judge:
            judge_val = judge_binary(q, gold, pred)
            all_judge.append(judge_val)
            cats[cat]["judge"].append(judge_val)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(items)}: F1={np.mean(all_f1):.4f}, "
                      f"BLEU-1={np.mean(all_bleu):.4f}, "
                      f"Judge={np.mean(all_judge)*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"  {args.label} — Standard Evaluation ({len(items)} questions)")
    print(f"  Judge model: {JUDGE_MODEL}")
    print(f"{'='*60}")
    print(f"  {'Category':<20} {'n':>4}  {'F1':>6}  {'BLEU-1':>6}", end="")
    if all_judge:
        print(f"  {'Judge%':>6}", end="")
    print()
    print(f"  {'-'*50}")

    for c in sorted(cats):
        n = len(cats[c]["f1"])
        f1_avg = np.mean(cats[c]["f1"])
        bleu_avg = np.mean(cats[c]["bleu"])
        line = f"  {cat_names.get(c, f'Cat {c}'):<20} {n:>4}  {f1_avg:>6.4f}  {bleu_avg:>6.4f}"
        if cats[c]["judge"]:
            judge_avg = np.mean(cats[c]["judge"]) * 100
            line += f"  {judge_avg:>5.1f}%"
        print(line)

    print(f"  {'-'*50}")
    overall_f1 = np.mean(all_f1)
    overall_bleu = np.mean(all_bleu)
    line = f"  {'Overall':<20} {len(items):>4}  {overall_f1:>6.4f}  {overall_bleu:>6.4f}"
    if all_judge:
        overall_judge = np.mean(all_judge) * 100
        line += f"  {overall_judge:>5.1f}%"
    print(line)

    # Save
    out = {
        "label": args.label, "judge_model": JUDGE_MODEL,
        "n": len(items),
        "overall_f1": float(overall_f1),
        "overall_bleu1": float(overall_bleu),
        "overall_judge": float(np.mean(all_judge) * 100) if all_judge else None,
        "per_category": {
            str(c): {
                "n": len(cats[c]["f1"]),
                "f1": float(np.mean(cats[c]["f1"])),
                "bleu1": float(np.mean(cats[c]["bleu"])),
                "judge": float(np.mean(cats[c]["judge"]) * 100) if cats[c]["judge"] else None,
            }
            for c in sorted(cats)
        }
    }
    outpath = args.results.replace(".json", "_standard.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
