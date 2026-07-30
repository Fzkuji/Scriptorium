"""
Evaluate our LoCoMo results using A-Mem's metrics (F1, ROUGE, BERTScore, etc.)
so results are directly comparable to A-Mem's paper numbers.

Usage:
    python3 eval_f1.py results_locomo_v2/answers_wiki_s0.jsonl
    python3 eval_f1.py results_locomo_v2/answers_flat_s0.jsonl
    python3 eval_f1.py --all
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "A-mem"))
from utils import calculate_metrics, aggregate_metrics

# A-Mem paper results (GPT-4o-mini, from their paper Table)
AMEM_RESULTS = {
    "A-Mem": {"1_single_hop": 44.65, "2_temporal": 45.85, "3_open_domain": None, "4_multi_hop": 27.02, "5_adversarial": 50.03},
    "LoCoMo": {"1_single_hop": 40.36, "2_temporal": 18.41, "3_open_domain": None, "4_multi_hop": 25.02, "5_adversarial": 69.23},
    "MemGPT": {"1_single_hop": 41.04, "2_temporal": 25.52, "3_open_domain": None, "4_multi_hop": 26.65, "5_adversarial": 43.29},
}

# LoCoMo category mapping
CATEGORY_NAMES = {1: "single_hop", 2: "temporal", 3: "open_domain", 4: "multi_hop", 5: "adversarial"}


def evaluate_file(filepath: str):
    results = []
    with open(filepath) as f:
        for line in f:
            results.append(json.loads(line))

    method = Path(filepath).stem
    print(f"\n{'='*60}")
    print(f"Evaluating: {method} ({len(results)} questions)")
    print(f"{'='*60}")

    all_metrics = []
    all_categories = []
    by_cat = defaultdict(list)

    for r in results:
        pred = str(r.get("hypothesis", ""))
        ref = str(r.get("reference", ""))
        cat = int(r.get("category", 0))
        metrics = calculate_metrics(pred, ref)
        all_metrics.append(metrics)
        all_categories.append(cat)
        by_cat[cat].append(metrics)

    # Overall
    overall_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics) * 100
    overall_bert = sum(m["bert_f1"] for m in all_metrics) / len(all_metrics) * 100
    overall_rouge = sum(m["rougeL_f"] for m in all_metrics) / len(all_metrics) * 100

    print(f"\nOverall ({len(all_metrics)} questions):")
    print(f"  F1:        {overall_f1:.1f}%")
    print(f"  ROUGE-L:   {overall_rouge:.1f}%")
    print(f"  BERTScore: {overall_bert:.1f}%")

    # Per category
    print(f"\nPer category (F1):")
    cat_f1 = {}
    for cat in sorted(by_cat.keys()):
        cat_name = CATEGORY_NAMES.get(cat, f"cat{cat}")
        f1 = sum(m["f1"] for m in by_cat[cat]) / len(by_cat[cat]) * 100
        cat_f1[cat] = f1
        print(f"  Cat {cat} ({cat_name}): {f1:.1f}% ({len(by_cat[cat])} questions)")

    # Compare with A-Mem paper
    print(f"\n--- Comparison with A-Mem paper (F1%) ---")
    print(f"{'Category':<20} {'Ours':>8} {'A-Mem':>8} {'MemGPT':>8} {'LoCoMo':>8}")
    for cat in sorted(by_cat.keys()):
        cat_name = CATEGORY_NAMES.get(cat, f"cat{cat}")
        ours = cat_f1.get(cat, 0)
        amem = AMEM_RESULTS["A-Mem"].get(f"{cat}_{cat_name}", None)
        memgpt = AMEM_RESULTS["MemGPT"].get(f"{cat}_{cat_name}", None)
        locomo = AMEM_RESULTS["LoCoMo"].get(f"{cat}_{cat_name}", None)
        print(f"  {cat_name:<18} {ours:>7.1f} {amem if amem else 'N/A':>8} {memgpt if memgpt else 'N/A':>8} {locomo if locomo else 'N/A':>8}")

    return {
        "method": method,
        "overall_f1": overall_f1,
        "overall_rouge": overall_rouge,
        "overall_bert": overall_bert,
        "per_category_f1": cat_f1,
        "n_questions": len(all_metrics),
    }


def main():
    if "--all" in sys.argv:
        files = sorted(Path("results_locomo_v2").glob("answers_*.jsonl"))
    else:
        files = [Path(a) for a in sys.argv[1:] if a.endswith(".jsonl")]

    if not files:
        print("Usage: python3 eval_f1.py --all")
        return

    all_results = {}
    for f in files:
        r = evaluate_file(str(f))
        all_results[r["method"]] = r

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Method':<25} {'F1':>8} {'ROUGE-L':>8} {'BERT':>8}")
        for name, r in all_results.items():
            print(f"  {name:<23} {r['overall_f1']:>7.1f}% {r['overall_rouge']:>7.1f}% {r['overall_bert']:>7.1f}%")
        print(f"\n  A-Mem (paper, GPT-4o-mini):  ~42% F1 (avg across categories)")
        print(f"  MemGPT (paper):              ~34% F1")


if __name__ == "__main__":
    main()
