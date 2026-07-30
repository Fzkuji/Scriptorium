"""Evaluate specific QA categories on existing wikis."""
import json, sys, os, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
import memory_builder
from memory_builder import verify_storage

sys.stdout.reconfigure(line_buffering=True)

CATEGORY_NAMES = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain", 5: "adversarial"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="locomo/data/locomo10.json")
    parser.add_argument("--outdir", default="memory_test_v2")
    parser.add_argument("--model", default="qwen3.6-flash")
    parser.add_argument("--samples", default="0-9")
    parser.add_argument("--category", type=int, required=True, help="Category to evaluate (1-5)")
    parser.add_argument("--max-qa", type=int, default=0, help="Max QA per sample (0=all)")
    args = parser.parse_args()

    memory_builder.ALIYUN_MODEL = args.model
    cat_name = CATEGORY_NAMES.get(args.category, f"cat{args.category}")

    with open(args.data) as f:
        data = json.load(f)

    if "-" in args.samples:
        start, end = args.samples.split("-")
        sample_ids = list(range(int(start), int(end) + 1))
    else:
        sample_ids = [int(args.samples)]

    all_correct = 0
    all_tested = 0
    all_lj_scores = []

    for sid in sample_ids:
        memory_dir = os.path.join(args.outdir, f"sample{sid}_{args.model}")
        if not os.path.exists(memory_dir):
            print(f"Sample {sid}: wiki not built, skipping")
            continue

        # Reset counters
        memory_builder.TOTAL_CALLS = 0
        memory_builder.TOTAL_TOKENS = 0
        memory_builder.CALL_LOG = []

        # Filter QA by category
        qa_list = [q for q in data[sid].get("qa", []) if q.get("category") == args.category]
        if not qa_list:
            print(f"Sample {sid}: no {cat_name} QA, skipping")
            continue

        n_qa = len(qa_list) if args.max_qa == 0 else min(args.max_qa, len(qa_list))
        qa_list = qa_list[:n_qa]

        print(f"\n{'='*60}")
        print(f"SAMPLE {sid} — {n_qa} {cat_name} QA")
        print(f"{'='*60}")

        t0 = time.time()
        correct, tested = verify_storage(memory_dir, qa_list, n_questions=n_qa)
        elapsed = time.time() - t0

        all_correct += correct
        all_tested += tested

        # Read LJ score from saved verify json
        verify_path = os.path.join(os.path.dirname(memory_dir), f"verify_{os.path.basename(memory_dir)}.json")
        sample_lj = 0.0
        if os.path.exists(verify_path):
            with open(verify_path) as f:
                vdata = json.load(f)
            sample_lj = vdata.get("avg_lj_score", 0.0)
            all_lj_scores.extend(r.get("judge_score", 0) for r in vdata.get("results", []))

        print(f"\n  Sample {sid}: {correct}/{tested} = {correct/tested:.1%} LJ={sample_lj:.1f} ({elapsed:.0f}s, {memory_builder.TOTAL_TOKENS} tokens)")

    avg_lj = sum(all_lj_scores) / len(all_lj_scores) if all_lj_scores else 0
    print(f"\n{'='*60}")
    print(f"OVERALL {cat_name}: {all_correct}/{all_tested} = {all_correct/all_tested:.1%} (LJ={avg_lj:.1f})")
    print(f"{'='*60}")

    # Save
    out_path = os.path.join(args.outdir, f"eval_{cat_name}_results.json")
    with open(out_path, "w") as f:
        json.dump({"category": args.category, "category_name": cat_name,
                   "correct": all_correct, "tested": all_tested,
                   "accuracy": all_correct/all_tested if all_tested else 0,
                   "avg_lj_score": avg_lj},
                  f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
