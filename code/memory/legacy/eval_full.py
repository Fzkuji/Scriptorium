"""Full evaluation: run QA on existing wikis, with per-category breakdown and token tracking."""
import json, sys, os, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
import memory_builder
from memory_builder import verify_storage

sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="locomo/data/locomo10.json")
    parser.add_argument("--outdir", default="memory_test_v2")
    parser.add_argument("--model", default="qwen3.6-flash")
    parser.add_argument("--samples", default="0-9", help="Sample range, e.g. '0-9' or '3'")
    parser.add_argument("--max-qa", type=int, default=20, help="Max QA per sample (0=all)")
    parser.add_argument("--wiki-model", default=None, help="Model used to build wiki (if different from --model)")
    args = parser.parse_args()

    memory_builder.ALIYUN_MODEL = args.model

    with open(args.data) as f:
        data = json.load(f)

    if "," in args.samples:
        sample_ids = [int(x) for x in args.samples.split(",")]
    elif "-" in args.samples:
        start, end = args.samples.split("-")
        sample_ids = list(range(int(start), int(end) + 1))
    else:
        sample_ids = [int(args.samples)]

    all_results = []
    overall_cats = {}

    for sid in sample_ids:
        wiki_model = args.wiki_model or args.model
        memory_dir = os.path.join(args.outdir, f"sample{sid}_{wiki_model}")
        if not os.path.exists(memory_dir):
            print(f"\nSample {sid}: wiki not built yet, skipping")
            continue

        # Reset counters for each sample
        memory_builder.TOTAL_CALLS = 0
        memory_builder.TOTAL_TOKENS = 0
        memory_builder.CALL_LOG = []

        qa_list = data[sid].get("qa", [])
        n_qa = len(qa_list) if args.max_qa == 0 else min(args.max_qa, len(qa_list))

        print(f"\n{'='*60}")
        print(f"SAMPLE {sid} — {n_qa} QA")
        print(f"{'='*60}")

        t0 = time.time()
        correct, tested = verify_storage(memory_dir, qa_list, n_questions=n_qa)
        elapsed = time.time() - t0

        print(f"\n  Sample {sid}: {correct}/{tested} = {correct/tested:.1%} ({elapsed:.0f}s)")
        print(f"  Token usage: {memory_builder.TOTAL_CALLS} calls, {memory_builder.TOTAL_TOKENS} tokens")

        # Load the saved verify json for per-category stats
        verify_path = os.path.join(os.path.dirname(memory_dir), f"verify_{os.path.basename(memory_dir)}.json")
        # Read saved verify json for per-category and LJ stats
        sample_lj = 0.0
        if os.path.exists(verify_path):
            with open(verify_path) as f:
                vdata = json.load(f)
            sample_lj = vdata.get("avg_lj_score", 0.0)
            if "per_category" in vdata:
                for cn, s in vdata["per_category"].items():
                    if cn not in overall_cats:
                        overall_cats[cn] = {"found": 0, "total": 0, "lj_sum": 0.0}
                    overall_cats[cn]["found"] += s["found"]
                    overall_cats[cn]["total"] += s["total"]
                    overall_cats[cn]["lj_sum"] += s.get("avg_lj_score", 0) * s["total"]

        all_results.append({
            "sample": sid,
            "correct": correct,
            "tested": tested,
            "accuracy": correct / tested if tested else 0,
            "avg_lj_score": sample_lj,
            "time": elapsed,
            "llm_calls": memory_builder.TOTAL_CALLS,
            "total_tokens": memory_builder.TOTAL_TOKENS,
            "call_log": memory_builder.CALL_LOG,
        })

    # Overall summary
    tf = sum(r["correct"] for r in all_results)
    tt = sum(r["tested"] for r in all_results)
    total_calls = sum(r["llm_calls"] for r in all_results)
    total_tokens = sum(r["total_tokens"] for r in all_results)

    overall_lj = sum(r["avg_lj_score"] * r["tested"] for r in all_results) / tt if tt else 0

    print(f"\n{'='*60}")
    print(f"OVERALL: {tf}/{tt} = {tf/tt:.1%} (LJ={overall_lj:.1f})")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  Sample {r['sample']}: {r['accuracy']:.1%} ({r['correct']}/{r['tested']}) LJ={r['avg_lj_score']:.1f} — {r['llm_calls']} calls, {r['total_tokens']} tokens")

    if overall_cats:
        print(f"\nPer-category:")
        for cn in ["single-hop", "temporal", "multi-hop", "open-domain", "adversarial"]:
            if cn in overall_cats:
                s = overall_cats[cn]
                cat_lj = s["lj_sum"] / s["total"] if s["total"] else 0
                print(f"  {cn}: {s['found']}/{s['total']} = {s['found']/s['total']:.1%} (LJ={cat_lj:.1f})")

    print(f"\nTotal: {total_calls} calls, {total_tokens} tokens")

    # Save (without call_log in summary, save separately)
    summary_results = [{k: v for k, v in r.items() if k != "call_log"} for r in all_results]
    out_path = os.path.join(args.outdir, "eval_full_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": summary_results, "per_category": overall_cats,
                   "overall": tf/tt if tt else 0,
                   "total_calls": total_calls, "total_tokens": total_tokens},
                  f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")

    # Save per-sample call logs
    for r in all_results:
        log_path = os.path.join(args.outdir, f"eval_call_log_sample{r['sample']}.json")
        with open(log_path, "w") as f:
            json.dump(r.get("call_log", []), f, indent=2)


if __name__ == "__main__":
    main()
