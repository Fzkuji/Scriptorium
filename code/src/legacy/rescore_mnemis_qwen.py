"""Re-score Mnemis with qwen3.6-flash judge (same as we used for our results)."""
import json
import re
import sys
import time
from collections import defaultdict
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)

# Use MiniMax since aliyun quota is exhausted - but we need qwen model
# Actually qwen3.6-flash is on aliyun which is exhausted. Let's use proxy with gpt-5.5
# Wait - the point is to use qwen3.6-flash as judge. We can't use aliyun (exhausted).
# But we can check if MiniMax has qwen models.
# Actually, the simplest approach: just run a subset (e.g. 200 questions) to get a meaningful comparison.

# Let's try aliyun first, if it fails we'll note it
ALIYUN_KEY = "sk-sp-D.LIEPX.T6GA.MEYCIQCmomYgcDaqBtAWGXYyua8ve/6RPImY/+0W2BuPqNXzZQIhAJwxunRW89LTfdar0B2A59wjemXSIAHYOPGBgIluBUCh"
ALIYUN_BASE = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
JUDGE_MODEL = "qwen3.6-flash"

client = OpenAI(api_key=ALIYUN_KEY, base_url=ALIYUN_BASE)

CATEGORY_MAP = {"1": "single-hop", "2": "temporal", "3": "multi-hop", "4": "open-domain", "5": "adversarial"}


def llm_judge_score(question, ground_truth, system_answer):
    if not system_answer or not system_answer.strip():
        return 0
    prompt = f"""You are evaluating a memory system's answer. Score 0-100 based on factual accuracy.

Question: {question}
Ground Truth: {ground_truth}
System Answer: {system_answer}

Score 0 if the answer is completely wrong or says "not found" when info exists.
Score 100 if the answer perfectly matches the ground truth.
Score 50+ if the core facts are correct even if wording differs.

Output ONLY a number 0-100."""

    for retry in range(3):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            score = int(re.search(r'\d+', text).group())
            return max(0, min(100, score))
        except Exception as e:
            print(f"  Judge error (retry {retry}): {e}")
            if retry < 2:
                time.sleep(2)
    return 0


def main():
    results_file = "mnemis/results/locomo/results_graphiti_gpt-41-mini-shortco-2025-04-14-Bing_gpt-41-mini-shortco-2025-04-14-Bing_ragtopk10_gtopk20_RAG_GRAPH.json"

    results = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    print(f"Loaded {len(results)} Mnemis predictions")
    print(f"Judge: {JUDGE_MODEL}")

    out_file = "mnemis_rescored_qwen.json"
    scored = []

    try:
        with open(out_file) as f:
            scored = json.load(f)
        print(f"Resuming from {len(scored)} already scored")
    except:
        pass

    start_idx = len(scored)
    cat_scores = defaultdict(list)
    cat_correct = defaultdict(lambda: [0, 0])

    for s in scored:
        cat = s["category_name"]
        cat_scores[cat].append(s["lj_score"])
        cat_correct[cat][1] += 1
        if s["lj_score"] >= 50:
            cat_correct[cat][0] += 1

    for i in range(start_idx, len(results)):
        r = results[i]
        question = r["question"]
        golden = r["golden_answer"]
        answer = r["answer"]
        cat_num = r["category"]
        cat_name = CATEGORY_MAP.get(cat_num, f"cat_{cat_num}")

        lj = llm_judge_score(question, golden, answer)

        entry = {
            "idx": i,
            "question": question,
            "golden_answer": golden,
            "mnemis_answer": answer[:200],
            "lj_score": lj,
            "category": cat_num,
            "category_name": cat_name,
        }
        scored.append(entry)

        cat_scores[cat_name].append(lj)
        cat_correct[cat_name][1] += 1
        if lj >= 50:
            cat_correct[cat_name][0] += 1

        done = i + 1
        if done % 20 == 0 or done == len(results):
            overall_lj = sum(s["lj_score"] for s in scored) / len(scored)
            overall_acc = sum(1 for s in scored if s["lj_score"] >= 50) / len(scored)
            print(f"[{done}/{len(results)}] Overall: LJ={overall_lj:.1f}, Acc={overall_acc:.1%}")

            with open(out_file, "w") as f:
                json.dump(scored, f, indent=2, ensure_ascii=False)

    print("\n=== FINAL RESULTS ===")
    overall_lj = sum(s["lj_score"] for s in scored) / len(scored)
    overall_acc = sum(1 for s in scored if s["lj_score"] >= 50) / len(scored)
    print(f"Overall: LJ={overall_lj:.1f}, Accuracy={overall_acc:.1%} ({len(scored)} questions)")
    for cat in sorted(cat_scores.keys()):
        clj = sum(cat_scores[cat]) / len(cat_scores[cat])
        cacc = cat_correct[cat][0] / cat_correct[cat][1]
        print(f"  {cat}: LJ={clj:.1f}, Acc={cacc:.1%} (n={len(cat_scores[cat])})")

    with open(out_file, "w") as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
