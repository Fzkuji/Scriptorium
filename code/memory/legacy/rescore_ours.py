"""Re-score our own LoCoMo results with gpt-5.5 judge for fair comparison.

Reads our verify_*.json files, extracts responses, re-judges with proxy.
"""
import json
import re
import sys
import time
import glob
from collections import defaultdict
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)

PROXY_BASE = "http://localhost:8199/v1"
JUDGE_MODEL = "gpt-5.5"

client = OpenAI(api_key="dummy", base_url=PROXY_BASE)


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
    verify_dir = "memory_test_v2"
    verify_files = sorted(glob.glob(f"{verify_dir}/verify_sample*.json"))
    print(f"Found {len(verify_files)} verify files")

    out_file = "ours_rescored.json"
    scored = []

    # Resume
    try:
        with open(out_file) as f:
            scored = json.load(f)
        print(f"Resuming from {len(scored)} already scored")
    except:
        pass

    done_keys = {(s["sample"], s["qa_idx"]) for s in scored}

    cat_scores = defaultdict(list)
    cat_correct = defaultdict(lambda: [0, 0])

    for s in scored:
        cat = s["category_name"]
        cat_scores[cat].append(s["lj_score"])
        cat_correct[cat][1] += 1
        if s["lj_score"] >= 50:
            cat_correct[cat][0] += 1

    total_qa = 0
    for vf in verify_files:
        sample_num = int(vf.split("sample")[1].split("_")[0])
        with open(vf) as f:
            data = json.load(f)
        results = data.get("results", [])
        for qi, r in enumerate(results):
            total_qa += 1
            key = (sample_num, qi)
            if key in done_keys:
                continue

            question = r["question"]
            answer = r["answer"]
            response = r["response"]
            cat_name = r.get("category_name", "unknown")

            lj = llm_judge_score(question, answer, response)

            entry = {
                "sample": sample_num,
                "qa_idx": qi,
                "question": question,
                "answer": answer[:200],
                "response": response[:200],
                "original_lj": r.get("judge_score", -1),
                "lj_score": lj,
                "category": r.get("category", 0),
                "category_name": cat_name,
            }
            scored.append(entry)
            done_keys.add(key)

            cat_scores[cat_name].append(lj)
            cat_correct[cat_name][1] += 1
            if lj >= 50:
                cat_correct[cat_name][0] += 1

            done = len(scored)
            if done % 20 == 0:
                overall_lj = sum(s["lj_score"] for s in scored) / len(scored)
                overall_acc = sum(1 for s in scored if s["lj_score"] >= 50) / len(scored)
                print(f"[{done}/{total_qa}] Overall: LJ={overall_lj:.1f}, Acc={overall_acc:.1%}")
                for cat in sorted(cat_scores.keys()):
                    clj = sum(cat_scores[cat]) / len(cat_scores[cat])
                    print(f"  {cat}: LJ={clj:.1f} (n={len(cat_scores[cat])})")

                with open(out_file, "w") as f:
                    json.dump(scored, f, indent=2, ensure_ascii=False)

    # Final save
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
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
