"""Re-score Mnemis LongMemEval-S predictions with gpt-5.5 judge."""
import json
import re
import sys
import time
from collections import defaultdict
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)

client = OpenAI(api_key="dummy", base_url="http://localhost:8199/v1")
JUDGE_MODEL = "gpt-5.5"


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
    results_file = "mnemis/results/lme-s/results_graphiti_gpt-41-mini-shortco-2025-04-14-Bing_gpt-41-mini-shortco-2025-04-14-Bing_ragtopk10_gtopk20_RAG_GRAPH.json"

    results = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    print(f"Loaded {len(results)} Mnemis LME predictions")

    out_file = "mnemis_lme_rescored.json"
    scored = []
    try:
        with open(out_file) as f:
            scored = json.load(f)
        print(f"Resuming from {len(scored)} already scored")
    except:
        pass

    start_idx = len(scored)
    type_scores = defaultdict(list)

    for s in scored:
        type_scores[s["question_type"]].append(s["lj_score"])

    for i in range(start_idx, len(results)):
        r = results[i]
        question = r["question"]
        golden = r["gold_answer"]
        answer = r["hypothesis"]
        q_type = r.get("question_type", "unknown")

        lj = llm_judge_score(question, golden, answer)

        entry = {
            "idx": i,
            "question": question,
            "gold_answer": golden,
            "mnemis_answer": answer[:200],
            "original_grade": r.get("grade", ""),
            "lj_score": lj,
            "question_type": q_type,
        }
        scored.append(entry)
        type_scores[q_type].append(lj)

        done = i + 1
        if done % 20 == 0 or done == len(results):
            overall_lj = sum(s["lj_score"] for s in scored) / len(scored)
            overall_acc = sum(1 for s in scored if s["lj_score"] >= 50) / len(scored)
            print(f"[{done}/{len(results)}] Overall: LJ={overall_lj:.1f}, Acc={overall_acc:.1%}")
            with open(out_file, "w") as f:
                json.dump(scored, f, indent=2, ensure_ascii=False)

    print("\n=== FINAL ===")
    overall_lj = sum(s["lj_score"] for s in scored) / len(scored)
    print(f"Overall: LJ={overall_lj:.1f} ({len(scored)} questions)")
    for t in sorted(type_scores):
        tlj = sum(type_scores[t]) / len(type_scores[t])
        print(f"  {t}: LJ={tlj:.1f} (n={len(type_scores[t])})")

    with open(out_file, "w") as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
