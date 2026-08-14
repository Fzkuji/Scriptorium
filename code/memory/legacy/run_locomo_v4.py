"""
LoCoMo Benchmark v4: MiniMax API (fast) + optimized prompts + Mem0 judge.

Key changes from v3:
- Uses MiniMax API instead of codex exec (~10x faster)
- Same optimized storage/retrieval prompts and Mem0 judge

Usage:
    python3 run_locomo_v4.py --n_samples 2 --n_qa 50
    python3 run_locomo_v4.py --n_samples 10  # full LoCoMo
"""

import json
import argparse
import os
import sys
import time
import re
import shutil
from pathlib import Path
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "memory-benchmarks" / "benchmarks"))

from locomo.prompts import get_judge_prompt, JUDGE_SYSTEM_PROMPT

MINIMAX_KEY = os.environ["MINIMAX_API_KEY"]
MINIMAX_MODEL = "MiniMax-M2.5-highspeed"

llm_client = OpenAI(api_key=MINIMAX_KEY, base_url="https://api.minimaxi.com/v1")


def call_llm(prompt: str, max_tokens: int = 4000) -> str:
    """Call MiniMax API — fast and cheap."""
    try:
        r = llm_client.chat.completions.create(
            model=MINIMAX_MODEL,
            messages=[
                {"role": "system", "content": "Answer directly and concisely. Follow instructions precisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        content = r.choices[0].message.content.strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return content
    except Exception as e:
        print(f"    LLM error: {e}")
        return ""


# =============================================================================
# Storage: Build wiki from conversations
# =============================================================================

STORAGE_PROMPT = """You are building a personal memory wiki from conversation sessions.
Your job is to extract and preserve ALL factual information — nothing should be lost.

CURRENT WIKI INDEX:
{index_content}

EXISTING FILES: {file_list}

NEW CONVERSATION TO PROCESS:
{batch_text}

## EXTRACTION RULES — preserve ALL of these:
- **Dates and times**: exact dates, relative time references ("last year", "next month"), durations
- **People**: names, relationships, roles, who said what
- **Personal facts**: relationship status, identity, age, where they live/moved from, birthdays
- **Events**: what happened, when, where, with whom
- **Preferences**: likes, dislikes, hobbies, habits, routines
- **Plans**: future intentions, goals, upcoming events
- **Emotions and opinions**: how people felt about events
- **Causal links**: why something happened, what led to what

## WIKI ORGANIZATION:
- For each fact, ask: "If I forgot this and needed to find it, where would I look?"
- File names = concrete nouns (person names, activity names)
- Cross-reference related files with markdown links [text](file.md)
- One file per entity/topic

## OUTPUT FORMAT:
Respond with a JSON object containing ONLY the files that need to be created or updated.
Do NOT include unchanged files.

{{
  "files": {{
    "index.md": "full updated index...",
    "new_or_updated_file.md": "full file content..."
  }}
}}

Output ONLY valid JSON."""


def store_wiki(sessions_data: list, wiki_dir: str) -> dict:
    wiki_path = Path(wiki_dir)
    if wiki_path.exists():
        shutil.rmtree(wiki_path)
    wiki_path.mkdir(parents=True)
    (wiki_path / "index.md").write_text("# Memory Index\n\n(empty)\n")

    stats = {"calls": 0, "sessions_processed": 0}

    # Smaller batches: 2 sessions at a time for better extraction
    batch_size = 2
    for i in range(0, len(sessions_data), batch_size):
        batch = sessions_data[i:i+batch_size]
        batch_text = ""
        for sess_key, sess_date, sess_turns in batch:
            batch_text += f"\n--- {sess_key} ({sess_date}) ---\n"
            for turn in sess_turns:
                batch_text += f"{turn['speaker']}: {turn['text']}\n"

        index_content = (wiki_path / "index.md").read_text()
        file_list = [str(f.relative_to(wiki_path)) for f in wiki_path.rglob("*.md") if f.name != "index.md"]

        prompt = STORAGE_PROMPT.format(
            index_content=index_content,
            file_list=json.dumps(file_list, ensure_ascii=False),
            batch_text=batch_text
        )

        response = call_llm(prompt)
        stats["calls"] += 1
        stats["sessions_processed"] += len(batch)

        try:
            json_str = response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            data = json.loads(json_str)
            files = data.get("files", data)
            for fname, content in files.items():
                fpath = wiki_path / fname
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    Warning: parse error: {e}")
            (wiki_path / "raw_notes.md").write_text(
                (wiki_path / "raw_notes.md").read_text() + "\n" + response
                if (wiki_path / "raw_notes.md").exists() else response
            )

        print(f"    Processed sessions {i+1}-{min(i+batch_size, len(sessions_data))}/{len(sessions_data)}")

    return stats


# =============================================================================
# Retrieval: Answer using wiki only (Mem0-inspired 7-step reasoning)
# =============================================================================

RETRIEVAL_PROMPT = """You are answering a question using ONLY the personal memory wiki below.
You must NOT use knowledge from your training data — only what is in the wiki.

## YOUR MEMORY WIKI:
{wiki_content}

## REASONING STEPS (follow in order):

1. **SCAN ALL FILES**: Read every file. Note ALL facts relevant to the question.
   Do NOT stop after finding the first relevant fact — details are scattered.

2. **ENTITY VERIFICATION**: Confirm each fact is about the correct person.
   If the question asks about Person A, do not confuse with Person B's facts.

3. **COMBINE AND CROSS-REFERENCE**: Connect facts from different files.
   Follow markdown links to find related information. For listing questions,
   collect ALL items across ALL files.

4. **TEMPORAL GROUNDING**: Use dates in the wiki to answer time-related questions.
   Calculate durations from explicit dates. "Last year" or "a few years ago"
   should be interpreted relative to the conversation dates in the wiki.

5. **CAUSAL REASONING**: For "would X do Y" or "what if" questions, follow
   the direct causal chain in the wiki. If the wiki shows X does Y BECAUSE of Z,
   then without Z, answer "likely no."

6. **SELECT BEST ANSWER**: Choose the most specific answer available.
   A proper name, date, or number beats a generic description.

7. **COMMIT**: Give a direct, specific answer. NEVER say "not mentioned",
   "not specified", "I don't have this information", or "the wiki doesn't say."
   If ANY file contains relevant information, give the best answer from
   available evidence. No hedging, no caveats.

## Question: {question}

Work through steps 1-7, then give your final answer after "ANSWER:"."""


def retrieve_wiki(question: str, wiki_dir: str) -> str:
    wiki_path = Path(wiki_dir)
    wiki_content = ""
    for f in sorted(wiki_path.rglob("*.md")):
        rel = str(f.relative_to(wiki_path))
        wiki_content += f"\n=== FILE: {rel} ===\n{f.read_text()}\n"

    prompt = RETRIEVAL_PROMPT.format(wiki_content=wiki_content, question=question)
    response = call_llm(prompt)

    if "ANSWER:" in response:
        return response.split("ANSWER:")[-1].strip()
    return response


# =============================================================================
# Evaluation: Mem0's official LOCOMO judge
# =============================================================================

def judge_mem0(question: str, answer: str, response: str, category: int) -> bool:
    """Use Mem0's official LOCOMO judge prompt."""
    judge_prompt = get_judge_prompt(
        category=category,
        question=question,
        answer=answer,
        response=response,
    )
    full_prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{judge_prompt}"

    result = call_llm(full_prompt, max_tokens=500)

    # Parse JSON response for label
    try:
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0]
        elif "```" in result:
            result = result.split("```")[1].split("```")[0]
        data = json.loads(result)
        return data.get("label", "").upper() == "CORRECT"
    except (json.JSONDecodeError, KeyError):
        return "correct" in result.lower() and "wrong" not in result.lower()


# =============================================================================
# Main
# =============================================================================

def load_locomo(data_path: str, n_samples: int):
    with open(data_path) as f:
        return json.load(f)[:n_samples]


def get_sessions(conversation: dict) -> list:
    session_keys = sorted(
        [k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1])
    )
    return [(key, conversation.get(key + "_date_time", ""), conversation[key]) for key in session_keys]


# LoCoMo category names
CAT_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--n_qa", type=int, default=20)
    parser.add_argument("--data", default="locomo/data/locomo10.json")
    parser.add_argument("--outdir", default="results_locomo_v4")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    samples = load_locomo(args.data, args.n_samples)
    print(f"Loaded {len(samples)} samples")

    all_results = []
    all_correct = 0
    all_total = 0
    cat_stats = {}

    for si, sample in enumerate(samples):
        sessions = get_sessions(sample["conversation"])
        qa_list = sample["qa"][:args.n_qa]
        print(f"\nSample {si}: {len(sessions)} sessions, {len(qa_list)} QA pairs")

        # Phase 1: Storage
        wiki_dir = os.path.join(args.outdir, f"wiki_s{si}")
        print(f"\n  Phase 1: Building wiki...")
        t0 = time.time()
        store_stats = store_wiki(sessions, wiki_dir)
        store_time = time.time() - t0
        wiki_files = list(Path(wiki_dir).rglob("*.md"))
        print(f"  Done in {store_time:.0f}s, {len(wiki_files)} files, {store_stats['calls']} calls")

        # Phase 2: Retrieval
        print(f"\n  Phase 2: Answering {len(qa_list)} questions (wiki only)...")
        results = []
        for qi, qa in enumerate(qa_list):
            question = qa["question"]
            reference = str(qa["answer"])
            category = qa.get("category", 0)

            t0 = time.time()
            hypothesis = retrieve_wiki(question, wiki_dir)
            elapsed = time.time() - t0

            print(f"    [{qi+1}/{len(qa_list)}] Q: {question[:55]}...")
            print(f"           A: {hypothesis[:70]}")
            print(f"           E: {reference[:70]}")

            results.append({
                "sample_id": si,
                "question": question,
                "hypothesis": hypothesis,
                "reference": reference,
                "category": category,
                "time": elapsed,
            })

        # Save answers
        ans_path = os.path.join(args.outdir, f"answers_s{si}.jsonl")
        with open(ans_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Phase 3: Evaluation with Mem0 judge
        print(f"\n  Phase 3: Mem0 judge evaluation...")
        correct = 0
        for r in results:
            is_correct = judge_mem0(r["question"], r["reference"], r["hypothesis"], r["category"])
            r["correct"] = is_correct
            if is_correct:
                correct += 1
            cat = r["category"]
            cat_name = CAT_NAMES.get(cat, f"cat{cat}")
            if cat_name not in cat_stats:
                cat_stats[cat_name] = {"correct": 0, "total": 0}
            cat_stats[cat_name]["total"] += 1
            if is_correct:
                cat_stats[cat_name]["correct"] += 1

        all_correct += correct
        all_total += len(results)
        all_results.extend(results)

        accuracy = correct / len(results) if results else 0
        print(f"\n  Sample {si} accuracy: {accuracy:.1%} ({correct}/{len(results)})")

        # Re-save with eval
        with open(ans_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final report
    overall_acc = all_correct / all_total if all_total else 0
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (Mem0 Judge, wiki-only retrieval)")
    print(f"{'='*60}")
    print(f"Overall: {overall_acc:.1%} ({all_correct}/{all_total})")
    for cat_name in ["single-hop", "temporal", "multi-hop", "open-domain"]:
        if cat_name in cat_stats:
            s = cat_stats[cat_name]
            acc = s["correct"] / s["total"] if s["total"] else 0
            print(f"  {cat_name:<12}: {acc:.1%} ({s['correct']}/{s['total']})")

    print(f"\n--- Comparison (Mem0 Judge, LoCoMo) ---")
    print(f"  Ours (wiki):    {overall_acc:.1%}")
    print(f"  Mem0 v3 top-50: 82.7%")
    print(f"  Mem0 v3 top-200: 92.5%")

    summary = {
        "overall_accuracy": overall_acc,
        "correct": all_correct,
        "total": all_total,
        "per_category": {k: {**v, "accuracy": v["correct"]/v["total"] if v["total"] else 0} for k, v in cat_stats.items()},
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.outdir}/summary.json")


if __name__ == "__main__":
    main()
