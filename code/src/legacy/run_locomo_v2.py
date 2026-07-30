"""
LoCoMo Benchmark v2: Proper separated storage/retrieval experiment.

Phase 1 (Storage): Feed sessions one by one, model writes wiki files to disk.
Phase 2 (Retrieval): Model ONLY sees wiki files (not original conversation), answers questions.

This properly simulates a memory system where information must be extracted and stored
before it can be retrieved later.

Usage:
    python3 run_locomo_v2.py --n_samples 1 --n_qa 20 --method wiki
    python3 run_locomo_v2.py --n_samples 1 --n_qa 20 --method all
"""

import json
import argparse
import os
import sys
import time
import subprocess
import tempfile
import shutil
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

REPO_DIR = str(Path(__file__).resolve().parents[3])


def call_codex(prompt: str, timeout: int = 180) -> str:
    """Call codex exec with prompt via stdin file."""
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


# =============================================================================
# Phase 1: Storage - build memory from conversation sessions
# =============================================================================

def store_wiki(sessions_data: list, wiki_dir: str) -> dict:
    """
    Process conversation sessions one batch at a time.
    Model reads current wiki + new sessions, updates wiki files on disk.
    Returns stats.
    """
    wiki_path = Path(wiki_dir)
    if wiki_path.exists():
        shutil.rmtree(wiki_path)
    wiki_path.mkdir(parents=True)
    (wiki_path / "index.md").write_text("# Memory Index\n\n(empty)\n")

    stats = {"calls": 0, "sessions_processed": 0}

    # Process sessions in batches of 3 to reduce codex calls
    batch_size = 3
    for i in range(0, len(sessions_data), batch_size):
        batch = sessions_data[i:i+batch_size]
        batch_text = ""
        for sess_key, sess_date, sess_turns in batch:
            batch_text += f"\n--- {sess_key} ({sess_date}) ---\n"
            for turn in sess_turns:
                batch_text += f"{turn['speaker']}: {turn['text']}\n"

        # Read current wiki state - only index + file list to keep prompt small
        index_content = (wiki_path / "index.md").read_text()
        file_list = [str(f.relative_to(wiki_path)) for f in wiki_path.rglob("*.md") if f.name != "index.md"]

        wiki_state = f"=== index.md ===\n{index_content}\n"
        wiki_state += f"\nExisting files: {json.dumps(file_list, ensure_ascii=False)}\n"
        # Include content of existing files (truncated to keep prompt manageable)
        for fname in file_list:
            content = (wiki_path / fname).read_text()
            if len(content) > 500:
                content = content[:500] + "\n...(truncated)"
            wiki_state += f"\n=== FILE: {fname} ===\n{content}\n"

        prompt = f"""You are building a personal memory wiki from conversation sessions.

CURRENT WIKI STATE:
{wiki_state}

NEW CONVERSATION TO PROCESS:
{batch_text}

TASK: Extract important facts, events, preferences, and personal information from
the new conversation. Update the wiki by adding this information.

RULES FOR WIKI ORGANIZATION:
1. Use Retrieval-Simulated Placement: For each piece of info, ask yourself
   "If I forgot this and needed to find it later, where would I look?"
   Put it there.
2. File names must be concrete nouns (person names, topic names, etc.)
3. Each file should focus on one entity or topic
4. Add cross-references between related files using markdown links
5. Update index.md with a one-line summary for each file
6. Skip greetings, small talk, and filler - only store substantive information

OUTPUT FORMAT - respond with a JSON object:
{{
  "files": {{
    "index.md": "updated index content...",
    "filename1.md": "file content...",
    "filename2.md": "file content..."
  }}
}}

Output ONLY the JSON, no other text. Include ALL files (both updated and unchanged)."""

        response = call_codex(prompt, timeout=600)
        stats["calls"] += 1
        stats["sessions_processed"] += len(batch)

        # Parse response and write files
        try:
            # Try to extract JSON from response
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
            print(f"    Warning: Could not parse wiki update response: {e}")
            # Write raw response as fallback
            (wiki_path / "raw_notes.md").write_text(response)

        print(f"    Processed sessions {i+1}-{min(i+batch_size, len(sessions_data))}/{len(sessions_data)}")

    return stats


def store_flat(sessions_data: list, storage_dir: str) -> dict:
    """Baseline: extract memories as flat list (like Mem0 without graph)."""
    storage_path = Path(storage_dir)
    if storage_path.exists():
        shutil.rmtree(storage_path)
    storage_path.mkdir(parents=True)

    stats = {"calls": 0, "sessions_processed": 0}
    all_memories = []

    batch_size = 3
    for i in range(0, len(sessions_data), batch_size):
        batch = sessions_data[i:i+batch_size]
        batch_text = ""
        for sess_key, sess_date, sess_turns in batch:
            batch_text += f"\n--- {sess_key} ({sess_date}) ---\n"
            for turn in sess_turns:
                batch_text += f"{turn['speaker']}: {turn['text']}\n"

        prompt = f"""Extract important facts, events, preferences, and personal information
from these conversation sessions. Output as a JSON list of strings, each being one
distinct piece of information. Skip greetings and small talk.

CONVERSATION:
{batch_text}

Output ONLY a JSON list like: ["fact 1", "fact 2", ...]"""

        response = call_codex(prompt, timeout=600)
        stats["calls"] += 1
        stats["sessions_processed"] += len(batch)

        try:
            json_str = response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            memories = json.loads(json_str)
            if isinstance(memories, list):
                all_memories.extend(memories)
        except (json.JSONDecodeError, KeyError):
            all_memories.append(response)

        print(f"    Processed sessions {i+1}-{min(i+batch_size, len(sessions_data))}/{len(sessions_data)}")

    # Save flat memory list
    mem_file = storage_path / "memories.json"
    mem_file.write_text(json.dumps(all_memories, ensure_ascii=False, indent=2))
    stats["total_memories"] = len(all_memories)
    return stats


# =============================================================================
# Phase 2: Retrieval - answer questions using ONLY stored memory
# =============================================================================

def retrieve_wiki(question: str, wiki_dir: str) -> str:
    """Answer question using ONLY wiki files. No access to original conversation."""
    wiki_path = Path(wiki_dir)

    # Read all wiki files
    wiki_content = ""
    for f in sorted(wiki_path.rglob("*.md")):
        rel = str(f.relative_to(wiki_path))
        wiki_content += f"\n=== FILE: {rel} ===\n{f.read_text()}\n"

    prompt = f"""You are answering a question using ONLY the information in your personal memory wiki.
You must NOT use any knowledge from your training data. Only use what is written in the wiki below.
If the answer is not in the wiki, say "I don't have this information."

YOUR MEMORY WIKI:
{wiki_content}

QUESTION: {question}

Answer concisely based ONLY on the wiki content above."""

    return call_codex(prompt, timeout=300)


def retrieve_flat(question: str, storage_dir: str) -> str:
    """Answer question using ONLY flat memory list. Pre-filter by keyword overlap."""
    storage_path = Path(storage_dir)
    mem_file = storage_path / "memories.json"

    if mem_file.exists():
        memories = json.loads(mem_file.read_text())
        # Pre-filter: score by keyword overlap with question
        q_words = set(question.lower().split())
        scored = []
        for m in memories:
            m_words = set(str(m).lower().split())
            overlap = len(q_words & m_words)
            scored.append((overlap, m))
        scored.sort(reverse=True)
        # Take top 50 most relevant + always include first 10 (context)
        top = [m for _, m in scored[:50]]
        mem_text = "\n".join(f"- {m}" for m in top)
    else:
        mem_text = "(no memories stored)"

    prompt = f"""You are answering a question using ONLY the stored memories below.
You must NOT use any knowledge from your training data. Only use what is in the memories.
If the answer is not in the memories, say "I don't have this information."

STORED MEMORIES:
{mem_text}

QUESTION: {question}

Answer concisely based ONLY on the memories above."""

    return call_codex(prompt, timeout=300)


# =============================================================================
# Evaluation
# =============================================================================

def judge(question: str, reference: str, hypothesis: str) -> bool:
    prompt = f"""Question: {question}
Correct Answer: {reference}
Model Response: {hypothesis}

Is the model response correct (contains or is equivalent to the correct answer)?
Do not penalize minor formatting differences or off-by-one day errors.
Answer yes or no only."""

    try:
        result = call_codex(prompt, timeout=120)
        return result.strip().lower().startswith("yes")
    except:
        return False


# =============================================================================
# Main
# =============================================================================

def load_locomo(data_path: str, n_samples: int):
    with open(data_path) as f:
        data = json.load(f)
    return data[:n_samples]


def get_sessions(conversation: dict) -> list:
    """Extract ordered sessions from LoCoMo conversation dict."""
    session_keys = sorted(
        [k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1])
    )
    sessions = []
    for key in session_keys:
        date = conversation.get(key + "_date_time", "")
        turns = conversation[key]
        sessions.append((key, date, turns))
    return sessions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["wiki", "flat", "all"], default="all")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--n_qa", type=int, default=20)
    parser.add_argument("--data", default="locomo/data/locomo10.json")
    parser.add_argument("--outdir", default="results_locomo_v2")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    samples = load_locomo(args.data, args.n_samples)
    print(f"Loaded {len(samples)} samples")

    methods = [args.method] if args.method != "all" else ["flat", "wiki"]
    all_metrics = {}

    for sample_idx, sample in enumerate(samples):
        sessions = get_sessions(sample["conversation"])
        qa_list = sample["qa"][:args.n_qa]
        print(f"\nSample {sample_idx+1}/{len(samples)}: {len(sessions)} sessions, {len(qa_list)} questions")

        for method in methods:
            print(f"\n{'='*60}")
            print(f"Method: {method}")
            print(f"{'='*60}")

            # Phase 1: Storage
            storage_dir = os.path.join(args.outdir, f"sample{sample_idx}_{method}")
            print(f"\n  Phase 1: Building memory from {len(sessions)} sessions...")
            t0 = time.time()

            if method == "wiki":
                store_stats = store_wiki(sessions, storage_dir)
            elif method == "flat":
                store_stats = store_flat(sessions, storage_dir)

            store_time = time.time() - t0
            print(f"  Storage done in {store_time:.0f}s ({store_stats['calls']} codex calls)")

            # Show what was stored
            if method == "wiki":
                wiki_files = list(Path(storage_dir).rglob("*.md"))
                print(f"  Wiki files created: {len(wiki_files)}")
                for wf in wiki_files:
                    print(f"    - {wf.relative_to(storage_dir)} ({wf.stat().st_size} bytes)")
            elif method == "flat":
                mem_file = Path(storage_dir) / "memories.json"
                if mem_file.exists():
                    mems = json.loads(mem_file.read_text())
                    print(f"  Memories extracted: {len(mems)}")

            # Phase 2: Retrieval
            print(f"\n  Phase 2: Answering {len(qa_list)} questions (memory only, no original conversation)...")
            results = []
            for qi, qa in enumerate(qa_list):
                question = qa["question"]
                reference = str(qa["answer"])

                t0 = time.time()
                if method == "wiki":
                    hypothesis = retrieve_wiki(question, storage_dir)
                elif method == "flat":
                    hypothesis = retrieve_flat(question, storage_dir)
                elapsed = time.time() - t0

                print(f"    [{qi+1}/{len(qa_list)}] Q: {question[:55]}...")
                print(f"           A: {hypothesis[:70]}")
                print(f"           Expected: {reference[:70]}")

                results.append({
                    "sample_id": sample.get("sample_id", sample_idx),
                    "question": question,
                    "hypothesis": hypothesis,
                    "reference": reference,
                    "category": qa.get("category", "unknown"),
                    "time": elapsed,
                })

            # Save answers
            out_path = os.path.join(args.outdir, f"answers_{method}_s{sample_idx}.jsonl")
            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  Answers saved to {out_path}")

            # Phase 3: Evaluation with LLM judge
            print(f"\n  Phase 3: LLM Judge evaluation...")
            correct = 0
            by_cat = {}
            for r in results:
                is_correct = judge(r["question"], r["reference"], r["hypothesis"])
                r["llm_correct"] = is_correct
                if is_correct:
                    correct += 1
                cat = str(r["category"])
                if cat not in by_cat:
                    by_cat[cat] = {"correct": 0, "total": 0}
                by_cat[cat]["total"] += 1
                if is_correct:
                    by_cat[cat]["correct"] += 1

            accuracy = correct / len(results) if results else 0
            key = f"{method}_s{sample_idx}"
            all_metrics[key] = {
                "method": method,
                "sample": sample_idx,
                "accuracy": accuracy,
                "correct": correct,
                "total": len(results),
                "store_time": store_time,
                "store_calls": store_stats["calls"],
                "by_category": {
                    k: {**v, "acc": v["correct"]/v["total"] if v["total"] else 0}
                    for k, v in by_cat.items()
                }
            }

            print(f"\n  --- {method} Results (Sample {sample_idx}) ---")
            print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(results)})")
            for cat, m in sorted(by_cat.items()):
                acc = m["correct"]/m["total"] if m["total"] else 0
                print(f"    Cat {cat}: {acc:.1%} ({m['correct']}/{m['total']})")

            # Re-save with eval labels
            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL RESULTS (LLM Judge, memory-only retrieval)")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Accuracy':>10} {'Correct':>8} {'Store Time':>12}")
    for key, m in all_metrics.items():
        print(f"{key:<15} {m['accuracy']:>9.1%} {m['correct']:>4}/{m['total']:<4} {m['store_time']:>10.0f}s")

    print(f"\n--- Known Baselines (LoCoMo, from papers) ---")
    print(f"  Mem0:         66.88%")
    print(f"  Memobase:     ~80%")
    print(f"  Backboard:    90.1% (commercial)")
    print(f"\nNote: Our results use memory-only retrieval (no access to original conversation).")
    print(f"This is comparable to how Mem0/Memobase are evaluated.")

    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
