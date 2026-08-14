"""
Run Mem0 baseline on LoCoMo dataset.

Usage:
    python3 run_mem0_locomo.py --sample 0 --max-sessions 3
"""
import json
import argparse
import os
import sys
import time
import re
from mem0 import Memory

sys.stdout.reconfigure(line_buffering=True)

ALIYUN_KEY = os.environ.get("ALIYUN_KEY",
    "sk-sp-D.LIEPX.T6GA.MEYCIQCmomYgcDaqBtAWGXYyua8ve/6RPImY/+0W2BuPqNXzZQIhAJwxunRW89LTfdar0B2A59wjemXSIAHYOPGBgIluBUCh")
ALIYUN_BASE = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODEL = os.environ.get("MODEL", "deepseek-v4-flash")


def setup_mem0():
    """Configure Mem0 to use Alibaba Cloud API for LLM, local embedding."""
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": MODEL,
                "api_key": ALIYUN_KEY,
                "openai_base_url": ALIYUN_BASE,
                "temperature": 0.3,
            }
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dims": 384,
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "locomo_mem0",
                "embedding_model_dims": 384,
            }
        },
        "version": "v1.1",
    }
    return Memory.from_config(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="locomo/data/locomo10.json")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-qa", type=int, default=100)
    parser.add_argument("--outdir", default="results_mem0")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    sample = data[args.sample]
    conv = sample["conversation"]
    speaker_a = conv.get("speaker_a", "Person A")
    speaker_b = conv.get("speaker_b", "Person B")

    os.makedirs(args.outdir, exist_ok=True)

    # Setup Mem0
    print(f"Setting up Mem0 with model={MODEL}...")
    try:
        m = setup_mem0()
    except Exception as e:
        print(f"Mem0 setup failed: {e}")
        print("Trying without custom embedder...")
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": MODEL,
                    "api_key": ALIYUN_KEY,
                    "openai_base_url": ALIYUN_BASE,
                    "temperature": 0.3,
                }
            },
            "version": "v1.1",
        }
        m = Memory.from_config(config)

    user_id = f"locomo_sample_{args.sample}"

    # Extract sessions
    sessions, dates = [], []
    i = 1
    while f"session_{i}" in conv:
        sessions.append(conv[f"session_{i}"])
        dates.append(conv.get(f"session_{i}_date_time", ""))
        i += 1

    if args.max_sessions:
        sessions = sessions[:args.max_sessions]
        dates = dates[:args.max_sessions]

    print(f"Sample {args.sample}: {len(sessions)} sessions")

    # Phase 1: Build memory
    t0 = time.time()
    total_messages = 0
    for si, (session, date) in enumerate(zip(sessions, dates)):
        turns = session if isinstance(session, list) else []
        messages = []
        for t in turns:
            if isinstance(t, dict):
                speaker = t.get("speaker", t.get("role", "user"))
                text = t.get("text", t.get("content", ""))
                role = "user" if speaker == speaker_a else "assistant"
                messages.append({"role": role, "content": f"{speaker}: {text}"})

        if not messages:
            continue

        try:
            m.add(messages, user_id=user_id, metadata={"session": si, "date": date})
            total_messages += len(messages)
            if (si + 1) % 5 == 0:
                print(f"  Session {si+1}/{len(sessions)} done ({total_messages} messages)")
        except Exception as e:
            print(f"  Session {si+1} error: {e}")

    build_time = time.time() - t0
    print(f"\nBuild done: {build_time:.0f}s, {total_messages} messages")

    # Get memory count
    try:
        all_mems = m.get_all(filters={"user_id": user_id})
        if isinstance(all_mems, dict) and "results" in all_mems:
            mem_count = len(all_mems["results"])
        elif isinstance(all_mems, list):
            mem_count = len(all_mems)
        else:
            mem_count = 0
        print(f"Total memories stored: {mem_count}")
    except Exception as e:
        mem_count = 0
        print(f"Could not count memories: {e}")

    # Phase 2: Evaluate
    qas = sample["qa"][:args.max_qa]
    print(f"\nEvaluating {len(qas)} questions...")

    results = []
    t1 = time.time()
    for qi, qa in enumerate(qas):
        question = qa["question"]
        gold = qa.get("answer", qa.get("adversarial_answer", "Not mentioned"))

        try:
            t_q = time.time()
            search_results = m.search(question, filters={"user_id": user_id}, limit=10)
            retrieve_time = time.time() - t_q

            # Format retrieved memories as context
            if isinstance(search_results, dict) and "results" in search_results:
                memories = search_results["results"]
            elif isinstance(search_results, list):
                memories = search_results
            else:
                memories = []

            context = "\n".join([
                mem.get("memory", mem.get("text", str(mem)))
                for mem in memories
            ]) if memories else "No relevant memories found."

            results.append({
                "question": question,
                "gold": gold,
                "category": qa.get("category", -1),
                "context": context[:500],
                "retrieve_time": round(retrieve_time, 3),
                "num_memories": len(memories),
            })

            if (qi + 1) % 25 == 0:
                print(f"  {qi+1}/{len(qas)} done")

        except Exception as e:
            results.append({
                "question": question, "gold": gold,
                "error": str(e), "retrieve_time": 0, "num_memories": 0,
            })

    eval_time = time.time() - t1

    # Save results
    output = {
        "method": "Mem0",
        "model": MODEL,
        "sample": args.sample,
        "build_time_s": round(build_time, 1),
        "eval_time_s": round(eval_time, 1),
        "num_sessions": len(sessions),
        "num_messages": total_messages,
        "num_questions": len(results),
        "results": results,
    }
    outfile = os.path.join(args.outdir, f"mem0_sample{args.sample}.json")
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outfile}")
    print(f"Build: {build_time:.0f}s, Eval: {eval_time:.0f}s")


if __name__ == "__main__":
    main()
