"""Run LongMemEval on sampled questions via ChatGPT proxy.

Samples N questions per type, builds wiki, evaluates.
"""
import json
import argparse
import os
import sys
import time
import re
import shutil
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
import memory_builder
from memory_builder import process_conversation_turn, llm_judge_score, log_usage, TOOLS, execute_tool

sys.stdout.reconfigure(line_buffering=True)

RETRIEVAL_SYSTEM_PROMPT = memory_builder.RETRIEVAL_SYSTEM_PROMPT
CHUNK_SIZE = 10


def build_wiki_for_sample(sample, sample_idx, outdir, model, client_obj):
    sessions = sample["haystack_sessions"]
    dates = sample.get("haystack_dates", [""] * len(sessions))

    memory_dir = os.path.join(outdir, f"lme_sample{sample_idx}")
    if os.path.exists(memory_dir):
        shutil.rmtree(memory_dir)
    os.makedirs(memory_dir)

    memory_builder.TOTAL_CALLS = 0
    memory_builder.TOTAL_TOKENS = 0
    memory_builder.CALL_LOG = []

    t0 = time.time()
    total_chunks = 0

    for si, (session, date) in enumerate(zip(sessions, dates)):
        turns = session
        for chunk_start in range(0, len(turns), CHUNK_SIZE):
            chunk = turns[chunk_start:chunk_start + CHUNK_SIZE]
            chunk_text = ""
            for t in chunk:
                chunk_text += f"{t.get('role', 'user')}: {t.get('content', '')}\n\n"
            if not chunk_text.strip():
                continue
            total_chunks += 1
            process_conversation_turn(chunk_text, date if isinstance(date, str) else "", memory_dir)

    build_time = time.time() - t0
    file_count = len(list(Path(memory_dir).rglob("*.md")))
    return memory_dir, build_time, total_chunks, file_count


def evaluate_sample(sample, memory_dir, model, client_obj):
    question = sample["question"]
    answer = sample["answer"]
    q_type = sample["question_type"]

    messages = [
        {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
        {"role": "user", "content": f"Search the memory wiki and answer this question:\n\n{question}"}
    ]

    response_text = ""
    files_visited = []

    for round_i in range(15):
        response = None
        for retry in range(3):
            try:
                response = client_obj.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    max_tokens=2000,
                    temperature=0.3,
                )
                if response and response.choices:
                    break
            except Exception as e:
                if retry < 2:
                    time.sleep(2)
                else:
                    response = None
                    break

        if not response or not response.choices:
            response_text = "NOT FOUND"
            break

        log_usage(response, phase="verify")
        choice = response.choices[0]
        msg = choice.message

        if msg.content:
            msg.content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()

        if not msg.tool_calls:
            response_text = msg.content or ""
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}
            if fn_name in ("cat", "grep_headings") and "path" in fn_args:
                files_visited.append(fn_args["path"])
            result = execute_tool(fn_name, fn_args, memory_dir)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    judge_score = llm_judge_score(question, answer, response_text)
    return {
        "question": question,
        "answer": answer,
        "response": response_text,
        "judge_score": judge_score,
        "found": judge_score >= 50,
        "question_type": q_type,
        "files_visited": list(dict.fromkeys(files_visited)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="longmemeval/data/longmemeval_s_cleaned.json")
    parser.add_argument("--per-type", type=int, default=5)
    parser.add_argument("--outdir", default="longmemeval_sampled")
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument("--model", default="gpt-5.5")
    args = parser.parse_args()

    if args.proxy:
        client_obj = OpenAI(api_key="dummy", base_url="http://localhost:8199/v1")
    else:
        client_obj = memory_builder.client

    memory_builder.ALIYUN_MODEL = args.model
    memory_builder.client = client_obj
    memory_builder.JUDGE_MODEL = args.model

    with open(args.data) as f:
        data = json.load(f)

    by_type = defaultdict(list)
    for i, d in enumerate(data):
        by_type[d["question_type"]].append(i)

    selected = []
    for qtype in sorted(by_type.keys()):
        indices = by_type[qtype][:args.per_type]
        selected.extend(indices)
        print(f"  {qtype}: {len(indices)} samples")

    print(f"\nTotal selected: {len(selected)} samples")
    os.makedirs(args.outdir, exist_ok=True)

    results = []
    results_file = os.path.join(args.outdir, "lme_results.json")
    done_indices = set()
    try:
        with open(results_file) as f:
            results = json.load(f)
        done_indices = {r["sample_idx"] for r in results}
        print(f"Resuming from {len(results)} already done")
    except:
        pass

    for qi, idx in enumerate(selected):
        if idx in done_indices:
            continue

        sample = data[idx]
        qtype = sample["question_type"]
        print(f"\n[{qi+1}/{len(selected)}] Sample {idx} ({qtype})")
        print(f"  Q: {sample['question'][:80]}...")

        memory_dir, build_time, chunks, files = build_wiki_for_sample(
            sample, idx, args.outdir, args.model, client_obj)
        print(f"  Built: {chunks} chunks, {files} files, {build_time:.0f}s")

        result = evaluate_sample(sample, memory_dir, args.model, client_obj)
        result["sample_idx"] = idx
        result["build_time"] = build_time
        result["build_chunks"] = chunks
        result["wiki_files"] = files

        status = "FOUND" if result["found"] else "MISS"
        print(f"  [{status}] LJ={result['judge_score']} A: {result['response'][:80]}...")

        results.append(result)

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        shutil.rmtree(memory_dir, ignore_errors=True)

    print(f"\n=== FINAL RESULTS ({len(results)} samples) ===")
    overall_lj = sum(r["judge_score"] for r in results) / len(results) if results else 0
    overall_acc = sum(1 for r in results if r["found"]) / len(results) if results else 0
    print(f"Overall: LJ={overall_lj:.1f}, Acc={overall_acc:.1%}")

    type_stats = defaultdict(lambda: {"scores": [], "correct": 0, "total": 0})
    for r in results:
        t = r["question_type"]
        type_stats[t]["scores"].append(r["judge_score"])
        type_stats[t]["total"] += 1
        if r["found"]:
            type_stats[t]["correct"] += 1

    for t in sorted(type_stats):
        s = type_stats[t]
        tlj = sum(s["scores"]) / len(s["scores"])
        print(f"  {t}: LJ={tlj:.1f}, Acc={s['correct']}/{s['total']}")


if __name__ == "__main__":
    main()
