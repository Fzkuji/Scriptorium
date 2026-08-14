"""
Efficiency & Cost Measurement for WikiMem.

Measures per-conversation and per-question costs:
- Number of LLM calls (build / retrieve)
- Token usage (prompt / completion / total)
- Context length per call
- Wall-clock time
- Wiki size (files, total bytes)

Run after alignment experiment to analyze existing logs,
or run standalone on a single conversation.

Usage:
    # Standalone: build + evaluate one LoCoMo conversation
    python3 run_efficiency.py --conv 0 --model deepseek-v4-flash --max-qa 50

    # Analyze existing alignment experiment logs
    python3 run_efficiency.py --analyze alignment_locomo
"""
import json
import os
import sys
import time
import re
import shutil
from pathlib import Path
from openai import OpenAI
import argparse

sys.path.insert(0, os.path.dirname(__file__))
import memory_builder
from memory_builder import process_conversation_turn, TOOLS, execute_tool

sys.stdout.reconfigure(line_buffering=True)

ALIYUN_KEY = "sk-sp-D.LIEPX.T6GA.MEYCIQCmomYgcDaqBtAWGXYyua8ve/6RPImY/+0W2BuPqNXzZQIhAJwxunRW89LTfdar0B2A59wjemXSIAHYOPGBgIluBUCh"
ALIYUN_BASE = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

RETRIEVAL_PROMPT = """You are answering questions using ONLY a personal memory wiki folder. You have access to file system tools to browse the wiki.

## How to find information
1. Use `ls` or `find` to see what files exist
2. Use `grep_headings` to see the structure of a file
3. Use `cat` to read file content
4. Follow markdown links to related files if needed

## Rules
- ONLY use information found in the wiki files. Do NOT use your training knowledge.
- If you cannot find the answer in the wiki after searching, say "NOT FOUND".
- Answer concisely once you find the information.
"""

READ_TOOLS = [t for t in TOOLS if t["function"]["name"] in ("ls", "cat", "grep_headings", "find")]


def get_sessions(conv):
    sessions = []
    dates = []
    i = 1
    while f"session_{i}" in conv:
        sessions.append(conv[f"session_{i}"])
        dates.append(conv.get(f"session_{i}_date_time", ""))
        i += 1
    return sessions, dates


def build_wiki_with_stats(conv_data, conv_id, model, outdir):
    """Build wiki and return detailed stats."""
    wiki_dir = os.path.join(outdir, f"efficiency_conv{conv_id}_{model}")
    if os.path.exists(wiki_dir):
        shutil.rmtree(wiki_dir)
    os.makedirs(wiki_dir)

    memory_builder.ALIYUN_MODEL = model
    memory_builder.API_KEY = ALIYUN_KEY
    memory_builder.API_BASE = ALIYUN_BASE
    memory_builder.client = OpenAI(api_key=ALIYUN_KEY, base_url=ALIYUN_BASE)
    memory_builder.TOTAL_CALLS = 0
    memory_builder.TOTAL_TOKENS = 0
    memory_builder.CALL_LOG = []

    conv = conv_data["conversation"]
    sessions, dates = get_sessions(conv)
    speaker_a = conv.get("speaker_a", "Person A")
    speaker_b = conv.get("speaker_b", "Person B")

    t0 = time.time()
    chunks = 0
    chunk_times = []

    for si, (session, date) in enumerate(zip(sessions, dates)):
        turns = session if isinstance(session, list) else []
        if isinstance(session, str):
            turns = [{"role": "user", "content": session}]

        for chunk_start in range(0, len(turns), 10):
            chunk = turns[chunk_start:chunk_start + 10]
            chunk_text = ""
            for t in chunk:
                if isinstance(t, dict):
                    role = t.get("role", "user")
                    content = t.get("content", "")
                elif isinstance(t, str):
                    content = t
                    role = "user"
                else:
                    continue
                name = speaker_a if role == "user" else speaker_b
                chunk_text += f"{name}: {content}\n\n"

            if not chunk_text.strip():
                continue

            chunks += 1
            ct0 = time.time()
            try:
                process_conversation_turn(chunk_text, date, wiki_dir)
            except Exception as e:
                print(f"  Error: {e}")
            chunk_times.append(time.time() - ct0)

    build_time = time.time() - t0
    build_log = list(memory_builder.CALL_LOG)

    # Wiki stats
    md_files = list(Path(wiki_dir).rglob("*.md"))
    file_count = len(md_files)
    total_bytes = sum(f.stat().st_size for f in md_files)
    total_lines = sum(len(f.read_text().splitlines()) for f in md_files)

    # Compute per-call stats
    prompt_tokens = [e.get("prompt_tokens", 0) for e in build_log]
    completion_tokens = [e.get("completion_tokens", 0) for e in build_log]
    total_tokens = [e.get("total_tokens", 0) for e in build_log]

    stats = {
        "phase": "build",
        "model": model,
        "conv_id": conv_id,
        "num_sessions": len(sessions),
        "num_chunks": chunks,
        "wall_clock_seconds": round(build_time, 1),
        "avg_chunk_time_seconds": round(sum(chunk_times) / len(chunk_times), 1) if chunk_times else 0,
        "num_llm_calls": len(build_log),
        "calls_per_chunk": round(len(build_log) / chunks, 1) if chunks else 0,
        "total_prompt_tokens": sum(prompt_tokens),
        "total_completion_tokens": sum(completion_tokens),
        "total_tokens": sum(total_tokens),
        "avg_prompt_tokens_per_call": round(sum(prompt_tokens) / len(prompt_tokens), 0) if prompt_tokens else 0,
        "avg_completion_tokens_per_call": round(sum(completion_tokens) / len(completion_tokens), 0) if completion_tokens else 0,
        "max_context_length": max(prompt_tokens) if prompt_tokens else 0,
        "wiki_files": file_count,
        "wiki_bytes": total_bytes,
        "wiki_lines": total_lines,
    }

    return wiki_dir, stats


def retrieve_with_stats(question, wiki_dir, model, max_rounds=10):
    """Retrieve and return answer + stats."""
    client = OpenAI(api_key=ALIYUN_KEY, base_url=ALIYUN_BASE)
    messages = [
        {"role": "system", "content": RETRIEVAL_PROMPT},
        {"role": "user", "content": f"Question: {question}"}
    ]

    t0 = time.time()
    calls = 0
    prompt_tokens_total = 0
    completion_tokens_total = 0
    context_lengths = []

    for _ in range(max_rounds):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                tools=READ_TOOLS, max_tokens=2000, temperature=0.3
            )
        except Exception as e:
            time.sleep(2)
            continue

        calls += 1
        if resp.usage:
            prompt_tokens_total += resp.usage.prompt_tokens
            completion_tokens_total += resp.usage.completion_tokens
            context_lengths.append(resp.usage.prompt_tokens)

        msg = resp.choices[0].message
        if msg.content:
            msg.content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()

        if not msg.tool_calls:
            elapsed = time.time() - t0
            stats = {
                "num_calls": calls,
                "prompt_tokens": prompt_tokens_total,
                "completion_tokens": completion_tokens_total,
                "total_tokens": prompt_tokens_total + completion_tokens_total,
                "max_context": max(context_lengths) if context_lengths else 0,
                "wall_clock": round(elapsed, 2),
            }
            return msg.content or "NOT FOUND", stats

        messages.append(msg)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except:
                args = {}
            result = execute_tool(tc.function.name, args, wiki_dir)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    elapsed = time.time() - t0
    stats = {
        "num_calls": calls,
        "prompt_tokens": prompt_tokens_total,
        "completion_tokens": completion_tokens_total,
        "total_tokens": prompt_tokens_total + completion_tokens_total,
        "max_context": max(context_lengths) if context_lengths else 0,
        "wall_clock": round(elapsed, 2),
    }
    return "NOT FOUND", stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv", type=int, default=0)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--max-qa", type=int, default=50)
    parser.add_argument("--outdir", default="efficiency_results")
    args = parser.parse_args()

    with open("locomo/data/locomo10.json") as f:
        data = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)

    # Build
    print(f"=== Building wiki: conv={args.conv}, model={args.model} ===")
    wiki_dir, build_stats = build_wiki_with_stats(data[args.conv], args.conv, args.model, args.outdir)
    print(f"\nBuild stats:")
    for k, v in build_stats.items():
        print(f"  {k}: {v}")

    # Retrieve
    qas = data[args.conv]["qa"][:args.max_qa]
    print(f"\n=== Evaluating: {len(qas)} questions ===")

    retrieval_stats_list = []
    for qi, qa in enumerate(qas):
        ans, rstats = retrieve_with_stats(qa["question"], wiki_dir, args.model)
        retrieval_stats_list.append(rstats)
        if (qi + 1) % 10 == 0:
            avg_calls = sum(s["num_calls"] for s in retrieval_stats_list) / len(retrieval_stats_list)
            avg_tokens = sum(s["total_tokens"] for s in retrieval_stats_list) / len(retrieval_stats_list)
            print(f"  {qi+1}/{len(qas)}: avg_calls={avg_calls:.1f}, avg_tokens={avg_tokens:.0f}")

    # Aggregate retrieval stats
    retrieval_summary = {
        "phase": "retrieve",
        "model": args.model,
        "num_questions": len(retrieval_stats_list),
        "avg_calls_per_question": round(sum(s["num_calls"] for s in retrieval_stats_list) / len(retrieval_stats_list), 1),
        "avg_prompt_tokens": round(sum(s["prompt_tokens"] for s in retrieval_stats_list) / len(retrieval_stats_list), 0),
        "avg_completion_tokens": round(sum(s["completion_tokens"] for s in retrieval_stats_list) / len(retrieval_stats_list), 0),
        "avg_total_tokens": round(sum(s["total_tokens"] for s in retrieval_stats_list) / len(retrieval_stats_list), 0),
        "avg_wall_clock": round(sum(s["wall_clock"] for s in retrieval_stats_list) / len(retrieval_stats_list), 2),
        "max_context_seen": max(s["max_context"] for s in retrieval_stats_list),
    }

    print(f"\nRetrieval stats:")
    for k, v in retrieval_summary.items():
        print(f"  {k}: {v}")

    # Save all
    output = {
        "build": build_stats,
        "retrieval": retrieval_summary,
        "per_question": retrieval_stats_list,
    }
    outfile = os.path.join(args.outdir, f"efficiency_conv{args.conv}_{args.model}.json")
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
