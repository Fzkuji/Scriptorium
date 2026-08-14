"""
Writer-Retriever Alignment on LoCoMo.

3 conversations x 3 writer models x 3 retriever models.
Each conversation: build wiki once per writer, evaluate 100 questions per retriever.

Usage:
    python3 run_alignment_locomo.py
"""
import json
import os
import sys
import time
import re
import shutil
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
import memory_builder
from memory_builder import process_conversation_turn, TOOLS, execute_tool

sys.stdout.reconfigure(line_buffering=True)

ALIYUN_KEY = "sk-sp-D.LIEPX.T6GA.MEYCIQCmomYgcDaqBtAWGXYyua8ve/6RPImY/+0W2BuPqNXzZQIhAJwxunRW89LTfdar0B2A59wjemXSIAHYOPGBgIluBUCh"
ALIYUN_BASE = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

MODELS = ["kimi-k2.6", "glm-5.2", "deepseek-v4-flash"]
JUDGE_MODEL = "deepseek-v4-flash"
CONV_IDS = [0, 1, 2]
MAX_QA = 100
CHUNK_SIZE = 10
OUTDIR = "alignment_locomo"

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
    """Extract sessions from LoCoMo conversation dict."""
    sessions = []
    dates = []
    i = 1
    while f"session_{i}" in conv:
        sessions.append(conv[f"session_{i}"])
        dates.append(conv.get(f"session_{i}_date_time", ""))
        i += 1
    return sessions, dates


def build_wiki(conv_data, conv_id, writer_model):
    """Build wiki for one conversation with one writer model."""
    wiki_dir = os.path.join(OUTDIR, f"conv{conv_id}_{writer_model}")
    if os.path.exists(wiki_dir) and len(list(Path(wiki_dir).rglob("*.md"))) > 0:
        fc = len(list(Path(wiki_dir).rglob("*.md")))
        print(f"  Wiki exists: {fc} files, skipping build")
        return wiki_dir

    if os.path.exists(wiki_dir):
        shutil.rmtree(wiki_dir)
    os.makedirs(wiki_dir)

    memory_builder.ALIYUN_MODEL = writer_model
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

    print(f"  Building: {len(sessions)} sessions, model={writer_model}")
    t0 = time.time()
    chunks = 0

    for si, (session, date) in enumerate(zip(sessions, dates)):
        turns = session if isinstance(session, list) else []
        if isinstance(session, str):
            turns = [{"role": "user", "content": session}]

        for chunk_start in range(0, len(turns), CHUNK_SIZE):
            chunk = turns[chunk_start:chunk_start + CHUNK_SIZE]
            chunk_text = ""
            for t in chunk:
                if isinstance(t, dict):
                    speaker = t.get("speaker", t.get("role", "user"))
                    text = t.get("text", t.get("content", ""))
                elif isinstance(t, str):
                    speaker = "user"
                    text = t
                else:
                    continue
                chunk_text += f"{speaker}: {text}\n\n"

            if not chunk_text.strip():
                continue
            chunks += 1
            try:
                process_conversation_turn(chunk_text, date, wiki_dir)
            except Exception as e:
                print(f"    Error session {si} chunk {chunk_start}: {e}")

    elapsed = time.time() - t0
    fc = len(list(Path(wiki_dir).rglob("*.md")))
    total_bytes = sum(f.stat().st_size for f in Path(wiki_dir).rglob("*.md"))

    build_stats = {
        "model": writer_model,
        "conv_id": conv_id,
        "num_sessions": len(sessions),
        "num_chunks": chunks,
        "wall_clock_s": round(elapsed, 1),
        "num_llm_calls": len(memory_builder.CALL_LOG),
        "calls_per_chunk": round(len(memory_builder.CALL_LOG) / max(chunks, 1), 1),
        "total_prompt_tokens": sum(e.get("prompt_tokens", 0) for e in memory_builder.CALL_LOG),
        "total_completion_tokens": sum(e.get("completion_tokens", 0) for e in memory_builder.CALL_LOG),
        "total_tokens": sum(e.get("total_tokens", 0) for e in memory_builder.CALL_LOG),
        "avg_prompt_per_call": round(sum(e.get("prompt_tokens", 0) for e in memory_builder.CALL_LOG) / max(len(memory_builder.CALL_LOG), 1)),
        "avg_completion_per_call": round(sum(e.get("completion_tokens", 0) for e in memory_builder.CALL_LOG) / max(len(memory_builder.CALL_LOG), 1)),
        "max_context": max((e.get("prompt_tokens", 0) for e in memory_builder.CALL_LOG), default=0),
        "wiki_files": fc,
        "wiki_bytes": total_bytes,
        "per_call_log": memory_builder.CALL_LOG,
    }

    # Save directory tree snapshot
    tree_lines = []
    for p in sorted(Path(wiki_dir).rglob("*")):
        rel = p.relative_to(wiki_dir)
        if p.is_file():
            tree_lines.append(f"{rel} ({p.stat().st_size}B)")
        else:
            tree_lines.append(f"{rel}/")
    build_stats["wiki_tree"] = tree_lines

    stats_file = os.path.join(OUTDIR, f"build_stats_conv{conv_id}_{writer_model}.json")
    with open(stats_file, "w") as f:
        json.dump(build_stats, f, indent=2)

    print(f"  Done: {fc} files, {chunks} chunks, {elapsed:.0f}s, {build_stats['num_llm_calls']} calls, {build_stats['total_tokens']} tokens")
    return wiki_dir


def retrieve(question, wiki_dir, retriever_model, max_rounds=10):
    """Retrieve answer using a specific model. Returns (answer, stats)."""
    client = OpenAI(api_key=ALIYUN_KEY, base_url=ALIYUN_BASE)
    messages = [
        {"role": "system", "content": RETRIEVAL_PROMPT},
        {"role": "user", "content": f"Question: {question}"}
    ]

    t0 = time.time()
    calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    context_lengths = []
    tools_used = []
    tools_used_detail = []

    for _ in range(max_rounds):
        try:
            resp = client.chat.completions.create(
                model=retriever_model, messages=messages,
                tools=READ_TOOLS, max_tokens=2000, temperature=0.3
            )
        except Exception as e:
            time.sleep(2)
            continue

        calls += 1
        if resp.usage:
            prompt_tokens += resp.usage.prompt_tokens
            completion_tokens += resp.usage.completion_tokens
            context_lengths.append(resp.usage.prompt_tokens)

        msg = resp.choices[0].message
        if msg.content:
            msg.content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()

        if not msg.tool_calls:
            stats = {
                "calls": calls, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "max_context": max(context_lengths) if context_lengths else 0,
                "wall_clock_s": round(time.time() - t0, 2),
                "tools_used": tools_used,
                "tools_used_detail": tools_used_detail,
            }
            return msg.content or "NOT FOUND", stats

        messages.append(msg)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except:
                args = {}
            tools_used.append(tc.function.name)
            tools_used_detail.append({
                "tool": tc.function.name,
                "path": args.get("path", ""),
                "round": calls,
            })
            result = execute_tool(tc.function.name, args, wiki_dir)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    stats = {
        "calls": calls, "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "max_context": max(context_lengths) if context_lengths else 0,
        "wall_clock_s": round(time.time() - t0, 2),
        "tools_used": tools_used,
        "tools_used_detail": tools_used_detail,
    }
    return "NOT FOUND", stats


def judge(question, gold, answer):
    """Score answer 0-100."""
    client = OpenAI(api_key=ALIYUN_KEY, base_url=ALIYUN_BASE)
    prompt = f"""Score 0-100 based on factual accuracy.
Question: {question}
Ground Truth: {gold}
System Answer: {answer}
Output ONLY a number 0-100."""
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=10, temperature=0
        )
        content = resp.choices[0].message.content.strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        nums = re.findall(r'\d+', content)
        return int(nums[0]) if nums else 0
    except:
        return 0


def main():
    with open("locomo/data/locomo10.json") as f:
        data = json.load(f)

    os.makedirs(OUTDIR, exist_ok=True)

    # Phase 1: Build wikis
    for conv_id in CONV_IDS:
        for writer in MODELS:
            print(f"\n=== Conv {conv_id}, Writer: {writer} ===")
            build_wiki(data[conv_id], conv_id, writer)

    # Phase 2: Evaluate
    results = {w: {r: [] for r in MODELS} for w in MODELS}

    for conv_id in CONV_IDS:
        qas = data[conv_id]["qa"][:MAX_QA]
        print(f"\n=== Evaluating Conv {conv_id}: {len(qas)} questions ===")

        for writer in MODELS:
            wiki_dir = os.path.join(OUTDIR, f"conv{conv_id}_{writer}")
            if not os.path.exists(wiki_dir):
                print(f"  Wiki missing: {wiki_dir}")
                continue

            for retriever in MODELS:
                print(f"  Writer={writer}, Retriever={retriever}")
                scores = []
                ret_stats_list = []
                for qi, qa in enumerate(qas):
                    gold = qa.get("answer", qa.get("adversarial_answer", "Not mentioned"))
                    ans, rstats = retrieve(qa["question"], wiki_dir, retriever)
                    sc = judge(qa["question"], gold, ans)
                    scores.append(sc)
                    rstats["question"] = qa["question"]
                    rstats["gold_answer"] = gold
                    rstats["system_answer"] = ans
                    rstats["score"] = sc
                    rstats["category"] = qa.get("category", -1)
                    rstats["first_file"] = next((t for t in rstats.get("tools_used_detail", []) if t["tool"] in ("cat", "grep_headings")), {}).get("path", "")
                    ret_stats_list.append(rstats)
                    if (qi + 1) % 25 == 0:
                        avg_sc = sum(scores) / len(scores)
                        avg_calls = sum(s["calls"] for s in ret_stats_list) / len(ret_stats_list)
                        avg_tok = sum(s["total_tokens"] for s in ret_stats_list) / len(ret_stats_list)
                        print(f"    {qi+1}/{len(qas)}, avg_score={avg_sc:.1f}, avg_calls={avg_calls:.1f}, avg_tokens={avg_tok:.0f}")

                results[writer][retriever].extend(scores)
                avg = sum(scores) / len(scores) if scores else 0

                ret_summary = {
                    "writer": writer, "retriever": retriever, "conv_id": conv_id,
                    "num_questions": len(ret_stats_list),
                    "avg_score": avg,
                    "avg_calls": round(sum(s["calls"] for s in ret_stats_list) / max(len(ret_stats_list), 1), 1),
                    "avg_prompt_tokens": round(sum(s["prompt_tokens"] for s in ret_stats_list) / max(len(ret_stats_list), 1)),
                    "avg_completion_tokens": round(sum(s["completion_tokens"] for s in ret_stats_list) / max(len(ret_stats_list), 1)),
                    "avg_total_tokens": round(sum(s["total_tokens"] for s in ret_stats_list) / max(len(ret_stats_list), 1)),
                    "avg_wall_clock_s": round(sum(s["wall_clock_s"] for s in ret_stats_list) / max(len(ret_stats_list), 1), 2),
                    "max_context": max((s["max_context"] for s in ret_stats_list), default=0),
                }
                ret_summary["per_question"] = ret_stats_list
                stats_file = os.path.join(OUTDIR, f"ret_stats_conv{conv_id}_{writer}_{retriever}.json")
                with open(stats_file, "w") as f:
                    json.dump(ret_summary, f, indent=2)

                print(f"    Conv {conv_id} done: avg={avg:.1f}, avg_calls={ret_summary['avg_calls']}, avg_tokens={ret_summary['avg_total_tokens']}")

    # Save
    final = {}
    for w in MODELS:
        final[w] = {}
        for r in MODELS:
            sc = results[w][r]
            final[w][r] = {"avg": sum(sc)/len(sc) if sc else 0, "n": len(sc)}

    with open(os.path.join(OUTDIR, "alignment_matrix.json"), "w") as f:
        json.dump(final, f, indent=2)

    # Print matrix
    print("\n\n=== ALIGNMENT MATRIX (LJ Score) ===")
    print(f"{'Writer \\ Retriever':<22}", end="")
    for r in MODELS:
        print(f"{r:<18}", end="")
    print()
    for w in MODELS:
        print(f"{w:<22}", end="")
        for r in MODELS:
            s = final[w][r]["avg"]
            mark = " *" if w == r else ""
            print(f"{s:<16.1f}{mark}", end="")
        print()


if __name__ == "__main__":
    main()
