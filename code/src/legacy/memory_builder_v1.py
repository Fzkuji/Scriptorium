"""
Memory Builder: LLM manages a markdown wiki folder as long-term memory.

Each conversation turn is processed one at a time. The LLM decides what to
extract, where to store it, and how to organize the file structure.

The LLM has access to shell commands (ls, cat, grep, mkdir, write) to
manage the memory folder. No index file, no embedding, no database.

Usage:
    python3 memory_builder.py --data locomo/data/locomo10.json --sample 0 --sessions 5
"""

import json
import argparse
import os
import sys
import time
import re
from pathlib import Path
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)

ALIYUN_KEY = "sk-sp-D.LIEPX.T6GA.MEYCIQCmomYgcDaqBtAWGXYyua8ve/6RPImY/+0W2BuPqNXzZQIhAJwxunRW89LTfdar0B2A59wjemXSIAHYOPGBgIluBUCh"
ALIYUN_MODEL = "qwen3.6-flash"
ALIYUN_BASE = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

API_KEY = os.environ.get("EVAL_API_KEY", ALIYUN_KEY)
API_BASE = os.environ.get("EVAL_API_BASE", ALIYUN_BASE)

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

TOTAL_CALLS = 0
TOTAL_TOKENS = 0
CALL_LOG = []  # Per-call token usage log


def log_usage(response, phase="unknown"):
    """Record per-call token usage."""
    global TOTAL_CALLS, TOTAL_TOKENS, CALL_LOG
    TOTAL_CALLS += 1
    entry = {"call_id": TOTAL_CALLS, "phase": phase, "model": ALIYUN_MODEL}
    if response.usage:
        u = response.usage
        entry["prompt_tokens"] = u.prompt_tokens
        entry["completion_tokens"] = u.completion_tokens
        entry["total_tokens"] = u.total_tokens
        entry["cached_tokens"] = getattr(u, "prompt_tokens_details", None)
        if hasattr(u, "prompt_tokens_details") and u.prompt_tokens_details:
            entry["cached_tokens"] = getattr(u.prompt_tokens_details, "cached_tokens", 0)
        TOTAL_TOKENS += u.total_tokens
    CALL_LOG.append(entry)


def call_llm(messages: list, max_tokens: int = 4000) -> str:
    global TOTAL_CALLS, TOTAL_TOKENS
    try:
        r = client.chat.completions.create(
            model=ALIYUN_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        content = r.choices[0].message.content.strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        log_usage(r, phase="call_llm")
        return content
    except Exception as e:
        print(f"  LLM error: {e}")
        return ""


# =============================================================================
# Shell tool simulation for the LLM
# =============================================================================

def execute_tool(tool_name: str, args: dict, memory_dir: str) -> str:
    """Execute a file system tool within the memory directory."""
    mem = Path(memory_dir)

    if tool_name == "ls":
        target = mem / args.get("path", "")
        if not target.exists():
            return f"Error: {args.get('path', '')} does not exist"
        if target.is_dir():
            items = sorted(target.iterdir())
            result = []
            for item in items:
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    result.append(f"{item.name}/")
                else:
                    result.append(item.name)
            return "\n".join(result) if result else "(empty directory)"
        return target.name

    elif tool_name == "cat":
        if "path" not in args or not args["path"]:
            return "Error: 'path' argument is required"
        target = mem / args["path"]
        if not target.exists():
            return f"Error: {args['path']} does not exist"
        if target.is_dir():
            return f"Error: {args['path']} is a directory, not a file. Use ls to list its contents."
        return target.read_text()

    elif tool_name == "grep_headings":
        if "path" not in args or not args["path"]:
            return "Error: 'path' argument is required"
        target = mem / args["path"]
        if not target.exists():
            return f"Error: {args['path']} does not exist"
        if target.is_dir():
            return f"Error: {args['path']} is a directory, not a file"
        lines = target.read_text().splitlines()
        headings = [l for l in lines if l.startswith("#")]
        return "\n".join(headings) if headings else "(no headings)"

    elif tool_name == "write_file":
        if "path" not in args or not args["path"]:
            return "Error: 'path' argument is required"
        target = mem / args["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["content"])
        return f"Written to {args['path']}"

    elif tool_name == "append_file":
        if "path" not in args or not args["path"]:
            return "Error: 'path' argument is required"
        target = mem / args["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = target.read_text() if target.exists() else ""
        target.write_text(existing + "\n" + args["content"])
        return f"Appended to {args['path']}"

    elif tool_name == "mkdir":
        if "path" not in args or not args["path"]:
            return "Error: 'path' argument is required"
        target = mem / args["path"]
        target.mkdir(parents=True, exist_ok=True)
        return f"Created directory {args['path']}"

    elif tool_name == "find":
        results = []
        for f in sorted(mem.rglob("*.md")):
            results.append(str(f.relative_to(mem)))
        return "\n".join(results) if results else "(no files)"

    else:
        return f"Unknown tool: {tool_name}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": "List files and directories in a path within the memory folder",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path within memory folder. Use '' for root."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cat",
            "description": "Read the full content of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep_headings",
            "description": "Extract all markdown headings (lines starting with #) from a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates or overwrites)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file"},
                    "content": {"type": "string", "description": "Full content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append content to an existing file (or create if not exists)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file"},
                    "content": {"type": "string", "description": "Content to append"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mkdir",
            "description": "Create a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path of directory to create"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "List all .md files in the entire memory folder recursively",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

SYSTEM_PROMPT = """You are a personal memory manager. You maintain a markdown wiki folder as long-term memory for a user.

## Your Task
After each conversation, organize and store information into the memory folder using the provided tools. You are an ORGANIZER, not a writer — copy relevant content directly from the conversation, do not rephrase or summarize it.

## Storage Principle (RSP - Retrieval-Simulated Placement)
For each piece of information, ask yourself: "If I forgot this and needed to find it later, where would I look?"
Store it there. This ensures you can find it when you need it.

## Information Structure (5W1H)
When storing information, preserve the complete event structure (Tulving, 1972; 5W1H framework):
- **Who**: people involved, their names, nicknames, how they address each other
- **What**: what happened, what was said, specific names of things (movies, books, places, brands, pets)
- **When**: dates, time references ("last week", "in 2019"), durations
- **Where**: locations, places
- **Why**: reasons, motivations, causes
- **How**: methods, manner, details of how something was done

## Content Rules
- **COPY, DON'T GENERATE**: Store conversation content in its original wording. Do not rephrase, summarize, or generalize. If someone says "I dyed my hair purple last week", store exactly that.
- **Filter**: Only store user-specific information (personal facts, preferences, experiences, decisions). Skip assistant's generic knowledge responses (e.g., a list of yoga poses, a recipe, a code snippet the user didn't write).
- **Conflicts**: When new information contradicts existing memory (e.g., user moved to a new city), append the new info with a timestamp — do NOT overwrite the old entry. Keep both versions; the timestamp shows which is newer.
- Each memory entry should have a timestamp: [YYYY-MM-DD] followed by the original conversation content.

## Structure Rules
- **Naming**: File and folder names must be concrete nouns (person names, project names, place names). Never use abstract words like "important", "work-related", or overly broad categories.
- **Organization**: Each file focuses on one entity or topic. Use markdown headings (##, ###) to organize sections within a file.
- **Cross-references**: Add markdown links [text](path.md) between related files in different folders.
- **Granularity**: Keep each directory manageable. If a folder has too many items, group them into subfolders with specific names. If a file grows very long, consider splitting it by subtopic.
- Always check the current state of the memory folder (ls, find) before making changes.
- You can create new files, new folders, append to existing files, or rewrite files as needed.
"""


def process_conversation_turn(turn_text: str, turn_date: str, memory_dir: str, max_rounds: int = 10):
    """Process one conversation turn: LLM decides what to store and where."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"New conversation ({turn_date}):\n\n{turn_text}\n\nPlease extract and store any important information from this conversation into the memory folder. First check the current state of the folder, then decide what to store and where."}
    ]

    for round_i in range(max_rounds):
        for retry in range(3):
            try:
                response = client.chat.completions.create(
                    model=ALIYUN_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    max_tokens=4000,
                    temperature=0.3,
                )
                break
            except Exception as e:
                if retry < 2:
                    import time as _t; _t.sleep(2)
                else:
                    raise

        log_usage(response, phase="build")

        choice = response.choices[0]
        msg = choice.message

        # Strip thinking tags from content
        if msg.content:
            msg.content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()

        # If no tool calls, we're done
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            break

        # Process tool calls
        messages.append(msg)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            result = execute_tool(fn_name, fn_args, memory_dir)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

    return len(messages)


def load_locomo_sessions(data_path: str, sample_idx: int):
    """Load conversation sessions from LoCoMo dataset."""
    with open(data_path) as f:
        data = json.load(f)
    sample = data[sample_idx]

    session_keys = sorted(
        [k for k in sample["conversation"] if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1])
    )

    CHUNK_SIZE = 10  # Process 10 turns at a time for better detail retention

    sessions = []
    for key in session_keys:
        date = sample["conversation"].get(key + "_date_time", "")
        turns = sample["conversation"][key]
        # Split into chunks of CHUNK_SIZE turns
        for chunk_start in range(0, len(turns), CHUNK_SIZE):
            chunk_turns = turns[chunk_start:chunk_start + CHUNK_SIZE]
            turn_text = ""
            for turn in chunk_turns:
                turn_text += f"{turn['speaker']}: {turn['text']}\n"
            chunk_label = f"{key}_chunk{chunk_start // CHUNK_SIZE + 1}" if len(turns) > CHUNK_SIZE else key
            sessions.append({"key": chunk_label, "date": date, "text": turn_text, "turns": chunk_turns})

    qa_list = sample.get("qa", [])
    return sessions, qa_list


RETRIEVAL_SYSTEM_PROMPT = """You are answering questions using ONLY a personal memory wiki folder. You have access to file system tools to browse the wiki.

## How to find information
1. Use `ls` or `find` to see what files exist
2. Use `grep_headings` to see the structure of a file
3. Use `cat` to read file content
4. Follow markdown links to related files if needed

## Rules
- ONLY use information found in the wiki files. Do NOT use your training knowledge.
- If you cannot find the answer in the wiki after searching, say "NOT FOUND".
- Answer concisely once you find the information.
- You decide which files to look at and how many — there is no limit.
"""


JUDGE_MODEL = "qwen3.6-flash"  # Fixed judge model for consistent scoring


def llm_judge_score(question: str, ground_truth: str, system_answer: str) -> int:
    """Use LLM-as-Judge to score the system's answer against ground truth (0-100)."""
    if not system_answer or not system_answer.strip():
        return 0

    judge_prompt = f"""You are evaluating a memory system's answer. Score 0-100 based on factual accuracy.

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
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            log_usage(response, phase="judge")
            text = response.choices[0].message.content.strip()
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            score = int(re.search(r'\d+', text).group())
            return max(0, min(100, score))
        except Exception:
            if retry < 2:
                import time as _t; _t.sleep(1)
    return 0


def verify_storage(memory_dir: str, qa_list: list, n_questions: int = 10):
    """Verify storage quality: LLM uses tools to search wiki and answer questions."""
    mem = Path(memory_dir)

    if not any(mem.rglob("*.md")):
        print("  WARNING: Memory folder is empty!")
        return 0, 0

    correct = 0
    tested = 0
    results = []
    selected_qa = qa_list[:n_questions]

    CATEGORY_NAMES = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain", 5: "adversarial"}

    for qa in selected_qa:
        question = qa["question"]
        answer = str(qa.get("answer", qa.get("adversarial_answer", "")))
        is_adversarial = qa.get("category") == 5
        category = qa.get("category", 0)

        messages = [
            {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Search the memory wiki and answer this question:\n\n{question}"}
        ]

        response_text = ""
        files_visited = []
        qa_calls_before = TOTAL_CALLS
        qa_tokens_before = TOTAL_TOKENS

        for round_i in range(15):  # max 15 tool call rounds
            response = None
            for retry in range(3):
                try:
                    response = client.chat.completions.create(
                        model=ALIYUN_MODEL,
                        messages=messages,
                        tools=TOOLS,
                        max_tokens=2000,
                        temperature=0.3,
                    )
                    if response and response.choices:
                        break
                except Exception as e:
                    if retry < 2:
                        import time as _t; _t.sleep(2)
                    else:
                        print(f"    WARNING: API error after retries: {e}")
                        response = None
                        break

            if not response or not response.choices:
                print(f"    WARNING: Empty API response, skipping question")
                response_text = "NOT FOUND"
                break

            log_usage(response, phase="verify")

            choice = response.choices[0]
            msg = choice.message

            if msg.content:
                msg.content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()

            if not msg.tool_calls:
                response_text = msg.content or ""
                messages.append({"role": "assistant", "content": response_text})
                break

            messages.append(msg)
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                # Track which files were visited
                if fn_name in ("cat", "grep_headings") and "path" in fn_args:
                    files_visited.append(fn_args["path"])

                result = execute_tool(fn_name, fn_args, memory_dir)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

        # LLM-as-Judge scoring
        judge_score = llm_judge_score(question, answer, response_text)
        is_found = judge_score >= 50

        if is_found:
            correct += 1
        tested += 1

        num_hops = len(set(files_visited))
        cat_name = CATEGORY_NAMES.get(category, f"cat{category}")
        status = "FOUND" if is_found else "MISS"
        print(f"    [{status}] [{cat_name}] (LJ={judge_score}) Q: {question[:50]}...")
        print(f"           A: {response_text[:80]}")
        if not is_found:
            print(f"           Expected: {answer[:60]}")
        if files_visited:
            print(f"           Files: {list(dict.fromkeys(files_visited))[:5]}, Hops: {num_hops}")

        qa_calls = TOTAL_CALLS - qa_calls_before
        qa_tokens = TOTAL_TOKENS - qa_tokens_before

        results.append({
            "question": question,
            "answer": answer,
            "response": response_text,
            "found": is_found,
            "judge_score": judge_score,
            "category": category,
            "category_name": cat_name,
            "files_visited": list(dict.fromkeys(files_visited)),
            "num_hops": num_hops,
            "llm_calls": qa_calls,
            "tokens_used": qa_tokens,
        })

    # Per-category stats
    cat_stats = {}
    for r in results:
        cn = r["category_name"]
        if cn not in cat_stats:
            cat_stats[cn] = {"found": 0, "total": 0, "scores": []}
        cat_stats[cn]["total"] += 1
        cat_stats[cn]["scores"].append(r["judge_score"])
        if r["found"]:
            cat_stats[cn]["found"] += 1

    avg_score = sum(r["judge_score"] for r in results) / len(results) if results else 0
    print(f"\n  Average LJ score: {avg_score:.1f}")
    print(f"  Per-category:")
    for cn in ["single-hop", "temporal", "multi-hop", "open-domain", "adversarial"]:
        if cn in cat_stats:
            s = cat_stats[cn]
            cat_avg = sum(s["scores"]) / len(s["scores"]) if s["scores"] else 0
            print(f"    {cn}: {s['found']}/{s['total']} = {s['found']/s['total']:.1%} (LJ={cat_avg:.1f})")

    # Save detailed results (convert scores list to avg for JSON)
    cat_stats_save = {}
    for cn, s in cat_stats.items():
        cat_stats_save[cn] = {"found": s["found"], "total": s["total"],
                              "avg_lj_score": sum(s["scores"]) / len(s["scores"]) if s["scores"] else 0}

    memory_dir = memory_dir.rstrip("/")
    results_path = os.path.join(os.path.dirname(memory_dir), f"verify_{os.path.basename(memory_dir)}.json")
    with open(results_path, "w") as f:
        json.dump({"results": results, "per_category": cat_stats_save,
                   "overall": correct/tested if tested else 0,
                   "avg_lj_score": avg_score},
                  f, indent=2, ensure_ascii=False)

    return correct, tested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="locomo/data/locomo10.json")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--sessions", type=int, default=5, help="Number of sessions to process (0=all)")
    parser.add_argument("--verify", type=int, default=10, help="Number of QA to verify")
    parser.add_argument("--outdir", default="memory_test")
    parser.add_argument("--model", default="qwen3.6-flash")
    args = parser.parse_args()

    global ALIYUN_MODEL
    ALIYUN_MODEL = args.model

    memory_dir = os.path.join(args.outdir, f"sample{args.sample}_{args.model}")
    if os.path.exists(memory_dir):
        import shutil
        shutil.rmtree(memory_dir)
    os.makedirs(memory_dir)

    sessions, qa_list = load_locomo_sessions(args.data, args.sample)

    if args.sessions > 0:
        sessions = sessions[:args.sessions]

    print(f"Sample {args.sample}: {len(sessions)} sessions, {len(qa_list)} QA pairs")
    print(f"Model: {args.model}")
    print(f"Memory dir: {memory_dir}")

    # Phase 1: Build memory
    print(f"\n=== Phase 1: Building memory ===")
    t0 = time.time()

    for i, sess in enumerate(sessions):
        print(f"\n  Session {i+1}/{len(sessions)} ({sess['key']}, {sess['date']})")
        print(f"    {len(sess['turns'])} turns, {len(sess['text'])} chars")

        msg_count = process_conversation_turn(sess["text"], sess["date"], memory_dir)
        print(f"    Processed ({msg_count} messages, {TOTAL_CALLS} total LLM calls)")

        # Show current wiki state
        file_count = len(list(Path(memory_dir).rglob("*.md")))
        total_size = sum(f.stat().st_size for f in Path(memory_dir).rglob("*.md"))
        print(f"    Wiki: {file_count} files, {total_size} bytes")

    build_time = time.time() - t0
    print(f"\n  Build complete in {build_time:.0f}s ({TOTAL_CALLS} LLM calls, {TOTAL_TOKENS} tokens)")

    # Show final wiki structure
    print(f"\n=== Wiki Structure ===")
    for f in sorted(Path(memory_dir).rglob("*.md")):
        rel = f.relative_to(memory_dir)
        size = f.stat().st_size
        print(f"  {rel} ({size} bytes)")

    # Phase 2: Verify storage quality
    if args.verify > 0 and qa_list:
        print(f"\n=== Phase 2: Verify storage ({args.verify} questions) ===")
        correct, tested = verify_storage(memory_dir, qa_list, args.verify)
        if tested > 0:
            print(f"\n  Storage quality: {correct}/{tested} = {correct/tested:.1%}")
        else:
            print(f"\n  No questions to verify")

    # Compute per-phase token stats
    build_calls = [c for c in CALL_LOG if c["phase"] == "build"]
    verify_calls = [c for c in CALL_LOG if c["phase"] == "verify"]
    build_tokens = sum(c.get("total_tokens", 0) for c in build_calls)
    verify_tokens = sum(c.get("total_tokens", 0) for c in verify_calls)
    build_prompt = sum(c.get("prompt_tokens", 0) for c in build_calls)
    build_completion = sum(c.get("completion_tokens", 0) for c in build_calls)
    verify_prompt = sum(c.get("prompt_tokens", 0) for c in verify_calls)
    verify_completion = sum(c.get("completion_tokens", 0) for c in verify_calls)
    cached = sum(c.get("cached_tokens", 0) for c in CALL_LOG if c.get("cached_tokens"))

    print(f"\n=== Token Usage ===")
    print(f"  Build:  {len(build_calls)} calls, {build_tokens} tokens (prompt: {build_prompt}, completion: {build_completion})")
    print(f"  Verify: {len(verify_calls)} calls, {verify_tokens} tokens (prompt: {verify_prompt}, completion: {verify_completion})")
    print(f"  Total:  {TOTAL_CALLS} calls, {TOTAL_TOKENS} tokens")
    if cached:
        print(f"  Cached: {cached} tokens")

    # Save stats
    stats = {
        "sample": args.sample,
        "model": args.model,
        "sessions_processed": len(sessions),
        "build_time": build_time,
        "total_llm_calls": TOTAL_CALLS,
        "total_tokens": TOTAL_TOKENS,
        "build_calls": len(build_calls),
        "build_tokens": build_tokens,
        "build_prompt_tokens": build_prompt,
        "build_completion_tokens": build_completion,
        "verify_calls": len(verify_calls),
        "verify_tokens": verify_tokens,
        "verify_prompt_tokens": verify_prompt,
        "verify_completion_tokens": verify_completion,
        "cached_tokens": cached,
        "file_count": len(list(Path(memory_dir).rglob("*.md"))),
    }
    with open(os.path.join(args.outdir, f"stats_sample{args.sample}.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Save full call log
    with open(os.path.join(args.outdir, f"call_log_sample{args.sample}.json"), "w") as f:
        json.dump(CALL_LOG, f, indent=2)


if __name__ == "__main__":
    main()
