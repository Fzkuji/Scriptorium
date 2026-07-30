"""Run ablation experiments on LoCoMo.

Variants:
  full        — Full method (RSP + Copy + 5W1H + structured wiki)
  topical     — Copy evidence into ordinary topical files, no RSP
  no_copy     — Remove "COPY DON'T GENERATE" rule, let LLM rephrase freely
  no_rsp      — Remove RSP principle, store chronologically
  flat        — All content in one flat file, no folders
  no_memory   — No wiki, answer from LLM's own knowledge (lower bound)
  full_context — Stuff all conversation into context (upper bound)

Usage:
  python3 run_ablation.py --variant no_copy --sample 0 --proxy
  python3 run_ablation.py --variant full_context --sample 0 --proxy
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
from memory_builder import (
    load_locomo_sessions, execute_tool, log_usage,
    TOOLS, RETRIEVAL_SYSTEM_PROMPT, llm_judge_score
)

sys.stdout.reconfigure(line_buffering=True)

# --- Ablation prompt variants ---

PROMPT_FULL = memory_builder.SYSTEM_PROMPT  # original

PROMPT_NO_COPY = PROMPT_FULL.replace(
    "You are an ORGANIZER, not a writer — copy relevant content directly from the conversation, do not rephrase or summarize it.",
    "You are a memory manager. Summarize and rephrase conversation content in your own words for clarity and conciseness."
).replace(
    "- **COPY, DON'T GENERATE**: Store conversation content in its original wording. Do not rephrase, summarize, or generalize. If someone says \"I dyed my hair purple last week\", store exactly that.",
    "- **SUMMARIZE**: Distill conversation content into clear, concise notes. Rephrase for readability."
)

PROMPT_NO_RSP = PROMPT_FULL.replace(
    """## Storage Principle (RSP - Retrieval-Simulated Placement)
For each piece of information, ask yourself: "If I forgot this and needed to find it later, where would I look?"
Store it there. This ensures you can find it when you need it.""",
    """## Storage Principle
Store information chronologically. Create one file per conversation session, named by date. Append new information to the end of the file for that session."""
)

PROMPT_TOPICAL = PROMPT_FULL.replace(
    """## Storage Principle (RSP - Retrieval-Simulated Placement)
For each piece of information, ask yourself: "If I forgot this and needed to find it later, where would I look?"
Store it there. This ensures you can find it when you need it.""",
    """## Storage Principle
Store information in ordinary topical files. Group copied evidence by explicit topic names such as people, activities, places, organizations, objects, plans, and events.
Choose paths based on semantic topic labels, not by simulating the future retrieval route for a question."""
)

PROMPT_FLAT = PROMPT_FULL.replace(
    """## Structure Rules
- **Naming**: File and folder names must be concrete nouns (person names, project names, place names). Never use abstract words like "important", "work-related", or overly broad categories.
- **Organization**: Each file focuses on one entity or topic. Use markdown headings (##, ###) to organize sections within a file.
- **Cross-references**: Add markdown links [text](path.md) between related files in different folders.
- **Granularity**: Keep each directory manageable. If a folder has too many items, group them into subfolders with specific names. If a file grows very long, consider splitting it by subtopic.
- Always check the current state of the memory folder (ls, find) before making changes.
- You can create new files, new folders, append to existing files, or rewrite files as needed.""",
    """## Structure Rules
- Store ALL information in a single file called `memory.md` in the root of the memory folder.
- Use markdown headings (##, ###) to organize sections within this single file.
- Do NOT create subfolders or multiple files.
- Append new information to the appropriate section of `memory.md`."""
)


def assistant_message_to_dict(msg):
    """Convert assistant tool calls to JSON-serializable chat messages."""
    out = {
        "role": "assistant",
        "content": msg.content or "",
    }
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = []
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            out["tool_calls"].append({
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })
    return out


def safe_execute_tool(fn_name, fn_args, memory_dir):
    try:
        return execute_tool(fn_name, fn_args, memory_dir)
    except Exception as exc:
        return f"Error executing {fn_name}: {exc}"


def process_turn_with_prompt(turn_text, turn_date, memory_dir, system_prompt, model, client, max_rounds=10):
    """Process one conversation turn with a custom system prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"New conversation ({turn_date}):\n\n{turn_text}\n\nPlease extract and store any important information from this conversation into the memory folder. First check the current state of the folder, then decide what to store and where."}
    ]

    for round_i in range(max_rounds):
        for retry in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    max_tokens=4000,
                    temperature=0.3,
                )
                break
            except Exception as e:
                if retry < 2:
                    time.sleep(2)
                else:
                    raise

        log_usage(response, phase="build")

        choice = response.choices[0]
        msg = choice.message

        if msg.content:
            msg.content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            break

        messages.append(assistant_message_to_dict(msg))
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {}
            result = safe_execute_tool(fn_name, fn_args, memory_dir)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return len(messages)


def verify_with_client(memory_dir, qa_list, n_questions, model, client_obj, outdir, variant):
    """Run verification with specified client."""
    mem = Path(memory_dir)
    if not any(mem.rglob("*.md")):
        print("  WARNING: Memory folder is empty!")
        return [], 0, 0

    results = []
    correct = 0
    tested = 0
    CATEGORY_NAMES = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain", 5: "adversarial"}
    selected_qa = qa_list[:n_questions]

    for qi, qa in enumerate(selected_qa):
        question = qa["question"]
        answer = str(qa.get("answer", qa.get("adversarial_answer", "")))
        category = qa.get("category", 0)
        cat_name = CATEGORY_NAMES.get(category, f"cat{category}")

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

            messages.append(assistant_message_to_dict(msg))
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    fn_args = {}
                if fn_name in ("cat", "grep_headings") and "path" in fn_args:
                    files_visited.append(fn_args["path"])
                result = safe_execute_tool(fn_name, fn_args, memory_dir)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        judge_score = llm_judge_score(question, answer, response_text)
        is_found = judge_score >= 50
        if is_found:
            correct += 1
        tested += 1

        status = "FOUND" if is_found else "MISS"
        print(f"  [{qi+1}/{len(selected_qa)}] [{status}] [{cat_name}] LJ={judge_score} Q: {question[:50]}...")

        results.append({
            "question": question,
            "answer": answer,
            "response": response_text,
            "found": is_found,
            "judge_score": judge_score,
            "category": category,
            "category_name": cat_name,
            "files_visited": list(dict.fromkeys(files_visited)),
        })

        # Incremental save
        verify_file = os.path.join(outdir, f"verify_{variant}.json")
        with open(verify_file, "w") as f:
            json.dump({"variant": variant, "correct": correct, "tested": tested, "results": results}, f, indent=2, ensure_ascii=False)

    return results, correct, tested


def run_full_context(sample_idx, data_path, qa_list, n_questions, model, client_obj, outdir):
    """Upper bound: stuff all conversation into context, no wiki."""
    with open(data_path) as f:
        data = json.load(f)
    sample = data[sample_idx]

    # Build full conversation text
    session_keys = sorted(
        [k for k in sample["conversation"] if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1])
    )
    full_text = ""
    for key in session_keys:
        date = sample["conversation"].get(key + "_date_time", "")
        turns = sample["conversation"][key]
        full_text += f"\n--- {key} ({date}) ---\n"
        for turn in turns:
            full_text += f"{turn['speaker']}: {turn['text']}\n"

    print(f"  Full context: {len(full_text)} chars, {len(full_text)//4} est. tokens")

    results = []
    correct = 0
    tested = 0
    CATEGORY_NAMES = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain", 5: "adversarial"}
    selected_qa = qa_list[:n_questions]

    for qi, qa in enumerate(selected_qa):
        question = qa["question"]
        answer = str(qa.get("answer", qa.get("adversarial_answer", "")))
        category = qa.get("category", 0)
        cat_name = CATEGORY_NAMES.get(category, f"cat{category}")

        messages = [
            {"role": "system", "content": "You are answering questions about a user based on their conversation history. Use ONLY the provided conversations to answer. If you cannot find the answer, say 'NOT FOUND'."},
            {"role": "user", "content": f"Here are the user's conversations:\n\n{full_text}\n\nQuestion: {question}\n\nAnswer concisely based on the conversations above."}
        ]

        for retry in range(3):
            try:
                response = client_obj.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=500,
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
        else:
            log_usage(response, phase="verify")
            response_text = response.choices[0].message.content or ""
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

        judge_score = llm_judge_score(question, answer, response_text)
        is_found = judge_score >= 50
        if is_found:
            correct += 1
        tested += 1

        status = "FOUND" if is_found else "MISS"
        print(f"  [{qi+1}/{len(selected_qa)}] [{status}] [{cat_name}] LJ={judge_score} Q: {question[:50]}...")

        results.append({
            "question": question,
            "answer": answer,
            "response": response_text,
            "found": is_found,
            "judge_score": judge_score,
            "category": category,
            "category_name": cat_name,
        })

        verify_file = os.path.join(outdir, "verify_full_context.json")
        with open(verify_file, "w") as f:
            json.dump({"variant": "full_context", "correct": correct, "tested": tested, "results": results}, f, indent=2, ensure_ascii=False)

    return results, correct, tested


def run_no_memory(qa_list, n_questions, model, client_obj, outdir):
    """Lower bound: no memory, pure LLM knowledge."""
    results = []
    correct = 0
    tested = 0
    CATEGORY_NAMES = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain", 5: "adversarial"}
    selected_qa = qa_list[:n_questions]

    for qi, qa in enumerate(selected_qa):
        question = qa["question"]
        answer = str(qa.get("answer", qa.get("adversarial_answer", "")))
        category = qa.get("category", 0)
        cat_name = CATEGORY_NAMES.get(category, f"cat{category}")

        messages = [
            {"role": "user", "content": f"Answer this question about a person's life. If you don't know, say 'NOT FOUND'.\n\nQuestion: {question}"}
        ]

        for retry in range(3):
            try:
                response = client_obj.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=500,
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
        else:
            response_text = response.choices[0].message.content or ""
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

        judge_score = llm_judge_score(question, answer, response_text)
        is_found = judge_score >= 50
        if is_found:
            correct += 1
        tested += 1

        status = "FOUND" if is_found else "MISS"
        print(f"  [{qi+1}/{len(selected_qa)}] [{status}] [{cat_name}] LJ={judge_score} Q: {question[:50]}...")

        results.append({
            "question": question,
            "answer": answer,
            "response": response_text,
            "found": is_found,
            "judge_score": judge_score,
            "category": category,
            "category_name": cat_name,
        })

        verify_file = os.path.join(outdir, "verify_no_memory.json")
        with open(verify_file, "w") as f:
            json.dump({"variant": "no_memory", "correct": correct, "tested": tested, "results": results}, f, indent=2, ensure_ascii=False)

    return results, correct, tested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["full", "topical", "no_copy", "no_rsp", "flat", "no_memory", "full_context"])
    parser.add_argument("--data", default="locomo/data/locomo10.json")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--verify", type=int, default=50, help="Number of QA to verify per sample")
    parser.add_argument("--outdir", default="ablation_results")
    parser.add_argument("--proxy", action="store_true", help="Use ChatGPT proxy on localhost:8199")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--skip-build", action="store_true", help="Skip wiki build, only verify")
    args = parser.parse_args()

    if args.proxy:
        client_obj = OpenAI(api_key="dummy", base_url="http://localhost:8199/v1")
    else:
        client_obj = memory_builder.client

    memory_builder.ALIYUN_MODEL = args.model
    # Override client in memory_builder for judge calls
    memory_builder.client = client_obj
    memory_builder.JUDGE_MODEL = args.model

    os.makedirs(args.outdir, exist_ok=True)

    sessions, qa_list = load_locomo_sessions(args.data, args.sample)
    print(f"Sample {args.sample}: {len(sessions)} chunks, {len(qa_list)} QA")

    variant = args.variant

    # Special cases: no_memory and full_context don't need wiki build
    if variant == "no_memory":
        print(f"\n=== No Memory (lower bound) ===")
        results, correct, tested = run_no_memory(qa_list, args.verify, args.model, client_obj, args.outdir)
        if tested > 0:
            lj = sum(r["judge_score"] for r in results) / len(results)
            print(f"\nNo Memory: {correct}/{tested} = {correct/tested:.1%}, LJ={lj:.1f}")
        return

    if variant == "full_context":
        print(f"\n=== Full Context (upper bound) ===")
        results, correct, tested = run_full_context(args.sample, args.data, qa_list, args.verify, args.model, client_obj, args.outdir)
        if tested > 0:
            lj = sum(r["judge_score"] for r in results) / len(results)
            print(f"\nFull Context: {correct}/{tested} = {correct/tested:.1%}, LJ={lj:.1f}")
        return

    # Select prompt variant
    prompts = {
        "full": PROMPT_FULL,
        "topical": PROMPT_TOPICAL,
        "no_copy": PROMPT_NO_COPY,
        "no_rsp": PROMPT_NO_RSP,
        "flat": PROMPT_FLAT,
    }
    system_prompt = prompts[variant]

    memory_dir = os.path.join(args.outdir, f"sample{args.sample}_{variant}")

    if not args.skip_build:
        if os.path.exists(memory_dir):
            shutil.rmtree(memory_dir)
        os.makedirs(memory_dir)

        print(f"\n=== Building wiki: {variant} ===")
        memory_builder.TOTAL_CALLS = 0
        memory_builder.TOTAL_TOKENS = 0
        memory_builder.CALL_LOG = []

        t0 = time.time()
        for i, sess in enumerate(sessions):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Chunk {i+1}/{len(sessions)}")
            process_turn_with_prompt(sess["text"], sess["date"], memory_dir, system_prompt, args.model, client_obj)

        build_time = time.time() - t0
        file_count = len(list(Path(memory_dir).rglob("*.md")))
        total_size = sum(f.stat().st_size for f in Path(memory_dir).rglob("*.md"))
        print(f"  Built in {build_time:.0f}s: {file_count} files, {total_size//1024}KB")
        print(f"  LLM calls: {memory_builder.TOTAL_CALLS}, tokens: {memory_builder.TOTAL_TOKENS}")

    # Verify
    if not os.path.exists(memory_dir) or not any(Path(memory_dir).rglob("*.md")):
        print(f"  No wiki at {memory_dir}, skipping verify")
        return

    print(f"\n=== Verifying: {variant} ({args.verify} QA) ===")
    memory_builder.TOTAL_CALLS = 0
    memory_builder.TOTAL_TOKENS = 0
    memory_builder.CALL_LOG = []

    results, correct, tested = verify_with_client(memory_dir, qa_list, args.verify, args.model, client_obj, args.outdir, variant)

    if tested > 0:
        lj = sum(r["judge_score"] for r in results) / len(results)
        print(f"\n{variant}: {correct}/{tested} = {correct/tested:.1%}, LJ={lj:.1f}")

        # Per-category breakdown
        cat_stats = defaultdict(lambda: {"scores": [], "correct": 0, "total": 0})
        for r in results:
            cat = r["category_name"]
            cat_stats[cat]["scores"].append(r["judge_score"])
            cat_stats[cat]["total"] += 1
            if r["found"]:
                cat_stats[cat]["correct"] += 1
        for cat in sorted(cat_stats.keys()):
            s = cat_stats[cat]
            clj = sum(s["scores"]) / len(s["scores"])
            print(f"  {cat}: {s['correct']}/{s['total']} = {s['correct']/s['total']:.1%}, LJ={clj:.1f}")


if __name__ == "__main__":
    main()
