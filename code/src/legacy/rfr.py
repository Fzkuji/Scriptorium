"""RFR (Retrieval-Feedback Reorganization): Optimize wiki structure based on retrieval failures.

Analyzes MISS cases from verification, identifies why retrieval failed,
and reorganizes the wiki to improve future retrieval.

Usage:
    python3 rfr.py --verify-json memory_test_v2/verify_sample3_qwen3.6-flash.json \
                    --wiki-dir memory_test_v2/sample3_qwen3.6-flash
"""
import json
import argparse
import os
import sys
import re
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from memory_builder import client, ALIYUN_MODEL, execute_tool, TOOLS, log_usage, CALL_LOG

sys.stdout.reconfigure(line_buffering=True)

RFR_PROMPT = """You are a wiki structure optimizer. A memory system stored information in a wiki folder, but some questions could not be answered by searching the wiki.

## Your Task
Analyze why retrieval failed and fix the wiki structure so the information can be found next time.

## Failure Analysis
For each failed question below, I'll show:
- The question that was asked
- The expected answer
- What files the retriever looked at
- The retriever's response

## Common Failure Modes
1. **Missing cross-reference**: Info exists in file A, but the retriever looked in file B. Fix: add a cross-reference link in file B pointing to file A.
2. **Bad file naming**: The file name doesn't match what someone would search for. Fix: rename or add an alias heading.
3. **Scattered info**: Related facts are split across too many files. Fix: consolidate or add a summary section.
4. **Missing aggregation**: Individual events are stored but no summary exists (e.g., "how many times has X done Y?"). Fix: add a count/summary line.

## Rules
- Use the provided tools (ls, cat, write_file, append_file, grep_headings) to inspect and modify the wiki.
- Do NOT delete any existing content — only add cross-references, summaries, or reorganize.
- Keep changes minimal and targeted — only fix what's needed for the failed retrievals.
- Explain each change you make.
"""


def analyze_failures(verify_json_path: str, wiki_dir: str, max_fixes: int = 5):
    """Analyze verification failures and reorganize wiki."""
    with open(verify_json_path) as f:
        data = json.load(f)

    results = data.get("results", data) if isinstance(data, dict) else data
    failures = [r for r in results if not r.get("found")]

    if not failures:
        print("No failures to analyze!")
        return

    print(f"Found {len(failures)} failures to analyze (processing up to {max_fixes})")

    failure_descriptions = []
    for f in failures[:max_fixes]:
        desc = f"Question: {f['question']}\n"
        desc += f"Expected answer: {f['answer']}\n"
        desc += f"Retriever response: {f.get('response', 'N/A')}\n"
        desc += f"Files visited: {f.get('files_visited', [])}\n"
        desc += f"Category: {f.get('category_name', 'unknown')}\n"
        failure_descriptions.append(desc)

    failure_text = "\n---\n".join(failure_descriptions)

    messages = [
        {"role": "system", "content": RFR_PROMPT},
        {"role": "user", "content": f"Here are the failed retrievals:\n\n{failure_text}\n\nPlease analyze each failure, inspect the wiki, and make targeted fixes to improve future retrieval."}
    ]

    print(f"\nAsking LLM to analyze and fix {len(failures[:max_fixes])} failures...")

    for round_i in range(20):
        response = None
        for retry in range(3):
            try:
                response = client.chat.completions.create(
                    model=ALIYUN_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    max_tokens=4000,
                    temperature=0.3,
                )
                if response and response.choices:
                    break
            except Exception as e:
                if retry < 2:
                    import time; time.sleep(2)
                else:
                    raise

        if not response or not response.choices:
            print("WARNING: Empty API response")
            break

        log_usage(response, phase="rfr")
        choice = response.choices[0]
        msg = choice.message

        if msg.content:
            msg.content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()

        if not msg.tool_calls:
            if msg.content:
                print(f"\nRFR Analysis:\n{msg.content}")
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            if fn_name in ("write_file", "append_file"):
                print(f"  [RFR] {fn_name}: {fn_args.get('path', '?')}")

            result = execute_tool(fn_name, fn_args, wiki_dir)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

    print(f"\nRFR complete. {len(CALL_LOG)} LLM calls made.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-json", required=True, help="Path to verify results JSON")
    parser.add_argument("--wiki-dir", required=True, help="Path to wiki directory")
    parser.add_argument("--max-fixes", type=int, default=5, help="Max failures to fix")
    parser.add_argument("--model", default="qwen3.6-flash")
    args = parser.parse_args()

    import memory_builder
    memory_builder.ALIYUN_MODEL = args.model

    analyze_failures(args.verify_json, args.wiki_dir, args.max_fixes)


if __name__ == "__main__":
    main()
