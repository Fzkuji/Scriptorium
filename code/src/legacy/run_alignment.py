"""
Writer-Retriever Alignment Experiment on LongMemEval.

Build wikis with 3 different writer models, then evaluate each wiki
with all 3 retriever models. Produces a 3x3 alignment matrix.

Usage:
    python3 run_alignment.py --num-samples 30
"""
import json
import argparse
import os
import sys
import time
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import memory_builder
from memory_builder import process_conversation_turn, llm_judge_score, TOOLS, execute_tool
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)

ALIYUN_KEY = "sk-sp-D.LIEPX.T6GA.MEYCIQCmomYgcDaqBtAWGXYyua8ve/6RPImY/+0W2BuPqNXzZQIhAJwxunRW89LTfdar0B2A59wjemXSIAHYOPGBgIluBUCh"
ALIYUN_BASE = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

WRITER_MODELS = ["kimi-k2.6", "glm-5.2", "deepseek-v4-flash"]
RETRIEVER_MODELS = ["kimi-k2.6", "glm-5.2", "deepseek-v4-flash"]
JUDGE_MODEL = "deepseek-v4-flash"

CHUNK_SIZE = 10

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

READ_TOOLS = [t for t in TOOLS if t["function"]["name"] in ("ls", "cat", "grep_headings", "find")]


def build_wiki(data_item, writer_model, outdir):
    """Build wiki from LongMemEval sessions using a specific writer model."""
    sessions = data_item["haystack_sessions"]
    dates = data_item.get("haystack_dates", [""] * len(sessions))
    qid = data_item["question_id"]

    memory_dir = os.path.join(outdir, f"wiki_{writer_model}_{qid}")
    if os.path.exists(memory_dir):
        return memory_dir

    os.makedirs(memory_dir)

    memory_builder.ALIYUN_MODEL = writer_model
    memory_builder.API_KEY = ALIYUN_KEY
    memory_builder.API_BASE = ALIYUN_BASE
    memory_builder.client = OpenAI(api_key=ALIYUN_KEY, base_url=ALIYUN_BASE)
    memory_builder.TOTAL_CALLS = 0
    memory_builder.TOTAL_TOKENS = 0

    for si, (session, date) in enumerate(zip(sessions, dates)):
        for chunk_start in range(0, len(session), CHUNK_SIZE):
            chunk = session[chunk_start:chunk_start + CHUNK_SIZE]
            chunk_text = ""
            for t in chunk:
                chunk_text += f"{t.get('role', 'user')}: {t.get('content', '')}\n\n"
            if not chunk_text.strip():
                continue
            try:
                process_conversation_turn(chunk_text, date, memory_dir)
            except Exception as e:
                print(f"  Build error (session {si}): {e}")

    file_count = len(list(Path(memory_dir).rglob("*.md")))
    print(f"  Built wiki: {file_count} files, model={writer_model}, qid={qid}")
    return memory_dir


def retrieve_answer(question, memory_dir, retriever_model, max_rounds=10):
    """Retrieve answer from wiki using a specific retriever model."""
    client = OpenAI(api_key=ALIYUN_KEY, base_url=ALIYUN_BASE)

    messages = [
        {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nPlease find the answer in the wiki folder."}
    ]

    for round_i in range(max_rounds):
        try:
            response = client.chat.completions.create(
                model=retriever_model,
                messages=messages,
                tools=READ_TOOLS,
                max_tokens=2000,
                temperature=0.3,
            )
        except Exception as e:
            print(f"    Retrieval error round {round_i}: {e}")
            time.sleep(2)
            continue

        choice = response.choices[0]
        msg = choice.message

        if msg.content:
            import re
            msg.content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()

        if not msg.tool_calls:
            return msg.content or "NOT FOUND"

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

    return "NOT FOUND"


def judge_answer(question, gold_answer, system_answer):
    """Score with judge model."""
    client = OpenAI(api_key=ALIYUN_KEY, base_url=ALIYUN_BASE)
    prompt = f"""You are evaluating a memory system's answer. Score 0-100 based on factual accuracy.

Question: {question}
Ground Truth: {gold_answer}
System Answer: {system_answer}

Score 0 if completely wrong or NOT FOUND when answer exists.
Score 50+ if core facts are correct.
Score 100 if perfectly matches ground truth.

Output ONLY a number 0-100."""

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        nums = re.findall(r'\d+', content)
        return int(nums[0]) if nums else 0
    except Exception as e:
        print(f"    Judge error: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--outdir", default="alignment_experiment")
    args = parser.parse_args()

    with open("longmemeval/data/longmemeval_s_cleaned.json") as f:
        data = json.load(f)

    # Sample questions with fewer sessions (faster to build)
    data_sorted = sorted(data, key=lambda x: len(x.get("haystack_sessions", [])))
    samples = data_sorted[:args.num_samples]
    print(f"Selected {len(samples)} samples (fewest sessions)")
    print(f"Session counts: {[len(s['haystack_sessions']) for s in samples[:5]]}...")

    os.makedirs(args.outdir, exist_ok=True)

    # Phase 1: Build wikis with each writer model
    for writer in WRITER_MODELS:
        print(f"\n=== Building wikis with writer: {writer} ===")
        for i, sample in enumerate(samples):
            print(f"Sample {i+1}/{len(samples)} (qid={sample['question_id']})")
            build_wiki(sample, writer, args.outdir)

    # Phase 2: Evaluate each writer's wiki with each retriever
    results = {}
    for writer in WRITER_MODELS:
        results[writer] = {}
        for retriever in RETRIEVER_MODELS:
            print(f"\n=== Evaluating: writer={writer}, retriever={retriever} ===")
            scores = []
            for i, sample in enumerate(samples):
                qid = sample["question_id"]
                wiki_dir = os.path.join(args.outdir, f"wiki_{writer}_{qid}")
                if not os.path.exists(wiki_dir):
                    print(f"  Wiki not found for {qid}, skipping")
                    continue

                answer = retrieve_answer(sample["question"], wiki_dir, retriever)
                score = judge_answer(sample["question"], sample["answer"], answer)
                scores.append(score)

                if (i + 1) % 10 == 0:
                    avg = sum(scores) / len(scores)
                    print(f"  Progress: {i+1}/{len(samples)}, running avg: {avg:.1f}")

            avg_score = sum(scores) / len(scores) if scores else 0
            results[writer][retriever] = {
                "avg_score": avg_score,
                "scores": scores,
                "count": len(scores)
            }
            print(f"  Result: writer={writer}, retriever={retriever}, LJ={avg_score:.1f}")

    # Save results
    results_file = os.path.join(args.outdir, "alignment_matrix.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print alignment matrix
    print("\n\n=== ALIGNMENT MATRIX (LJ Score) ===")
    print(f"{'Writer / Retriever':<25}", end="")
    for r in RETRIEVER_MODELS:
        print(f"{r:<20}", end="")
    print()
    for w in WRITER_MODELS:
        print(f"{w:<25}", end="")
        for r in RETRIEVER_MODELS:
            score = results[w][r]["avg_score"]
            marker = " *" if w == r else ""
            print(f"{score:<18.1f}{marker}", end="")
        print()

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
