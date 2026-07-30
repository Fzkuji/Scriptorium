"""Trace-smoke runner for NativeMem-style retrieval.

This script validates the artifact and per-question trace schema before larger
experiments. It can reuse an existing memory folder, run a small LoCoMo split,
and write auditable retrieval traces plus per-question outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None

sys.path.insert(0, os.path.dirname(__file__))
import memory_builder
from memory_builder import TOOLS, execute_tool, load_locomo_sessions


CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}

READ_TOOLS = [
    t for t in TOOLS
    if t["function"]["name"] in {"ls", "cat", "grep_headings", "find"}
]

RETRIEVAL_PROMPT = """You answer questions using only the personal memory folder.
Use the file tools to browse the folder. Do not use outside knowledge.

Procedure:
1. Start with ls at the root.
2. Choose the path whose name you would inspect first for the question.
3. Use grep_headings before reading a long file when helpful.
4. Use cat to read the needed section or file.
5. Follow links when the question involves multiple people, events, places, or projects.
6. If the first path fails, backtrack and inspect another plausible path.
7. Answer only after finding evidence. If evidence is absent, answer NOT FOUND.

Return a concise answer and include the evidence path you used."""


def get_encoder(name: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, enc: Any) -> int:
    if not text:
        return 0
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256_bytes(data)


def folder_sha256(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists():
        return ""
    for p in sorted(path.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(path)).replace(os.sep, "/")
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes()) if path.exists() else ""


def usage_dict(resp: Any) -> dict[str, int]:
    usage = getattr(resp, "usage", None)
    if not usage:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def assistant_message_to_dict(msg: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "role": "assistant",
        "content": getattr(msg, "content", None) or "",
    }
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return out


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def select_questions(qa_list: list[dict[str, Any]], per_category: int, max_questions: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    for cat in [1, 2, 3, 4, 5]:
        count = 0
        for idx, qa in enumerate(qa_list):
            if idx in used_ids or qa.get("category") != cat:
                continue
            item = dict(qa)
            item["_qa_index"] = idx
            selected.append(item)
            used_ids.add(idx)
            count += 1
            if count >= per_category:
                break
    for idx, qa in enumerate(qa_list):
        if len(selected) >= max_questions:
            break
        if idx in used_ids:
            continue
        item = dict(qa)
        item["_qa_index"] = idx
        selected.append(item)
        used_ids.add(idx)
    return selected[:max_questions]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def ensure_memory_artifact(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def judge_answer(
    client: OpenAI,
    model: str,
    question: str,
    gold_answer: str,
    system_answer: str,
    max_retries: int = 3,
) -> tuple[int, str, dict[str, int], float]:
    prompt = f"""You are evaluating a memory system answer against a reference.
Score factual accuracy and completeness from 0 to 100.

Question: {question}
Reference answer: {gold_answer}
System answer: {system_answer}

Guidelines:
0 means wrong or NOT FOUND when the answer exists.
25 means related topic but key facts are wrong.
50 means core facts are partly correct but major details are missing.
100 means the answer matches the reference with all important facts.

Output only one integer from 0 to 100."""

    last_error = ""
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            latency = time.time() - t0
            text = strip_thinking(resp.choices[0].message.content or "")
            match = re.search(r"\d+", text)
            score = int(match.group()) if match else 0
            return max(0, min(100, score)), text, usage_dict(resp), latency
        except Exception as exc:  # pragma: no cover
            last_error = str(exc)
            if attempt + 1 < max_retries:
                time.sleep(2)
    return 0, f"JUDGE_ERROR: {last_error}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0.0


def retrieve_question(
    client: OpenAI,
    model: str,
    memory_dir: Path,
    question: str,
    enc: Any,
    max_rounds: int,
    max_answer_tokens: int,
) -> tuple[str, dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": RETRIEVAL_PROMPT},
        {"role": "user", "content": f"Search the memory folder and answer this question:\n\n{question}"},
    ]
    trace: dict[str, Any] = {
        "retrieval_model": model,
        "max_rounds": max_rounds,
        "tool_calls": [],
        "files_opened": [],
        "first_content_file": None,
        "retrieval_usage": [],
        "answer_raw": "",
        "errors": [],
    }

    response_text = "NOT FOUND"
    retrieval_visible_tokens = 0
    content_observation_tokens = 0
    retrieval_prompt_tokens = 0
    retrieval_completion_tokens = 0
    retrieval_billable_tokens = 0
    retrieval_start = time.time()

    for round_i in range(max_rounds):
        round_start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=READ_TOOLS,
                max_tokens=max_answer_tokens,
                temperature=0,
            )
        except Exception as exc:
            trace["errors"].append({"round": round_i, "error": str(exc)})
            time.sleep(2)
            continue

        latency = time.time() - round_start
        u = usage_dict(resp)
        retrieval_prompt_tokens += u["prompt_tokens"]
        retrieval_completion_tokens += u["completion_tokens"]
        retrieval_billable_tokens += u["total_tokens"]
        trace["retrieval_usage"].append({"round": round_i, "latency_s": latency, **u})

        msg = resp.choices[0].message
        if msg.content:
            msg.content = strip_thinking(msg.content)

        if not msg.tool_calls:
            response_text = msg.content or "NOT FOUND"
            trace["answer_raw"] = response_text
            messages.append({"role": "assistant", "content": response_text})
            break

        messages.append(assistant_message_to_dict(msg))
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {}

            observation = execute_tool(fn_name, fn_args, str(memory_dir))
            observation_tokens = count_tokens(observation, enc)
            retrieval_visible_tokens += observation_tokens

            path = fn_args.get("path") if isinstance(fn_args, dict) else None
            content_bearing = fn_name in {"cat", "grep_headings"}
            if content_bearing:
                content_observation_tokens += observation_tokens
            if fn_name == "cat" and path:
                trace["files_opened"].append(path)
                if trace["first_content_file"] is None:
                    trace["first_content_file"] = path
            elif fn_name == "grep_headings" and path:
                if path not in trace["files_opened"]:
                    trace["files_opened"].append(path)

            call_record = {
                "round": round_i,
                "tool_call_id": tc.id,
                "function": fn_name,
                "arguments": fn_args,
                "path": path,
                "observation_chars": len(observation),
                "observation_tokens": observation_tokens,
                "content_bearing": content_bearing,
                "observation_preview": observation[:500],
            }
            trace["tool_calls"].append(call_record)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": observation})

    trace["retrieval_latency_s"] = round(time.time() - retrieval_start, 3)
    trace["retrieval_visible_tokens"] = retrieval_visible_tokens
    trace["content_observation_tokens"] = content_observation_tokens
    trace["retrieval_prompt_tokens"] = retrieval_prompt_tokens
    trace["retrieval_completion_tokens"] = retrieval_completion_tokens
    trace["retrieval_billable_tokens"] = retrieval_billable_tokens
    trace["retrieval_calls"] = len(trace["retrieval_usage"])
    trace["files_opened"] = list(dict.fromkeys(trace["files_opened"]))
    return response_text, trace


def write_manifest(root: Path, manifest_path: Path) -> None:
    rows: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p == manifest_path:
            continue
        rows.append(f"{file_sha256(p)}  {p.relative_to(root)}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="locomo/data/locomo10.json")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--method-id", default="S8")
    parser.add_argument("--method-name", default="NativeMem")
    parser.add_argument("--run-id", default="R001_trace_smoke")
    parser.add_argument("--build-id", default="build_smoke_qwen36")
    parser.add_argument("--memory-dir", required=True)
    parser.add_argument("--artifacts-dir", default="../artifacts")
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--retriever-model", default="qwen3.6-flash")
    parser.add_argument("--answer-model", default="qwen3.6-flash")
    parser.add_argument("--judge-model", default="qwen3.6-flash")
    parser.add_argument("--provider-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--tokenizer", default="o200k_base")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-answer-tokens", type=int, default=1024)
    parser.add_argument("--final-evidence-cap", type=int, default=6000)
    parser.add_argument("--intermediate-observation-cap", type=int, default=20000)
    args = parser.parse_args()

    memory_dir = Path(args.memory_dir)
    if not memory_dir.exists():
        raise SystemExit(f"memory folder does not exist: {memory_dir}")

    api_key = args.api_key or os.environ.get("EVAL_API_KEY") or memory_builder.API_KEY
    base_url = args.provider_base or os.environ.get("EVAL_API_BASE") or memory_builder.API_BASE
    client = OpenAI(api_key=api_key, base_url=base_url)
    enc = get_encoder(args.tokenizer)

    _, qa_list = load_locomo_sessions(args.data, args.sample)
    selected = select_questions(qa_list, args.per_category, args.max_questions)

    artifacts = Path(args.artifacts_dir)
    dataset_id = "D1"
    split_id = f"locomo_sample{args.sample}_smoke_{len(selected)}_v1"
    method_id = args.method_id
    build_id = args.build_id

    split_rows = []
    for qa in selected:
        qidx = qa["_qa_index"]
        category = qa.get("category", 0)
        split_rows.append({
            "split_id": split_id,
            "dataset_id": dataset_id,
            "source_dataset": args.data,
            "conversation_id": f"conv{args.sample}",
            "question_id": f"conv{args.sample}_q{qidx}",
            "qa_index": qidx,
            "category": category,
            "category_name": CATEGORY_NAMES.get(category, f"cat{category}"),
            "question": qa["question"],
        })
    split_path = artifacts / "splits" / f"{split_id}.jsonl"
    write_jsonl(split_path, split_rows)

    memory_artifact_dir = artifacts / "memory_folders" / dataset_id / method_id / build_id
    ensure_memory_artifact(memory_dir, memory_artifact_dir)

    config = {
        "run_id": args.run_id,
        "dataset_id": dataset_id,
        "split_id": split_id,
        "method_id": method_id,
        "method_name": args.method_name,
        "build_id": build_id,
        "source_memory_dir": str(memory_dir),
        "memory_artifact_dir": str(memory_artifact_dir),
        "retriever_model": args.retriever_model,
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "tokenizer": args.tokenizer,
        "max_rounds": args.max_rounds,
        "max_answer_tokens": args.max_answer_tokens,
        "final_evidence_cap": args.final_evidence_cap,
        "intermediate_observation_cap": args.intermediate_observation_cap,
        "retrieval_temperature": 0,
        "judge_temperature": 0,
        "question_count": len(selected),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    config["config_sha256"] = sha256_json(config)
    config_path = artifacts / "configs" / f"{args.run_id}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    memory_hash = folder_sha256(memory_artifact_dir)
    outputs: list[dict[str, Any]] = []
    judge_outputs: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    trace_dir = artifacts / "retrieval_traces" / dataset_id / method_id / build_id
    trace_dir.mkdir(parents=True, exist_ok=True)

    for i, qa in enumerate(selected, start=1):
        qidx = qa["_qa_index"]
        question_id = f"conv{args.sample}_q{qidx}"
        category = qa.get("category", 0)
        category_name = CATEGORY_NAMES.get(category, f"cat{category}")
        gold_answer = str(qa.get("answer", qa.get("adversarial_answer", "")))
        print(f"[{i}/{len(selected)}] {question_id} {category_name}: {qa['question'][:80]}")

        answer, trace = retrieve_question(
            client=client,
            model=args.retriever_model,
            memory_dir=memory_artifact_dir,
            question=qa["question"],
            enc=enc,
            max_rounds=args.max_rounds,
            max_answer_tokens=args.max_answer_tokens,
        )
        score, judge_raw, judge_usage, judge_latency = judge_answer(
            client=client,
            model=args.judge_model,
            question=qa["question"],
            gold_answer=gold_answer,
            system_answer=answer,
        )

        trace.update({
            "run_id": args.run_id,
            "dataset_id": dataset_id,
            "split_id": split_id,
            "question_id": question_id,
            "conversation_id": f"conv{args.sample}",
            "category_or_type": category_name,
            "question": qa["question"],
            "gold_answer": gold_answer,
            "system_answer": answer,
            "judge_model": args.judge_model,
            "judge_raw": judge_raw,
            "judge_usage": judge_usage,
            "judge_latency_s": judge_latency,
            "judge_score": score,
            "memory_folder_sha256": memory_hash,
            "config_sha256": config["config_sha256"],
        })
        trace_path = trace_dir / f"{question_id}.json"
        trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

        total_visible_tokens = trace["retrieval_visible_tokens"] + min(
            trace["content_observation_tokens"], args.final_evidence_cap
        )
        total_billable_tokens = trace["retrieval_billable_tokens"] + judge_usage["total_tokens"]
        record = {
            "run_id": args.run_id,
            "dataset_id": dataset_id,
            "split_id": split_id,
            "question_id": question_id,
            "conversation_id": f"conv{args.sample}",
            "category_or_type": category_name,
            "method_id": method_id,
            "method_name": args.method_name,
            "build_id": build_id,
            "writer_model": "existing_memory_folder",
            "retriever_model": args.retriever_model,
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
            "tokenizer": args.tokenizer,
            "final_evidence_cap": args.final_evidence_cap,
            "intermediate_observation_cap": args.intermediate_observation_cap,
            "total_visible_cap": None,
            "tool_call_limit": None,
            "retrieval_calls": trace["retrieval_calls"],
            "tool_calls": trace["tool_calls"],
            "files_opened": trace["files_opened"],
            "first_content_file": trace["first_content_file"],
            "target_evidence_files": [],
            "first_path_hit": None,
            "retrieval_visible_tokens": trace["retrieval_visible_tokens"],
            "final_evidence_tokens": min(trace["content_observation_tokens"], args.final_evidence_cap),
            "answer_prompt_tokens": trace["retrieval_prompt_tokens"],
            "answer_completion_tokens": trace["retrieval_completion_tokens"],
            "judge_prompt_tokens": judge_usage["prompt_tokens"],
            "judge_completion_tokens": judge_usage["completion_tokens"],
            "total_visible_tokens": total_visible_tokens,
            "total_billable_tokens": total_billable_tokens,
            "storage_amortized_tokens": 0,
            "latency_retrieval_s": trace["retrieval_latency_s"],
            "latency_answer_s": trace["retrieval_latency_s"],
            "latency_judge_s": judge_latency,
            "gold_answer": gold_answer,
            "system_answer": answer,
            "judge_score": score,
            "f1": None,
            "bleu1": None,
            "failure_tag": None,
            "trace_path": str(trace_path),
            "memory_folder_sha256": memory_hash,
            "config_sha256": config["config_sha256"],
        }
        outputs.append(record)
        judge_outputs.append({
            "question_id": question_id,
            "judge_model": args.judge_model,
            "judge_score": score,
            "judge_raw": judge_raw,
            "usage": judge_usage,
            "latency_s": judge_latency,
        })
        summary_rows.append({
            "question_id": question_id,
            "category": category_name,
            "judge_score": score,
            "retrieval_calls": trace["retrieval_calls"],
            "retrieval_visible_tokens": trace["retrieval_visible_tokens"],
            "final_evidence_tokens": record["final_evidence_tokens"],
            "total_visible_tokens": total_visible_tokens,
            "total_billable_tokens": total_billable_tokens,
            "files_opened": len(trace["files_opened"]),
            "first_content_file": trace["first_content_file"] or "",
        })
        print(
            f"  score={score} calls={trace['retrieval_calls']} "
            f"visible={total_visible_tokens} billable={total_billable_tokens}"
        )

    output_path = artifacts / "per_question_outputs" / dataset_id / method_id / f"{build_id}.jsonl"
    write_jsonl(output_path, outputs)
    judge_path = artifacts / "judge_outputs" / args.judge_model / dataset_id / method_id / f"{build_id}.jsonl"
    write_jsonl(judge_path, judge_outputs)

    table_path = artifacts / "tables" / f"{args.run_id}_summary.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest_path = artifacts / "manifests" / f"{args.run_id}.sha256"
    write_manifest(artifacts, manifest_path)

    avg_score = sum(r["judge_score"] for r in summary_rows) / len(summary_rows) if summary_rows else 0
    avg_visible = sum(r["total_visible_tokens"] for r in summary_rows) / len(summary_rows) if summary_rows else 0
    print("\nTrace smoke complete")
    print(f"  questions: {len(summary_rows)}")
    print(f"  avg_judge_score: {avg_score:.1f}")
    print(f"  avg_total_visible_tokens: {avg_visible:.0f}")
    print(f"  split: {split_path}")
    print(f"  outputs: {output_path}")
    print(f"  traces: {trace_dir}")
    print(f"  summary: {table_path}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
