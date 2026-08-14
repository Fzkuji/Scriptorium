"""Formal trace runner for file-memory retrieval experiments.

This runner separates retrieval from final answering. The retrieval model may
browse a memory folder with file tools, but the answer model receives only a
bounded evidence context assembled from retrieved observations. This keeps the
final answer evidence cap explicit while preserving retrieval traces and cost
accounting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
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


class QuestionTimeout(RuntimeError):
    pass


def _raise_question_timeout(signum, frame):
    raise QuestionTimeout("question timed out")

READ_TOOLS = [
    t for t in TOOLS
    if t["function"]["name"] in {"ls", "cat", "grep_headings", "find"}
]

RETRIEVAL_PROMPT = """You retrieve evidence from a personal memory folder.
Use the file tools to browse the folder. Do not use outside knowledge.

Procedure:
1. Start with ls at the root.
2. Choose the path whose name you would inspect first for the question.
3. Use grep_headings before reading a long file when helpful.
4. Use cat to read the needed section or file.
5. Follow links when the question involves multiple people, events, places, or projects.
6. If the first path fails, backtrack and inspect another plausible path.
7. Stop after you have found enough evidence for a separate answer model.

Do not answer the question. Return only the evidence paths and a short note
about what each path contains. If evidence is absent, return NOT FOUND."""

ANSWER_PROMPT = """You answer a benchmark question using only the retrieved evidence.
Do not use outside knowledge. If the evidence is absent or insufficient, answer
NOT FOUND. Keep the answer concise.

For causal or counterfactual questions, use directly relevant causal statements
in the evidence to answer the most likely outcome. Do not output NOT FOUND only
because the evidence does not state the counterfactual verbatim.

Question:
{question}

Retrieved evidence:
{evidence}

Answer:"""


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


def truncate_to_token_cap(text: str, cap: int, enc: Any) -> str:
    if cap <= 0:
        return ""
    if count_tokens(text, enc) <= cap:
        return text
    if enc is None:
        return text[: max(1, cap * 4)]
    return enc.decode(enc.encode(text)[:cap])


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


def gold_answer_for_qa(qa: dict[str, Any]) -> str:
    if qa.get("category") == 5:
        if "answer" in qa and qa["answer"] is not None:
            return str(qa["answer"])
        return "Not mentioned in the conversation"
    return str(qa.get("answer", qa.get("adversarial_answer", "")))


def build_evidence_context(evidence_items: list[dict[str, Any]], cap: int, enc: Any) -> tuple[str, int]:
    parts: list[str] = []
    used = 0
    for idx, item in enumerate(evidence_items, start=1):
        if used >= cap:
            break
        header = f"\n[Evidence {idx}: {item['function']} {item.get('path') or ''}]\n"
        header_tokens = count_tokens(header, enc)
        if used + header_tokens >= cap:
            break
        remaining = cap - used - header_tokens
        content = truncate_to_token_cap(item["content"], remaining, enc)
        content_tokens = count_tokens(content, enc)
        parts.append(header + content)
        used += header_tokens + content_tokens
    context = "\n".join(parts).strip()
    return context, count_tokens(context, enc)


def rank_evidence_items_by_selector(
    evidence_items: list[dict[str, Any]],
    selector_text: str,
) -> list[dict[str, Any]]:
    if not selector_text:
        return evidence_items
    selector_lower = selector_text.lower()

    def rank(pair: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        idx, item = pair
        path = str(item.get("path") or "").lower()
        if not path:
            return (10**9, idx)
        pos = selector_lower.find(path)
        if pos >= 0:
            return (pos, idx)
        basename = path.rsplit("/", 1)[-1]
        pos = selector_lower.find(basename)
        if pos >= 0:
            return (pos, idx)
        return (10**9, idx)

    return [item for _, item in sorted(enumerate(evidence_items), key=rank)]


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summary_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": record["question_id"],
        "category": record["category_or_type"],
        "judge_score": record["judge_score"],
        "retrieval_calls": record["retrieval_calls"],
        "retrieval_visible_tokens": record["retrieval_visible_tokens"],
        "final_evidence_tokens": record["final_evidence_tokens"],
        "answer_prompt_tokens": record["answer_prompt_tokens"],
        "total_visible_tokens": record["total_visible_tokens"],
        "total_billable_tokens": record["total_billable_tokens"],
        "files_opened": len(record.get("files_opened", [])),
        "first_content_file": record.get("first_content_file") or "",
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_memory_artifact(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def load_locomo_data(data_path: str) -> list[dict[str, Any]]:
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


def load_selected_questions(
    data_path: str,
    sample_idx: int,
    per_category: int,
    max_questions: int,
    split_file: str | None,
) -> tuple[list[dict[str, Any]], str, Path | None]:
    if split_file:
        split_path = Path(split_file)
        rows = [
            json.loads(line)
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        data = load_locomo_data(data_path)
        selected: list[dict[str, Any]] = []
        for row in rows:
            row_sample_idx = int(row["sample_idx"])
            qa_idx = int(row["qa_index"])
            qa = dict(data[row_sample_idx]["qa"][qa_idx])
            qa["_qa_index"] = qa_idx
            qa["_sample_idx"] = row_sample_idx
            qa["_conversation_id"] = row.get("conversation_id", f"conv{row_sample_idx}")
            qa["_question_id"] = row.get("question_id", f"conv{row_sample_idx}_q{qa_idx}")
            selected.append(qa)
        split_id = rows[0].get("split_id", split_path.stem) if rows else split_path.stem
        return selected, split_id, split_path

    _, qa_list = load_locomo_sessions(data_path, sample_idx)
    selected = select_questions(qa_list, per_category, max_questions)
    for qa in selected:
        qa["_sample_idx"] = sample_idx
        qa["_conversation_id"] = f"conv{sample_idx}"
        qa["_question_id"] = f"conv{sample_idx}_q{qa['_qa_index']}"
    split_id = f"locomo_sample{sample_idx}_smoke_{len(selected)}_v1"
    return selected, split_id, None


def memory_source_for_question(
    sample_idx: int,
    memory_dir: str | None,
    memory_dir_template: str | None,
) -> Path:
    if memory_dir_template:
        return Path(memory_dir_template.format(sample=sample_idx, sample_idx=sample_idx))
    if memory_dir:
        return Path(memory_dir)
    raise ValueError("Either --memory-dir or --memory-dir-template is required")


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
If the reference answer says the information is not mentioned, NOT FOUND or an
equivalent abstention is correct.

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


def answer_from_evidence(
    client: OpenAI,
    model: str,
    question: str,
    evidence_context: str,
    max_answer_tokens: int,
    max_retries: int = 3,
) -> tuple[str, dict[str, int], float]:
    if not evidence_context.strip():
        return "NOT FOUND", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0.0

    prompt = ANSWER_PROMPT.format(question=question, evidence=evidence_context)
    last_error = ""
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_answer_tokens,
                temperature=0,
            )
            latency = time.time() - t0
            answer = strip_thinking(resp.choices[0].message.content or "") or "NOT FOUND"
            return answer, usage_dict(resp), latency
        except Exception as exc:
            last_error = str(exc)
            if attempt + 1 < max_retries:
                time.sleep(2)
    return f"ANSWER_ERROR: {last_error}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0.0


def retrieve_evidence(
    client: OpenAI,
    model: str,
    memory_dir: Path,
    question: str,
    enc: Any,
    max_rounds: int,
    max_selector_tokens: int,
    intermediate_observation_cap: int,
    tool_call_limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": RETRIEVAL_PROMPT},
        {"role": "user", "content": f"Search the memory folder and retrieve evidence for this question:\n\n{question}"},
    ]
    trace: dict[str, Any] = {
        "retrieval_model": model,
        "max_rounds": max_rounds,
        "tool_call_limit": tool_call_limit,
        "tool_calls": [],
        "files_opened": [],
        "first_content_file": None,
        "retrieval_usage": [],
        "selector_raw": "",
        "errors": [],
        "evidence_items": [],
    }

    retrieval_visible_tokens = 0
    retrieval_prompt_tokens = 0
    retrieval_completion_tokens = 0
    retrieval_billable_tokens = 0
    retrieval_start = time.time()
    tool_calls_seen = 0
    evidence_items: list[dict[str, Any]] = []

    for round_i in range(max_rounds):
        round_start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=READ_TOOLS,
                max_tokens=max_selector_tokens,
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
            trace["selector_raw"] = msg.content or "NOT FOUND"
            messages.append({"role": "assistant", "content": trace["selector_raw"]})
            break

        messages.append(assistant_message_to_dict(msg))
        for tc in msg.tool_calls:
            if tool_call_limit is not None and tool_calls_seen >= tool_call_limit:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "TOOL_CALL_LIMIT_REACHED",
                })
                trace["tool_calls"].append({
                    "round": round_i,
                    "tool_call_id": tc.id,
                    "function": tc.function.name,
                    "arguments": {},
                    "path": None,
                    "observation_chars": len("TOOL_CALL_LIMIT_REACHED"),
                    "observation_tokens": count_tokens("TOOL_CALL_LIMIT_REACHED", enc),
                    "raw_observation_tokens": 0,
                    "content_bearing": False,
                    "truncated": False,
                    "observation_preview": "TOOL_CALL_LIMIT_REACHED",
                })
                continue

            tool_calls_seen += 1
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {}

            raw_observation = execute_tool(fn_name, fn_args, str(memory_dir))
            raw_observation_tokens = count_tokens(raw_observation, enc)
            remaining_observation_budget = max(0, intermediate_observation_cap - retrieval_visible_tokens)
            observation = truncate_to_token_cap(raw_observation, remaining_observation_budget, enc)
            observation_tokens = count_tokens(observation, enc)
            retrieval_visible_tokens += observation_tokens

            path = fn_args.get("path") if isinstance(fn_args, dict) else None
            content_bearing = fn_name in {"cat", "grep_headings"}
            if fn_name == "cat" and path:
                trace["files_opened"].append(path)
                if trace["first_content_file"] is None:
                    trace["first_content_file"] = path
            elif fn_name == "grep_headings" and path:
                if path not in trace["files_opened"]:
                    trace["files_opened"].append(path)

            if content_bearing and observation.strip():
                evidence_item = {
                    "round": round_i,
                    "function": fn_name,
                    "path": path,
                    "content": observation,
                    "tokens": observation_tokens,
                    "raw_tokens": raw_observation_tokens,
                    "truncated": raw_observation_tokens > observation_tokens,
                }
                evidence_items.append(evidence_item)

            call_record = {
                "round": round_i,
                "tool_call_id": tc.id,
                "function": fn_name,
                "arguments": fn_args,
                "path": path,
                "observation_chars": len(observation),
                "observation_tokens": observation_tokens,
                "raw_observation_tokens": raw_observation_tokens,
                "content_bearing": content_bearing,
                "truncated": raw_observation_tokens > observation_tokens,
                "observation_preview": observation[:500],
            }
            trace["tool_calls"].append(call_record)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": observation})

    trace["retrieval_latency_s"] = round(time.time() - retrieval_start, 3)
    trace["retrieval_visible_tokens"] = retrieval_visible_tokens
    trace["retrieval_prompt_tokens"] = retrieval_prompt_tokens
    trace["retrieval_completion_tokens"] = retrieval_completion_tokens
    trace["retrieval_billable_tokens"] = retrieval_billable_tokens
    trace["retrieval_calls"] = len(trace["retrieval_usage"])
    trace["files_opened"] = list(dict.fromkeys(trace["files_opened"]))
    trace["evidence_items"] = [
        {k: v for k, v in item.items() if k != "content"} | {"content_preview": item["content"][:500]}
        for item in evidence_items
    ]
    return evidence_items, trace


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
    parser.add_argument("--memory-dir", default=None)
    parser.add_argument("--memory-dir-template", default=None)
    parser.add_argument("--artifacts-dir", default="../artifacts")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--retriever-model", default="qwen3.6-flash")
    parser.add_argument("--answer-model", default="qwen3.6-flash")
    parser.add_argument("--judge-model", default="qwen3.6-flash")
    parser.add_argument("--writer-model", default="existing_memory_folder")
    parser.add_argument("--provider-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--tokenizer", default="o200k_base")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-answer-tokens", type=int, default=1024)
    parser.add_argument("--max-selector-tokens", type=int, default=512)
    parser.add_argument("--final-evidence-cap", type=int, default=6000)
    parser.add_argument("--intermediate-observation-cap", type=int, default=20000)
    parser.add_argument("--total-visible-cap", type=int, default=None)
    parser.add_argument("--tool-call-limit", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--question-timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.memory_dir and not args.memory_dir_template:
        raise SystemExit("Either --memory-dir or --memory-dir-template is required")

    api_key = args.api_key or os.environ.get("EVAL_API_KEY") or memory_builder.API_KEY
    base_url = args.provider_base or os.environ.get("EVAL_API_BASE") or memory_builder.API_BASE
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.request_timeout)
    enc = get_encoder(args.tokenizer)

    selected, split_id, input_split_path = load_selected_questions(
        data_path=args.data,
        sample_idx=args.sample,
        per_category=args.per_category,
        max_questions=args.max_questions,
        split_file=args.split_file,
    )

    artifacts = Path(args.artifacts_dir)
    dataset_id = "D1"
    method_id = args.method_id
    build_id = args.build_id

    split_rows = []
    for qa in selected:
        sample_idx = int(qa["_sample_idx"])
        qidx = qa["_qa_index"]
        category = qa.get("category", 0)
        split_rows.append({
            "split_id": split_id,
            "dataset_id": dataset_id,
            "source_dataset": args.data,
            "conversation_id": qa["_conversation_id"],
            "sample_idx": sample_idx,
            "question_id": qa["_question_id"],
            "qa_index": qidx,
            "category": category,
            "category_name": CATEGORY_NAMES.get(category, f"cat{category}"),
            "question": qa["question"],
        })
    split_path = artifacts / "splits" / f"{split_id}.jsonl"
    if input_split_path is None:
        write_jsonl(split_path, split_rows)
    else:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        if input_split_path.resolve() != split_path.resolve():
            shutil.copyfile(input_split_path, split_path)

    memory_artifact_dir = artifacts / "memory_folders" / dataset_id / method_id / build_id
    if memory_artifact_dir.exists():
        shutil.rmtree(memory_artifact_dir)
    memory_artifact_dir.mkdir(parents=True, exist_ok=True)
    memory_artifact_by_sample: dict[int, Path] = {}
    for qa in selected:
        sample_idx = int(qa["_sample_idx"])
        if sample_idx in memory_artifact_by_sample:
            continue
        source_memory_dir = memory_source_for_question(
            sample_idx=sample_idx,
            memory_dir=args.memory_dir,
            memory_dir_template=args.memory_dir_template,
        )
        if not source_memory_dir.exists():
            raise SystemExit(f"memory folder does not exist: {source_memory_dir}")
        dest = memory_artifact_dir / f"sample{sample_idx}"
        shutil.copytree(source_memory_dir, dest)
        memory_artifact_by_sample[sample_idx] = dest

    config = {
        "run_id": args.run_id,
        "dataset_id": dataset_id,
        "split_id": split_id,
        "method_id": method_id,
        "method_name": args.method_name,
        "build_id": build_id,
        "source_memory_dir": args.memory_dir,
        "source_memory_dir_template": args.memory_dir_template,
        "memory_artifact_dir": str(memory_artifact_dir),
        "split_file": str(split_path),
        "retriever_model": args.retriever_model,
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "writer_model": args.writer_model,
        "tokenizer": args.tokenizer,
        "protocol": "two_stage_retrieval_then_answer",
        "max_rounds": args.max_rounds,
        "max_answer_tokens": args.max_answer_tokens,
        "max_selector_tokens": args.max_selector_tokens,
        "final_evidence_cap": args.final_evidence_cap,
        "intermediate_observation_cap": args.intermediate_observation_cap,
        "total_visible_cap": args.total_visible_cap,
        "tool_call_limit": args.tool_call_limit,
        "request_timeout": args.request_timeout,
        "retrieval_temperature": 0,
        "answer_temperature": 0,
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
    output_path = artifacts / "per_question_outputs" / dataset_id / method_id / f"{build_id}.jsonl"
    judge_path = artifacts / "judge_outputs" / args.judge_model / dataset_id / method_id / f"{build_id}.jsonl"
    table_path = artifacts / "tables" / f"{args.run_id}_summary.csv"
    if args.resume:
        outputs = read_jsonl(output_path)
        judge_outputs = read_jsonl(judge_path)
        summary_rows = [summary_from_record(record) for record in outputs]
    completed_question_ids = {record["question_id"] for record in outputs}

    for i, qa in enumerate(selected, start=1):
        sample_idx = int(qa["_sample_idx"])
        qidx = qa["_qa_index"]
        question_id = qa["_question_id"]
        conversation_id = qa["_conversation_id"]
        category = qa.get("category", 0)
        category_name = CATEGORY_NAMES.get(category, f"cat{category}")
        gold_answer = gold_answer_for_qa(qa)
        if question_id in completed_question_ids:
            print(f"[{i}/{len(selected)}] {question_id} {category_name}: SKIP existing")
            continue
        print(f"[{i}/{len(selected)}] {question_id} {category_name}: {qa['question'][:80]}")

        try:
            if args.question_timeout > 0:
                signal.signal(signal.SIGALRM, _raise_question_timeout)
                signal.alarm(args.question_timeout)

            evidence_items, trace = retrieve_evidence(
                client=client,
                model=args.retriever_model,
                memory_dir=memory_artifact_by_sample[sample_idx],
                question=qa["question"],
                enc=enc,
                max_rounds=args.max_rounds,
                max_selector_tokens=args.max_selector_tokens,
                intermediate_observation_cap=args.intermediate_observation_cap,
                tool_call_limit=args.tool_call_limit,
            )
            effective_evidence_cap = args.final_evidence_cap
            if args.total_visible_cap is not None:
                effective_evidence_cap = min(
                    effective_evidence_cap,
                    max(0, args.total_visible_cap - trace["retrieval_visible_tokens"]),
                )
            ranked_evidence_items = rank_evidence_items_by_selector(
                evidence_items,
                trace.get("selector_raw", ""),
            )
            evidence_context, final_evidence_tokens = build_evidence_context(
                ranked_evidence_items,
                effective_evidence_cap,
                enc,
            )
            answer, answer_usage, answer_latency = answer_from_evidence(
                client=client,
                model=args.answer_model,
                question=qa["question"],
                evidence_context=evidence_context,
                max_answer_tokens=args.max_answer_tokens,
            )
            score, judge_raw, judge_usage, judge_latency = judge_answer(
                client=client,
                model=args.judge_model,
                question=qa["question"],
                gold_answer=gold_answer,
                system_answer=answer,
            )
            failure_tag = None
            final_evidence_order = [
                {
                    "path": item.get("path"),
                    "function": item.get("function"),
                    "tokens": item.get("tokens"),
                    "raw_tokens": item.get("raw_tokens"),
                }
                for item in ranked_evidence_items
            ]
            final_evidence_sha256 = sha256_bytes(evidence_context.encode("utf-8"))
            final_evidence_preview = evidence_context[:1000]
        except QuestionTimeout as exc:
            trace = {
                "retrieval_model": args.retriever_model,
                "max_rounds": args.max_rounds,
                "tool_call_limit": args.tool_call_limit,
                "tool_calls": [],
                "files_opened": [],
                "first_content_file": None,
                "retrieval_usage": [],
                "selector_raw": "",
                "errors": [{"error": str(exc)}],
                "retrieval_latency_s": args.question_timeout,
                "retrieval_visible_tokens": 0,
                "retrieval_prompt_tokens": 0,
                "retrieval_completion_tokens": 0,
                "retrieval_billable_tokens": 0,
                "retrieval_calls": 0,
            }
            effective_evidence_cap = args.final_evidence_cap
            final_evidence_tokens = 0
            answer = "QUESTION_TIMEOUT"
            answer_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            answer_latency = 0.0
            score = 0
            judge_raw = "QUESTION_TIMEOUT"
            judge_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            judge_latency = 0.0
            failure_tag = "question_timeout"
            final_evidence_order = []
            final_evidence_sha256 = sha256_bytes(b"")
            final_evidence_preview = ""
        finally:
            if args.question_timeout > 0:
                signal.alarm(0)

        trace.update({
            "run_id": args.run_id,
            "dataset_id": dataset_id,
            "split_id": split_id,
            "question_id": question_id,
            "conversation_id": conversation_id,
            "sample_idx": sample_idx,
            "category_or_type": category_name,
            "question": qa["question"],
            "gold_answer": gold_answer,
            "system_answer": answer,
            "answer_model": args.answer_model,
            "answer_usage": answer_usage,
            "answer_latency_s": answer_latency,
            "final_evidence_tokens": final_evidence_tokens,
            "final_evidence_cap_effective": effective_evidence_cap,
            "final_evidence_sha256": final_evidence_sha256,
            "final_evidence_preview": final_evidence_preview,
            "final_evidence_order": final_evidence_order,
            "judge_model": args.judge_model,
            "judge_raw": judge_raw,
            "judge_usage": judge_usage,
            "judge_latency_s": judge_latency,
            "judge_score": score,
            "failure_tag": failure_tag,
            "memory_folder_sha256": memory_hash,
            "config_sha256": config["config_sha256"],
        })
        trace_path = trace_dir / f"{question_id}.json"
        trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

        total_visible_tokens = trace["retrieval_visible_tokens"] + final_evidence_tokens
        total_billable_tokens = (
            trace["retrieval_billable_tokens"]
            + answer_usage["total_tokens"]
            + judge_usage["total_tokens"]
        )
        record = {
            "run_id": args.run_id,
            "dataset_id": dataset_id,
            "split_id": split_id,
            "question_id": question_id,
            "conversation_id": conversation_id,
            "sample_idx": sample_idx,
            "category_or_type": category_name,
            "method_id": method_id,
            "method_name": args.method_name,
            "build_id": build_id,
            "writer_model": args.writer_model,
            "retriever_model": args.retriever_model,
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
            "tokenizer": args.tokenizer,
            "final_evidence_cap": args.final_evidence_cap,
            "intermediate_observation_cap": args.intermediate_observation_cap,
            "total_visible_cap": args.total_visible_cap,
            "tool_call_limit": args.tool_call_limit,
            "retrieval_calls": trace["retrieval_calls"],
            "tool_calls": trace["tool_calls"],
            "files_opened": trace["files_opened"],
            "first_content_file": trace["first_content_file"],
            "target_evidence_files": [],
            "first_path_hit": None,
            "retrieval_visible_tokens": trace["retrieval_visible_tokens"],
            "final_evidence_tokens": final_evidence_tokens,
            "answer_prompt_tokens": answer_usage["prompt_tokens"],
            "answer_completion_tokens": answer_usage["completion_tokens"],
            "judge_prompt_tokens": judge_usage["prompt_tokens"],
            "judge_completion_tokens": judge_usage["completion_tokens"],
            "total_visible_tokens": total_visible_tokens,
            "total_billable_tokens": total_billable_tokens,
            "storage_amortized_tokens": 0,
            "latency_retrieval_s": trace["retrieval_latency_s"],
            "latency_answer_s": answer_latency,
            "latency_judge_s": judge_latency,
            "gold_answer": gold_answer,
            "system_answer": answer,
            "judge_score": score,
            "f1": None,
            "bleu1": None,
            "failure_tag": failure_tag,
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
        summary_rows.append(summary_from_record(record))
        write_jsonl(output_path, outputs)
        write_jsonl(judge_path, judge_outputs)
        write_summary_csv(table_path, summary_rows)
        print(
            f"  score={score} calls={trace['retrieval_calls']} "
            f"visible={total_visible_tokens} billable={total_billable_tokens}"
        )

    write_jsonl(output_path, outputs)
    write_jsonl(judge_path, judge_outputs)

    write_summary_csv(table_path, summary_rows)

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
