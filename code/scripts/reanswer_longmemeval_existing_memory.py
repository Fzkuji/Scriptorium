#!/usr/bin/env python3
"""Re-answer frozen LongMemEval items from existing NativeMem libraries."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_v88_gpt55_longmemeval as lme  # noqa: E402
from scripts.run_v11_gpt55_frontier_longmemeval import V11_ANSWER_PROMPT  # noqa: E402
from src.v11_memory import _chat_completion_with_retry  # noqa: E402

lme.LME_SINGLE_PROMPT = V11_ANSWER_PROMPT


V11_TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": (
            "Run read-only shell commands in the memory workspace. Supported "
            "commands include find, rg, grep, cat, sed -n, ls, head, tail, wc, "
            "sort, uniq, and cut."
        ),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
        }, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "list_memory_files",
        "description": "List Markdown files in the memory workspace.",
        "parameters": {"type": "object", "properties": {
            "prefix": {"type": "string"},
        }},
    }},
    {"type": "function", "function": {
        "name": "read_memory_file",
        "description": "Read one Markdown file from the memory workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "search_memory",
        "description": "Literal case-insensitive search across Markdown memory files.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "resolve_sources",
        "description": "Read original conversation turns for complete [Dn:m] source anchors.",
        "parameters": {"type": "object", "properties": {
            "source_ids": {"type": "array", "items": {"type": "string"}},
        }, "required": ["source_ids"]},
    }},
]

V11_RETRIEVAL_PROMPT = """Answer one memory-benchmark question from a read-only V11 memory workspace.
Condition: {condition}
The workspace has no fixed directory taxonomy. Use the general read-only bash
tool or the specialized memory tools and the
actual inventory below. Search wording may differ from the question, so inspect
semantically relevant files and use several literal queries when needed.

For temporal, update, counting, comparison, and multi-session questions, inspect
all relevant events. Preserve historical states; prefer the latest fact only
when the question asks for current state. Use resolve_sources on relevant
[Dn:m] anchors when exact wording or nearby turns matter.

You must read non-empty memory evidence before answering. Do not answer from the
inventory, file names, prior knowledge, or assumptions. If the recorded history
does not contain the requested fact, state that directly.

Inventory:
{inventory}

Current Date: {question_date}
Question: {question}

After tool use, output exactly one <answer>...</answer> block and no reasoning.
"""

V11_CONDITION_VIEWS = {
    "dual_source": ("topics", "timeline", "recent"),
    "topic_source": ("topics", "recent"),
    "timeline_source": ("timeline", "recent"),
    "dual_no_source": ("topics", "timeline", "recent"),
}


def stop_on_signal(_signum: object, _frame: object) -> None:
    raise KeyboardInterrupt


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def configure(
    gateway_base_url: str,
    workers: int,
    *,
    api_format: str = "openai",
    model: str = "openai/gpt-4o-mini",
    api_key: str = "local-openrouter-gateway",
    max_tokens: int = 1200,
    map_mode: str = "dir",
    map_inline: int = 8,
    read_context: int = 8,
) -> Any:
    config = dict(lme.METHOD_ENV)
    config["NATIVEMEM_V8_MAP"] = map_mode
    config["NATIVEMEM_V8_MAP_INLINE"] = str(map_inline)
    config["NATIVEMEM_V8_READ_CONTEXT"] = str(read_context)
    config["NATIVEMEM_V8_CONCURRENCY"] = str(workers)
    config["NATIVEMEM_V8_MAX_TOKENS"] = str(max_tokens)
    os.environ.update(config)
    os.environ.update({
        "MODEL": model,
        "BUILDER_MODEL": model,
        "BUILDER_BASE": gateway_base_url,
        "BUILDER_KEY": api_key,
        "ALIYUN_KEY": api_key,
        "NATIVEMEM_TRUST_PROXY": "0",
        "NO_PROXY": "localhost,127.0.0.1",
        "no_proxy": "localhost,127.0.0.1",
    })
    os.environ.pop("NATIVEMEM_V9_PIPELINE", None)
    os.environ.pop("NATIVEMEM_V9_SCRIBE_MODE", None)
    backend = lme.load_backend()
    if api_format == "anthropic":
        from src.anthropic_openai_compat import AnthropicOpenAICompat

        anthropic_client = AnthropicOpenAICompat(api_key, gateway_base_url)
        backend.client = anthropic_client
        backend.nativemem_runtime.client = anthropic_client
        backend.v8_memory.client = anthropic_client
    lme.verify_backend_contract(backend)
    return backend


def source_records(analysis_path: Path) -> list[dict[str, Any]]:
    value = read_json(analysis_path)
    records = value.get("records") if isinstance(value, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("analysis file has no records")
    indices = [int(record["dataset_index"]) for record in records]
    if len(indices) != len(set(indices)):
        raise ValueError("analysis file has duplicate dataset indices")
    return sorted(records, key=lambda record: int(record["dataset_index"]))


def load_completed_results(
    output_dir: Path,
    dataset: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    expected = {int(source["dataset_index"]) for source in sources}
    completed: dict[int, dict[str, Any]] = {}
    for path in sorted((output_dir / "items").glob("*.json")):
        record = read_json(path)
        if not isinstance(record, dict):
            raise ValueError(f"invalid item record: {path}")
        try:
            index = int(record["dataset_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid item index: {path}") from exc
        if index not in expected or index >= len(dataset):
            raise ValueError(f"item {index} is not part of this run")
        item = dataset[index]
        if (
            str(record.get("question_id")) != str(item["question_id"])
            or record.get("question") != item["question"]
            or not str(record.get("answer", "")).strip()
        ):
            raise ValueError(f"item {index} identity or answer is invalid")
        if index in completed:
            raise ValueError(f"duplicate item {index}")
        completed[index] = record
    return completed


def write_results(output_dir: Path, completed: dict[int, dict[str, Any]]) -> None:
    atomic_json(
        output_dir / "results.json",
        [completed[index] for index in sorted(completed)],
    )


def run_pending(
    sources: list[dict[str, Any]],
    *,
    workers: int,
    work: Callable[[dict[str, Any]], dict[str, Any]],
    on_complete: Callable[[dict[str, Any]], None],
) -> tuple[dict[int, dict[str, Any]], bool]:
    """Run a bounded number of items and retain completions on interruption."""
    source_iter = iter(sources)
    completed: dict[int, dict[str, Any]] = {}
    pool = ThreadPoolExecutor(max_workers=min(workers, len(sources) or 1))
    active = {}
    interrupted = False

    def submit_one() -> bool:
        try:
            source = next(source_iter)
        except StopIteration:
            return False
        active[pool.submit(work, source)] = int(source["dataset_index"])
        return True

    try:
        for _ in range(min(workers, len(sources))):
            submit_one()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                index = active.pop(future)
                try:
                    record = future.result()
                except Exception as exc:  # one bad QA must not stop the batch
                    print(
                        f"failed item={index} error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    submit_one()
                    continue
                completed[index] = record
                on_complete(record)
                submit_one()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return completed, interrupted


def install_trace_hooks(backend: Any) -> threading.local:
    state = threading.local()
    original_execute = backend.execute_tool
    original_read = backend.read_turns

    def execute(tool_name: str, args: object, base_dir: str, hide_raw: bool = False) -> str:
        output = original_execute(tool_name, args, base_dir, hide_raw=hide_raw)
        trace = getattr(state, "trace", None)
        if trace is not None:
            trace.append({
                "type": tool_name,
                "args": args,
                "accepted": not str(output).startswith("Rejected"),
            })
        return output

    def read(turn_index: dict[str, Any], dia_ids: object, context: int = 1) -> str:
        output = original_read(turn_index, dia_ids, context=context)
        trace = getattr(state, "trace", None)
        if trace is not None:
            trace.append({
                "type": "read_original",
                "dia_ids": dia_ids,
                "context": context,
            })
        return output

    backend.execute_tool = execute
    backend.read_turns = read
    return state


def _v11_files(
    memory_dir: Path,
    condition: str = "native",
    *,
    include_recent: bool = True,
) -> list[Path]:
    root = memory_dir.resolve()
    views = V11_CONDITION_VIEWS.get(condition)
    if condition != "native" and views is None:
        raise ValueError(f"unknown V11 condition: {condition}")
    files = [
        path for path in root.rglob("*.md")
        if path.is_file()
        and not path.is_symlink()
        and "sources" not in path.relative_to(root).parts
        and (
            views is None
            or path.relative_to(root).parts[0] in views
        )
    ]
    recent = root / "recent_events.jsonl"
    if (
        recent.is_file()
        and not recent.is_symlink()
        and include_recent
        and (views is None or "recent" in views)
    ):
        files.append(recent)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _v11_read(
    memory_dir: Path,
    raw_path: object,
    condition: str = "native",
    *,
    include_recent: bool = True,
) -> str:
    root = memory_dir.resolve()
    relative = Path(str(raw_path or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("memory path escapes workspace")
    path = (root / relative).resolve()
    path.relative_to(root)
    allowed = path in _v11_files(root, condition, include_recent=include_recent)
    if path.is_symlink() or not path.is_file() or not allowed:
        raise ValueError("memory path is not an allowed memory file")
    return path.read_text(encoding="utf-8")


def _v11_tools(condition: str) -> list[dict[str, Any]]:
    if condition == "native":
        return V11_TOOLS
    if condition not in V11_CONDITION_VIEWS:
        raise ValueError(f"unknown V11 condition: {condition}")
    return [
        tool for tool in V11_TOOLS
        if tool["function"]["name"] != "bash"
        and not (
            condition == "dual_no_source"
            and tool["function"]["name"] == "resolve_sources"
        )
    ]


def collect_answer_v11(
    backend: Any,
    item: dict[str, Any],
    memory_dir: Path,
    turn_index: dict[str, Any],
    condition: str = "native",
    *,
    include_recent: bool = True,
) -> tuple[list[dict[str, str]], int, str, list[dict[str, Any]]]:
    files = _v11_files(memory_dir, condition, include_recent=include_recent)
    tools = _v11_tools(condition)
    inventory = "\n".join(path.relative_to(memory_dir).as_posix() for path in files)
    messages: list[Any] = [{"role": "user", "content": V11_RETRIEVAL_PROMPT.format(
        condition=condition,
        inventory=inventory or "(no Markdown files)",
        question_date=item.get("question_date", ""),
        question=item["question"],
    )}]
    trace: list[dict[str, Any]] = []
    evidence: list[dict[str, str]] = []
    steps = 0
    max_rounds = int(os.environ.get("NATIVEMEM_V8_MAX_ROUNDS", "20"))
    max_tokens = int(os.environ.get("NATIVEMEM_V8_MAX_TOKENS", "1200"))
    context = int(os.environ.get("NATIVEMEM_V8_READ_CONTEXT", "8"))

    for _ in range(max_rounds):
        response = _chat_completion_with_retry(
            backend.client.chat.completions.create,
            model=backend.ALIYUN_MODEL,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        backend.log_usage(response, phase="v11_memory_qa")
        steps += 1
        message = response.choices[0].message
        calls = getattr(message, "tool_calls", None) or []
        content = message.content or ""
        if not calls:
            match = lme._ANSWER_RE.search(content)
            if match and evidence:
                return evidence, steps, match.group(1).strip(), trace
            trace.append({"type": "rejected_no_evidence"})
            messages.extend([
                {"role": "assistant", "content": content},
                {"role": "user", "content": (
                    "No final answer is accepted yet. Read non-empty memory evidence "
                    "with the tools, then answer the original question."
                )},
            ])
            continue

        messages.append(message)
        for call in calls:
            name = call.function.name
            args: dict[str, Any] = {}
            accepted = False
            try:
                args = json.loads(call.function.arguments or "{}")
                if name == "bash":
                    accepted, reason = lme.validate_read_only_command(
                        str(args.get("command", ""))
                    )
                    if not accepted:
                        output = f"Command rejected: {reason}"
                    else:
                        output = backend.execute_tool(
                            "bash", args, str(memory_dir), hide_raw=True
                        )
                elif name == "list_memory_files":
                    prefix = str(args.get("prefix", ""))
                    output = "\n".join(
                        path.relative_to(memory_dir).as_posix()
                        for path in files
                        if path.relative_to(memory_dir).as_posix().startswith(prefix)
                    )
                elif name == "read_memory_file":
                    output = _v11_read(
                        memory_dir,
                        args.get("path"),
                        condition,
                        include_recent=include_recent,
                    )
                elif name == "search_memory":
                    query = str(args.get("query", "")).strip()
                    if not query:
                        raise ValueError("search query is empty")
                    limit = max(1, min(int(args.get("max_results", 20)), 100))
                    matches = []
                    for path in files:
                        relative = path.relative_to(memory_dir).as_posix()
                        for line_no, line in enumerate(
                            path.read_text(encoding="utf-8").splitlines(), start=1
                        ):
                            if query.casefold() in line.casefold():
                                matches.append(f"{relative}:{line_no}:{line}")
                                if len(matches) == limit:
                                    break
                        if len(matches) == limit:
                            break
                    output = "\n".join(matches)
                elif name == "resolve_sources":
                    if condition == "dual_no_source":
                        raise ValueError("source resolution is disabled")
                    source_ids = args.get("source_ids", [])
                    if isinstance(source_ids, str):
                        source_ids = [source_ids]
                    output = backend.read_turns(turn_index, source_ids, context=context)
                else:
                    raise ValueError(f"unknown tool: {name}")
            except Exception as exc:  # noqa: BLE001
                output = f"Tool error: {exc}"
            entry = {"type": name, "args": args, "nonempty": bool(output.strip())}
            if name == "bash":
                entry["accepted"] = accepted
            trace.append(entry)
            if name != "list_memory_files" and output.strip() and not output.startswith("Tool error:"):
                evidence.append({"text": output, "date": ""})
            messages.append({
                "role": "tool", "tool_call_id": call.id, "content": output[:100_000]
            })
    raise RuntimeError("V11 retrieval agent did not produce an evidence-grounded answer")


def collect_answer(
    backend: Any,
    item: dict[str, Any],
    memory_dir: Path,
    turn_index: dict[str, Any],
    prompt: str,
    trace_state: threading.local | None = None,
    v11_condition: str = "native",
) -> tuple[list[dict[str, str]], int, str, list[dict[str, Any]]]:
    if prompt == "v11-agent":
        return collect_answer_v11(
            backend, item, memory_dir, turn_index, condition=v11_condition
        )
    if prompt == "longmemeval":
        for attempt in range(2):
            try:
                return lme.collect_and_answer_longmemeval(
                    backend, item, Path(memory_dir), turn_index
                )
            except lme.EmptyStageError:
                if attempt == 1:
                    raise
    if prompt != "adapter-v8":
        raise ValueError(f"unknown retrieval prompt: {prompt}")

    if trace_state is not None:
        trace_state.trace = []
    memories, steps, answer = backend._collect_and_answer_v8(
        lme.model_question(item), str(memory_dir), turn_index
    )
    trace = list(getattr(trace_state, "trace", []))
    if trace_state is not None:
        trace_state.trace = None
    return memories, steps, answer, trace


def answer_one(
    backend: Any,
    trace_state: threading.local,
    dataset: list[dict[str, Any]],
    source: dict[str, Any],
    output_dir: Path,
    prompt: str,
    v11_condition: str = "native",
) -> dict[str, Any]:
    index = int(source["dataset_index"])
    item = dataset[index]
    checkpoint_path = (ROOT / str(source["checkpoint"])).resolve()
    checkpoint = read_json(checkpoint_path)
    if (
        (
            checkpoint.get("status") != "complete"
            and checkpoint.get("build", {}).get("status") != "complete"
        )
        or int(checkpoint.get("dataset_index", -1)) != index
        or str(checkpoint.get("question_id")) != str(item["question_id"])
    ):
        raise ValueError(f"invalid source checkpoint for item {index}")
    memory_dir = Path(str(checkpoint["paths"]["memory_dir"])).resolve()
    if not lme.memory_is_valid(memory_dir):
        raise ValueError(f"invalid source memory for item {index}")

    conv = lme.longmemeval_to_locomo(item, index)
    turn_index = backend.build_turn_index(conv)
    memories, steps, answer, trace = collect_answer(
        backend, item, memory_dir, turn_index, prompt, trace_state,
        v11_condition=v11_condition,
    )
    answer = str(answer).strip()
    if not answer or steps <= 0:
        raise RuntimeError(f"empty answer for item {index}")
    record = {
        "dataset_index": index,
        "question_id": item["question_id"],
        "question": item["question"],
        "question_type": item["question_type"],
        "abstention": str(item.get("question_type", "")).startswith("abstention"),
        "gold": item["answer"],
        "answer": answer,
        "old_answer": source["answer"],
        "retrieval_steps": steps,
        "memories_count": len(memories),
        "memories": memories,
        "tool_trace": trace,
        "source_checkpoint": str(checkpoint_path),
        "model": os.environ["MODEL"],
        "v11_condition": v11_condition,
        "variant": (
            f"existing_memory_{prompt}_map_{os.environ['NATIVEMEM_V8_MAP']}"
            f"_inline_{os.environ['NATIVEMEM_V8_MAP_INLINE']}"
            f"_read_context_{os.environ['NATIVEMEM_V8_READ_CONTEXT']}"
        ),
    }
    atomic_json(output_dir / "items" / f"{index:04d}.json", record)
    return record


def main() -> int:
    signal.signal(signal.SIGTERM, stop_on_signal)
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=lme.DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gateway-base-url", required=True)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument(
        "--api-format", choices=("openai", "anthropic"), default="openai"
    )
    parser.add_argument("--api-key-env")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--prompt", choices=("adapter-v8", "longmemeval", "v11-agent"),
        default="longmemeval",
    )
    parser.add_argument(
        "--v11-condition",
        choices=("native", *V11_CONDITION_VIEWS),
        default="native",
    )
    parser.add_argument("--map-mode", choices=("dir", "files"), default="dir")
    parser.add_argument("--map-inline", type=int, default=8)
    parser.add_argument("--read-context", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.map_inline < 1:
        parser.error("--map-inline must be positive")
    if args.read_context < 0:
        parser.error("--read-context must be non-negative")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    api_key = "local-openrouter-gateway"
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            parser.error(f"environment variable {args.api_key_env} is not set")

    dataset = lme.load_dataset(args.data.resolve(), lme.EXPECTED_LONGMEMEVAL_SIZE)
    lme.validate_longmemeval_s(dataset)
    sources = source_records(args.analysis.resolve())
    output_dir = args.output_dir.resolve()
    backend = configure(
        args.gateway_base_url,
        args.workers,
        api_format=args.api_format,
        model=args.model,
        api_key=api_key,
        max_tokens=args.max_tokens,
        map_mode=args.map_mode,
        map_inline=args.map_inline,
        read_context=args.read_context,
    )
    trace_state = install_trace_hooks(backend)

    completed = load_completed_results(output_dir, dataset, sources)
    write_results(output_dir, completed)
    pending = [source for source in sources if int(source["dataset_index"]) not in completed]
    print(f"resumed={len(completed)} pending={len(pending)} workers={args.workers}", flush=True)

    def work(source: dict[str, Any]) -> dict[str, Any]:
        return answer_one(
            backend, trace_state, dataset, source, output_dir, args.prompt,
            args.v11_condition,
        )

    def save_completion(record: dict[str, Any]) -> None:
        index = int(record["dataset_index"])
        completed[index] = record
        write_results(output_dir, completed)
        print(
            f"complete={len(completed)}/{len(sources)} item={index} "
            f"answer={record['answer']!r}",
            flush=True,
        )

    _, interrupted = run_pending(
        pending,
        workers=args.workers,
        work=work,
        on_complete=save_completion,
    )
    if interrupted:
        print(f"interrupted; saved={len(completed)}/{len(sources)}", flush=True)
        return 130

    if len(completed) != len(sources):
        raise RuntimeError(f"only completed {len(completed)}/{len(sources)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
