"""Concurrent retrieval and answering over frozen LongMemEval memories."""

import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

from scripts import run_v88_gpt55_longmemeval as lme
from scripts.v11_common import atomic_json, read_json
from src.nativemem_versions.v11 import retrieval


CODE_ROOT = Path(__file__).resolve().parents[2]
lme.LME_SINGLE_PROMPT = retrieval.ANSWER_PROMPT


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

    def execute(
        tool_name: str, args: object, base_dir: str, hide_raw: bool = False
    ) -> str:
        output = original_execute(tool_name, args, base_dir, hide_raw=hide_raw)
        if (trace := getattr(state, "trace", None)) is not None:
            trace.append({
                "type": tool_name,
                "args": args,
                "accepted": not str(output).startswith("Rejected"),
            })
        return output

    def read(
        turn_index: dict[str, Any], dia_ids: object, context: int = 1
    ) -> str:
        output = original_read(turn_index, dia_ids, context=context)
        if (trace := getattr(state, "trace", None)) is not None:
            trace.append({
                "type": "read_original",
                "dia_ids": dia_ids,
                "context": context,
            })
        return output

    backend.execute_tool = execute
    backend.read_turns = read
    return state


def collect_answer(
    backend: Any,
    item: dict[str, Any],
    memory_dir: Path,
    turn_index: dict[str, Any],
    prompt: str,
    trace_state: threading.local | None = None,
    v11_condition: str = "native",
    query_config: retrieval.QueryConfig | None = None,
) -> tuple[list[dict[str, str]], int, str, list[dict[str, Any]]]:
    if prompt == "v11-agent":
        return retrieval.collect_answer(
            backend,
            item,
            memory_dir,
            turn_index,
            condition=v11_condition,
            config=query_config,
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
    *,
    model: str,
    query_config: retrieval.QueryConfig,
) -> dict[str, Any]:
    index = int(source["dataset_index"])
    item = dataset[index]
    checkpoint_path = (CODE_ROOT / str(source["checkpoint"])).resolve()
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
    conversation = lme.longmemeval_to_locomo(item, index)
    turn_index = backend.build_turn_index(conversation)
    memories, steps, answer, trace = collect_answer(
        backend,
        item,
        memory_dir,
        turn_index,
        prompt,
        trace_state,
        v11_condition=v11_condition,
        query_config=query_config,
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
        "model": model,
        "v11_condition": v11_condition,
        "variant": f"existing_memory_{prompt}",
    }
    atomic_json(output_dir / "items" / f"{index:04d}.json", record)
    return record
