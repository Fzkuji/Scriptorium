"""Concurrent retrieval and answering over frozen LongMemEval memories."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

from scripts.runners.common import atomic_json, read_json
from memory import retrieval

from . import support as lme


CODE_ROOT = Path(__file__).resolve().parents[3]
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


def collect_answer(
    backend: Any,
    item: dict[str, Any],
    memory_dir: Path,
    turn_index: dict[str, Any],
    condition: str = "native",
    query_config: retrieval.QueryConfig | None = None,
) -> tuple[list[dict[str, str]], int, str, list[dict[str, Any]]]:
    return retrieval.collect_answer(
        backend,
        item,
        memory_dir,
        turn_index,
        condition=condition,
        config=query_config,
    )


def answer_one(
    backend: Any,
    dataset: list[dict[str, Any]],
    source: dict[str, Any],
    output_dir: Path,
    condition: str = "native",
    *,
    model: str,
    query_config: retrieval.QueryConfig,
) -> dict[str, Any]:
    index = int(source["dataset_index"])
    item = dataset[index]
    checkpoint_raw = Path(str(source["checkpoint"]))
    checkpoint_path = (CODE_ROOT / checkpoint_raw).resolve()
    if not checkpoint_path.is_file():
        checkpoint_path = (
            Path("/Users/fzkuji/Scriptorium/code") / checkpoint_raw
        ).resolve()
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
        memory_dir = checkpoint_path.parent / "memory"
    if not lme.memory_is_valid(memory_dir):
        raise ValueError(f"invalid source memory for item {index}")
    conversation = lme.to_conversation(item, index)
    turn_index = backend.build_turn_index(conversation)
    memories, steps, answer, trace = collect_answer(
        backend,
        item,
        memory_dir,
        turn_index,
        condition=condition,
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
        "condition": condition,
        "variant": "existing_memory_nativemem",
    }
    atomic_json(output_dir / "items" / f"{index:04d}.json", record)
    return record
