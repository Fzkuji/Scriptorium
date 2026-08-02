"""Answer one LoCoMo question from a frozen NativeMem workspace."""

import time
from pathlib import Path
from typing import Any

from src import retrieval


def answer_question(
    backend: Any,
    query_config: retrieval.QueryConfig,
    memory_dir: Path,
    turn_index: dict[str, Any],
    sample_index: int,
    sample_id: str,
    question_index: int,
    question: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    memories, steps, answer, trace = retrieval.collect_answer(
        backend,
        {"question": question["question"]},
        memory_dir,
        turn_index,
        config=query_config,
    )
    return {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "question_index": question_index,
        "question_id": f"s{sample_index}_q{question_index}",
        "question": question["question"],
        "gold": str(
            question.get("answer", question.get("adversarial_answer", ""))
        ),
        "category": int(question["category"]),
        "memories": memories,
        "answer": str(answer).strip(),
        "retrieval": {
            "latency_s": round(time.monotonic() - started, 3),
            "steps": steps,
            "tool_calls": int(trace[-1].get("tool_calls", 0)) if trace else 0,
            "visible_tokens": int(
                trace[-1].get("memory_visible_tokens", 0)
            ) if trace else 0,
        },
        "tool_trace": trace,
    }
