"""Answer one benchmark question from a frozen Scriptorium workspace."""

import time
from pathlib import Path
from typing import Any

from src import retrieval

# Fields a benchmark may attach to a question that the judge needs later.
# An allowlist, not a copy of everything: the question's own `answer` is the
# gold and must never become the record's `answer`.
CARRIED_FIELDS = ("beam_category", "rubric", "abstention", "difficulty")


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
        **{
            field: question[field]
            for field in CARRIED_FIELDS
            if field in question
        },
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
