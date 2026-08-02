"""Write-time retrieval verification and repair."""

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .config import MemoryConfig
from .agent import _run_agent, render_conversation
from .provider import _json_response
from .prompts import (
    TOOLS,
    VERIFICATION_PROBE_TASK,
    VERIFICATION_REPAIR_TASK,
    VERIFICATION_RETRIEVAL_TASK,
)


def _verification_retrieve(
    memory_dir: str | Path,
    *,
    client: Any,
    model: str,
    question: str,
    usage_logger: Any | None,
    config: MemoryConfig,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="nativemem-verify-read-") as temporary:
        copied = Path(temporary) / "memory"
        shutil.copytree(Path(memory_dir), copied)
        final: list[str] = []
        audit = _run_agent(
            copied,
            client=client,
            model=model,
            task=VERIFICATION_RETRIEVAL_TASK.format(question=question),
            usage_logger=usage_logger,
            final_output=final,
            tools=TOOLS[:1],
            config=config,
        )
    text = final[-1] if final else ""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("verification retrieval did not return an answer block")
    return {"question": question, "answer": match.group(1).strip(), "trace": audit}


def _verification_answer_supported(
    *,
    client: Any,
    model: str,
    question: str,
    expected_answer: str,
    retrieved_answer: str,
    usage_logger: Any | None,
    config: MemoryConfig,
) -> bool:
    result = _json_response(
        client=client,
        model=model,
        usage_logger=usage_logger,
        config=config,
        messages=[{
            "role": "user",
            "content": (
                "Decide whether the retrieved answer correctly answers the question "
                "according to the expected source-grounded answer. Output only "
                '{"supported":true} or {"supported":false}.\n\n'
                f"Question: {question}\n"
                f"Expected answer: {expected_answer}\n"
                f"Retrieved answer: {retrieved_answer}"
            ),
        }],
    )
    if not isinstance(result.get("supported"), bool):
        raise ValueError("verification answer check must return supported boolean")
    return result["supported"]


def verify_session(
    memory_dir: str | Path,
    *,
    client: Any,
    model: str,
    observation_date: str,
    turns: list[tuple[str, str]],
    refs: list[str],
    usage_logger: Any | None = None,
    config: MemoryConfig | None = None,
) -> dict[str, Any]:
    config = config or MemoryConfig()
    probe = _json_response(
        client=client,
        model=model,
        usage_logger=usage_logger,
        config=config,
        messages=[{
            "role": "user",
            "content": VERIFICATION_PROBE_TASK.format(
                observation_date=observation_date,
                conversation=render_conversation(turns, refs),
            ),
        }],
    )
    question = str(probe.get("question", "")).strip()
    expected_answer = str(probe.get("expected_answer", "")).strip()
    probe_refs = [str(ref) for ref in probe.get("refs", [])]
    if (
        not question
        or not expected_answer
        or not probe_refs
        or any(ref not in refs for ref in probe_refs)
    ):
        raise ValueError("verification probe is incomplete or cites another session")
    probe = {
        "question": question,
        "expected_answer": expected_answer,
        "refs": probe_refs,
    }

    initial = _verification_retrieve(
        memory_dir,
        client=client,
        model=model,
        question=question,
        usage_logger=usage_logger,
        config=config,
    )
    initial["supported"] = _verification_answer_supported(
        client=client,
        model=model,
        question=question,
        expected_answer=expected_answer,
        retrieved_answer=initial["answer"],
        usage_logger=usage_logger,
        config=config,
    )
    if initial["supported"]:
        return {
            "probe": probe,
            "initial": initial,
            "repaired": False,
            "repair_trace": [],
            "post_repair": None,
        }

    repair_trace = _run_agent(
        memory_dir,
        client=client,
        model=model,
        task=VERIFICATION_REPAIR_TASK.format(
            question=question,
            expected_answer=expected_answer,
            refs=", ".join(probe_refs),
            retrieved_answer=initial["answer"],
            trace=json.dumps(initial["trace"], ensure_ascii=False, indent=2),
        ),
        usage_logger=usage_logger,
        tools=TOOLS[:1],
        config=config,
    )
    post_repair = _verification_retrieve(
        memory_dir,
        client=client,
        model=model,
        question=question,
        usage_logger=usage_logger,
        config=config,
    )
    post_repair["supported"] = _verification_answer_supported(
        client=client,
        model=model,
        question=question,
        expected_answer=expected_answer,
        retrieved_answer=post_repair["answer"],
        usage_logger=usage_logger,
        config=config,
    )
    return {
        "probe": probe,
        "initial": initial,
        "repaired": True,
        "repair_trace": repair_trace,
        "post_repair": post_repair,
    }
