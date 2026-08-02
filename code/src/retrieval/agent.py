"""Agent-controlled NativeMem retrieval loop."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..management import _chat_completion_with_retry
from .config import QueryConfig
from .context import evidence_keys, initialize_context
from .prompts import FINAL_PROMPT
from .tools import execute_tool_call
from .views import memory_files, tools_for


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
def collect_answer(
    runtime: Any,
    item: dict[str, Any],
    memory_dir: Path,
    turn_index: dict[str, Any],
    condition: str = "native",
    *,
    include_recent: bool = True,
    config: QueryConfig | None = None,
) -> tuple[list[dict[str, str]], int, str, list[dict[str, Any]]]:
    config = config or getattr(runtime, "query_config", None) or QueryConfig()
    memory_dir = Path(memory_dir).resolve()
    files = memory_files(memory_dir, condition, include_recent=include_recent)
    tools = tools_for(condition)
    verify_sources = config.verify_sources
    messages, trace, evidence, seen_evidence, budget = initialize_context(
        memory_dir=memory_dir,
        files=files,
        condition=condition,
        item=item,
        verify_sources=verify_sources,
        model=runtime.model,
        config=config,
    )

    steps = 0
    tool_calls_used = 0
    no_new_evidence = 0
    indexes: dict[str, Any] = {}
    termination_reason = (
        "visible_token_limit"
        if budget.used >= config.visible_token_limit
        else None
    )

    if termination_reason is None:
        for _ in range(config.max_rounds):
            response = _chat_completion_with_retry(
                runtime.client.chat.completions.create,
                model=runtime.model,
                messages=messages,
                tools=tools,
                max_tokens=config.max_output_tokens,
                temperature=0.0,
            )
            runtime.log_usage(response, phase="memory_qa")
            steps += 1
            message = response.choices[0].message
            calls = getattr(message, "tool_calls", None) or []
            content = message.content or ""
            if not calls:
                match = _ANSWER_RE.search(content)
                if match and evidence:
                    trace.append({
                        "type": "termination",
                        "termination_reason": "evidence_sufficient",
                        "retrieval_rounds": steps,
                        "tool_calls": tool_calls_used,
                        "memory_visible_tokens": budget.used,
                        "source_verification": verify_sources,
                    })
                    return evidence, steps, match.group(1).strip(), trace
                trace.append({"type": "rejected_no_evidence"})
                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": (
                        "No final answer is accepted yet. Read non-empty memory "
                        "evidence with the tools, then answer the original question."
                    )},
                ])
                continue

            messages.append(message)
            for call in calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except (TypeError, ValueError):
                    args = {}
                if termination_reason is not None or tool_calls_used >= config.max_tool_calls:
                    termination_reason = termination_reason or "tool_call_limit"
                    trace.append({
                        "type": name, "args": args,
                        "executed": False, "nonempty": False,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": f"Retrieval stopped: {termination_reason}.",
                    })
                    continue

                tool_calls_used += 1
                try:
                    output, executed, accepted = execute_tool_call(
                        runtime,
                        name,
                        args,
                        memory_dir=memory_dir,
                        files=files,
                        condition=condition,
                        include_recent=include_recent,
                        indexes=indexes,
                    )
                except Exception as exc:  # noqa: BLE001
                    output, executed, accepted = f"Tool error: {exc}", False, None

                rejected = output.startswith("Command rejected:")
                excluded = (
                    output.startswith("Tool error:")
                    or rejected
                    or output in {"No BM25 matches.", "No embedding matches."}
                )
                memory_output = output if output.strip() and not excluded else ""
                if memory_output:
                    delivered, raw_visible, delivered_visible = budget.consume(
                        memory_output
                    )
                else:
                    delivered, raw_visible, delivered_visible = output, 0, 0
                entry = {
                    "type": name,
                    "args": args,
                    "executed": executed,
                    "nonempty": False if rejected else bool(delivered.strip()),
                    "raw_visible_tokens": raw_visible,
                    "delivered_visible_tokens": delivered_visible,
                    "cumulative_visible_tokens": budget.used,
                }
                if accepted is not None:
                    entry["accepted"] = accepted
                trace.append(entry)

                new_keys = set()
                if name != "list_memory_files" and memory_output and delivered.strip():
                    evidence.append({"text": delivered, "date": ""})
                    new_keys = evidence_keys(delivered) - seen_evidence
                    seen_evidence.update(new_keys)
                if not rejected:
                    no_new_evidence = 0 if new_keys else no_new_evidence + 1
                messages.append({
                    "role": "tool", "tool_call_id": call.id, "content": delivered,
                })

                if budget.used >= config.visible_token_limit:
                    termination_reason = "visible_token_limit"
                elif no_new_evidence >= 2:
                    termination_reason = "no_new_evidence"
                elif tool_calls_used >= config.max_tool_calls:
                    termination_reason = "tool_call_limit"
            if termination_reason is not None:
                break
        if termination_reason is None:
            termination_reason = "round_limit"

    trace.append({
        "type": "termination",
        "termination_reason": termination_reason,
        "retrieval_rounds": steps,
        "tool_calls": tool_calls_used,
        "memory_visible_tokens": budget.used,
        "source_verification": verify_sources,
    })
    response = _chat_completion_with_retry(
        runtime.client.chat.completions.create,
        model=runtime.model,
        messages=messages + [{"role": "user", "content": FINAL_PROMPT}],
        max_tokens=config.max_output_tokens,
        temperature=0.0,
    )
    runtime.log_usage(response, phase="memory_qa_finalize")
    steps += 1
    content = response.choices[0].message.content or ""
    match = _ANSWER_RE.search(content)
    answer = match.group(1).strip() if match else content.strip()
    return evidence, steps, answer or "Insufficient information.", trace
