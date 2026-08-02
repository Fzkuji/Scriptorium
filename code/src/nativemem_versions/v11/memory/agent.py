"""Memory writer and organizer agent loop."""

import json
import shutil
from pathlib import Path
from typing import Any

from .config import MemoryConfig
from .model_reconciliation import _make_reconciler
from .prompts import SYSTEM_PROMPT, TOOLS
from .provider import _chat_completion_with_retry, _provider_options
from .workspace import MemoryWorkspace


def render_conversation(
    turns: list[tuple[str, str]], refs: list[str]
) -> str:
    return "\n".join(
        f"[{ref}] {speaker}: {text}"
        for (speaker, text), ref in zip(turns, refs)
    )


def _compact_tool_history(messages: list[Any]) -> None:
    """Retain full output for only the latest tool-call round."""
    def role(message: Any) -> object:
        return (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )

    last_assistant = max(
        (
            index
            for index, message in enumerate(messages)
            if role(message) == "assistant"
        ),
        default=-1,
    )
    for message in messages[:last_assistant]:
        if role(message) == "tool" and len(message.get("content", "")) > 1000:
            message["content"] = "[previous tool output omitted]"


def _run_agent(
    memory_dir: str | Path,
    *,
    client: Any,
    model: str,
    task: str,
    source_sessions: list[dict[str, Any]] | None = None,
    usage_logger: Any | None = None,
    final_output: list[str] | None = None,
    max_rounds: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    config: MemoryConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or MemoryConfig()
    if max_rounds is None:
        max_rounds = config.agent_max_rounds
    if max_rounds < 1:
        raise ValueError("V11 agent max rounds must be positive")
    available_tools = TOOLS if tools is None else tools
    allowed_tools = {tool["function"]["name"] for tool in available_tools}
    workspace = MemoryWorkspace(
        memory_dir,
        reconciler=_make_reconciler(client, model, usage_logger, config),
        config=config,
    )
    if source_sessions:
        workspace.archive_sessions(source_sessions)
        workspace._refresh_stage()
    task = f"{task}\n\nCurrent workspace structure:\n{workspace.structure()}"
    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    audit: list[dict[str, Any]] = []
    completed = False
    for round_no in range(max_rounds):
        _compact_tool_history(messages)
        request = {
            "model": model,
            "messages": messages,
            "tools": available_tools,
            "temperature": 0.1,
        }
        request.update(_provider_options(config))
        response = _chat_completion_with_retry(
            client.chat.completions.create,
            retry_log=config.retry_log,
            **request,
        )
        if usage_logger is not None:
            usage_logger(response)
        message = response.choices[0].message
        messages.append(message)
        calls = message.tool_calls or []
        if not calls:
            if final_output is not None:
                final_output.append(message.content or "")
            completed = True
            break
        for call in calls:
            try:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise ValueError("invalid JSON arguments") from exc
                if call.function.name not in allowed_tools:
                    raise ValueError(
                        f"tool is not allowed in this phase: "
                        f"{call.function.name}"
                    )
                if call.function.name == "shell":
                    result = workspace.shell(args["command"])
                    output = result.stdout + result.stderr
                    record = {
                        "round": round_no,
                        "tool": "shell",
                        "command": args["command"],
                        "returncode": result.returncode,
                        "output": output,
                        "count": workspace.last_created_blocks,
                        "topic_paths": workspace.last_changed_topics,
                    }
                elif call.function.name == "save_memory":
                    output = workspace.save_memory(args["events"])
                    record = {
                        "round": round_no,
                        "tool": "save_memory",
                        "count": len(args["events"]),
                        "output": output,
                        "topic_paths": sorted({
                            "topics/"
                            + MemoryWorkspace._validate_event(event)["topic_path"]
                            for event in args["events"]
                        }),
                    }
                else:
                    raise ValueError(f"unknown tool: {call.function.name}")
                record["status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                output = f"Tool error: {exc}"
                record = {
                    "round": round_no,
                    "tool": call.function.name,
                    "status": "error",
                    "output": output,
                }
            audit.append(record)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": output[-100_000:],
            })
    if not completed:
        audit.append({
            "tool": "agent",
            "status": "stopped",
            "reason": "round_limit",
            "rounds": max_rounds,
        })
    shutil.rmtree(workspace.stage_dir, ignore_errors=True)
    return audit
