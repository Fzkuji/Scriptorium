"""Writer agent backed by an OpenAI chat-completions endpoint.

Scriptorium's writer normally runs inside Claude Code, which speaks Anthropic's
protocol. The Agent Memory Leaderboard mandates gpt-4o-mini for Add and Search,
so this module drives the same writing protocol against an OpenAI-compatible
endpoint instead.

It is a drop-in for ``ClaudeCodeAgent``: same ``run(...)`` keyword arguments,
same ``AgentResult`` back. The workspace, the ``shell`` tool, the system prompt,
and the transaction rules are all unchanged — only the model transport differs.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import APITimeoutError, OpenAI

from .claude_code import AgentExecutionError, AgentResult

# A call given less than this has no chance of returning, and cutting the first
# turn off leaves the pass with nothing to commit at all.
_MINIMUM_CALL_SECONDS = 20.0


class _CallOverran(Exception):
    """One endpoint call outlived the time the caller had left for it."""


def _within(seconds: float | None, call: Any) -> Any:
    """Run ``call``, giving up on it once ``seconds`` have passed.

    The client's own timeout does not bound this. It is an idle timeout: a
    gateway that trickles bytes, or sends keepalives while an upstream model
    generates, resets it on every read, and one measured call ran 786s against
    a 180s setting. Only the wall clock bounds a wall clock, so the call runs
    on a thread this one stops waiting for.

    The abandoned thread finishes into nothing. That wastes the tokens the
    call had already earned, which is the price of answering the caller who is
    still holding the line — and at 18 abandonments in 768 writes, it is a
    price worth paying.
    """
    if seconds is None:
        return call()

    outcome: dict[str, Any] = {}

    def settle() -> None:
        try:
            outcome["value"] = call()
        except BaseException as exc:  # carried back to the waiting thread
            outcome["error"] = exc

    worker = threading.Thread(target=settle, daemon=True)
    # The clock starts before the thread does. Starting one waits for it to be
    # scheduled, and in a process already running hundreds of them that wait
    # is time the caller has spent but the budget never saw.
    deadline = time.monotonic() + seconds
    worker.start()
    worker.join(max(0.0, deadline - time.monotonic()))
    if worker.is_alive():
        raise _CallOverran(f"no response within {seconds:.0f}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


@dataclass(frozen=True)
class OpenAIAgentConfig:
    base_url: str
    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    request_timeout_s: float = 180.0
    max_retries: int = 4

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url is required")
        if not self.api_key:
            raise ValueError("api_key is required")
        if not self.model.strip():
            raise ValueError("model is required")


def _redact(message: str, api_key: str) -> str:
    return message.replace(api_key, "***") if api_key else message


def _tool_schemas(tools: list[Any]) -> list[dict[str, Any]]:
    """Translate the SDK tool definitions into OpenAI function schemas."""
    schemas = []
    for definition in tools:
        schemas.append({
            "type": "function",
            "function": {
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.input_schema,
            },
        })
    return schemas


# Enough of a call to recognise it, not so much that a trajectory outweighs
# the memory it wrote.
_ARGUMENT_PREVIEW = 400
_RESULT_PREVIEW = 400


class OpenAIWriterAgent:
    # Nothing here but the tools it is handed: the writing prompts assume a
    # Write and an Edit tool, and this runtime has to be given them.
    has_file_tools = False

    """Run one writing trajectory against an OpenAI-compatible endpoint."""

    def __init__(self, config: OpenAIAgentConfig, *, client: Any | None = None):
        self.config = config
        self._client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.request_timeout_s,
            max_retries=config.max_retries,
        )

    def run(
        self,
        *,
        prompt: str,
        system_prompt: str,
        cwd: str | Path,
        tools: list[Any] | None = None,
        max_turns: int = 20,
        max_seconds: float | None = None,
        max_budget_usd: float | None = None,
        output_schema: dict[str, Any] | None = None,
        # Offer these schemas and hand the calls back instead of dispatching
        # them. A pass that only reads has no handlers to run and wants the
        # arguments themselves.
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> AgentResult:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        if not Path(cwd).is_dir():
            raise ValueError(f"agent working directory does not exist: {cwd}")

        # The SDK tools are async callables carrying their own schema; keep a
        # lookup so tool calls can be dispatched by name.
        handlers = {definition.name: definition.handler for definition in (tools or [])}
        schemas = tool_schemas or _tool_schemas(tools or [])

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        started = time.time()
        api_ms = 0
        input_tokens = 0
        output_tokens = 0
        texts: list[str] = []
        stop_reason = "complete"
        turns = 0
        # What each turn called and what came back. Without it a run that
        # spent every turn retrying one rejected Edit is indistinguishable
        # from one that did the work, and both report `max_turns`.
        trajectory: list[dict[str, Any]] = []

        for turns in range(1, max_turns + 1):
            # Whoever is waiting on this pass has a deadline of their own, and
            # one endpoint call is what overruns it: the client's own timeout
            # is three minutes and it retries, so a single turn can outlast the
            # whole budget several times over. Each call is given only the time
            # that is left, which makes the budget a ceiling on the pass rather
            # than on the number of turns it opens.
            left = None
            if max_seconds is not None:
                left = max_seconds - (time.time() - started)
                # Too little left to be worth a round trip: stop here rather
                # than spend the remainder on a call that cannot land.
                if turns > 1 and left < _MINIMUM_CALL_SECONDS:
                    turns -= 1
                    stop_reason = "max_seconds"
                    break
                # The first turn is the only chance this pass has to write
                # anything, so it gets the floor even when that overruns a
                # budget already spent before the pass began.
                left = max(_MINIMUM_CALL_SECONDS, left)

            client = self._client
            if left is not None:
                # Retries stay on under a deadline. Dropping them looked like
                # the way to keep a call inside its budget, but the budget is
                # already kept by the clock below, and a gateway's transient
                # 429 or 502 would instead end the write outright and be
                # reported to the caller as a failed ingest.
                client = client.with_options(timeout=left)
            call_started = time.time()
            try:
                response = _within(left, lambda: client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    tools=schemas or None,
                    temperature=self.config.temperature,
                ))
            except (APITimeoutError, _CallOverran):
                # Out of time. Earlier turns already wrote through the tools,
                # so the pass reports what it got instead of failing whole.
                api_ms += int((time.time() - call_started) * 1000)
                turns -= 1
                stop_reason = "max_seconds"
                break
            except Exception as exc:
                raise AgentExecutionError(
                    _redact(f"{type(exc).__name__}: {exc}", self.config.api_key)
                ) from exc
            api_ms += int((time.time() - call_started) * 1000)

            usage = getattr(response, "usage", None)
            if usage:
                input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

            choice = response.choices[0]
            message = choice.message
            if message.content:
                texts.append(message.content)

            calls = list(message.tool_calls or [])
            if not calls:
                stop_reason = choice.finish_reason or "complete"
                break

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [{
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                } for call in calls],
            })

            if tool_schemas is not None:
                # Reading, not editing: the caller asked what the model would
                # record, and records it itself. Arguments come back whole,
                # because here they are the answer rather than a note on how
                # the turn went.
                trajectory.extend({
                    "tool": call.function.name,
                    "arguments": call.function.arguments,
                } for call in calls)
                stop_reason = choice.finish_reason or "complete"
                break

            for call in calls:
                output = self._dispatch(handlers, call)
                trajectory.append({
                    "tool": call.function.name,
                    "arguments": str(call.function.arguments)[:_ARGUMENT_PREVIEW],
                    "result": output[:_RESULT_PREVIEW],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": output,
                })
        else:
            stop_reason = "max_turns"

        return AgentResult(
            text="\n".join(texts),
            structured_output=None,
            num_turns=turns,
            turns=trajectory,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            anthropic_equivalent_cost_usd=None,
            duration_ms=int((time.time() - started) * 1000),
            duration_api_ms=api_ms,
            stop_reason=stop_reason,
            session_id="",
        )

    def _dispatch(self, handlers: dict[str, Any], call: Any) -> str:
        """Run one tool call and render its result as text for the model."""
        import asyncio

        handler = handlers.get(call.function.name)
        if handler is None:
            return f"error: unknown tool {call.function.name}"
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"error: arguments were not valid JSON: {exc}"
        try:
            result = asyncio.run(handler(arguments))
        except Exception as exc:  # a failing tool is fed back, not fatal
            return f"error: {type(exc).__name__}: {exc}"
        blocks = result.get("content", []) if isinstance(result, dict) else []
        text = "\n".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict)
        )
        return text or "(no output)"
