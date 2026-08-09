"""Isolated synchronous adapter for the Claude Agent SDK."""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    query as sdk_query,
)


class AgentExecutionError(RuntimeError):
    """Claude Code could not complete an agent trajectory."""


@dataclass(frozen=True)
class ClaudeCodeConfig:
    base_url: str
    api_key: str
    model: str
    cli_path: str | None = None

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url is required")
        if not self.api_key:
            raise ValueError("api_key is required")
        if not self.model.strip():
            raise ValueError("model is required")
        if self.cli_path is not None and not self.cli_path.strip():
            raise ValueError("cli_path must not be empty")


@dataclass(frozen=True)
class AgentResult:
    text: str
    structured_output: Any
    num_turns: int
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    # Priced by the SDK against Anthropic's own rate table. When the run is
    # routed elsewhere via ANTHROPIC_BASE_URL this is not the amount billed;
    # it is only "what this token volume would cost on Anthropic".
    anthropic_equivalent_cost_usd: float | None
    duration_ms: int
    duration_api_ms: int
    stop_reason: str | None
    session_id: str


QueryFunction = Callable[..., AsyncIterator[Any]]
ProgressFunction = Callable[[dict[str, Any]], None]


class ClaudeCodeAgent:
    """Run one non-persistent Claude Code process per trajectory."""

    def __init__(
        self,
        config: ClaudeCodeConfig,
        *,
        query_fn: QueryFunction | None = None,
    ) -> None:
        self.config = config
        self._query = query_fn or sdk_query

    def run(
        self,
        *,
        prompt: str,
        system_prompt: str,
        cwd: str | Path,
        tools: list[Any] | None = None,
        max_turns: int = 20,
        max_budget_usd: float | None = None,
        output_schema: dict[str, Any] | None = None,
        progress_fn: ProgressFunction | None = None,
    ) -> AgentResult:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        if max_budget_usd is not None and max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be positive")
        # Resolve before building the coroutine: an exception raised while
        # evaluating these arguments would leave _run() created but never
        # awaited, which surfaces as a RuntimeWarning far from its cause.
        resolved_cwd = Path(cwd).resolve()
        return asyncio.run(self._run(
            prompt=prompt,
            system_prompt=system_prompt,
            cwd=resolved_cwd,
            tools=tools or [],
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            output_schema=output_schema,
            progress_fn=progress_fn,
        ))

    async def _run(
        self,
        *,
        prompt: str,
        system_prompt: str,
        cwd: Path,
        tools: list[Any],
        max_turns: int,
        max_budget_usd: float | None,
        output_schema: dict[str, Any] | None,
        progress_fn: ProgressFunction | None,
    ) -> AgentResult:
        if not cwd.is_dir():
            raise ValueError(f"agent working directory does not exist: {cwd}")
        server_name = "agent_memory"
        mcp_servers = {}
        # The workspace is a scratch directory, so the built-in file tools are
        # safe here and are what the model is trained to reach for. Editing
        # through them beats scripting the same change in a shell heredoc.
        builtin_tools = ["Read", "Edit", "Write", "Grep", "Glob"]
        allowed_tools = list(builtin_tools)
        if tools:
            mcp_servers[server_name] = create_sdk_mcp_server(
                server_name, tools=tools
            )
            allowed_tools += [
                f"mcp__{server_name}__{definition.name}"
                for definition in tools
            ]

        with tempfile.TemporaryDirectory(
            prefix="agent-memory-claude-config-"
        ) as config_root:
            options = ClaudeAgentOptions(
                # `tools` is what puts schemas on the wire; `allowed_tools` only
                # filters what may run. Leaving this empty left weaker models
                # with the MCP shell as their sole visible tool, so they wrote
                # tool names into shell commands instead of calling the tools.
                tools=builtin_tools,
                allowed_tools=allowed_tools,
                system_prompt=system_prompt,
                mcp_servers=mcp_servers,
                permission_mode="dontAsk",
                model=self.config.model,
                thinking={"type": "disabled"},
                max_turns=max_turns,
                max_budget_usd=max_budget_usd,
                cwd=cwd,
                cli_path=self.config.cli_path,
                setting_sources=[],
                env={
                    "ANTHROPIC_BASE_URL": self.config.base_url,
                    "ANTHROPIC_API_KEY": self.config.api_key,
                    "ANTHROPIC_AUTH_TOKEN": "",
                    "CLAUDE_CONFIG_DIR": config_root,
                },
                extra_args={
                    "bare": None,
                    "no-session-persistence": None,
                    "strict-mcp-config": None,
                },
                output_format=(
                    {"type": "json_schema", "schema": output_schema}
                    if output_schema is not None
                    else None
                ),
            )
            texts: list[str] = []
            final: ResultMessage | None = None
            assistant_messages = 0
            try:
                async for message in self._query(prompt=prompt, options=options):
                    if isinstance(message, AssistantMessage):
                        assistant_messages += 1
                        texts.extend(
                            block.text
                            for block in message.content
                            if isinstance(block, TextBlock)
                        )
                        if progress_fn is not None:
                            progress_fn({
                                "event": "assistant_message",
                                "assistant_messages": assistant_messages,
                                "content_blocks": len(message.content),
                            })
                    elif isinstance(message, ResultMessage):
                        final = message
                        if progress_fn is not None:
                            progress_fn({
                                "event": "result",
                                "assistant_messages": assistant_messages,
                                "num_turns": int(message.num_turns),
                                "is_error": bool(message.is_error),
                            })
            except Exception as exc:
                if final is None:
                    message = str(exc).replace(self.config.api_key, "[redacted]")
                    raise AgentExecutionError(message) from exc

            if final is None:
                raise AgentExecutionError(
                    "Claude Code ended without a result message"
                )
            if final.is_error:
                details = "; ".join(final.errors or []) or (
                    final.result or final.subtype
                )
                if getattr(final, "api_error_status", None) is not None:
                    details += f"; API status {final.api_error_status}"
                details = details.replace(self.config.api_key, "[redacted]")
                raise AgentExecutionError(details)
            usage = final.usage or {}
            return AgentResult(
                text=(final.result or "\n".join(texts)).strip(),
                structured_output=final.structured_output,
                num_turns=int(final.num_turns),
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
                cache_creation_input_tokens=int(
                    usage.get("cache_creation_input_tokens", 0) or 0
                ),
                cache_read_input_tokens=int(
                    usage.get("cache_read_input_tokens", 0) or 0
                ),
                anthropic_equivalent_cost_usd=final.total_cost_usd,
                duration_ms=int(final.duration_ms),
                duration_api_ms=int(final.duration_api_ms),
                stop_reason=final.stop_reason,
                session_id=final.session_id,
            )
