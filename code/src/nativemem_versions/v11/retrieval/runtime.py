"""Explicit API client and usage accounting for V11."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from .. import adapter
from .config import QueryConfig
from .shell import execute_workspace_bash


class UsageTracker:
    def __init__(self) -> None:
        self._local = threading.local()
        self._values: dict[str, dict[str, float | int]] = {}
        self._lock = threading.Lock()

    def reset(self, phase: str) -> None:
        self._local.phase = phase
        with self._lock:
            self._values[phase] = {
                "calls": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "llm_time_s": 0.0,
            }

    def record(self, tokens_in: int, tokens_out: int) -> None:
        phase = getattr(self._local, "phase", None)
        if phase is None:
            return
        with self._lock:
            value = self._values.get(phase)
            if value is not None:
                value["calls"] += 1
                value["tokens_in"] += tokens_in
                value["tokens_out"] += tokens_out

    def snapshot(self, phase: str) -> dict[str, float | int]:
        with self._lock:
            value = self._values.pop(phase, None)
        if getattr(self._local, "phase", None) == phase:
            self._local.phase = None
        return dict(value or {
            "calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "llm_time_s": 0.0,
        })


class Runtime:
    def __init__(
        self,
        client: Any,
        model: str,
        *,
        build_config: adapter.BuildConfig | None = None,
        query_config: QueryConfig | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.build_config = build_config
        self.query_config = query_config or QueryConfig()
        self.call_log: list[dict[str, Any]] = []
        self._usage_lock = threading.Lock()
        self.tracker = UsageTracker()

    def log_usage(self, response: Any, phase: str = "unknown") -> None:
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        with self._usage_lock:
            self.call_log.append({
                "phase": phase,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })
        self.tracker.record(prompt_tokens, completion_tokens)

    @staticmethod
    def build_turn_index(conv: dict[str, Any]) -> dict[str, Any]:
        return adapter.build_turn_index(conv)

    @staticmethod
    def read_turns(
        turn_index: dict[str, Any], source_ids: list[str], context: int = 1
    ) -> str:
        return adapter.read_turns(turn_index, source_ids, context=context)

    @staticmethod
    def execute_tool(
        tool_name: str,
        args: dict[str, Any],
        base_dir: str,
        hide_raw: bool = False,
    ) -> str:
        del hide_raw
        if tool_name != "bash":
            return f"Unknown tool: {tool_name}"
        return execute_workspace_bash(args.get("command", ""), Path(base_dir))

    def build_memory(self, conv: dict[str, Any], memory_dir: str):
        if self.build_config is None:
            raise RuntimeError("V11 build_config was not supplied")
        return adapter.build_memory(
            conv,
            memory_dir,
            client=self.client,
            model=self.model,
            usage_logger=lambda response: self.log_usage(response, "v11_agent"),
            config=self.build_config,
        )

    def collect_and_answer_longmemeval(
        self,
        item: dict[str, Any],
        memory_dir: Path,
        turn_index: dict[str, Any],
    ):
        from .agent import collect_answer

        return collect_answer(
            self, item, memory_dir, turn_index, config=self.query_config
        )


def create_runtime(
    base_url: str,
    *,
    api_key: str,
    api_format: str = "openai",
    model: str = "openai/gpt-4o-mini",
    max_retries: int = 2,
    timeout_seconds: float = 180.0,
    trust_proxy: bool = False,
    client: Any | None = None,
    build_config: adapter.BuildConfig | None = None,
    query_config: QueryConfig | None = None,
) -> Runtime:
    if not base_url:
        raise ValueError("base_url is required")
    if not api_key:
        raise ValueError("api_key is required")
    if api_format not in {"openai", "anthropic"}:
        raise ValueError("api_format must be 'openai' or 'anthropic'")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if client is None and api_format == "anthropic":
        from src.anthropic_openai_compat import AnthropicOpenAICompat

        client = AnthropicOpenAICompat(api_key, base_url)
    elif client is None:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            http_client=httpx.Client(
                trust_env=trust_proxy,
                timeout=timeout_seconds,
            ),
        )
    return Runtime(
        client,
        model,
        build_config=build_config,
        query_config=query_config,
    )
