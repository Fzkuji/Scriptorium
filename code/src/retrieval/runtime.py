"""Claude Code agent and usage accounting for Agent Memory Harness."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .. import build as adapter
from ..agent_runtime import ClaudeCodeAgent, ClaudeCodeConfig
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

    def record(
        self,
        tokens_in: int,
        tokens_out: int,
        *,
        calls: int = 1,
        llm_time_s: float = 0.0,
    ) -> None:
        phase = getattr(self._local, "phase", None)
        if phase is None:
            return
        with self._lock:
            value = self._values.get(phase)
            if value is not None:
                value["calls"] += calls
                value["tokens_in"] += tokens_in
                value["tokens_out"] += tokens_out
                value["llm_time_s"] += llm_time_s

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
    supports_batch_checkpoint = True
    def __init__(
        self,
        agent: Any,
        model: str,
        *,
        query_agent: Any | None = None,
        query_model: str | None = None,
        build_config: adapter.BuildConfig | None = None,
        query_config: QueryConfig | None = None,
    ) -> None:
        self.builder_agent = agent
        self.builder_model = model
        self.agent = query_agent or agent
        self.model = query_model or model
        self.build_config = build_config
        self.query_config = query_config or QueryConfig()
        self.call_log: list[dict[str, Any]] = []
        self._usage_lock = threading.Lock()
        self._retrieval_index_lock = threading.Lock()
        self._retrieval_indexes: dict[tuple[Any, ...], Any] = {}
        self.tracker = UsageTracker()

    def get_retrieval_index(self, key: tuple[Any, ...], factory: Any) -> Any:
        with self._retrieval_index_lock:
            if key not in self._retrieval_indexes:
                self._retrieval_indexes[key] = factory()
            return self._retrieval_indexes[key]

    def log_agent_result(
        self, result: Any, phase: str = "unknown"
    ) -> None:
        prompt_tokens = int(getattr(result, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(result, "output_tokens", 0) or 0)
        cache_write_tokens = int(
            getattr(result, "cache_creation_input_tokens", 0) or 0
        )
        cache_read_tokens = int(
            getattr(result, "cache_read_input_tokens", 0) or 0
        )
        calls = int(getattr(result, "num_turns", 0) or 0)
        duration_ms = int(getattr(result, "duration_ms", 0) or 0)
        with self._usage_lock:
            self.call_log.append({
                "phase": phase,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cache_write_tokens": cache_write_tokens,
                "cache_read_tokens": cache_read_tokens,
                "calls": calls,
                "anthropic_equivalent_cost_usd": getattr(
                    result, "anthropic_equivalent_cost_usd", None
                ),
                "duration_ms": duration_ms,
                "duration_api_ms": int(
                    getattr(result, "duration_api_ms", 0) or 0
                ),
                "stop_reason": getattr(result, "stop_reason", None),
                "session_id": getattr(result, "session_id", ""),
            })
        self.tracker.record(
            prompt_tokens,
            completion_tokens,
            calls=calls,
            llm_time_s=duration_ms / 1000,
        )

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
        return execute_workspace_bash(
            args.get("command", ""),
            Path(base_dir),
        )

    def build_memory(
        self,
        conv: dict[str, Any],
        memory_dir: str,
        *,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ):
        if self.build_config is None:
            raise RuntimeError("Scriptorium build_config was not supplied")
        result = adapter.build_memory(
            conv,
            memory_dir,
            agent=self.builder_agent,
            model=self.builder_model,
            usage_logger=lambda result: self.log_agent_result(
                result, "writer"
            ),
            config=self.build_config,
            checkpoint_path=checkpoint_path,
            resume=resume,
        )
        with self._retrieval_index_lock:
            self._retrieval_indexes.clear()
        return result

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
    model: str = "openai/gpt-4o-mini",
    cli_path: str | None = None,
    query_base_url: str | None = None,
    query_api_key: str | None = None,
    query_model: str | None = None,
    query_cli_path: str | None = None,
    agent: Any | None = None,
    query_agent: Any | None = None,
    build_config: adapter.BuildConfig | None = None,
    query_config: QueryConfig | None = None,
) -> Runtime:
    if not base_url:
        raise ValueError("base_url is required")
    if not api_key:
        raise ValueError("api_key is required")
    if agent is None:
        agent = ClaudeCodeAgent(
            ClaudeCodeConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
                cli_path=cli_path,
            )
        )
    if query_agent is None and any(
        value is not None
        for value in (query_base_url, query_api_key, query_model, query_cli_path)
    ):
        query_agent = ClaudeCodeAgent(
            ClaudeCodeConfig(
                base_url=query_base_url or base_url,
                api_key=query_api_key or api_key,
                model=query_model or model,
                cli_path=query_cli_path if query_cli_path is not None else cli_path,
            )
        )
    return Runtime(
        agent,
        model,
        query_agent=query_agent,
        query_model=query_model,
        build_config=build_config,
        query_config=query_config,
    )
