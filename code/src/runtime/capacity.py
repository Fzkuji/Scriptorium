"""Writer-capacity artifacts and complete-session packing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .tokenization import TokenCounter


SCHEMA = "nativemem-writer-capacity-v1"


class SessionTooLargeError(ValueError):
    """A complete session cannot fit within the calibrated Writer limit."""


@dataclass(frozen=True)
class WriterCapacity:
    model: str
    writer_protocol_sha256: str
    safe_input_tokens: int
    tokenizer: dict[str, Any]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        model: str,
        writer_protocol_sha256: str,
    ) -> WriterCapacity:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("schema") != SCHEMA:
            raise ValueError("unsupported Writer capacity schema")
        if value.get("model") != model:
            raise ValueError("Writer capacity model does not match")
        if value.get("writer_protocol_sha256") != writer_protocol_sha256:
            raise ValueError("Writer capacity protocol does not match")
        safe_input_tokens = value.get("safe_input_tokens")
        if not isinstance(safe_input_tokens, int) or safe_input_tokens < 1:
            raise ValueError("safe_input_tokens must be a positive integer")
        tokenizer = value.get("tokenizer")
        if not isinstance(tokenizer, dict):
            raise ValueError("Writer capacity tokenizer is missing")
        return cls(
            model=model,
            writer_protocol_sha256=writer_protocol_sha256,
            safe_input_tokens=safe_input_tokens,
            tokenizer=tokenizer,
        )


def pack_complete_sessions(
    sessions: list[dict[str, Any]],
    *,
    max_input_tokens: int,
    render_batch: Callable[[list[dict[str, Any]]], str],
    token_counter: TokenCounter,
) -> list[list[dict[str, Any]]]:
    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be positive")
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for session in sessions:
        candidate = [*current, session]
        if token_counter.count(render_batch(candidate)) <= max_input_tokens:
            current = candidate
            continue
        if not current:
            raise SessionTooLargeError(
                "one complete session exceeds the calibrated Writer input limit"
            )
        batches.append(current)
        current = [session]
        if token_counter.count(render_batch(current)) > max_input_tokens:
            raise SessionTooLargeError(
                "one complete session exceeds the calibrated Writer input limit"
            )
    if current:
        batches.append(current)
    return batches


def select_safe_capacity(
    probes: list[dict[str, Any]], *, probe_ids: set[str]
) -> int:
    if not probe_ids:
        raise ValueError("probe_ids must not be empty")
    candidates = sorted({int(row["candidate_tokens"]) for row in probes})
    selected = 0
    for candidate in candidates:
        rows = [row for row in probes if int(row["candidate_tokens"]) == candidate]
        if rows and all(bool(row.get("skipped")) for row in rows):
            continue
        by_probe = {str(row["probe_id"]): bool(row.get("passed")) for row in rows}
        if set(by_probe) != probe_ids or not all(by_probe.values()):
            break
        selected = candidate
    if selected < 1:
        raise ValueError("no Writer capacity level passed every probe")
    return selected
