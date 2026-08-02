"""Initial visible context and token accounting for retrieval."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..runtime.tokenization import TokenCounter

from .config import QueryConfig
from .prompts import RETRIEVAL_PROMPT


_SOURCE_REF_RE = re.compile(
    r"D\d+:\d+(?:-(?:D\d+:)?\d+)?|"
    r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+"
)


def evidence_keys(text: str) -> set[str]:
    keys = {f"ref:{ref}" for ref in _SOURCE_REF_RE.findall(text)}
    for line in text.splitlines():
        normalized = " ".join(line.casefold().split())
        if normalized:
            keys.add(
                "line:" + hashlib.sha256(normalized.encode()).hexdigest()
            )
    return keys


@dataclass
class VisibleBudget:
    counter: TokenCounter
    limit: int
    used: int = 0

    def consume(self, text: str) -> tuple[str, int, int]:
        raw_tokens = self.counter.count(text)
        delivered = self.counter.truncate(
            text, max(0, self.limit - self.used)
        )
        delivered_tokens = self.counter.count(delivered)
        self.used += delivered_tokens
        return delivered, raw_tokens, delivered_tokens


def initialize_context(
    *,
    memory_dir: Path,
    files: list[Path],
    condition: str,
    item: dict[str, Any],
    verify_sources: bool,
    model: str,
    config: QueryConfig,
) -> tuple[
    list[Any],
    list[dict[str, Any]],
    list[dict[str, str]],
    set[str],
    VisibleBudget,
]:
    budget = VisibleBudget(
        TokenCounter.resolve(requested_model=model),
        config.visible_token_limit,
    )
    core_path = memory_dir / "core.md"
    recent_path = memory_dir / "recent_events.jsonl"
    core_raw = (
        core_path.read_text(encoding="utf-8") if core_path in files else ""
    )
    recent_raw = (
        recent_path.read_text(encoding="utf-8")
        if recent_path in files
        else ""
    )
    inventory_raw = "\n".join(
        path.relative_to(memory_dir).as_posix() for path in files
    )
    core, core_raw_tokens, core_tokens = budget.consume(core_raw)
    recent, recent_raw_tokens, recent_tokens = budget.consume(recent_raw)
    inventory, inventory_raw_tokens, inventory_tokens = budget.consume(
        inventory_raw
    )
    messages: list[Any] = [{
        "role": "user",
        "content": RETRIEVAL_PROMPT.format(
            condition=condition,
            workspace_root=memory_dir,
            core_memory=core or "(empty)",
            recent_memory=recent or "(empty)",
            inventory=inventory or "(no visible files)",
            source_verification_guidance=(
                "Source files are directly accessible through the standard "
                "read and search tools. Verify relevant Topic evidence against "
                "its Source file before answering."
                if verify_sources
                else "Source files remain directly accessible through the "
                "standard read and search tools; explicit source verification "
                "is optional for this run."
            ),
            question_date=item.get("question_date", ""),
            question=item["question"],
        ),
    }]
    trace: list[dict[str, Any]] = []
    evidence: list[dict[str, str]] = []
    seen: set[str] = set()
    for initial in (core, recent):
        if initial.strip():
            evidence.append({"text": initial, "date": ""})
            seen.update(evidence_keys(initial))
    if core_raw.strip() or recent_raw.strip():
        trace.append({
            "type": "initial_context",
            "core_visible_tokens": core_tokens,
            "recent_visible_tokens": recent_tokens,
            "inventory_visible_tokens": inventory_tokens,
            "raw_visible_tokens": (
                core_raw_tokens
                + recent_raw_tokens
                + inventory_raw_tokens
            ),
            "delivered_visible_tokens": (
                core_tokens + recent_tokens + inventory_tokens
            ),
            "cumulative_visible_tokens": budget.used,
        })
    return messages, trace, evidence, seen, budget
