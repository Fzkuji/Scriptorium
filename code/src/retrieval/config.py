"""Validated retrieval settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryConfig:
    max_turns: int = 20
    max_budget_usd: float | None = None
    verify_sources: bool = True

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        if self.max_budget_usd is not None and self.max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be positive")
