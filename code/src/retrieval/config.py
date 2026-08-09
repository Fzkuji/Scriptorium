"""Validated retrieval settings."""

from dataclasses import dataclass

from .schemas import normalize_memory_components


@dataclass(frozen=True)
class QueryConfig:
    max_turns: int = 20
    max_budget_usd: float | None = None
    verify_sources: bool = True
    memory_components: tuple[str, ...] | str | None = None

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        if self.max_budget_usd is not None and self.max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be positive")
        object.__setattr__(
            self,
            "memory_components",
            normalize_memory_components(self.memory_components),
        )
