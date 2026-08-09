"""Memory writer configuration."""

from dataclasses import dataclass

@dataclass(frozen=True)
class MemoryConfig:
    core_max_tokens: int = 2_000
    few_shot_instructions: bool = False
    recent_limit: int = 50
    max_turns: int = 20
    # Wall-clock budget for one writing pass. A caller that must answer
    # inside a deadline — the served endpoint, whose client hangs up at about
    # a hundred seconds — sets this so the pass stops opening new turns once
    # the budget is spent and commits what it has written. Turns already in
    # flight run to completion, so the pass overruns by at most one turn.
    max_seconds: float | None = None
    max_budget_usd: float | None = None

    def __post_init__(self) -> None:
        if self.core_max_tokens < 0:
            raise ValueError("core_max_tokens must be non-negative")
        if self.recent_limit < 0:
            raise ValueError("recent_limit must be non-negative")
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        if self.max_budget_usd is not None and self.max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be positive")
