"""Memory writer configuration."""

from dataclasses import dataclass

@dataclass(frozen=True)
class MemoryConfig:
    core_max_tokens: int = 2_000
    recent_limit: int = 50
    agent_max_rounds: int = 12
    manager_max_rounds: int = 8
    reasoning_effort: str | None = None
    thinking: str | None = None
    retry_log: bool = False

    def __post_init__(self) -> None:
        if self.core_max_tokens < 0:
            raise ValueError("core_max_tokens must be non-negative")
        if self.recent_limit < 0:
            raise ValueError("recent_limit must be non-negative")
        if self.agent_max_rounds < 1:
            raise ValueError("agent_max_rounds must be positive")
        if self.manager_max_rounds < 1:
            raise ValueError("manager_max_rounds must be positive")

