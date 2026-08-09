"""Memory writer configuration."""

from dataclasses import dataclass

@dataclass(frozen=True)
class MemoryConfig:
    core_max_tokens: int = 3_000
    core_repair_target_tokens: int = 2_700
    core_repair_max_checks: int = 8
    core_repair_max_trajectories: int = 2
    core_repair_stagnation_limit: int = 2
    writer_shell_examples: bool = False
    recent_limit: int = 50
    max_turns: int = 20
    max_budget_usd: float | None = None
    shell_backend: str = "auto"

    def __post_init__(self) -> None:
        if self.core_max_tokens < 0:
            raise ValueError("core_max_tokens must be non-negative")
        if self.core_repair_target_tokens < 0:
            raise ValueError("core_repair_target_tokens must be non-negative")
        if self.core_repair_max_checks < 1:
            raise ValueError("core_repair_max_checks must be positive")
        if self.core_repair_max_trajectories < 1:
            raise ValueError("core_repair_max_trajectories must be positive")
        if self.core_repair_stagnation_limit < 1:
            raise ValueError("core_repair_stagnation_limit must be positive")
        if self.recent_limit < 0:
            raise ValueError("recent_limit must be non-negative")
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        if self.max_budget_usd is not None and self.max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be positive")
        if self.shell_backend not in {"auto", "posix-bash", "native"}:
            raise ValueError(
                "shell_backend must be auto, posix-bash, or native"
            )

    @property
    def effective_core_repair_target_tokens(self) -> int:
        """Return a target with roughly ten percent headroom below the limit."""
        headroom_target = int(self.core_max_tokens * 0.9)
        return min(self.core_repair_target_tokens, headroom_target)
