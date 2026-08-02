"""Validated retrieval settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryConfig:
    max_rounds: int = 8
    max_tool_calls: int = 5
    visible_token_limit: int = 10_000
    max_output_tokens: int = 1_200
    verify_sources: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_rounds",
            "max_tool_calls",
            "visible_token_limit",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
