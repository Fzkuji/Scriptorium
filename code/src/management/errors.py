"""Structured memory-management failures."""

from __future__ import annotations


class CoreCapacityError(ValueError):
    """The staged Core Memory exceeds its configured token budget."""

    def __init__(self, *, token_count: int, limit: int) -> None:
        self.token_count = int(token_count)
        self.limit = int(limit)
        super().__init__(
            f"Core Memory exceeds {self.limit} tokens: {self.token_count}"
        )
