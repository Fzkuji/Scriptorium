"""Scriptorium: model-managed Markdown files as external memory.

Public facade so installed users need not import from the repository's
internal ``src`` layout.
"""

from src.build import BuildConfig, build_memory
from src.management import MemoryWorkspace
from src.management.config import MemoryConfig
from src.management.transaction import (
    TransactionError,
    TransactionLimits,
    TransactionResult,
)
from src.retrieval import inspect
from src.retrieval.agent import collect_answer
from src.retrieval.config import QueryConfig

__all__ = [
    "BuildConfig",
    "MemoryConfig",
    "MemoryWorkspace",
    "QueryConfig",
    "TransactionError",
    "TransactionLimits",
    "TransactionResult",
    "build_memory",
    "collect_answer",
    "inspect",
]
