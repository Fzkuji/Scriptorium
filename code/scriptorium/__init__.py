"""Scriptorium: model-managed Markdown files as external memory.

Public facade so installed users need not import from the repository's
internal ``src`` layout. Attributes resolve lazily: the CLI and MCP server
must not pay for — or depend on — the experiment stack behind
``build_memory`` and ``collect_answer``.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BuildConfig": ("src.build", "BuildConfig"),
    "build_memory": ("src.build", "build_memory"),
    "MemoryWorkspace": ("src.management", "MemoryWorkspace"),
    "MemoryConfig": ("src.management.config", "MemoryConfig"),
    "TransactionError": ("src.management.transaction", "TransactionError"),
    "TransactionLimits": ("src.management.transaction", "TransactionLimits"),
    "TransactionResult": ("src.management.transaction", "TransactionResult"),
    "inspect": ("src.retrieval.inspect", None),
    "collect_answer": ("src.retrieval.agent", "collect_answer"),
    "QueryConfig": ("src.retrieval.config", "QueryConfig"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, attribute) if attribute else module
