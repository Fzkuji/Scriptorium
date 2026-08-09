"""Write-time verification, plus what outside callers still reach here.

Writing is `memory.writing`; organizing is `memory.organizing`. This package
is what a write needs checked against its sources after the fact —
`verify_session` — and two re-exports so nothing outside had to move in
lockstep: ``write_sessions`` from writing, and ``organize_topics`` from
organizing, kept as the flat audit list it always returned since that is
the shape a caller of this name already depends on.
"""

from pathlib import Path
from typing import Any

from .agent import _run_agent
from .verification import verify_session
from ..config import MemoryConfig
from ..organizing.reorganize import reorganize
from ..workspace import MemoryWorkspace
from ..writing import write_sessions


def organize_topics(
    memory_dir: str | Path,
    *,
    agent: Any,
    touched: set[str] | None = None,
    usage_logger: Any | None = None,
    config: MemoryConfig | None = None,
) -> list[dict[str, Any]]:
    """Compatibility name for `memory.organizing.reorganize`.

    `reorganize` now runs `tidy` and the model pass and reports both; this
    name predates that split and a caller still using it gets only the model
    pass's audit trail, which is all this function ever returned.
    """
    return reorganize(
        memory_dir, agent=agent, touched=touched,
        usage_logger=usage_logger, config=config,
    )["organize"]


__all__ = [
    "MemoryConfig",
    "MemoryWorkspace",
    "organize_topics",
    "verify_session",
    "write_sessions",
]
