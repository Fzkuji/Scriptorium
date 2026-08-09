"""Process-wide cache of retrieval indexes, keyed by workspace revision.

Building the BM25 and embedding indexes means re-parsing and re-embedding
every memory file, which is most of a search's latency and none of its
work: the result is identical until a write changes the workspace. Keying
on the workspace's content revision means the first read after a write
rebuilds and every read after it is free — inside one request and across
the next ones — and the revision a write superseded is dropped rather than
accumulating.

Both `search.nearest` and `read.read` build their tools through this, so a
Search's own seed lookup and the reading pass that follows it build an
index once between them rather than twice. It is process-wide rather than
per-call for the same reason: a Search that follows an Add should rebuild
once, not once per request.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from ..workspace.transaction import workspace_revision

_indexes: dict[tuple[Any, ...], Any] = {}
_guard = threading.Lock()


class IndexCache:
    """The `get_retrieval_index` a retrieval tool call needs, for one workspace.

    Retrieval tool dispatch (`tools.execute_tool_call`) is written against a
    duck-typed runtime exposing `get_retrieval_index(key, factory)`; this is
    the minimal object satisfying that for a workspace path rather than a
    full harness Runtime.
    """

    def __init__(self, memory_dir: Path):
        self.revision = workspace_revision(memory_dir)

    def get_retrieval_index(
        self, key: tuple[Any, ...], factory: Callable[[], Any]
    ) -> Any:
        stamped = (self.revision, *key)
        with _guard:
            if stamped not in _indexes:
                # Only the current revision of a workspace is reachable, so
                # the ones this write superseded are dead weight. A run that
                # writes a thousand times keeps one index, not a thousand.
                for stale in [
                    held for held in _indexes
                    if held[1:] == key and held[0] != self.revision
                ]:
                    del _indexes[stale]
                _indexes[stamped] = factory()
            return _indexes[stamped]
