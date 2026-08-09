"""Deterministic seed search: the closest memory to a query, no model.

Every retrieval starts by searching for the words in the query. That is not
a judgement — it is what every retrieval does before it can think — so it
is a named function here rather than something each caller (the service's
`/search`, or a reading pass's own opening turn) assembles for itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .index_cache import IndexCache
from .tools import execute_tool_call, is_memory_output
from .views import memory_files

_log = logging.getLogger(__name__)


def nearest(
    memory_dir: Path,
    query: str,
    *,
    top_k: int = 10,
    search_tools: str = "fused",
) -> list[str]:
    """The closest memory to `query`: zero or one rendered passage.

    `memory_search` merges BM25 and embedding ranking into one rendered
    block, so there is nothing further to split until a model reads it.
    `search_tools="bm25"` asks for the lexical half alone, which needs no
    embedding backend and so no torch in the image.
    """
    tool = "memory_search" if search_tools == "fused" else "bm25_search"
    memory_dir = Path(memory_dir).resolve()
    files = memory_files(memory_dir, "native", include_recent=True)
    try:
        output, _executed, _accepted = execute_tool_call(
            IndexCache(memory_dir),
            tool,
            {"query": query, "top_k": top_k},
            memory_dir=memory_dir,
            files=files,
            condition="native",
            include_recent=True,
            indexes={},
        )
    except Exception:  # noqa: BLE001 - a seed that fails is still a floor
        # Loudly. A search backend that cannot be built returns nothing, and
        # nothing is indistinguishable from "memory holds no answer" — an
        # image shipped without the embedding backend once answered every
        # query with silence and looked healthy doing it.
        _log.exception("seed search failed for %r", query)
        return []
    output = output.strip()
    return [output] if is_memory_output(output) else []
