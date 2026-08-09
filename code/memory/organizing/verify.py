"""Deterministic reachability: is every recorded fact still findable.

Organizing rearranges prose; nothing before this checked whether the
rearranged prose is still something a search would surface. A fact memory
holds but the index cannot return is invisible to retrieval, and nothing
today would notice — this is what would.

This is a search-side check with no model, unlike
`memory.management.verify_session`, which has a model answer a question
against memory and checks the answer. That one verifies one session's write
against its sources; this one verifies the whole of `topics/` against its
own index, which is a different, cheaper question asked of everything at
once rather than of one write at a time.

`memory.retrieval`'s public surface has no plain "search returns these rows"
function — what it exports either answers through a model (`collect_answer`,
`Runtime`) or serves the interactive read-only path (`memory_files`,
`read_memory_file`), neither of which is a deterministic index search. So
this goes straight to `MemoryBM25Index`, the index class BM25 search already
runs against. Matching is by block ID rather than by comparing prose: the
index and the topic parser normalize a paragraph's text slightly
differently (the index also flattens Markdown links, for instance), but both
derive a merged paragraph's identity the same way — the last `^id` in its
suffix run — so an ID match is exact where a text match would be fuzzy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..markdown import parse_topic_tree
from ..retrieval.bm25 import MemoryBM25Index


def unreachable(
    memory_dir: str | Path, *, top_k: int = 10
) -> list[dict[str, Any]]:
    """Recorded facts a search for their own wording does not return.

    Each row names the file, the block ID, and the fact's own text, so the
    result can be handed to a reader or an organizer without another lookup.
    """
    memory_dir = Path(memory_dir)
    topics = memory_dir / "topics"
    if not topics.is_dir():
        return []
    index = MemoryBM25Index(memory_dir, persist=False)
    missing: list[dict[str, Any]] = []
    for unit in parse_topic_tree(topics):
        if not unit.content.strip():
            continue
        hits = index.search(unit.content, top_k=top_k)
        found = any(
            hit["event_id"].split(":", 1)[0] == unit.memory_id
            for hit in hits
        )
        if not found:
            missing.append({
                "path": f"topics/{unit.topic_path}",
                "block_id": unit.memory_id,
                "fact": unit.content,
            })
    return missing
