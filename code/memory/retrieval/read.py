"""The model reads memory and reports what bears on a query.

Search up to this point is deterministic; from here a model decides what to
look at further and what belongs in the report. It is a separate module
from `search.py` because it needs an agent and a model, where `nearest` is
called from contexts — a plain HTTP request, a unit test — that must not
need either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..prompts import FIND_MEMORY
from .config import QueryConfig
from .context import initialize_context
from .index_cache import IndexCache
from .tool_server import RetrievalToolState, retrieval_tools
from .tools import is_memory_output
from .views import memory_files


@dataclass
class ReadResult:
    """What one reading pass over memory produced.

    A caller that only needs to shape a response — the leaderboard's
    Search, say — wants the report, what was actually read, and enough of
    the run's accounting to log one usage line, without needing to know how
    the agent got there.
    """

    text: str
    passages: list[str]
    trace: list[dict[str, Any]]
    num_turns: int
    input_tokens: int
    output_tokens: int
    stop_reason: str
    tool_calls: int
    visible_tokens: int


def read(
    memory_dir: Path,
    query: str,
    *,
    agent: Any,
    model: str,
    config: QueryConfig,
    seed: list[str] | None = None,
) -> ReadResult:
    """Build the context, run the reading agent, and report what it found.

    `seed` is the deterministic search a caller already ran (typically
    `search.nearest`); folding it into the opening prompt means a weak model
    that would otherwise stop after one call already has something to
    work from.
    """
    memory_dir = Path(memory_dir).resolve()
    files = memory_files(memory_dir, "native", include_recent=True)
    prompt, trace, evidence, initial_tokens = initialize_context(
        memory_dir=memory_dir,
        files=files,
        condition="native",
        item={"question": query, "question_date": ""},
        verify_sources=config.verify_sources,
        model=model,
    )
    if seed:
        prompt = (
            f"{prompt}\n\nClosest memory to the request, already looked up:\n\n"
            + "\n\n".join(seed)
        )
    state = RetrievalToolState(
        trace=trace, evidence=evidence, model=model, visible_tokens=initial_tokens,
    )
    result = agent.run(
        prompt=prompt,
        system_prompt=FIND_MEMORY,
        cwd=memory_dir,
        tools=retrieval_tools(
            IndexCache(memory_dir),
            memory_dir=memory_dir,
            files=files,
            condition="native",
            include_recent=True,
            state=state,
            search_tools=config.search_tools,
        ),
        max_turns=config.max_turns,
        max_budget_usd=config.max_budget_usd,
    )
    reported = re.sub(r"</?answer>", "", result.text or "").strip()
    # What the agent actually read, verbatim, in the order it read it —
    # filtered to memory rather than the directory listings and misses it
    # also saw along the way.
    passages = []
    for row in state.evidence:
        text = str(row.get("text", "")).strip()
        if is_memory_output(text, row.get("tool")):
            passages.append(text)
    return ReadResult(
        text=reported,
        passages=passages,
        trace=trace,
        num_turns=result.num_turns,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        stop_reason=result.stop_reason,
        tool_calls=state.tool_calls,
        visible_tokens=state.visible_tokens,
    )
