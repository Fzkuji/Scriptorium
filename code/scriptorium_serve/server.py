"""Add / Search HTTP service for the Agent Memory Leaderboard.

The platform owns Answer and Eval; this service owns exactly two operations:

  POST /add     synchronous ingest of one chunk of messages for one user
  POST /search  BM25 retrieval of memory evidence for one user
  GET  /health  unauthenticated liveness probe

Contract notes that shaped this file:
  - Add must return 200 only after the write is committed AND retrievable.
  - Search returns evidence only. It never synthesises an answer.
  - user_id is the sole retrieval-isolation key: one workspace per user_id.
  - Academic track mandates gpt-4o-mini as the Add/Search model.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.agent_runtime import (
    AgentExecutionError,
    OpenAIAgentConfig,
    OpenAIWriterAgent,
)
from memory.management import MemoryConfig, write_sessions
from memory.retrieval import QueryConfig, nearest, read
from scriptorium.cli import ensure_workspace

WORKSPACE_ROOT = Path(os.environ.get("SCRIPTORIUM_WORKSPACES", "./workspaces")).resolve()
SERVICE_TOKEN = os.environ.get("SCRIPTORIUM_SERVICE_TOKEN", "")

# Writer model. The academic track requires gpt-4o-mini for Add/Search, so the
# default is pinned rather than left to the environment.
WRITER_MODEL = os.environ.get("SCRIPTORIUM_WRITER_MODEL", "gpt-4o-mini")
WRITER_BASE_URL = os.environ.get("SCRIPTORIUM_WRITER_BASE_URL", "")
WRITER_API_KEY = os.environ.get("SCRIPTORIUM_WRITER_API_KEY", "")

# One Add request carries at most 20 messages, so a single writer pass covers it.
# Cross-session reorganisation is deliberately not run inside a request: it would
# put the 1200s platform timeout at risk for no gain on a 20-message chunk.
# few_shot_instructions is the successor to the shell-era worked examples:
# the writer edits through file tools now, and weaker models still need the
# shapes spelled out.
MEMORY_CONFIG = MemoryConfig(few_shot_instructions=True)

# Retrieval is one shot here, so the agent is given room to look more than
# once; the caller's own timeout is the real ceiling. Source verification is
# off because the caller never sees a citation to check — it sees the memory.
# Lexical ranking alone. Measured against `fused` (BM25 plus embeddings) on
# 20 LoCoMo questions and on 97 BEAM questions across five conversations, the
# two are level on accuracy — 50.8% against 51.2% median gold-term coverage —
# while this one answers at a p90 of 3.7s against 12.3s and needs no torch,
# which is 2.2GB of the image. `fused` stays available for a memory where the
# wording of a question and the wording of its record diverge more than they
# do here.
SEARCH_TOOLS = os.environ.get("SCRIPTORIUM_SEARCH_TOOLS", "bm25")
QUERY_CONFIG = QueryConfig(
    max_turns=8, verify_sources=False, search_tools=SEARCH_TOOLS
)

# One line per request, so what a run costs and where its time goes is
# readable without re-deriving it from the platform's own timings.
_usage = logging.getLogger("scriptorium.usage")
if not _usage.handlers:
    # The server runs under uvicorn, which configures its own loggers and not
    # this one; without a handler of its own every line here goes nowhere.
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("usage %(message)s"))
    _usage.addHandler(_handler)
    _usage.setLevel(logging.INFO)
    _usage.propagate = False


def _record(operation: str, began: float, **fields: Any) -> None:
    _usage.info(json.dumps({
        "op": operation,
        "seconds": round(time.monotonic() - began, 2),
        **fields,
    }, ensure_ascii=False))


app = FastAPI(title="Scriptorium Memory Service")
_bearer = HTTPBearer(auto_error=False)

# Both handlers offload to the default thread pool, whose anyio limit is 40. The
# platform fans out up to 64 Add and 256 Search workers, so 40 would queue them
# behind each other and inflate latency under a load we can otherwise serve.
THREAD_LIMIT = int(os.environ.get("SCRIPTORIUM_THREAD_LIMIT", "384"))


@app.on_event("startup")
async def _raise_thread_limit() -> None:
    import anyio.to_thread

    anyio.to_thread.current_default_thread_limiter().total_tokens = THREAD_LIMIT


@app.on_event("startup")
async def _check_search_backend() -> None:
    """Refuse to start rather than answer every query with silence.

    A search backend that cannot be built is caught downstream and read as an
    empty result, so a service missing one serves every request successfully
    and returns nothing — a whole evaluation scored zero against a healthy
    process and a clean log. This is where that costs one line instead of a
    run.
    """
    if SEARCH_TOOLS == "fused":
        from sentence_transformers import SentenceTransformer

        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    else:
        import rank_bm25  # noqa: F401

# Writes to one workspace are serialised by Scriptorium's own file lock, but the
# platform fans out 64 workers and may hit the same user_id concurrently. A
# per-user lock keeps those requests queued in-process instead of contending on
# the file lock and burning the request timeout.
# ponytail: dict of locks, never evicted. Bounded by user_id count per run.
_user_locks: dict[str, threading.Lock] = {}
_user_locks_guard = threading.Lock()


_SAFE_ID = re.compile(r"[^A-Za-z0-9._:-]")


class Message(BaseModel):
    role: str
    content: str
    timestamp: int | None = None


class AddRequest(BaseModel):
    request_id: str
    messages: list[Message]
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str
    user_id: str
    top_k: int = 100
    options: list[str] | None = None


def _authorize(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    if not SERVICE_TOKEN:
        return
    if credentials is None or credentials.credentials != SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")


def _workspace(user_id: str) -> Path:
    """Map a user_id to its own workspace directory.

    user_id arrives from the platform (``eval:run_abc123:locomo:conv-0``) and is
    used as a path component, so anything outside a conservative allowlist is
    escaped rather than trusted.
    """
    safe = _SAFE_ID.sub("_", user_id)
    if not safe or safe in (".", ".."):
        raise HTTPException(status_code=400, detail="invalid user_id")
    path = WORKSPACE_ROOT / safe
    if WORKSPACE_ROOT not in path.resolve().parents:
        raise HTTPException(status_code=400, detail="invalid user_id")
    return path


def _ref_component(value: str) -> str:
    """Make an identifier safe to use inside a `provider/thread/message` ref."""
    cleaned = _SAFE_ID.sub("_", value).replace("/", "_").replace(":", "-")
    return cleaned or "unknown"


def _user_lock(user_id: str) -> threading.Lock:
    with _user_locks_guard:
        return _user_locks.setdefault(user_id, threading.Lock())


def _agent() -> OpenAIWriterAgent:
    if not WRITER_BASE_URL or not WRITER_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="writer endpoint not configured",
        )
    return OpenAIWriterAgent(OpenAIAgentConfig(
        base_url=WRITER_BASE_URL,
        api_key=WRITER_API_KEY,
        model=WRITER_MODEL,
    ))


def _observation_date(messages: list[Message]) -> str:
    """Session date for the chunk, taken from the first timestamped message."""
    for message in messages:
        if message.timestamp:
            moment = datetime.fromtimestamp(
                message.timestamp / 1000, tz=timezone.utc
            )
            return moment.strftime("%Y-%m-%d")
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _ingest(payload: AddRequest) -> None:
    workspace = _workspace(payload.user_id)
    with _user_lock(payload.user_id):
        ensure_workspace(workspace)
        turns = [(m.role, m.content) for m in payload.messages]
        # refs are the per-turn citation handles the writer footnotes against.
        # Scriptorium accepts `provider/thread/message`, which maps cleanly onto
        # the platform's identifiers: the source session becomes the thread and
        # each message is addressed within its chunk.
        thread = _ref_component(payload.session_id)
        chunk = _ref_component(payload.request_id)
        refs = [
            f"leaderboard/{thread}/{chunk}-{position}"
            for position, _ in enumerate(payload.messages)
        ]
        began = time.monotonic()
        written = write_sessions(
            workspace,
            agent=_agent(),
            sessions=[{
                "observation_date": _observation_date(payload.messages),
                "turns": turns,
                "refs": refs,
            }],
            config=MEMORY_CONFIG,
        )
        # write_sessions reports an audit trail; its closing row carries what
        # the one agent pass cost.
        closing = next(
            (row for row in reversed(written or []) if row.get("tool") == "agent"),
            {},
        )
        _record(
            "add", began,
            turns=closing.get("rounds"),
            input_tokens=closing.get("input_tokens"),
            output_tokens=closing.get("output_tokens"),
            stop=closing.get("reason"),
            messages=len(payload.messages),
            user=payload.user_id,
        )
        # Nothing to invalidate: retrieval rebuilds its index from the
        # workspace on every call, so a committed write is visible to the
        # very next Search, which the contract requires.


@app.post("/add")
async def add(payload: AddRequest, _: None = Depends(_authorize)) -> dict[str, Any]:
    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    if any(not m.content.strip() for m in payload.messages):
        raise HTTPException(status_code=400, detail="content must not be empty")
    try:
        await run_in_threadpool(_ingest, payload)
    except AgentExecutionError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {
        "success": True,
        "request_id": payload.request_id,
        "user_id": payload.user_id,
        "session_id": payload.session_id,
    }


def _passages(text: str) -> list[str]:
    """Split the model's report into one passage per remembered thing.

    Blank lines are what the prompt asks for, but a model that answers in a
    bulleted or numbered list is reporting the same thing in a different
    shape, and collapsing that into one row would throw away the separation
    it just made.
    """
    passages = []
    for block in re.split(r"\n\s*\n", text or ""):
        block = block.strip()
        if not block:
            continue
        lines = [line.strip() for line in block.splitlines()]
        listed = [
            re.sub(r"^(?:[-*\u2022]|\d+[.)])\s+", "", line)
            for line in lines if _LIST_ITEM.match(line)
        ]
        if len(listed) > 1 and len(listed) == len([l for l in lines if l]):
            passages.extend(item for item in listed if item)
        else:
            passages.append(block)
    return passages


_LIST_ITEM = re.compile(r"^(?:[-*\u2022]|\d+[.)])\s+\S")


def _found_memory(text: str) -> list[dict[str, Any]]:
    """What the model reports finding, as the contract's result rows."""
    results = []
    for position, passage in enumerate(_passages(text)):
        results.append({
            "id": f"m{position}",
            "content": passage,
            # The caller ranks nothing; order is the model's judgement.
            "score": round(1.0 - position * 0.01, 4),
            "created_at": _stated_date(passage),
        })
    return results


_DATE_IN_TEXT = re.compile(r"\b(\d{4}(?:-\d{2}){0,2})\b")


def _stated_date(passage: str) -> str:
    """The date the passage carries, if it carries one."""
    found = _DATE_IN_TEXT.search(passage)
    return found.group(1) if found else ""


def _retrieve(payload: SearchRequest) -> list[dict[str, Any]]:
    """Let the model read the memory and report what bears on the query.

    Reading a file, following a footnote to the message behind it,
    searching again with different words — that is how this memory is
    meant to be read. A keyword query over the index gets one shot at
    guessing the wording, and the caller never gets to ask again.
    """
    began = time.monotonic()
    workspace = _workspace(payload.user_id)
    if not workspace.exists():
        return []
    # The first thing a search does is search, every time, for the words in
    # the query. That is not a judgement, so run it here and hand the result
    # to the reading pass as its seed: the model starts from evidence instead
    # of spending a turn getting some, and a weak one that would have stopped
    # after one call has already had it made for it.
    opening = nearest(workspace, payload.query, search_tools=SEARCH_TOOLS)
    outcome = read(
        workspace, payload.query,
        agent=_agent(), model=WRITER_MODEL, config=QUERY_CONFIG, seed=opening,
    )
    rows = _found_memory(outcome.text)
    # What the model read, verbatim, under what it chose to say about it. Its
    # report is a judgement and belongs first; the passages behind that
    # judgement are the memory itself, and the caller answers from what it is
    # handed rather than from what it can ask for next.
    rows += _found_memory("\n\n".join(outcome.passages))
    if not rows:
        # An empty hand scores zero however good the reasoning behind it was.
        rows = _found_memory("\n\n".join(opening))
    kept = _distinct(rows)[: max(1, payload.top_k)]
    _record(
        "search", began,
        turns=outcome.num_turns,
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
        stop=outcome.stop_reason,
        rows=len(kept),
        tool_calls=outcome.tool_calls,
        visible_tokens=outcome.visible_tokens,
        user=payload.user_id,
    )
    return kept


def _distinct(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeats, keeping the first ranking of each passage."""
    seen, kept = set(), []
    for row in rows:
        key = " ".join(row["content"].split())
        if key in seen:
            continue
        seen.add(key)
        row["id"] = f"m{len(kept)}"
        row["score"] = round(1.0 - len(kept) * 0.01, 4)
        kept.append(row)
    return kept


@app.post("/search")
async def search(payload: SearchRequest, _: None = Depends(_authorize)) -> dict[str, Any]:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    data = await run_in_threadpool(_retrieve, payload)
    return {"data": data}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
