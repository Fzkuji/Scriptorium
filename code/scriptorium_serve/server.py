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

import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent_runtime import (
    AgentExecutionError,
    OpenAIAgentConfig,
    OpenAIWriterAgent,
)
from src.management import MemoryConfig, write_sessions
from src.retrieval.bm25 import MemoryBM25Index
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
MEMORY_CONFIG = MemoryConfig(writer_shell_examples=True)

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

# Writes to one workspace are serialised by Scriptorium's own file lock, but the
# platform fans out 64 workers and may hit the same user_id concurrently. A
# per-user lock keeps those requests queued in-process instead of contending on
# the file lock and burning the request timeout.
# ponytail: dict of locks, never evicted. Bounded by user_id count per run.
_user_locks: dict[str, threading.Lock] = {}
_user_locks_guard = threading.Lock()

# BM25 indexes are expensive to rebuild (O(corpus) per refresh), so hold one per
# workspace. MemoryBM25Index is internally RLock-guarded.
_indexes: dict[str, MemoryBM25Index] = {}
_indexes_guard = threading.Lock()

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


def _index(workspace: Path) -> MemoryBM25Index:
    key = str(workspace)
    with _indexes_guard:
        index = _indexes.get(key)
        if index is None:
            index = MemoryBM25Index(workspace, persist=True)
            _indexes[key] = index
        return index


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
        write_sessions(
            workspace,
            agent=_agent(),
            sessions=[{
                "observation_date": _observation_date(payload.messages),
                "turns": turns,
                "refs": refs,
            }],
            config=MEMORY_CONFIG,
        )
        # The write is committed; drop the cached index so the very next Search
        # observes it. The contract requires Add to be retrievable on return.
        with _indexes_guard:
            _indexes.pop(str(workspace), None)


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


def _retrieve(payload: SearchRequest) -> list[dict[str, Any]]:
    workspace = _workspace(payload.user_id)
    if not workspace.exists():
        return []
    hits = _index(workspace).search(payload.query, top_k=payload.top_k)
    results = []
    for hit in hits:
        # `date` is the event's own date when the writer recorded one.
        created_at = hit.get("date") or ""
        results.append({
            "id": hit["event_id"],
            "content": hit["content"],
            "score": hit["final_score"],
            "created_at": created_at,
        })
    return results


@app.post("/search")
async def search(payload: SearchRequest, _: None = Depends(_authorize)) -> dict[str, Any]:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    data = await run_in_threadpool(_retrieve, payload)
    return {"data": data}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
