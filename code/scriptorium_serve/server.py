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

from memory.agent_runtime import (
    AgentExecutionError,
    OpenAIAgentConfig,
    OpenAIWriterAgent,
)
from memory.management import MemoryConfig, write_sessions
from memory.retrieval import QueryConfig
from memory.retrieval.context import initialize_context
from memory.retrieval.tool_server import RetrievalToolState, retrieval_tools
from memory.retrieval.views import memory_files
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
QUERY_CONFIG = QueryConfig(max_turns=8, verify_sources=False)

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


class _Runtime:
    """The little the retrieval agent needs from a benchmark Runtime.

    `collect_answer` was written for the harness and reaches for the
    model, the agent and a usage log. Here there is no run to account
    for, so the log goes nowhere.
    """

    def __init__(self, agent: Any, model: str, query_config: QueryConfig):
        self.agent = agent
        self.model = model
        self.query_config = query_config

    def log_agent_result(self, result: Any, *, phase: str = "") -> None:
        return None


FIND_MEMORY = (
    "Find everything in this memory workspace that bears on the request "
    "below, and return it.\n\n"
    "You are not answering. Something else will do that, and it sees only "
    "what you return — not this workspace, not your reasoning. So return "
    "the memory itself: the sentences as they are written, each with the "
    "date its footnote carries and enough of its heading or subject that "
    "it still means something on its own.\n\n"
    "Search more than one way before deciding nothing is there; wording in "
    "memory rarely matches the wording of a request. Where memory records "
    "a change, return both what it was and what it became. Return nothing "
    "at all rather than something you did not find."
)


def _passages(text: str) -> list[str]:
    """Split the model's report into one passage per remembered thing."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text or "")]
    return [block for block in blocks if block]


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
    workspace = _workspace(payload.user_id)
    if not workspace.exists():
        return []
    files = memory_files(workspace, "native", include_recent=True)
    prompt, trace, evidence, initial_tokens = initialize_context(
        memory_dir=workspace,
        files=files,
        condition="native",
        item={"question": payload.query, "question_date": ""},
        verify_sources=False,
        model=WRITER_MODEL,
    )
    state = RetrievalToolState(
        trace=trace,
        evidence=evidence,
        model=WRITER_MODEL,
        visible_tokens=initial_tokens,
    )
    runtime = _Runtime(_agent(), WRITER_MODEL, QUERY_CONFIG)
    result = runtime.agent.run(
        prompt=prompt,
        system_prompt=FIND_MEMORY,
        cwd=workspace,
        tools=retrieval_tools(
            runtime,
            memory_dir=workspace,
            files=files,
            condition="native",
            include_recent=True,
            state=state,
            search_tools=QUERY_CONFIG.search_tools,
        ),
        max_turns=QUERY_CONFIG.max_turns,
        max_budget_usd=QUERY_CONFIG.max_budget_usd,
    )
    reported = re.sub(r"</?answer>", "", result.text or "").strip()
    return _found_memory(reported)[: max(1, payload.top_k)]


@app.post("/search")
async def search(payload: SearchRequest, _: None = Depends(_authorize)) -> dict[str, Any]:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    data = await run_in_threadpool(_retrieve, payload)
    return {"data": data}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
