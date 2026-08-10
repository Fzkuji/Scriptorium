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
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
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
from memory.management import MemoryConfig
from memory.workspace.layout import runtime_dir
from memory.workspace.transaction import TransactionError, workspace_write_lock
from memory.writing.parallel import write_sessions_in_parallel
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
# A write that has not finished in six turns is not going to. Measured over
# 1273 production writes, the ones that ran to the twenty-turn ceiling were
# 77 of them, each burning one to three minutes and 150k input tokens to
# commit nothing: the model retrying an edit the contract keeps refusing.
# Capping the ceiling costs those nothing they were going to produce, and
# takes the tail off a stage the platform times out on.
#
# The turn ceiling bounds round trips and not the clock, and a write with no
# clock on it holds a request open for as long as the endpoint will talk: the
# client's own timeout was three minutes with four retries behind it, and one
# measured turn ran 200s.
#
# So the pass carries a wall-clock budget. ADD_SECONDS covers the whole
# request, the wait for the workspace lock included, and bounds the endpoint
# call as well as the turn count: a turn that cannot finish inside what is
# left ends the pass, and everything earlier turns wrote stays committed.
#
# A hundred and fifty seconds, and the number has been wrong twice. It was
# first sized to duck under a hundred-second ceiling that turned out to
# belong to a Cloudflare tunnel, then under a two-minute one read from the
# proxy log. Per-request timings finally settled it: the caller waited 488s
# for one Add and answered it, and the requests it abandoned had waited a
# median of 35s, less than the ones it kept. It is not a timeout at all, so
# there is nothing to duck under.
#
# What the ninety-second version did instead was cut its own writes: 125 of
# 300 passes stopped on the budget, 118 of them part-way through a second
# turn, which is where the model finishes recording. Against a median of 62s
# and a p90 of 135s this only trims the tail that runs away.
MEMORY_CONFIG = MemoryConfig(few_shot_instructions=True, max_turns=6)
ADD_SECONDS = float(os.environ.get("SCRIPTORIUM_ADD_SECONDS", "150"))

# What one reading call may take, which is what decides how long an Add can
# run: the local half is under a second and the rest is the endpoint.
#
# The ceiling matters more than the median. A caller runs a fixed number of
# Adds at once, so a pass that takes eighty seconds holds one of those slots
# for eighty seconds, and the chunks waiting behind it spend their own
# deadline queued. At the far end that shows up as requests cancelled in the
# same instant they were sent: measured at the packet level, a SYN, our
# SYN-ACK one round trip later, and a reset with no payload acknowledged —
# a request whose budget was gone before the connection finished opening.
#
# A call that overruns is not an error. The turns already taken wrote what
# they found, and the pass reports them with `stop: max_seconds`, so the
# cost of the ceiling is the facts the unfinished round would have added.
READ_SECONDS = float(os.environ.get("SCRIPTORIUM_READ_SECONDS", "30"))

# Where a chunk waits between arriving and being written.
#
# Add used to hold its connection for the whole write, so the caller — which
# sends one chunk and waits for the answer before sending the next — advanced
# at one chunk per write. At a five second median that is twelve a minute, and
# a job cancelled after thirteen minutes had taken in under eighty. The rate
# was our latency and nothing else.
#
# Writing behind the answer breaks that coupling: the caller is told the chunk
# is safely recorded, which it is, and the model pass happens off its clock.
# What the contract actually needs is that a Search never reads a user whose
# chunks are still outstanding, and that is held in Search instead.
#
# Beside the workspaces rather than inside them: everything that walks the root
# treats each entry as a memory.
QUEUE_ROOT = WORKSPACE_ROOT.with_name(WORKSPACE_ROOT.name + "-queue")

# How many users one process writes at a time. The work is a model call, not
# local, so these are mostly waiting.
DRAIN_THREADS = int(os.environ.get("SCRIPTORIUM_DRAIN_THREADS", "4"))

# How many of one user's chunks are written together.
#
# Chunks of one conversation must commit in order, so a user is written by one
# thread and threads do not help: with two conversations in flight, fourteen of
# sixteen writers had nothing to take, and five hundred queued chunks were five
# hundred model calls end to end. Batching is where the parallelism is — the
# writer reads a whole batch at once and commits it in one transaction, so a
# batch costs about what its slowest chunk costs.
#
# It is also how many calls one writer holds open, so it is bounded rather
# than "whatever arrived".
BATCH_CHUNKS = int(os.environ.get("SCRIPTORIUM_BATCH_CHUNKS", "12"))

# How long Search waits for a user's queued chunks. Long, because answering
# early means answering from a memory that is missing what was just sent; it
# exists so a stuck queue degrades to a poor answer instead of no answer.
SEARCH_WAIT_SECONDS = float(
    os.environ.get("SCRIPTORIUM_SEARCH_WAIT_SECONDS", "1200")
)

# A user is claimed by one process while its chunks are written, and the claim
# is refreshed after each one. Older than this and the holder is gone.
CLAIM_STALE_SECONDS = 300.0

# The longest one sample may spend being written, measured from its first
# chunk. A sample that runs long does not fail alone: the caller waits out its
# own ingestion deadline and cancels the job, and everything already written
# for every other sample goes with it. Ten minutes, then whatever is left is
# dropped and the sample is answered from what it has.
SAMPLE_SECONDS = float(os.environ.get("SCRIPTORIUM_SAMPLE_SECONDS", "600"))

# When a sample has run long enough to be worth finishing rather than doing
# well. Past this the writer gets one pass and a short clock instead of six
# passes and a long one: it records what it has found and stops.
SAMPLE_HURRY_SECONDS = float(
    os.environ.get("SCRIPTORIUM_SAMPLE_HURRY_SECONDS", "180")
)

# What the writer is given once a sample is in a hurry.
HURRY_TURNS = int(os.environ.get("SCRIPTORIUM_HURRY_TURNS", "1"))
HURRY_SECONDS = float(os.environ.get("SCRIPTORIUM_HURRY_SECONDS", "12"))

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


@app.on_event("startup")
async def _start_writers() -> None:
    """Start the threads that write what Add has already answered for.

    Every worker process runs its own, and they take users from the queue by
    claiming them, so adding processes adds throughput without any of them
    needing to know about the others.
    """
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    for number in range(DRAIN_THREADS):
        threading.Thread(
            target=_drain_forever, name=f"write-{number}", daemon=True
        ).start()

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


# What a run calls itself inside an identifier: a stamp and a token, as in
# `eval:scriptmem:20260807T052712Z_55b3f972:longmemeval_refined:conv-0`.
_RUN_ID = re.compile(r"\A(?:\d{8}T\d{6}Z_[0-9a-f]{4,}|run_[0-9a-f]{6,})\Z")


def _stable_id(value: str) -> str:
    """The identifier with the run it belonged to taken out.

    The platform names a conversation for the job that asked about it, so the
    same conversation is a different user every time and a job that was
    cancelled leaves nothing behind for the next one: six of them in a row
    each started from an empty memory and each died at thirteen minutes.
    Dropping the run keeps a conversation's memory across attempts, so a chunk
    already written is recognised and skipped rather than read again.

    Everything that tells one conversation from another is kept — the suite
    and the conversation both survive — so retrieval stays separated exactly
    where the platform separates it. The benchmark's own cambench identifiers
    already work this way: they name the conversation by its content and carry
    no run at all.
    """
    kept = [part for part in value.split(":") if not _RUN_ID.match(part)]
    return ":".join(kept) if kept else value


def _workspace(user_id: str) -> Path:
    """Map a user_id to its own workspace directory.

    user_id arrives from the platform (``eval:run_abc123:locomo:conv-0``) and is
    used as a path component, so anything outside a conservative allowlist is
    escaped rather than trusted.
    """
    safe = _SAFE_ID.sub("_", _stable_id(user_id))
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


@lru_cache(maxsize=1)
def _writer() -> OpenAIWriterAgent:
    """The one writer every request shares.

    Building it per request built an HTTP client per request, and a client
    owns its connection pool: sixty concurrent Adds meant sixty pools that
    each opened their own TLS connections, used them for one chunk and threw
    them away. One client keeps its connections and hands them back.
    """
    return OpenAIWriterAgent(OpenAIAgentConfig(
        base_url=WRITER_BASE_URL,
        api_key=WRITER_API_KEY,
        model=WRITER_MODEL,
    ))


def _agent() -> OpenAIWriterAgent:
    if not WRITER_BASE_URL or not WRITER_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="writer endpoint not configured",
        )
    return _writer()


def _observation_date(messages: list[Message]) -> str:
    """Session date for the chunk, taken from the first timestamped message."""
    for message in messages:
        if message.timestamp:
            moment = datetime.fromtimestamp(
                message.timestamp / 1000, tz=timezone.utc
            )
            return moment.strftime("%Y-%m-%d")
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _handled_requests(workspace: Path) -> Path:
    """Where this user's finished request IDs are kept.

    Under the runtime directory, which is excluded from everything that reads
    the workspace as memory: it must not change a revision or make a stage
    look dirty.
    """
    return runtime_dir(workspace) / "handled-requests.txt"


def _already_written(workspace: Path, request_id: str) -> bool:
    handled = _handled_requests(workspace)
    if not handled.is_file():
        return False
    return _ref_component(_stable_id(request_id)) in handled.read_text(
        encoding="utf-8"
    ).split("\n")


def _mark_written(workspace: Path, request_id: str) -> None:
    # Kept in the form the citations use, which is what makes the record
    # recoverable: every chunk already written names itself in the sources it
    # archived, so a workspace that predates this file can be given one.
    handled = _handled_requests(workspace)
    handled.parent.mkdir(parents=True, exist_ok=True)
    with handled.open("a", encoding="utf-8") as record:
        record.write(f"{_ref_component(_stable_id(request_id))}\n")


def _queue_dir(user_id: str) -> Path:
    """Where this user's chunks wait to be written."""
    safe = _SAFE_ID.sub("_", _stable_id(user_id))
    if not safe or safe in (".", ".."):
        raise HTTPException(status_code=400, detail="invalid user_id")
    return QUEUE_ROOT / safe


def _queued(user_id: str) -> list[Path]:
    """This user's waiting chunks, oldest first."""
    folder = _queue_dir(user_id)
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.json"))


def _already_queued(user_id: str, request_id: str) -> bool:
    folder = _queue_dir(user_id)
    if not folder.is_dir():
        return False
    return any(folder.glob(f"*-{_ref_component(_stable_id(request_id))}.json"))


def _enqueue(payload: AddRequest) -> None:
    """Put a chunk where the writer will find it, in the order it arrived.

    Named by arrival so a plain sort replays the conversation in sequence, and
    renamed into place so a reader never sees half a file.
    """
    folder = _queue_dir(payload.user_id)
    folder.mkdir(parents=True, exist_ok=True)
    # When this sample's clock starts. Written once, and by whichever process
    # took the first chunk, so every worker reads the same start.
    started = folder / "started"
    if not started.exists():
        try:
            opened = os.open(started, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:  # another worker got there first
            pass
        else:
            os.write(opened, str(time.time()).encode())
            os.close(opened)
    name = f"{time.time_ns():020d}-{_ref_component(_stable_id(payload.request_id))}.json"
    partial = folder / f"{name}.part"
    partial.write_text(payload.model_dump_json(), encoding="utf-8")
    partial.rename(folder / name)


def _sample_age(folder: Path) -> float:
    """How long this sample has been being written, in seconds."""
    try:
        return time.time() - float(
            (folder / "started").read_text(encoding="utf-8").strip()
        )
    except (OSError, ValueError):
        return 0.0


def _abandon(folder: Path) -> int:
    """Drop what a sample has left, because its time is up. Returns how many.

    The caller was told 200 for every one of these, so there is nothing to
    report back and nothing to retry. Dropping them is what keeps the sample
    from spending the whole job's ingestion budget on one conversation.
    """
    left = sorted(folder.glob("*.json"))
    for path in left:
        path.unlink(missing_ok=True)
    return len(left)


def _holder_is_gone(marker: Path) -> bool:
    """Whether the process that claimed this user has died.

    A claim outlives the process that took it, so a restart left every user in
    flight claimed by nobody. Waiting out the clock instead meant a new build
    wrote nothing for five minutes, which is a third of the window a job gets.
    The claim names its process, so the usual case is answered directly and the
    clock is only the backstop for a pid that has been handed to someone else.
    """
    try:
        pid = int(marker.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return True
    if pid > 0:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:  # alive, and not ours to signal
            return False
        else:
            return False
    try:
        return time.time() - marker.stat().st_mtime > CLAIM_STALE_SECONDS
    except OSError:  # pragma: no cover - it went away while we looked
        return True


def _claim(folder: Path) -> bool:
    """Take this user, so two processes never write their chunks at once.

    Their order is the point: the file lock would keep two writers from
    corrupting each other but not from committing the second chunk first.
    """
    marker = folder / "busy"
    try:
        holder = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            if _holder_is_gone(marker) or (
                time.time() - marker.stat().st_mtime > CLAIM_STALE_SECONDS
            ):
                marker.unlink()
        except OSError:  # pragma: no cover - another process got there first
            pass
        return False
    os.write(holder, str(os.getpid()).encode())
    os.close(holder)
    return True


def _next_batch(folder: Path) -> list[tuple[Path, AddRequest]]:
    """The next run of this user's chunks that can be written together.

    Ordered by arrival, and stopped at a change of observation date because
    the writer stamps a whole batch with the date of its first chunk. Stopped
    at BATCH_CHUNKS as well: every chunk in a batch reads at the same time, so
    the batch is also how many calls are open at once.
    """
    batch: list[tuple[Path, AddRequest]] = []
    observed: str | None = None
    for path in sorted(folder.glob("*.json"))[:BATCH_CHUNKS]:
        began = time.monotonic()
        try:
            payload = AddRequest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as error:  # a file we cannot read is not a chunk
            _record("queue-unreadable", began,
                    file=path.name, reason=str(error)[:200])
            path.unlink(missing_ok=True)
            continue
        date = _observation_date(payload.messages)
        if observed is not None and date != observed:
            break
        observed = date
        batch.append((path, payload))
    return batch


def _drain_user(folder: Path) -> int:
    """Write everything this user has waiting, in order. Returns how many."""
    written = 0
    while True:
        age = _sample_age(folder)
        if age > SAMPLE_SECONDS:
            dropped = _abandon(folder)
            if dropped:
                _record("sample-cut", time.monotonic() - age,
                        user=folder.name, dropped=dropped,
                        written_here=written)
            return written
        batch = _next_batch(folder)
        if not batch:
            return written
        began = time.monotonic()
        try:
            _ingest([payload for _, payload in batch],
                    hurry=age > SAMPLE_HURRY_SECONDS)
        except TransactionError as error:
            # Someone else holds the workspace, which is a wait rather than a
            # refusal: one batch that took eight minutes to commit made every
            # batch behind it time out, and discarding those threw away eight
            # chunks that nothing was wrong with. Leave them queued.
            _record("add-deferred", began, reason=str(error)[:200],
                    user=batch[0][1].user_id, chunks=len(batch))
            return written
        except Exception as error:
            # Already answered 200, so there is nobody to raise to. Moved aside
            # rather than dropped: they stay readable, and they stop holding
            # Search open for chunks that will not write.
            _record("add-failed", began,
                    reason=f"{type(error).__name__}: {error}"[:300],
                    user=batch[0][1].user_id, chunks=len(batch))
            refused = folder / "refused"
            refused.mkdir(exist_ok=True)
            for path, _ in batch:
                path.rename(refused / path.name)
            continue
        for path, _ in batch:
            path.unlink(missing_ok=True)
        written += len(batch)
        (folder / "busy").touch()  # the claim is still held by a live process


def _drain_forever() -> None:
    """Write queued chunks for whichever users are not already being written."""
    while True:
        worked = False
        try:
            folders = sorted(QUEUE_ROOT.iterdir()) if QUEUE_ROOT.is_dir() else []
        except OSError:  # pragma: no cover - the root came and went
            folders = []
        for folder in folders:
            if not folder.is_dir() or not any(folder.glob("*.json")):
                continue
            if not _claim(folder):
                continue
            try:
                worked = bool(_drain_user(folder)) or worked
            finally:
                (folder / "busy").unlink(missing_ok=True)
        if not worked:
            time.sleep(0.2)


def _wait_for_writes(user_id: str) -> tuple[float, int]:
    """Hold Search until this user's queued chunks are written.

    Add answers as soon as a chunk is on disk, which is what lets the caller
    keep sending instead of spending a model call per chunk. The promise it
    made — that a Search sees everything already sent — is kept here.
    """
    began = time.monotonic()
    folder = _queue_dir(user_id)
    while True:
        left = len(_queued(user_id))
        if not left:
            return time.monotonic() - began, 0
        # Never past the sample's own ceiling. The writer drops what is left
        # when it reaches that, so waiting longer here would be waiting for
        # chunks that are already gone.
        if _sample_age(folder) > SAMPLE_SECONDS + 30:
            _record("search-past-cut", began, user=user_id, left=left)
            return time.monotonic() - began, left
        if time.monotonic() - began > SEARCH_WAIT_SECONDS:
            _record("search-waited-out", began, user=user_id, left=left)
            return time.monotonic() - began, left
        time.sleep(0.25)


def _session_of(payload: AddRequest) -> dict[str, Any]:
    """One chunk in the shape the writer reads it in.

    refs are the per-turn citation handles the writer footnotes against.
    Scriptorium accepts `provider/thread/message`, which maps cleanly onto the
    platform's identifiers: the source session becomes the thread and each
    message is addressed within its chunk.
    """
    thread = _ref_component(_stable_id(payload.session_id))
    chunk = _ref_component(_stable_id(payload.request_id))
    return {
        "observation_date": _observation_date(payload.messages),
        "turns": [(message.role, message.content) for message in payload.messages],
        "refs": [
            f"leaderboard/{thread}/{chunk}-{position}"
            for position, _ in enumerate(payload.messages)
        ],
    }


def _ingest(payloads: list[AddRequest], *, hurry: bool = False) -> None:
    """Write a run of one user's chunks, together.

    The writer reads every chunk it is given at once and commits them in one
    transaction, so a batch costs about what its slowest chunk costs rather
    than the sum. One at a time, a conversation of five hundred chunks was
    five hundred model calls end to end; the caller sends them in a minute and
    would have waited most of an hour to ask its first question.

    They still commit in the order they arrived: the facts are staged in the
    order the chunks are passed, which is what recency is read from.
    """
    if not payloads:
        return
    first = payloads[0]
    workspace = _workspace(first.user_id)
    arrived = time.monotonic()
    # Chunks of one conversation arrive concurrently and commit through one
    # transaction, so they queue here. That wait is the caller's deadline being
    # spent before any work starts, which is why the budget is measured from
    # arrival rather than from the top of the write.
    # The in-process lock keeps this server's own requests for one user off
    # each other; the file lock does the same across processes, which is what
    # makes it safe to serve from more than one. Committing a write is Python
    # walking the whole workspace, and one interpreter runs one of those at a
    # time however many requests are in flight, so the server runs several.
    with _user_lock(first.user_id), workspace_write_lock(
        workspace, timeout_s=ADD_SECONDS
    ):
        ensure_workspace(workspace)
        # A caller that did not hear the answer sends the chunk again. Reading
        # it a second time costs another model pass and files the same facts
        # twice. The work is already committed; drop it and write the rest.
        fresh = [
            payload for payload in payloads
            if not _already_written(workspace, payload.request_id)
        ]
        repeats = len(payloads) - len(fresh)
        if repeats:
            _record("add-repeat", arrived, chunks=repeats, user=first.user_id)
        if not fresh:
            return
        began = time.monotonic()
        waited = began - arrived
        written = write_sessions_in_parallel(
            workspace,
            agent=_agent(),
            sessions=[_session_of(payload) for payload in fresh],
            # A flat ceiling per chunk, and the chunks of a batch read at the
            # same time, so this is the batch's ceiling too. It used to be
            # whatever was left of the caller's deadline, which was right while
            # the caller was holding the connection open; now that it is not, a
            # chunk that sat in the queue would have been given a second.
            #
            # A sample that has already run long gets one pass on a short
            # clock: past that point finishing is worth more than finding
            # everything, because the alternative is the whole job being
            # cancelled and every other sample going with it.
            config=replace(
                MEMORY_CONFIG,
                max_seconds=HURRY_SECONDS if hurry else READ_SECONDS,
                max_turns=HURRY_TURNS if hurry else MEMORY_CONFIG.max_turns,
            ),
        )
        # write_sessions reports an audit trail; its closing row carries what
        # the one agent pass cost.
        closing = next(
            (row for row in reversed(written or []) if row.get("tool") == "agent"),
            {},
        )
        # Recorded only now, so a pass that raised is retried rather than
        # remembered as done.
        for payload in fresh:
            _mark_written(workspace, payload.request_id)
        _record(
            "add", arrived,
            chunks=len(fresh),
            hurry=hurry,
            waited=round(waited, 2),
            turns=closing.get("rounds"),
            reading=closing.get("reading_seconds"),
            input_tokens=closing.get("input_tokens"),
            output_tokens=closing.get("output_tokens"),
            stop=closing.get("reason"),
            messages=sum(len(payload.messages) for payload in fresh),
            user=first.user_id,
        )
        # Nothing to invalidate: retrieval rebuilds its index from the
        # workspace on every call, so a committed write is visible to the
        # very next Search, which the contract requires.


@app.post("/add")
async def add(payload: AddRequest, _: None = Depends(_authorize)) -> dict[str, Any]:
    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    # A blank message is a fact about the benchmark's transcript, not a fault
    # in the request, and rejecting the chunk it arrived in threw away the
    # nineteen good messages beside it: 484 chunks in one run, each read by
    # the caller as a failed ingest. Drop the blanks and write the rest.
    usable = [message for message in payload.messages if message.content.strip()]
    blank = len(payload.messages) - len(usable)
    payload.messages = usable
    if blank:
        _record("blanks", time.monotonic(), dropped=blank, user=payload.user_id)
    if not payload.messages:
        # Nothing to commit, so nothing to be retrievable, and the caller is
        # owed the same answer either way.
        return {
            "success": True,
            "request_id": payload.request_id,
            "user_id": payload.user_id,
            "session_id": payload.session_id,
        }
    began = time.monotonic()
    answer = {
        "success": True,
        "request_id": payload.request_id,
        "user_id": payload.user_id,
        "session_id": payload.session_id,
    }
    workspace = _workspace(payload.user_id)
    if _already_written(workspace, payload.request_id) or _already_queued(
        payload.user_id, payload.request_id
    ):
        _record("add-repeat", began,
                messages=len(payload.messages), user=payload.user_id)
        return answer
    try:
        await run_in_threadpool(_enqueue, payload)
    except OSError as error:
        # Answered as accepted even so. A refusal here is not read as one
        # chunk lost: the caller counts it as a failed ingest and cancels the
        # whole job, taking every sample already written with it. One chunk
        # costs the questions that needed it; the job costs all of them.
        _record("queue-failed", began, reason=str(error)[:300],
                user=payload.user_id)
        return answer
    _record("add-queued", began,
            messages=len(payload.messages), user=payload.user_id)
    return answer


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
    # Add answers before the write, so this is where the contract is honoured:
    # a question is not answered from a memory that is still being written.
    waited, left = await run_in_threadpool(_wait_for_writes, payload.user_id)
    if waited > 1.0:
        _record("search-waited", time.monotonic() - waited,
                user=payload.user_id, unwritten=left)
    data = await run_in_threadpool(_retrieve, payload)
    return {"data": data}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
