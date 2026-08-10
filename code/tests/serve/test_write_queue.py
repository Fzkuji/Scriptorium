"""Add answers before the write; Search waits for it.

The caller sends one chunk and waits for the answer before sending the next,
so its rate was our write time: twelve chunks a minute at a five second
median, and a job cancelled at thirteen minutes had taken in under eighty.
Answering as soon as the chunk is on disk breaks that coupling. What the
contract needs in exchange is that a question is never answered from a memory
that is still being written, which is Search's job now.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from scriptorium_serve import server
from scriptorium_serve.server import AddRequest, Message


@pytest.fixture
def queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "queue"
    monkeypatch.setattr(server, "QUEUE_ROOT", root)
    return root


def chunk(request_id: str, user: str = "u1", text: str = "hello") -> AddRequest:
    return AddRequest(
        request_id=request_id, user_id=user, session_id="s1",
        messages=[Message(role="user", content=text)],
    )


def test_a_queued_chunk_is_waiting_for_its_user(queue: Path) -> None:
    server._enqueue(chunk("r1"))

    assert len(server._queued("u1")) == 1
    assert server._queued("u2") == [], "queues do not leak between users"


def test_chunks_are_replayed_in_the_order_they_arrived(queue: Path) -> None:
    for number in range(5):
        server._enqueue(chunk(f"r{number}"))
        time.sleep(0.002)  # the name carries arrival, so they must differ

    order = [path.name.split("-", 1)[1] for path in server._queued("u1")]

    assert order == [f"r{number}.json" for number in range(5)]


def test_a_repeat_is_recognised_while_it_is_still_waiting(queue: Path) -> None:
    """The record of what was written is only written after the write.

    Between answering and writing, that record does not yet name the chunk, so
    a caller that resends in the meantime would be queued a second time and
    the same facts filed twice.
    """
    server._enqueue(chunk("r1"))

    assert server._already_queued("u1", "r1")
    assert not server._already_queued("u1", "r2")
    assert not server._already_queued("u2", "r1")


def test_only_one_writer_holds_a_user(queue: Path) -> None:
    folder = queue / "u1"
    folder.mkdir(parents=True)

    assert server._claim(folder)
    assert not server._claim(folder), "a second writer must not take it"

    (folder / "busy").unlink()
    assert server._claim(folder), "released, so it can be taken again"


def test_a_claim_left_by_a_dead_writer_is_taken_back(
    queue: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = queue / "u1"
    folder.mkdir(parents=True)
    server._claim(folder)
    # Older than a live writer would ever leave it: each written chunk touches
    # the marker, and one chunk is capped well below this.
    stale = time.time() - server.CLAIM_STALE_SECONDS - 1
    import os

    os.utime(folder / "busy", (stale, stale))

    assert not server._claim(folder), "the first attempt only clears it"
    assert server._claim(folder), "and the next one takes it"


def test_search_waits_for_what_is_still_queued(
    queue: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "SEARCH_WAIT_SECONDS", 5.0)
    server._enqueue(chunk("r1"))
    done = threading.Event()

    def answer() -> None:
        server._wait_for_writes("u1")
        done.set()

    threading.Thread(target=answer, daemon=True).start()
    assert not done.wait(0.5), "the chunk is still waiting, so Search is too"

    for path in server._queued("u1"):  # the writer gets to it
        path.unlink()

    assert done.wait(3), "and Search goes on once nothing is left"


def test_search_gives_up_rather_than_never_answering(
    queue: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queue that will not drain must degrade, not hang.

    Answering from an incomplete memory scores badly on one question.
    Answering never fails the run.
    """
    monkeypatch.setattr(server, "SEARCH_WAIT_SECONDS", 0.3)
    server._enqueue(chunk("r1"))

    waited, left = server._wait_for_writes("u1")

    assert left == 1, "it reports what it gave up on"
    assert 0.3 <= waited < 3


def test_a_chunk_that_cannot_be_written_stops_holding_search_open(
    queue: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused write is moved aside, not retried until the run ends."""
    server._enqueue(chunk("r1"))
    folder = queue / "u1"

    def refuse(payload: AddRequest) -> None:
        raise RuntimeError("the writer said no")

    monkeypatch.setattr(server, "_ingest", refuse)
    server._drain_user(folder)

    assert server._queued("u1") == [], "Search is no longer waiting on it"
    assert list((folder / "refused").glob("*.json")), "and it is still readable"


def test_a_written_chunk_leaves_the_queue(
    queue: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server._enqueue(chunk("r1"))
    server._enqueue(chunk("r2"))
    seen: list[str] = []
    monkeypatch.setattr(server, "_ingest", lambda p: seen.append(p.request_id))

    written = server._drain_user(queue / "u1")

    assert written == 2
    assert seen == ["r1", "r2"], "in order"
    assert server._queued("u1") == []
