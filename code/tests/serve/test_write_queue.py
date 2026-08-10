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


def chunk(
    request_id: str,
    user: str = "u1",
    text: str = "hello",
    stamp: int | None = None,
) -> AddRequest:
    return AddRequest(
        request_id=request_id, user_id=user, session_id="s1",
        messages=[Message(role="user", content=text, timestamp=stamp)],
    )


DAY = 86_400_000  # one day, in the milliseconds the platform stamps with


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


def test_a_claim_left_by_a_restart_is_taken_back_at_once(queue: Path) -> None:
    """A restart must not stall writing until the claim clock runs out.

    Killing a build left every user in flight claimed by a process that no
    longer exists, and the next build wrote nothing until those claims aged
    out: five minutes of a thirteen minute window, spent on nothing.
    """
    folder = queue / "u1"
    folder.mkdir(parents=True)
    # A pid that is not running. Claimed just now, so the clock says nothing.
    (folder / "busy").write_text("2147483646", encoding="utf-8")

    assert not server._claim(folder), "the first attempt clears it"
    assert server._claim(folder), "and the next one takes it straight away"


def test_a_claim_held_by_a_live_writer_is_left_alone(queue: Path) -> None:
    folder = queue / "u1"
    folder.mkdir(parents=True)

    assert server._claim(folder)  # ours, and we are running

    assert not server._claim(folder)
    assert (folder / "busy").exists(), "not cleared while its holder lives"


def test_a_sample_that_runs_long_is_cut(
    queue: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One slow sample must not spend the whole job's ingestion budget.

    The caller cancels the job when ingestion runs past its own deadline, and
    everything already written for every other sample goes with it. Ten
    minutes, then whatever is left is dropped and the sample answers from what
    it has.
    """
    monkeypatch.setattr(server, "SAMPLE_SECONDS", 600.0)
    for number in range(4):
        server._enqueue(chunk(f"r{number}"))
        time.sleep(0.002)
    folder = queue / "u1"
    (folder / "started").write_text(str(time.time() - 601), encoding="utf-8")
    monkeypatch.setattr(server, "_ingest", lambda batch, hurry=False: None)

    written = server._drain_user(folder)

    assert written == 0, "its time was up before this pass"
    assert server._queued("u1") == [], "and nothing is left holding Search open"


def test_a_sample_still_inside_its_time_is_written(
    queue: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server._enqueue(chunk("r1"))
    (queue / "u1" / "started").write_text(str(time.time()), encoding="utf-8")
    monkeypatch.setattr(server, "_ingest", lambda batch, hurry=False: None)

    assert server._drain_user(queue / "u1") == 1


def test_a_long_sample_is_told_to_wrap_up(
    queue: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the hurry mark the writer gets one pass and a short clock."""
    monkeypatch.setattr(server, "SAMPLE_HURRY_SECONDS", 180.0)
    server._enqueue(chunk("r1"))
    folder = queue / "u1"
    (folder / "started").write_text(str(time.time() - 181), encoding="utf-8")
    hurried: list[bool] = []
    monkeypatch.setattr(
        server, "_ingest", lambda batch, hurry=False: hurried.append(hurry)
    )

    server._drain_user(folder)

    assert hurried == [True]


def test_a_fresh_sample_is_not_hurried(
    queue: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server._enqueue(chunk("r1"))
    hurried: list[bool] = []
    monkeypatch.setattr(
        server, "_ingest", lambda batch, hurry=False: hurried.append(hurry)
    )

    server._drain_user(queue / "u1")

    assert hurried == [False]


def test_the_clock_starts_at_the_first_chunk_and_does_not_restart(
    queue: Path,
) -> None:
    server._enqueue(chunk("r1"))
    first = (queue / "u1" / "started").read_text(encoding="utf-8")
    time.sleep(0.01)

    server._enqueue(chunk("r2"))

    assert (queue / "u1" / "started").read_text(encoding="utf-8") == first


def test_search_does_not_wait_past_the_cut(
    queue: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Waiting longer than the sample lives is waiting for nothing."""
    monkeypatch.setattr(server, "SAMPLE_SECONDS", 600.0)
    monkeypatch.setattr(server, "SEARCH_WAIT_SECONDS", 9999.0)
    server._enqueue(chunk("r1"))
    (queue / "u1" / "started").write_text(str(time.time() - 700), encoding="utf-8")

    waited, left = server._wait_for_writes("u1")

    assert left == 1
    assert waited < 3, "it did not sit out the search budget"


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

    def refuse(batch: list[AddRequest], hurry: bool = False) -> None:
        raise RuntimeError("the writer said no")

    monkeypatch.setattr(server, "_ingest", refuse)
    server._drain_user(folder)

    assert server._queued("u1") == [], "Search is no longer waiting on it"
    assert list((folder / "refused").glob("*.json")), "and it is still readable"


def test_a_written_chunk_leaves_the_queue(
    queue: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server._enqueue(chunk("r1"))
    time.sleep(0.002)
    server._enqueue(chunk("r2"))
    seen: list[list[str]] = []
    monkeypatch.setattr(
        server, "_ingest",
        lambda batch, hurry=False: seen.append([p.request_id for p in batch])
    )

    written = server._drain_user(queue / "u1")

    assert written == 2
    assert seen == [["r1", "r2"]], "written together, in order"
    assert server._queued("u1") == []


def test_a_batch_is_bounded(queue: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every chunk in a batch reads at once, so the batch is the call count."""
    monkeypatch.setattr(server, "BATCH_CHUNKS", 3)
    for number in range(7):
        server._enqueue(chunk(f"r{number}"))
        time.sleep(0.002)
    seen: list[list[str]] = []
    monkeypatch.setattr(
        server, "_ingest",
        lambda batch, hurry=False: seen.append([p.request_id for p in batch])
    )

    written = server._drain_user(queue / "u1")

    assert written == 7
    assert seen == [["r0", "r1", "r2"], ["r3", "r4", "r5"], ["r6"]]


def test_a_batch_stops_at_a_change_of_day(
    queue: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer stamps a batch with the date of its first chunk.

    Carrying one day's date onto the next day's facts would put them on the
    wrong date in a memory whose questions are largely about when.
    """
    for number, stamp in enumerate([DAY, DAY, 2 * DAY, 2 * DAY]):
        server._enqueue(chunk(f"r{number}", stamp=stamp))
        time.sleep(0.002)
    seen: list[list[str]] = []
    monkeypatch.setattr(
        server, "_ingest",
        lambda batch, hurry=False: seen.append([p.request_id for p in batch])
    )

    server._drain_user(queue / "u1")

    assert seen == [["r0", "r1"], ["r2", "r3"]]
    dates = {server._observation_date(p.messages) for p in [chunk("x", stamp=DAY)]}
    assert dates, "the date is what the grouping is on"
