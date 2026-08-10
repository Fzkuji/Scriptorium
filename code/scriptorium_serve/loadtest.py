"""Drive the service the way the leaderboard drives it, before it does.

`smoke` proves the contract on one request. It cannot see what only appears
under load and over time: a call that outlives its timeout, latency that grows
with the workspace, a tail the caller hangs up on. Three runs died on exactly
that, each one found by launching the real evaluation and watching it fail two
minutes later.

So this replays recorded traffic at the platform's concurrency for as long as
you ask, and reports the distribution rather than a pass mark:

    PYTHONPATH=code python3 -m scriptorium_serve.loadtest \
      --base-url http://127.0.0.1:8600 --token "$TOKEN" \
      --workers 16 --minutes 10

Payloads come from the `sources/` archives of workspaces the platform already
wrote, so chunk sizes and message counts are the real ones. Every request goes
to a `loadtest-` user so it lands in throwaway workspaces; `--clean` removes
them afterwards.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# How the runtime archives one message it was sent: `[date] role: content`.
_MESSAGE = re.compile(r"^\[\d{4}-\d\d-\d\d\]\s*([A-Za-z0-9_ -]{1,40}?):\s*(.+)$")


def _chunks_from(archive: Path, per_chunk: int) -> list[list[dict]]:
    """Read one archived source file back into Add-shaped message chunks."""
    messages = []
    for line in archive.read_text(errors="replace").splitlines():
        found = _MESSAGE.match(line.strip())
        if found:
            speaker, content = found.group(1).strip().lower(), found.group(2).strip()
            messages.append({
                "role": "assistant" if speaker == "assistant" else "user",
                "content": content,
            })
    return [
        messages[at:at + per_chunk] for at in range(0, len(messages), per_chunk)
    ]


def corpus(workspaces: Path, per_chunk: int, limit: int) -> list[list[dict]]:
    """Sample across every archive rather than draining the first few.

    Benchmarks differ by an order of magnitude in how much text one chunk
    carries, and it is the heavy ones that produce the long writes. Reading
    files in order until the limit is reached would fill the corpus from
    whichever benchmark sorts first and hide them.
    """
    per_archive: list[list[list[dict]]] = []
    for archive in sorted(workspaces.rglob("sources/**/*.md")):
        found = [c for c in _chunks_from(archive, per_chunk) if c]
        if found:
            per_archive.append(found)
    chunks: list[list[dict]] = []
    depth = 0
    while per_archive and len(chunks) < limit:
        taken = False
        for found in per_archive:
            if depth < len(found):
                chunks.append(found[depth])
                taken = True
                if len(chunks) >= limit:
                    break
        if not taken:
            break
        depth += 1
    return chunks


def _post(base: str, path: str, token: str, body: dict, timeout: float) -> tuple[int, float, str]:
    request = urllib.request.Request(
        f"{base.rstrip('/')}{path}",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    began = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return response.status, time.monotonic() - began, ""
    except urllib.error.HTTPError as error:
        return error.code, time.monotonic() - began, error.read()[:200].decode(errors="replace")
    except Exception as error:  # a hang-up is the failure being hunted
        return 0, time.monotonic() - began, f"{type(error).__name__}: {error}"


def main(argv: list[str] | None = None) -> int:
    parse = argparse.ArgumentParser(description=__doc__)
    parse.add_argument("--base-url", required=True)
    parse.add_argument("--token", required=True)
    parse.add_argument("--workers", type=int, default=16,
                       help="concurrent Add requests; the platform uses 16")
    parse.add_argument("--minutes", type=float, default=10.0)
    parse.add_argument("--conversations", type=int, default=8,
                       help="distinct user_ids, so writes to one workspace queue "
                            "the way they do in a run")
    parse.add_argument("--per-chunk", type=int, default=20,
                       help="messages per Add; the platform sends at most 20")
    parse.add_argument("--ceiling", type=float, default=120.0,
                       help="seconds above which a request counts as a tail risk")
    parse.add_argument("--min-chars", type=int, default=0,
                       help="drop chunks lighter than this. The first run of "
                            "this tool sampled a median of 1673 characters and "
                            "reported a 24.8s median, while production was "
                            "writing 8k-character chunks in 45s and hanging on "
                            "the tail; a test that never sends the heavy ones "
                            "cannot see what fails")
    parse.add_argument("--workspaces", type=Path,
                       default=Path("/Users/fzkuji/scriptorium-runs/live-workspaces"))
    parse.add_argument("--clean", action="store_true",
                       help="delete the loadtest workspaces when done")
    args = parse.parse_args(argv)

    chunks = corpus(args.workspaces, args.per_chunk, limit=1500)
    if args.min_chars:
        chunks = [c for c in chunks
                  if sum(len(m['content']) for m in c) >= args.min_chars]
    if not chunks:
        print(f"no archived sources under {args.workspaces}", file=sys.stderr)
        return 2
    sizes = sorted(sum(len(m["content"]) for m in c) for c in chunks)
    print(f"{len(chunks)} recorded chunks, "
          f"median {statistics.median(len(c) for c in chunks):.0f} messages, "
          f"chars median {sizes[len(sizes) // 2]} p90 {sizes[int(len(sizes) * .9)]} "
          f"max {sizes[-1]}")

    stamp = time.strftime("%H%M%S")
    deadline = time.monotonic() + args.minutes * 60
    results: list[tuple[int, float, str]] = []
    guard = threading.Lock()
    counter = iter(range(1_000_000))

    def worker(slot: int) -> None:
        picker = random.Random(slot)
        while time.monotonic() < deadline:
            with guard:
                sequence = next(counter)
            conversation = sequence % args.conversations
            messages = picker.choice(chunks)
            outcome = _post(args.base_url, "/add", args.token, {
                "request_id": f"loadtest-{stamp}-{sequence}",
                "user_id": f"loadtest-{stamp}:conv-{conversation}",
                "session_id": f"loadtest-session-{conversation}",
                # Every fourth chunk carries a blank turn, because the
                # benchmark's transcripts do and rejecting the chunk over one
                # cost a run a fifth of its writes.
                "messages": [
                    {**message, "timestamp": 1_700_000_000_000 + sequence * 86_400_000}
                    for message in messages
                ] + ([{"role": "assistant", "content": "  ",
                       "timestamp": 1_700_000_000_000}] if sequence % 4 == 0 else []),
            }, timeout=args.ceiling * 5)
            with guard:
                results.append(outcome)
                status, seconds, detail = outcome
                if status != 200:
                    print(f"  !! {status} after {seconds:.1f}s {detail}", flush=True)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for slot in range(args.workers):
            pool.submit(worker, slot)
        while time.monotonic() < deadline:
            time.sleep(30)
            with guard:
                done = len(results)
                bad = sum(1 for status, _, _ in results if status != 200)
            elapsed = (time.monotonic() - started) / 60
            print(f"  {elapsed:4.1f}min  {done} adds  {bad} failed  "
                  f"{done / max(elapsed, 0.01):.0f}/min", flush=True)

    total = len(results)
    failed = [r for r in results if r[0] != 200]
    times = sorted(seconds for status, seconds, _ in results if status == 200)
    print(f"\n{total} adds, {len(failed)} failed, "
          f"{total / args.minutes:.0f}/min sustained at {args.workers} workers")
    if times:
        print(f"latency: median={times[len(times) // 2]:.1f}s "
              f"p90={times[int(len(times) * .9)]:.1f}s "
              f"p99={times[int(len(times) * .99)]:.1f}s "
              f"max={times[-1]:.1f}s")
        over = sum(1 for t in times if t > args.ceiling)
        print(f"over {args.ceiling:.0f}s: {over} ({over / len(times) * 100:.1f}%)")
    for status, seconds, detail in failed[:10]:
        print(f"  failed: {status} after {seconds:.1f}s {detail}")

    if args.clean:
        root = args.workspaces
        for spent in root.glob(f"loadtest-{stamp}*"):
            shutil.rmtree(spent, ignore_errors=True)
        print(f"removed loadtest workspaces under {root}")

    # A tail above the ceiling is what ends a run, so it is the exit code.
    return 1 if failed or (times and times[-1] > args.ceiling) else 0


if __name__ == "__main__":
    raise SystemExit(main())
