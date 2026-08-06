"""Live smoke check against a running Add / Search service.

Exercises the platform's own contract end to end: Add a chunk, confirm the write
is retrievable the moment Add returns, and confirm one user's memory never
surfaces for another.

    python -m scriptorium_serve.smoke --base-url http://127.0.0.1:8000

Run it before submitting to the leaderboard; it is the same sequence the
platform's smoke stage performs.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid

import httpx

CONVERSATION = [
    ("user", "I started learning saxophone in May 2023.", 1683504000000),
    ("assistant", "That's great! How often do you practise?", 1683504060000),
    ("user", "Every morning before work, about an hour.", 1683504120000),
]


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", default=None)
    parser.add_argument("--timeout", type=float, default=1200.0)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    headers = _headers(args.token)
    run = uuid.uuid4().hex[:8]
    user = f"eval:smoke_{run}:locomo:conv-0"
    other = f"eval:smoke_{run}:locomo:conv-1"
    failures: list[str] = []

    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(f"{base}/health")
        if health.status_code // 100 != 2:
            print(f"FAIL health -> {health.status_code}")
            return 1
        print(f"ok   health -> {health.status_code}")

        started = time.time()
        add = client.post(f"{base}/add", headers=headers, json={
            "request_id": f"eval:smoke_{run}:locomo_refined:conv-0:chunk-0",
            "messages": [
                {"role": role, "content": content, "timestamp": stamp}
                for role, content, stamp in CONVERSATION
            ],
            "user_id": user,
            "session_id": f"eval:smoke_{run}:sample:0",
        })
        elapsed = time.time() - started
        if add.status_code != 200:
            print(f"FAIL add -> {add.status_code} {add.text[:300]}")
            return 1
        body = add.json()
        print(f"ok   add -> 200 in {elapsed:.1f}s")

        if body.get("success") is not True:
            failures.append("add.success must be boolean true")
        for field, expected in (
            ("request_id", f"eval:smoke_{run}:locomo_refined:conv-0:chunk-0"),
            ("user_id", user),
            ("session_id", f"eval:smoke_{run}:sample:0"),
        ):
            if body.get(field) != expected:
                failures.append(f"add.{field} must echo the request exactly")
        if elapsed > 1200:
            failures.append(f"add took {elapsed:.0f}s, over the 1200s timeout")

        # The contract requires the write to be retrievable as soon as Add
        # returns, so this runs with no delay in between.
        search = client.post(f"{base}/search", headers=headers, json={
            "query": "When did the user start learning saxophone?",
            "user_id": user,
            "top_k": 100,
        })
        if search.status_code != 200:
            print(f"FAIL search -> {search.status_code} {search.text[:300]}")
            return 1
        data = search.json().get("data")
        if not isinstance(data, list):
            print("FAIL search.data must be an array")
            return 1
        print(f"ok   search -> 200, {len(data)} memories")

        if not data:
            failures.append("search returned nothing for a memory just added")
        if len(data) > 100:
            failures.append("search returned more than top_k results")
        for hit in data:
            if not isinstance(hit.get("id"), str) or not hit["id"]:
                failures.append("each memory needs a stable string id")
                break
            if not isinstance(hit.get("content"), str) or not hit["content"]:
                failures.append("each memory needs non-empty content")
                break
        scores = [h["score"] for h in data if isinstance(h.get("score"), (int, float))]
        if scores != sorted(scores, reverse=True):
            failures.append("memories must be ordered by descending relevance")

        isolated = client.post(f"{base}/search", headers=headers, json={
            "query": "saxophone", "user_id": other, "top_k": 100,
        })
        if isolated.status_code == 200 and isolated.json().get("data"):
            failures.append("another user_id must not see this user's memory")
        else:
            print("ok   isolation -> other user_id sees nothing")

    if failures:
        print()
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("\nAll contract checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
