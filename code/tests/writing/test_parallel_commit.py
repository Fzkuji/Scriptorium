"""The parallel writer commits its facts once, and a bad one costs only itself.

Committing validates the whole workspace — every topic parsed, every source
reference re-checked — so doing it per fact made one Add cost the workspace
once per fact it recorded, and grow as the memory grew. These pin the two
things that change: the batch commits together, and a refusal still leaves the
good facts written.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory.agent_runtime import AgentResult
from memory.config import MemoryConfig
from memory.writing.parallel import write_sessions_in_parallel


class _Answers:
    """Returns the same `remember` calls for every group it is asked about."""

    def __init__(self, facts: list[dict]):
        self.facts = facts
        self.calls = 0

    def run(self, **_: object) -> AgentResult:
        self.calls += 1
        # Only the first group reports; the rest find nothing, so the test
        # counts facts rather than groups.
        turns = [
            {"tool": "remember", "arguments": json.dumps(fact)}
            for fact in (self.facts if self.calls == 1 else [])
        ]
        return AgentResult(
            text="", structured_output=None, num_turns=1, turns=turns,
            input_tokens=1, output_tokens=1, cache_creation_input_tokens=0,
            cache_read_input_tokens=0, anthropic_equivalent_cost_usd=None,
            duration_ms=0, duration_api_ms=0, stop_reason="stop", session_id="",
        )


def _fact(number: int, **overrides: object) -> dict:
    fact = {
        "subject": f"Person {number}",
        "kind": "person",
        "fact": f"Person {number} moved to Berlin in March.",
        "sources": ["leaderboard/thread/chunk-0"],
    }
    fact.update(overrides)
    return fact


def _session() -> dict:
    return {
        "observation_date": "2026-05-08",
        "turns": [("user", "Something was said.")],
        "refs": ["leaderboard/thread/chunk-0"],
    }


def _audit_for(memory_dir: Path, facts: list[dict]) -> list[dict]:
    return write_sessions_in_parallel(
        memory_dir,
        agent=_Answers(facts),
        sessions=[_session()],
        config=MemoryConfig(max_seconds=30),
    )


def test_every_fact_lands_in_one_commit(tmp_path: Path) -> None:
    audit = _audit_for(tmp_path, [_fact(number) for number in range(4)])

    commits = [row for row in audit if row.get("tool") == "commit"]
    assert len(commits) == 1, "one Add, one transaction"
    assert commits[0]["status"] == "ok"
    assert commits[0]["count"] == 4

    written = " ".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "topics").rglob("*.md")
    )
    for number in range(4):
        assert f"Person {number} moved to Berlin" in written


def test_a_refused_fact_does_not_cost_the_others(tmp_path: Path) -> None:
    # A source the workspace cannot resolve is refused when the batch commits,
    # not when it is staged, which is the case the retry exists for: the whole
    # transaction rolls back, and only re-recording one fact at a time leaves
    # the good ones written.
    facts = [_fact(0), _fact(1, sources=["nowhere/at/all"]), _fact(2)]

    audit = _audit_for(tmp_path, facts)

    written = " ".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "topics").rglob("*.md")
    )
    assert "Person 0 moved to Berlin" in written
    assert "Person 2 moved to Berlin" in written
    assert "Person 1 moved to Berlin" not in written
    closing = next(row for row in reversed(audit) if row.get("tool") == "agent")
    assert closing["refused_facts"] == 1, "the refusal is still reported"
    assert closing["status"] == "rejected"


def test_finding_a_refused_fact_costs_a_handful_of_commits_not_one_each(
    tmp_path: Path,
) -> None:
    """Halving, because a commit costs the whole workspace however small it is.

    Retrying every fact on its own made one refusal cost one full validation
    per fact: on a memory grown to six thousand topics, a batch spent eight and
    a half minutes holding the write lock against nineteen seconds of reading,
    and everything queued behind it timed out.
    """
    facts = [_fact(number) for number in range(16)]
    facts[9] = _fact(9, sources=["nowhere/at/all"])

    audit = _audit_for(tmp_path, facts)

    commits = [row for row in audit if row.get("tool") == "commit"]
    assert len(commits) <= 12, f"halving, not one per fact: {len(commits)}"

    written = " ".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "topics").rglob("*.md")
    )
    assert "Person 9 moved to Berlin" not in written
    for number in [n for n in range(16) if n != 9]:
        assert f"Person {number} moved to Berlin" in written, number
