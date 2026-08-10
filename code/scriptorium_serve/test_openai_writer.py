"""End-to-end Add through the OpenAI writer, against a stubbed endpoint.

The stub stands in for gpt-4o-mini: it issues one `remember` call that records a
memory line, then finishes. What this proves is the wiring — tool schemas reach
the model, tool calls dispatch into the workspace, the write commits, and the
result is retrievable through Search, which is what Add's contract demands.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

WORKSPACES = tempfile.mkdtemp(prefix="scriptorium-writer-")
os.environ["SCRIPTORIUM_WORKSPACES"] = WORKSPACES

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from scriptorium_serve import server  # noqa: E402
from memory.agent_runtime import OpenAIAgentConfig, OpenAIWriterAgent  # noqa: E402

USER = "eval:run_stub:locomo:conv-0"


class _StubCompletions:
    """Replies with one `remember` call, then a plain closing message."""

    def __init__(self, arguments: str):
        self.arguments = arguments
        self.calls = 0
        self.seen_tools: list[str] = []

    def create(self, *, model, messages, tools, temperature):
        self.calls += 1
        if tools:
            self.seen_tools = [t["function"]["name"] for t in tools]
        if self.calls == 1:
            call = SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="remember", arguments=self.arguments,
                ),
            )
            message = SimpleNamespace(content=None, tool_calls=[call])
        else:
            message = SimpleNamespace(content="written", tool_calls=[])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def _json_string(value: str) -> str:
    import json

    return json.dumps(value)


class _StubClient:
    def __init__(self, arguments: str):
        self.completions = _StubCompletions(arguments)
        self.chat = SimpleNamespace(completions=self.completions)

    def with_options(self, **_: object) -> "_StubClient":
        """The writer narrows the client's timeout per call under a deadline."""
        return self


def _agent_with(arguments: str) -> tuple[OpenAIWriterAgent, _StubClient]:
    client = _StubClient(arguments)
    agent = OpenAIWriterAgent(
        OpenAIAgentConfig(
            base_url="https://stub.invalid/v1",
            api_key="stub-key",
            model="gpt-4o-mini",
        ),
        client=client,
    )
    return agent, client


REQUEST_ID = "eval:run_stub:locomo_refined:conv-0:chunk-0"
SESSION_ID = "eval:run_stub:sample:0"

# `remember` cites the ref by the label the server derives for the message,
# because a footnote may not rest on evidence the workspace lacks.
_THREAD = SESSION_ID.replace(":", "-")
_MESSAGE = f"{REQUEST_ID.replace(':', '-')}-0"
REMEMBER = json.dumps({
    "subject": "Calvin",
    "kind": "person",
    "fact": "Calvin plays saxophone in a jazz quartet.",
    "sources": [f"leaderboard/{_THREAD}/{_MESSAGE}"],
})


class _ReadingModel:
    """Reports the topic files it finds, as a real retrieval pass would."""

    def run(self, *, prompt, system_prompt, cwd, tools=None, **kwargs):
        from memory.agent_runtime import AgentResult
        from pathlib import Path as _Path

        passages = [
            path.read_text(encoding="utf-8").strip()
            for path in sorted(_Path(cwd).glob("topics/*.md"))
        ]
        return AgentResult(
            text="\n\n".join(passages), structured_output=None, num_turns=1,
            input_tokens=1, output_tokens=1, cache_creation_input_tokens=0,
            cache_read_input_tokens=0, anthropic_equivalent_cost_usd=0.0,
            duration_ms=1, duration_api_ms=1, stop_reason="end_turn",
            session_id="s",
        )


def test_add_writes_memory_and_search_returns_it(monkeypatch) -> None:
    agent, client = _agent_with(REMEMBER)

    monkeypatch.setattr(server, "_agent", lambda: agent)
    monkeypatch.setattr(server, "WRITER_BASE_URL", "https://stub.invalid/v1")
    monkeypatch.setattr(server, "WRITER_API_KEY", "stub-key")

    with TestClient(server.app) as client_http:
        added = client_http.post("/add", json={
            "request_id": REQUEST_ID,
            "messages": [
                {"role": "user", "content": "I play sax in a quartet.",
                 "timestamp": 1683504000000},
            ],
            "user_id": USER,
            "session_id": SESSION_ID,
        })
        assert added.status_code == 200, added.text
        assert added.json() == {
            "success": True,
            "request_id": REQUEST_ID,
            "user_id": USER,
            "session_id": SESSION_ID,
        }

        # A served write only ever records, so recording is all it is offered.
        # Across 1905 production passes the model called `remember` 14142
        # times and `update` or `forget` not once; offering them here would be
        # latitude a weak model spends on rejected edits instead of on the
        # conversation, and keeping them would forbid reading the chunk in
        # several groups at once, which is where its latency went.
        assert client.completions.seen_tools == ["remember"]

        # Retrieval runs a model too, and it is a different job: read the
        # workspace and report the memory that bears on the query. Standing
        # in for that keeps this a round trip — the file the writer really
        # wrote is the file retrieval really reads.
        monkeypatch.setattr(server, "_agent", lambda: _ReadingModel())

        found = client_http.post("/search", json={
            "query": "saxophone quartet", "user_id": USER, "top_k": 100,
        })
        assert found.status_code == 200
        data = found.json()["data"]
        assert data, "the just-written memory must be retrievable"
        # Both the topic the writer authored and the archived source message are
        # retrievable; the contract only requires that the Add be searchable.
        contents = " ".join(hit["content"].lower() for hit in data)
        assert "saxophone" in contents
        assert "quartet" in contents
        for hit in data:
            assert set(hit) == {"id", "content", "score", "created_at"}


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def test_writer_reports_endpoint_failures(monkeypatch) -> None:
    class _Failing:
        def create(self, **_):
            raise RuntimeError("upstream 500")

    agent = OpenAIWriterAgent(
        OpenAIAgentConfig(
            base_url="https://stub.invalid/v1", api_key="stub-key",
            model="gpt-4o-mini",
        ),
        client=SimpleNamespace(chat=SimpleNamespace(completions=_Failing())),
    )
    from memory.agent_runtime import AgentExecutionError

    with tempfile.TemporaryDirectory() as cwd:
        with pytest.raises(AgentExecutionError):
            agent.run(prompt="p", system_prompt="s", cwd=cwd, tools=[])


def test_api_key_is_redacted_from_errors() -> None:
    class _Leaky:
        def create(self, **_):
            raise RuntimeError("bad key sk-secret-123 rejected")

    agent = OpenAIWriterAgent(
        OpenAIAgentConfig(
            base_url="https://stub.invalid/v1", api_key="sk-secret-123",
            model="gpt-4o-mini",
        ),
        client=SimpleNamespace(chat=SimpleNamespace(completions=_Leaky())),
    )
    from memory.agent_runtime import AgentExecutionError

    with tempfile.TemporaryDirectory() as cwd:
        with pytest.raises(AgentExecutionError) as caught:
            agent.run(prompt="p", system_prompt="s", cwd=cwd, tools=[])
    assert "sk-secret-123" not in str(caught.value)


def test_a_blank_message_does_not_cost_the_chunk_it_arrived_in(tmp_path, monkeypatch):
    """The benchmark's transcripts carry blank turns; the good ones travel with them.

    Rejecting the whole chunk read to the caller as a failed ingest, 484 times
    in one run, and threw away the nineteen messages beside each blank.
    """
    from fastapi.testclient import TestClient

    from scriptorium_serve import server

    monkeypatch.setattr(server, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(server, "SERVICE_TOKEN", "")
    written: list[int] = []
    monkeypatch.setattr(server, "_ingest", lambda payload: written.append(len(payload.messages)))

    client = TestClient(server.app)
    answer = client.post("/add", json={
        "request_id": "r1",
        "user_id": "u1",
        "session_id": "s1",
        "messages": [
            {"role": "user", "content": "I moved to Berlin in March."},
            {"role": "assistant", "content": "   "},
            {"role": "user", "content": "The flat is near Tempelhof."},
        ],
    })

    assert answer.status_code == 200
    assert written == [2], "the blank is dropped and the other two are written"


def test_a_chunk_that_is_entirely_blank_is_accepted_and_writes_nothing(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from scriptorium_serve import server

    monkeypatch.setattr(server, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(server, "SERVICE_TOKEN", "")
    written: list[int] = []
    monkeypatch.setattr(server, "_ingest", lambda payload: written.append(len(payload.messages)))

    client = TestClient(server.app)
    answer = client.post("/add", json={
        "request_id": "r2", "user_id": "u2", "session_id": "s2",
        "messages": [{"role": "user", "content": " "}, {"role": "assistant", "content": ""}],
    })

    assert answer.status_code == 200
    assert written == [], "nothing to commit, so nothing was committed"
