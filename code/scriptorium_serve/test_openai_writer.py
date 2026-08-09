"""End-to-end Add through the OpenAI writer, against a stubbed endpoint.

The stub stands in for gpt-4o-mini: it issues one shell tool call that writes a
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

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from scriptorium_serve import server  # noqa: E402
from memory.agent_runtime import OpenAIAgentConfig, OpenAIWriterAgent  # noqa: E402

USER = "eval:run_stub:locomo:conv-0"


class _StubCompletions:
    """Replies with one shell tool call, then a plain closing message."""

    def __init__(self, command: str):
        self.command = command
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
                    name="shell",
                    arguments='{"command": %s}' % _json_string(self.command),
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
    def __init__(self, command: str):
        self.completions = _StubCompletions(command)
        self.chat = SimpleNamespace(completions=self.completions)


def _agent_with(command: str) -> tuple[OpenAIWriterAgent, _StubClient]:
    client = _StubClient(command)
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

# The footnote must cite a source the Add actually archived, so it is built from
# the same ref the server derives. A note may not rest on evidence it lacks.
_THREAD = SESSION_ID.replace(":", "-")
_MESSAGE = f"{REQUEST_ID.replace(':', '-')}-0"

TOPIC = f"""# Music

## Career

Calvin plays saxophone in a jazz quartet.[^e-aaaa111122] ^abc12345

[^e-aaaa111122]: Time: `2023-05-08`; Sources: \
[leaderboard/{_THREAD}/{_MESSAGE}](../sources/leaderboard/{_THREAD}.md#{_MESSAGE})
"""


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
    command = f"cat > topics/music.md <<'SCRIPTORIUM_EOF'\n{TOPIC}SCRIPTORIUM_EOF"
    agent, client = _agent_with(command)

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

        # The shell tool was actually offered to the model and driven by it.
        assert client.completions.seen_tools == ["shell"]

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
