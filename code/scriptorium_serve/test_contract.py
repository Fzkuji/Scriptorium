"""Contract checks for the Add / Search service.

Retrieval reads the workspace with a model, so these stub the model and
report what a real one would: the memory it found. What they protect is
the platform-facing contract — response shape, user isolation, top_k and
auth — not the model's judgement.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

WORKSPACES = tempfile.mkdtemp(prefix="scriptorium-contract-")
os.environ["SCRIPTORIUM_WORKSPACES"] = WORKSPACES

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from scriptorium_serve import server  # noqa: E402
from scriptorium.cli import ensure_workspace  # noqa: E402

client = TestClient(server.app)

USER_A = "eval:run_abc123:locomo:conv-0"
USER_B = "eval:run_abc123:locomo:conv-1"


def _line(date: str, text: str, ref: str) -> str:
    """One memory line in Scriptorium's dated record format."""
    return f"[{date}] {text} · [{ref}]\n"


def _seed(user_id: str, topic: str, body: str) -> None:
    """Write a topic file directly, standing in for a completed Add."""
    workspace = server._workspace(user_id)
    ensure_workspace(workspace)
    (workspace / "topics").mkdir(parents=True, exist_ok=True)
    (workspace / "topics" / f"{topic}.md").write_text(body, encoding="utf-8")


class _ReportingModel:
    """Reports back whatever the workspace holds, one passage per file.

    A real model searches and reads before deciding what is relevant.
    Standing in for that here keeps these checks about the contract.
    """

    def __init__(self, workspace_of):
        self._workspace_of = workspace_of

    def run(self, *, prompt, system_prompt, cwd, tools=None, **kwargs):
        from memory.agent_runtime import AgentResult

        passages = [
            path.read_text(encoding="utf-8").strip()
            for path in sorted(Path(cwd).glob("topics/*.md"))
        ]
        return AgentResult(
            text="\n\n".join(passages),
            structured_output=None, num_turns=1,
            input_tokens=1, output_tokens=1,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
            anthropic_equivalent_cost_usd=0.0,
            duration_ms=1, duration_api_ms=1,
            stop_reason="end_turn", session_id="s",
        )


@pytest.fixture(autouse=True)
def _stub_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server, "_agent", lambda: _ReportingModel(server._workspace)
    )


def test_health_needs_no_auth() -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_search_returns_contract_shape() -> None:
    _seed(
        USER_A,
        "music",
        _line("2023-05-08", "Calvin plays saxophone in a jazz quartet.", "D1:1"),
    )
    response = client.post(
        "/search",
        json={"query": "saxophone jazz", "user_id": USER_A, "top_k": 100},
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"data"}
    assert payload["data"], "expected a hit for a seeded term"
    hit = payload["data"][0]
    assert set(hit) == {"id", "content", "score", "created_at"}
    assert isinstance(hit["id"], str) and hit["id"]
    assert isinstance(hit["score"], (int, float))
    assert "saxophone" in hit["content"].lower()


def test_results_are_isolated_by_user_id() -> None:
    _seed(USER_A, "pets", _line("2023-06-01", "Calvin owns a beagle named Rex.", "D2:1"))
    _seed(USER_B, "pets", _line("2023-06-01", "Dana owns a tabby named Miso.", "D2:1"))

    a = client.post("/search", json={"query": "beagle Rex", "user_id": USER_A}).json()
    b = client.post("/search", json={"query": "beagle Rex", "user_id": USER_B}).json()

    assert any("beagle" in hit["content"].lower() for hit in a["data"])
    assert not any("beagle" in hit["content"].lower() for hit in b["data"])


def test_unknown_user_returns_empty_data() -> None:
    response = client.post(
        "/search", json={"query": "anything", "user_id": "eval:never:seen"}
    )
    assert response.status_code == 200
    assert response.json() == {"data": []}


def test_search_respects_top_k() -> None:
    body = [
        _line("2023-07-01", f"Calvin recorded takeaway {index}.", f"D3:{index}")
        for index in range(30)
    ]
    _seed(USER_A, "notes", "".join(body))

    response = client.post(
        "/search", json={"query": "Calvin takeaway", "user_id": USER_A, "top_k": 5}
    )
    assert len(response.json()["data"]) <= 5


def test_a_large_top_k_is_not_clipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The platform fixes top_k=100, and nothing may cut the report short
    of what the model actually found."""
    class _Verbose(_ReportingModel):
        def run(self, **kwargs):
            result = super().run(**kwargs)
            return type(result)(
                **{**result.__dict__,
                   "text": "\n\n".join(f"passage {i}" for i in range(120))}
            )

    monkeypatch.setattr(server, "_agent", lambda: _Verbose(server._workspace))
    _seed(USER_B, "bulk", _line("2023-08-01", "Calvin discussed a proposal.", "D4:1"))

    response = client.post(
        "/search", json={"query": "Calvin proposal", "user_id": USER_B, "top_k": 100}
    )
    assert len(response.json()["data"]) == 100


def test_empty_query_is_rejected() -> None:
    response = client.post("/search", json={"query": "   ", "user_id": USER_A})
    assert response.status_code == 400


def test_add_rejects_empty_messages() -> None:
    response = client.post(
        "/add",
        json={
            "request_id": "r1",
            "messages": [],
            "user_id": USER_A,
            "session_id": "s1",
        },
    )
    assert response.status_code == 400


def test_add_without_writer_config_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the failure mode where the writer endpoint is unset."""
    monkeypatch.undo()  # drop the stubbed model; this is about not having one
    response = client.post(
        "/add",
        json={
            "request_id": "r2",
            "messages": [{"role": "user", "content": "hello", "timestamp": 1704067200000}],
            "user_id": "eval:run_abc123:locomo:conv-writer",
            "session_id": "s1",
        },
    )
    assert response.status_code == 503


def test_user_id_cannot_escape_workspace_root() -> None:
    response = client.post(
        "/search", json={"query": "x", "user_id": "../../etc"}
    )
    assert response.status_code in (200, 400)
    assert not (Path(WORKSPACES).parent / "etc").exists()


def test_bearer_token_enforced_when_configured() -> None:
    server.SERVICE_TOKEN = "secret-token"
    try:
        assert client.get("/health").status_code == 200
        rejected = client.post("/search", json={"query": "x", "user_id": USER_A})
        assert rejected.status_code == 401
        accepted = client.post(
            "/search",
            json={"query": "saxophone", "user_id": USER_A},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert accepted.status_code == 200
    finally:
        server.SERVICE_TOKEN = ""


# -- retrieval shaping ------------------------------------------------------

def test_the_report_is_split_into_one_row_per_remembered_thing():
    from scriptorium_serve.server import _found_memory

    rows = _found_memory(
        "Dave > Relocation\n[2024-03] He moved to Shanghai.\n\n"
        "[2024-06] His wife stayed in Beijing."
    )

    assert len(rows) == 2
    assert "Shanghai" in rows[0]["content"]
    assert "Beijing" in rows[1]["content"]


def test_rows_carry_the_date_the_passage_states():
    """The caller sorts and filters on created_at, and the only date that
    means anything is the one the memory's own footnote recorded."""
    from scriptorium_serve.server import _found_memory

    rows = _found_memory("[2024-03] He moved to Shanghai.")

    assert rows[0]["created_at"] == "2024-03"


def test_a_passage_without_a_date_reports_none():
    from scriptorium_serve.server import _found_memory

    assert _found_memory("He moved to Shanghai.")[0]["created_at"] == ""


def test_finding_nothing_returns_nothing():
    """The prompt tells the model to return nothing rather than something
    it did not find, so an empty report must not become a row."""
    from scriptorium_serve.server import _found_memory

    assert _found_memory("") == []
    assert _found_memory("   \n\n  ") == []


def test_order_is_the_models_judgement_not_a_score():
    from scriptorium_serve.server import _found_memory

    rows = _found_memory("first\n\nsecond\n\nthird")

    assert [row["score"] for row in rows] == sorted(
        (row["score"] for row in rows), reverse=True
    )


def test_the_service_refuses_to_start_without_its_search_backend(monkeypatch):
    """A missing backend must not look like an empty memory.

    A backend that cannot be built is caught downstream and read as "nothing
    found", so a service without one serves every request successfully and
    returns nothing: a whole evaluation scored zero against a healthy process
    and a clean log. That happened once, with an image built without the
    embedding backend while search was still fusing both.
    """
    import builtins

    missing = "rank_bm25" if server.SEARCH_TOOLS != "fused" else "sentence_transformers"
    real = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.startswith(missing):
            raise ModuleNotFoundError(f"No module named {missing!r}")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)

    with pytest.raises(ModuleNotFoundError):
        with TestClient(server.app):
            pass
