"""Contract checks for the Add / Search service.

Retrieval is exercised against a hand-built workspace, so these run with no
writer endpoint and no API key. What they protect is the platform-facing
contract: response shape, user isolation, top_k, and auth.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

WORKSPACES = tempfile.mkdtemp(prefix="scriptorium-contract-")
os.environ["SCRIPTORIUM_WORKSPACES"] = WORKSPACES

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    server._indexes.pop(str(workspace), None)


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


def test_top_k_100_is_not_truncated_at_50() -> None:
    """The platform fixes top_k=100; the BM25 cap must not clip it to 50."""
    body = [
        _line("2023-08-01", f"Calvin discussed proposal {index}.", f"D4:{index}")
        for index in range(120)
    ]
    _seed(USER_B, "bulk", "".join(body))

    response = client.post(
        "/search", json={"query": "Calvin proposal", "user_id": USER_B, "top_k": 100}
    )
    assert len(response.json()["data"]) > 50


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


def test_add_without_writer_config_reports_unavailable() -> None:
    """Guards the failure mode where the writer endpoint is unset."""
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
