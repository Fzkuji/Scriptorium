from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluation.durable_model_ledger import (
    DurableLedgerError,
    DurableModelObserver,
    HashChainLedger,
    ledger_state,
    read_ledger,
)
from scripts.evaluation.visible_token_budget import TokenCounter


class FakeResponse:
    def __init__(self, *, response_id: str = "response-1", model: str = "model-1"):
        self.id = response_id
        self.model = model
        self.usage = SimpleNamespace(
            prompt_tokens=11, completion_tokens=3, total_tokens=14
        )

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        del mode
        return {
            "id": self.id,
            "model": self.model,
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            },
            "choices": [{"message": {"content": "ok"}}],
        }


def operation_start(ledger: HashChainLedger, operation_id: str = "op-1") -> None:
    ledger.append(
        "operation_started",
        operation_id=operation_id,
        kind="test",
        operation_input_sha256="a" * 64,
        metadata={},
        continuing_tree_before={},
    )


def operation_commit(ledger: HashChainLedger, operation_id: str = "op-1") -> None:
    ledger.append(
        "operation_committed",
        operation_id=operation_id,
        kind="test",
        operation_input_sha256="a" * 64,
        continuing_tree_before={},
        continuing_tree_after={},
        latency_s=0.1,
        cost={},
        result={},
    )


def test_synthetic_observer_persists_request_response_and_usage(tmp_path: Path) -> None:
    ledger = HashChainLedger(tmp_path / "operations.jsonl", run_id="run-1")
    operation_start(ledger)
    resource = SimpleNamespace(create=lambda **kwargs: FakeResponse())
    observer = DurableModelObserver(
        ledger=ledger,
        artifact_root=tmp_path,
        token_counter=TokenCounter.utf8_bytes(requested_model="model-1"),
        expected_model="model-1",
        proxy_log=None,
        formal=False,
    )
    observer.install(resource)
    with observer.operation("op-1"):
        response = resource.create(
            model="model-1",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=20,
        )
    observer.restore()
    operation_commit(ledger)

    assert response.id == "response-1"
    records = read_ledger(tmp_path / "operations.jsonl")
    state = ledger_state(records)
    assert state["model_call_count"] == 1
    started = next(item for item in records if item["event"] == "model_call_started")
    finished = next(item for item in records if item["event"] == "model_call_finished")
    request_path = tmp_path / started["request_path"]
    response_path = tmp_path / finished["response_path"]
    assert request_path.is_file() and response_path.is_file()
    request = json.loads(request_path.read_text())
    stored_response = json.loads(response_path.read_text())
    assert request["transport_headers"] == {
        "X-Controlled-Logical-Call-ID": started["logical_call_id"]
    }
    assert request["local_visible_tokens"] == started["local_visible_tokens"]
    assert stored_response["response"]["id"] == "response-1"
    assert finished["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 3,
        "total_tokens": 14,
    }
    assert finished["proxy_evidence"]["mode"] == "synthetic"


def test_formal_observer_links_exclusive_proxy_attempt(tmp_path: Path) -> None:
    proxy_log = tmp_path / "proxy.jsonl"
    proxy_log.touch()
    usage = {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}

    def create(**kwargs):
        logical_id = kwargs["extra_headers"]["X-Controlled-Logical-Call-ID"]
        event = {
            "run_id": "proxy-launch",
            "status": "success",
            "http_status": 200,
            "requested_model": "model-1",
            "actual_model": "model-1",
            "response_id": "response-1",
            "event_id": "event-1",
            "logical_call_id": logical_id,
            "request_sha256": "b" * 64,
            "response_sha256": "c" * 64,
            "client_http_attempts": 1,
            "upstream_http_attempts": 2,
            "unsupported_parameters": ["max_output_tokens"],
            "usage": usage,
            "error": None,
        }
        with proxy_log.open("ab") as handle:
            handle.write(json.dumps(event, separators=(",", ":")).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return FakeResponse()

    ledger = HashChainLedger(tmp_path / "operations.jsonl", run_id="run-1")
    operation_start(ledger)
    resource = SimpleNamespace(create=create)
    observer = DurableModelObserver(
        ledger=ledger,
        artifact_root=tmp_path,
        token_counter=TokenCounter.utf8_bytes(requested_model="model-1"),
        expected_model="model-1",
        proxy_log=proxy_log,
        formal=True,
    )
    observer.install(resource)
    with observer.operation("op-1"):
        resource.create(
            model="model-1", messages=[{"role": "user", "content": "hello"}]
        )
    observer.restore()
    operation_commit(ledger)
    terminal = next(
        item for item in ledger.records if item["event"] == "model_call_finished"
    )
    assert terminal["proxy_evidence"]["client_http_attempts"] == 1
    assert terminal["proxy_evidence"]["upstream_http_attempts"] == 2
    assert terminal["proxy_evidence"]["unsupported_parameters"] == [
        "max_output_tokens"
    ]
    assert terminal["proxy_evidence"]["events"][0]["event_id"] == "event-1"


def test_formal_observer_fails_closed_without_proxy_event(tmp_path: Path) -> None:
    proxy_log = tmp_path / "proxy.jsonl"
    proxy_log.touch()
    ledger = HashChainLedger(tmp_path / "operations.jsonl", run_id="run-1")
    operation_start(ledger)
    resource = SimpleNamespace(create=lambda **kwargs: FakeResponse())
    observer = DurableModelObserver(
        ledger=ledger,
        artifact_root=tmp_path,
        token_counter=TokenCounter.utf8_bytes(requested_model="model-1"),
        expected_model="model-1",
        proxy_log=proxy_log,
        formal=True,
    )
    observer.install(resource)
    with pytest.raises(DurableLedgerError, match="no event"):
        with observer.operation("op-1"):
            resource.create(model="model-1", messages=[])
    observer.restore()
    assert any(item["event"] == "model_call_failed" for item in ledger.records)


def test_ledger_state_rejects_orphans_failures_and_tampering(tmp_path: Path) -> None:
    path = tmp_path / "operations.jsonl"
    ledger = HashChainLedger(path, run_id="run-1")
    operation_start(ledger)
    with pytest.raises(DurableLedgerError, match="unresolved ledger state"):
        ledger_state(ledger.records)

    operation_commit(ledger)
    assert ledger_state(ledger.records)["model_call_count"] == 0
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["kind"] = "tampered"
    lines[0] = json.dumps(first, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(DurableLedgerError, match="event hash mismatch"):
        read_ledger(path)


def test_ledger_rejects_hardlink(tmp_path: Path) -> None:
    path = tmp_path / "operations.jsonl"
    ledger = HashChainLedger(path, run_id="run-1")
    operation_start(ledger)
    os.link(path, tmp_path / "second-link.jsonl")
    with pytest.raises(DurableLedgerError, match="hardlinked"):
        ledger.append("unused")
