from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _health() -> dict[str, object]:
    return {
        "status": "ok",
        "auth_readable": True,
        "max_concurrency": 2,
        "max_attempts": 2,
        "connect_timeout_s": 30,
        "read_timeout_s": 300,
        "requested_reasoning_effort": "none",
        "code_sha256": "a" * 64,
        "request_log": "/private/upstream.jsonl",
    }


def _success() -> dict[str, object]:
    return {
        "id": "chatcmpl-subscription-test",
        "object": "chat.completion",
        "model": "gpt-5.5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        "proxy_meta": {
            "attempts": 1,
            "http_request_id": "request-test",
            "unsupported_parameters": [],
            "ignored_client_parameters": ["max_output_tokens"],
            "requested_reasoning_effort": "none",
            "actual_reasoning_effort": "none",
        },
    }


class _FakeUpstream(BaseHTTPRequestHandler):
    health_payload: object = _health()
    response_payload: object = _success()

    def _write(self, status: int, payload: object) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._write(200, type(self).health_payload)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self._write(200, type(self).response_payload)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@contextmanager
def _serve(
    tmp_path: Path,
    *,
    response: object | None = None,
) -> Iterator[tuple[str, Path]]:
    import controlled_subscription_qa_proxy as controlled

    class Fake(_FakeUpstream):
        health_payload = _health()
        response_payload = _success() if response is None else response

    upstream = HTTPServer(("127.0.0.1", 0), Fake)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    log_path = tmp_path / "exclusive.jsonl"
    log_path.touch(mode=0o600)
    wrapper = controlled.base.ThreadedHTTPServer(
        ("127.0.0.1", 0),
        controlled.make_handler(
            upstream=f"http://127.0.0.1:{upstream.server_port}",
            log_path=log_path,
            run_id="qa-proxy-test",
            expected_upstream_sha256="a" * 64,
        ),
    )
    wrapper_thread = threading.Thread(target=wrapper.serve_forever, daemon=True)
    wrapper_thread.start()
    try:
        yield f"http://127.0.0.1:{wrapper.server_port}", log_path
    finally:
        wrapper.shutdown()
        wrapper.server_close()
        wrapper_thread.join(timeout=5)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)


def _post(
    url: str,
    *,
    body: dict[str, object],
    headers: dict[str, str],
) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        response = opener.open(request, timeout=5)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())
    return response.status, json.loads(response.read())


def test_success_records_question_and_logical_call_without_prompt_or_key(
    tmp_path: Path,
) -> None:
    prompt_secret = "PROMPT_SECRET_subscription_qa"
    api_secret = "sk-client-subscription-qa"
    with _serve(tmp_path) as (origin, log_path):
        status, payload = _post(
            f"{origin}/v1/chat/completions",
            body={
                "model": "gpt-5.5",
                "messages": [{"role": "user", "content": prompt_secret}],
                "max_tokens": 1200,
            },
            headers={
                "Authorization": f"Bearer {api_secret}",
                "X-Controlled-Question-ID": "question-1",
                "X-Controlled-Logical-Call-ID": "logical-call-1",
            },
        )

    entry_text = log_path.read_text(encoding="utf-8")
    entry = json.loads(entry_text)
    assert status == 200
    assert entry["status"] == "success"
    assert entry["question_id"] == "question-1"
    assert entry["logical_call_id"] == "logical-call-1"
    assert entry["response_id"] == "chatcmpl-subscription-test"
    assert entry["actual_model"] == "gpt-5.5"
    assert entry["client_http_attempts"] == 1
    assert entry["upstream_http_attempts"] == 1
    assert entry["unsupported_parameters"] == []
    assert entry["ignored_client_parameters"] == ["max_output_tokens"]
    assert payload["exclusive_proxy_meta"]["logical_call_id"] == "logical-call-1"
    assert prompt_secret not in entry_text
    assert api_secret not in entry_text


def test_unexpected_ignored_parameter_fails_closed(tmp_path: Path) -> None:
    response = _success()
    proxy_meta = response["proxy_meta"]
    assert isinstance(proxy_meta, dict)
    proxy_meta["ignored_client_parameters"] = ["temperature"]

    with _serve(tmp_path, response=response) as (origin, log_path):
        status, _ = _post(
            f"{origin}/v1/chat/completions",
            body={"model": "gpt-5.5", "messages": [{"role": "user", "content": "q"}]},
            headers={
                "X-Controlled-Question-ID": "question-2",
                "X-Controlled-Logical-Call-ID": "logical-call-2",
            },
        )

    entry = json.loads(log_path.read_text(encoding="utf-8"))
    assert status == 502
    assert entry["status"] == "error"
    assert "ignored parameter evidence" in entry["error"]
    assert entry["child_upstream_http_attempts"] == 1
    assert entry["upstream_http_attempts"] is None


def test_unsupported_parameter_fails_closed(tmp_path: Path) -> None:
    response = _success()
    proxy_meta = response["proxy_meta"]
    assert isinstance(proxy_meta, dict)
    proxy_meta["unsupported_parameters"] = ["tools"]

    with _serve(tmp_path, response=response) as (origin, log_path):
        status, _ = _post(
            f"{origin}/v1/chat/completions",
            body={"model": "gpt-5.5", "messages": [{"role": "user", "content": "q"}]},
            headers={
                "X-Controlled-Question-ID": "question-3",
                "X-Controlled-Logical-Call-ID": "logical-call-3",
            },
        )

    entry = json.loads(log_path.read_text(encoding="utf-8"))
    assert status == 502
    assert entry["status"] == "error"
    assert "unsupported parameter evidence" in entry["error"]


def test_subscription_upstream_preflight_rejects_code_hash_mismatch() -> None:
    import controlled_subscription_qa_proxy as controlled

    health = _health()
    health["code_sha256"] = "b" * 64

    class WrongHealth(_FakeUpstream):
        health_payload = health

    upstream = HTTPServer(("127.0.0.1", 0), WrongHealth)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(ValueError, match="code SHA-256"):
            controlled.validate_subscription_upstream(
                f"http://127.0.0.1:{upstream.server_port}",
                expected_code_sha256="a" * 64,
            )
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)


def test_missing_reasoning_evidence_fails_closed(tmp_path: Path) -> None:
    response = _success()
    proxy_meta = response["proxy_meta"]
    usage = response["usage"]
    assert isinstance(proxy_meta, dict)
    assert isinstance(usage, dict)
    proxy_meta["actual_reasoning_effort"] = None
    usage.pop("completion_tokens_details")

    with _serve(tmp_path, response=response) as (origin, log_path):
        status, _ = _post(
            f"{origin}/v1/chat/completions",
            body={"model": "gpt-5.5", "messages": [{"role": "user", "content": "q"}]},
            headers={
                "X-Controlled-Question-ID": "question-4",
                "X-Controlled-Logical-Call-ID": "logical-call-4",
            },
        )

    entry = json.loads(log_path.read_text(encoding="utf-8"))
    assert status == 502
    assert entry["status"] == "error"
    assert "reasoning evidence" in entry["error"]


def test_usage_total_mismatch_fails_closed(tmp_path: Path) -> None:
    response = _success()
    usage = response["usage"]
    assert isinstance(usage, dict)
    usage["total_tokens"] = 99

    with _serve(tmp_path, response=response) as (origin, log_path):
        status, _ = _post(
            f"{origin}/v1/chat/completions",
            body={"model": "gpt-5.5", "messages": [{"role": "user", "content": "q"}]},
            headers={
                "X-Controlled-Question-ID": "question-usage",
                "X-Controlled-Logical-Call-ID": "logical-call-usage",
            },
        )

    entry = json.loads(log_path.read_text(encoding="utf-8"))
    assert status == 502
    assert entry["status"] == "error"
    assert "usage" in entry["error"]
    assert entry["child_upstream_http_attempts"] == 1
    assert entry["upstream_http_attempts"] is None


def test_provider_attempt_count_above_frozen_max_fails_closed(
    tmp_path: Path,
) -> None:
    response = _success()
    proxy_meta = response["proxy_meta"]
    assert isinstance(proxy_meta, dict)
    proxy_meta["attempts"] = 3

    with _serve(tmp_path, response=response) as (origin, log_path):
        status, _ = _post(
            f"{origin}/v1/chat/completions",
            body={"model": "gpt-5.5", "messages": [{"role": "user", "content": "q"}]},
            headers={
                "X-Controlled-Question-ID": "question-attempts",
                "X-Controlled-Logical-Call-ID": "logical-call-attempts",
            },
        )

    entry = json.loads(log_path.read_text(encoding="utf-8"))
    assert status == 502
    assert entry["status"] == "error"
    assert "attempt count" in entry["error"]
    assert entry["child_upstream_http_attempts"] == 1
    assert entry["upstream_http_attempts"] is None


def test_completion_usage_above_requested_cap_fails_closed(
    tmp_path: Path,
) -> None:
    response = _success()
    usage = response["usage"]
    assert isinstance(usage, dict)
    usage["completion_tokens"] = 1_201
    usage["total_tokens"] = 1_211

    with _serve(tmp_path, response=response) as (origin, log_path):
        status, _ = _post(
            f"{origin}/v1/chat/completions",
            body={
                "model": "gpt-5.5",
                "messages": [{"role": "user", "content": "q"}],
                "max_tokens": 1_200,
            },
            headers={
                "X-Controlled-Question-ID": "question-cap",
                "X-Controlled-Logical-Call-ID": "logical-call-cap",
            },
        )

    entry = json.loads(log_path.read_text(encoding="utf-8"))
    assert status == 502
    assert entry["status"] == "error"
    assert "completion token cap" in entry["error"]
    assert entry["child_upstream_http_attempts"] == 1
    assert entry["upstream_http_attempts"] == 1
    assert entry["ignored_client_parameters"] == ["max_output_tokens"]


def test_proxy_integrates_with_durable_model_observer(tmp_path: Path) -> None:
    import httpx
    from openai import OpenAI

    from src.evaluation.durable_model_ledger import (
        DurableModelObserver,
        HashChainLedger,
        ledger_state,
        read_ledger,
    )
    from src.evaluation.visible_token_budget import TokenCounter

    with _serve(tmp_path) as (origin, proxy_log):
        ledger_path = tmp_path / "ledger.jsonl"
        ledger = HashChainLedger(ledger_path, run_id="qa-observer-test")
        observer = DurableModelObserver(
            ledger=ledger,
            artifact_root=tmp_path / "artifacts",
            token_counter=TokenCounter.utf8_bytes(requested_model="gpt-5.5"),
            expected_model="gpt-5.5",
            proxy_log=proxy_log,
            formal=True,
        )
        client = OpenAI(
            api_key="test-only",
            base_url=f"{origin}/v1",
            max_retries=0,
            http_client=httpx.Client(trust_env=False),
        )
        operation_id = "answer:question-5"
        ledger.append("operation_started", operation_id=operation_id)
        observer.install(client.chat.completions)
        try:
            with observer.operation(operation_id):
                response = client.chat.completions.create(
                    model="gpt-5.5",
                    messages=[{"role": "user", "content": "q"}],
                    max_tokens=1200,
                    extra_headers={"X-Controlled-Question-ID": "question-5"},
                )
        finally:
            observer.restore()
        ledger.append("operation_committed", operation_id=operation_id)

    state = ledger_state(read_ledger(ledger_path))
    assert response.id == "chatcmpl-subscription-test"
    assert state["model_call_count"] == 1
    assert state["successful_model_calls"] == 1
