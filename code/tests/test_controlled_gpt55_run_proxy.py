from __future__ import annotations

import copy
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

from baselines.controlled_locomo import controlled_gpt55_run_proxy as controlled  # noqa: E402


def _health() -> dict[str, object]:
    return {
        "status": "ok",
        "schema": "openai-gpt55-flex-health/v1",
        "requested_model": "gpt-5.5",
        "provider_model": "gpt-5.5-2026-04-23",
        "service_tier": "flex",
        "budget": {
            "max_cost_usd": "100",
            "committed_cost_usd": "1.25",
            "reserved_cost_usd": "0.75",
            "remaining_cost_usd": "98",
            "in_flight": 1,
        },
    }


def _success() -> dict[str, object]:
    return {
        "id": "chatcmpl-fake",
        "model": "gpt-5.5",
        "service_tier": "flex",
        "choices": [
            {
                "message": {"role": "assistant", "content": "<answer>x</answer>"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        },
        "flex_gateway_meta": {
            "request_id": "gateway-request-123",
            "request_sha256": "a" * 64,
            "provider_request_sha256": "b" * 64,
            "provider_actual_model": "gpt-5.5-2026-04-23",
            "returned_alias": "gpt-5.5",
            "service_tier": "flex",
            "physical_attempt_count": 2,
            "physical_retry_count": 1,
        },
        "proxy_meta": {
            "attempts": 2,
            "http_request_id": "openai-request-123",
            "unsupported_parameters": [],
            "provider_actual_model": "gpt-5.5-2026-04-23",
            "service_tier": "flex",
        },
    }


class _FakeUpstream(BaseHTTPRequestHandler):
    health_payload: object = _health()
    response_payload: object = _success()
    response_status = 200

    def _write(self, status: int, value: object) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        self._write(200, type(self).health_payload)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self._write(type(self).response_status, type(self).response_payload)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@contextmanager
def _serve(
    tmp_path: Path, *, health: object | None = None, response: object | None = None
) -> Iterator[tuple[str, Path]]:
    class Fake(_FakeUpstream):
        health_payload = _health() if health is None else health
        response_payload = _success() if response is None else response

    upstream = HTTPServer(("127.0.0.1", 0), Fake)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    log_path = tmp_path / "exclusive.jsonl"
    log_path.touch(mode=0o600)
    upstream_origin = f"http://127.0.0.1:{upstream.server_port}"
    wrapper = controlled.base.ThreadedHTTPServer(
        ("127.0.0.1", 0),
        controlled.make_handler(
            upstream=upstream_origin,
            log_path=log_path,
            run_id="controlled-test",
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


def _request_json(
    url: str,
    *,
    body: dict[str, object] | None = None,
    authorization: str | None = None,
) -> tuple[int, dict[str, object]]:
    headers = {"Content-Type": "application/json"}
    if authorization is not None:
        headers["Authorization"] = authorization
    request = urllib.request.Request(
        url,
        data=(
            json.dumps(body, separators=(",", ":")).encode()
            if body is not None
            else None
        ),
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        response = opener.open(request, timeout=5)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())
    return response.status, json.loads(response.read())


def test_health_rejects_subscription_like_upstream_without_echoing_secret(
    tmp_path: Path,
) -> None:
    secret = "sk-subscription-like-secret"
    subscription_health = {
        "status": "ok",
        "auth_readable": True,
        "code_sha256": "c" * 64,
        "api_key": secret,
    }
    with _serve(tmp_path, health=subscription_health) as (origin, log_path):
        status, payload = _request_json(f"{origin}/healthz")
    assert status == 503
    assert payload["status"] == "error"
    assert secret not in json.dumps(payload)
    assert log_path.read_text() == ""


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "chatgpt-subscription-health/v1"),
        ("requested_model", "gpt-5.5-auto"),
        ("provider_model", "gpt-5.5"),
        ("service_tier", "standard"),
        ("budget", {"max_cost_usd": "100"}),
        (
            "budget",
            {
                "max_cost_usd": "100",
                "committed_cost_usd": "1",
                "reserved_cost_usd": "1",
                "remaining_cost_usd": "99",
                "in_flight": 1,
            },
        ),
    ],
)
def test_health_requires_exact_flex_contract(
    tmp_path: Path, field: str, value: object
) -> None:
    health = _health()
    health[field] = value
    with _serve(tmp_path, health=health) as (origin, _):
        status, _ = _request_json(f"{origin}/healthz")
    assert status == 503


def _without(response: dict[str, object], *path: str) -> dict[str, object]:
    result = copy.deepcopy(response)
    target = result
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target.pop(path[-1])
    return result


@pytest.mark.parametrize(
    "response",
    [
        _without(_success(), "service_tier"),
        _without(_success(), "flex_gateway_meta"),
        _without(_success(), "flex_gateway_meta", "provider_actual_model"),
        _without(_success(), "flex_gateway_meta", "service_tier"),
        _without(_success(), "flex_gateway_meta", "request_id"),
        {**_success(), "service_tier": "standard"},
    ],
)
def test_success_without_complete_gateway_evidence_becomes_error(
    tmp_path: Path, response: dict[str, object]
) -> None:
    with _serve(tmp_path, response=response) as (origin, log_path):
        status, payload = _request_json(
            f"{origin}/v1/chat/completions",
            body={"model": "gpt-5.5", "messages": [{"role": "user", "content": "q"}]},
        )
    entry = json.loads(log_path.read_text())
    assert status == 502
    assert entry["status"] == "error"
    assert entry["http_status"] == 502
    assert entry["gateway_request_id"] is None
    assert payload["exclusive_proxy_meta"]["gateway_request_id"] is None


def test_success_persists_gateway_evidence_but_not_prompt_or_key(tmp_path: Path) -> None:
    prompt_secret = "PROMPT_SECRET_2d04f657"
    api_secret = "sk-client-secret-49c251"
    with _serve(tmp_path) as (origin, log_path):
        status, payload = _request_json(
            f"{origin}/v1/chat/completions",
            body={
                "model": "gpt-5.5",
                "service_tier": "flex",
                "messages": [{"role": "user", "content": prompt_secret}],
            },
            authorization=f"Bearer {api_secret}",
        )
    entry_text = log_path.read_text()
    entry = json.loads(entry_text)
    assert status == 200
    assert entry["status"] == "success"
    assert entry["actual_model"] == "gpt-5.5"
    assert entry["provider_actual_model"] == "gpt-5.5-2026-04-23"
    assert entry["service_tier"] == "flex"
    assert entry["gateway_request_id"] == "gateway-request-123"
    assert entry["gateway_request_sha256"] == "a" * 64
    assert entry["provider_request_sha256"] == "b" * 64
    assert entry["upstream_http_attempts"] == 2
    assert payload["service_tier"] == "flex"
    assert payload["exclusive_proxy_meta"]["gateway_request_id"] == (
        entry["gateway_request_id"]
    )
    assert prompt_secret not in entry_text
    assert api_secret not in entry_text
