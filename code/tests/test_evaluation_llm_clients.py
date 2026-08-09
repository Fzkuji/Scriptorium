from types import SimpleNamespace

import httpx
import pytest

import scripts.evaluation.llm_clients as clients
from scripts.evaluation.llm_clients import LLMCallError, _should_trust_proxy, chat


def _response(*, choices=None, finish_reason="stop", refusal=None):
    if choices is None:
        message = SimpleNamespace(content="yes", refusal=refusal)
        choices = [SimpleNamespace(
            message=message,
            finish_reason=finish_reason,
        )]
    return SimpleNamespace(
        choices=choices,
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=1),
        model="actual-model",
        id="response-id",
    )


def _client(response):
    def create(**_kwargs):
        return response

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def _sequence_client(values):
    values = iter(values)

    def create(**_kwargs):
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def test_proxy_routing_distinguishes_local_and_external_endpoints():
    assert not _should_trust_proxy("http://127.0.0.1:8199/v1")
    assert not _should_trust_proxy("http://localhost:8199/v1")
    assert not _should_trust_proxy(
        "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )
    assert _should_trust_proxy("https://openrouter.ai/api/v1")
    assert _should_trust_proxy("https://api.openai.com/v1")


def test_chat_records_completion_evidence():
    text, usage = chat(
        _client(_response()), "requested-model", [], retries=1, retry_wait=0
    )

    assert text == "yes"
    assert usage == {
        "prompt_tokens": 4,
        "completion_tokens": 1,
        "request_attempts": 1,
        "physical_http_attempts": 1,
        "logical_client_calls": 1,
        "failed_request_attempts": 0,
        "unknown_token_attempts": 0,
        "request_attempt_details": [{
            "request_attempt": 1,
            "status": "accepted",
            "failure_type": None,
            "error_type": None,
            "error_message": None,
            "requested_model": "requested-model",
            "response_model": "actual-model",
            "response_id": "response-id",
            "finish_reason": "stop",
            "refusal": None,
            "choice_count": 1,
            "prompt_tokens": 4,
            "completion_tokens": 1,
        }],
        "requested_model": "requested-model",
        "response_model": "actual-model",
        "response_id": "response-id",
        "finish_reason": "stop",
        "refusal": None,
        "choice_count": 1,
    }


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (_response(choices=[]), "expected exactly one"),
        (_response(choices=[object(), object()]), "expected exactly one"),
        (_response(finish_reason="length"), "finish_reason"),
        (_response(refusal="policy refusal"), "contains a refusal"),
    ],
)
def test_chat_rejects_invalid_completion_evidence(response, error):
    with pytest.raises(LLMCallError, match=error):
        chat(_client(response), "model", [], retries=1, retry_wait=0)


def test_chat_records_transport_and_invalid_response_retries():
    client = _sequence_client([
        RuntimeError("connection reset"),
        _response(finish_reason="length"),
        _response(),
    ])

    text, usage = chat(client, "requested-model", [], retries=3, retry_wait=0)

    assert text == "yes"
    assert usage["request_attempts"] == 3
    assert usage["failed_request_attempts"] == 2
    assert usage["unknown_token_attempts"] == 1
    assert usage["prompt_tokens"] == 8
    assert usage["completion_tokens"] == 2
    first, second, third = usage["request_attempt_details"]
    assert first == {
        "request_attempt": 1,
        "status": "error",
        "failure_type": "transport_error",
        "error_type": "RuntimeError",
        "error_message": "connection reset",
        "requested_model": "requested-model",
        "response_model": None,
        "response_id": None,
        "finish_reason": None,
        "refusal": None,
        "choice_count": None,
        "prompt_tokens": None,
        "completion_tokens": None,
    }
    assert second["status"] == "rejected"
    assert second["failure_type"] == "invalid_response"
    assert second["finish_reason"] == "length"
    assert second["response_id"] == "response-id"
    assert third["status"] == "accepted"


def test_openai_client_disables_sdk_retries_and_counts_physical_http(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(500, json={"error": {"message": "retry"}})
        return httpx.Response(
            200,
            json={
                "id": "response-id",
                "object": "chat.completion",
                "created": 1,
                "model": "actual-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "yes"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        clients,
        "_http_client",
        lambda _base: httpx.Client(transport=transport, timeout=10),
    )
    client = clients._client("test", "https://example.test/v1")
    text, usage = chat(
        client, "requested-model", [], retries=3, retry_wait=0
    )
    assert text == "yes"
    assert len(calls) == 3
    assert usage["physical_http_attempts"] == 3
    assert usage["request_attempts"] == 3
