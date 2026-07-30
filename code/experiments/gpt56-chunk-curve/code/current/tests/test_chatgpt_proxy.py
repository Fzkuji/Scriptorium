import json

import src.chatgpt_proxy as proxy
from src.chatgpt_proxy import parse_responses_stream


class FakeResponse:
    def __init__(self, events=(), *, status_code=200, text="", headers=None):
        self.events = events
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self.closed = False

    def iter_lines(self):
        for event in self.events:
            yield ("data: " + json.dumps(event)).encode()

    def close(self):
        self.closed = True


def test_parse_completed_text_and_usage():
    result = parse_responses_stream(FakeResponse([
        {"type": "response.output_text.delta", "delta": "OK"},
        {"type": "response.completed", "response": {"usage": {
            "input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}},
    ]))
    assert result["text"] == "OK"
    assert result["usage"]["total_tokens"] == 5


def test_rejects_truncated_stream():
    result = parse_responses_stream(FakeResponse([
        {"type": "response.output_text.delta", "delta": "partial"},
    ]))
    assert "error" in result and "response.completed" in result["error"]


def test_rejects_failed_stream_even_with_partial_text():
    result = parse_responses_stream(FakeResponse([
        {"type": "response.output_text.delta", "delta": "partial"},
        {"type": "response.failed", "response": {"error": {"code": "bad"}}},
    ]))
    assert "error" in result and "stream failed" in result["error"]


def test_rejects_empty_completed_response():
    result = parse_responses_stream(FakeResponse([
        {"type": "response.completed", "response": {
            "output": [], "usage": {}}},
    ]))
    assert "error" in result and "empty output" in result["error"]


def test_recovers_output_text_present_only_in_completed_response():
    result = parse_responses_stream(FakeResponse([
        {"type": "response.completed", "response": {
            "output": [{
                "id": "msg-final",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "Recovered final text",
                }],
            }],
            "usage": {},
        }},
    ]))

    assert result["text"] == "Recovered final text"
    assert result["tool_calls"] is None


def test_recovers_function_call_present_only_in_completed_response():
    result = parse_responses_stream(FakeResponse([
        {"type": "response.completed", "response": {
            "output": [{
                "id": "item-final",
                "type": "function_call",
                "call_id": "call-final",
                "name": "read_original",
                "arguments": '{"dia_ids":["D1:1"]}',
                "status": "completed",
            }],
            "usage": {},
        }},
    ]))

    assert result["text"] is None
    assert result["tool_calls"] == [{
        "id": "call-final",
        "type": "function",
        "function": {
            "name": "read_original",
            "arguments": '{"dia_ids":["D1:1"]}',
        },
    }]


def test_preserves_two_interleaved_function_calls():
    result = parse_responses_stream(FakeResponse([
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"id": "item-a", "type": "function_call",
                  "call_id": "call-a", "name": "bash"}},
        {"type": "response.output_item.added", "output_index": 1,
         "item": {"id": "item-b", "type": "function_call",
                  "call_id": "call-b", "name": "read_original"}},
        {"type": "response.function_call_arguments.delta", "item_id": "item-a",
         "delta": '{"command":"ls"}'},
        {"type": "response.function_call_arguments.delta", "item_id": "item-b",
         "delta": '{"dia_ids":["D1:1"]}'},
        {"type": "response.function_call_arguments.done", "item_id": "item-b",
         "arguments": '{"dia_ids":["D1:1"]}'},
        {"type": "response.function_call_arguments.done", "item_id": "item-a",
         "arguments": '{"command":"ls"}'},
        {"type": "response.completed", "response": {
            "id": "resp-1", "model": "gpt-5.5", "usage": {}}},
    ]))
    assert [x["id"] for x in result["tool_calls"]] == ["call-b", "call-a"]
    assert result["response_id"] == "resp-1"
    assert result["actual_model"] == "gpt-5.5"


def completed_response(
    *, reasoning_effort="none", reasoning_tokens=0,
    include_reasoning_tokens=True,
):
    response = {
        "id": "resp-off",
        "model": "gpt-5.5",
        "usage": {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "output_tokens_details": {},
        },
    }
    if reasoning_effort is not None:
        response["reasoning"] = {"effort": reasoning_effort}
    if include_reasoning_tokens:
        response["usage"]["output_tokens_details"]["reasoning_tokens"] = (
            reasoning_tokens
        )
    return FakeResponse([
        {"type": "response.output_text.delta", "delta": "OK"},
        {"type": "response.completed", "response": response},
    ])


def test_call_forces_reasoning_none_and_records_actual_effort(
    monkeypatch, tmp_path,
):
    sent = []
    response = completed_response()
    request_log = tmp_path / "requests.jsonl"

    def fake_post(*_args, **kwargs):
        sent.append(kwargs["json"])
        return response

    monkeypatch.setattr(proxy, "get_access_token", lambda: "test-token")
    monkeypatch.setattr(proxy.requests, "post", fake_post)
    monkeypatch.setattr(proxy, "REQUEST_LOG", str(request_log))

    result = proxy.call_responses_api([
        {"role": "user", "content": "Return OK."},
    ])

    assert len(sent) == 1
    assert sent[0]["reasoning"] == {"effort": "none"}
    assert result["requested_reasoning_effort"] == "none"
    assert result["actual_reasoning_effort"] == "none"
    assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0
    assert response.closed
    log = json.loads(request_log.read_text())
    assert log["requested_reasoning_effort"] == "none"
    assert log["actual_reasoning_effort"] == "none"


def test_call_ignores_unsupported_max_output_tokens_without_retry(monkeypatch):
    sent = []

    def fake_post(*_args, **kwargs):
        sent.append(kwargs["json"])
        return completed_response()

    monkeypatch.setattr(proxy, "get_access_token", lambda: "test-token")
    monkeypatch.setattr(proxy.requests, "post", fake_post)

    result = proxy.call_responses_api(
        [{"role": "user", "content": "Return OK."}],
        max_output_tokens=1200,
    )

    assert len(sent) == 1
    assert "max_output_tokens" not in sent[0]
    assert result["ignored_client_parameters"] == ["max_output_tokens"]


def test_call_fails_closed_on_non_none_actual_effort(monkeypatch):
    calls = 0

    def fake_post(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return completed_response(reasoning_effort="medium")

    monkeypatch.setattr(proxy, "get_access_token", lambda: "test-token")
    monkeypatch.setattr(proxy.requests, "post", fake_post)

    result = proxy.call_responses_api([
        {"role": "user", "content": "Return OK."},
    ])

    assert calls == 1
    assert "reported actual effort 'medium'" in result["error"]


def test_call_fails_closed_on_nonzero_reasoning_tokens(monkeypatch):
    calls = 0

    def fake_post(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return completed_response(reasoning_tokens=1)

    monkeypatch.setattr(proxy, "get_access_token", lambda: "test-token")
    monkeypatch.setattr(proxy.requests, "post", fake_post)

    result = proxy.call_responses_api([
        {"role": "user", "content": "Return OK."},
    ])

    assert calls == 1
    assert "violated reasoning.effort=none" in result["error"]


def test_usage_limit_is_not_retried(monkeypatch):
    calls = 0

    def fake_post(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return FakeResponse(
            status_code=429,
            text='{"error":{"code":"usage_limit_reached"}}',
        )

    monkeypatch.setattr(proxy, "get_access_token", lambda: "test-token")
    monkeypatch.setattr(proxy.requests, "post", fake_post)

    result = proxy.call_responses_api([
        {"role": "user", "content": "Return OK."},
    ])

    assert calls == 1
    assert "usage_limit_reached" in result["error"]


def test_call_rejects_missing_reasoning_evidence(monkeypatch):
    monkeypatch.setattr(proxy, "get_access_token", lambda: "test-token")
    monkeypatch.setattr(
        proxy.requests,
        "post",
        lambda *_args, **_kwargs: completed_response(
            reasoning_effort=None,
            include_reasoning_tokens=False,
        ),
    )

    result = proxy.call_responses_api([
        {"role": "user", "content": "Return OK."},
    ])

    assert "no evidence" in result["error"]


def test_transport_timeout_is_not_retried(monkeypatch):
    calls = 0

    def fake_post(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise proxy.requests.ReadTimeout("timed out")

    monkeypatch.setattr(proxy, "get_access_token", lambda: "test-token")
    monkeypatch.setattr(proxy.requests, "post", fake_post)

    result = proxy.call_responses_api([
        {"role": "user", "content": "Return OK."},
    ])

    assert calls == 1
    assert "ReadTimeout" in result["error"]


def test_ssl_handshake_failure_retries_once(monkeypatch):
    calls = 0

    def fake_post(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise proxy.requests.exceptions.SSLError(
                "UNEXPECTED_EOF_WHILE_READING"
            )
        return completed_response()

    monkeypatch.setattr(proxy, "get_access_token", lambda: "test-token")
    monkeypatch.setattr(proxy.requests, "post", fake_post)
    monkeypatch.setattr(proxy.time, "sleep", lambda _seconds: None)

    result = proxy.call_responses_api([
        {"role": "user", "content": "Return OK."},
    ])

    assert result["text"] == "OK"
    assert result["attempts"] == 2
    assert calls == 2


def test_chunked_stream_disconnect_retries_once(monkeypatch):
    calls = 0

    class DisconnectedResponse(FakeResponse):
        def iter_lines(self):
            raise proxy.requests.exceptions.ChunkedEncodingError(
                "Response ended prematurely"
            )

    def fake_post(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return DisconnectedResponse()
        return completed_response()

    monkeypatch.setattr(proxy, "get_access_token", lambda: "test-token")
    monkeypatch.setattr(proxy.requests, "post", fake_post)
    monkeypatch.setattr(proxy.time, "sleep", lambda _seconds: None)

    result = proxy.call_responses_api([
        {"role": "user", "content": "Return OK."},
    ])

    assert result["text"] == "OK"
    assert result["attempts"] == 2
    assert calls == 2


def test_explicit_retryable_http_status_retries_once(monkeypatch):
    responses = [
        FakeResponse(status_code=503, text="temporarily unavailable"),
        completed_response(),
    ]

    monkeypatch.setattr(proxy, "get_access_token", lambda: "test-token")
    monkeypatch.setattr(
        proxy.requests, "post", lambda *_args, **_kwargs: responses.pop(0)
    )
    monkeypatch.setattr(proxy.time, "sleep", lambda _seconds: None)

    result = proxy.call_responses_api([
        {"role": "user", "content": "Return OK."},
    ])

    assert result["text"] == "OK"
    assert result["attempts"] == 2
    assert responses == []


def test_explicit_stream_server_error_retries_once(monkeypatch):
    responses = [
        FakeResponse([
            {
                "type": "response.failed",
                "response": {
                    "error": {
                        "code": "server_error",
                        "message": "Please retry.",
                    }
                },
            }
        ]),
        completed_response(),
    ]

    monkeypatch.setattr(proxy, "get_access_token", lambda: "test-token")
    monkeypatch.setattr(
        proxy.requests, "post", lambda *_args, **_kwargs: responses.pop(0)
    )
    monkeypatch.setattr(proxy.time, "sleep", lambda _seconds: None)

    result = proxy.call_responses_api([
        {"role": "user", "content": "Return OK."},
    ])

    assert result["text"] == "OK"
    assert result["attempts"] == 2
    assert responses == []


def test_unsupported_reasoning_parameter_is_not_removed(monkeypatch):
    sent = []

    def fake_post(*_args, **kwargs):
        sent.append(kwargs["json"])
        return FakeResponse(
            status_code=400,
            text='{"detail":"Unsupported parameter: reasoning"}',
        )

    monkeypatch.setattr(proxy, "get_access_token", lambda: "test-token")
    monkeypatch.setattr(proxy.requests, "post", fake_post)

    result = proxy.call_responses_api([
        {"role": "user", "content": "Return OK."},
    ])

    assert len(sent) == 1
    assert sent[0]["reasoning"] == {"effort": "none"}
    assert "Unsupported parameter: reasoning" in result["error"]
