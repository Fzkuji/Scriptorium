from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from baselines.gateways import audit_openai_gpt55_flex_gateway as auditor
from baselines.gateways import openai_gpt55_flex_gateway as gateway_module


class FakeTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.payloads: list[bytes] = []
        self._lock = threading.Lock()

    def send(self, payload: bytes) -> gateway_module.UpstreamResponse:
        with self._lock:
            self.payloads.append(payload)
            if not self.responses:
                raise AssertionError("fake upstream has no queued response")
            value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        assert isinstance(value, gateway_module.UpstreamResponse)
        return value


class BlockingTransport:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.payloads: list[bytes] = []

    def send(self, payload: bytes) -> gateway_module.UpstreamResponse:
        self.payloads.append(payload)
        self.entered.set()
        assert self.release.wait(timeout=5)
        return success_response(prompt=1, cached=0, completion=1, reasoning=0)


@pytest.fixture(autouse=True)
def prohibit_real_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_connection(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("network access is forbidden in gateway tests")

    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden_connection)


def success_response(
    *,
    prompt: int = 100,
    cached: int = 20,
    completion: int = 10,
    reasoning: int = 4,
    model: str = gateway_module.PROVIDER_MODEL,
    service_tier: str = gateway_module.SERVICE_TIER,
    response_id: str = "chatcmpl-fake",
    message: dict[str, Any] | None = None,
) -> gateway_module.UpstreamResponse:
    value = {
        "id": response_id,
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "service_tier": service_tier,
        "choices": [
            {
                "index": 0,
                "message": message
                or {"role": "assistant", "content": "fake response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }
    return gateway_module.UpstreamResponse(
        200,
        {"x-request-id": f"req-{response_id}"},
        json.dumps(value).encode(),
    )


def resource_unavailable() -> gateway_module.UpstreamResponse:
    return gateway_module.UpstreamResponse(
        429,
        {"x-request-id": "req-capacity"},
        json.dumps(
            {
                "error": {
                    "message": "Resource unavailable for Flex processing",
                    "type": "resource_unavailable",
                    "code": "resource_unavailable",
                }
            }
        ).encode(),
    )


def request(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "test only"}],
        "max_tokens": 64,
    }
    value.update(overrides)
    return value


def make_gateway(
    root: Path,
    transport: object,
    *,
    max_cost_usd: str = "100",
    attempts: int = 5,
    sleep=lambda _delay: None,
) -> gateway_module.GPT55FlexGateway:
    return gateway_module.GPT55FlexGateway(
        result_root=root,
        max_cost_usd=max_cost_usd,
        transport=transport,  # type: ignore[arg-type]
        max_physical_attempts=attempts,
        initial_backoff_seconds=1,
        sleep=sleep,
    )


def read_log(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / gateway_module.REQUEST_LOG_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_flex_pricing_cached_reasoning_and_long_context() -> None:
    regular = gateway_module.calculate_cost_nanos(
        prompt_tokens=1_000,
        cached_tokens=400,
        completion_tokens=100,
    )
    assert regular == 3_100_000

    boundary = gateway_module.calculate_cost_nanos(
        prompt_tokens=272_000,
        cached_tokens=1_000,
        completion_tokens=100,
    )
    expected_boundary = 271_000 * 2_500 + 1_000 * 250 + 100 * 15_000
    assert boundary == expected_boundary

    long_context = gateway_module.calculate_cost_nanos(
        prompt_tokens=272_001,
        cached_tokens=1_000,
        completion_tokens=100,
    )
    expected_long = 271_001 * 5_000 + 1_000 * 500 + 100 * 22_500
    assert long_context == expected_long
    assert gateway_module.nanos_to_usd(regular) == "0.0031"


def test_alias_conversion_tools_temperature_usage_and_offline_audit(
    tmp_path: Path,
) -> None:
    tool_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"id":1}'},
            }
        ],
    }
    transport = FakeTransport([success_response(message=tool_message)])
    gateway = make_gateway(tmp_path / "flex", transport)
    local_request = request(
        temperature=0.2,
        messages=[
            {"role": "user", "content": "find"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "prior-call",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "prior-call", "content": "done"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )
    status, response = gateway.handle(local_request)

    assert status == 200
    assert response["model"] == "gpt-5.5"
    assert response["service_tier"] == "flex"
    assert response["choices"][0]["message"]["tool_calls"][0]["id"] == "call-1"
    assert response["proxy_meta"]["attempts"] == 1
    assert response["proxy_meta"]["provider_actual_model"] == (
        "gpt-5.5-2026-04-23"
    )
    outbound = json.loads(transport.payloads[0])
    assert outbound["model"] == "gpt-5.5-2026-04-23"
    assert outbound["service_tier"] == "flex"
    assert outbound["max_completion_tokens"] == 64
    assert "max_tokens" not in outbound
    assert outbound["temperature"] == 0.2
    assert outbound["messages"][1]["tool_calls"][0]["id"] == "prior-call"
    assert outbound["tools"][0]["function"]["name"] == "lookup"

    entry = read_log(tmp_path / "flex")[0]
    assert entry["provider_actual_model"] == "gpt-5.5-2026-04-23"
    assert entry["actual_model"] == "gpt-5.5"
    assert entry["returned_alias"] == "gpt-5.5"
    assert entry["service_tier"] == "flex"
    assert entry["response_id"] == "chatcmpl-fake"
    assert entry["usage"]["cached_tokens"] == 20
    assert entry["usage"]["reasoning_tokens"] == 4
    assert entry["physical_attempt_count"] == 1
    assert entry["physical_retry_count"] == 0
    assert entry["transformations"] == ["max_tokens_to_max_completion_tokens"]
    assert entry["max_completion_tokens"] == 64
    assert entry["reservation_nanos"] == gateway_module.maximum_request_cost_nanos(64)

    report = auditor.audit(
        result_root=tmp_path / "flex",
        max_cost_usd="100",
    )
    assert report["status"] == "pass"
    assert report["billable_request_count"] == 1


@pytest.mark.parametrize(
    ("bad_request", "expected_code"),
    [
        (request(model="gpt-5.5-2026-04-23"), "model_mismatch"),
        (request(service_tier="standard"), "service_tier_mismatch"),
        (request(service_tier="auto"), "service_tier_mismatch"),
    ],
)
def test_local_model_and_service_tier_mismatch_fail_before_upstream(
    tmp_path: Path,
    bad_request: dict[str, Any],
    expected_code: str,
) -> None:
    transport = FakeTransport([])
    gateway = make_gateway(tmp_path / expected_code, transport)
    status, response = gateway.handle(bad_request)
    assert status == 400
    assert response["error"]["code"] == expected_code
    assert transport.payloads == []
    entry = read_log(tmp_path / expected_code)[0]
    assert entry["physical_attempt_count"] == 0
    assert entry["billable"] is False


@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (
            success_response(model="gpt-5.5"),
            "provider_model_mismatch",
        ),
        (
            success_response(service_tier="default"),
            "provider_service_tier_mismatch",
        ),
    ],
)
def test_provider_model_or_tier_mismatch_is_billable_and_fails_closed(
    tmp_path: Path,
    response: gateway_module.UpstreamResponse,
    expected_code: str,
) -> None:
    root = tmp_path / expected_code
    gateway = make_gateway(root, FakeTransport([response]))
    status, returned = gateway.handle(request())
    assert status == 502
    assert returned["error"]["code"] == expected_code
    entry = read_log(root)[0]
    assert entry["status"] == "error"
    assert entry["billable"] is True
    assert entry["cost_nanos"] > 0
    state = json.loads((root / gateway_module.STATE_NAME).read_text())
    assert state["committed_cost_nanos"] == entry["cost_nanos"]
    report = auditor.audit(result_root=root, max_cost_usd="100")
    assert report["billable_request_count"] == 1
    assert report["error_count"] == 1


def test_resource_unavailable_exponential_backoff_never_changes_flex_payload(
    tmp_path: Path,
) -> None:
    delays: list[float] = []
    transport = FakeTransport(
        [resource_unavailable(), resource_unavailable(), success_response()]
    )
    gateway = make_gateway(
        tmp_path / "retry",
        transport,
        attempts=3,
        sleep=delays.append,
    )
    status, response = gateway.handle(request())
    assert status == 200
    assert response["model"] == "gpt-5.5"
    assert delays == [1, 2]
    assert len(transport.payloads) == 3
    assert len(set(transport.payloads)) == 1
    for payload in transport.payloads:
        outbound = json.loads(payload)
        assert outbound["service_tier"] == "flex"
        assert outbound["model"] == "gpt-5.5-2026-04-23"
    entry = read_log(tmp_path / "retry")[0]
    assert entry["physical_attempt_count"] == 3
    assert entry["physical_retry_count"] == 2
    assert [item["http_status"] for item in entry["physical_attempts"]] == [
        429,
        429,
        200,
    ]


def test_resource_unavailable_exhaustion_releases_reservation(tmp_path: Path) -> None:
    transport = FakeTransport([resource_unavailable(), resource_unavailable()])
    root = tmp_path / "unavailable"
    gateway = make_gateway(root, transport, attempts=2)
    status, response = gateway.handle(request())
    assert status == 503
    assert response["error"]["code"] == "flex_resource_unavailable"
    state = gateway.cost_store.snapshot()
    assert state["committed_cost_nanos"] == 0
    assert state["reserved_cost_nanos"] == 0
    assert state["reservations"] == {}
    assert all(json.loads(payload)["service_tier"] == "flex" for payload in transport.payloads)


def test_transport_uncertainty_retains_reservation_and_audit_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "uncertain"
    gateway = make_gateway(root, FakeTransport([TimeoutError("fake timeout")]))
    status, response = gateway.handle(request(max_tokens=1))
    assert status == 502
    assert response["error"]["code"] == "upstream_transport_billing_uncertain"
    state = gateway.cost_store.snapshot()
    assert state["committed_cost_nanos"] == 0
    assert state["reserved_cost_nanos"] > 0
    assert len(state["reservations"]) == 1
    entry = read_log(root)[0]
    assert entry["reservation_retained"] is True
    with pytest.raises(auditor.AuditError, match="in-flight reservations remain"):
        auditor.audit(result_root=root, max_cost_usd="100")
    report = auditor.audit(
        result_root=root,
        max_cost_usd="100",
        allow_in_flight=True,
    )
    assert report["in_flight"] == 1


def test_concurrent_budget_gate_counts_in_flight_reservation(tmp_path: Path) -> None:
    transport = BlockingTransport()
    root = tmp_path / "budget"
    gateway = make_gateway(root, transport, max_cost_usd="5.26")
    competing_transport = FakeTransport([])
    competing_gateway = make_gateway(
        root,
        competing_transport,
        max_cost_usd="5.26",
    )
    first_result: list[tuple[int, dict[str, Any]]] = []
    first = threading.Thread(
        target=lambda: first_result.append(
            gateway.handle(request(max_tokens=1))
        ),
    )
    first.start()
    assert transport.entered.wait(timeout=5)

    status, response = competing_gateway.handle(request(max_tokens=1))
    assert status == 402
    assert response["error"]["code"] == "local_cost_budget_exceeded"
    assert competing_transport.payloads == []
    in_flight = gateway.cost_store.snapshot()
    assert len(in_flight["reservations"]) == 1
    assert in_flight["committed_cost_nanos"] + in_flight["reserved_cost_nanos"] <= (
        gateway.max_cost_nanos
    )

    transport.release.set()
    first.join(timeout=5)
    assert first_result[0][0] == 200
    final = gateway.cost_store.snapshot()
    assert final["reserved_cost_nanos"] == 0
    assert final["committed_cost_nanos"] <= gateway.max_cost_nanos
    report = auditor.audit(result_root=root, max_cost_usd="5.26")
    assert report["success_count"] == 1
    assert report["error_count"] == 1


def test_health_does_not_expose_secret_or_paths(tmp_path: Path) -> None:
    root = tmp_path / "private" / "gateway"
    gateway = make_gateway(root, FakeTransport([]))
    rendered = json.dumps(gateway.health(), sort_keys=True)
    assert "secret-api-key" not in rendered
    assert str(root) not in rendered
    assert "OPENAI_API_KEY" not in rendered
    health = gateway.health()
    assert health["provider_model"] == "gpt-5.5-2026-04-23"
    assert health["service_tier"] == "flex"
    assert health["budget"]["in_flight"] == 0


def test_nonempty_unmarked_root_is_rejected_to_prevent_subscription_mixing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "subscription-results"
    root.mkdir()
    (root / "chatgpt_subscription_proxy.jsonl").write_text("{}\n")
    with pytest.raises(gateway_module.StateError, match="separate roots"):
        make_gateway(root, FakeTransport([]))


def test_state_resume_requires_identical_cost_cap(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    first = make_gateway(root, FakeTransport([success_response()]), max_cost_usd="100")
    assert first.handle(request())[0] == 200
    resumed = make_gateway(root, FakeTransport([]), max_cost_usd="100")
    assert resumed.cost_store.snapshot()["billable_request_count"] == 1
    with pytest.raises(gateway_module.StateError, match="max_cost_nanos mismatch"):
        make_gateway(root, FakeTransport([]), max_cost_usd="101")


def test_independent_auditor_rejects_tampered_service_tier(tmp_path: Path) -> None:
    root = tmp_path / "tamper"
    gateway = make_gateway(root, FakeTransport([success_response()]))
    assert gateway.handle(request())[0] == 200
    log_path = root / gateway_module.REQUEST_LOG_NAME
    entry = read_log(root)[0]
    entry["service_tier"] = "default"
    log_path.write_text(json.dumps(entry, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(auditor.AuditError, match="service tier mismatch"):
        auditor.audit(result_root=root, max_cost_usd="100")


def test_independent_auditor_rejects_tampered_reservation_and_failure_count(
    tmp_path: Path,
) -> None:
    reservation_root = tmp_path / "reservation-tamper"
    reservation_gateway = make_gateway(
        reservation_root,
        FakeTransport([success_response()]),
    )
    assert reservation_gateway.handle(request(max_tokens=32))[0] == 200
    log_path = reservation_root / gateway_module.REQUEST_LOG_NAME
    entry = read_log(reservation_root)[0]
    entry["reservation_nanos"] += 1
    log_path.write_text(json.dumps(entry, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(auditor.AuditError, match="reservation differs"):
        auditor.audit(result_root=reservation_root, max_cost_usd="100")

    failure_root = tmp_path / "failure-count-tamper"
    failure_gateway = make_gateway(
        failure_root,
        FakeTransport([resource_unavailable()]),
        attempts=1,
    )
    assert failure_gateway.handle(request())[0] == 503
    state_path = failure_root / gateway_module.STATE_NAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["failed_request_count"] = 0
    state_path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(auditor.AuditError, match="released-failure count"):
        auditor.audit(result_root=failure_root, max_cost_usd="100")


def test_production_transport_endpoint_is_fixed_in_source() -> None:
    assert gateway_module.UPSTREAM_HOST == "api.openai.com"
    assert gateway_module.UPSTREAM_PATH == "/v1/chat/completions"
    assert gateway_module.PROVIDER_MODEL == "gpt-5.5-2026-04-23"
    assert gateway_module.SERVICE_TIER == "flex"
