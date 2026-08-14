from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from src import openrouter_gpt4o_mini_gateway as gateway
from src import openrouter_gateway_evidence


AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_openrouter_gpt4o_mini_gateway.py"
)
SPEC = importlib.util.spec_from_file_location("openrouter_gateway_audit", AUDIT_PATH)
assert SPEC is not None and SPEC.loader is not None
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


class FakeTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    def send(self, payload: bytes) -> gateway.UpstreamResponse:
        self.payloads.append(json.loads(payload))
        if not self.responses:
            raise AssertionError("unexpected fake transport call")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        assert isinstance(response, gateway.UpstreamResponse)
        return response


class BlockingTransport:
    def __init__(self, response: gateway.UpstreamResponse) -> None:
        self.response = response
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def send(self, payload: bytes) -> gateway.UpstreamResponse:
        del payload
        self.calls += 1
        self.entered.set()
        assert self.release.wait(timeout=5)
        return self.response


class FakeHTTPSConnection:
    def __init__(self, host: str, port: int, *, timeout: float) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.tunnel: tuple[str, int] | None = None
        self.request_args: tuple[object, ...] | None = None
        self.closed = False

    def set_tunnel(self, host: str, port: int) -> None:
        self.tunnel = (host, port)

    def request(self, *args: object, **kwargs: object) -> None:
        self.request_args = args

    def getresponse(self) -> object:
        class Response:
            status = 200

            @staticmethod
            def read() -> bytes:
                return b"{}"

            @staticmethod
            def getheaders() -> list[tuple[str, str]]:
                return []

        return Response()

    def close(self) -> None:
        self.closed = True


def test_transport_uses_explicit_https_proxy_connect_tunnel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[FakeHTTPSConnection] = []

    def build(host: str, port: int, *, timeout: float) -> FakeHTTPSConnection:
        value = FakeHTTPSConnection(host, port, timeout=timeout)
        created.append(value)
        return value

    monkeypatch.setattr(gateway.http.client, "HTTPSConnection", build)
    transport = gateway.OpenRouterChatCompletionsTransport(
        api_key="test-key", https_proxy="http://127.0.0.1:7890"
    )
    assert transport.send(b"{}") == gateway.UpstreamResponse(200, {}, b"{}")
    assert len(created) == 1
    assert (created[0].host, created[0].port) == ("127.0.0.1", 7890)
    assert created[0].tunnel == (gateway.UPSTREAM_HOST, 443)
    assert created[0].closed is True


@pytest.mark.parametrize(
    "proxy",
    ["https://127.0.0.1:7890", "socks5://127.0.0.1:7891", "http://:7890"],
)
def test_transport_rejects_unsupported_https_proxy(proxy: str) -> None:
    with pytest.raises(gateway.GatewayError):
        gateway.OpenRouterChatCompletionsTransport(
            api_key="test-key", https_proxy=proxy
        )


def response(
    *,
    response_id: str = "gen-test-1",
    model: str = gateway.PROVIDER_MODEL,
    prompt: int = 1_000,
    cached: int = 200,
    completion: int = 100,
    cost: object = "0.000195",
) -> gateway.UpstreamResponse:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    if cost is not None:
        usage["cost"] = cost
    return gateway.UpstreamResponse(
        200,
        {"x-request-id": f"upstream-{response_id}"},
        json.dumps(
            {
                "id": response_id,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": usage,
            }
        ).encode(),
    )


def request(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "model": gateway.REQUESTED_MODEL,
        "messages": [{"role": "user", "content": "test"}],
        "temperature": 0,
        "max_tokens": 400,
    }
    value.update(updates)
    return value


def make_gateway(
    root: Path,
    transport: object,
    *,
    cap: str = "1",
) -> gateway.OpenRouterCostGateway:
    return gateway.OpenRouterCostGateway(
        result_root=root,
        max_cost_usd=cap,
        transport=transport,  # type: ignore[arg-type]
    )


def log_entries(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / gateway.REQUEST_LOG_NAME).read_text().splitlines()
        if line.strip()
    ]


def test_alias_snapshot_tools_exact_cost_and_independent_audit(tmp_path: Path) -> None:
    transport = FakeTransport([response()])
    instance = make_gateway(tmp_path / "openrouter", transport)
    payload = request(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="auto",
    )
    status, returned = instance.handle(payload)
    assert status == 200
    assert returned["model"] == gateway.REQUESTED_MODEL
    assert returned["openrouter_gateway_meta"] == {
        "schema": "openrouter-gpt4o-mini-response-meta/v1",
        "request_id": returned["openrouter_gateway_meta"]["request_id"],
        "request_sha256": returned["openrouter_gateway_meta"]["request_sha256"],
        "provider_request_sha256": returned["openrouter_gateway_meta"][
            "provider_request_sha256"
        ],
        "provider_actual_model": gateway.PROVIDER_MODEL,
        "returned_alias": gateway.REQUESTED_MODEL,
        "response_id": "gen-test-1",
        "provider_usage_cost_usd": "0.000195",
        "token_derived_cost_usd": "0.000195",
        "committed_cost_source": "provider_usage_cost",
        "committed_cost_usd": "0.000195",
    }
    outbound = transport.payloads[0]
    assert outbound["model"] == gateway.PROVIDER_MODEL
    assert outbound["stream"] is False
    assert outbound["provider"] == {"allow_fallbacks": False}
    assert outbound["tools"] == payload["tools"]
    assert outbound["tool_choice"] == "auto"
    state = json.loads((instance.state_path).read_text())
    assert state["committed_cost_nanos"] == 195_000
    assert state["provider_exact_cost_count"] == 1
    report = audit_module.audit(instance.result_root, max_cost_usd="1")
    assert report["status"] == "passed"
    assert report["cost_sources"] == {
        "provider_usage_cost": 1,
        "token_derived": 0,
    }


def test_missing_usage_cost_commits_token_derived_cost(tmp_path: Path) -> None:
    transport = FakeTransport([response(cost=None)])
    instance = make_gateway(tmp_path / "root", transport)
    status, _ = instance.handle(request())
    assert status == 200
    entry = log_entries(instance.result_root)[0]
    assert entry["provider_usage_cost_nanos"] is None
    assert entry["token_derived_cost_nanos"] == 195_000
    assert entry["committed_cost_source"] == "token_derived"
    assert audit_module.audit(instance.result_root, max_cost_usd="1")[
        "status"
    ] == "passed"


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("gpt-4o-mini", "model_mismatch"),
        (gateway.PROVIDER_MODEL, "model_mismatch"),
        ("anthropic/claude", "model_mismatch"),
    ],
)
def test_local_gateway_accepts_only_alias(
    tmp_path: Path, value: str, code: str
) -> None:
    transport = FakeTransport([])
    instance = make_gateway(tmp_path / value.replace("/", "_"), transport)
    status, returned = instance.handle(request(model=value))
    assert status == 400
    assert returned["error"]["code"] == code
    assert transport.payloads == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("models", [gateway.PROVIDER_MODEL, "openai/gpt-4.1-mini"]),
        ("route", "fallback"),
        ("provider", {"allow_fallbacks": True}),
        ("fallbacks", ["openai/gpt-4.1-mini"]),
    ],
)
def test_cross_model_routing_fields_fail_before_transport(
    tmp_path: Path, field: str, value: object
) -> None:
    transport = FakeTransport([])
    instance = make_gateway(tmp_path / field, transport)
    status, returned = instance.handle(request(**{field: value}))
    assert status == 400
    assert returned["error"]["code"] == "model_routing_not_allowed"
    assert transport.payloads == []
    state = json.loads(instance.state_path.read_text())
    assert state["billable_request_count"] == 0
    assert state["reserved_cost_nanos"] == 0


def test_provider_snapshot_mismatch_is_billable_error(tmp_path: Path) -> None:
    transport = FakeTransport([response(model="openai/gpt-4o-mini")])
    instance = make_gateway(tmp_path / "root", transport)
    status, returned = instance.handle(request())
    assert status == 502
    assert returned["error"]["code"] == "provider_model_mismatch"
    state = json.loads(instance.state_path.read_text())
    assert state["billable_request_count"] == 1
    assert state["reserved_cost_nanos"] == 0
    report = audit_module.audit(instance.result_root, max_cost_usd="1")
    assert report["status"] == "passed"


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("Authorization: Bearer or-secret-value"),
        gateway.UpstreamResponse(
            429,
            {},
            b'{"error":{"type":"rate_limit","message":"sk-secret-value"}}',
        ),
        gateway.UpstreamResponse(
            500, {}, b'{"error":{"type":"server_error","message":"failed"}}'
        ),
    ],
)
def test_uncertain_transport_and_upstream_errors_retain_without_retry(
    tmp_path: Path, failure: object
) -> None:
    transport = FakeTransport([failure])
    instance = make_gateway(tmp_path / "root", transport)
    status, _ = instance.handle(request())
    assert status == 502
    assert len(transport.payloads) == 1
    state = json.loads(instance.state_path.read_text())
    assert state["retained_failure_count"] == 1
    assert state["reserved_cost_nanos"] > 0
    entry = log_entries(instance.result_root)[0]
    assert entry["reservation_retained"] is True
    serialized = json.dumps(entry)
    assert "or-secret-value" not in serialized
    assert "sk-secret-value" not in serialized
    with pytest.raises(audit_module.AuditError, match="residual uncertain"):
        audit_module.audit(instance.result_root, max_cost_usd="1")


def test_usage_or_cost_uncertainty_retains_reservation(tmp_path: Path) -> None:
    invalid_cost = response(cost="0.0000000001")
    transport = FakeTransport([invalid_cost])
    instance = make_gateway(tmp_path / "root", transport)
    status, returned = instance.handle(request())
    assert status == 502
    assert returned["error"]["code"] == "upstream_usage_cost_billing_uncertain"
    assert json.loads(instance.state_path.read_text())["reserved_cost_nanos"] > 0


def test_budget_reservation_is_concurrent_and_fail_closed(tmp_path: Path) -> None:
    maximum = gateway.maximum_request_cost_nanos(400)
    cap = gateway.nanos_to_usd(maximum)
    transport = BlockingTransport(response())
    instance = make_gateway(tmp_path / "root", transport, cap=cap)
    first: list[tuple[int, dict[str, Any]]] = []

    thread = threading.Thread(target=lambda: first.append(instance.handle(request())))
    thread.start()
    assert transport.entered.wait(timeout=5)
    status, rejected = instance.handle(request())
    assert status == 402
    assert rejected["error"]["code"] == "local_cost_budget_exceeded"
    assert transport.calls == 1
    transport.release.set()
    thread.join(timeout=5)
    assert first[0][0] == 200
    assert audit_module.audit(instance.result_root, max_cost_usd=cap)[
        "status"
    ] == "passed"


def test_cap_and_root_identity_are_immutable(tmp_path: Path) -> None:
    root = tmp_path / "root"
    make_gateway(root, FakeTransport([]), cap="2")
    with pytest.raises(gateway.StateError, match="max_cost"):
        make_gateway(root, FakeTransport([]), cap="3")
    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()
    (unmarked / "foreign.json").write_text("{}")
    with pytest.raises(gateway.StateError, match="non-empty"):
        make_gateway(unmarked, FakeTransport([]))


def test_auditor_rejects_state_log_and_model_tampering(tmp_path: Path) -> None:
    root = tmp_path / "root"
    instance = make_gateway(root, FakeTransport([response()]))
    assert instance.handle(request())[0] == 200
    state = json.loads(instance.state_path.read_text())
    state["committed_cost_nanos"] += 1
    instance.state_path.write_text(json.dumps(state) + "\n")
    with pytest.raises(audit_module.AuditError, match="committed_cost_nanos"):
        audit_module.audit(root, max_cost_usd="1")

    root2 = tmp_path / "root2"
    instance2 = make_gateway(root2, FakeTransport([response()]))
    assert instance2.handle(request())[0] == 200
    entries = log_entries(root2)
    entries[0]["provider_actual_model"] = "openai/gpt-4o-mini"
    (root2 / gateway.REQUEST_LOG_NAME).write_text(
        "".join(json.dumps(item) + "\n" for item in entries)
    )
    with pytest.raises(audit_module.AuditError, match="provider model"):
        audit_module.audit(root2, max_cost_usd="1")

    root3 = tmp_path / "root3"
    instance3 = make_gateway(root3, FakeTransport([response()]))
    assert instance3.handle(request())[0] == 200
    entries = log_entries(root3)
    entries[0]["provider_usage_cost_nanos"] += 1
    (root3 / gateway.REQUEST_LOG_NAME).write_text(
        "".join(json.dumps(item) + "\n" for item in entries)
    )
    with pytest.raises(audit_module.AuditError, match="usage.cost"):
        audit_module.audit(root3, max_cost_usd="1")


def test_log_contains_hashes_and_no_request_payload_or_key(tmp_path: Path) -> None:
    secret = "or-secret-not-for-log"
    instance = make_gateway(tmp_path / "root", FakeTransport([response()]))
    assert instance.handle(
        request(messages=[{"role": "user", "content": secret}])
    )[0] == 200
    text = (instance.result_root / gateway.REQUEST_LOG_NAME).read_text()
    assert secret not in text
    entry = json.loads(text)
    assert len(entry["request_sha256"]) == 64
    assert len(entry["provider_request_sha256"]) == 64
    assert len(entry["response_sha256"]) == 64
    assert "messages" not in text


def test_production_endpoint_is_fixed() -> None:
    source = Path(gateway.__file__).read_text()
    assert 'UPSTREAM_HOST = "openrouter.ai"' in source
    assert 'UPSTREAM_PATH = "/api/v1/chat/completions"' in source
    assert "--upstream" not in source
    assert gateway.PROVIDER_MODEL == "openai/gpt-4o-mini-2024-07-18"


def test_consumer_binding_requires_marked_live_loopback_gateway(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    instance = make_gateway(root, FakeTransport([response(response_id="bound-1")]))
    gateway._write_ready(
        root / gateway.READY_NAME, server_port=8421, gateway=instance
    )
    binding = openrouter_gateway_evidence.capture_binding(
        root, base_url="http://127.0.0.1:8421/v1"
    )
    assert binding["provider"] == "OpenRouter"
    assert binding["provider_model"] == gateway.PROVIDER_MODEL
    assert instance.handle(request())[0] == 200
    final_binding = openrouter_gateway_evidence.finalize_binding(binding)
    assert openrouter_gateway_evidence.verify_response_ids(
        final_binding, ["bound-1"]
    )["bound_response_ids"] == 1
    with pytest.raises(
        openrouter_gateway_evidence.GatewayEvidenceError,
        match="marked loopback",
    ):
        openrouter_gateway_evidence.capture_binding(
            root, base_url="https://openrouter.ai/api/v1"
        )
    (root / gateway.READY_NAME).unlink()
    # Frozen bindings remain independently checkable after gateway shutdown.
    assert openrouter_gateway_evidence.validate_binding(
        binding,
        result_root=root,
        base_url="http://127.0.0.1:8421/v1",
    ) == binding


def test_primary_scorer_and_beam_require_marked_gateway_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import evaluate_v88_gpt55_beam as beam
    from scripts import run_r301_organizer_retriever as r301
    from scripts import score_v88_gpt55_benchmarks as scorer

    root = tmp_path / "root"
    instance = make_gateway(root, FakeTransport([]))
    gateway._write_ready(
        root / gateway.READY_NAME, server_port=8422, gateway=instance
    )
    binding = openrouter_gateway_evidence.capture_binding(
        root, base_url="http://127.0.0.1:8422/v1"
    )
    assert r301.capture_formal_openrouter_gateway(
        root, upstream="http://127.0.0.1:8422/v1"
    ) == binding
    with pytest.raises(r301.R301Error, match="marked loopback"):
        r301.capture_formal_openrouter_gateway(
            root, upstream="https://openrouter.ai/api/v1"
        )
    for name in ("JUDGE_MODEL", "JUDGE_BASE", "JUDGE_KEY", "JUDGE_HTTP_RETRIES"):
        monkeypatch.setenv(name, "test-placeholder")
    with pytest.raises(scorer.ScoringError, match="marked loopback"):
        scorer.configure_judge("primary")
    profile = scorer.configure_judge(
        "primary", openrouter_gateway=binding
    )
    assert profile["base_url"] == "http://127.0.0.1:8422/v1"
    assert scorer.os.environ["JUDGE_HTTP_RETRIES"] == "1"

    config = beam.evaluation_config(
        profile="primary",
        base_url="http://127.0.0.1:8422/v1",
        max_tokens=400,
        max_retries=1,
        proxy_log=None,
        openrouter_gateway=binding,
        formal_transport_contract=True,
    )
    beam.validate_formal_config(config)
    assert config["transport_contract"] == "marked_openrouter_gateway"
    assert config["base_url"] == binding["base_url"]
    direct = dict(config)
    direct["base_url"] = "https://openrouter.ai/api/v1"
    with pytest.raises(beam.EvaluationError, match="gateway base URL"):
        beam.validate_formal_config(direct)
