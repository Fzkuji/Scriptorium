from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from src import openai_gpt55_flex_gateway as gateway
from src import openai_gpt55_flex_gateway_evidence as evidence


class FakeTransport:
    def __init__(self) -> None:
        self.calls = 0

    def send(self, payload: bytes) -> gateway.UpstreamResponse:
        self.calls += 1
        request = json.loads(payload)
        assert request["model"] == gateway.PROVIDER_MODEL
        assert request["service_tier"] == gateway.SERVICE_TIER
        response = {
            "id": f"provider-response-{self.calls}",
            "model": gateway.PROVIDER_MODEL,
            "service_tier": gateway.SERVICE_TIER,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 1},
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        }
        return gateway.UpstreamResponse(
            200,
            {"x-request-id": f"physical-{self.calls}"},
            json.dumps(response).encode(),
        )


def active_gateway(tmp_path: Path) -> gateway.GPT55FlexGateway:
    result = gateway.GPT55FlexGateway(
        result_root=tmp_path / "gateway",
        max_cost_usd="100",
        transport=FakeTransport(),
    )
    ready = {
        "schema": "openai-gpt55-flex-ready/v1",
        "pid": 123,
        "base_url": "http://127.0.0.1:38401/v1",
        "health_url": "http://127.0.0.1:38401/healthz",
        "requested_model": gateway.REQUESTED_MODEL,
        "provider_model": gateway.PROVIDER_MODEL,
        "service_tier": gateway.SERVICE_TIER,
        "max_cost_usd": "100",
        "started_at": "2026-07-14T00:00:00+00:00",
    }
    (result.result_root / "gateway_ready.json").write_text(
        json.dumps(ready, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def one_call(
    instance: gateway.GPT55FlexGateway,
    index: int,
) -> dict[str, object]:
    status, response = instance.handle(
        {
            "model": gateway.REQUESTED_MODEL,
            "messages": [{"role": "user", "content": f"question {index}"}],
            "max_completion_tokens": 16,
        }
    )
    assert status == 200
    meta = response["flex_gateway_meta"]
    return {
        "actual_model": response["model"],
        "provider_actual_model": meta["provider_actual_model"],
        "service_tier": meta["service_tier"],
        "gateway_request_id": meta["request_id"],
        "gateway_request_sha256": meta["request_sha256"],
        "provider_request_sha256": meta["provider_request_sha256"],
        "response_id": response["id"],
    }


def test_window_binds_root_state_prefix_snapshot_tier_and_consumer_ids(tmp_path):
    instance = active_gateway(tmp_path)
    start = evidence.capture_start(instance.result_root)
    consumers = [one_call(instance, 1), one_call(instance, 2)]
    closed = evidence.capture_end(start)
    report = evidence.audit_window(closed, consumer_records=consumers)
    assert report["status"] == "passed"
    assert report["provider_model"] == gateway.PROVIDER_MODEL
    assert report["service_tier"] == "flex"
    assert report["requests"] == 2
    assert report["committed_cost_nanos"] > 0

    # Later consumers may append to the shared root without invalidating this
    # bounded prefix.
    one_call(instance, 3)
    assert evidence.audit_window(closed, consumer_records=consumers)["requests"] == 2


def test_window_rejects_consumer_mismatch_and_log_prefix_rewrite(tmp_path):
    instance = active_gateway(tmp_path)
    start = evidence.capture_start(instance.result_root)
    consumers = [one_call(instance, 1)]
    closed = evidence.capture_end(start)

    bad_consumer = copy.deepcopy(consumers)
    bad_consumer[0]["provider_actual_model"] = "gpt-5.5"
    with pytest.raises(evidence.EvidenceError, match="provider_actual_model"):
        evidence.audit_window(closed, consumer_records=bad_consumer)

    log_path = instance.result_root / gateway.REQUEST_LOG_NAME
    payload = bytearray(log_path.read_bytes())
    payload[0] = ord("[")
    log_path.write_bytes(payload)
    with pytest.raises(evidence.EvidenceError, match="prefix hash"):
        evidence.audit_window(closed, consumer_records=consumers)


def test_capture_requires_idle_serial_gateway_state(tmp_path):
    instance = active_gateway(tmp_path)
    request_id = "retained-test-reservation"
    instance.cost_store.reserve(
        request_id=request_id,
        cost_nanos=1,
        request_sha256="a" * 64,
        max_completion_tokens=1,
    )
    with pytest.raises(evidence.EvidenceError, match="reservations"):
        evidence.capture_start(instance.result_root)


def test_child_proxy_lifecycle_binds_dynamic_gateway_and_exact_ids(tmp_path):
    instance = gateway.GPT55FlexGateway(
        result_root=tmp_path / "gateway",
        max_cost_usd="100",
        transport=FakeTransport(),
    )
    server = gateway.ThreadedHTTPServer(
        ("127.0.0.1", 0), gateway.make_handler(instance)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    ready = {
        "schema": "openai-gpt55-flex-ready/v1",
        "pid": 123,
        "base_url": f"http://127.0.0.1:{port}/v1",
        "health_url": f"http://127.0.0.1:{port}/healthz",
        "requested_model": gateway.REQUESTED_MODEL,
        "provider_model": gateway.PROVIDER_MODEL,
        "service_tier": gateway.SERVICE_TIER,
        "max_cost_usd": "100",
        "started_at": "2026-07-14T00:00:00+00:00",
    }
    (instance.result_root / "gateway_ready.json").write_text(
        json.dumps(ready, indent=2) + "\n", encoding="utf-8"
    )
    invocation = None
    try:
        invocation = evidence.begin_child_invocation(
            instance.result_root,
            tmp_path / "consumer-evidence",
            run_id="fixture-child",
        )
        assert invocation.base_url != ready["base_url"]
        payload = json.dumps({
            "model": gateway.REQUESTED_MODEL,
            "service_tier": gateway.SERVICE_TIER,
            "messages": [{"role": "user", "content": "fixture"}],
            "max_completion_tokens": 16,
        }).encode()
        request = Request(
            f"{invocation.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with build_opener(ProxyHandler({})).open(
            request, timeout=5
        ) as response:  # noqa: S310
            assert response.status == 200
            result = json.loads(response.read())
        record = invocation.finish()
        invocation = None
        assert record["requests"] == 1
        assert record["response_ids"] == [result["id"]]
        assert evidence.audit_invocation(record)["requests"] == 1
    finally:
        if invocation is not None:
            invocation.abort()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
