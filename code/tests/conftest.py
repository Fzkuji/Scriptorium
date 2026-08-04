import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

# 让 tests 能 import 当前 src 与 scripts package。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

_DATA = os.path.join(_ROOT, "benchmarks", "locomo", "data", "locomo10.json")


@pytest.fixture
def locomo_session():
    """LoCoMo sample0 的 session_1（18 个 turn，均为 dict，有 speaker/dia_id/text）。"""
    with open(_DATA) as f:
        data = json.load(f)
    return data[0]["conversation"]["session_1"]


class _FakeTransport:
    def __init__(self):
        self.calls = 0

    def send(self, payload):
        from baselines.gateways import openai_gpt55_flex_gateway as gateway

        self.calls += 1
        request = json.loads(payload)
        assert request["model"] == gateway.PROVIDER_MODEL
        response = {
            "id": f"fake-provider-response-{self.calls}",
            "model": gateway.PROVIDER_MODEL,
            "service_tier": gateway.SERVICE_TIER,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "fixture"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 2,
                "total_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }
        return gateway.UpstreamResponse(
            200,
            {"x-request-id": f"fake-physical-{self.calls}"},
            json.dumps(response).encode(),
        )


class FakeFlexEvidenceProvider:
    def __init__(self, root: Path):
        from baselines.gateways import openai_gpt55_flex_gateway as gateway

        self.gateway = gateway.GPT55FlexGateway(
            result_root=root / "fake-flex-gateway",
            max_cost_usd="100",
            transport=_FakeTransport(),
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
        (self.gateway.result_root / "gateway_ready.json").write_text(
            json.dumps(ready, indent=2) + "\n", encoding="utf-8"
        )

    @property
    def root(self):
        return self.gateway.result_root

    def close_window(self, request_label: str | None = None):
        """Create a valid local-only provider window for integration fixtures."""
        from baselines.gateways import openai_gpt55_flex_gateway as gateway
        from baselines.gateways import openai_gpt55_flex_gateway_evidence as evidence

        start = evidence.capture_start(self.root)
        response = None
        consumer_records = []
        if request_label is not None:
            status, response = self.gateway.handle(
                {
                    "model": gateway.REQUESTED_MODEL,
                    "messages": [{"role": "user", "content": request_label}],
                    "max_completion_tokens": 16,
                }
            )
            assert status == 200
            meta = response["flex_gateway_meta"]
            consumer_records = [
                {
                    "gateway_request_id": meta["request_id"],
                    "gateway_request_sha256": meta["request_sha256"],
                    "provider_request_sha256": meta["provider_request_sha256"],
                    "provider_actual_model": meta["provider_actual_model"],
                    "service_tier": response["service_tier"],
                    "actual_model": response["model"],
                    "response_id": response["id"],
                }
            ]
        window = evidence.capture_end(start)
        assert evidence.audit_window(
            window, consumer_records=consumer_records
        )["status"] == "passed"
        return {
            "contract": start["contract"],
            "window": window,
            "response": response,
            "consumer_records": consumer_records,
        }

    def attach(self, run_dir: Path, manifest: dict, run_id: str):
        from baselines.gateways import openai_gpt55_flex_gateway as gateway
        from baselines.gateways import openai_gpt55_flex_gateway_evidence as evidence

        start = evidence.capture_start(self.root)
        status, response = self.gateway.handle({
            "model": gateway.REQUESTED_MODEL,
            "messages": [{"role": "user", "content": run_id}],
            "max_completion_tokens": 16,
        })
        assert status == 200
        meta = response["flex_gateway_meta"]
        evidence_dir = run_dir / "provider_evidence" / run_id
        evidence_dir.mkdir(parents=True)
        child_record = {
            "run_id": run_id,
            "started_at": "2026-07-14T00:00:00+00:00",
            "finished_at": "2026-07-14T00:00:01+00:00",
            "status": "success",
            "http_status": 200,
            "requested_model": gateway.REQUESTED_MODEL,
            "requested_service_tier": gateway.SERVICE_TIER,
            "actual_model": response["model"],
            "provider_actual_model": meta["provider_actual_model"],
            "service_tier": response["service_tier"],
            "gateway_request_id": meta["request_id"],
            "gateway_request_sha256": meta["request_sha256"],
            "provider_request_sha256": meta["provider_request_sha256"],
            "response_id": response["id"],
            "request_sha256": "f" * 64,
            "error": None,
        }
        log_path = evidence_dir / evidence.CHILD_LOG_NAME
        log_path.write_text(json.dumps(child_record) + "\n", encoding="utf-8")
        ready_path = evidence_dir / evidence.CHILD_READY_NAME
        child_ready = {
            "run_id": run_id,
            "pid": 456,
            "port": 49001,
            "base_url": "http://127.0.0.1:49001/v1",
            "upstream": start["contract"]["origin"],
            "log": str(log_path.resolve()),
            "started_at": "2026-07-14T00:00:00+00:00",
        }
        ready_path.write_text(
            json.dumps(child_ready, indent=2) + "\n", encoding="utf-8"
        )
        window = evidence.capture_end(start)
        window_path = evidence_dir / evidence.WINDOW_NAME
        window_path.write_text(
            json.dumps(window, indent=2) + "\n", encoding="utf-8"
        )
        stored = {
            "schema": evidence.INVOCATION_SCHEMA,
            "run_id": run_id,
            "gateway_root": str(self.root),
            "provider_model": gateway.PROVIDER_MODEL,
            "returned_alias": gateway.REQUESTED_MODEL,
            "service_tier": gateway.SERVICE_TIER,
            "billing": "api_flex",
            "child_base_url": child_ready["base_url"],
            "requests": 1,
            "gateway_request_ids": [meta["request_id"]],
            "response_ids": [response["id"]],
            "window": {
                "path": str(window_path.resolve()),
                "sha256": hashlib.sha256(window_path.read_bytes()).hexdigest(),
            },
            "consumer_log": {
                "path": str(log_path.resolve()),
                "sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
            },
            "child_ready": {
                "path": str(ready_path.resolve()),
                "sha256": hashlib.sha256(ready_path.read_bytes()).hexdigest(),
            },
        }
        stored["audit"] = evidence.audit_window(
            window, consumer_records=[child_record]
        )
        record_path = evidence_dir / evidence.INVOCATION_NAME
        record_path.write_text(
            json.dumps(stored, indent=2) + "\n", encoding="utf-8"
        )
        manifest_record = dict(stored)
        manifest_record["record_path"] = str(record_path.resolve())
        manifest_record["record_sha256"] = hashlib.sha256(
            record_path.read_bytes()
        ).hexdigest()
        provider = manifest.setdefault("provider_evidence", {
            "schema": "openai-gpt55-flex-invocations/v1",
            "gateway_root": str(self.root),
            "active_run_id": None,
            "invocations": [],
        })
        provider["invocations"].append(manifest_record)
        return manifest_record


@pytest.fixture
def fake_flex_provider(tmp_path):
    return FakeFlexEvidenceProvider(tmp_path)
