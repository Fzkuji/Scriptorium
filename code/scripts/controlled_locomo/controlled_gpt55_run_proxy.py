#!/usr/bin/env python3
"""Question-linked extension of the frozen per-run GPT-5.5 proxy wrapper.

The retrieval baseline wrapper remains byte-identical to its preregistered
source.  This module reuses its transport, server, fsync, health, and atomic
ready-file primitives while adding answer-question linkage and inner-proxy
attempt metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import gpt55_run_proxy as base


HEALTH_SCHEMA = "openai-gpt55-flex-health/v1"
REQUESTED_MODEL = "gpt-5.5"
PROVIDER_MODEL = "gpt-5.5-2026-04-23"
SERVICE_TIER = "flex"
_HEALTH_FIELDS = {
    "status",
    "schema",
    "requested_model",
    "provider_model",
    "service_tier",
    "budget",
}
_BUDGET_FIELDS = {
    "max_cost_usd",
    "committed_cost_usd",
    "reserved_cost_usd",
    "remaining_cost_usd",
    "in_flight",
}
_GATEWAY_META_FIELDS = {
    "request_id",
    "request_sha256",
    "provider_request_sha256",
    "provider_actual_model",
    "returned_alias",
    "service_tier",
    "physical_attempt_count",
    "physical_retry_count",
}


def _safe_header(handler: BaseHTTPRequestHandler, name: str) -> str | None:
    value = handler.headers.get(name) or None
    return value[:200] if value is not None else None


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_budget(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _BUDGET_FIELDS:
        raise ValueError("upstream budget schema differs")
    parsed: dict[str, Decimal] = {}
    for field in (
        "max_cost_usd",
        "committed_cost_usd",
        "reserved_cost_usd",
        "remaining_cost_usd",
    ):
        raw = value.get(field)
        if not isinstance(raw, str):
            raise ValueError(f"upstream budget {field} is not a decimal string")
        try:
            amount = Decimal(raw)
        except InvalidOperation as exc:
            raise ValueError(f"upstream budget {field} is invalid") from exc
        if not amount.is_finite() or amount < 0:
            raise ValueError(f"upstream budget {field} is negative or non-finite")
        parsed[field] = amount
    if parsed["max_cost_usd"] <= 0:
        raise ValueError("upstream maximum budget is not positive")
    if (
        parsed["committed_cost_usd"]
        + parsed["reserved_cost_usd"]
        + parsed["remaining_cost_usd"]
        != parsed["max_cost_usd"]
    ):
        raise ValueError("upstream budget components do not reconcile")
    in_flight = value.get("in_flight")
    if not isinstance(in_flight, int) or isinstance(in_flight, bool) or in_flight < 0:
        raise ValueError("upstream in-flight reservation count is invalid")
    return {field: value[field] for field in sorted(_BUDGET_FIELDS)}


def _validate_health(value: object) -> dict[str, object]:
    """Validate and sanitize the frozen Flex gateway health contract."""

    if not isinstance(value, dict) or set(value) != _HEALTH_FIELDS:
        raise ValueError("upstream health schema differs")
    expected = {
        "status": "ok",
        "schema": HEALTH_SCHEMA,
        "requested_model": REQUESTED_MODEL,
        "provider_model": PROVIDER_MODEL,
        "service_tier": SERVICE_TIER,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ValueError(f"upstream health {field} differs")
    return {**expected, "budget": _validate_budget(value.get("budget"))}


def _validate_gateway_success(
    response: object,
) -> tuple[dict[str, object], dict[str, object], int, list[str]]:
    """Return sanitized success evidence or fail closed."""

    if not isinstance(response, dict):
        raise ValueError("upstream success response is not an object")
    if response.get("model") != REQUESTED_MODEL:
        raise ValueError("upstream returned an unexpected alias")
    if response.get("service_tier") != SERVICE_TIER:
        raise ValueError("upstream returned an unexpected service tier")
    response_id = response.get("id")
    if not isinstance(response_id, str) or not response_id or len(response_id) > 200:
        raise ValueError("upstream response ID is missing or invalid")
    raw_meta = response.get("flex_gateway_meta")
    if not isinstance(raw_meta, dict) or set(raw_meta) != _GATEWAY_META_FIELDS:
        raise ValueError("upstream Flex gateway metadata schema differs")
    expected_meta = {
        "provider_actual_model": PROVIDER_MODEL,
        "returned_alias": REQUESTED_MODEL,
        "service_tier": SERVICE_TIER,
    }
    for field, expected_value in expected_meta.items():
        if raw_meta.get(field) != expected_value:
            raise ValueError(f"upstream Flex gateway metadata {field} differs")
    gateway_request_id = raw_meta.get("request_id")
    if (
        not isinstance(gateway_request_id, str)
        or not gateway_request_id
        or len(gateway_request_id) > 200
    ):
        raise ValueError("upstream Flex gateway request ID is missing or invalid")
    for field in ("request_sha256", "provider_request_sha256"):
        if not _is_sha256(raw_meta.get(field)):
            raise ValueError(f"upstream Flex gateway metadata {field} is invalid")
    physical_attempts = raw_meta.get("physical_attempt_count")
    physical_retries = raw_meta.get("physical_retry_count")
    if (
        not isinstance(physical_attempts, int)
        or isinstance(physical_attempts, bool)
        or physical_attempts < 1
        or not isinstance(physical_retries, int)
        or isinstance(physical_retries, bool)
        or physical_retries != physical_attempts - 1
    ):
        raise ValueError("upstream Flex gateway physical-attempt evidence differs")
    raw_proxy_meta = response.get("proxy_meta")
    if not isinstance(raw_proxy_meta, dict):
        raise ValueError("upstream compatibility proxy metadata is missing")
    unsupported = raw_proxy_meta.get("unsupported_parameters")
    if unsupported != []:
        raise ValueError("upstream reports unsupported parameters")
    if (
        raw_proxy_meta.get("attempts") != physical_attempts
        or raw_proxy_meta.get("provider_actual_model") != PROVIDER_MODEL
        or raw_proxy_meta.get("service_tier") != SERVICE_TIER
    ):
        raise ValueError("upstream compatibility proxy metadata differs")
    gateway_evidence = {
        "gateway_request_id": gateway_request_id,
        "gateway_request_sha256": raw_meta["request_sha256"],
        "provider_request_sha256": raw_meta["provider_request_sha256"],
        "provider_actual_model": PROVIDER_MODEL,
        "service_tier": SERVICE_TIER,
    }
    return response, gateway_evidence, physical_attempts, []


def make_handler(*, upstream: str, log_path: Path, run_id: str):
    class Handler(BaseHTTPRequestHandler):
        def _write(self, status: int, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/healthz":
                self._write(404, b'{"error":"not found"}')
                return
            try:
                status, payload = base.request_upstream("GET", f"{upstream}/healthz")
                parsed_health = json.loads(payload) if status == 200 else None
                upstream_health = _validate_health(parsed_health)
            except Exception as exc:  # noqa: BLE001
                self._write(
                    503,
                    json.dumps(
                        {
                            "status": "error",
                            "error": "upstream Flex health validation failed",
                            "detail": str(exc)[:300],
                        }
                    ).encode(),
                )
                return
            if status != 200:
                self._write(
                    503,
                    json.dumps(
                        {
                            "status": "error",
                            "upstream_status": status,
                            "error": "upstream Flex gateway is not healthy",
                        }
                    ).encode(),
                )
                return
            self._write(
                200,
                json.dumps(
                    {
                        "status": "ok",
                        "run_id": run_id,
                        "exclusive_log": str(log_path),
                        "upstream": upstream,
                        "upstream_health": upstream_health,
                        "base_wrapper_sha256": hashlib.sha256(
                            Path(base.__file__).read_bytes()
                        ).hexdigest(),
                        "controlled_wrapper_sha256": hashlib.sha256(
                            Path(__file__).read_bytes()
                        ).hexdigest(),
                    }
                ).encode(),
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._write(404, b'{"error":"not found"}')
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            requested_model = None
            request_hash = hashlib.sha256(body).hexdigest()
            event_id = f"exclusive-{uuid.uuid4().hex}"
            question_id = _safe_header(self, "X-Controlled-Question-ID")
            logical_call_id = _safe_header(self, "X-Controlled-Logical-Call-ID")
            request_json: object = None
            try:
                request_json = json.loads(body)
                if isinstance(request_json, dict):
                    requested_model = request_json.get("model")
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            started_at = base.utc_now()
            requested_tier = (
                request_json.get("service_tier")
                if isinstance(request_json, dict)
                else None
            )
            if requested_model != REQUESTED_MODEL or requested_tier not in (
                None,
                SERVICE_TIER,
            ):
                error = (
                    "exclusive proxy only accepts model gpt-5.5 and "
                    "service_tier omitted or flex"
                )
                response_payload = json.dumps(
                    {
                        "error": error,
                        "exclusive_proxy_meta": {
                            "event_id": event_id,
                            "run_id": run_id,
                            "question_id": question_id,
                            "logical_call_id": logical_call_id,
                            "request_sha256": request_hash,
                            "client_http_attempts": 1,
                            "upstream_http_attempts": 0,
                            "unsupported_parameters": [],
                        },
                    },
                    separators=(",", ":"),
                ).encode()
                base.append_jsonl(
                    log_path,
                    {
                        "run_id": run_id,
                        "started_at": started_at,
                        "finished_at": base.utc_now(),
                        "status": "error",
                        "http_status": 400,
                        "requested_model": requested_model,
                        "requested_service_tier": requested_tier,
                        "actual_model": None,
                        "provider_actual_model": None,
                        "service_tier": None,
                        "gateway_request_id": None,
                        "gateway_request_sha256": None,
                        "provider_request_sha256": None,
                        "response_id": None,
                        "event_id": event_id,
                        "question_id": question_id,
                        "logical_call_id": logical_call_id,
                        "request_sha256": request_hash,
                        "response_sha256": hashlib.sha256(response_payload).hexdigest(),
                        "client_http_attempts": 1,
                        "upstream_http_attempts": 0,
                        "unsupported_parameters": [],
                        "usage": None,
                        "error": error,
                    },
                )
                self._write(400, response_payload)
                return

            status = 502
            response_payload = b""
            error = None
            try:
                status, response_payload = base.request_upstream(
                    "POST", f"{upstream}/v1/chat/completions", body
                )
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                response_payload = json.dumps({"error": error}).encode()

            response_model = response_id = None
            usage = None
            upstream_http_attempts = None
            unsupported_parameters: list[str] = []
            response_json: dict | None = None
            gateway_evidence: dict[str, object] = {}
            try:
                parsed = json.loads(response_payload)
                if not isinstance(parsed, dict):
                    raise ValueError("upstream JSON response is not an object")
                response_json = parsed
                response_model = response_json.get("model")
                response_id = response_json.get("id")
                usage = response_json.get("usage")
                proxy_meta = response_json.get("proxy_meta")
                if isinstance(proxy_meta, dict):
                    upstream_http_attempts = proxy_meta.get("attempts")
                    raw_unsupported = proxy_meta.get("unsupported_parameters", [])
                    if isinstance(raw_unsupported, list) and all(
                        isinstance(value, str) for value in raw_unsupported
                    ):
                        unsupported_parameters = raw_unsupported
                if status >= 400 and error is None:
                    error = f"upstream HTTP {status}"
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                if error is None and status >= 400:
                    error = f"HTTP {status}: non-JSON response"

            if 200 <= status < 300:
                try:
                    (
                        response_json,
                        gateway_evidence,
                        upstream_http_attempts,
                        unsupported_parameters,
                    ) = _validate_gateway_success(response_json)
                    response_model = REQUESTED_MODEL
                except ValueError as exc:
                    status = 502
                    error = str(exc)
                    response_json = {"error": error}
                    usage = None
                    gateway_evidence = {}
            elif response_json is None:
                response_json = {"error": error or f"HTTP {status}"}

            response_json["exclusive_proxy_meta"] = {
                "event_id": event_id,
                "run_id": run_id,
                "question_id": question_id,
                "logical_call_id": logical_call_id,
                "request_sha256": request_hash,
                "client_http_attempts": 1,
                "upstream_http_attempts": upstream_http_attempts,
                "unsupported_parameters": unsupported_parameters,
                "gateway_request_id": gateway_evidence.get("gateway_request_id"),
                "gateway_request_sha256": gateway_evidence.get(
                    "gateway_request_sha256"
                ),
                "provider_request_sha256": gateway_evidence.get(
                    "provider_request_sha256"
                ),
                "provider_actual_model": gateway_evidence.get(
                    "provider_actual_model"
                ),
                "service_tier": gateway_evidence.get("service_tier"),
            }
            response_payload = json.dumps(
                response_json, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")

            response_hash = hashlib.sha256(response_payload).hexdigest()
            base.append_jsonl(
                log_path,
                {
                    "run_id": run_id,
                    "started_at": started_at,
                    "finished_at": base.utc_now(),
                    "status": "success" if 200 <= status < 300 else "error",
                    "http_status": status,
                    "requested_model": requested_model,
                    "requested_service_tier": requested_tier,
                    "actual_model": response_model,
                    "provider_actual_model": gateway_evidence.get(
                        "provider_actual_model"
                    ),
                    "service_tier": gateway_evidence.get("service_tier"),
                    "gateway_request_id": gateway_evidence.get(
                        "gateway_request_id"
                    ),
                    "gateway_request_sha256": gateway_evidence.get(
                        "gateway_request_sha256"
                    ),
                    "provider_request_sha256": gateway_evidence.get(
                        "provider_request_sha256"
                    ),
                    "response_id": response_id,
                    "event_id": event_id,
                    "question_id": question_id,
                    "logical_call_id": logical_call_id,
                    "request_sha256": request_hash,
                    "response_sha256": response_hash,
                    "client_http_attempts": 1,
                    "upstream_http_attempts": upstream_http_attempts,
                    "unsupported_parameters": unsupported_parameters,
                    "usage": usage,
                    "error": error,
                },
            )
            self._write(status, response_payload)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Question-linked exclusive proxy for controlled answers"
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        args.upstream = base.validate_upstream(args.upstream)
    except ValueError as exc:
        parser.error(str(exc))
    log_path = args.log.expanduser().resolve()
    ready_path = args.ready.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise SystemExit(f"exclusive log already exists: {log_path}")
    log_path.touch(mode=0o600)
    server = base.ThreadedHTTPServer(
        ("127.0.0.1", args.port),
        make_handler(
            upstream=args.upstream.rstrip("/"),
            log_path=log_path,
            run_id=args.run_id,
        ),
    )
    base.atomic_json(
        ready_path,
        {
            "run_id": args.run_id,
            "pid": os.getpid(),
            "port": server.server_port,
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "upstream": args.upstream.rstrip("/"),
            "log": str(log_path),
            "base_wrapper_sha256": hashlib.sha256(
                Path(base.__file__).read_bytes()
            ).hexdigest(),
            "controlled_wrapper_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "started_at": base.utc_now(),
        },
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
