#!/usr/bin/env python3
"""Exclusive per-launch proxy for the fixed OpenAI GPT-5.5 Flex gateway.

This process provides exact R110 run linkage while leaving provider selection,
snapshot pinning, service-tier enforcement, and the durable cost cap to
``scripts.gateways.openai_gpt55_flex_gateway``.  It accepts only a loopback upstream whose
health contract proves that fixed gateway is active.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import urlparse


REQUESTED_MODEL = "gpt-5.5"
PROVIDER_MODEL = "gpt-5.5-2026-04-23"
SERVICE_TIER = "flex"


WRITE_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def append_jsonl(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    with WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def request_upstream(method: str, url: str, body: bytes = b"") -> tuple[int, bytes]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid upstream URL: {url}")
    cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = cls(parsed.hostname, parsed.port, timeout=300)
    try:
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"
        connection.request(
            method,
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer x",
            },
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def validate_upstream(value: str) -> str:
    """Return a normalized loopback Flex-gateway origin or fail closed."""
    parsed = urlparse(value.rstrip("/"))
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port is None
        or not (0 < parsed.port < 65536)
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("upstream must be a loopback HTTP origin with an explicit port")
    return f"http://127.0.0.1:{parsed.port}"


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


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
                status, payload = request_upstream("GET", f"{upstream}/healthz")
                upstream_health = json.loads(payload) if status == 200 else None
            except Exception as exc:  # noqa: BLE001
                self._write(
                    503,
                    json.dumps({"status": "error", "error": str(exc)}).encode(),
                )
                return
            if (
                status != 200
                or not isinstance(upstream_health, dict)
                or upstream_health.get("status") != "ok"
                or upstream_health.get("schema")
                != "openai-gpt55-flex-health/v1"
                or upstream_health.get("requested_model") != REQUESTED_MODEL
                or upstream_health.get("provider_model") != PROVIDER_MODEL
                or upstream_health.get("service_tier") != SERVICE_TIER
                or not isinstance(upstream_health.get("budget"), dict)
            ):
                self._write(
                    503,
                    json.dumps(
                        {
                            "status": "error",
                            "upstream_status": status,
                            "upstream_health": upstream_health,
                        }
                    ).encode(),
                )
                return
            self._write(
                200,
                json.dumps({
                    "status": "ok",
                    "run_id": run_id,
                    "exclusive_log": str(log_path),
                    "upstream": upstream,
                    "upstream_health": upstream_health,
                }).encode(),
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._write(404, b'{"error":"not found"}')
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            requested_model = None
            requested_tier = None
            request_hash = hashlib.sha256(body).hexdigest()
            try:
                request_json = json.loads(body)
                if isinstance(request_json, dict):
                    requested_model = request_json.get("model")
                    requested_tier = request_json.get("service_tier")
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            started_at = utc_now()
            if requested_model != REQUESTED_MODEL or requested_tier not in (
                None,
                SERVICE_TIER,
            ):
                response_payload = json.dumps(
                    {
                        "error": (
                            "exclusive proxy only accepts model gpt-5.5 and "
                            "service_tier omitted or flex"
                        )
                    }
                ).encode()
                append_jsonl(
                    log_path,
                    {
                        "run_id": run_id,
                        "started_at": started_at,
                        "finished_at": utc_now(),
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
                        "request_sha256": request_hash,
                        "error": "requested model or service tier differs",
                    },
                )
                self._write(400, response_payload)
                return
            status = 502
            response_payload = b""
            error = None
            try:
                status, response_payload = request_upstream(
                    "POST", f"{upstream}/v1/chat/completions", body
                )
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                response_payload = json.dumps({"error": error}).encode()
            response_model = response_id = None
            response_tier = None
            gateway_meta: dict[str, object] = {}
            try:
                response_json = json.loads(response_payload)
                response_model = response_json.get("model")
                response_id = response_json.get("id")
                response_tier = response_json.get("service_tier")
                raw_gateway_meta = response_json.get("flex_gateway_meta")
                if isinstance(raw_gateway_meta, dict):
                    gateway_meta = raw_gateway_meta
                if status >= 400 and error is None:
                    error = str(response_json.get("error", f"HTTP {status}"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if error is None and status >= 400:
                    error = f"HTTP {status}: non-JSON response"
            if 200 <= status < 300:
                mismatch = None
                if response_model != REQUESTED_MODEL:
                    mismatch = f"upstream returned unexpected model {response_model!r}"
                elif response_tier != SERVICE_TIER:
                    mismatch = (
                        "upstream returned unexpected service tier "
                        f"{response_tier!r}"
                    )
                elif gateway_meta.get("provider_actual_model") != PROVIDER_MODEL:
                    mismatch = "upstream omitted the fixed provider snapshot evidence"
                elif gateway_meta.get("service_tier") != SERVICE_TIER:
                    mismatch = "upstream omitted the fixed Flex evidence"
                elif not isinstance(gateway_meta.get("request_id"), str):
                    mismatch = "upstream omitted the gateway request ID"
                if mismatch is not None:
                    status = 502
                    error = mismatch
                    response_payload = json.dumps({"error": error}).encode()
            append_jsonl(
                log_path,
                {
                    "run_id": run_id,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "status": "success" if 200 <= status < 300 else "error",
                    "http_status": status,
                    "requested_model": requested_model,
                    "requested_service_tier": requested_tier,
                    "actual_model": response_model,
                    "provider_actual_model": gateway_meta.get(
                        "provider_actual_model"
                    ),
                    "service_tier": response_tier,
                    "gateway_request_id": gateway_meta.get("request_id"),
                    "gateway_request_sha256": gateway_meta.get("request_sha256"),
                    "provider_request_sha256": gateway_meta.get(
                        "provider_request_sha256"
                    ),
                    "response_id": response_id,
                    "request_sha256": request_hash,
                    "error": error,
                },
            )
            self._write(status, response_payload)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Exclusive proxy for one baseline launch")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        args.upstream = validate_upstream(args.upstream)
    except ValueError as exc:
        parser.error(str(exc))
    log_path = args.log.expanduser().resolve()
    ready_path = args.ready.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise SystemExit(f"exclusive log already exists: {log_path}")
    log_path.touch(mode=0o600)
    server = ThreadedHTTPServer(
        ("127.0.0.1", args.port),
        make_handler(
            upstream=args.upstream.rstrip("/"),
            log_path=log_path,
            run_id=args.run_id,
        ),
    )
    atomic_json(
        ready_path,
        {
            "run_id": args.run_id,
            "pid": os.getpid(),
            "port": server.server_port,
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "upstream": args.upstream.rstrip("/"),
            "log": str(log_path),
            "started_at": utc_now(),
        },
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
