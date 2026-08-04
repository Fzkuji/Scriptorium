#!/usr/bin/env python3
"""Exclusive, model-bound HTTP proxy used only by the R301 runner.

The proxy accepts one preregistered requested model and one regular expression
for the provider's returned model identity.  It writes one fsynced JSONL event
per client request and exposes the logical-call identifier supplied by the
runner.  The upstream API key is read from an environment variable and is
never written to an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import urlparse


WRITE_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
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
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"
    with WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def request_upstream(
    *, method: str, url: str, body: bytes, api_key: str
) -> tuple[int, bytes]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid upstream URL: {url}")
    connection_class = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_class(parsed.hostname, parsed.port, timeout=360)
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
                "Authorization": f"Bearer {api_key}",
            },
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _safe_header(handler: BaseHTTPRequestHandler, name: str) -> str | None:
    value = handler.headers.get(name)
    return value[:400] if value else None


def make_handler(
    *,
    upstream: str,
    api_key: str,
    log_path: Path,
    run_id: str,
    expected_requested_model: str,
    accepted_actual_model: re.Pattern[str],
):
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
            self._write(
                200,
                json.dumps(
                    {
                        "status": "ok",
                        "run_id": run_id,
                        "expected_requested_model": expected_requested_model,
                        "accepted_actual_model_regex": accepted_actual_model.pattern,
                    },
                    separators=(",", ":"),
                ).encode("utf-8"),
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._write(404, b'{"error":"not found"}')
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            request_sha256 = hashlib.sha256(body).hexdigest()
            event_id = f"r301-exclusive-{uuid.uuid4().hex}"
            logical_call_id = _safe_header(
                self, "X-Controlled-Logical-Call-ID"
            )
            question_id = _safe_header(self, "X-Controlled-Question-ID")
            started_at = utc_now()
            requested_model: object = None
            try:
                request_value = json.loads(body)
                if isinstance(request_value, dict):
                    requested_model = request_value.get("model")
            except (UnicodeDecodeError, json.JSONDecodeError):
                request_value = None

            if requested_model != expected_requested_model:
                error = (
                    "requested model differs from the exclusive launch: "
                    f"{requested_model!r}"
                )
                response = {
                    "error": error,
                    "exclusive_proxy_meta": {
                        "event_id": event_id,
                        "run_id": run_id,
                        "logical_call_id": logical_call_id,
                        "question_id": question_id,
                        "request_sha256": request_sha256,
                        "client_http_attempts": 1,
                        "upstream_http_attempts": 0,
                        "unsupported_parameters": [],
                    },
                }
                payload = json.dumps(response, separators=(",", ":")).encode()
                append_jsonl(
                    log_path,
                    {
                        "event_id": event_id,
                        "run_id": run_id,
                        "started_at": started_at,
                        "finished_at": utc_now(),
                        "status": "error",
                        "http_status": 400,
                        "logical_call_id": logical_call_id,
                        "question_id": question_id,
                        "requested_model": requested_model,
                        "actual_model": None,
                        "response_id": None,
                        "request_sha256": request_sha256,
                        "response_sha256": hashlib.sha256(payload).hexdigest(),
                        "client_http_attempts": 1,
                        "upstream_http_attempts": 0,
                        "unsupported_parameters": [],
                        "usage": None,
                        "error": error,
                    },
                )
                self._write(400, payload)
                return

            status = 502
            raw_response = b""
            error: str | None = None
            try:
                status, raw_response = request_upstream(
                    method="POST",
                    url=f"{upstream}/v1/chat/completions",
                    body=body,
                    api_key=api_key,
                )
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"

            response_value: dict[str, object]
            try:
                parsed = json.loads(raw_response)
                if not isinstance(parsed, dict):
                    raise ValueError("upstream response is not an object")
                response_value = parsed
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                response_value = {"error": error or "upstream response is not JSON"}
                if status < 400:
                    status = 502
            actual_model = response_value.get("model")
            response_id = response_value.get("id")
            usage = response_value.get("usage")
            upstream_meta = response_value.get("proxy_meta")
            attempts = 1
            unsupported: list[str] = []
            if isinstance(upstream_meta, dict):
                raw_attempts = upstream_meta.get("attempts")
                if isinstance(raw_attempts, int) and raw_attempts >= 1:
                    attempts = raw_attempts
                raw_unsupported = upstream_meta.get("unsupported_parameters", [])
                if isinstance(raw_unsupported, list) and all(
                    isinstance(item, str) for item in raw_unsupported
                ):
                    unsupported = raw_unsupported
            if status < 300 and (
                not isinstance(actual_model, str)
                or accepted_actual_model.fullmatch(actual_model) is None
            ):
                error = f"provider returned unaccepted model {actual_model!r}"
                status = 502
                response_value = {"error": error}
            elif status >= 300 and error is None:
                error = str(response_value.get("error", f"HTTP {status}"))

            response_value["exclusive_proxy_meta"] = {
                "event_id": event_id,
                "run_id": run_id,
                "logical_call_id": logical_call_id,
                "question_id": question_id,
                "request_sha256": request_sha256,
                "client_http_attempts": 1,
                "upstream_http_attempts": attempts,
                "unsupported_parameters": unsupported,
            }
            payload = json.dumps(
                response_value, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            append_jsonl(
                log_path,
                {
                    "event_id": event_id,
                    "run_id": run_id,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "status": "success" if 200 <= status < 300 else "error",
                    "http_status": status,
                    "logical_call_id": logical_call_id,
                    "question_id": question_id,
                    "requested_model": requested_model,
                    "actual_model": actual_model,
                    "accepted_actual_model_regex": accepted_actual_model.pattern,
                    "response_id": response_id,
                    "request_sha256": request_sha256,
                    "response_sha256": hashlib.sha256(payload).hexdigest(),
                    "client_http_attempts": 1,
                    "upstream_http_attempts": attempts,
                    "unsupported_parameters": unsupported,
                    "usage": usage,
                    "error": error,
                },
            )
            self._write(status, payload)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--expected-requested-model", required=True)
    parser.add_argument("--accepted-actual-model-regex", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    try:
        accepted = re.compile(args.accepted_actual_model_regex)
    except re.error as exc:
        raise SystemExit(f"invalid actual-model regex: {exc}") from exc
    log_path = args.log.expanduser().resolve()
    ready_path = args.ready.expanduser().resolve()
    if log_path.exists() or ready_path.exists():
        raise SystemExit("proxy log/ready artifact already exists")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(mode=0o600)
    server = ThreadedHTTPServer(
        ("127.0.0.1", args.port),
        make_handler(
            upstream=args.upstream.rstrip("/").removesuffix("/v1"),
            api_key=api_key,
            log_path=log_path,
            run_id=args.run_id,
            expected_requested_model=args.expected_requested_model,
            accepted_actual_model=accepted,
        ),
    )
    atomic_json(
        ready_path,
        {
            "schema": "nativemem.r301-exclusive-proxy-ready.v1",
            "run_id": args.run_id,
            "pid": os.getpid(),
            "port": server.server_port,
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "upstream": args.upstream.rstrip("/"),
            "log": str(log_path),
            "api_key_env": args.api_key_env,
            "api_key_recorded": False,
            "expected_requested_model": args.expected_requested_model,
            "accepted_actual_model_regex": accepted.pattern,
            "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
