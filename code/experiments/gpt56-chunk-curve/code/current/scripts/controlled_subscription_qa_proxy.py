#!/usr/bin/env python3
"""Exclusive question-linked proxy for subscription-backed QA calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import gpt55_run_proxy as base


REQUESTED_MODEL = "gpt-5.5"
REASONING_EFFORT = "none"


class CompletionTokenCapError(ValueError):
    """A rejected upstream response whose provider attempts remain auditable."""

    def __init__(
        self,
        *,
        attempts: int,
        unsupported: list[str],
        ignored: list[str],
    ) -> None:
        super().__init__("upstream response exceeds the completion token cap")
        self.attempts = attempts
        self.unsupported = unsupported
        self.ignored = ignored


def _safe_header(handler: BaseHTTPRequestHandler, name: str) -> str | None:
    value = handler.headers.get(name)
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned[:200] or None


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_subscription_upstream(
    upstream: str,
    *,
    expected_code_sha256: str,
) -> dict[str, Any]:
    """Validate the live subscription proxy before accepting QA calls."""

    if not _valid_sha256(expected_code_sha256):
        raise ValueError("expected upstream code SHA-256 is invalid")
    status, payload = base.request_upstream("GET", f"{upstream}/healthz")
    try:
        health = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("upstream health is not JSON") from exc
    if status != 200 or not isinstance(health, dict):
        raise ValueError("upstream subscription health is unavailable")
    if health.get("code_sha256") != expected_code_sha256:
        raise ValueError("upstream code SHA-256 differs")
    if (
        health.get("status") != "ok"
        or health.get("auth_readable") is not True
        or health.get("requested_reasoning_effort") != REASONING_EFFORT
        or health.get("max_attempts") != 2
        or not isinstance(health.get("max_concurrency"), int)
        or isinstance(health.get("max_concurrency"), bool)
        or int(health["max_concurrency"]) < 1
    ):
        raise ValueError("upstream subscription health differs")
    return {
        "status": "ok",
        "auth_readable": True,
        "max_concurrency": health["max_concurrency"],
        "max_attempts": 2,
        "connect_timeout_s": health.get("connect_timeout_s"),
        "read_timeout_s": health.get("read_timeout_s"),
        "requested_reasoning_effort": REASONING_EFFORT,
        "code_sha256": expected_code_sha256,
        "request_log": health.get("request_log"),
    }


def _validate_upstream_success(
    payload: object,
    *,
    requested_completion_token_cap: int | None,
) -> tuple[dict[str, Any], int, list[str], list[str]]:
    if not isinstance(payload, dict):
        raise ValueError("upstream response is not an object")
    if payload.get("model") != REQUESTED_MODEL:
        raise ValueError("upstream returned an unexpected model")
    response_id = payload.get("id")
    if not isinstance(response_id, str) or not response_id or len(response_id) > 200:
        raise ValueError("upstream response ID is invalid")
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("upstream response usage is missing")
    token_counts: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError("upstream response usage is invalid")
        token_counts[name] = value
    if token_counts["total_tokens"] != (
        token_counts["prompt_tokens"] + token_counts["completion_tokens"]
    ):
        raise ValueError("upstream response usage total differs")
    proxy_meta = payload.get("proxy_meta")
    if not isinstance(proxy_meta, dict):
        raise ValueError("upstream proxy metadata is missing")
    attempts = proxy_meta.get("attempts")
    if (
        not isinstance(attempts, int)
        or isinstance(attempts, bool)
        or not 1 <= attempts <= 2
    ):
        raise ValueError("upstream attempt count is invalid")
    unsupported = proxy_meta.get("unsupported_parameters", [])
    ignored = proxy_meta.get("ignored_client_parameters", [])
    if not (
        isinstance(unsupported, list)
        and all(isinstance(value, str) for value in unsupported)
        and isinstance(ignored, list)
        and all(isinstance(value, str) for value in ignored)
    ):
        raise ValueError("upstream parameter evidence is invalid")
    if unsupported:
        raise ValueError("upstream unsupported parameter evidence differs")
    if ignored not in ([], ["max_output_tokens"]):
        raise ValueError("upstream ignored parameter evidence differs")
    if proxy_meta.get("requested_reasoning_effort") != REASONING_EFFORT:
        raise ValueError("upstream requested reasoning effort differs")
    actual_reasoning = proxy_meta.get("actual_reasoning_effort")
    if actual_reasoning not in (None, REASONING_EFFORT):
        raise ValueError("upstream actual reasoning effort differs")
    details = usage.get("completion_tokens_details")
    if actual_reasoning is None and (
        not isinstance(details, dict) or "reasoning_tokens" not in details
    ):
        raise ValueError("upstream reasoning evidence is missing")
    if isinstance(details, dict):
        reasoning_tokens = details.get("reasoning_tokens", 0)
        if (
            not isinstance(reasoning_tokens, int)
            or isinstance(reasoning_tokens, bool)
            or reasoning_tokens != 0
        ):
            raise ValueError("upstream reported invalid reasoning tokens")
    if (
        requested_completion_token_cap is not None
        and token_counts["completion_tokens"] > requested_completion_token_cap
    ):
        raise CompletionTokenCapError(
            attempts=attempts,
            unsupported=unsupported,
            ignored=ignored,
        )
    return payload, attempts, unsupported, ignored


def make_handler(
    *,
    upstream: str,
    log_path: Path,
    run_id: str,
    expected_upstream_sha256: str,
):
    if not _valid_sha256(expected_upstream_sha256):
        raise ValueError("expected upstream SHA-256 is invalid")

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
                validate_subscription_upstream(
                    upstream,
                    expected_code_sha256=expected_upstream_sha256,
                )
            except Exception as exc:  # noqa: BLE001
                self._write(
                    503,
                    json.dumps(
                        {"status": "error", "error": str(exc)[:300]},
                        separators=(",", ":"),
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
                        "upstream_code_sha256": expected_upstream_sha256,
                    },
                    separators=(",", ":"),
                ).encode(),
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._write(404, b'{"error":"not found"}')
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            event_id = f"subscription-qa-{uuid.uuid4().hex}"
            request_sha256 = hashlib.sha256(body).hexdigest()
            question_id = _safe_header(self, "X-Controlled-Question-ID")
            logical_call_id = _safe_header(self, "X-Controlled-Logical-Call-ID")
            requested_model: object = None
            requested_completion_token_cap: int | None = None
            try:
                request_json = json.loads(body)
                if not isinstance(request_json, dict):
                    raise ValueError("request is not an object")
                requested_model = request_json.get("model")
                raw_cap = request_json.get(
                    "max_tokens", request_json.get("max_completion_tokens")
                )
                if raw_cap is not None:
                    if (
                        not isinstance(raw_cap, int)
                        or isinstance(raw_cap, bool)
                        or raw_cap < 1
                    ):
                        raise ValueError("completion token cap is invalid")
                    requested_completion_token_cap = raw_cap
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                request_json = None
            started_at = base.utc_now()
            status = 400
            error: str | None = None
            response: dict[str, Any]
            child_upstream_attempts = 0
            upstream_attempts: int | None = 0
            unsupported: list[str] = []
            ignored: list[str] = []
            if request_json is None:
                error = "request JSON is invalid"
                response = {"error": error}
            elif requested_model != REQUESTED_MODEL:
                error = "exclusive proxy only accepts model gpt-5.5"
                response = {"error": error}
            elif question_id is None or logical_call_id is None:
                error = "controlled question and logical call headers are required"
                response = {"error": error}
            else:
                try:
                    child_upstream_attempts = 1
                    upstream_attempts = None
                    status, raw_response = base.request_upstream(
                        "POST", f"{upstream}/v1/chat/completions", body
                    )
                    parsed = json.loads(raw_response)
                    if 200 <= status < 300:
                        response, upstream_attempts, unsupported, ignored = (
                            _validate_upstream_success(
                                parsed,
                                requested_completion_token_cap=(
                                    requested_completion_token_cap
                                ),
                            )
                        )
                    else:
                        response = parsed if isinstance(parsed, dict) else {
                            "error": f"upstream HTTP {status}"
                        }
                        error = f"upstream HTTP {status}"
                except CompletionTokenCapError as exc:
                    status = 502
                    error = f"{type(exc).__name__}: {exc}"
                    response = {"error": error}
                    upstream_attempts = exc.attempts
                    unsupported = exc.unsupported
                    ignored = exc.ignored
                except Exception as exc:  # noqa: BLE001
                    status = 502
                    error = f"{type(exc).__name__}: {exc}"
                    response = {"error": error}
            meta = {
                "event_id": event_id,
                "run_id": run_id,
                "question_id": question_id,
                "logical_call_id": logical_call_id,
                "request_sha256": request_sha256,
                "client_http_attempts": 1,
                "requested_completion_token_cap": requested_completion_token_cap,
                "child_upstream_http_attempts": child_upstream_attempts,
                "upstream_http_attempts": upstream_attempts,
                "unsupported_parameters": unsupported,
                "ignored_client_parameters": ignored,
            }
            response["exclusive_proxy_meta"] = meta
            response_payload = json.dumps(
                response, ensure_ascii=False, separators=(",", ":")
            ).encode()
            response_id = response.get("id")
            actual_model = response.get("model")
            usage = response.get("usage")
            base.append_jsonl(
                log_path,
                {
                    "run_id": run_id,
                    "started_at": started_at,
                    "finished_at": base.utc_now(),
                    "status": "success" if 200 <= status < 300 else "error",
                    "http_status": status,
                    "requested_model": requested_model,
                    "actual_model": actual_model,
                    "response_id": response_id,
                    "event_id": event_id,
                    "question_id": question_id,
                    "logical_call_id": logical_call_id,
                    "request_sha256": request_sha256,
                    "response_sha256": hashlib.sha256(response_payload).hexdigest(),
                    "client_http_attempts": 1,
                    "requested_completion_token_cap": (
                        requested_completion_token_cap
                    ),
                    "child_upstream_http_attempts": child_upstream_attempts,
                    "upstream_http_attempts": upstream_attempts,
                    "unsupported_parameters": unsupported,
                    "ignored_client_parameters": ignored,
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
        description="Exclusive subscription proxy for controlled QA"
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--upstream-code-sha256", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        upstream = base.validate_upstream(args.upstream)
    except ValueError as exc:
        parser.error(str(exc))
    validate_subscription_upstream(
        upstream,
        expected_code_sha256=args.upstream_code_sha256,
    )
    log_path = args.log.expanduser().resolve()
    ready_path = args.ready.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise SystemExit(f"exclusive log already exists: {log_path}")
    log_path.touch(mode=0o600)
    server = base.ThreadedHTTPServer(
        ("127.0.0.1", args.port),
        make_handler(
            upstream=upstream,
            log_path=log_path,
            run_id=args.run_id,
            expected_upstream_sha256=args.upstream_code_sha256,
        ),
    )
    base.atomic_json(
        ready_path,
        {
            "run_id": args.run_id,
            "pid": os.getpid(),
            "port": server.server_port,
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "upstream": upstream,
            "upstream_code_sha256": args.upstream_code_sha256,
            "log": str(log_path),
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
