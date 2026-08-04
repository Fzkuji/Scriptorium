"""Fail-closed local gateway for OpenAI GPT-5.5 Flex Chat Completions.

The public HTTP surface accepts the existing ``gpt-5.5`` alias, while every
provider request is pinned to ``gpt-5.5-2026-04-23`` and ``service_tier=flex``.
The module deliberately has no configurable provider URL.  Tests inject a
transport object directly and therefore never need provider network access.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import http.client
import json
import os
import stat
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Callable, Iterator, Mapping, Protocol


REQUESTED_MODEL = "gpt-5.5"
PROVIDER_MODEL = "gpt-5.5-2026-04-23"
SERVICE_TIER = "flex"
UPSTREAM_HOST = "api.openai.com"
UPSTREAM_PATH = "/v1/chat/completions"

CONTEXT_WINDOW_TOKENS = 1_050_000
MAX_OUTPUT_TOKENS = 128_000
LONG_CONTEXT_THRESHOLD = 272_000

NANODOLLARS_PER_USD = 1_000_000_000
UNCACHED_INPUT_NANOS_PER_TOKEN = 2_500
CACHED_INPUT_NANOS_PER_TOKEN = 250
OUTPUT_NANOS_PER_TOKEN = 15_000
LONG_UNCACHED_INPUT_NANOS_PER_TOKEN = 5_000
LONG_CACHED_INPUT_NANOS_PER_TOKEN = 500
LONG_OUTPUT_NANOS_PER_TOKEN = 22_500

ROOT_SCHEMA = "openai-gpt55-flex-result-root/v1"
STATE_SCHEMA = "openai-gpt55-flex-cost-state/v1"
REQUEST_LOG_SCHEMA = "openai-gpt55-flex-request/v1"
PRICING_ID = "openai-gpt55-flex-2026-04-23"
ROOT_MARKER_NAME = "openai_gpt55_flex_root.json"
STATE_NAME = "flex_cost_state.json"
REQUEST_LOG_NAME = "flex_requests.jsonl"


class GatewayError(RuntimeError):
    """Base class for local gateway failures."""


class RequestRejected(GatewayError):
    """A local request violates the gateway contract."""

    def __init__(self, message: str, *, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.code = code


class BudgetExceeded(GatewayError):
    """A request cannot be reserved under the durable cost cap."""


class StateError(GatewayError):
    """Durable state or append-only audit data is invalid."""


@dataclass(frozen=True)
class UpstreamResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class UpstreamTransport(Protocol):
    def send(self, payload: bytes) -> UpstreamResponse:
        """Send one physical request and return the complete HTTP response."""


class OpenAIChatCompletionsTransport:
    """Production transport with a compile-time fixed OpenAI endpoint."""

    def __init__(self, *, api_key: str, timeout_seconds: float = 900.0) -> None:
        if not api_key or not api_key.strip():
            raise GatewayError("OpenAI API key is empty")
        if timeout_seconds <= 0:
            raise GatewayError("timeout_seconds must be positive")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def send(self, payload: bytes) -> UpstreamResponse:
        connection = http.client.HTTPSConnection(
            UPSTREAM_HOST,
            443,
            timeout=self._timeout_seconds,
        )
        try:
            connection.request(
                "POST",
                UPSTREAM_PATH,
                body=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "model-aligned-wiki-gpt55-flex-gateway/1",
                },
            )
            response = connection.getresponse()
            body = response.read()
            headers = {key.lower(): value for key, value in response.getheaders()}
            return UpstreamResponse(response.status, headers, body)
        finally:
            connection.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def usd_to_nanos(value: str | Decimal | int | float) -> int:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise GatewayError(f"invalid USD value: {value!r}") from exc
    if not amount.is_finite() or amount <= 0:
        raise GatewayError("USD value must be finite and positive")
    nanos = amount * NANODOLLARS_PER_USD
    if nanos != nanos.to_integral_value():
        raise GatewayError("USD value supports at most 9 decimal places")
    return int(nanos)


def nanos_to_usd(value: int) -> str:
    if not isinstance(value, int) or value < 0:
        raise ValueError("nanodollar value must be a non-negative integer")
    result = Decimal(value) / NANODOLLARS_PER_USD
    rendered = format(result, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _token_count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StateError(f"{name} must be a non-negative integer")
    return value


def calculate_cost_nanos(
    *,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
) -> int:
    """Calculate exact Flex token cost using integer nanodollars."""

    prompt = _token_count(prompt_tokens, name="prompt_tokens")
    cached = _token_count(cached_tokens, name="cached_tokens")
    completion = _token_count(completion_tokens, name="completion_tokens")
    if cached > prompt:
        raise StateError("cached_tokens exceeds prompt_tokens")
    if prompt > LONG_CONTEXT_THRESHOLD:
        uncached_rate = LONG_UNCACHED_INPUT_NANOS_PER_TOKEN
        cached_rate = LONG_CACHED_INPUT_NANOS_PER_TOKEN
        output_rate = LONG_OUTPUT_NANOS_PER_TOKEN
    else:
        uncached_rate = UNCACHED_INPUT_NANOS_PER_TOKEN
        cached_rate = CACHED_INPUT_NANOS_PER_TOKEN
        output_rate = OUTPUT_NANOS_PER_TOKEN
    return (
        (prompt - cached) * uncached_rate
        + cached * cached_rate
        + completion * output_rate
    )


def maximum_request_cost_nanos(max_completion_tokens: int) -> int:
    """Return a published-model-limit upper bound for one successful request.

    The reservation assumes every possible input token is uncached and uses
    long-context rates for the complete request.  This is intentionally more
    conservative than estimating the prompt locally.
    """

    maximum = _positive_int(
        max_completion_tokens,
        name="max_completion_tokens",
        upper=MAX_OUTPUT_TOKENS,
    )
    maximum_prompt = CONTEXT_WINDOW_TOKENS - maximum
    if maximum_prompt <= LONG_CONTEXT_THRESHOLD:
        return calculate_cost_nanos(
            prompt_tokens=maximum_prompt,
            cached_tokens=0,
            completion_tokens=maximum,
        )
    return (
        maximum_prompt * LONG_UNCACHED_INPUT_NANOS_PER_TOKEN
        + maximum * LONG_OUTPUT_NANOS_PER_TOKEN
    )


def _positive_int(value: object, *, name: str, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestRejected(f"{name} must be an integer")
    if value < 1 or value > upper:
        raise RequestRejected(f"{name} must be between 1 and {upper}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(raw_temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _read_regular_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StateError(f"state is not a regular file: {path}")
    metadata = path.stat(follow_symlinks=False)
    if metadata.st_nlink != 1:
        raise StateError(f"state file must not be hardlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StateError(f"cannot read state JSON: {path}") from exc
    if not isinstance(value, dict):
        raise StateError(f"state JSON is not an object: {path}")
    return value


class DurableCostStore:
    """Process-safe reservation and cumulative-cost state."""

    def __init__(self, *, state_path: Path, max_cost_nanos: int) -> None:
        self.state_path = state_path
        self.lock_path = state_path.with_suffix(state_path.suffix + ".lock")
        self.max_cost_nanos = max_cost_nanos
        self._thread_lock = threading.RLock()
        with self._locked():
            if not self.state_path.exists():
                _atomic_json(self.state_path, self._initial_state())
            self._validate(self._load())

    def _initial_state(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "pricing_id": PRICING_ID,
            "provider_model": PROVIDER_MODEL,
            "returned_alias": REQUESTED_MODEL,
            "service_tier": SERVICE_TIER,
            "context_window_tokens": CONTEXT_WINDOW_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "long_context_threshold": LONG_CONTEXT_THRESHOLD,
            "pricing_nanos_per_token": {
                "uncached_input": UNCACHED_INPUT_NANOS_PER_TOKEN,
                "cached_input": CACHED_INPUT_NANOS_PER_TOKEN,
                "output": OUTPUT_NANOS_PER_TOKEN,
                "long_uncached_input": LONG_UNCACHED_INPUT_NANOS_PER_TOKEN,
                "long_cached_input": LONG_CACHED_INPUT_NANOS_PER_TOKEN,
                "long_output": LONG_OUTPUT_NANOS_PER_TOKEN,
            },
            "max_cost_nanos": self.max_cost_nanos,
            "max_cost_usd": nanos_to_usd(self.max_cost_nanos),
            "committed_cost_nanos": 0,
            "reserved_cost_nanos": 0,
            "billable_request_count": 0,
            "failed_request_count": 0,
            "usage": {
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
            },
            "reservations": {},
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.lock_path, flags, 0o600)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise StateError("cost-state lock must be one regular file")
            with self._thread_lock:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _load(self) -> dict[str, Any]:
        return _read_regular_json(self.state_path)

    def _validate(self, state: Mapping[str, Any]) -> None:
        exact = {
            "schema": STATE_SCHEMA,
            "pricing_id": PRICING_ID,
            "provider_model": PROVIDER_MODEL,
            "returned_alias": REQUESTED_MODEL,
            "service_tier": SERVICE_TIER,
            "context_window_tokens": CONTEXT_WINDOW_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "long_context_threshold": LONG_CONTEXT_THRESHOLD,
            "pricing_nanos_per_token": {
                "uncached_input": UNCACHED_INPUT_NANOS_PER_TOKEN,
                "cached_input": CACHED_INPUT_NANOS_PER_TOKEN,
                "output": OUTPUT_NANOS_PER_TOKEN,
                "long_uncached_input": LONG_UNCACHED_INPUT_NANOS_PER_TOKEN,
                "long_cached_input": LONG_CACHED_INPUT_NANOS_PER_TOKEN,
                "long_output": LONG_OUTPUT_NANOS_PER_TOKEN,
            },
            "max_cost_nanos": self.max_cost_nanos,
            "max_cost_usd": nanos_to_usd(self.max_cost_nanos),
        }
        for name, expected in exact.items():
            if state.get(name) != expected:
                raise StateError(f"cost-state {name} mismatch")
        committed = _token_count(
            state.get("committed_cost_nanos"),
            name="committed_cost_nanos",
        )
        reserved = _token_count(
            state.get("reserved_cost_nanos"),
            name="reserved_cost_nanos",
        )
        if committed + reserved > self.max_cost_nanos:
            raise StateError("cost-state exceeds max_cost_nanos")
        reservations = state.get("reservations")
        if not isinstance(reservations, dict):
            raise StateError("cost-state reservations is not an object")
        reservation_sum = 0
        for request_id, reservation in reservations.items():
            if not isinstance(request_id, str) or not isinstance(reservation, dict):
                raise StateError("invalid cost-state reservation")
            reservation_sum += _token_count(
                reservation.get("cost_nanos"),
                name="reservation.cost_nanos",
            )
        if reservation_sum != reserved:
            raise StateError("cost-state reservation sum mismatch")
        for name in ("billable_request_count", "failed_request_count"):
            _token_count(state.get(name), name=name)
        usage = state.get("usage")
        if not isinstance(usage, dict):
            raise StateError("cost-state usage is not an object")
        for name in (
            "prompt_tokens",
            "cached_tokens",
            "completion_tokens",
            "reasoning_tokens",
        ):
            _token_count(usage.get(name), name=f"usage.{name}")

    def reserve(
        self,
        *,
        request_id: str,
        cost_nanos: int,
        max_completion_tokens: int,
        request_sha256: str,
    ) -> dict[str, Any]:
        with self._locked():
            state = self._load()
            self._validate(state)
            if request_id in state["reservations"]:
                raise StateError("duplicate request reservation")
            projected = (
                state["committed_cost_nanos"]
                + state["reserved_cost_nanos"]
                + cost_nanos
            )
            if projected > self.max_cost_nanos:
                raise BudgetExceeded(
                    "max-cost gate rejected request: "
                    f"committed={nanos_to_usd(state['committed_cost_nanos'])}, "
                    f"reserved={nanos_to_usd(state['reserved_cost_nanos'])}, "
                    f"requested_reservation={nanos_to_usd(cost_nanos)}, "
                    f"cap={nanos_to_usd(self.max_cost_nanos)}"
                )
            state["reservations"][request_id] = {
                "cost_nanos": cost_nanos,
                "max_completion_tokens": max_completion_tokens,
                "request_sha256": request_sha256,
                "created_at": utc_now(),
            }
            state["reserved_cost_nanos"] += cost_nanos
            state["updated_at"] = utc_now()
            _atomic_json(self.state_path, state)
            return copy.deepcopy(state)

    def commit(
        self,
        *,
        request_id: str,
        actual_cost_nanos: int,
        usage: Mapping[str, int],
    ) -> dict[str, Any]:
        with self._locked():
            state = self._load()
            self._validate(state)
            reservation = state["reservations"].get(request_id)
            if not isinstance(reservation, dict):
                raise StateError("request reservation is absent at commit")
            reserved = reservation["cost_nanos"]
            if actual_cost_nanos > reserved:
                raise StateError("actual cost exceeds conservative reservation")
            if (
                state["committed_cost_nanos"]
                + state["reserved_cost_nanos"]
                - reserved
                + actual_cost_nanos
                > self.max_cost_nanos
            ):
                raise StateError("actual cost would exceed max-cost gate")
            state["reservations"].pop(request_id)
            state["reserved_cost_nanos"] -= reserved
            state["committed_cost_nanos"] += actual_cost_nanos
            state["billable_request_count"] += 1
            for name in (
                "prompt_tokens",
                "cached_tokens",
                "completion_tokens",
                "reasoning_tokens",
            ):
                state["usage"][name] += _token_count(
                    usage[name],
                    name=f"usage.{name}",
                )
            state["updated_at"] = utc_now()
            _atomic_json(self.state_path, state)
            return copy.deepcopy(state)

    def release(self, *, request_id: str, failed: bool = True) -> dict[str, Any]:
        with self._locked():
            state = self._load()
            self._validate(state)
            reservation = state["reservations"].pop(request_id, None)
            if isinstance(reservation, dict):
                state["reserved_cost_nanos"] -= reservation["cost_nanos"]
            if failed:
                state["failed_request_count"] += 1
            state["updated_at"] = utc_now()
            _atomic_json(self.state_path, state)
            return copy.deepcopy(state)

    def snapshot(self) -> dict[str, Any]:
        with self._locked():
            state = self._load()
            self._validate(state)
            return copy.deepcopy(state)


class AppendOnlyRequestLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self._thread_lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._validate_file()
        else:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.path, flags, 0o600)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.lock_path, flags, 0o600)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise StateError("request-log lock must be one regular file")
            with self._thread_lock:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _validate_file(self) -> None:
        if self.path.is_symlink() or not self.path.is_file():
            raise StateError("request log must be a regular file")
        metadata = self.path.stat(follow_symlinks=False)
        if metadata.st_nlink != 1:
            raise StateError("request log must not be hardlinked")
        payload = self.path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            raise StateError("request log has an incomplete final line")

    def append(self, record: Mapping[str, Any]) -> None:
        payload = canonical_json_bytes(record) + b"\n"
        with self._locked():
            if self.path.exists():
                self._validate_file()
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.path, flags, 0o600)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise StateError("request log must be one regular file")
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise StateError("short write to request log")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


@dataclass(frozen=True)
class NormalizedUsage:
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }


def normalize_usage(
    value: object,
    *,
    max_completion_tokens: int,
) -> NormalizedUsage:
    if not isinstance(value, dict):
        raise StateError("provider usage is absent or not an object")

    def integer(mapping: Mapping[str, Any], name: str) -> int:
        raw = mapping.get(name)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise StateError(f"provider usage {name} is invalid")
        return raw

    prompt = integer(value, "prompt_tokens")
    completion = integer(value, "completion_tokens")
    total = integer(value, "total_tokens")
    prompt_details = value.get("prompt_tokens_details") or {}
    completion_details = value.get("completion_tokens_details") or {}
    if not isinstance(prompt_details, dict) or not isinstance(
        completion_details, dict
    ):
        raise StateError("provider usage details are invalid")
    cached = prompt_details.get("cached_tokens", 0)
    reasoning = completion_details.get("reasoning_tokens", 0)
    if isinstance(cached, bool) or not isinstance(cached, int) or cached < 0:
        raise StateError("provider cached token usage is invalid")
    if isinstance(reasoning, bool) or not isinstance(reasoning, int) or reasoning < 0:
        raise StateError("provider reasoning token usage is invalid")
    if cached > prompt:
        raise StateError("provider cached tokens exceed prompt tokens")
    if reasoning > completion:
        raise StateError("provider reasoning tokens exceed completion tokens")
    if total != prompt + completion:
        raise StateError("provider total_tokens differs from prompt+completion")
    if completion > max_completion_tokens:
        raise StateError("provider completion usage exceeds requested maximum")
    if total > CONTEXT_WINDOW_TOKENS:
        raise StateError("provider usage exceeds GPT-5.5 context window")
    return NormalizedUsage(prompt, cached, completion, reasoning, total)


def normalize_request(
    request: Mapping[str, Any],
    *,
    default_max_completion_tokens: int,
) -> tuple[dict[str, Any], list[str], int]:
    if not isinstance(request, Mapping):
        raise RequestRejected("request JSON must be an object")
    requested_model = request.get("model")
    if requested_model != REQUESTED_MODEL:
        raise RequestRejected(
            f"local gateway only accepts model {REQUESTED_MODEL}",
            code="model_mismatch",
        )
    requested_tier = request.get("service_tier")
    if requested_tier not in (None, SERVICE_TIER):
        raise RequestRejected(
            "service_tier must be omitted or flex",
            code="service_tier_mismatch",
        )
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestRejected("messages must be a non-empty list")
    if request.get("stream") not in (None, False):
        raise RequestRejected(
            "streaming is disabled because complete usage is required for billing",
            code="streaming_not_supported",
        )

    outbound = copy.deepcopy(dict(request))
    transformations: list[str] = []
    legacy_max = outbound.pop("max_tokens", None)
    current_max = outbound.get("max_completion_tokens")
    if legacy_max is not None:
        legacy_max = _positive_int(
            legacy_max,
            name="max_tokens",
            upper=MAX_OUTPUT_TOKENS,
        )
        if current_max is not None and current_max != legacy_max:
            raise RequestRejected(
                "max_tokens and max_completion_tokens differ",
                code="max_token_conflict",
            )
        outbound["max_completion_tokens"] = legacy_max
        current_max = legacy_max
        transformations.append("max_tokens_to_max_completion_tokens")
    if current_max is None:
        current_max = _positive_int(
            default_max_completion_tokens,
            name="default_max_completion_tokens",
            upper=MAX_OUTPUT_TOKENS,
        )
        outbound["max_completion_tokens"] = current_max
        transformations.append("default_max_completion_tokens_added")
    else:
        current_max = _positive_int(
            current_max,
            name="max_completion_tokens",
            upper=MAX_OUTPUT_TOKENS,
        )

    temperature = outbound.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise RequestRejected("temperature must be numeric")
        if not 0 <= float(temperature) <= 2:
            raise RequestRejected("temperature must be between 0 and 2")

    outbound["model"] = PROVIDER_MODEL
    outbound["service_tier"] = SERVICE_TIER
    outbound["stream"] = False
    return outbound, transformations, current_max


def _safe_error_fields(body: bytes) -> tuple[str | None, str | None, str | None]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, None, None
    if not isinstance(value, dict):
        return None, None, None
    error = value.get("error")
    if not isinstance(error, dict):
        return None, None, None
    code = error.get("code")
    kind = error.get("type")
    message = error.get("message")
    safe_code = str(code)[:100] if code is not None else None
    safe_kind = str(kind)[:100] if kind is not None else None
    safe_message = str(message)[:300] if message is not None else None
    return safe_code, safe_kind, safe_message


def _is_resource_unavailable(response: UpstreamResponse) -> bool:
    if response.status != 429:
        return False
    code, kind, message = _safe_error_fields(response.body)
    identifiers = {str(code).lower(), str(kind).lower()}
    if identifiers & {
        "resource_unavailable",
        "flex_resource_unavailable",
        "flex_capacity_exceeded",
    }:
        return True
    return message is not None and "resource unavailable" in message.lower()


def _provider_request_id(headers: Mapping[str, str]) -> str | None:
    value = headers.get("x-request-id") or headers.get("request-id")
    return str(value)[:200] if value else None


class GPT55FlexGateway:
    """Core request processor; safe to use from multiple HTTP threads."""

    def __init__(
        self,
        *,
        result_root: Path,
        max_cost_usd: str | Decimal | int | float,
        transport: UpstreamTransport,
        default_max_completion_tokens: int = 4_096,
        max_physical_attempts: int = 5,
        initial_backoff_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.result_root = result_root.expanduser().resolve()
        self.max_cost_nanos = usd_to_nanos(max_cost_usd)
        self.transport = transport
        self.default_max_completion_tokens = _positive_int(
            default_max_completion_tokens,
            name="default_max_completion_tokens",
            upper=MAX_OUTPUT_TOKENS,
        )
        if max_physical_attempts < 1 or max_physical_attempts > 20:
            raise GatewayError("max_physical_attempts must be between 1 and 20")
        if initial_backoff_seconds < 0:
            raise GatewayError("initial_backoff_seconds must be non-negative")
        self.max_physical_attempts = max_physical_attempts
        self.initial_backoff_seconds = initial_backoff_seconds
        self.sleep = sleep
        self._prepare_result_root()
        self.state_path = self.result_root / STATE_NAME
        self.request_log_path = self.result_root / REQUEST_LOG_NAME
        self.cost_store = DurableCostStore(
            state_path=self.state_path,
            max_cost_nanos=self.max_cost_nanos,
        )
        self.request_log = AppendOnlyRequestLog(self.request_log_path)

    def _prepare_result_root(self) -> None:
        if self.result_root.exists() and not self.result_root.is_dir():
            raise StateError("result root is not a directory")
        self.result_root.mkdir(parents=True, exist_ok=True)
        marker = self.result_root / ROOT_MARKER_NAME
        expected = {
            "schema": ROOT_SCHEMA,
            "provider": "openai_api",
            "api": "chat_completions",
            "upstream_origin": f"https://{UPSTREAM_HOST}",
            "upstream_path": UPSTREAM_PATH,
            "provider_model": PROVIDER_MODEL,
            "returned_alias": REQUESTED_MODEL,
            "service_tier": SERVICE_TIER,
            "billing": "api_flex",
        }
        if marker.exists():
            actual = _read_regular_json(marker)
            for name, value in expected.items():
                if actual.get(name) != value:
                    raise StateError(f"result-root marker {name} mismatch")
            return
        existing = [path for path in self.result_root.iterdir()]
        if existing:
            raise StateError(
                "result root is non-empty without the GPT-5.5 Flex marker; "
                "subscription and API artifacts must use separate roots"
            )
        _atomic_json(marker, {**expected, "created_at": utc_now()})

    def health(self) -> dict[str, Any]:
        state = self.cost_store.snapshot()
        remaining = (
            state["max_cost_nanos"]
            - state["committed_cost_nanos"]
            - state["reserved_cost_nanos"]
        )
        return {
            "status": "ok",
            "schema": "openai-gpt55-flex-health/v1",
            "requested_model": REQUESTED_MODEL,
            "provider_model": PROVIDER_MODEL,
            "service_tier": SERVICE_TIER,
            "budget": {
                "max_cost_usd": state["max_cost_usd"],
                "committed_cost_usd": nanos_to_usd(
                    state["committed_cost_nanos"]
                ),
                "reserved_cost_usd": nanos_to_usd(state["reserved_cost_nanos"]),
                "remaining_cost_usd": nanos_to_usd(remaining),
                "in_flight": len(state["reservations"]),
            },
        }

    @staticmethod
    def _base_log_record(
        *,
        request_id: str,
        started_at: str,
        requested_model: object,
        requested_service_tier: object,
        request_sha256: str,
    ) -> dict[str, Any]:
        return {
            "schema": REQUEST_LOG_SCHEMA,
            "request_id": request_id,
            "started_at": started_at,
            "finished_at": None,
            "status": None,
            "http_status": None,
            "requested_model": requested_model,
            "requested_service_tier": requested_service_tier,
            "provider_target_model": PROVIDER_MODEL,
            "provider_target_service_tier": SERVICE_TIER,
            "provider_actual_model": None,
            "actual_model": None,
            "returned_alias": None,
            "service_tier": None,
            "response_id": None,
            "request_sha256": request_sha256,
            "provider_request_sha256": None,
            "transformations": [],
            "max_completion_tokens": None,
            "usage": None,
            "cost_nanos": 0,
            "cost_usd": "0",
            "reservation_nanos": 0,
            "physical_attempt_count": 0,
            "physical_retry_count": 0,
            "physical_attempts": [],
            "reservation_retained": False,
            "billable": False,
            "error_code": None,
            "error": None,
        }

    def _append_error(
        self,
        record: dict[str, Any],
        *,
        http_status: int,
        code: str,
        error: str,
    ) -> None:
        record.update(
            {
                "finished_at": utc_now(),
                "status": "error",
                "http_status": http_status,
                "error_code": code,
                "error": error[:500],
            }
        )
        self.request_log.append(record)

    def handle(self, request: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        request_id = f"flex-{uuid.uuid4().hex}"
        started_at = utc_now()
        try:
            local_bytes = canonical_json_bytes(request)
        except (TypeError, ValueError) as exc:
            local_bytes = b"invalid-json-object"
            request = {}
            serialization_error = str(exc)
        else:
            serialization_error = None
        requested_model = request.get("model") if isinstance(request, Mapping) else None
        requested_tier = (
            request.get("service_tier") if isinstance(request, Mapping) else None
        )
        record = self._base_log_record(
            request_id=request_id,
            started_at=started_at,
            requested_model=requested_model,
            requested_service_tier=requested_tier,
            request_sha256=sha256_bytes(local_bytes),
        )
        if serialization_error is not None:
            self._append_error(
                record,
                http_status=400,
                code="invalid_json_value",
                error=serialization_error,
            )
            return self._error_response(400, "invalid_json_value", serialization_error)

        try:
            outbound, transformations, max_completion_tokens = normalize_request(
                request,
                default_max_completion_tokens=self.default_max_completion_tokens,
            )
        except RequestRejected as exc:
            self._append_error(
                record,
                http_status=400,
                code=exc.code,
                error=str(exc),
            )
            return self._error_response(400, exc.code, str(exc))

        outbound_bytes = canonical_json_bytes(outbound)
        provider_request_sha256 = sha256_bytes(outbound_bytes)
        reservation_nanos = maximum_request_cost_nanos(max_completion_tokens)
        record.update(
            {
                "provider_request_sha256": provider_request_sha256,
                "transformations": transformations,
                "max_completion_tokens": max_completion_tokens,
                "reservation_nanos": reservation_nanos,
            }
        )
        try:
            self.cost_store.reserve(
                request_id=request_id,
                cost_nanos=reservation_nanos,
                max_completion_tokens=max_completion_tokens,
                request_sha256=provider_request_sha256,
            )
        except BudgetExceeded as exc:
            self._append_error(
                record,
                http_status=402,
                code="local_cost_budget_exceeded",
                error=str(exc),
            )
            return self._error_response(
                402,
                "local_cost_budget_exceeded",
                str(exc),
            )

        final_response: UpstreamResponse | None = None
        physical_attempts: list[dict[str, Any]] = []
        transport_error: Exception | None = None
        for attempt in range(1, self.max_physical_attempts + 1):
            attempt_started = time.monotonic()
            try:
                response = self.transport.send(outbound_bytes)
            except Exception as exc:  # noqa: BLE001
                transport_error = exc
                physical_attempts.append(
                    {
                        "attempt": attempt,
                        "http_status": None,
                        "provider_request_id": None,
                        "error_code": type(exc).__name__,
                        "error_type": "transport_error",
                        "resource_unavailable": False,
                        "latency_ms": round(
                            (time.monotonic() - attempt_started) * 1_000,
                            3,
                        ),
                    }
                )
                break
            code, kind, _message = _safe_error_fields(response.body)
            resource_unavailable = _is_resource_unavailable(response)
            physical_attempts.append(
                {
                    "attempt": attempt,
                    "http_status": response.status,
                    "provider_request_id": _provider_request_id(response.headers),
                    "error_code": code,
                    "error_type": kind,
                    "resource_unavailable": resource_unavailable,
                    "latency_ms": round(
                        (time.monotonic() - attempt_started) * 1_000,
                        3,
                    ),
                }
            )
            final_response = response
            if not resource_unavailable:
                break
            if attempt < self.max_physical_attempts:
                self.sleep(self.initial_backoff_seconds * (2 ** (attempt - 1)))

        record["physical_attempts"] = physical_attempts
        record["physical_attempt_count"] = len(physical_attempts)
        record["physical_retry_count"] = max(0, len(physical_attempts) - 1)

        if transport_error is not None:
            message = f"{type(transport_error).__name__}: {transport_error}"
            record["reservation_retained"] = True
            self._append_error(
                record,
                http_status=502,
                code="upstream_transport_billing_uncertain",
                error=message,
            )
            return self._error_response(
                502,
                "upstream_transport_billing_uncertain",
                message,
            )
        if final_response is None:
            self.cost_store.release(request_id=request_id)
            message = "no upstream response was produced"
            self._append_error(
                record,
                http_status=502,
                code="upstream_missing_response",
                error=message,
            )
            return self._error_response(502, "upstream_missing_response", message)
        if _is_resource_unavailable(final_response):
            self.cost_store.release(request_id=request_id)
            message = "OpenAI Flex returned 429 Resource Unavailable after retries"
            self._append_error(
                record,
                http_status=503,
                code="flex_resource_unavailable",
                error=message,
            )
            return self._error_response(503, "flex_resource_unavailable", message)
        if final_response.status < 200 or final_response.status >= 300:
            self.cost_store.release(request_id=request_id)
            code, kind, message = _safe_error_fields(final_response.body)
            detail = message or code or kind or f"HTTP {final_response.status}"
            self._append_error(
                record,
                http_status=502,
                code="upstream_rejected_request",
                error=detail,
            )
            return self._error_response(502, "upstream_rejected_request", detail)

        try:
            parsed = json.loads(final_response.body)
            if not isinstance(parsed, dict):
                raise StateError("provider response is not a JSON object")
            usage = normalize_usage(
                parsed.get("usage"),
                max_completion_tokens=max_completion_tokens,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, StateError) as exc:
            record["reservation_retained"] = True
            self._append_error(
                record,
                http_status=502,
                code="upstream_usage_billing_uncertain",
                error=str(exc),
            )
            return self._error_response(
                502,
                "upstream_usage_billing_uncertain",
                str(exc),
            )

        provider_model = parsed.get("model")
        provider_tier = parsed.get("service_tier")
        response_id = parsed.get("id")
        actual_cost_nanos = calculate_cost_nanos(
            prompt_tokens=usage.prompt_tokens,
            cached_tokens=usage.cached_tokens,
            completion_tokens=usage.completion_tokens,
        )
        record.update(
            {
                "provider_actual_model": provider_model,
                "service_tier": provider_tier,
                "response_id": response_id,
                "usage": usage.as_dict(),
                "cost_nanos": actual_cost_nanos,
                "cost_usd": nanos_to_usd(actual_cost_nanos),
                "billable": True,
            }
        )
        try:
            self.cost_store.commit(
                request_id=request_id,
                actual_cost_nanos=actual_cost_nanos,
                usage=usage.as_dict(),
            )
        except StateError as exc:
            record["reservation_retained"] = True
            self._append_error(
                record,
                http_status=500,
                code="cost_state_invariant_failed",
                error=str(exc),
            )
            return self._error_response(500, "cost_state_invariant_failed", str(exc))

        mismatch: tuple[str, str] | None = None
        if provider_model != PROVIDER_MODEL:
            mismatch = (
                "provider_model_mismatch",
                f"provider returned model {provider_model!r}, expected {PROVIDER_MODEL!r}",
            )
        elif provider_tier != SERVICE_TIER:
            mismatch = (
                "provider_service_tier_mismatch",
                f"provider returned service_tier {provider_tier!r}, expected flex",
            )
        elif not isinstance(response_id, str) or not response_id:
            mismatch = ("provider_response_id_missing", "provider response id is absent")
        elif not isinstance(parsed.get("choices"), list) or not parsed["choices"]:
            mismatch = ("provider_choices_missing", "provider choices are absent")
        if mismatch is not None:
            self._append_error(
                record,
                http_status=502,
                code=mismatch[0],
                error=mismatch[1],
            )
            return self._error_response(502, mismatch[0], mismatch[1])

        returned = copy.deepcopy(parsed)
        returned["model"] = REQUESTED_MODEL
        returned["service_tier"] = SERVICE_TIER
        returned["flex_gateway_meta"] = {
            "request_id": request_id,
            "request_sha256": record["request_sha256"],
            "provider_request_sha256": provider_request_sha256,
            "provider_actual_model": PROVIDER_MODEL,
            "returned_alias": REQUESTED_MODEL,
            "service_tier": SERVICE_TIER,
            "physical_attempt_count": len(physical_attempts),
            "physical_retry_count": max(0, len(physical_attempts) - 1),
        }
        returned["proxy_meta"] = {
            "attempts": len(physical_attempts),
            "http_request_id": physical_attempts[-1].get("provider_request_id"),
            "unsupported_parameters": [],
            "provider_actual_model": PROVIDER_MODEL,
            "service_tier": SERVICE_TIER,
        }
        record.update(
            {
                "finished_at": utc_now(),
                "status": "success",
                "http_status": 200,
                "actual_model": REQUESTED_MODEL,
                "returned_alias": REQUESTED_MODEL,
            }
        )
        self.request_log.append(record)
        return 200, returned

    @staticmethod
    def _error_response(
        status: int,
        code: str,
        message: str,
    ) -> tuple[int, dict[str, Any]]:
        return status, {
            "error": {
                "message": message[:500],
                "type": "gpt55_flex_gateway_error",
                "code": code,
            }
        }


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def make_handler(gateway: GPT55FlexGateway) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "GPT55FlexGateway/1"

        def _write_json(self, status: int, value: object) -> None:
            payload = canonical_json_bytes(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/healthz":
                self._write_json(404, {"error": "not found"})
                return
            try:
                self._write_json(200, gateway.health())
            except GatewayError:
                self._write_json(
                    503,
                    {
                        "status": "error",
                        "error": {
                            "type": "local_state_error",
                            "message": "durable cost state is unavailable",
                        },
                    },
                )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._write_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._write_json(400, {"error": "invalid Content-Length"})
                return
            if length < 1 or length > 64 * 1024 * 1024:
                self._write_json(413, {"error": "request body size is invalid"})
                return
            raw = self.rfile.read(length)
            try:
                request = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._write_json(400, {"error": "request body is not valid JSON"})
                return
            if not isinstance(request, dict):
                self._write_json(400, {"error": "request JSON must be an object"})
                return
            try:
                status, response = gateway.handle(request)
            except GatewayError:
                self._write_json(
                    500,
                    {
                        "error": {
                            "type": "local_state_error",
                            "message": "durable gateway state is unavailable",
                        }
                    },
                )
                return
            self._write_json(status, response)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return Handler


def _write_ready(path: Path, *, server_port: int, gateway: GPT55FlexGateway) -> None:
    _atomic_json(
        path,
        {
            "schema": "openai-gpt55-flex-ready/v1",
            "pid": os.getpid(),
            "base_url": f"http://127.0.0.1:{server_port}/v1",
            "health_url": f"http://127.0.0.1:{server_port}/healthz",
            "requested_model": REQUESTED_MODEL,
            "provider_model": PROVIDER_MODEL,
            "service_tier": SERVICE_TIER,
            "max_cost_usd": nanos_to_usd(gateway.max_cost_nanos),
            "started_at": utc_now(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local OpenAI GPT-5.5 snapshot gateway with Flex-only billing",
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--max-cost-usd", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-physical-attempts", type=int, default=5)
    parser.add_argument("--initial-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--default-max-completion-tokens", type=int, default=4_096)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("gateway must bind to loopback (127.0.0.1 or localhost)")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"environment variable {args.api_key_env} is empty or absent")
    transport = OpenAIChatCompletionsTransport(
        api_key=api_key,
        timeout_seconds=args.timeout_seconds,
    )
    gateway = GPT55FlexGateway(
        result_root=args.result_root,
        max_cost_usd=args.max_cost_usd,
        transport=transport,
        default_max_completion_tokens=args.default_max_completion_tokens,
        max_physical_attempts=args.max_physical_attempts,
        initial_backoff_seconds=args.initial_backoff_seconds,
    )
    server = ThreadedHTTPServer((args.host, args.port), make_handler(gateway))
    ready = gateway.result_root / "gateway_ready.json"
    _write_ready(ready, server_port=server.server_port, gateway=gateway)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        ready.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
