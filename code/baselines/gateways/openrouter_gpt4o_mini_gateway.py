"""Fail-closed local cost gateway for OpenRouter GPT-4o-mini.

The HTTP surface accepts only ``openai/gpt-4o-mini``.  Every provider request
is pinned to ``openai/gpt-4o-mini-2024-07-18`` at OpenRouter's fixed Chat
Completions endpoint.  The durable cost gate reserves the maximum possible
request cost before network I/O and never retries an uncertain request.

Tests inject a transport and therefore do not need network access or a key.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import http.client
import json
import os
import re
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
from typing import Any, Iterator, Mapping, Protocol
from urllib.parse import urlsplit


REQUESTED_MODEL = "openai/gpt-4o-mini"
PROVIDER_MODEL = "openai/gpt-4o-mini-2024-07-18"
UPSTREAM_HOST = "openrouter.ai"
UPSTREAM_PATH = "/api/v1/chat/completions"

CONTEXT_WINDOW_TOKENS = 128_000
MAX_OUTPUT_TOKENS = 16_384
NANODOLLARS_PER_USD = 1_000_000_000
UNCACHED_INPUT_NANOS_PER_TOKEN = 150
CACHED_INPUT_NANOS_PER_TOKEN = 75
OUTPUT_NANOS_PER_TOKEN = 600

ROOT_SCHEMA = "openrouter-gpt4o-mini-result-root/v1"
STATE_SCHEMA = "openrouter-gpt4o-mini-cost-state/v1"
REQUEST_LOG_SCHEMA = "openrouter-gpt4o-mini-request/v1"
PRICING_ID = "openrouter-openai-gpt4o-mini-2024-07-18"
ROOT_MARKER_NAME = "openrouter_gpt4o_mini_root.json"
STATE_NAME = "openrouter_cost_state.json"
REQUEST_LOG_NAME = "openrouter_requests.jsonl"
READY_NAME = "gateway_ready.json"
FORBIDDEN_MODEL_ROUTING_FIELDS = frozenset(
    {"models", "route", "provider", "fallbacks"}
)


class GatewayError(RuntimeError):
    """Base class for local gateway failures."""


class RequestRejected(GatewayError):
    def __init__(self, message: str, *, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.code = code


class BudgetExceeded(GatewayError):
    """The conservative reservation would exceed the durable cap."""


class StateError(GatewayError):
    """Durable state or append-only evidence is invalid."""


@dataclass(frozen=True)
class UpstreamResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class UpstreamTransport(Protocol):
    def send(self, payload: bytes) -> UpstreamResponse:
        """Send exactly one physical request."""


class OpenRouterChatCompletionsTransport:
    """Production transport with a non-configurable OpenRouter endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float = 360.0,
        https_proxy: str | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise GatewayError("OpenRouter API key is empty")
        if timeout_seconds <= 0:
            raise GatewayError("timeout_seconds must be positive")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._https_proxy = _parse_https_proxy(https_proxy)

    def send(self, payload: bytes) -> UpstreamResponse:
        if self._https_proxy is None:
            connection = http.client.HTTPSConnection(
                UPSTREAM_HOST, 443, timeout=self._timeout_seconds
            )
        else:
            proxy_host, proxy_port = self._https_proxy
            connection = http.client.HTTPSConnection(
                proxy_host, proxy_port, timeout=self._timeout_seconds
            )
            connection.set_tunnel(UPSTREAM_HOST, 443)
        try:
            connection.request(
                "POST",
                UPSTREAM_PATH,
                body=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "model-aligned-wiki-openrouter-cost-gateway/1",
                },
            )
            response = connection.getresponse()
            body = response.read()
            headers = {key.lower(): value for key, value in response.getheaders()}
            return UpstreamResponse(response.status, headers, body)
        finally:
            connection.close()


def _parse_https_proxy(value: str | None) -> tuple[str, int] | None:
    """Parse an explicit HTTP CONNECT proxy without exposing credentials."""

    if value is None or not value.strip():
        return None
    parsed = urlsplit(value)
    if parsed.scheme.lower() != "http":
        raise GatewayError("HTTPS proxy must use an http:// URL")
    if not parsed.hostname:
        raise GatewayError("HTTPS proxy host is empty")
    if parsed.username is not None or parsed.password is not None:
        raise GatewayError("HTTPS proxy credentials are not supported")
    try:
        port = parsed.port or 80
    except ValueError as exc:
        raise GatewayError("HTTPS proxy port is invalid") from exc
    return parsed.hostname, port


def configured_https_proxy() -> str | None:
    """Use the standard HTTPS proxy environment variables, if configured."""

    return os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")


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


def usd_to_nanos(
    value: str | Decimal | int | float, *, allow_zero: bool = False
) -> int:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise GatewayError(f"invalid USD value: {value!r}") from exc
    if not amount.is_finite() or amount < 0 or (amount == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise GatewayError(f"USD value must be finite and {qualifier}")
    nanos = amount * NANODOLLARS_PER_USD
    if nanos != nanos.to_integral_value():
        raise GatewayError("USD value supports at most 9 decimal places")
    return int(nanos)


def nanos_to_usd(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("nanodollar value must be a non-negative integer")
    rendered = format(Decimal(value) / NANODOLLARS_PER_USD, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StateError(f"{name} must be a non-negative integer")
    return value


def calculate_token_cost_nanos(
    *, prompt_tokens: int, cached_tokens: int, completion_tokens: int
) -> int:
    prompt = _integer(prompt_tokens, name="prompt_tokens")
    cached = _integer(cached_tokens, name="cached_tokens")
    completion = _integer(completion_tokens, name="completion_tokens")
    if cached > prompt:
        raise StateError("cached_tokens exceeds prompt_tokens")
    return (
        (prompt - cached) * UNCACHED_INPUT_NANOS_PER_TOKEN
        + cached * CACHED_INPUT_NANOS_PER_TOKEN
        + completion * OUTPUT_NANOS_PER_TOKEN
    )


def _positive_int(value: object, *, name: str, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestRejected(f"{name} must be an integer")
    if not 1 <= value <= upper:
        raise RequestRejected(f"{name} must be between 1 and {upper}")
    return value


def maximum_request_cost_nanos(max_completion_tokens: int) -> int:
    maximum = _positive_int(
        max_completion_tokens,
        name="max_completion_tokens",
        upper=MAX_OUTPUT_TOKENS,
    )
    maximum_prompt = CONTEXT_WINDOW_TOKENS - maximum
    return (
        maximum_prompt * UNCACHED_INPUT_NANOS_PER_TOKEN
        + maximum * OUTPUT_NANOS_PER_TOKEN
    )


def _atomic_json(path: Path, value: object) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
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
        raise StateError(f"JSON is not a regular file: {path}")
    metadata = path.stat(follow_symlinks=False)
    if metadata.st_nlink != 1:
        raise StateError(f"JSON file must not be hardlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StateError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise StateError(f"JSON is not an object: {path}")
    return value


class DurableCostStore:
    """Process-safe conservative reservations and committed provider cost."""

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
        now = utc_now()
        return {
            "schema": STATE_SCHEMA,
            "pricing_id": PRICING_ID,
            "provider": "OpenRouter",
            "provider_model": PROVIDER_MODEL,
            "returned_alias": REQUESTED_MODEL,
            "cross_model_fallbacks": False,
            "context_window_tokens": CONTEXT_WINDOW_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "pricing_nanos_per_token": {
                "uncached_input": UNCACHED_INPUT_NANOS_PER_TOKEN,
                "cached_input": CACHED_INPUT_NANOS_PER_TOKEN,
                "output": OUTPUT_NANOS_PER_TOKEN,
            },
            "max_cost_nanos": self.max_cost_nanos,
            "max_cost_usd": nanos_to_usd(self.max_cost_nanos),
            "committed_cost_nanos": 0,
            "reserved_cost_nanos": 0,
            "billable_request_count": 0,
            "retained_failure_count": 0,
            "provider_exact_cost_count": 0,
            "token_derived_cost_count": 0,
            "usage": {
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "completion_tokens": 0,
            },
            "reservations": {},
            "created_at": now,
            "updated_at": now,
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
            "provider": "OpenRouter",
            "provider_model": PROVIDER_MODEL,
            "returned_alias": REQUESTED_MODEL,
            "cross_model_fallbacks": False,
            "context_window_tokens": CONTEXT_WINDOW_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "pricing_nanos_per_token": {
                "uncached_input": UNCACHED_INPUT_NANOS_PER_TOKEN,
                "cached_input": CACHED_INPUT_NANOS_PER_TOKEN,
                "output": OUTPUT_NANOS_PER_TOKEN,
            },
            "max_cost_nanos": self.max_cost_nanos,
            "max_cost_usd": nanos_to_usd(self.max_cost_nanos),
        }
        for name, expected in exact.items():
            if state.get(name) != expected:
                raise StateError(f"cost-state {name} mismatch")
        committed = _integer(
            state.get("committed_cost_nanos"), name="committed_cost_nanos"
        )
        reserved = _integer(
            state.get("reserved_cost_nanos"), name="reserved_cost_nanos"
        )
        if committed + reserved > self.max_cost_nanos:
            raise StateError("cost-state exceeds max_cost_nanos")
        reservations = state.get("reservations")
        if not isinstance(reservations, dict):
            raise StateError("cost-state reservations is not an object")
        reservation_sum = 0
        retained = 0
        for request_id, reservation in reservations.items():
            if not isinstance(request_id, str) or not isinstance(reservation, dict):
                raise StateError("invalid cost-state reservation")
            reservation_sum += _integer(
                reservation.get("cost_nanos"), name="reservation.cost_nanos"
            )
            retained += int(reservation.get("retained_reason") is not None)
        if reservation_sum != reserved:
            raise StateError("cost-state reservation sum mismatch")
        if retained != _integer(
            state.get("retained_failure_count"), name="retained_failure_count"
        ):
            raise StateError("cost-state retained failure count mismatch")
        for name in (
            "billable_request_count",
            "provider_exact_cost_count",
            "token_derived_cost_count",
        ):
            _integer(state.get(name), name=name)
        if (
            state["provider_exact_cost_count"] + state["token_derived_cost_count"]
            != state["billable_request_count"]
        ):
            raise StateError("cost-state cost-source count mismatch")
        usage = state.get("usage")
        if not isinstance(usage, dict):
            raise StateError("cost-state usage is not an object")
        for name in ("prompt_tokens", "cached_tokens", "completion_tokens"):
            _integer(usage.get(name), name=f"usage.{name}")

    def reserve(
        self,
        *,
        request_id: str,
        cost_nanos: int,
        max_completion_tokens: int,
        request_sha256: str,
    ) -> None:
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
                    f"request={nanos_to_usd(cost_nanos)}, "
                    f"cap={nanos_to_usd(self.max_cost_nanos)}"
                )
            state["reservations"][request_id] = {
                "cost_nanos": cost_nanos,
                "max_completion_tokens": max_completion_tokens,
                "request_sha256": request_sha256,
                "retained_reason": None,
                "created_at": utc_now(),
            }
            state["reserved_cost_nanos"] += cost_nanos
            state["updated_at"] = utc_now()
            _atomic_json(self.state_path, state)

    def retain_failure(self, *, request_id: str, reason: str) -> None:
        with self._locked():
            state = self._load()
            self._validate(state)
            reservation = state["reservations"].get(request_id)
            if not isinstance(reservation, dict):
                raise StateError("request reservation is absent at retain")
            if reservation.get("retained_reason") is None:
                reservation["retained_reason"] = reason[:200]
                reservation["retained_at"] = utc_now()
                state["retained_failure_count"] += 1
            state["updated_at"] = utc_now()
            _atomic_json(self.state_path, state)

    def commit(
        self,
        *,
        request_id: str,
        actual_cost_nanos: int,
        cost_source: str,
        usage: Mapping[str, int],
    ) -> None:
        if cost_source not in {"provider_usage_cost", "token_derived"}:
            raise StateError("invalid committed cost source")
        with self._locked():
            state = self._load()
            self._validate(state)
            reservation = state["reservations"].get(request_id)
            if not isinstance(reservation, dict):
                raise StateError("request reservation is absent at commit")
            if reservation.get("retained_reason") is not None:
                raise StateError("retained uncertain reservation cannot be committed")
            reserved = reservation["cost_nanos"]
            if actual_cost_nanos > reserved:
                raise StateError("actual cost exceeds conservative reservation")
            state["reservations"].pop(request_id)
            state["reserved_cost_nanos"] -= reserved
            state["committed_cost_nanos"] += actual_cost_nanos
            state["billable_request_count"] += 1
            counter = (
                "provider_exact_cost_count"
                if cost_source == "provider_usage_cost"
                else "token_derived_cost_count"
            )
            state[counter] += 1
            for name in ("prompt_tokens", "cached_tokens", "completion_tokens"):
                state["usage"][name] += _integer(
                    usage[name], name=f"usage.{name}"
                )
            state["updated_at"] = utc_now()
            _atomic_json(self.state_path, state)

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
            self._validate_file()
            flags = os.O_WRONLY | os.O_APPEND
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.path, flags)
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
    total_tokens: int
    provider_cost: str | None
    provider_cost_nanos: int | None

    def token_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def normalize_usage(value: object, *, max_completion_tokens: int) -> NormalizedUsage:
    if not isinstance(value, dict):
        raise StateError("provider usage is absent or not an object")

    def count(name: str) -> int:
        return _integer(value.get(name), name=f"provider usage {name}")

    prompt = count("prompt_tokens")
    completion = count("completion_tokens")
    total = count("total_tokens")
    details = value.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        raise StateError("provider prompt_tokens_details is invalid")
    cached = _integer(details.get("cached_tokens", 0), name="provider cached_tokens")
    if cached > prompt:
        raise StateError("provider cached tokens exceed prompt tokens")
    if total != prompt + completion:
        raise StateError("provider total_tokens differs from prompt+completion")
    if completion > max_completion_tokens:
        raise StateError("provider completion usage exceeds requested maximum")
    if total > CONTEXT_WINDOW_TOKENS:
        raise StateError("provider usage exceeds GPT-4o-mini context window")
    raw_cost = value.get("cost")
    if raw_cost is None:
        provider_cost = None
        provider_cost_nanos = None
    else:
        provider_cost_nanos = usd_to_nanos(raw_cost, allow_zero=True)
        provider_cost = nanos_to_usd(provider_cost_nanos)
    return NormalizedUsage(
        prompt,
        cached,
        completion,
        total,
        provider_cost,
        provider_cost_nanos,
    )


def normalize_request(
    request: Mapping[str, Any], *, default_max_tokens: int
) -> tuple[dict[str, Any], list[str], int]:
    if not isinstance(request, Mapping):
        raise RequestRejected("request JSON must be an object")
    if request.get("model") != REQUESTED_MODEL:
        raise RequestRejected(
            f"local gateway only accepts model {REQUESTED_MODEL}",
            code="model_mismatch",
        )
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestRejected("messages must be a non-empty list")
    if request.get("stream") not in (None, False):
        raise RequestRejected(
            "streaming is disabled because complete usage is required",
            code="streaming_not_supported",
        )
    forbidden_routing = FORBIDDEN_MODEL_ROUTING_FIELDS.intersection(request)
    if forbidden_routing:
        raise RequestRejected(
            "model routing fields are forbidden: "
            f"{sorted(forbidden_routing)}",
            code="model_routing_not_allowed",
        )
    outbound = copy.deepcopy(dict(request))
    legacy = outbound.get("max_tokens")
    current = outbound.get("max_completion_tokens")
    if legacy is not None and current is not None and legacy != current:
        raise RequestRejected(
            "max_tokens and max_completion_tokens differ",
            code="max_token_conflict",
        )
    selected = legacy if legacy is not None else current
    transformations: list[str] = []
    if selected is None:
        selected = default_max_tokens
        outbound["max_tokens"] = selected
        transformations.append("default_max_tokens_added")
    maximum = _positive_int(selected, name="max_tokens", upper=MAX_OUTPUT_TOKENS)
    temperature = outbound.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise RequestRejected("temperature must be numeric")
        if not 0 <= float(temperature) <= 2:
            raise RequestRejected("temperature must be between 0 and 2")
    outbound["model"] = PROVIDER_MODEL
    outbound["provider"] = {"allow_fallbacks": False}
    transformations.append("provider_allow_fallbacks_false_added")
    outbound["stream"] = False
    return outbound, transformations, maximum


def _safe_error(body: bytes) -> tuple[str | None, str | None, str | None]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, None, None
    if not isinstance(value, dict) or not isinstance(value.get("error"), dict):
        return None, None, None
    error = value["error"]
    return tuple(
        str(error.get(name))[:300] if error.get(name) is not None else None
        for name in ("code", "type", "message")
    )


def _provider_request_id(headers: Mapping[str, str]) -> str | None:
    value = headers.get("x-request-id") or headers.get("request-id")
    return str(value)[:200] if value else None


def _redacted_error(value: str) -> str:
    """Keep diagnostics without persisting credentials or bearer-like tokens."""
    text = re.sub(
        r"(?i)(authorization|api[-_ ]?key|bearer)\s*[:=]?\s*[^\s,;]+",
        r"\1=[REDACTED]",
        value,
    )
    text = re.sub(r"\b(?:sk|or)-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", text)
    return text[:500]


class OpenRouterCostGateway:
    """Core request processor; safe for concurrent HTTP threads."""

    def __init__(
        self,
        *,
        result_root: Path,
        max_cost_usd: str | Decimal | int | float,
        transport: UpstreamTransport,
        default_max_tokens: int = 1_024,
    ) -> None:
        self.result_root = result_root.expanduser().resolve()
        self.max_cost_nanos = usd_to_nanos(max_cost_usd)
        self.transport = transport
        self.default_max_tokens = _positive_int(
            default_max_tokens, name="default_max_tokens", upper=MAX_OUTPUT_TOKENS
        )
        self._prepare_result_root()
        self.state_path = self.result_root / STATE_NAME
        self.request_log_path = self.result_root / REQUEST_LOG_NAME
        self.cost_store = DurableCostStore(
            state_path=self.state_path, max_cost_nanos=self.max_cost_nanos
        )
        self.request_log = AppendOnlyRequestLog(self.request_log_path)

    def _prepare_result_root(self) -> None:
        if self.result_root.exists() and not self.result_root.is_dir():
            raise StateError("result root is not a directory")
        self.result_root.mkdir(parents=True, exist_ok=True)
        marker_path = self.result_root / ROOT_MARKER_NAME
        expected = {
            "schema": ROOT_SCHEMA,
            "provider": "OpenRouter",
            "api": "chat_completions",
            "upstream_url": f"https://{UPSTREAM_HOST}{UPSTREAM_PATH}",
            "provider_model": PROVIDER_MODEL,
            "returned_alias": REQUESTED_MODEL,
            "cross_model_fallbacks": False,
            "billing": "openrouter_api",
            "routing_policy": "exact_model_no_fallback",
        }
        if marker_path.exists():
            marker = _read_regular_json(marker_path)
            for name, expected_value in expected.items():
                if marker.get(name) != expected_value:
                    raise StateError(f"result-root marker {name} mismatch")
            return
        if any(self.result_root.iterdir()):
            raise StateError(
                "result root is non-empty without the OpenRouter marker; "
                "use an isolated result root"
            )
        _atomic_json(marker_path, {**expected, "created_at": utc_now()})

    def health(self) -> dict[str, Any]:
        state = self.cost_store.snapshot()
        remaining = (
            state["max_cost_nanos"]
            - state["committed_cost_nanos"]
            - state["reserved_cost_nanos"]
        )
        return {
            "status": "ok",
            "schema": "openrouter-gpt4o-mini-health/v1",
            "requested_model": REQUESTED_MODEL,
            "provider_model": PROVIDER_MODEL,
            "cross_model_fallbacks": False,
            "budget": {
                "max_cost_usd": state["max_cost_usd"],
                "committed_cost_usd": nanos_to_usd(state["committed_cost_nanos"]),
                "reserved_cost_usd": nanos_to_usd(state["reserved_cost_nanos"]),
                "remaining_cost_usd": nanos_to_usd(remaining),
                "in_flight": len(state["reservations"]),
            },
        }

    @staticmethod
    def _base_record(
        *, request_id: str, request_sha256: str, requested_model: object
    ) -> dict[str, Any]:
        return {
            "schema": REQUEST_LOG_SCHEMA,
            "request_id": request_id,
            "started_at": utc_now(),
            "finished_at": None,
            "status": None,
            "http_status": None,
            "requested_model": requested_model,
            "provider_target_model": PROVIDER_MODEL,
            "provider_actual_model": None,
            "actual_model": None,
            "returned_alias": None,
            "response_id": None,
            "provider_request_id": None,
            "request_sha256": request_sha256,
            "provider_request_sha256": None,
            "response_sha256": None,
            "transformations": [],
            "max_completion_tokens": None,
            "usage": None,
            "provider_usage_cost_usd": None,
            "provider_usage_cost_nanos": None,
            "token_derived_cost_usd": None,
            "token_derived_cost_nanos": None,
            "committed_cost_source": None,
            "committed_cost_usd": "0",
            "committed_cost_nanos": 0,
            "reservation_nanos": 0,
            "physical_attempt_count": 0,
            "reservation_retained": False,
            "billable": False,
            "error_code": None,
            "error_type": None,
            "error": None,
        }

    def _append_error(
        self,
        record: dict[str, Any],
        *,
        http_status: int,
        code: str,
        error: str,
        error_type: str | None = None,
    ) -> None:
        persisted_error = _redacted_error(error)
        if code in {
            "upstream_transport_billing_uncertain",
            "upstream_error_billing_uncertain",
            "upstream_usage_cost_billing_uncertain",
        }:
            persisted_error = error_type or code
        record.update(
            {
                "finished_at": utc_now(),
                "status": "error",
                "http_status": http_status,
                "error_code": code,
                "error_type": error_type,
                "error": persisted_error,
            }
        )
        self.request_log.append(record)

    def _retain_and_error(
        self,
        *,
        request_id: str,
        record: dict[str, Any],
        http_status: int,
        code: str,
        error: str,
        error_type: str | None = None,
    ) -> tuple[int, dict[str, Any]]:
        self.cost_store.retain_failure(request_id=request_id, reason=code)
        record["reservation_retained"] = True
        self._append_error(
            record,
            http_status=http_status,
            code=code,
            error=error,
            error_type=error_type,
        )
        return self._error_response(http_status, code, error)

    def handle(self, request: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        request_id = f"openrouter-{uuid.uuid4().hex}"
        try:
            local_bytes = canonical_json_bytes(request)
        except (TypeError, ValueError) as exc:
            local_bytes = b"invalid-json-object"
            request = {}
            serialization_error = str(exc)
        else:
            serialization_error = None
        requested_model = request.get("model") if isinstance(request, Mapping) else None
        record = self._base_record(
            request_id=request_id,
            request_sha256=sha256_bytes(local_bytes),
            requested_model=requested_model,
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
            outbound, transformations, max_tokens = normalize_request(
                request, default_max_tokens=self.default_max_tokens
            )
        except RequestRejected as exc:
            self._append_error(
                record, http_status=400, code=exc.code, error=str(exc)
            )
            return self._error_response(400, exc.code, str(exc))

        outbound_bytes = canonical_json_bytes(outbound)
        provider_request_sha256 = sha256_bytes(outbound_bytes)
        reservation_nanos = maximum_request_cost_nanos(max_tokens)
        record.update(
            {
                "provider_request_sha256": provider_request_sha256,
                "transformations": transformations,
                "max_completion_tokens": max_tokens,
                "reservation_nanos": reservation_nanos,
            }
        )
        try:
            self.cost_store.reserve(
                request_id=request_id,
                cost_nanos=reservation_nanos,
                max_completion_tokens=max_tokens,
                request_sha256=provider_request_sha256,
            )
        except BudgetExceeded as exc:
            self._append_error(
                record,
                http_status=402,
                code="local_cost_budget_exceeded",
                error=str(exc),
            )
            return self._error_response(402, "local_cost_budget_exceeded", str(exc))

        started = time.monotonic()
        try:
            response = self.transport.send(outbound_bytes)
        except Exception as exc:  # noqa: BLE001
            record["physical_attempt_count"] = 1
            return self._retain_and_error(
                request_id=request_id,
                record=record,
                http_status=502,
                code="upstream_transport_billing_uncertain",
                error=f"{type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
            )
        record["physical_attempt_count"] = 1
        record["provider_request_id"] = _provider_request_id(response.headers)
        record["response_sha256"] = sha256_bytes(response.body)
        record["latency_ms"] = round((time.monotonic() - started) * 1_000, 3)
        if not 200 <= response.status < 300:
            code, kind, message = _safe_error(response.body)
            detail = message or code or kind or f"HTTP {response.status}"
            return self._retain_and_error(
                request_id=request_id,
                record=record,
                http_status=502,
                code="upstream_error_billing_uncertain",
                error=detail,
                error_type=kind,
            )

        try:
            parsed = json.loads(response.body)
            if not isinstance(parsed, dict):
                raise StateError("provider response is not a JSON object")
            usage = normalize_usage(
                parsed.get("usage"), max_completion_tokens=max_tokens
            )
            provider_model = parsed.get("model")
            response_id = parsed.get("id")
            token_cost_nanos = calculate_token_cost_nanos(
                prompt_tokens=usage.prompt_tokens,
                cached_tokens=usage.cached_tokens,
                completion_tokens=usage.completion_tokens,
            )
            committed_cost_nanos = (
                usage.provider_cost_nanos
                if usage.provider_cost_nanos is not None
                else token_cost_nanos
            )
            cost_source = (
                "provider_usage_cost"
                if usage.provider_cost_nanos is not None
                else "token_derived"
            )
        except (UnicodeDecodeError, json.JSONDecodeError, GatewayError) as exc:
            return self._retain_and_error(
                request_id=request_id,
                record=record,
                http_status=502,
                code="upstream_usage_cost_billing_uncertain",
                error=str(exc),
                error_type=type(exc).__name__,
            )

        record.update(
            {
                "provider_actual_model": provider_model,
                "response_id": response_id,
                "usage": usage.token_dict(),
                "provider_usage_cost_usd": usage.provider_cost,
                "provider_usage_cost_nanos": usage.provider_cost_nanos,
                "token_derived_cost_usd": nanos_to_usd(token_cost_nanos),
                "token_derived_cost_nanos": token_cost_nanos,
                "committed_cost_source": cost_source,
                "committed_cost_usd": nanos_to_usd(committed_cost_nanos),
                "committed_cost_nanos": committed_cost_nanos,
                "billable": True,
            }
        )
        try:
            self.cost_store.commit(
                request_id=request_id,
                actual_cost_nanos=committed_cost_nanos,
                cost_source=cost_source,
                usage=usage.token_dict(),
            )
        except StateError as exc:
            return self._retain_and_error(
                request_id=request_id,
                record=record,
                http_status=500,
                code="cost_state_invariant_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )

        mismatch: tuple[str, str] | None = None
        if provider_model != PROVIDER_MODEL:
            mismatch = (
                "provider_model_mismatch",
                f"provider returned model {provider_model!r}, expected {PROVIDER_MODEL!r}",
            )
        elif not isinstance(response_id, str) or not response_id.strip():
            mismatch = ("provider_response_id_missing", "provider response id is absent")
        elif not isinstance(parsed.get("choices"), list) or not parsed["choices"]:
            mismatch = ("provider_choices_missing", "provider choices are absent")
        if mismatch is not None:
            self._append_error(
                record, http_status=502, code=mismatch[0], error=mismatch[1]
            )
            return self._error_response(502, mismatch[0], mismatch[1])

        returned = copy.deepcopy(parsed)
        returned["model"] = REQUESTED_MODEL
        returned["openrouter_gateway_meta"] = {
            "schema": "openrouter-gpt4o-mini-response-meta/v1",
            "request_id": request_id,
            "request_sha256": record["request_sha256"],
            "provider_request_sha256": provider_request_sha256,
            "provider_actual_model": PROVIDER_MODEL,
            "returned_alias": REQUESTED_MODEL,
            "response_id": response_id,
            "provider_usage_cost_usd": usage.provider_cost,
            "token_derived_cost_usd": nanos_to_usd(token_cost_nanos),
            "committed_cost_source": cost_source,
            "committed_cost_usd": nanos_to_usd(committed_cost_nanos),
        }
        returned["proxy_meta"] = {
            "attempts": 1,
            "http_request_id": record["provider_request_id"],
            "unsupported_parameters": [],
            "provider_actual_model": PROVIDER_MODEL,
            "provider": "OpenRouter",
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
    def _error_response(status: int, code: str, message: str) -> tuple[int, dict[str, Any]]:
        return status, {
            "error": {
                "message": message[:500],
                "type": "openrouter_cost_gateway_error",
                "code": code,
            }
        }


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def make_handler(gateway: OpenRouterCostGateway) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "OpenRouterCostGateway/1"

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
                self._write_json(503, {"error": "durable cost state unavailable"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._write_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._write_json(400, {"error": "invalid Content-Length"})
                return
            if not 1 <= length <= 64 * 1024 * 1024:
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
                self._write_json(500, {"error": "durable gateway state unavailable"})
                return
            self._write_json(status, response)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return Handler


def _write_ready(path: Path, *, server_port: int, gateway: OpenRouterCostGateway) -> None:
    _atomic_json(
        path,
        {
            "schema": "openrouter-gpt4o-mini-ready/v1",
            "pid": os.getpid(),
            "base_url": f"http://127.0.0.1:{server_port}/v1",
            "health_url": f"http://127.0.0.1:{server_port}/healthz",
            "result_root": str(gateway.result_root),
            "root_marker": str(gateway.result_root / ROOT_MARKER_NAME),
            "state": str(gateway.state_path),
            "request_log": str(gateway.request_log_path),
            "requested_model": REQUESTED_MODEL,
            "provider_model": PROVIDER_MODEL,
            "cross_model_fallbacks": False,
            "max_cost_usd": nanos_to_usd(gateway.max_cost_nanos),
            "started_at": utc_now(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local OpenRouter GPT-4o-mini gateway with a durable USD cap"
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--max-cost-usd", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8201)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--https-proxy",
        default=configured_https_proxy(),
        help="HTTP CONNECT proxy for OpenRouter (defaults to HTTPS_PROXY)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=360.0)
    parser.add_argument("--default-max-tokens", type=int, default=1_024)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("gateway must bind to loopback")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"environment variable {args.api_key_env} is empty or absent")
    gateway = OpenRouterCostGateway(
        result_root=args.result_root,
        max_cost_usd=args.max_cost_usd,
        transport=OpenRouterChatCompletionsTransport(
            api_key=api_key,
            timeout_seconds=args.timeout_seconds,
            https_proxy=args.https_proxy,
        ),
        default_max_tokens=args.default_max_tokens,
    )
    server = ThreadedHTTPServer((args.host, args.port), make_handler(gateway))
    ready = gateway.result_root / READY_NAME
    _write_ready(ready, server_port=server.server_port, gateway=gateway)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        ready.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
