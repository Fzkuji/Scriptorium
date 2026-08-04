#!/usr/bin/env python3
"""Independent offline audit for the GPT-5.5 Flex gateway artifacts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator, Mapping


REQUESTED_MODEL = "gpt-5.5"
PROVIDER_MODEL = "gpt-5.5-2026-04-23"
SERVICE_TIER = "flex"
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

FORBIDDEN_LOG_KEYS = {
    "authorization",
    "api_key",
    "openai_api_key",
    "messages",
    "request_body",
    "provider_payload",
    "outbound_payload",
}


class AuditError(RuntimeError):
    """Gateway artifacts do not prove the required execution contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usd_to_nanos(value: str) -> int:
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise AuditError(f"invalid --max-cost-usd: {value!r}") from exc
    if not amount.is_finite() or amount <= 0:
        raise AuditError("--max-cost-usd must be finite and positive")
    nanos = amount * NANODOLLARS_PER_USD
    if nanos != nanos.to_integral_value():
        raise AuditError("--max-cost-usd supports at most 9 decimal places")
    return int(nanos)


def _nanos_to_usd(value: int) -> str:
    rendered = format(Decimal(value) / NANODOLLARS_PER_USD, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuditError(f"{name} must be a non-negative integer")
    return value


def _cost_nanos(*, prompt: int, cached: int, completion: int) -> int:
    if cached > prompt:
        raise AuditError("cached tokens exceed prompt tokens")
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


def _maximum_request_cost_nanos(max_completion_tokens: object) -> int:
    maximum = _integer(
        max_completion_tokens,
        name="request max_completion_tokens",
    )
    if maximum < 1 or maximum > MAX_OUTPUT_TOKENS:
        raise AuditError("request max_completion_tokens is outside model limits")
    maximum_prompt = CONTEXT_WINDOW_TOKENS - maximum
    if maximum_prompt <= LONG_CONTEXT_THRESHOLD:
        return _cost_nanos(prompt=maximum_prompt, cached=0, completion=maximum)
    return (
        maximum_prompt * LONG_UNCACHED_INPUT_NANOS_PER_TOKEN
        + maximum * LONG_OUTPUT_NANOS_PER_TOKEN
    )


def _read_regular_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"required artifact is not a regular file: {path}")
    metadata = path.stat(follow_symlinks=False)
    if metadata.st_nlink != 1:
        raise AuditError(f"artifact must not be hardlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON artifact is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"request log is not a regular file: {path}")
    metadata = path.stat(follow_symlinks=False)
    if metadata.st_nlink != 1:
        raise AuditError("request log must not be hardlinked")
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise AuditError("request log has an incomplete final line")
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuditError(f"invalid request-log JSON at line {line_number}") from exc
        if not isinstance(value, dict):
            raise AuditError(f"request-log line {line_number} is not an object")
        result.append(value)
    return result


def _walk_keys(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _sha256(value: object, *, name: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise AuditError(f"{name} is not a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise AuditError(f"{name} is not a SHA-256 hex digest") from exc
    return value


def _usage(value: object, *, line_number: int) -> dict[str, int]:
    if not isinstance(value, dict):
        raise AuditError(f"line {line_number} billable usage is absent")
    names = (
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
    )
    result = {
        name: _integer(value.get(name), name=f"line {line_number} usage.{name}")
        for name in names
    }
    if result["cached_tokens"] > result["prompt_tokens"]:
        raise AuditError(f"line {line_number} cached tokens exceed prompt tokens")
    if result["reasoning_tokens"] > result["completion_tokens"]:
        raise AuditError(f"line {line_number} reasoning tokens exceed completion tokens")
    if result["total_tokens"] != (
        result["prompt_tokens"] + result["completion_tokens"]
    ):
        raise AuditError(f"line {line_number} total token count differs")
    if result["total_tokens"] > CONTEXT_WINDOW_TOKENS:
        raise AuditError(f"line {line_number} exceeds the model context window")
    return result


def _audit_attempts(entry: Mapping[str, Any], *, line_number: int) -> None:
    attempts = entry.get("physical_attempts")
    if not isinstance(attempts, list):
        raise AuditError(f"line {line_number} physical_attempts is not a list")
    count = _integer(
        entry.get("physical_attempt_count"),
        name=f"line {line_number} physical_attempt_count",
    )
    retries = _integer(
        entry.get("physical_retry_count"),
        name=f"line {line_number} physical_retry_count",
    )
    if count != len(attempts) or retries != max(0, count - 1):
        raise AuditError(f"line {line_number} physical attempt totals differ")
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict) or attempt.get("attempt") != index:
            raise AuditError(f"line {line_number} physical attempt sequence differs")
        if index < len(attempts):
            if (
                attempt.get("http_status") != 429
                or attempt.get("resource_unavailable") is not True
            ):
                raise AuditError(
                    f"line {line_number} retried a non-resource-unavailable response"
                )


def _audit_root_marker(root: Path) -> None:
    marker = _read_regular_json(root / ROOT_MARKER_NAME)
    expected = {
        "schema": ROOT_SCHEMA,
        "provider": "openai_api",
        "api": "chat_completions",
        "upstream_origin": "https://api.openai.com",
        "upstream_path": "/v1/chat/completions",
        "provider_model": PROVIDER_MODEL,
        "returned_alias": REQUESTED_MODEL,
        "service_tier": SERVICE_TIER,
        "billing": "api_flex",
    }
    for name, value in expected.items():
        if marker.get(name) != value:
            raise AuditError(f"result-root marker {name} mismatch")


def audit(
    *,
    result_root: Path,
    max_cost_usd: str,
    allow_in_flight: bool = False,
) -> dict[str, Any]:
    root = result_root.expanduser().resolve()
    if not root.is_dir():
        raise AuditError(f"result root is not a directory: {root}")
    expected_max_nanos = _usd_to_nanos(max_cost_usd)
    _audit_root_marker(root)
    state = _read_regular_json(root / STATE_NAME)
    entries = _read_jsonl(root / REQUEST_LOG_NAME)

    expected_state = {
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
        "max_cost_nanos": expected_max_nanos,
        "max_cost_usd": _nanos_to_usd(expected_max_nanos),
    }
    for name, value in expected_state.items():
        if state.get(name) != value:
            raise AuditError(f"cost-state {name} mismatch")

    seen: set[str] = set()
    billable_count = 0
    success_count = 0
    error_count = 0
    released_failed_count = 0
    retained_reservation_ids: set[str] = set()
    committed_cost_nanos = 0
    aggregate_usage = {
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
    }
    for line_number, entry in enumerate(entries, start=1):
        if entry.get("schema") != REQUEST_LOG_SCHEMA:
            raise AuditError(f"request-log schema mismatch at line {line_number}")
        request_id = entry.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise AuditError(f"line {line_number} request_id is absent")
        if request_id in seen:
            raise AuditError(f"duplicate request_id at line {line_number}")
        seen.add(request_id)
        forbidden = FORBIDDEN_LOG_KEYS.intersection(_walk_keys(entry))
        if forbidden:
            raise AuditError(
                f"line {line_number} contains forbidden raw/secret keys: "
                f"{sorted(forbidden)}"
            )
        if entry.get("provider_target_model") != PROVIDER_MODEL:
            raise AuditError(f"line {line_number} provider target model mismatch")
        if entry.get("provider_target_service_tier") != SERVICE_TIER:
            raise AuditError(f"line {line_number} provider target tier mismatch")
        _sha256(entry.get("request_sha256"), name=f"line {line_number} request_sha256")
        provider_sha = entry.get("provider_request_sha256")
        _sha256(
            provider_sha,
            name=f"line {line_number} provider_request_sha256",
            nullable=True,
        )
        _audit_attempts(entry, line_number=line_number)
        reservation = _integer(
            entry.get("reservation_nanos"),
            name=f"line {line_number} reservation_nanos",
        )
        max_completion = entry.get("max_completion_tokens")
        if provider_sha is None:
            if reservation != 0 or max_completion is not None:
                raise AuditError(
                    f"line {line_number} local rejection has a provider reservation"
                )
        else:
            expected_reservation = _maximum_request_cost_nanos(max_completion)
            if reservation != expected_reservation:
                raise AuditError(
                    f"line {line_number} conservative reservation differs"
                )
        status = entry.get("status")
        if status not in {"success", "error"}:
            raise AuditError(f"line {line_number} status is invalid")
        if status == "success":
            success_count += 1
            if entry.get("http_status") != 200:
                raise AuditError(f"line {line_number} successful HTTP status differs")
            if entry.get("requested_model") != REQUESTED_MODEL:
                raise AuditError(f"line {line_number} requested alias mismatch")
            if entry.get("provider_actual_model") != PROVIDER_MODEL:
                raise AuditError(f"line {line_number} provider actual model mismatch")
            if entry.get("actual_model") != REQUESTED_MODEL:
                raise AuditError(f"line {line_number} returned model mismatch")
            if entry.get("returned_alias") != REQUESTED_MODEL:
                raise AuditError(f"line {line_number} returned alias mismatch")
            if entry.get("service_tier") != SERVICE_TIER:
                raise AuditError(f"line {line_number} service tier mismatch")
            if not isinstance(entry.get("response_id"), str) or not entry["response_id"]:
                raise AuditError(f"line {line_number} response_id is absent")
            if provider_sha is None:
                raise AuditError(f"line {line_number} provider request SHA is absent")
        else:
            error_count += 1
            if not isinstance(entry.get("error_code"), str):
                raise AuditError(f"line {line_number} error_code is absent")
            if (
                entry.get("billable") is True
                and entry.get("provider_actual_model") != PROVIDER_MODEL
                and entry.get("error_code") != "provider_model_mismatch"
            ):
                raise AuditError(f"line {line_number} did not flag provider model mismatch")
            if (
                entry.get("billable") is True
                and entry.get("provider_actual_model") == PROVIDER_MODEL
                and entry.get("service_tier") != SERVICE_TIER
                and entry.get("error_code") != "provider_service_tier_mismatch"
            ):
                raise AuditError(f"line {line_number} did not flag provider tier mismatch")
            if entry.get("error_code") in {
                "upstream_missing_response",
                "flex_resource_unavailable",
                "upstream_rejected_request",
            }:
                released_failed_count += 1

        retained = entry.get("reservation_retained")
        if not isinstance(retained, bool):
            raise AuditError(f"line {line_number} reservation_retained is not boolean")
        if retained:
            if status != "error":
                raise AuditError(f"line {line_number} retained a successful reservation")
            retained_reservation_ids.add(request_id)

        billable = entry.get("billable")
        if not isinstance(billable, bool):
            raise AuditError(f"line {line_number} billable is not boolean")
        cost = _integer(entry.get("cost_nanos"), name=f"line {line_number} cost")
        if entry.get("cost_usd") != _nanos_to_usd(cost):
            raise AuditError(f"line {line_number} cost rendering mismatch")
        if billable:
            billable_count += 1
            usage = _usage(entry.get("usage"), line_number=line_number)
            expected_cost = _cost_nanos(
                prompt=usage["prompt_tokens"],
                cached=usage["cached_tokens"],
                completion=usage["completion_tokens"],
            )
            if cost != expected_cost:
                raise AuditError(f"line {line_number} token cost mismatch")
            committed_cost_nanos += cost
            for name in aggregate_usage:
                aggregate_usage[name] += usage[name]
        elif cost != 0 or entry.get("usage") is not None:
            raise AuditError(f"line {line_number} non-billable entry has usage/cost")

    state_committed = _integer(
        state.get("committed_cost_nanos"),
        name="state committed_cost_nanos",
    )
    state_reserved = _integer(
        state.get("reserved_cost_nanos"),
        name="state reserved_cost_nanos",
    )
    if state_committed != committed_cost_nanos:
        raise AuditError("request log cost sum differs from durable cost state")
    if state.get("billable_request_count") != billable_count:
        raise AuditError("request log billable count differs from durable cost state")
    if state.get("failed_request_count") != released_failed_count:
        raise AuditError("request log released-failure count differs from durable cost state")
    if state.get("usage") != aggregate_usage:
        raise AuditError("request log usage sum differs from durable cost state")
    reservations = state.get("reservations")
    if not isinstance(reservations, dict):
        raise AuditError("cost-state reservations is not an object")
    reservation_sum = 0
    for reservation in reservations.values():
        if not isinstance(reservation, dict):
            raise AuditError("invalid in-flight reservation")
        reservation_sum += _integer(
            reservation.get("cost_nanos"),
            name="reservation cost_nanos",
        )
    for request_id in reservations:
        if request_id in seen and request_id not in retained_reservation_ids:
            raise AuditError(
                "completed request has a reservation without reservation_retained"
            )
    missing_retained = retained_reservation_ids.difference(reservations)
    if missing_retained:
        raise AuditError("retained request reservation is absent from durable state")
    if reservation_sum != state_reserved:
        raise AuditError("reservation sum differs from reserved_cost_nanos")
    if reservations and not allow_in_flight:
        raise AuditError("in-flight reservations remain; stop the gateway before audit")
    if state_committed + state_reserved > expected_max_nanos:
        raise AuditError("committed plus reserved cost exceeds max-cost gate")

    return {
        "schema": "openai-gpt55-flex-audit/v1",
        "status": "pass",
        "audited_at": _utc_now(),
        "result_root": str(root),
        "provider_model": PROVIDER_MODEL,
        "returned_alias": REQUESTED_MODEL,
        "service_tier": SERVICE_TIER,
        "request_count": len(entries),
        "success_count": success_count,
        "error_count": error_count,
        "billable_request_count": billable_count,
        "committed_cost_nanos": state_committed,
        "committed_cost_usd": _nanos_to_usd(state_committed),
        "reserved_cost_nanos": state_reserved,
        "reserved_cost_usd": _nanos_to_usd(state_reserved),
        "max_cost_nanos": expected_max_nanos,
        "max_cost_usd": _nanos_to_usd(expected_max_nanos),
        "usage": aggregate_usage,
        "in_flight": len(reservations),
    }


def _atomic_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline audit of OpenAI GPT-5.5 Flex gateway accounting",
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--max-cost-usd", required=True)
    parser.add_argument("--allow-in-flight", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = audit(
            result_root=args.result_root,
            max_cost_usd=args.max_cost_usd,
            allow_in_flight=args.allow_in_flight,
        )
    except AuditError as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, ensure_ascii=False))
        return 1
    if args.output is not None:
        _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
