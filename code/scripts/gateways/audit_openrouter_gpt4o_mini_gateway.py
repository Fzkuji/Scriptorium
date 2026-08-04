#!/usr/bin/env python3
"""Independent offline auditor for the OpenRouter GPT-4o-mini cost gateway.

This module deliberately does not import the gateway implementation.  It
recomputes reservations, token-derived charges, cumulative state, model
identity, and provider-``usage.cost`` commitment from constants frozen here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator


REQUESTED_MODEL = "openai/gpt-4o-mini"
PROVIDER_MODEL = "openai/gpt-4o-mini-2024-07-18"
UPSTREAM_URL = "https://openrouter.ai/api/v1/chat/completions"
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
FORBIDDEN_LOG_KEYS = {
    "authorization",
    "api_key",
    "api-key",
    "key",
    "messages",
    "message",
    "prompt",
    "content",
    "tools",
    "tool_choice",
}


class AuditError(RuntimeError):
    """The durable gateway evidence is incomplete or inconsistent."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def usd_to_nanos(value: object, *, allow_zero: bool = False) -> int:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise AuditError(f"invalid USD value: {value!r}") from exc
    if not amount.is_finite() or amount < 0 or (amount == 0 and not allow_zero):
        raise AuditError("USD value is not finite and valid")
    nanos = amount * NANODOLLARS_PER_USD
    if nanos != nanos.to_integral_value():
        raise AuditError("USD value has more than nine decimal places")
    return int(nanos)


def nanos_to_usd(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuditError("nanodollar value is invalid")
    rendered = format(Decimal(value) / NANODOLLARS_PER_USD, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuditError(f"{name} is not a non-negative integer")
    return value


def sha256(value: object, *, name: str, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AuditError(f"{name} is not a lowercase SHA-256")
    return value


def token_cost(*, prompt: int, cached: int, completion: int) -> int:
    if cached > prompt:
        raise AuditError("cached token count exceeds prompt token count")
    return (
        (prompt - cached) * UNCACHED_INPUT_NANOS_PER_TOKEN
        + cached * CACHED_INPUT_NANOS_PER_TOKEN
        + completion * OUTPUT_NANOS_PER_TOKEN
    )


def maximum_reservation(max_completion_tokens: object) -> int:
    maximum = integer(max_completion_tokens, name="max_completion_tokens")
    if not 1 <= maximum <= MAX_OUTPUT_TOKENS:
        raise AuditError("max_completion_tokens exceeds the frozen model limit")
    return (
        (CONTEXT_WINDOW_TOKENS - maximum) * UNCACHED_INPUT_NANOS_PER_TOKEN
        + maximum * OUTPUT_NANOS_PER_TOKEN
    )


def read_regular_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"not a regular JSON file: {path}")
    metadata = path.stat(follow_symlinks=False)
    if metadata.st_nlink != 1 or not stat.S_ISREG(metadata.st_mode):
        raise AuditError(f"JSON file is hardlinked or irregular: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot parse JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON root is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise AuditError("request log is not a regular file")
    metadata = path.stat(follow_symlinks=False)
    if metadata.st_nlink != 1:
        raise AuditError("request log is hardlinked")
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise AuditError("request log has an incomplete final line")
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        if not raw.strip():
            raise AuditError(f"request log line {line_number} is blank")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuditError(f"request log line {line_number} is invalid") from exc
        if not isinstance(value, dict):
            raise AuditError(f"request log line {line_number} is not an object")
        records.append(value)
    return records


def walk_keys(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower()
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def normalized_usage(value: object, *, line_number: int) -> dict[str, int]:
    if not isinstance(value, dict):
        raise AuditError(f"line {line_number} usage is absent")
    exact_keys = {
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "total_tokens",
    }
    if set(value) != exact_keys:
        raise AuditError(f"line {line_number} usage keys differ")
    result = {
        name: integer(value.get(name), name=f"line {line_number} usage.{name}")
        for name in exact_keys
    }
    if result["cached_tokens"] > result["prompt_tokens"]:
        raise AuditError(f"line {line_number} cached tokens exceed prompt tokens")
    if result["total_tokens"] != (
        result["prompt_tokens"] + result["completion_tokens"]
    ):
        raise AuditError(f"line {line_number} total token count differs")
    if result["total_tokens"] > CONTEXT_WINDOW_TOKENS:
        raise AuditError(f"line {line_number} exceeds the context window")
    return result


def audit(result_root: Path, *, max_cost_usd: str) -> dict[str, Any]:
    root = result_root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise AuditError("result root is not a regular directory")
    marker = read_regular_json(root / ROOT_MARKER_NAME)
    expected_marker = {
        "schema": ROOT_SCHEMA,
        "provider": "OpenRouter",
        "api": "chat_completions",
        "upstream_url": UPSTREAM_URL,
        "provider_model": PROVIDER_MODEL,
        "returned_alias": REQUESTED_MODEL,
        "cross_model_fallbacks": False,
        "billing": "openrouter_api",
        "routing_policy": "exact_model_no_fallback",
    }
    for name, expected in expected_marker.items():
        if marker.get(name) != expected:
            raise AuditError(f"result-root marker {name} differs")

    cap = usd_to_nanos(max_cost_usd)
    state = read_regular_json(root / STATE_NAME)
    expected_state = {
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
        "max_cost_nanos": cap,
        "max_cost_usd": nanos_to_usd(cap),
    }
    for name, expected in expected_state.items():
        if state.get(name) != expected:
            raise AuditError(f"cost state {name} differs")

    entries = read_jsonl(root / REQUEST_LOG_NAME)
    request_ids: set[str] = set()
    response_ids: set[str] = set()
    committed_sum = 0
    committed_usage = {
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
    }
    billable_count = 0
    exact_count = 0
    derived_count = 0
    retained: dict[str, dict[str, Any]] = {}
    status_counts: dict[str, int] = {}

    for line_number, entry in enumerate(entries, start=1):
        if entry.get("schema") != REQUEST_LOG_SCHEMA:
            raise AuditError(f"line {line_number} schema differs")
        leaked = FORBIDDEN_LOG_KEYS.intersection(walk_keys(entry))
        if leaked:
            raise AuditError(
                f"line {line_number} contains forbidden log keys: {sorted(leaked)}"
            )
        request_id = entry.get("request_id")
        if not isinstance(request_id, str) or not request_id.startswith("openrouter-"):
            raise AuditError(f"line {line_number} request id differs")
        if request_id in request_ids:
            raise AuditError(f"line {line_number} duplicates a request id")
        request_ids.add(request_id)
        sha256(entry.get("request_sha256"), name="request_sha256")
        sha256(
            entry.get("provider_request_sha256"),
            name="provider_request_sha256",
            nullable=True,
        )
        sha256(
            entry.get("response_sha256"),
            name="response_sha256",
            nullable=True,
        )
        if entry.get("provider_target_model") != PROVIDER_MODEL:
            raise AuditError(f"line {line_number} provider target differs")
        status = entry.get("status")
        if status not in {"success", "error"}:
            raise AuditError(f"line {line_number} status differs")
        status_counts[status] = status_counts.get(status, 0) + 1
        reservation = integer(
            entry.get("reservation_nanos"), name=f"line {line_number} reservation"
        )
        max_tokens = entry.get("max_completion_tokens")
        if reservation:
            if reservation != maximum_reservation(max_tokens):
                raise AuditError(f"line {line_number} reservation differs")
            if entry.get("provider_request_sha256") is None:
                raise AuditError(f"line {line_number} reserved without provider hash")
            transformations = entry.get("transformations")
            if (
                not isinstance(transformations, list)
                or "provider_allow_fallbacks_false_added" not in transformations
                or any(
                    item
                    not in {
                        "default_max_tokens_added",
                        "provider_allow_fallbacks_false_added",
                    }
                    for item in transformations
                )
            ):
                raise AuditError(
                    f"line {line_number} exact-model routing evidence differs"
                )

        billable = entry.get("billable")
        retained_flag = entry.get("reservation_retained")
        if not isinstance(billable, bool) or not isinstance(retained_flag, bool):
            raise AuditError(f"line {line_number} boolean accounting differs")
        if retained_flag:
            if status != "error" or not reservation or billable:
                raise AuditError(f"line {line_number} retained reservation differs")
            retained[request_id] = entry
            continue

        if billable:
            billable_count += 1
            usage = normalized_usage(entry.get("usage"), line_number=line_number)
            if usage["completion_tokens"] > integer(
                max_tokens, name=f"line {line_number} max_completion_tokens"
            ):
                raise AuditError(f"line {line_number} completion exceeds maximum")
            derived = token_cost(
                prompt=usage["prompt_tokens"],
                cached=usage["cached_tokens"],
                completion=usage["completion_tokens"],
            )
            if (
                entry.get("token_derived_cost_nanos") != derived
                or entry.get("token_derived_cost_usd") != nanos_to_usd(derived)
            ):
                raise AuditError(f"line {line_number} token-derived cost differs")
            provider_nanos = entry.get("provider_usage_cost_nanos")
            provider_usd = entry.get("provider_usage_cost_usd")
            source = entry.get("committed_cost_source")
            if provider_nanos is None:
                if provider_usd is not None or source != "token_derived":
                    raise AuditError(f"line {line_number} missing-cost source differs")
                expected_committed = derived
                derived_count += 1
            else:
                provider_nanos = integer(
                    provider_nanos, name=f"line {line_number} provider cost"
                )
                if (
                    provider_usd != nanos_to_usd(provider_nanos)
                    or source != "provider_usage_cost"
                ):
                    raise AuditError(f"line {line_number} provider usage.cost differs")
                expected_committed = provider_nanos
                exact_count += 1
            if (
                entry.get("committed_cost_nanos") != expected_committed
                or entry.get("committed_cost_usd")
                != nanos_to_usd(expected_committed)
                or expected_committed > reservation
            ):
                raise AuditError(f"line {line_number} committed cost differs")
            committed_sum += expected_committed
            for name in committed_usage:
                committed_usage[name] += usage[name]
            response_id = entry.get("response_id")
            if not isinstance(response_id, str) or not response_id.strip():
                raise AuditError(f"line {line_number} response id is absent")
            if response_id in response_ids:
                raise AuditError(f"line {line_number} response id is duplicated")
            response_ids.add(response_id)
            if entry.get("provider_actual_model") != PROVIDER_MODEL:
                # A model mismatch is still billable but must remain an error.
                if status != "error" or entry.get("error_code") != "provider_model_mismatch":
                    raise AuditError(f"line {line_number} provider model differs")
            elif status == "success":
                if (
                    entry.get("actual_model") != REQUESTED_MODEL
                    or entry.get("returned_alias") != REQUESTED_MODEL
                    or entry.get("http_status") != 200
                ):
                    raise AuditError(f"line {line_number} returned alias differs")
            continue

        if status == "success":
            raise AuditError(f"line {line_number} success is not billable")
        if any(
            entry.get(name) not in (None, 0, "0")
            for name in (
                "usage",
                "provider_usage_cost_nanos",
                "provider_usage_cost_usd",
                "token_derived_cost_nanos",
                "token_derived_cost_usd",
                "committed_cost_source",
                "committed_cost_nanos",
            )
        ):
            raise AuditError(f"line {line_number} non-billable cost fields differ")

    reservations = state.get("reservations")
    if not isinstance(reservations, dict):
        raise AuditError("cost-state reservations is not an object")
    if set(reservations) != set(retained):
        raise AuditError("cost-state retained reservations differ from request log")
    reservation_sum = 0
    for request_id, reservation in reservations.items():
        if not isinstance(reservation, dict):
            raise AuditError("cost-state reservation is not an object")
        entry = retained[request_id]
        expected = maximum_reservation(reservation.get("max_completion_tokens"))
        if (
            reservation.get("cost_nanos") != expected
            or expected != entry.get("reservation_nanos")
            or reservation.get("request_sha256")
            != entry.get("provider_request_sha256")
            or reservation.get("retained_reason") != entry.get("error_code")
        ):
            raise AuditError("cost-state retained reservation evidence differs")
        reservation_sum += expected

    exact_state = {
        "committed_cost_nanos": committed_sum,
        "reserved_cost_nanos": reservation_sum,
        "billable_request_count": billable_count,
        "retained_failure_count": len(retained),
        "provider_exact_cost_count": exact_count,
        "token_derived_cost_count": derived_count,
        "usage": committed_usage,
    }
    for name, expected in exact_state.items():
        if state.get(name) != expected:
            raise AuditError(f"cost-state {name} differs from request log")
    if committed_sum + reservation_sum > cap:
        raise AuditError("committed plus reserved cost exceeds the cap")
    if retained:
        raise AuditError(
            "residual uncertain reservations remain; manual billing resolution is required"
        )

    return {
        "schema": "openrouter-gpt4o-mini-gateway-audit/v1",
        "status": "passed",
        "audited_at": utc_now(),
        "result_root": str(root),
        "root_marker": {
            "path": str(root / ROOT_MARKER_NAME),
            "sha256": hashlib.sha256((root / ROOT_MARKER_NAME).read_bytes()).hexdigest(),
        },
        "state": {
            "path": str(root / STATE_NAME),
            "sha256": hashlib.sha256((root / STATE_NAME).read_bytes()).hexdigest(),
            "max_cost_usd": nanos_to_usd(cap),
            "committed_cost_usd": nanos_to_usd(committed_sum),
            "reserved_cost_usd": "0",
        },
        "request_log": {
            "path": str(root / REQUEST_LOG_NAME),
            "sha256": hashlib.sha256((root / REQUEST_LOG_NAME).read_bytes()).hexdigest(),
            "bytes": (root / REQUEST_LOG_NAME).stat().st_size,
            "entries": len(entries),
            "status_counts": status_counts,
            "billable_requests": billable_count,
            "response_ids": len(response_ids),
        },
        "cost_sources": {
            "provider_usage_cost": exact_count,
            "token_derived": derived_count,
        },
        "provider": "OpenRouter",
        "requested_model": REQUESTED_MODEL,
        "provider_model": PROVIDER_MODEL,
    }


def atomic_json(path: Path, value: object) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--max-cost-usd", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.result_root, max_cost_usd=args.max_cost_usd)
    except (AuditError, OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    if args.output:
        atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
