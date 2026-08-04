"""Reusable consumer evidence for the fixed OpenAI GPT-5.5 Flex gateway.

The provider gateway owns billing and its append-only request log.  Experiment
runners use this module to capture an idle start prefix and an idle end prefix.
Auditors can later prove that the bounded segment contains exactly the consumer
responses, even after subsequent experiments append to the same provider root.
No request content or credential is copied into consumer artifacts.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Iterable, Mapping
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from baselines.gateways import openai_gpt55_flex_gateway as gateway


WINDOW_SCHEMA = "openai-gpt55-flex-consumer-window/v1"
CONTRACT_SCHEMA = "openai-gpt55-flex-consumer-contract/v1"
PREFIX_SCHEMA = "openai-gpt55-flex-log-prefix/v1"
STATE_SNAPSHOT_SCHEMA = "openai-gpt55-flex-state-snapshot/v1"
CONSUMER_LOCK_NAME = ".flex_consumer_window.lock"
INVOCATION_SCHEMA = "openai-gpt55-flex-child-invocation/v1"
CHILD_READY_NAME = "child_proxy_ready.json"
CHILD_LOG_NAME = "child_proxy_requests.jsonl"
WINDOW_NAME = "gateway_window.json"
INVOCATION_NAME = "invocation.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
CHILD_PROXY = REPO_ROOT / "baselines" / "controlled_locomo" / "gpt55_run_proxy.py"


class EvidenceError(RuntimeError):
    """Raised when provider-root or consumer-window evidence is invalid."""


def _atomic_json(path: Path, value: object) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def acquire_consumer_lock(result_root: Path) -> IO[str]:
    """Acquire the process-wide serial-consumer lock without waiting."""
    root = result_root.expanduser().resolve()
    _validate_root_marker(root)
    path = root / CONSUMER_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise EvidenceError("Flex consumer lock is not one regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvidenceError(
                "another evidence-integrated consumer owns the Flex gateway root"
            ) from exc
        return os.fdopen(descriptor, "a+", encoding="utf-8")
    except Exception:
        os.close(descriptor)
        raise


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"{label} is not a regular file: {path}")
    metadata = path.stat(follow_symlinks=False)
    if metadata.st_nlink != 1:
        raise EvidenceError(f"{label} must not be hardlinked: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read {label}: {path}") from exc


def _read_regular_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular_bytes(path, label=label)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} is not a JSON object: {path}")
    return value, payload


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError(f"{label} must be a non-negative integer")
    return value


def _validate_root_marker(root: Path) -> tuple[dict[str, Any], bytes]:
    marker, payload = _read_regular_json(
        root / gateway.ROOT_MARKER_NAME,
        label="Flex root marker",
    )
    expected = {
        "schema": gateway.ROOT_SCHEMA,
        "provider": "openai_api",
        "api": "chat_completions",
        "upstream_origin": f"https://{gateway.UPSTREAM_HOST}",
        "upstream_path": gateway.UPSTREAM_PATH,
        "provider_model": gateway.PROVIDER_MODEL,
        "returned_alias": gateway.REQUESTED_MODEL,
        "service_tier": gateway.SERVICE_TIER,
        "billing": "api_flex",
    }
    for key, expected_value in expected.items():
        if marker.get(key) != expected_value:
            raise EvidenceError(f"Flex root marker {key} differs")
    return marker, payload


def _validate_state(value: Mapping[str, Any]) -> None:
    expected = {
        "schema": gateway.STATE_SCHEMA,
        "pricing_id": gateway.PRICING_ID,
        "provider_model": gateway.PROVIDER_MODEL,
        "returned_alias": gateway.REQUESTED_MODEL,
        "service_tier": gateway.SERVICE_TIER,
        "context_window_tokens": gateway.CONTEXT_WINDOW_TOKENS,
        "max_output_tokens": gateway.MAX_OUTPUT_TOKENS,
        "long_context_threshold": gateway.LONG_CONTEXT_THRESHOLD,
        "pricing_nanos_per_token": {
            "uncached_input": gateway.UNCACHED_INPUT_NANOS_PER_TOKEN,
            "cached_input": gateway.CACHED_INPUT_NANOS_PER_TOKEN,
            "output": gateway.OUTPUT_NANOS_PER_TOKEN,
            "long_uncached_input": gateway.LONG_UNCACHED_INPUT_NANOS_PER_TOKEN,
            "long_cached_input": gateway.LONG_CACHED_INPUT_NANOS_PER_TOKEN,
            "long_output": gateway.LONG_OUTPUT_NANOS_PER_TOKEN,
        },
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise EvidenceError(f"Flex cost state {key} differs")
    maximum = _nonnegative_int(value.get("max_cost_nanos"), label="max cost")
    if maximum <= 0 or value.get("max_cost_usd") != gateway.nanos_to_usd(maximum):
        raise EvidenceError("Flex cost state has no valid fixed cap")
    committed = _nonnegative_int(
        value.get("committed_cost_nanos"), label="committed cost"
    )
    reserved = _nonnegative_int(
        value.get("reserved_cost_nanos"), label="reserved cost"
    )
    if committed + reserved > maximum:
        raise EvidenceError("Flex cost state exceeds its fixed cap")
    reservations = value.get("reservations")
    if not isinstance(reservations, dict):
        raise EvidenceError("Flex cost-state reservations are invalid")
    reservation_sum = 0
    for record in reservations.values():
        if not isinstance(record, dict):
            raise EvidenceError("Flex reservation record is invalid")
        reservation_sum += _nonnegative_int(
            record.get("cost_nanos"), label="reservation cost"
        )
    if reserved != reservation_sum:
        raise EvidenceError("Flex reservation inventory differs")
    for key in ("billable_request_count", "failed_request_count"):
        _nonnegative_int(value.get(key), label=key)
    usage = value.get("usage")
    if not isinstance(usage, dict):
        raise EvidenceError("Flex cost-state usage is invalid")
    for key in (
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "reasoning_tokens",
    ):
        _nonnegative_int(usage.get(key), label=f"usage.{key}")


def _state_snapshot(path: Path, *, require_idle: bool) -> dict[str, Any]:
    value, payload = _read_regular_json(path, label="Flex cost state")
    _validate_state(value)
    reservations = value["reservations"]
    if require_idle and (reservations or value["reserved_cost_nanos"] != 0):
        raise EvidenceError("Flex gateway has in-flight or retained reservations")
    return {
        "schema": STATE_SNAPSHOT_SCHEMA,
        "file_sha256": sha256_bytes(payload),
        "canonical_sha256": sha256_bytes(_canonical_bytes(value)),
        "value": value,
    }


def _validate_state_snapshot(record: object, *, require_idle: bool) -> dict[str, Any]:
    if not isinstance(record, dict) or record.get("schema") != STATE_SNAPSHOT_SCHEMA:
        raise EvidenceError("recorded Flex state snapshot is invalid")
    value = record.get("value")
    if not isinstance(value, dict):
        raise EvidenceError("recorded Flex state value is invalid")
    _validate_state(value)
    if record.get("canonical_sha256") != sha256_bytes(_canonical_bytes(value)):
        raise EvidenceError("recorded Flex state canonical hash differs")
    if not isinstance(record.get("file_sha256"), str) or re.fullmatch(
        r"[0-9a-f]{64}", record["file_sha256"]
    ) is None:
        raise EvidenceError("recorded Flex state file hash is invalid")
    if require_idle and (value["reservations"] or value["reserved_cost_nanos"] != 0):
        raise EvidenceError("recorded Flex state is not idle")
    return value


def _prefix_record(payload: bytes) -> dict[str, Any]:
    if payload and not payload.endswith(b"\n"):
        raise EvidenceError("Flex request log has an incomplete final line")
    return {
        "schema": PREFIX_SCHEMA,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "entries": len([line for line in payload.splitlines() if line.strip()]),
    }


def _validate_prefix(record: object, payload: bytes, *, label: str) -> None:
    if not isinstance(record, dict) or record.get("schema") != PREFIX_SCHEMA:
        raise EvidenceError(f"{label} prefix record is invalid")
    size = _nonnegative_int(record.get("bytes"), label=f"{label} prefix bytes")
    entries = _nonnegative_int(
        record.get("entries"), label=f"{label} prefix entries"
    )
    if size > len(payload):
        raise EvidenceError(f"{label} prefix exceeds the current request log")
    prefix = payload[:size]
    if prefix and not prefix.endswith(b"\n"):
        raise EvidenceError(f"{label} prefix ends inside a request-log record")
    if record.get("sha256") != sha256_bytes(prefix):
        raise EvidenceError(f"{label} request-log prefix hash differs")
    if entries != len([line for line in prefix.splitlines() if line.strip()]):
        raise EvidenceError(f"{label} request-log prefix entry count differs")


def _validate_ready(value: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    expected = {
        "schema": "openai-gpt55-flex-ready/v1",
        "requested_model": gateway.REQUESTED_MODEL,
        "provider_model": gateway.PROVIDER_MODEL,
        "service_tier": gateway.SERVICE_TIER,
        "max_cost_usd": state["max_cost_usd"],
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise EvidenceError(f"Flex ready record {key} differs")
    base_url = value.get("base_url")
    health_url = value.get("health_url")
    if not isinstance(base_url, str) or re.fullmatch(
        r"http://127\.0\.0\.1:[1-9][0-9]{0,4}/v1", base_url
    ) is None:
        raise EvidenceError("Flex ready base URL is not a fixed loopback URL")
    port = int(base_url.split(":", 2)[2].split("/", 1)[0])
    if port >= 65536 or health_url != f"http://127.0.0.1:{port}/healthz":
        raise EvidenceError("Flex ready health URL differs")
    return base_url


def active_contract(result_root: Path) -> dict[str, Any]:
    """Validate an active provider root and return a serializable contract."""
    root = result_root.expanduser().resolve()
    if not root.is_dir():
        raise EvidenceError(f"Flex gateway result root is absent: {root}")
    _, marker_payload = _validate_root_marker(root)
    state_record = _state_snapshot(root / gateway.STATE_NAME, require_idle=True)
    state = state_record["value"]
    ready, ready_payload = _read_regular_json(
        root / "gateway_ready.json",
        label="Flex ready record",
    )
    base_url = _validate_ready(ready, state)
    _read_regular_bytes(root / gateway.REQUEST_LOG_NAME, label="Flex request log")
    return {
        "schema": CONTRACT_SCHEMA,
        "result_root": str(root),
        "base_url": base_url,
        "origin": base_url.removesuffix("/v1"),
        "provider": "openai_api",
        "provider_model": gateway.PROVIDER_MODEL,
        "returned_alias": gateway.REQUESTED_MODEL,
        "service_tier": gateway.SERVICE_TIER,
        "billing": "api_flex",
        "max_cost_nanos": state["max_cost_nanos"],
        "max_cost_usd": state["max_cost_usd"],
        "root_marker_sha256": sha256_bytes(marker_payload),
        "ready_sha256": sha256_bytes(ready_payload),
        "ready_canonical_sha256": sha256_bytes(_canonical_bytes(ready)),
        "ready": ready,
    }


def validate_recorded_contract(record: object) -> dict[str, Any]:
    """Validate a saved contract against persistent root and current prefix."""
    if not isinstance(record, dict) or record.get("schema") != CONTRACT_SCHEMA:
        raise EvidenceError("recorded Flex consumer contract is invalid")
    root_value = record.get("result_root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise EvidenceError("recorded Flex result root is invalid")
    root = Path(root_value).resolve()
    _, marker_payload = _validate_root_marker(root)
    expected = {
        "provider": "openai_api",
        "provider_model": gateway.PROVIDER_MODEL,
        "returned_alias": gateway.REQUESTED_MODEL,
        "service_tier": gateway.SERVICE_TIER,
        "billing": "api_flex",
        "root_marker_sha256": sha256_bytes(marker_payload),
    }
    for key, expected_value in expected.items():
        if record.get(key) != expected_value:
            raise EvidenceError(f"recorded Flex contract {key} differs")
    ready = record.get("ready")
    if not isinstance(ready, dict):
        raise EvidenceError("recorded Flex ready object is invalid")
    state, _ = _read_regular_json(root / gateway.STATE_NAME, label="Flex cost state")
    _validate_state(state)
    base_url = _validate_ready(ready, state)
    if record.get("base_url") != base_url:
        raise EvidenceError("recorded Flex base URL differs")
    if record.get("origin") != base_url.removesuffix("/v1"):
        raise EvidenceError("recorded Flex origin differs")
    if record.get("max_cost_nanos") != state["max_cost_nanos"]:
        raise EvidenceError("recorded Flex max-cost nanos differs")
    if record.get("max_cost_usd") != state["max_cost_usd"]:
        raise EvidenceError("recorded Flex max-cost USD differs")
    if record.get("ready_canonical_sha256") != sha256_bytes(
        _canonical_bytes(ready)
    ):
        raise EvidenceError("recorded Flex ready canonical hash differs")
    ready_path = root / "gateway_ready.json"
    if ready_path.is_file():
        current_ready, ready_payload = _read_regular_json(
            ready_path, label="Flex ready record"
        )
        _validate_ready(current_ready, state)
        if current_ready == ready and record.get("ready_sha256") != sha256_bytes(
            ready_payload
        ):
            raise EvidenceError("recorded Flex ready file hash differs")
    elif not isinstance(record.get("ready_sha256"), str) or re.fullmatch(
        r"[0-9a-f]{64}", str(record["ready_sha256"])
    ) is None:
        raise EvidenceError("recorded Flex ready file hash is invalid")
    return record


def capture_start(result_root: Path) -> dict[str, Any]:
    """Capture an idle provider-root prefix before a consumer starts."""
    contract = active_contract(result_root)
    root = Path(contract["result_root"])
    log_payload = _read_regular_bytes(
        root / gateway.REQUEST_LOG_NAME,
        label="Flex request log",
    )
    return {
        "schema": WINDOW_SCHEMA,
        "contract": contract,
        "request_log_start": _prefix_record(log_payload),
        "state_start": _state_snapshot(root / gateway.STATE_NAME, require_idle=True),
        "request_log_end": None,
        "state_end": None,
        "segment": None,
    }


def _parse_segment(payload: bytes) -> list[dict[str, Any]]:
    if payload and not payload.endswith(b"\n"):
        raise EvidenceError("Flex request-log segment is incomplete")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(
                f"Flex request-log segment line {line_number} is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise EvidenceError(
                f"Flex request-log segment line {line_number} is not an object"
            )
        records.append(value)
    return records


def _validate_success_entry(entry: Mapping[str, Any]) -> None:
    expected = {
        "schema": gateway.REQUEST_LOG_SCHEMA,
        "status": "success",
        "http_status": 200,
        "requested_model": gateway.REQUESTED_MODEL,
        "provider_target_model": gateway.PROVIDER_MODEL,
        "provider_target_service_tier": gateway.SERVICE_TIER,
        "provider_actual_model": gateway.PROVIDER_MODEL,
        "actual_model": gateway.REQUESTED_MODEL,
        "returned_alias": gateway.REQUESTED_MODEL,
        "service_tier": gateway.SERVICE_TIER,
        "billable": True,
        "reservation_retained": False,
    }
    for key, expected_value in expected.items():
        if entry.get(key) != expected_value:
            raise EvidenceError(f"Flex request entry {key} differs")
    for key in (
        "request_id",
        "response_id",
        "request_sha256",
        "provider_request_sha256",
    ):
        value = entry.get(key)
        if not isinstance(value, str) or not value:
            raise EvidenceError(f"Flex request entry {key} is invalid")
    usage = entry.get("usage")
    if not isinstance(usage, dict):
        raise EvidenceError("Flex request entry usage is invalid")
    prompt = _nonnegative_int(usage.get("prompt_tokens"), label="prompt tokens")
    cached = _nonnegative_int(usage.get("cached_tokens"), label="cached tokens")
    completion = _nonnegative_int(
        usage.get("completion_tokens"), label="completion tokens"
    )
    _nonnegative_int(usage.get("reasoning_tokens"), label="reasoning tokens")
    expected_cost = gateway.calculate_cost_nanos(
        prompt_tokens=prompt,
        cached_tokens=cached,
        completion_tokens=completion,
    )
    if (
        entry.get("cost_nanos") != expected_cost
        or entry.get("cost_usd") != gateway.nanos_to_usd(expected_cost)
    ):
        raise EvidenceError("Flex request entry cost differs from usage")


def _usage_sum(entries: Iterable[Mapping[str, Any]], key: str) -> int:
    return sum(int(entry["usage"][key]) for entry in entries)


def _validate_state_delta(
    start: Mapping[str, Any],
    end: Mapping[str, Any],
    entries: list[dict[str, Any]],
) -> None:
    if start["max_cost_nanos"] != end["max_cost_nanos"]:
        raise EvidenceError("Flex max-cost cap changed inside consumer window")
    expected = {
        "committed_cost_nanos": sum(int(entry["cost_nanos"]) for entry in entries),
        "billable_request_count": len(entries),
        "failed_request_count": 0,
    }
    for key, delta in expected.items():
        if int(end[key]) - int(start[key]) != delta:
            raise EvidenceError(f"Flex state delta {key} differs")
    for key in (
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "reasoning_tokens",
    ):
        if int(end["usage"][key]) - int(start["usage"][key]) != _usage_sum(
            entries, key
        ):
            raise EvidenceError(f"Flex state usage delta {key} differs")


def capture_end(start_record: Mapping[str, Any]) -> dict[str, Any]:
    """Close a consumer window and bind exact request/state deltas."""
    if start_record.get("schema") != WINDOW_SCHEMA:
        raise EvidenceError("Flex consumer start window is invalid")
    contract = validate_recorded_contract(start_record.get("contract"))
    root = Path(contract["result_root"])
    payload = _read_regular_bytes(
        root / gateway.REQUEST_LOG_NAME,
        label="Flex request log",
    )
    _validate_prefix(start_record.get("request_log_start"), payload, label="start")
    start_bytes = int(start_record["request_log_start"]["bytes"])
    end_prefix = _prefix_record(payload)
    segment_payload = payload[start_bytes:]
    entries = _parse_segment(segment_payload)
    for entry in entries:
        _validate_success_entry(entry)
    start_state = _validate_state_snapshot(
        start_record.get("state_start"), require_idle=True
    )
    end_state_record = _state_snapshot(
        root / gateway.STATE_NAME,
        require_idle=True,
    )
    _validate_state_delta(start_state, end_state_record["value"], entries)
    return {
        "schema": WINDOW_SCHEMA,
        "contract": contract,
        "request_log_start": start_record["request_log_start"],
        "state_start": start_record["state_start"],
        "request_log_end": end_prefix,
        "state_end": end_state_record,
        "segment": {
            "bytes": len(segment_payload),
            "sha256": sha256_bytes(segment_payload),
            "entries": len(entries),
            "request_ids": [entry["request_id"] for entry in entries],
            "response_ids": [entry["response_id"] for entry in entries],
        },
    }


def _consumer_index(records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        request_id = record.get("gateway_request_id")
        if not isinstance(request_id, str) or not request_id or request_id in result:
            raise EvidenceError("consumer gateway request ID is missing or duplicated")
        result[request_id] = record
    return result


def audit_window(
    window: object,
    *,
    consumer_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit a closed window and exact consumer-to-provider correspondence."""
    if not isinstance(window, dict) or window.get("schema") != WINDOW_SCHEMA:
        raise EvidenceError("closed Flex consumer window is invalid")
    contract = validate_recorded_contract(window.get("contract"))
    root = Path(contract["result_root"])
    payload = _read_regular_bytes(
        root / gateway.REQUEST_LOG_NAME,
        label="Flex request log",
    )
    _validate_prefix(window.get("request_log_start"), payload, label="start")
    _validate_prefix(window.get("request_log_end"), payload, label="end")
    start_bytes = int(window["request_log_start"]["bytes"])
    end_bytes = int(window["request_log_end"]["bytes"])
    if end_bytes < start_bytes:
        raise EvidenceError("Flex request-log prefixes are reversed")
    segment_payload = payload[start_bytes:end_bytes]
    segment_record = window.get("segment")
    if not isinstance(segment_record, dict):
        raise EvidenceError("Flex request-log segment record is invalid")
    if (
        segment_record.get("bytes") != len(segment_payload)
        or segment_record.get("sha256") != sha256_bytes(segment_payload)
    ):
        raise EvidenceError("Flex request-log segment hash differs")
    entries = _parse_segment(segment_payload)
    if segment_record.get("entries") != len(entries):
        raise EvidenceError("Flex request-log segment entry count differs")
    for entry in entries:
        _validate_success_entry(entry)
    if segment_record.get("request_ids") != [entry["request_id"] for entry in entries]:
        raise EvidenceError("Flex request-log segment request IDs differ")
    if segment_record.get("response_ids") != [entry["response_id"] for entry in entries]:
        raise EvidenceError("Flex request-log segment response IDs differ")
    start_state = _validate_state_snapshot(window.get("state_start"), require_idle=True)
    end_state = _validate_state_snapshot(window.get("state_end"), require_idle=True)
    _validate_state_delta(start_state, end_state, entries)
    current_state_record = _state_snapshot(
        root / gateway.STATE_NAME,
        require_idle=False,
    )
    current_state = current_state_record["value"]
    if current_state["max_cost_nanos"] != end_state["max_cost_nanos"]:
        raise EvidenceError("current Flex cost cap differs from the closed window")
    for key in (
        "committed_cost_nanos",
        "billable_request_count",
        "failed_request_count",
    ):
        if int(current_state[key]) < int(end_state[key]):
            raise EvidenceError(f"current Flex state regressed at {key}")
    for key in (
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "reasoning_tokens",
    ):
        if int(current_state["usage"][key]) < int(end_state["usage"][key]):
            raise EvidenceError(f"current Flex usage regressed at {key}")

    consumers = _consumer_index(consumer_records)
    providers = {str(entry["request_id"]): entry for entry in entries}
    if set(consumers) != set(providers):
        raise EvidenceError("consumer IDs do not equal the bounded Flex request segment")
    for request_id, consumer in consumers.items():
        provider = providers[request_id]
        comparisons = {
            "response_id": "response_id",
            "gateway_request_sha256": "request_sha256",
            "provider_request_sha256": "provider_request_sha256",
            "provider_actual_model": "provider_actual_model",
            "service_tier": "service_tier",
        }
        for consumer_key, provider_key in comparisons.items():
            if consumer.get(consumer_key) != provider.get(provider_key):
                raise EvidenceError(
                    f"consumer {consumer_key} differs for gateway request {request_id}"
                )
        if consumer.get("actual_model") != gateway.REQUESTED_MODEL:
            raise EvidenceError("consumer returned alias differs")
    return {
        "status": "passed",
        "provider_model": gateway.PROVIDER_MODEL,
        "returned_alias": gateway.REQUESTED_MODEL,
        "service_tier": gateway.SERVICE_TIER,
        "billing": "api_flex",
        "max_cost_usd": contract["max_cost_usd"],
        "requests": len(entries),
        "request_log_start_bytes": start_bytes,
        "request_log_end_bytes": end_bytes,
        "segment_sha256": segment_record["sha256"],
        "committed_cost_nanos": (
            int(end_state["committed_cost_nanos"])
            - int(start_state["committed_cost_nanos"])
        ),
    }


def load_child_proxy_records(
    path: Path,
    *,
    expected_run_id: str,
) -> list[dict[str, Any]]:
    """Read and strictly validate one exclusive child-proxy log.

    A formal invocation cannot contain a rejected or failed child request.  A
    failed provider request may retain a gateway reservation, so accepting it
    as completed experiment evidence would make the provider-cost delta and
    consumer inventory ambiguous.
    """
    payload = _read_regular_bytes(path, label="exclusive child-proxy log")
    if payload and not payload.endswith(b"\n"):
        raise EvidenceError("exclusive child-proxy log has an incomplete final line")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(
                f"exclusive child-proxy log line {line_number} is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise EvidenceError(
                f"exclusive child-proxy log line {line_number} is not an object"
            )
        if value.get("run_id") != expected_run_id:
            raise EvidenceError("exclusive child-proxy run ID differs")
        if value.get("status") != "success" or value.get("http_status") != 200:
            raise EvidenceError("exclusive child-proxy log contains a failed request")
        expected = {
            "actual_model": gateway.REQUESTED_MODEL,
            "provider_actual_model": gateway.PROVIDER_MODEL,
            "service_tier": gateway.SERVICE_TIER,
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise EvidenceError(f"exclusive child-proxy {key} differs")
        for key in (
            "gateway_request_id",
            "gateway_request_sha256",
            "provider_request_sha256",
            "response_id",
            "request_sha256",
        ):
            field = value.get(key)
            if not isinstance(field, str) or not field:
                raise EvidenceError(f"exclusive child-proxy {key} is invalid")
        records.append(value)
    return records


def audit_invocation(record: object) -> dict[str, Any]:
    """Audit saved child-proxy evidence against its bounded provider segment."""
    if not isinstance(record, dict) or record.get("schema") != INVOCATION_SCHEMA:
        raise EvidenceError("Flex child invocation record is invalid")
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise EvidenceError("Flex child invocation run ID is invalid")
    window_value = record.get("window")
    consumer_value = record.get("consumer_log")
    ready_value = record.get("child_ready")
    if (
        not isinstance(window_value, dict)
        or not isinstance(consumer_value, dict)
        or not isinstance(ready_value, dict)
    ):
        raise EvidenceError("Flex child invocation paths are invalid")
    window_path_value = window_value.get("path")
    consumer_path_value = consumer_value.get("path")
    ready_path_value = ready_value.get("path")
    if not isinstance(window_path_value, str) or not Path(window_path_value).is_absolute():
        raise EvidenceError("Flex child invocation window path is invalid")
    if not isinstance(consumer_path_value, str) or not Path(consumer_path_value).is_absolute():
        raise EvidenceError("Flex child invocation consumer path is invalid")
    if not isinstance(ready_path_value, str) or not Path(ready_path_value).is_absolute():
        raise EvidenceError("Flex child invocation ready path is invalid")
    window_path = Path(window_path_value).resolve()
    consumer_path = Path(consumer_path_value).resolve()
    ready_path = Path(ready_path_value).resolve()
    window, window_payload = _read_regular_json(
        window_path, label="Flex child invocation window"
    )
    consumer_payload = _read_regular_bytes(
        consumer_path, label="exclusive child-proxy log"
    )
    child_ready, ready_payload = _read_regular_json(
        ready_path, label="exclusive child-proxy ready record"
    )
    if window_value.get("sha256") != sha256_bytes(window_payload):
        raise EvidenceError("Flex child invocation window hash differs")
    if consumer_value.get("sha256") != sha256_bytes(consumer_payload):
        raise EvidenceError("Flex child invocation consumer-log hash differs")
    if ready_value.get("sha256") != sha256_bytes(ready_payload):
        raise EvidenceError("Flex child invocation ready hash differs")
    contract = window.get("contract")
    expected_ready = {
        "run_id": run_id,
        "upstream": contract.get("origin") if isinstance(contract, dict) else None,
    }
    for key, expected_value in expected_ready.items():
        if child_ready.get(key) != expected_value:
            raise EvidenceError(f"exclusive child-proxy saved ready {key} differs")
    saved_log = child_ready.get("log")
    if not isinstance(saved_log, str) or not Path(saved_log).is_absolute():
        raise EvidenceError("exclusive child-proxy saved ready log is invalid")
    if Path(saved_log).resolve() != consumer_path:
        saved_parts = Path(saved_log).parts
        actual_parts = consumer_path.parts
        try:
            saved_suffix = saved_parts[saved_parts.index("provider_evidence") :]
            actual_suffix = actual_parts[actual_parts.index("provider_evidence") :]
        except ValueError as exc:
            raise EvidenceError(
                "exclusive child-proxy saved ready log differs"
            ) from exc
        if saved_suffix != actual_suffix:
            raise EvidenceError("exclusive child-proxy saved ready log differs")
    child_base_url = child_ready.get("base_url")
    if (
        not isinstance(child_base_url, str)
        or re.fullmatch(r"http://127\.0\.0\.1:[1-9][0-9]{0,4}/v1", child_base_url)
        is None
        or child_base_url == (contract.get("base_url") if isinstance(contract, dict) else None)
        or child_base_url.endswith(":8199/v1")
        or record.get("child_base_url") != child_base_url
    ):
        raise EvidenceError("exclusive child-proxy saved base URL differs")
    consumers = load_child_proxy_records(consumer_path, expected_run_id=run_id)
    report = audit_window(window, consumer_records=consumers)
    if record.get("gateway_request_ids") != [
        item["gateway_request_id"] for item in consumers
    ]:
        raise EvidenceError("Flex child invocation gateway request IDs differ")
    if record.get("response_ids") != [item["response_id"] for item in consumers]:
        raise EvidenceError("Flex child invocation response IDs differ")
    if record.get("requests") != len(consumers):
        raise EvidenceError("Flex child invocation request count differs")
    expected = {
        "provider_model": gateway.PROVIDER_MODEL,
        "returned_alias": gateway.REQUESTED_MODEL,
        "service_tier": gateway.SERVICE_TIER,
        "billing": "api_flex",
    }
    for key, expected_value in expected.items():
        if record.get(key) != expected_value:
            raise EvidenceError(f"Flex child invocation {key} differs")
    return report


def _validate_child_ready(
    value: object,
    *,
    run_id: str,
    contract: Mapping[str, Any],
    process: subprocess.Popen[bytes],
    log_path: Path,
) -> str:
    if not isinstance(value, dict):
        raise EvidenceError("exclusive child-proxy ready record is invalid")
    expected = {
        "run_id": run_id,
        "pid": process.pid,
        "upstream": contract["origin"],
        "log": str(log_path),
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise EvidenceError(f"exclusive child-proxy ready {key} differs")
    base_url = value.get("base_url")
    if not isinstance(base_url, str) or re.fullmatch(
        r"http://127\.0\.0\.1:[1-9][0-9]{0,4}/v1", base_url
    ) is None:
        raise EvidenceError("exclusive child-proxy base URL is invalid")
    port = int(base_url.split(":", 2)[2].split("/", 1)[0])
    if port >= 65536 or port == 8199:
        raise EvidenceError("exclusive child-proxy port is invalid")
    if base_url == contract["base_url"]:
        raise EvidenceError("consumer must use the exclusive child proxy")
    return base_url


def _wait_for_child_ready(
    ready_path: Path,
    *,
    run_id: str,
    contract: Mapping[str, Any],
    process: subprocess.Popen[bytes],
    log_path: Path,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise EvidenceError(
                f"exclusive child proxy exited before ready: {process.returncode}"
            )
        if ready_path.is_file():
            try:
                value, _ = _read_regular_json(
                    ready_path, label="exclusive child-proxy ready record"
                )
                base_url = _validate_child_ready(
                    value,
                    run_id=run_id,
                    contract=contract,
                    process=process,
                    log_path=log_path,
                )
                health_url = f"{base_url.removesuffix('/v1')}/healthz"
                request = Request(health_url, method="GET")
                with build_opener(ProxyHandler({})).open(
                    request, timeout=2
                ) as response:  # noqa: S310
                    health = json.loads(response.read())
                if (
                    response.status != 200
                    or health.get("status") != "ok"
                    or health.get("run_id") != run_id
                    or health.get("upstream") != contract["origin"]
                ):
                    raise EvidenceError("exclusive child-proxy health differs")
                return value, base_url
            except (EvidenceError, OSError, URLError, json.JSONDecodeError) as exc:
                last_error = exc
        time.sleep(0.05)
    detail = f": {last_error}" if last_error is not None else ""
    raise EvidenceError(f"exclusive child proxy did not become ready{detail}")


@dataclass
class ChildProxyInvocation:
    """Live exclusive child proxy plus one serial provider evidence window."""

    run_id: str
    gateway_root: Path
    evidence_dir: Path
    contract: dict[str, Any]
    start_window: dict[str, Any]
    lock_handle: IO[str]
    process: subprocess.Popen[bytes]
    ready_path: Path
    log_path: Path
    base_url: str
    _finished: bool = False

    def finish(self) -> dict[str, Any]:
        if self._finished:
            raise EvidenceError("exclusive child invocation is already finished")
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            consumers = load_child_proxy_records(
                self.log_path, expected_run_id=self.run_id
            )
            window = capture_end(self.start_window)
            window_path = self.evidence_dir / WINDOW_NAME
            _atomic_json(window_path, window)
            window_payload = _read_regular_bytes(
                window_path, label="Flex child invocation window"
            )
            consumer_payload = _read_regular_bytes(
                self.log_path, label="exclusive child-proxy log"
            )
            ready_payload = _read_regular_bytes(
                self.ready_path, label="exclusive child-proxy ready record"
            )
            record: dict[str, Any] = {
                "schema": INVOCATION_SCHEMA,
                "run_id": self.run_id,
                "gateway_root": str(self.gateway_root),
                "provider_model": gateway.PROVIDER_MODEL,
                "returned_alias": gateway.REQUESTED_MODEL,
                "service_tier": gateway.SERVICE_TIER,
                "billing": "api_flex",
                "child_base_url": self.base_url,
                "requests": len(consumers),
                "gateway_request_ids": [
                    item["gateway_request_id"] for item in consumers
                ],
                "response_ids": [item["response_id"] for item in consumers],
                "window": {
                    "path": str(window_path),
                    "sha256": sha256_bytes(window_payload),
                },
                "consumer_log": {
                    "path": str(self.log_path),
                    "sha256": sha256_bytes(consumer_payload),
                },
                "child_ready": {
                    "path": str(self.ready_path),
                    "sha256": sha256_bytes(ready_payload),
                },
            }
            report = audit_window(window, consumer_records=consumers)
            record["audit"] = report
            invocation_path = self.evidence_dir / INVOCATION_NAME
            _atomic_json(invocation_path, record)
            record["record_path"] = str(invocation_path)
            record["record_sha256"] = sha256_bytes(
                _read_regular_bytes(
                    invocation_path, label="Flex child invocation record"
                )
            )
            return record
        finally:
            self._finished = True
            self.lock_handle.close()

    def abort(self) -> None:
        if self._finished:
            return
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        finally:
            self._finished = True
            self.lock_handle.close()


def begin_child_invocation(
    gateway_root: Path,
    evidence_dir: Path,
    *,
    run_id: str | None = None,
    timeout_seconds: float = 15,
) -> ChildProxyInvocation:
    """Start one fixed child proxy from an active ready-marker contract."""
    root = gateway_root.expanduser().resolve()
    identifier = run_id or f"native-{uuid.uuid4().hex}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", identifier):
        raise EvidenceError("exclusive child-proxy run ID is invalid")
    directory = evidence_dir.expanduser().resolve()
    if directory.exists():
        raise EvidenceError(f"provider evidence directory already exists: {directory}")
    directory.mkdir(parents=True, mode=0o700)
    lock_handle = acquire_consumer_lock(root)
    process: subprocess.Popen[bytes] | None = None
    try:
        start = capture_start(root)
        contract = start["contract"]
        ready_path = directory / CHILD_READY_NAME
        log_path = directory / CHILD_LOG_NAME
        process = subprocess.Popen(
            [
                sys.executable,
                str(CHILD_PROXY),
                "--port",
                "0",
                "--upstream",
                str(contract["origin"]),
                "--log",
                str(log_path),
                "--ready",
                str(ready_path),
                "--run-id",
                identifier,
            ],
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _, base_url = _wait_for_child_ready(
            ready_path,
            run_id=identifier,
            contract=contract,
            process=process,
            log_path=log_path,
            timeout_seconds=timeout_seconds,
        )
        return ChildProxyInvocation(
            run_id=identifier,
            gateway_root=root,
            evidence_dir=directory,
            contract=contract,
            start_window=start,
            lock_handle=lock_handle,
            process=process,
            ready_path=ready_path,
            log_path=log_path,
            base_url=base_url,
        )
    except Exception:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        lock_handle.close()
        raise
