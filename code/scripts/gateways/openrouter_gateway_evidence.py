"""Consumer-side binding for the marked OpenRouter GPT-4o-mini gateway.

This module reads only local evidence.  It does not import the gateway, access
credentials, or make HTTP requests.  Formal consumers bind a result artifact
to an immutable root marker and to a complete-line request-log interval.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


REQUESTED_MODEL = "openai/gpt-4o-mini"
PROVIDER_MODEL = "openai/gpt-4o-mini-2024-07-18"
ROOT_SCHEMA = "openrouter-gpt4o-mini-result-root/v1"
STATE_SCHEMA = "openrouter-gpt4o-mini-cost-state/v1"
REQUEST_LOG_SCHEMA = "openrouter-gpt4o-mini-request/v1"
ROOT_MARKER_NAME = "openrouter_gpt4o_mini_root.json"
STATE_NAME = "openrouter_cost_state.json"
REQUEST_LOG_NAME = "openrouter_requests.jsonl"
READY_NAME = "gateway_ready.json"
BINDING_SCHEMA = "openrouter-gpt4o-mini-consumer-binding/v1"


class GatewayEvidenceError(RuntimeError):
    """The local gateway identity or frozen log prefix is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GatewayEvidenceError(f"missing regular JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GatewayEvidenceError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prefix(path: Path, byte_count: int | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GatewayEvidenceError("OpenRouter gateway request log is absent")
    payload = path.read_bytes()
    count = len(payload) if byte_count is None else byte_count
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 0 <= count <= len(payload)
        or (count and payload[count - 1 : count] != b"\n")
    ):
        raise GatewayEvidenceError("OpenRouter gateway log prefix is invalid")
    return {
        "bytes": count,
        "sha256": hashlib.sha256(payload[:count]).hexdigest(),
    }


def _canonical_base_url(value: str) -> str:
    normalized = value.rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.path != "/v1"
        or parsed.query
        or parsed.fragment
    ):
        raise GatewayEvidenceError(
            "formal OpenRouter traffic requires a marked loopback /v1 gateway"
        )
    return normalized


def _validate_root(
    root: Path, base_url: str, *, require_ready: bool
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise GatewayEvidenceError("OpenRouter gateway result root is invalid")
    marker_path = root / ROOT_MARKER_NAME
    state_path = root / STATE_NAME
    log_path = root / REQUEST_LOG_NAME
    ready_path = root / READY_NAME
    marker = _read_json(marker_path)
    state = _read_json(state_path)
    ready = _read_json(ready_path) if ready_path.exists() else None
    expected_marker = {
        "schema": ROOT_SCHEMA,
        "provider": "OpenRouter",
        "api": "chat_completions",
        "upstream_url": "https://openrouter.ai/api/v1/chat/completions",
        "provider_model": PROVIDER_MODEL,
        "returned_alias": REQUESTED_MODEL,
        "cross_model_fallbacks": False,
        "billing": "openrouter_api",
        "routing_policy": "exact_model_no_fallback",
    }
    for name, expected in expected_marker.items():
        if marker.get(name) != expected:
            raise GatewayEvidenceError(f"OpenRouter marker {name} differs")
    if (
        state.get("schema") != STATE_SCHEMA
        or state.get("provider_model") != PROVIDER_MODEL
        or state.get("returned_alias") != REQUESTED_MODEL
        or state.get("provider") != "OpenRouter"
        or state.get("cross_model_fallbacks") is not False
    ):
        raise GatewayEvidenceError("OpenRouter gateway state identity differs")
    canonical = _canonical_base_url(base_url)
    if require_ready and ready is None:
        raise GatewayEvidenceError("OpenRouter gateway ready marker is absent")
    if ready is not None:
        if (
            ready.get("schema") != "openrouter-gpt4o-mini-ready/v1"
            or ready.get("base_url") != canonical
            or Path(str(ready.get("result_root", ""))).resolve() != root
            or Path(str(ready.get("root_marker", ""))).resolve() != marker_path
            or Path(str(ready.get("state", ""))).resolve() != state_path
            or Path(str(ready.get("request_log", ""))).resolve() != log_path
            or ready.get("requested_model") != REQUESTED_MODEL
            or ready.get("provider_model") != PROVIDER_MODEL
            or ready.get("cross_model_fallbacks") is not False
            or ready.get("max_cost_usd") != state.get("max_cost_usd")
        ):
            raise GatewayEvidenceError("OpenRouter gateway ready marker differs")
    _prefix(log_path)
    return {
        "root": root,
        "base_url": canonical,
        "marker_path": marker_path,
        "state_path": state_path,
        "log_path": log_path,
        "ready_path": ready_path,
        "state": state,
    }


def capture_binding(result_root: Path, *, base_url: str) -> dict[str, Any]:
    evidence = _validate_root(result_root, base_url, require_ready=True)
    return {
        "schema": BINDING_SCHEMA,
        "provider": "OpenRouter",
        "requested_model": REQUESTED_MODEL,
        "provider_model": PROVIDER_MODEL,
        "result_root": str(evidence["root"]),
        "base_url": evidence["base_url"],
        "root_marker": {
            "path": str(evidence["marker_path"]),
            "sha256": _sha256(evidence["marker_path"]),
        },
        "state_path": str(evidence["state_path"]),
        "request_log_path": str(evidence["log_path"]),
        "max_cost_usd": evidence["state"]["max_cost_usd"],
        "start_prefix": _prefix(evidence["log_path"]),
    }


def validate_binding(
    binding: Mapping[str, Any], *, result_root: Path, base_url: str
) -> dict[str, Any]:
    evidence = _validate_root(result_root, base_url, require_ready=False)
    exact = {
        "schema": BINDING_SCHEMA,
        "provider": "OpenRouter",
        "requested_model": REQUESTED_MODEL,
        "provider_model": PROVIDER_MODEL,
        "result_root": str(evidence["root"]),
        "base_url": evidence["base_url"],
        "state_path": str(evidence["state_path"]),
        "request_log_path": str(evidence["log_path"]),
        "max_cost_usd": evidence["state"]["max_cost_usd"],
    }
    for name, expected in exact.items():
        if binding.get(name) != expected:
            raise GatewayEvidenceError(f"OpenRouter binding {name} differs")
    marker = binding.get("root_marker")
    if not isinstance(marker, dict) or marker != {
        "path": str(evidence["marker_path"]),
        "sha256": _sha256(evidence["marker_path"]),
    }:
        raise GatewayEvidenceError("OpenRouter binding marker differs")
    start = binding.get("start_prefix")
    if not isinstance(start, dict):
        raise GatewayEvidenceError("OpenRouter binding start prefix is absent")
    if _prefix(evidence["log_path"], start.get("bytes")) != start:
        raise GatewayEvidenceError("OpenRouter binding start prefix changed")
    return dict(binding)


def finalize_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(binding.get("result_root", ""))).expanduser().resolve()
    validated = validate_binding(
        binding, result_root=root, base_url=str(binding.get("base_url", ""))
    )
    state_path = Path(validated["state_path"])
    log_path = Path(validated["request_log_path"])
    state = _read_json(state_path)
    return {
        **validated,
        "end_prefix": _prefix(log_path),
        "state_at_end": {
            "sha256": _sha256(state_path),
            "committed_cost_nanos": state.get("committed_cost_nanos"),
            "reserved_cost_nanos": state.get("reserved_cost_nanos"),
            "billable_request_count": state.get("billable_request_count"),
        },
    }


def log_interval(binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(binding.get("end_prefix"), dict):
        raise GatewayEvidenceError("OpenRouter binding end prefix is absent")
    root = Path(str(binding.get("result_root", ""))).expanduser().resolve()
    validated = validate_binding(
        binding, result_root=root, base_url=str(binding.get("base_url", ""))
    )
    log_path = Path(validated["request_log_path"])
    start = binding["start_prefix"]["bytes"]
    end = binding["end_prefix"]["bytes"]
    if _prefix(log_path, end) != binding["end_prefix"] or end < start:
        raise GatewayEvidenceError("OpenRouter binding end prefix changed")
    payload = log_path.read_bytes()[start:end]
    records: list[dict[str, Any]] = []
    for line in payload.splitlines():
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("schema") != REQUEST_LOG_SCHEMA:
            raise GatewayEvidenceError("OpenRouter gateway interval record differs")
        records.append(value)
    return records


def verify_response_ids(
    binding: Mapping[str, Any], response_ids: Iterable[str]
) -> dict[str, Any]:
    expected = list(response_ids)
    if len(expected) != len(set(expected)):
        raise GatewayEvidenceError("consumer response IDs are duplicated")
    interval = log_interval(binding)
    successes: dict[str, dict[str, Any]] = {}
    for entry in interval:
        response_id = entry.get("response_id")
        if entry.get("status") == "success" and isinstance(response_id, str):
            if response_id in successes:
                raise GatewayEvidenceError("gateway interval response ID is duplicated")
            successes[response_id] = entry
    missing = set(expected) - set(successes)
    if missing:
        raise GatewayEvidenceError(
            f"consumer response IDs are absent from gateway interval: {len(missing)}"
        )
    for response_id in expected:
        entry = successes[response_id]
        if (
            entry.get("requested_model") != REQUESTED_MODEL
            or entry.get("provider_actual_model") != PROVIDER_MODEL
            or entry.get("returned_alias") != REQUESTED_MODEL
            or entry.get("billable") is not True
        ):
            raise GatewayEvidenceError("gateway response identity differs")
    return {
        "status": "passed",
        "bound_response_ids": len(expected),
        "interval_entries": len(interval),
        "interval_successes": len(successes),
    }
