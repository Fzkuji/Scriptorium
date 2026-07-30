"""Independent reconstruction audit for visible-token budget traces."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.visible_token_budget import (
    SCHEMA_VERSION,
    TRUNCATION_ALGORITHM,
    TokenCounter,
    canonical_json,
    snapshot_memory_path,
)


ZERO_HASH = "0" * 64


class DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_strict(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number: {value}")
        ),
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _record_hash(record_without_hash: Mapping[str, Any]) -> str:
    return _sha256(canonical_json(record_without_hash).encode("utf-8"))


def _text_measurements(text: str, tokenizer: TokenCounter) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    return {
        "sha256": _sha256(encoded),
        "utf8_bytes": len(encoded),
        "tokens": tokenizer.count(text),
    }


def _reconstruct_truncation(
    raw_text: str,
    limit: int,
    tokenizer: TokenCounter,
) -> str:
    """Reimplement the recorded algorithm without trusting gate output."""

    token_ids = tokenizer.encode(raw_text)
    if len(token_ids) <= limit:
        return raw_text
    prefix_ids = token_ids[:limit]
    while True:
        try:
            candidate = tokenizer.decode(prefix_ids)
            is_valid = (
                (bool(candidate) or not prefix_ids)
                and raw_text.startswith(candidate)
                and tokenizer.count(candidate) <= limit
            )
        except Exception:  # noqa: BLE001 - invalid tokenizer boundary
            is_valid = False
            candidate = ""
        if is_valid:
            return candidate
        if not prefix_ids:
            return ""
        prefix_ids.pop()


def _directory_root(entries: list[dict[str, Any]]) -> str:
    normalized = [
        {
            "path": entry.get("path"),
            "sha256": entry.get("sha256"),
            "byte_count": entry.get("byte_count"),
        }
        for entry in entries
    ]
    return _sha256(canonical_json(normalized).encode("utf-8"))


def _validate_memory_descriptor(
    descriptor: Any,
    *,
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(descriptor, dict):
        errors.append(f"{label}: descriptor is not an object")
        return
    if not _valid_sha256(descriptor.get("sha256")):
        errors.append(f"{label}: invalid sha256")
    if not isinstance(descriptor.get("byte_count"), int) or descriptor["byte_count"] < 0:
        errors.append(f"{label}: invalid byte_count")
    kind = descriptor.get("kind")
    if kind not in {"bytes", "file", "directory"}:
        errors.append(f"{label}: unsupported kind {kind!r}")
        return
    if kind != "directory":
        return

    entries = descriptor.get("entries")
    if not isinstance(entries, list):
        errors.append(f"{label}: directory entries are missing")
        return
    paths: list[str] = []
    valid_entries = True
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"{label}: entry {index} is not an object")
            valid_entries = False
            continue
        path = entry.get("path")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or "\\" in path
        ):
            errors.append(f"{label}: unsafe entry path at index {index}")
            valid_entries = False
        else:
            paths.append(path)
        if not _valid_sha256(entry.get("sha256")):
            errors.append(f"{label}: invalid entry hash at index {index}")
            valid_entries = False
        if not isinstance(entry.get("byte_count"), int) or entry["byte_count"] < 0:
            errors.append(f"{label}: invalid entry size at index {index}")
            valid_entries = False
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        errors.append(f"{label}: directory entries are not uniquely sorted")
        valid_entries = False
    if descriptor.get("file_count") != len(entries):
        errors.append(f"{label}: file_count mismatch")
    total_bytes = sum(
        entry.get("byte_count", 0)
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("byte_count"), int)
    )
    if descriptor.get("byte_count") != total_bytes:
        errors.append(f"{label}: directory byte_count mismatch")
    if valid_entries and descriptor.get("sha256") != _directory_root(entries):
        errors.append(f"{label}: directory root hash mismatch")


def _compare_live_memory(
    descriptor: Mapping[str, Any],
    live_path: str | os.PathLike[str] | None,
    *,
    label: str,
    errors: list[str],
) -> None:
    if live_path is None:
        return
    try:
        observed = snapshot_memory_path(live_path).descriptor
    except Exception as exc:  # noqa: BLE001 - audit must return all failures
        errors.append(f"{label}: could not hash live path: {exc}")
        return
    for field in ("kind", "sha256", "byte_count"):
        if descriptor.get(field) != observed.get(field):
            errors.append(
                f"{label}: live {field} mismatch: "
                f"recorded={descriptor.get(field)!r}, observed={observed.get(field)!r}"
            )


def _check_equal(
    record: Mapping[str, Any],
    field: str,
    expected: Any,
    *,
    context: str,
    errors: list[str],
) -> None:
    if record.get(field) != expected:
        errors.append(
            f"{context}: {field} mismatch: "
            f"recorded={record.get(field)!r}, reconstructed={expected!r}"
        )


def audit_visible_token_trace(
    trace_path: str | os.PathLike[str],
    *,
    manifest_path: str | os.PathLike[str] | None = None,
    memory_before_path: str | os.PathLike[str] | None = None,
    memory_after_path: str | os.PathLike[str] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Audit a trace by reconstructing all content and budget measurements."""

    trace = Path(trace_path)
    manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else trace.with_suffix(trace.suffix + ".manifest.json")
    )
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit_status": "fail",
        "trace_path": str(trace.resolve()) if trace.exists() else str(trace),
        "manifest_path": str(manifest.resolve()) if manifest.exists() else str(manifest),
        "errors": errors,
        "warnings": warnings,
    }

    if not trace.is_file():
        errors.append("trace file is missing")
        return report
    trace_bytes = trace.read_bytes()
    trace_sha256 = _sha256(trace_bytes)
    report["trace_sha256"] = trace_sha256

    manifest_data: dict[str, Any] | None = None
    if not manifest.is_file():
        errors.append("complete manifest is missing")
    else:
        try:
            loaded = _load_json_strict(manifest.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("manifest root is not an object")
            manifest_data = loaded
        except Exception as exc:  # noqa: BLE001
            errors.append(f"manifest parse failure: {exc}")
    if manifest_data is not None:
        if manifest_data.get("trace_sha256") != trace_sha256:
            errors.append("manifest trace_sha256 does not match trace bytes")
        if manifest_data.get("trace_file") != trace.name:
            errors.append("manifest trace_file does not match audited trace")
        if require_complete and manifest_data.get("status") != "complete":
            errors.append("manifest status is not complete")

    records: list[dict[str, Any]] = []
    try:
        text = trace_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        errors.append(f"trace is not UTF-8: {exc}")
        return report
    if text and not text.endswith("\n"):
        errors.append("trace does not end with a complete newline")
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            value = _load_json_strict(line)
            if not isinstance(value, dict):
                raise ValueError("record is not an object")
            records.append(value)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"line {line_number}: parse failure: {exc}")
    report["record_count"] = len(records)
    if not records:
        errors.append("trace has no records")
        return report

    previous_hash = ZERO_HASH
    for index, record in enumerate(records):
        context = f"record {index}"
        if record.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{context}: schema_version mismatch")
        if record.get("sequence") != index:
            errors.append(f"{context}: non-contiguous sequence")
        if record.get("previous_record_hash") != previous_hash:
            errors.append(f"{context}: previous_record_hash mismatch")
        actual_hash = record.get("record_hash")
        record_without_hash = dict(record)
        record_without_hash.pop("record_hash", None)
        reconstructed_hash = _record_hash(record_without_hash)
        if actual_hash != reconstructed_hash:
            errors.append(f"{context}: record_hash mismatch")
        previous_hash = str(actual_hash)

    header = records[0]
    if header.get("record_type") != "header":
        errors.append("record 0 is not a header")
        return report
    run_id = header.get("run_id")
    config = header.get("config")
    if not isinstance(config, dict):
        errors.append("header config is not an object")
        return report
    budget = config.get("configured_budget_tokens")
    if not isinstance(budget, int) or budget < 0:
        errors.append("configured budget is invalid")
        return report
    overflow_policy = config.get("overflow_policy")
    if overflow_policy not in {"truncate", "reject"}:
        errors.append("overflow policy is invalid")
        return report
    if config.get("provider_exact") is not False:
        errors.append("header must declare provider_exact=false")
    if config.get("truncation_algorithm") != TRUNCATION_ALGORITHM:
        errors.append("truncation algorithm identity mismatch")
    if config.get("overflow_ends_retrieval") is not True:
        errors.append("overflow_ends_retrieval must be true")
    if config.get("finalizer_required") is not True:
        errors.append("finalizer_required must be true")

    tokenizer_identity = config.get("tokenizer")
    if not isinstance(tokenizer_identity, dict):
        errors.append("tokenizer identity is missing")
        return report
    if tokenizer_identity.get("provider_exact") is not False:
        errors.append("tokenizer identity must declare provider_exact=false")
    try:
        tokenizer = TokenCounter.from_identity(tokenizer_identity)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"tokenizer reconstruction failed: {exc}")
        return report

    memory_before = header.get("memory_before")
    _validate_memory_descriptor(memory_before, label="memory_before", errors=errors)
    if isinstance(memory_before, dict):
        _compare_live_memory(
            memory_before,
            memory_before_path,
            label="memory_before",
            errors=errors,
        )

    cumulative = 0
    source_cumulative = 0
    exhausted = budget == 0
    exhaustion_sequence: int | None = 0 if exhausted else None
    event_ids: set[str] = set()
    finalizer: dict[str, Any] | None = None
    delivery_count = 0

    for index, record in enumerate(records[1:], start=1):
        context = f"record {index}"
        if record.get("run_id") != run_id:
            errors.append(f"{context}: run_id mismatch")
        record_type = record.get("record_type")
        if record_type == "finalizer":
            if finalizer is not None:
                errors.append(f"{context}: multiple finalizers")
            finalizer = record
            if index != len(records) - 1:
                errors.append(f"{context}: finalizer is not the last record")
            continue
        if record_type != "delivery":
            errors.append(f"{context}: unsupported record_type {record_type!r}")
            continue
        if finalizer is not None:
            errors.append(f"{context}: delivery occurs after finalizer")
        delivery_count += 1

        event_id = record.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in event_ids:
            errors.append(f"{context}: event_id is invalid or duplicated")
        else:
            event_ids.add(event_id)
        kind = record.get("kind")
        if kind not in {"tool_result", "source_resolution"}:
            errors.append(f"{context}: invalid delivery kind")

        raw = record.get("raw")
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
            errors.append(f"{context}: raw text record is invalid")
            continue
        raw_text = raw["text"]
        raw_measurements = _text_measurements(raw_text, tokenizer)
        for field, expected in raw_measurements.items():
            _check_equal(raw, field, expected, context=f"{context}.raw", errors=errors)
        raw_tokens = raw_measurements["tokens"]
        budget_before = budget - cumulative

        if exhausted:
            expected_text = None
            expected_decision = "rejected_budget_exhausted"
            expected_exhausted = True
        elif raw_tokens <= budget_before:
            expected_text = raw_text
            expected_decision = "delivered"
            expected_exhausted = raw_tokens == budget_before
        elif overflow_policy == "reject":
            expected_text = None
            expected_decision = "rejected_overflow"
            expected_exhausted = True
        else:
            try:
                candidate = _reconstruct_truncation(raw_text, budget_before, tokenizer)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{context}: truncation reconstruction failed: {exc}")
                candidate = ""
            expected_text = candidate or None
            expected_decision = "truncated" if candidate else "rejected_truncation_empty"
            expected_exhausted = True

        delivered = record.get("delivered")
        if expected_text is None:
            if delivered is not None:
                errors.append(f"{context}: rejected content has a delivered record")
            delivered_tokens = 0
        elif not isinstance(delivered, dict) or not isinstance(delivered.get("text"), str):
            errors.append(f"{context}: expected delivered text is missing")
            delivered_tokens = tokenizer.count(expected_text)
        else:
            if delivered["text"] != expected_text:
                errors.append(f"{context}: delivered text differs from reconstructed result")
            delivered_measurements = _text_measurements(delivered["text"], tokenizer)
            for field, expected in delivered_measurements.items():
                _check_equal(
                    delivered,
                    field,
                    expected,
                    context=f"{context}.delivered",
                    errors=errors,
                )
            delivered_tokens = delivered_measurements["tokens"]

        if delivered_tokens > budget_before:
            errors.append(f"{context}: delivered tokens exceed remaining budget")
        cumulative += delivered_tokens
        source_tokens = delivered_tokens if kind == "source_resolution" else 0
        source_cumulative += source_tokens
        exhausted = expected_exhausted
        if exhausted and exhaustion_sequence is None:
            exhaustion_sequence = index

        expected_fields = {
            "decision": expected_decision,
            "budget_before_tokens": budget_before,
            "budget_after_tokens": budget - cumulative,
            "cumulative_visible_tokens": cumulative,
            "source_resolution_tokens": source_tokens,
            "cumulative_source_resolution_tokens": source_cumulative,
            "exhausted": exhausted,
            "finalizer_required": True,
        }
        for field, expected in expected_fields.items():
            _check_equal(record, field, expected, context=context, errors=errors)

    if finalizer is None:
        errors.append("required finalizer is missing")
    else:
        memory_after = finalizer.get("memory_after")
        _validate_memory_descriptor(memory_after, label="memory_after", errors=errors)
        if isinstance(memory_after, dict):
            _compare_live_memory(
                memory_after,
                memory_after_path,
                label="memory_after",
                errors=errors,
            )
        before_hash = memory_before.get("sha256") if isinstance(memory_before, dict) else None
        after_hash = memory_after.get("sha256") if isinstance(memory_after, dict) else None
        _check_equal(
            finalizer,
            "memory_before_sha256",
            before_hash,
            context="finalizer",
            errors=errors,
        )
        _check_equal(
            finalizer,
            "memory_changed",
            before_hash != after_hash,
            context="finalizer",
            errors=errors,
        )
        if not isinstance(finalizer.get("actual_model_usage"), dict):
            errors.append("finalizer: actual_model_usage is not an object")
        expected_summary = {
            "configured_budget_tokens": budget,
            "cumulative_visible_tokens": cumulative,
            "cumulative_source_resolution_tokens": source_cumulative,
            "remaining_tokens": budget - cumulative,
            "exhausted": exhausted,
            "exhaustion_sequence": exhaustion_sequence,
            "delivery_record_count": delivery_count,
        }
        _check_equal(
            finalizer,
            "summary",
            expected_summary,
            context="finalizer",
            errors=errors,
        )

    if manifest_data is not None:
        expected_manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "record_count": len(records),
            "final_record_hash": records[-1].get("record_hash"),
            "tokenizer": tokenizer_identity,
            "configured_budget_tokens": budget,
            "memory_before_sha256": (
                memory_before.get("sha256") if isinstance(memory_before, dict) else None
            ),
            "memory_after_sha256": (
                finalizer.get("memory_after", {}).get("sha256")
                if isinstance(finalizer, dict)
                and isinstance(finalizer.get("memory_after"), dict)
                else None
            ),
            "summary": finalizer.get("summary") if isinstance(finalizer, dict) else None,
        }
        for field, expected in expected_manifest.items():
            if manifest_data.get(field) != expected:
                errors.append(f"manifest: {field} mismatch")

    report["run_id"] = run_id
    report["configured_budget_tokens"] = budget
    report["cumulative_visible_tokens"] = cumulative
    report["cumulative_source_resolution_tokens"] = source_cumulative
    report["remaining_tokens"] = budget - cumulative
    report["exhausted"] = exhausted
    report["delivery_record_count"] = delivery_count
    report["tokenizer"] = tokenizer_identity
    report["provider_exact"] = False
    report["memory_before_sha256"] = (
        memory_before.get("sha256") if isinstance(memory_before, dict) else None
    )
    report["memory_after_sha256"] = (
        finalizer.get("memory_after", {}).get("sha256")
        if isinstance(finalizer, dict) and isinstance(finalizer.get("memory_after"), dict)
        else None
    )
    report["audit_status"] = "pass" if not errors else "fail"
    return report


__all__ = ["DuplicateJsonKeyError", "audit_visible_token_trace"]
