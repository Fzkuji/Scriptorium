#!/usr/bin/env python3
"""Resumable LoCoMo/LongMemEval-S judging for the v8.8 GPT-5.5 runs.

The benchmark method uses GPT-5.5, but the protocol-comparable judge remains
``openai/gpt-4o-mini`` through OpenRouter.  ``gpt-5.5`` through the local
subscription proxy is available only as a separately labelled secondary
analysis.  One output file must contain exactly one judge profile.

This script intentionally does not implement prompts or aggregation itself.
It calls :mod:`src.evaluation.judges` and :mod:`src.evaluation.evaluate`, then
records hashes of those files so a resumed file cannot mix implementations.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import tempfile
import unicodedata
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src import openai_gpt55_flex_gateway_evidence as flex_evidence
from src import openrouter_gateway_evidence


ROOT = Path(__file__).resolve().parents[1]
LOCOMO_DATA = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
LONGMEMEVAL_DATA = (
    ROOT / "benchmarks" / "longmemeval" / "data" / "longmemeval_s_cleaned.json"
)

JUDGE_PROFILES: dict[str, dict[str, Any]] = {
    "primary": {
        "id": "protocol-primary-openrouter-gpt4o-mini",
        "provider": "OpenRouter",
        "requested_model": "openai/gpt-4o-mini",
        "base_url": "dynamic-openrouter-gateway",
        "comparison_status": "protocol-comparable",
        "comparable_to_published_primary": True,
        "note": "Primary judge required by docs/experiments/protocols/unified_evaluation_protocol.md section 6.4.",
    },
    "secondary": {
        "id": "secondary-openai-flex-gpt5.5",
        "provider": "OpenAI API Flex",
        "requested_model": "gpt-5.5",
        "base_url": "dynamic-openai-flex-gateway",
        "comparison_status": "non-comparable",
        "comparable_to_published_primary": False,
        "note": (
            "Secondary sensitivity analysis only; scores are not comparable "
            "to the protocol primary or published gpt-4o-mini results."
        ),
    },
}

SCOPES = {
    "locomo": "LoCoMo cat1-4 only (cat5 excluded)",
    "locomo-cat5": (
        "LoCoMo category 5 abstention-only 446; reported separately from cat1-4"
    ),
    "longmemeval": "LongMemEval-S full 500 including 30 abstention questions",
}
UPSTREAM_BENCHMARKS = {
    "locomo": "LoCoMo",
    "locomo-cat5": "LoCoMo",
    "longmemeval": "LongMemEval-S",
}
LOCOMO_BENCHMARKS = frozenset({"locomo", "locomo-cat5"})
UPSTREAM_PROXY_EVIDENCE = {
    "evidence_scope": "shared_proxy_window",
    "exact_run_response_id_linkage": "unavailable_not_persisted",
    "run_call_count_comparison": "not_performed_without_exact_linkage",
    "global_actual_model_consistency": True,
}

# The requested artifacts are judge artifacts.  Lexical metrics are kept out of
# this online/resumable pass so missing optional NLTK/ROUGE dependencies cannot
# interrupt judging; they can be computed independently from the same answers.
METRICS = ["judge"]
DERIVED_QUESTION_KEYS = {
    "lexical",
    "lexical_refusal",
    "judge_score",
    "judge_raw",
    "judge_usage",
}
ATTEMPT_LEDGER_SCHEMA = 2
CAT5_CANONICAL_GOLD = "Not mentioned in the conversation"
CAT5_REFUSAL_PHRASES = (
    "not mentioned in the conversation",
    "not provided in the conversation",
    "not stated in the conversation",
    "cannot be determined from the conversation",
    "can't be determined from the conversation",
    "cannot determine from the conversation",
    "insufficient information in the conversation",
    "no information in the conversation",
)


def metrics_for_benchmark(benchmark: str) -> list[str]:
    if benchmark == "locomo-cat5":
        return ["judge", "lexical_refusal_diagnostic"]
    return list(METRICS)


class ScoringError(RuntimeError):
    """Raised when an input, partial file, or judge response is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ensure_distinct_paths(paths: dict[str, Path | None]) -> None:
    """Reject lexical, symlink, hard-link, case, and Unicode path aliases."""
    entries = [
        (label, path.expanduser().absolute())
        for label, path in paths.items() if path is not None
    ]
    for index, (left_label, left) in enumerate(entries):
        for right_label, right in entries[index + 1:]:
            aliases = left.resolve() == right.resolve()
            if left.exists() and right.exists():
                try:
                    aliases = aliases or os.path.samefile(left, right)
                except OSError:
                    pass
            left_parent = left.parent.resolve()
            right_parent = right.parent.resolve()
            left_name = unicodedata.normalize("NFC", left.name).casefold()
            right_name = unicodedata.normalize("NFC", right.name).casefold()
            aliases = aliases or (
                left_parent == right_parent and left_name == right_name
            )
            if aliases:
                raise ScoringError(
                    "artifact path collision: "
                    f"{left_label}={left}, {right_label}={right}"
                )


def attempt_ledger_path(output: Path) -> Path:
    return Path(f"{output.expanduser().resolve()}.attempts.jsonl")


def append_attempt_event(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    """Append one durable event and fsync before another model call may start."""
    path.parent.mkdir(parents=True, exist_ok=True)
    persisted = {
        "schema_version": ATTEMPT_LEDGER_SCHEMA,
        "event_id": str(uuid.uuid4()),
        "timestamp": utc_now(),
        **copy.deepcopy(event),
    }
    payload = (json.dumps(persisted, ensure_ascii=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("attempt ledger append wrote zero bytes")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return persisted


def read_attempt_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise ScoringError("attempt ledger has a non-durable partial tail")
    events: list[dict[str, Any]] = []
    ids: set[str] = set()
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        try:
            event = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScoringError(
                f"attempt ledger line {line_number} is invalid"
            ) from exc
        if (
            not isinstance(event, dict)
            or event.get("schema_version") != ATTEMPT_LEDGER_SCHEMA
            or not isinstance(event.get("event_id"), str)
            or not event["event_id"]
            or event["event_id"] in ids
        ):
            raise ScoringError(f"attempt ledger line {line_number} has invalid identity")
        ids.add(event["event_id"])
        events.append(event)
    return events


def ledger_report(path: Path, run_key: str) -> dict[str, Any]:
    events = read_attempt_ledger(path)
    relevant = [event for event in events if event.get("run_key") == run_key]
    return {
        "schema_version": ATTEMPT_LEDGER_SCHEMA,
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() else sha256_json([]),
        "entries": len(events),
        "run_entries": len(relevant),
        "physical_http_attempt_events": sum(
            event.get("event") == "physical_http_attempt" for event in relevant
        ),
        "logical_judge_call_events": sum(
            event.get("event") == "logical_judge_call" for event in relevant
        ),
        "judge_failures": sum(
            event.get("event") == "judge_failure" for event in relevant
        ),
        "judge_results": sum(
            event.get("event") == "judge_result" for event in relevant
        ),
    }


def _usage_attempt_without_logical_indices(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in detail.items()
        if key not in {"parse_attempt", "logical_judge_call"}
    }


def _validate_observed_attempt_group(
    *,
    question_id: str,
    observed: list[dict[str, Any]],
    usage: object,
    profile_name: str,
) -> None:
    """Bind durable observer events to one terminal judge call."""
    normalized = normalize_judge_usage(usage)
    details = validate_request_attempt_details(
        normalized, profile_name, question_id
    )
    del details  # validation side effect is the purpose; all attempts follow below.
    physical = [
        event for event in observed if event.get("event") == "physical_http_attempt"
    ]
    logical = [
        event for event in observed if event.get("event") == "logical_judge_call"
    ]
    expected_physical = [
        _usage_attempt_without_logical_indices(detail)
        for detail in normalized["request_attempt_details"]
    ]
    actual_physical = [event.get("attempt") for event in physical]
    if actual_physical != expected_physical:
        raise ScoringError(
            f"{question_id} durable physical attempts differ from terminal usage"
        )
    logical_calls = normalized["logical_judge_calls"]
    if len(logical) != logical_calls:
        raise ScoringError(
            f"{question_id} durable logical-call count differs from terminal usage"
        )
    for index, event in enumerate(logical, start=1):
        if (
            event.get("logical_judge_call") != index
            or event.get("status")
            not in {"http_failed", "parse_rejected", "parse_accepted"}
        ):
            raise ScoringError(
                f"{question_id} durable logical-call event {index} is invalid"
            )


def reconcile_attempt_ledger(
    events: list[dict[str, Any]],
    run_key: str,
    *,
    benchmark: str,
    profile_name: str,
) -> dict[str, list[dict[str, Any]]]:
    """Validate every observed call has exactly one later terminal event.

    An observed physical or logical event without a terminal failure/result may
    represent a completed provider call.  Resuming it would risk a duplicate,
    so callers must fail before issuing another request.
    """
    observed_by_id: dict[str, dict[str, Any]] = {}
    consumed_observed: set[str] = set()
    terminals: dict[str, list[dict[str, Any]]] = {}
    result_questions: set[str] = set()
    expected_profile = judge_profile(benchmark, profile_name)["id"]
    for event in events:
        if event.get("run_key") != run_key:
            continue
        event_type = event.get("event")
        if event_type in {"physical_http_attempt", "logical_judge_call"}:
            question_id = event.get("question_id")
            if not isinstance(question_id, str) or not question_id:
                raise ScoringError("attempt ledger observation lacks question id")
            observed_by_id[event["event_id"]] = event
            continue
        if event_type not in {"judge_failure", "judge_result"}:
            continue
        question_id = event.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ScoringError("attempt ledger terminal event lacks question id")
        if (
            event.get("benchmark") != benchmark
            or event.get("judge_profile") != expected_profile
        ):
            raise ScoringError(
                f"{question_id} attempt ledger terminal identity mismatch"
            )
        invocation_id = event.get("invocation_id")
        if not isinstance(invocation_id, str) or not invocation_id:
            raise ScoringError(
                f"{question_id} attempt ledger terminal lacks invocation identity"
            )
        observed_ids = event.get("observed_event_ids")
        if (
            not isinstance(observed_ids, list)
            or any(not isinstance(value, str) or not value for value in observed_ids)
            or len(set(observed_ids)) != len(observed_ids)
        ):
            raise ScoringError(
                f"{question_id} attempt ledger terminal has invalid observations"
            )
        if any(value in consumed_observed for value in observed_ids):
            raise ScoringError(
                f"{question_id} attempt ledger observation was consumed twice"
            )
        try:
            observed = [observed_by_id[value] for value in observed_ids]
        except KeyError as exc:
            raise ScoringError(
                f"{question_id} attempt ledger terminal references an unknown event"
            ) from exc
        if any(
            item.get("question_id") != question_id
            or item.get("benchmark") != benchmark
            or item.get("judge_profile") != expected_profile
            or item.get("invocation_id") != invocation_id
            for item in observed
        ):
            raise ScoringError(
                f"{question_id} attempt ledger terminal crosses call identities"
            )
        current_usage = event.get("current_usage")
        if observed_ids or current_usage is not None:
            if not isinstance(current_usage, dict):
                raise ScoringError(
                    f"{question_id} observed calls have no terminal usage"
                )
            _validate_observed_attempt_group(
                question_id=question_id,
                observed=observed,
                usage=current_usage,
                profile_name=profile_name,
            )
        consumed_observed.update(observed_ids)
        if event_type == "judge_result":
            if question_id in result_questions:
                raise ScoringError(
                    f"{question_id} has duplicate attempt-ledger results"
                )
            result_questions.add(question_id)
        terminals.setdefault(question_id, []).append(event)
    orphan_ids = set(observed_by_id) - consumed_observed
    if orphan_ids:
        first = observed_by_id[sorted(orphan_ids)[0]]
        raise ScoringError(
            "attempt ledger contains an unresolved observed call; refusing resume: "
            f"question={first.get('question_id')}, event_id={first.get('event_id')}"
        )
    return terminals


def default_score_audit_path(evaluation: Path) -> Path:
    return evaluation.expanduser().resolve().with_suffix(".audit.json")


def score_audit_registry_path(evaluation: Path) -> Path:
    evaluation = evaluation.expanduser().resolve()
    return Path(f"{evaluation}.audit-targets.json")


def register_score_audit(evaluation: Path, audit_output: Path) -> None:
    """Record an audit target so later score rewrites can remove stale passes."""
    evaluation = evaluation.expanduser().resolve()
    audit_output = audit_output.expanduser().resolve()
    registry = score_audit_registry_path(evaluation)
    targets: list[str] = []
    if registry.is_file():
        try:
            value = read_json(registry)
            if isinstance(value, dict) and isinstance(value.get("targets"), list):
                targets = [
                    str(Path(item).expanduser().resolve())
                    for item in value["targets"] if isinstance(item, str)
                ]
        except (OSError, json.JSONDecodeError):
            targets = []
    target = str(audit_output)
    if target not in targets:
        targets.append(target)
    atomic_json(registry, {
        "schema_version": 1,
        "evaluation": str(evaluation),
        "targets": sorted(set(targets)),
    })


def invalidate_score_audits(evaluation: Path) -> None:
    """Remove registered reports that authenticate an evaluation being changed."""
    evaluation = evaluation.expanduser().resolve()
    default = default_score_audit_path(evaluation)
    registry = score_audit_registry_path(evaluation)
    candidates = {default}
    if registry.is_file():
        try:
            value = read_json(registry)
            if (
                isinstance(value, dict)
                and value.get("evaluation") == str(evaluation)
                and isinstance(value.get("targets"), list)
            ):
                candidates.update(
                    Path(item).expanduser().resolve()
                    for item in value["targets"] if isinstance(item, str)
                )
        except (OSError, json.JSONDecodeError):
            pass
    for candidate in candidates:
        if not candidate.is_file():
            continue
        safe_to_remove = candidate == default
        if not safe_to_remove:
            try:
                report = read_json(candidate)
                safe_to_remove = (
                    isinstance(report, dict)
                    and report.get("evaluation") == str(evaluation)
                )
            except (OSError, json.JSONDecodeError):
                safe_to_remove = False
        if safe_to_remove:
            candidate.unlink(missing_ok=True)
    registry.unlink(missing_ok=True)


def recorded_proxy_path(source_audit: Path) -> Path | None:
    """Read only the proxy path needed for destructive-output preflight."""
    try:
        report = read_json(source_audit.expanduser().resolve())
    except (OSError, json.JSONDecodeError):
        return None
    value = report.get("proxy_window", {}).get("path") if isinstance(report, dict) else None
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_sha256(benchmark: str) -> str:
    path = LOCOMO_DATA if benchmark in LOCOMO_BENCHMARKS else LONGMEMEVAL_DATA
    return sha256_file(path)


def judge_profile(benchmark: str, profile_name: str) -> dict[str, Any]:
    """Return benchmark-specific comparison metadata for a judge profile."""
    profile = copy.deepcopy(JUDGE_PROFILES[profile_name])
    if benchmark == "locomo-cat5":
        profile["comparison_status"] = "separate-category-experiment"
        profile["comparable_to_published_primary"] = False
        profile["note"] = (
            "LoCoMo category 5 is evaluated and reported separately from the "
            "published category 1-4 primary result. Missing-answer records use "
            "the semantic abstention judge; records with an explicit source "
            "answer use the ordinary LoCoMo judge."
        )
    elif benchmark == "longmemeval" and profile_name == "primary":
        profile["comparable_to_published_primary"] = False
        profile["note"] = (
            "Primary judge required by the unified protocol. LongMemEval's "
            "published primary used gpt-4o-2024-08-06, so this gpt-4o-mini "
            "score is not directly comparable to that published number."
        )
    return profile


def gateway_start_binding(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the immutable fields used in the scorer run identity."""
    if value is None:
        return None
    return {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key not in {"end_prefix", "state_at_end"}
    }


def flex_gateway_start_binding(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    fields = (
        "schema",
        "result_root",
        "base_url",
        "origin",
        "provider",
        "provider_model",
        "returned_alias",
        "service_tier",
        "billing",
        "max_cost_nanos",
        "max_cost_usd",
        "root_marker_sha256",
        "ready_canonical_sha256",
    )
    return {key: copy.deepcopy(value.get(key)) for key in fields}


def _secondary_response_ids(records: list[dict[str, Any]]) -> list[str]:
    response_ids: list[str] = []
    for record in records:
        if record.get("question_id") == "_build_stats" or "judge_usage" not in record:
            continue
        usage = normalize_judge_usage(record["judge_usage"])
        response_ids.extend(
            str(detail["response_id"])
            for detail in usage["request_attempt_details"]
            if detail["response_id"] is not None
        )
    if len(response_ids) != len(set(response_ids)):
        raise ScoringError("secondary judge response IDs are duplicated")
    return response_ids


def validate_secondary_flex_evidence(
    records: list[dict[str, Any]], evidence_root: Path
) -> dict[str, Any]:
    """Independently audit every bounded Flex invocation for one evaluation."""

    root = evidence_root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ScoringError("secondary Flex evidence root is missing")
    directories = sorted(root.glob("invocation-*"))
    expected_names = {
        f"invocation-{index:04d}" for index in range(1, len(directories) + 1)
    }
    if not directories or {path.name for path in root.iterdir()} != expected_names:
        raise ScoringError("secondary Flex invocation inventory differs")
    invocation_rows: list[dict[str, Any]] = []
    provider_response_ids: list[str] = []
    gateway_request_ids: list[str] = []
    for index, directory in enumerate(directories, start=1):
        if (
            directory.name != f"invocation-{index:04d}"
            or not directory.is_dir()
            or directory.is_symlink()
        ):
            raise ScoringError("secondary Flex invocation directory differs")
        expected_files = {
            flex_evidence.CHILD_READY_NAME,
            flex_evidence.CHILD_LOG_NAME,
            flex_evidence.WINDOW_NAME,
            flex_evidence.INVOCATION_NAME,
        }
        if {path.name for path in directory.iterdir()} != expected_files:
            raise ScoringError("secondary Flex invocation files differ")
        invocation_path = directory / flex_evidence.INVOCATION_NAME
        invocation = read_json(invocation_path)
        if not isinstance(invocation, dict):
            raise ScoringError("secondary Flex invocation is not an object")
        for field, filename in (
            ("window", flex_evidence.WINDOW_NAME),
            ("consumer_log", flex_evidence.CHILD_LOG_NAME),
            ("child_ready", flex_evidence.CHILD_READY_NAME),
        ):
            binding = invocation.get(field)
            if (
                not isinstance(binding, dict)
                or Path(str(binding.get("path"))).resolve() != directory / filename
            ):
                raise ScoringError(f"secondary Flex {field} path differs")
        try:
            report = flex_evidence.audit_invocation(invocation)
            consumers = flex_evidence.load_child_proxy_records(
                directory / flex_evidence.CHILD_LOG_NAME,
                expected_run_id=str(invocation.get("run_id")),
            )
        except flex_evidence.EvidenceError as exc:
            raise ScoringError(str(exc)) from exc
        provider_response_ids.extend(str(row["response_id"]) for row in consumers)
        gateway_request_ids.extend(str(row["gateway_request_id"]) for row in consumers)
        invocation_rows.append(
            {
                "path": str(invocation_path),
                "sha256": sha256_file(invocation_path),
                "requests": report["requests"],
                "segment_sha256": report["segment_sha256"],
                "committed_cost_nanos": report["committed_cost_nanos"],
            }
        )
    expected_response_ids = _secondary_response_ids(records)
    if (
        len(provider_response_ids) != len(set(provider_response_ids))
        or set(provider_response_ids) != set(expected_response_ids)
    ):
        raise ScoringError(
            "secondary judge responses do not equal bounded Flex provider responses"
        )
    if len(gateway_request_ids) != len(set(gateway_request_ids)):
        raise ScoringError("secondary Flex gateway request IDs are duplicated")
    return {
        "status": "passed",
        "transport": "exclusive-child-proxy-to-openai-gpt55-flex-gateway",
        "evidence_root": str(root),
        "invocations": invocation_rows,
        "requests": len(provider_response_ids),
        "response_ids": len(expected_response_ids),
        "gateway_request_ids_sha256": sha256_json(gateway_request_ids),
        "committed_cost_nanos": sum(
            int(row["committed_cost_nanos"]) for row in invocation_rows
        ),
    }


def _rerun_upstream_audit(
    benchmark: str, run_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute the benchmark auditor instead of trusting its saved report."""
    if benchmark in LOCOMO_BENCHMARKS:
        manifest = read_json(run_dir / "run_manifest.json")
        if not isinstance(manifest, dict):
            raise ScoringError("LoCoMo run manifest is not an object")
        config = manifest.get("config")
        config_provider = (
            config.get("provider") if isinstance(config, dict) else None
        )
        provider = manifest.get("provider") or config_provider
        if (
            provider == "chatgpt_pro_subscription"
            and manifest.get("formal_flex_result") is False
        ):
            from scripts import audit_v88_gpt55_locomo_subscription as upstream
        elif provider == "openai_api_flex_via_exclusive_child_proxy":
            from scripts import audit_v88_gpt55_locomo as upstream
        else:
            raise ScoringError(
                f"unsupported LoCoMo generation provider: {provider!r}"
            )
    elif benchmark == "longmemeval":
        from scripts import audit_v88_gpt55_longmemeval as upstream
    else:
        raise ScoringError(f"unsupported benchmark: {benchmark}")
    try:
        return upstream.audit(run_dir)
    except Exception as exc:  # noqa: BLE001
        raise ScoringError(
            f"upstream audit re-execution failed: {type(exc).__name__}: {exc}"
        ) from exc


def _json_normalize(value: object) -> Any:
    """Normalize in-memory auditor output to its JSON representation."""
    return json.loads(json.dumps(value, ensure_ascii=False))


def validate_upstream_audit(
    benchmark: str, source_input: Path, source_audit: Path
) -> dict[str, Any]:
    """Bind a scoring input to its completed benchmark run and manifest."""
    source_input = source_input.expanduser().resolve()
    source_audit = source_audit.expanduser().resolve()
    run_dir = source_input.parent
    expected_input_name = (
        "questions_all.json"
        if benchmark in LOCOMO_BENCHMARKS
        else "evaluation_input.json"
    )
    if source_input != run_dir / expected_input_name:
        raise ScoringError(
            f"scoring input must be <run_dir>/{expected_input_name}"
        )
    if source_audit != run_dir / "audit.json":
        raise ScoringError("source audit must be <run_dir>/audit.json")
    report = read_json(source_audit)
    if not isinstance(report, dict):
        raise ScoringError("source audit is not a JSON object")
    expected = {
        "schema_version": 1,
        "status": "passed",
        "benchmark": UPSTREAM_BENCHMARKS[benchmark],
        "method": "NativeMem-v8.8+calendar",
        "model": "gpt-5.5",
        "run_dir": str(run_dir),
        "source_hashes_match": True,
        "empty_answers": 0,
    }
    if benchmark in LOCOMO_BENCHMARKS:
        expected.update({"questions": 1986, "cat1_4": 1540})
        legacy_hash_key = "combined_sha256"
    else:
        expected.update({"questions": 500, "abstention_questions": 30})
        legacy_hash_key = "evaluation_input_sha256"
    mismatches = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise ScoringError(f"source audit metadata mismatch: {mismatches}")

    input_sha256 = sha256_file(source_input)
    scoring_input = report.get("scoring_input")
    expected_input = {"path": str(source_input), "sha256": input_sha256}
    if scoring_input != expected_input or report.get(legacy_hash_key) != input_sha256:
        raise ScoringError("source audit does not authenticate the scoring input")

    manifest_path = run_dir / "run_manifest.json"
    manifest_info = report.get("manifest")
    if not isinstance(manifest_info, dict):
        raise ScoringError("source audit does not record its run manifest")
    if manifest_info.get("path") != str(manifest_path):
        raise ScoringError("source audit manifest path does not match the run directory")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_info.get("sha256") != manifest_sha256:
        raise ScoringError("source audit manifest hash does not match the run manifest")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ScoringError("upstream run manifest is not complete")
    if benchmark in LOCOMO_BENCHMARKS:
        if (
            manifest.get("benchmark") != "locomo"
            or manifest.get("method") != "NativeMem-v8.8+calendar"
            or manifest.get("backbone") != "gpt-5.5"
        ):
            raise ScoringError("LoCoMo run manifest identity mismatch")
    else:
        models = manifest.get("models", {})
        if (
            manifest.get("benchmark") != "LongMemEval-S"
            or manifest.get("completed") != 500
            or manifest.get("method", {}).get("version") != "v8.8+calendar"
            or any(
                models.get(role) != "gpt-5.5"
                for role in ("builder", "retriever", "answerer")
            )
        ):
            raise ScoringError("LongMemEval-S run manifest identity mismatch")

    proxy_window = report.get("proxy_window")
    if not isinstance(proxy_window, dict):
        raise ScoringError("source audit does not record its proxy request log")
    proxy_path_value = proxy_window.get("path")
    if not isinstance(proxy_path_value, str) or not proxy_path_value.strip():
        raise ScoringError("source audit proxy request log path is empty")
    proxy_path = Path(proxy_path_value).expanduser().resolve()
    if not proxy_path.is_file():
        raise ScoringError(f"source audit proxy request log is missing: {proxy_path}")
    proxy_evidence = {
        key: proxy_window.get(key) for key in UPSTREAM_PROXY_EVIDENCE
    }
    if proxy_evidence != UPSTREAM_PROXY_EVIDENCE:
        raise ScoringError(
            "source audit does not disclose the shared proxy evidence limitation"
        )
    if proxy_window.get("actual_models") != ["gpt-5.5"]:
        raise ScoringError("source audit has unexpected actual models")

    recomputed_records, recomputed_report = _rerun_upstream_audit(
        benchmark, run_dir
    )
    source_records = _load_record_list(source_input)
    if _json_normalize(recomputed_records) != _json_normalize(source_records):
        raise ScoringError(
            "scoring input differs from records recomputed by the upstream auditor"
        )
    authenticated_report = copy.deepcopy(recomputed_report)
    authenticated_report[legacy_hash_key] = input_sha256
    authenticated_report["scoring_input"] = expected_input
    if _json_normalize(authenticated_report) != _json_normalize(report):
        raise ScoringError(
            "source audit differs from the recomputed upstream audit report"
        )

    return {
        "run_dir": str(run_dir),
        "audit_path": str(source_audit),
        "audit_sha256": sha256_file(source_audit),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "proxy_log_path": str(proxy_path),
        "upstream_proxy_evidence": proxy_evidence,
    }


@contextmanager
def exclusive_output_lock(output: Path):
    """Reject concurrent writers targeting the same evaluation artifact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{output}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ScoringError(f"evaluation output is already locked: {output}") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def code_hashes() -> dict[str, str]:
    paths = {
        "scorer": Path(__file__).resolve(),
        "evaluate": ROOT / "src" / "evaluation" / "evaluate.py",
        "judges": ROOT / "src" / "evaluation" / "judges.py",
        "llm_clients": ROOT / "src" / "evaluation" / "llm_clients.py",
        "openrouter_gateway_evidence": (
            ROOT / "src" / "openrouter_gateway_evidence.py"
        ),
        "openai_flex_gateway_evidence": (
            ROOT / "src" / "openai_gpt55_flex_gateway_evidence.py"
        ),
        "openai_flex_gateway": ROOT / "src" / "openai_gpt55_flex_gateway.py",
        "prompts": ROOT / "src" / "evaluation" / "prompts.py",
        "metrics": ROOT / "src" / "evaluation" / "metrics.py",
        "upstream_audit_locomo": (
            ROOT / "scripts" / "audit_v88_gpt55_locomo.py"
        ),
        "upstream_audit_locomo_subscription": (
            ROOT / "scripts" / "audit_v88_gpt55_locomo_subscription.py"
        ),
        "upstream_audit_longmemeval": (
            ROOT / "scripts" / "audit_v88_gpt55_longmemeval.py"
        ),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _load_record_list(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if isinstance(value, dict):
        for key in ("results", "questions", "individual_results"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ScoringError(f"input is not a JSON record list: {path}")
    return value


def _expected_locomo() -> dict[str, dict[str, Any]]:
    data = read_json(LOCOMO_DATA)
    if not isinstance(data, list) or len(data) != 10:
        raise ScoringError("LoCoMo source must contain 10 conversations")
    expected: dict[str, dict[str, Any]] = {}
    for sample, conversation in enumerate(data):
        for index, qa in enumerate(conversation.get("qa", [])):
            category = int(qa["category"])
            if category not in (1, 2, 3, 4):
                continue
            expected[f"s{sample}_q{index}"] = {
                "question": qa["question"],
                "gold": str(qa["answer"]),
                "category": category,
            }
    counts = Counter(item["category"] for item in expected.values())
    if len(expected) != 1540 or dict(counts) != {1: 282, 2: 321, 3: 96, 4: 841}:
        raise ScoringError(f"unexpected LoCoMo cat1-4 inventory: {dict(counts)}")
    return expected


def _expected_locomo_cat5() -> dict[str, dict[str, Any]]:
    """Rebuild the category-5 inventory from raw LoCoMo, never from audit gold."""
    data = read_json(LOCOMO_DATA)
    if not isinstance(data, list) or len(data) != 10:
        raise ScoringError("LoCoMo source must contain 10 conversations")
    expected: dict[str, dict[str, Any]] = {}
    missing_explicit_answer = 0
    for sample, conversation in enumerate(data):
        for index, qa in enumerate(conversation.get("qa", [])):
            if int(qa["category"]) != 5:
                continue
            explicit_answer = qa.get("answer")
            is_abstention = explicit_answer is None or not str(explicit_answer).strip()
            if is_abstention:
                gold = CAT5_CANONICAL_GOLD
                missing_explicit_answer += 1
            else:
                gold = str(explicit_answer)
            distractor = qa.get("adversarial_answer")
            if not isinstance(distractor, str) or not distractor.strip():
                raise ScoringError(
                    f"LoCoMo category-5 s{sample}_q{index} lacks a distractor"
                )
            expected[f"s{sample}_q{index}"] = {
                "question": qa["question"],
                "gold": gold,
                "category": 5,
                "cat5_abstention": is_abstention,
                "distractor": distractor,
            }
    if len(expected) != 446 or missing_explicit_answer != 444:
        raise ScoringError(
            "unexpected LoCoMo category-5 inventory: "
            f"questions={len(expected)}, missing_answer={missing_explicit_answer}"
        )
    return expected


def _expected_longmemeval() -> dict[str, dict[str, Any]]:
    data = read_json(LONGMEMEVAL_DATA)
    if not isinstance(data, list) or len(data) != 500:
        raise ScoringError("LongMemEval-S source must contain 500 questions")
    expected: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(data):
        question_id = str(item.get("question_id", ""))
        if not question_id or question_id in expected:
            raise ScoringError("LongMemEval-S question ids are missing or duplicated")
        expected[question_id] = {
            "dataset_index": index,
            "question": item["question"],
            "gold": item["answer"],
            "question_type": item["question_type"],
            "abstention": question_id.endswith("_abs"),
        }
    types = Counter(item["question_type"] for item in expected.values())
    required = {
        "multi-session": 133,
        "temporal-reasoning": 133,
        "knowledge-update": 78,
        "single-session-user": 70,
        "single-session-assistant": 56,
        "single-session-preference": 30,
    }
    if dict(types) != required:
        raise ScoringError(f"unexpected LongMemEval-S inventory: {dict(types)}")
    if sum(item["abstention"] for item in expected.values()) != 30:
        raise ScoringError("LongMemEval-S abstention inventory is not 30")
    return expected


def select_official_records(
    records: list[dict[str, Any]], benchmark: str
) -> list[dict[str, Any]]:
    """Validate raw-audit records and select the protocol's scoring scope."""
    if benchmark == "locomo":
        expected = _expected_locomo()
        required_builds = 10
    elif benchmark == "locomo-cat5":
        expected = _expected_locomo_cat5()
        required_builds = 10
    elif benchmark == "longmemeval":
        expected = _expected_longmemeval()
        required_builds = 500
    else:
        raise ScoringError(f"unsupported benchmark: {benchmark}")

    selected: list[dict[str, Any]] = []
    observed: set[str] = set()
    build_identities: set[int] = set()
    for source in records:
        record = copy.deepcopy(source)
        if record.get("question_id") == "_build_stats":
            identity_key = (
                "sample" if benchmark in LOCOMO_BENCHMARKS else "dataset_index"
            )
            identity = record.get(identity_key)
            if isinstance(identity, bool) or not isinstance(identity, int):
                raise ScoringError(
                    f"{benchmark} build record has invalid {identity_key}"
                )
            if identity in build_identities or not 0 <= identity < required_builds:
                raise ScoringError(
                    f"{benchmark} build identity is duplicated or out of range: "
                    f"{identity}"
                )
            if benchmark == "longmemeval":
                expected_question_id = next(
                    question_id
                    for question_id, item in expected.items()
                    if item["dataset_index"] == identity
                )
                if str(record.get("source_question_id", "")) != expected_question_id:
                    raise ScoringError(
                        f"longmemeval build {identity} source question id mismatch"
                    )
            imported_accounting = record.get("build_accounting")
            authenticated_import = (
                benchmark in LOCOMO_BENCHMARKS
                and identity == 0
                and isinstance(imported_accounting, dict)
                and imported_accounting.get("status")
                == "unavailable_imported_subscription_memory"
                and imported_accounting.get("source_provider")
                == "chatgpt_pro_subscription"
                and imported_accounting.get("build_calls") is None
                and imported_accounting.get("build_tokens_in") is None
                and imported_accounting.get("build_tokens_out") is None
                and imported_accounting.get("build_llm_time_s") is None
                and all(
                    isinstance(imported_accounting.get(name), dict)
                    for name in (
                        "source_manifest", "source_memory", "imported_memory"
                    )
                )
                and all(
                    re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256", "")))
                    is not None
                    for binding in (
                        imported_accounting.get("source_manifest", {}),
                        imported_accounting.get("source_memory", {}),
                        imported_accounting.get("imported_memory", {}),
                    )
                )
                and record.get("build_calls") is None
                and record.get("build_tokens_in") is None
            )
            for key in ("build_calls", "build_tokens_in", "num_memories"):
                value = record.get(key)
                if authenticated_import and key in {
                    "build_calls", "build_tokens_in"
                }:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                    raise ScoringError(
                        f"{benchmark} build {identity} has invalid {key}"
                    )
            build_identities.add(identity)
            selected.append(record)
            continue
        question_id = str(record.get("question_id", ""))
        if benchmark == "locomo" and record.get("category") == 5:
            continue
        if benchmark == "locomo-cat5" and record.get("category") != 5:
            continue
        if question_id not in expected:
            raise ScoringError(f"unknown {benchmark} question id: {question_id!r}")
        if question_id in observed:
            raise ScoringError(f"duplicate {benchmark} question id: {question_id}")
        reference = expected[question_id]
        source_reference = {
            key: value
            for key, value in reference.items()
            if key not in {"distractor", "cat5_abstention"}
        }
        # The upstream source audit historically wrote adversarial_answer into
        # category-5 gold.  Authenticate all placement-independent source fields,
        # then replace that value from the raw dataset before any judge call.
        if benchmark == "locomo-cat5":
            source_reference.pop("gold")
        for key, value in source_reference.items():
            if record.get(key) != value:
                raise ScoringError(
                    f"{benchmark} {question_id} differs from source field {key}"
                )
        if benchmark == "locomo-cat5":
            record["gold"] = reference["gold"]
            record["cat5_abstention"] = reference["cat5_abstention"]
            record["distractor"] = reference["distractor"]
        if not str(record.get("answer", "")).strip():
            raise ScoringError(f"{benchmark} {question_id} has an empty answer")
        retrieval = record.get("retrieval")
        if not isinstance(retrieval, dict):
            raise ScoringError(f"{benchmark} {question_id} lacks retrieval metadata")
        for key in ("calls", "steps", "tokens_in"):
            value = retrieval.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ScoringError(
                    f"{benchmark} {question_id} has invalid retrieval {key}"
                )
        conflicts = DERIVED_QUESTION_KEYS.intersection(record)
        if conflicts:
            raise ScoringError(
                f"raw input {question_id} already contains scoring fields: "
                f"{sorted(conflicts)}"
            )
        observed.add(question_id)
        selected.append(record)

    missing = set(expected) - observed
    if missing:
        raise ScoringError(
            f"{benchmark} input is missing {len(missing)} questions; "
            f"first={sorted(missing)[0]}"
        )
    expected_build_identities = set(range(required_builds))
    if build_identities != expected_build_identities:
        raise ScoringError(
            f"{benchmark} input has invalid build identities"
        )
    return selected


def configure_judge(
    profile_name: str,
    *,
    openrouter_gateway: dict[str, Any] | None = None,
    secondary_base_url: str | None = None,
) -> dict[str, Any]:
    profile = copy.deepcopy(JUDGE_PROFILES[profile_name])
    if profile_name == "primary":
        if not isinstance(openrouter_gateway, dict):
            raise ScoringError(
                "primary formal scoring requires a marked loopback OpenRouter gateway"
            )
        profile["base_url"] = str(openrouter_gateway["base_url"])
    elif secondary_base_url is not None:
        profile["base_url"] = secondary_base_url
    os.environ["JUDGE_MODEL"] = profile["requested_model"]
    os.environ["JUDGE_BASE"] = profile["base_url"]
    if profile_name == "primary":
        # The marked loopback gateway owns the real credential.  The SDK still
        # requires a non-empty value but this local bearer value has no billing
        # authority and is not persisted.
        os.environ["JUDGE_KEY"] = "local-openrouter-gateway"
        os.environ["JUDGE_HTTP_RETRIES"] = "1"
    elif secondary_base_url is not None:
        os.environ["JUDGE_KEY"] = "local-openai-flex-gateway"
        os.environ["JUDGE_HTTP_RETRIES"] = "1"
    else:
        os.environ["JUDGE_KEY"] = os.environ.get("JUDGE_KEY") or "local-proxy"
    return profile


def response_model_matches(profile_name: str, value: object) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        return False
    model = value
    if profile_name == "secondary":
        return model == "gpt-5.5"
    canonical = "openai/gpt-4o-mini"
    if model == canonical:
        return True
    prefix = f"{canonical}-"
    if not model.startswith(prefix):
        return False
    date_suffix = model.removeprefix(prefix)
    try:
        parsed = datetime.strptime(date_suffix, "%Y-%m-%d")
    except ValueError:
        return False
    return parsed.strftime("%Y-%m-%d") == date_suffix


def normalize_judge_usage(usage: object) -> dict[str, Any]:
    """Normalize explicit logical-call and physical-attempt counters."""
    if not isinstance(usage, dict):
        raise ScoringError("judge usage is not an object")
    normalized = copy.deepcopy(usage)
    normalized.setdefault(
        "logical_judge_calls", normalized.get("parse_attempts", 0)
    )
    normalized.setdefault(
        "physical_http_attempts", normalized.get("request_attempts", 0)
    )
    details = normalized.get("request_attempt_details")
    if isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict):
                detail.setdefault("logical_judge_call", detail.get("parse_attempt"))
    return normalized


def merge_judge_usages(*values: object) -> dict[str, Any]:
    """Combine historical failed calls and the current successful call."""
    usages = [normalize_judge_usage(value) for value in values if value is not None]
    if not usages:
        raise ScoringError("cannot merge an empty judge-usage history")
    merged = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "parse_attempts": 0,
        "logical_judge_calls": 0,
        "request_attempts": 0,
        "physical_http_attempts": 0,
        "failed_request_attempts": 0,
        "unknown_token_attempts": 0,
        "request_attempt_details": [],
        "requested_models": [],
        "response_models": [],
        "response_ids": [],
        "finish_reasons": [],
        "refusals": [],
        "choice_counts": [],
    }
    for usage in usages:
        logical_offset = merged["logical_judge_calls"]
        for key in (
            "prompt_tokens", "completion_tokens", "parse_attempts",
            "logical_judge_calls", "request_attempts", "physical_http_attempts",
            "failed_request_attempts", "unknown_token_attempts",
        ):
            merged[key] += usage.get(key, 0) or 0
        for detail in usage.get("request_attempt_details", []):
            copied = copy.deepcopy(detail)
            logical_call = copied.get(
                "logical_judge_call", copied.get("parse_attempt")
            )
            if isinstance(logical_call, int) and not isinstance(logical_call, bool):
                copied["logical_judge_call"] = logical_call + logical_offset
                copied["parse_attempt"] = logical_call + logical_offset
            merged["request_attempt_details"].append(copied)
        for key in (
            "requested_models", "response_models", "response_ids",
            "finish_reasons", "refusals", "choice_counts",
        ):
            merged[key].extend(copy.deepcopy(usage.get(key, [])))
    return merged


def validate_request_attempt_details(
    usage: dict[str, Any], profile_name: str, question_id: str,
) -> list[dict[str, Any]]:
    """Validate every provider request, including rejected/error attempts."""
    parse_attempts = usage.get("parse_attempts")
    logical_calls = usage.get("logical_judge_calls")
    request_attempts = usage.get("request_attempts")
    if (
        isinstance(logical_calls, bool)
        or not isinstance(logical_calls, int)
        or logical_calls < parse_attempts
        or logical_calls <= 0
        or usage.get("physical_http_attempts") != request_attempts
        or
        isinstance(request_attempts, bool)
        or not isinstance(request_attempts, int)
        or not logical_calls <= request_attempts <= logical_calls * 3
    ):
        raise ScoringError(f"{question_id} has invalid request_attempts")
    details = usage.get("request_attempt_details")
    if not isinstance(details, list) or len(details) != request_attempts:
        raise ScoringError(f"{question_id} request attempt details do not match")
    required_keys = {
        "parse_attempt", "logical_judge_call", "request_attempt", "status", "failure_type",
        "error_type", "error_message", "requested_model", "response_model",
        "response_id", "finish_reason", "refusal", "choice_count",
        "prompt_tokens", "completion_tokens",
    }
    expected_model = JUDGE_PROFILES[profile_name]["requested_model"]
    known_prompt = 0
    known_completion = 0
    failed = 0
    unknown = 0
    accepted_by_parse: dict[int, dict[str, Any]] = {}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for detail in details:
        if not isinstance(detail, dict) or set(detail) != required_keys:
            raise ScoringError(f"{question_id} has invalid request attempt schema")
        parse_index = detail["logical_judge_call"]
        request_index = detail["request_attempt"]
        if (
            isinstance(parse_index, bool) or not isinstance(parse_index, int)
            or not 1 <= parse_index <= logical_calls
            or detail["parse_attempt"] != parse_index
            or isinstance(request_index, bool) or not isinstance(request_index, int)
            or request_index <= 0
        ):
            raise ScoringError(f"{question_id} has invalid request attempt index")
        grouped.setdefault(parse_index, []).append(detail)
        if detail["requested_model"] != expected_model:
            raise ScoringError(f"{question_id} request attempt used unexpected model")
        response_model = detail["response_model"]
        if response_model is not None and not response_model_matches(
            profile_name, response_model
        ):
            raise ScoringError(f"{question_id} request attempt response model mismatch")
        response_id = detail["response_id"]
        if response_id is not None and (
            not isinstance(response_id, str)
            or not response_id.strip()
            or response_id != response_id.strip()
        ):
            raise ScoringError(f"{question_id} request attempt has invalid response id")
        for token_key in ("prompt_tokens", "completion_tokens"):
            value = detail[token_key]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ScoringError(
                    f"{question_id} request attempt has invalid {token_key}"
                )
        known_prompt += detail["prompt_tokens"] or 0
        known_completion += detail["completion_tokens"] or 0
        unknown += int(
            detail["prompt_tokens"] is None
            or detail["completion_tokens"] is None
        )

        status = detail["status"]
        if status == "accepted":
            if parse_index in accepted_by_parse:
                raise ScoringError(f"{question_id} has duplicate accepted request")
            if (
                detail["failure_type"] is not None
                or detail["error_type"] is not None
                or detail["error_message"] is not None
                or detail["finish_reason"] != "stop"
                or detail["refusal"] is not None
                or detail["choice_count"] != 1
                or response_id is None
                or detail["prompt_tokens"] is None
                or detail["completion_tokens"] is None
            ):
                raise ScoringError(f"{question_id} has invalid accepted request evidence")
            accepted_by_parse[parse_index] = detail
        elif status == "rejected":
            failed += 1
            if (
                detail["failure_type"] != "invalid_response"
                or not isinstance(detail["error_type"], str)
                or not detail["error_type"]
                or not isinstance(detail["error_message"], str)
                or not detail["error_message"]
            ):
                raise ScoringError(f"{question_id} has invalid rejected request evidence")
        elif status == "error":
            failed += 1
            if (
                detail["failure_type"] != "transport_error"
                or not isinstance(detail["error_type"], str)
                or not detail["error_type"]
                or not isinstance(detail["error_message"], str)
                or not detail["error_message"]
                or any(detail[key] is not None for key in (
                    "response_model", "response_id", "finish_reason", "refusal",
                    "choice_count", "prompt_tokens", "completion_tokens",
                ))
            ):
                raise ScoringError(f"{question_id} has invalid transport error evidence")
        else:
            raise ScoringError(f"{question_id} has invalid request attempt status")

    if set(grouped) != set(range(1, logical_calls + 1)):
        raise ScoringError(f"{question_id} request attempts omit a logical call")
    for parse_index, group in grouped.items():
        if [item["request_attempt"] for item in group] != list(
            range(1, len(group) + 1)
        ):
            raise ScoringError(f"{question_id} request attempt order is invalid")
        if parse_index in accepted_by_parse:
            if group[-1]["status"] != "accepted":
                raise ScoringError(
                    f"{question_id} accepted logical call does not end in acceptance"
                )
        elif any(item["status"] == "accepted" for item in group):
            raise ScoringError(f"{question_id} accepted-attempt accounting differs")
    if len(accepted_by_parse) != parse_attempts:
        raise ScoringError(f"{question_id} parse-attempt count mismatch")

    for count_key in ("failed_request_attempts", "unknown_token_attempts"):
        count_value = usage.get(count_key)
        if (
            isinstance(count_value, bool)
            or not isinstance(count_value, int)
            or count_value < 0
        ):
            raise ScoringError(f"{question_id} has invalid {count_key}")
    if usage.get("failed_request_attempts") != failed:
        raise ScoringError(f"{question_id} failed request count mismatch")
    if usage.get("unknown_token_attempts") != unknown:
        raise ScoringError(f"{question_id} unknown token count mismatch")
    if usage.get("prompt_tokens") != known_prompt:
        raise ScoringError(f"{question_id} request prompt token total mismatch")
    if usage.get("completion_tokens") != known_completion:
        raise ScoringError(f"{question_id} request completion token total mismatch")
    return [accepted_by_parse[index] for index in sorted(accepted_by_parse)]


def validate_judge_result(
    score: object,
    raw: object,
    usage: object,
    benchmark: str,
    profile_name: str,
    question_id: str,
) -> None:
    if isinstance(score, bool) or not isinstance(score, int) or score not in (0, 1):
        raise ScoringError(f"{question_id} judge score is not integer 0/1")
    if not isinstance(raw, str) or not raw.strip():
        raise ScoringError(f"{question_id} judge raw response is empty")
    from src.evaluation.judges import _parse_label, _parse_yes_no_strict

    try:
        parsed_score = (
            _parse_label(raw)
            if benchmark in LOCOMO_BENCHMARKS
            else _parse_yes_no_strict(raw)
        )
    except ValueError as exc:
        raise ScoringError(
            f"{question_id} judge raw response is not parseable"
        ) from exc
    if parsed_score != score:
        raise ScoringError(f"{question_id} judge raw response disagrees with score")
    usage = normalize_judge_usage(usage)
    attempts = usage.get("parse_attempts")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ScoringError(f"{question_id} has invalid parse_attempts")
    for token_key in ("prompt_tokens", "completion_tokens"):
        value = usage.get(token_key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ScoringError(f"{question_id} has invalid {token_key}")
    requested = usage.get("requested_models")
    responses = usage.get("response_models")
    response_ids = usage.get("response_ids")
    if not isinstance(requested, list) or len(requested) != attempts:
        raise ScoringError(f"{question_id} requested_models do not match attempts")
    expected_model = JUDGE_PROFILES[profile_name]["requested_model"]
    if any(model != expected_model for model in requested):
        raise ScoringError(f"{question_id} used an unexpected requested model")
    if not isinstance(responses, list) or len(responses) != attempts:
        raise ScoringError(f"{question_id} response_models do not match attempts")
    if any(not response_model_matches(profile_name, model) for model in responses):
        raise ScoringError(f"{question_id} used an unexpected response model")
    if not isinstance(response_ids, list) or len(response_ids) != attempts:
        raise ScoringError(f"{question_id} response_ids do not match attempts")
    if any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in response_ids
    ):
        raise ScoringError(f"{question_id} has an empty response id")
    finish_reasons = usage.get("finish_reasons")
    refusals = usage.get("refusals")
    choice_counts = usage.get("choice_counts")
    if finish_reasons != ["stop"] * attempts:
        raise ScoringError(f"{question_id} has an invalid finish reason")
    if refusals != [None] * attempts:
        raise ScoringError(f"{question_id} has a refusal")
    if choice_counts != [1] * attempts:
        raise ScoringError(f"{question_id} has an invalid choice count")
    accepted_attempts = validate_request_attempt_details(
        usage, profile_name, question_id
    )
    if [item["requested_model"] for item in accepted_attempts] != requested:
        raise ScoringError(f"{question_id} accepted requested model evidence mismatch")
    if [item["response_model"] for item in accepted_attempts] != responses:
        raise ScoringError(f"{question_id} accepted response model evidence mismatch")
    if [item["response_id"] for item in accepted_attempts] != response_ids:
        raise ScoringError(f"{question_id} accepted response id evidence mismatch")


def _proxy_log_entries(
    path: Path, *, byte_limit: int | None = None
) -> tuple[list[dict[str, Any]], int, str]:
    """Read one immutable complete-line prefix of a shared JSONL log."""
    payload = path.read_bytes()
    if byte_limit is None:
        final_newline = payload.rfind(b"\n")
        cutoff = final_newline + 1 if final_newline >= 0 else 0
    else:
        if (
            isinstance(byte_limit, bool)
            or not isinstance(byte_limit, int)
            or byte_limit < 0
            or byte_limit > len(payload)
            or (byte_limit and payload[byte_limit - 1:byte_limit] != b"\n")
        ):
            raise ScoringError("frozen proxy JSONL byte cutoff is invalid")
        cutoff = byte_limit
    complete = payload[:cutoff]
    entries = []
    for line_number, raw_line in enumerate(complete.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            line = raw_line.decode("utf-8")
            entry = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScoringError(
                f"invalid proxy JSONL line {line_number}: {type(exc).__name__}"
            ) from exc
        if not isinstance(entry, dict):
            raise ScoringError(f"proxy JSONL line {line_number} is not an object")
        entries.append(entry)
    return entries, cutoff, hashlib.sha256(complete).hexdigest()


def validate_secondary_proxy_evidence(
    records: list[dict[str, Any]],
    proxy_log: Path,
    *,
    frozen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Match responses and account for every proxy-internal upstream attempt."""
    proxy_log = proxy_log.expanduser().resolve()
    by_response_id: dict[str, list[dict[str, Any]]] = {}
    byte_limit = frozen.get("log_prefix_bytes") if isinstance(frozen, dict) else None
    proxy_entries, prefix_bytes, prefix_sha256 = _proxy_log_entries(
        proxy_log, byte_limit=byte_limit
    )
    if (
        isinstance(frozen, dict)
        and frozen.get("log_prefix_sha256") != prefix_sha256
    ):
        raise ScoringError("frozen proxy JSONL prefix hash mismatch")
    normalized_errors: list[dict[str, Any]] = []
    proxy_response_ids: set[str] = set()
    total_upstream_attempts = 0
    for line_number, entry in enumerate(proxy_entries, start=1):
        status = entry.get("status")
        attempts = entry.get("attempts")
        unsupported = entry.get("unsupported_parameters")
        # Older proxy records predate this field.  Treat absence as an empty
        # list while validating and preserving it separately when present.
        ignored = entry.get("ignored_client_parameters", [])
        if (
            status not in {"success", "error"}
            or isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or not 1 <= attempts <= 8
            or not isinstance(unsupported, list)
            or any(not isinstance(value, str) or not value for value in unsupported)
            or len(set(unsupported)) != len(unsupported)
            or not isinstance(ignored, list)
            or any(not isinstance(value, str) or not value for value in ignored)
            or len(set(ignored)) != len(ignored)
            or not isinstance(entry.get("requested_model"), str)
            or not entry["requested_model"]
        ):
            raise ScoringError(
                f"proxy JSONL line {line_number} has invalid attempt evidence"
            )
        total_upstream_attempts += attempts
        response_id = entry.get("response_id")
        if status == "success":
            if (
                not isinstance(response_id, str)
                or not response_id
                or response_id in proxy_response_ids
                or not isinstance(entry.get("actual_model"), str)
                or not entry["actual_model"]
                ):
                    raise ScoringError(
                    f"proxy JSONL line {line_number} is not a successful GPT-5.5 request"
                    )
            proxy_response_ids.add(response_id)
            by_response_id.setdefault(response_id, []).append(entry)
        else:
            if response_id is not None or not isinstance(entry.get("error"), str):
                raise ScoringError(
                    f"proxy JSONL line {line_number} has invalid terminal error"
                )
            normalized_errors.append({
                "attempts": attempts,
                "unsupported_parameters": unsupported,
                "ignored_client_parameters": ignored,
                "error": entry["error"],
                "http_status": entry.get("http_status"),
                "http_request_id": entry.get("http_request_id"),
            })

    matched_entries = []
    recorded_ids: set[str] = set()
    request_attempts = 0
    unlinked_transport_errors = 0
    for record in records:
        if record.get("question_id") == "_build_stats" or "judge_usage" not in record:
            continue
        usage = normalize_judge_usage(record["judge_usage"])
        prompt_tokens = 0
        completion_tokens = 0
        request_attempts += usage["request_attempts"]
        for detail in usage["request_attempt_details"]:
            response_id = detail["response_id"]
            if response_id is None:
                if detail["status"] != "error":
                    raise ScoringError(
                        f"{record['question_id']} has an unlinked non-transport attempt"
                    )
                unlinked_transport_errors += 1
                continue
            if response_id in recorded_ids:
                raise ScoringError("judge response ids are duplicated")
            candidates = by_response_id.get(response_id, [])
            if len(candidates) != 1:
                raise ScoringError(
                    f"judge response id {response_id!r} has {len(candidates)} "
                    "matching proxy records"
                )
            entry = candidates[0]
            if (
                entry.get("status") != "success"
                or entry.get("requested_model") != "gpt-5.5"
                or entry.get("actual_model") != "gpt-5.5"
            ):
                raise ScoringError(
                    f"judge response id {response_id!r} is not a successful "
                    "GPT-5.5 proxy request"
                )
            proxy_usage = entry.get("usage")
            if not isinstance(proxy_usage, dict):
                raise ScoringError(
                    f"judge response id {response_id!r} has no proxy usage"
                )
            proxy_prompt_tokens = proxy_usage.get("prompt_tokens")
            proxy_completion_tokens = proxy_usage.get("completion_tokens")
            for name, value in (
                ("prompt_tokens", proxy_prompt_tokens),
                ("completion_tokens", proxy_completion_tokens),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ScoringError(
                        f"judge response id {response_id!r} has invalid proxy {name}"
                    )
            prompt_tokens += proxy_prompt_tokens
            completion_tokens += proxy_completion_tokens
            if (
                detail["prompt_tokens"] != proxy_prompt_tokens
                or detail["completion_tokens"] != proxy_completion_tokens
            ):
                raise ScoringError(
                    f"judge response id {response_id!r} token evidence mismatch"
                )
            recorded_ids.add(response_id)
            matched_entries.append({
                "response_id": response_id,
                "requested_model": entry["requested_model"],
                "actual_model": entry["actual_model"],
                "http_request_id": entry.get("http_request_id"),
                "upstream_attempts": entry["attempts"],
                "unsupported_parameters": entry["unsupported_parameters"],
                "ignored_client_parameters": entry.get(
                    "ignored_client_parameters", []
                ),
                "prompt_tokens": proxy_prompt_tokens,
                "completion_tokens": proxy_completion_tokens,
            })
        if prompt_tokens != usage["prompt_tokens"]:
            raise ScoringError(
                f"{record['question_id']} proxy prompt token count mismatch"
            )
        if completion_tokens != usage["completion_tokens"]:
            raise ScoringError(
                f"{record['question_id']} proxy completion token count mismatch"
            )
    if not matched_entries:
        raise ScoringError("secondary judge output has no proxy response evidence")
    matched_entries.sort(key=lambda entry: entry["response_id"])
    matched_upstream_attempts = sum(
        entry["upstream_attempts"] for entry in matched_entries
    )
    unsupported = sorted({
        value
        for entry in matched_entries
        for value in entry["unsupported_parameters"]
    })
    ignored = sorted({
        value
        for entry in matched_entries
        for value in entry["ignored_client_parameters"]
    })
    return {
        "path": str(proxy_log),
        "log_prefix_bytes": prefix_bytes,
        "log_prefix_sha256": prefix_sha256,
        "evidence_scope": "response_id_linked_except_transport_errors",
        "content_hash_linkage": "unavailable_not_persisted",
        "prompt_response_content_verified": False,
        "logical_judge_calls": sum(
            normalize_judge_usage(record["judge_usage"])["logical_judge_calls"]
            for record in records
            if record.get("question_id") != "_build_stats"
            and "judge_usage" in record
        ),
        "request_attempts": request_attempts,
        "client_physical_http_attempts": request_attempts,
        "matched_proxy_requests": len(matched_entries),
        "matched_proxy_upstream_attempts": matched_upstream_attempts,
        "matched_proxy_internal_retries": (
            matched_upstream_attempts - len(matched_entries)
        ),
        "matched_response_ids": len(matched_entries),
        "unlinked_transport_errors": unlinked_transport_errors,
        "shared_proxy_entries": len(proxy_entries),
        "shared_proxy_upstream_attempts": total_upstream_attempts,
        "shared_proxy_terminal_errors": len(normalized_errors),
        "shared_proxy_error_evidence_sha256": sha256_json(normalized_errors),
        "requested_models": ["gpt-5.5"],
        "actual_models": ["gpt-5.5"],
        "unsupported_parameters": unsupported,
        "ignored_client_parameters": ignored,
        "requested_output_limit_enforced": (
            "max_output_tokens" not in set(unsupported) | set(ignored)
        ),
        "prompt_tokens": sum(entry["prompt_tokens"] for entry in matched_entries),
        "completion_tokens": sum(
            entry["completion_tokens"] for entry in matched_entries
        ),
        "evidence_sha256": sha256_json(matched_entries),
    }


def _base_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in DERIVED_QUESTION_KEYS}


def merge_partial(
    selected: list[dict[str, Any]],
    partial: dict[str, Any],
    benchmark: str,
    profile_name: str,
    source_sha256: str,
    selected_sha256: str,
    current_code_hashes: dict[str, str],
    provenance: dict[str, Any],
    proxy_log: Path | None,
    run_key: str,
    ledger_path: Path,
    generation_id: str,
    openrouter_gateway: dict[str, Any] | None,
    flex_gateway_contract: dict[str, Any] | None,
    flex_evidence_root: str | None,
    transport_contract: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta = partial.get("meta")
    old_records = partial.get("results")
    if not isinstance(meta, dict) or not isinstance(old_records, list):
        raise ScoringError("partial output must contain meta and results")
    profile = judge_profile(benchmark, profile_name)
    if isinstance(openrouter_gateway, dict):
        profile["base_url"] = str(openrouter_gateway["base_url"])
    required_meta = {
        "schema_version": 3,
        "benchmark": benchmark,
        "scope": SCOPES[benchmark],
        "method": "NativeMem-v8.8+calendar",
        "backbone": "gpt-5.5",
        "judge_profile": profile["id"],
        "judge_provider": profile["provider"],
        "judge_requested_model": profile["requested_model"],
        "judge_base_url": profile["base_url"],
        "comparison_status": profile["comparison_status"],
        "comparable_to_published_primary": profile[
            "comparable_to_published_primary"
        ],
        "comparison_note": profile["note"],
        "source_input_sha256": source_sha256,
        "source_input": str(Path(provenance["run_dir"]) / (
            "questions_all.json" if benchmark in LOCOMO_BENCHMARKS
            else "evaluation_input.json"
        )),
        "selected_input_sha256": selected_sha256,
        "source_dataset_sha256": dataset_sha256(benchmark),
        "source_audit": provenance["audit_path"],
        "source_audit_sha256": provenance["audit_sha256"],
        "source_run_dir": provenance["run_dir"],
        "source_manifest": provenance["manifest_path"],
        "source_manifest_sha256": provenance["manifest_sha256"],
        "upstream_proxy_evidence": provenance["upstream_proxy_evidence"],
        "judge_proxy_log": str(proxy_log) if proxy_log else None,
        "metrics": metrics_for_benchmark(benchmark),
        "code_hashes": current_code_hashes,
        "run_key": run_key,
        "generation_id": generation_id,
        "transport_contract": transport_contract,
        "openrouter_gateway": openrouter_gateway,
        "openai_flex_gateway": flex_gateway_contract,
        "flex_evidence_root": flex_evidence_root,
    }
    mismatches = {
        key: {"expected": value, "actual": meta.get(key)}
        for key, value in required_meta.items()
        if meta.get(key) != value
    }
    if mismatches:
        raise ScoringError(f"partial output metadata mismatch: {mismatches}")
    ledger = meta.get("attempt_ledger")
    if not isinstance(ledger, dict) or ledger.get("path") != str(ledger_path):
        raise ScoringError("partial output attempt ledger differs")
    if len(old_records) != len(selected):
        raise ScoringError("partial output record count differs from selected input")

    merged = copy.deepcopy(selected)
    for index, (base, old) in enumerate(zip(merged, old_records)):
        if not isinstance(old, dict) or _base_record(old) != base:
            raise ScoringError(f"partial output source record differs at index {index}")
        if base.get("question_id") == "_build_stats":
            continue
        present = DERIVED_QUESTION_KEYS.intersection(old)
        if benchmark == "locomo-cat5" and "lexical_refusal" not in present:
            raise ScoringError(
                f"partial output lacks category-5 refusal diagnostic at index {index}"
            )
        if any(metric != "judge" for metric in METRICS) and "lexical" not in present:
            raise ScoringError(f"partial output lacks lexical metrics at index {index}")
        judge_present = {"judge_score", "judge_raw", "judge_usage"}.intersection(old)
        if judge_present and judge_present != {"judge_score", "judge_raw", "judge_usage"}:
            raise ScoringError(f"partial output has incomplete judge fields at index {index}")
        for key in present:
            base[key] = copy.deepcopy(old[key])
        if judge_present:
            validate_judge_result(
                base["judge_score"], base["judge_raw"], base["judge_usage"],
                benchmark, profile_name, str(base["question_id"]),
            )
    return merged, copy.deepcopy(meta)


def _evaluation_api():
    from src.evaluation import evaluate

    return evaluate


def recompute_aggregate(records: list[dict[str, Any]], benchmark: str) -> dict[str, Any]:
    evaluate = _evaluation_api()
    if benchmark == "locomo":
        aggregate = evaluate.aggregate_locomo(records)
    elif benchmark == "locomo-cat5":
        questions = [
            record for record in records
            if record.get("question_id") != "_build_stats"
        ]
        scores = [record["judge_score"] for record in questions if "judge_score" in record]
        abstention_scores = [
            record["judge_score"]
            for record in questions
            if record.get("cat5_abstention") is True and "judge_score" in record
        ]
        explicit_answer_scores = [
            record["judge_score"]
            for record in questions
            if record.get("cat5_abstention") is False and "judge_score" in record
        ]
        lexical = [
            record["lexical_refusal"]
            for record in questions
            if isinstance(record.get("lexical_refusal"), bool)
        ]
        aggregate = {
            "n": len(questions),
            "n_abstention": sum(
                bool(record.get("cat5_abstention")) for record in questions
            ),
            "n_explicit_answer": sum(
                not bool(record.get("cat5_abstention")) for record in questions
            ),
            "semantic_judge_accuracy": (
                sum(scores) / len(scores) if scores else None
            ),
            "semantic_abstention_judge_accuracy": (
                sum(abstention_scores) / len(abstention_scores)
                if abstention_scores else None
            ),
            "explicit_answer_judge_accuracy": (
                sum(explicit_answer_scores) / len(explicit_answer_scores)
                if explicit_answer_scores else None
            ),
            "lexical_refusal_rate_diagnostic": (
                sum(lexical) / len(lexical) if lexical else None
            ),
            "lexical_refusal_detector": {
                "status": "diagnostic_only_not_semantic_correctness",
                "normalization": "NFKC casefold and whitespace collapse",
                "phrases": list(CAT5_REFUSAL_PHRASES),
            },
            "excluded_from_cat1_4": True,
        }
    else:
        # aggregate_longmemeval expects only QA records; independent-history
        # accounting records are retained solely for aggregate_efficiency.
        questions = [
            record for record in records
            if record.get("question_id") != "_build_stats"
        ]
        aggregate = evaluate.aggregate_longmemeval(questions)
    aggregate["efficiency"] = evaluate.aggregate_efficiency(records)
    return aggregate


def add_lexical_metrics(records: list[dict[str, Any]]) -> None:
    evaluate = _evaluation_api()
    evaluate.run_lexical(records, METRICS)


def lexical_refusal_diagnostic(answer: object) -> bool:
    """Return a frozen phrase match; this is not a semantic correctness score."""
    normalized = unicodedata.normalize("NFKC", str(answer)).casefold()
    normalized = " ".join(normalized.split())
    return any(phrase in normalized for phrase in CAT5_REFUSAL_PHRASES)


def add_cat5_refusal_diagnostics(records: list[dict[str, Any]]) -> None:
    for record in records:
        if record.get("question_id") == "_build_stats":
            continue
        record["lexical_refusal"] = (
            lexical_refusal_diagnostic(record.get("answer"))
            if record.get("cat5_abstention") is True
            else None
        )


def _progress_meta(
    meta: dict[str, Any], records: list[dict[str, Any]], benchmark: str,
    status: str,
) -> dict[str, Any]:
    questions = [record for record in records if record.get("question_id") != "_build_stats"]
    scored = [record for record in questions if "judge_score" in record]
    updated = copy.deepcopy(meta)
    updated.update({
        "status": status,
        "updated_at": utc_now(),
        "question_count": len(questions),
        "completed_questions": len(scored),
        "correct_questions": sum(record.get("judge_score", 0) for record in scored),
        "aggregate": recompute_aggregate(records, benchmark),
    })
    ledger = updated.get("attempt_ledger")
    run_key = updated.get("run_key")
    if isinstance(ledger, dict) and isinstance(ledger.get("path"), str):
        updated["attempt_ledger"] = ledger_report(Path(ledger["path"]), run_key)
    if status == "complete":
        updated["finished_at"] = updated["updated_at"]
        updated.pop("last_error", None)
    return updated


def _write_progress(
    output: Path, records: list[dict[str, Any]], meta: dict[str, Any],
    benchmark: str, status: str,
) -> dict[str, Any]:
    updated = _progress_meta(meta, records, benchmark, status)
    atomic_json(output, {"meta": updated, "results": records})
    return updated


def _replay_ledger_results(
    records: list[dict[str, Any]], events: list[dict[str, Any]], run_key: str,
    benchmark: str, profile_name: str,
) -> None:
    """Restore committed judgments that are newer than the JSON snapshot."""
    by_question = {
        str(record.get("question_id")): record
        for record in records if record.get("question_id") != "_build_stats"
    }
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("run_key") == run_key and event.get("event") == "judge_result":
            question_id = event.get("question_id")
            if not isinstance(question_id, str) or question_id not in by_question:
                raise ScoringError("attempt ledger result has an unknown question id")
            latest[question_id] = event
    for question_id, event in latest.items():
        usage = normalize_judge_usage(event.get("usage"))
        validate_judge_result(
            event.get("score"), event.get("raw"), usage,
            benchmark, profile_name, question_id,
        )
        record = by_question[question_id]
        committed = {
            "judge_score": event["score"],
            "judge_raw": event["raw"],
            "judge_usage": usage,
        }
        existing = {key: record.get(key) for key in committed if key in record}
        if existing and existing != committed:
            raise ScoringError(
                f"attempt ledger and evaluation snapshot differ for {question_id}"
            )
        record.update(committed)


def _pending_failure_events(
    events: list[dict[str, Any]], run_key: str, question_id: str,
) -> list[dict[str, Any]]:
    failures = [
        event for event in events
        if event.get("run_key") == run_key
        and event.get("question_id") == question_id
        and event.get("event") == "judge_failure"
    ]
    consumed: set[str] = set()
    for event in events:
        if (
            event.get("run_key") == run_key
            and event.get("question_id") == question_id
            and event.get("event") == "judge_result"
        ):
            values = event.get("consumed_failure_event_ids", [])
            if not isinstance(values, list) or any(
                not isinstance(value, str) for value in values
            ):
                raise ScoringError("attempt ledger has invalid consumed failures")
            consumed.update(values)
    return [event for event in failures if event["event_id"] not in consumed]


def _score_selected_records_unlocked(
    *,
    selected: list[dict[str, Any]],
    benchmark: str,
    profile_name: str,
    source_input: Path,
    source_audit: Path,
    output: Path,
    judge_fn: Callable[..., tuple[int, str, dict[str, Any]]],
    proxy_log: Path | None = None,
    save_every: int = 20,
    restart: bool = False,
    openrouter_gateway: dict[str, Any] | None = None,
    flex_gateway_contract: dict[str, Any] | None = None,
    formal_transport_contract: bool = False,
) -> dict[str, Any]:
    """Score validated records, resuming a compatible atomic output if present."""
    if save_every <= 0:
        raise ScoringError("save_every must be positive")
    source_input = source_input.expanduser().resolve()
    source_audit = source_audit.expanduser().resolve()
    output = output.expanduser().resolve()
    flex_evidence_root = (
        str(Path(f"{output}.provider-evidence").resolve())
        if profile_name == "secondary" and formal_transport_contract
        else None
    )
    ledger_path = attempt_ledger_path(output)
    if restart:
        ensure_distinct_paths({
            "evaluation_output": output,
            "source_input": source_input,
            "source_audit": source_audit,
            "source_manifest": source_input.parent / "run_manifest.json",
            "proxy_log": (
                proxy_log.expanduser().resolve() if proxy_log
                else recorded_proxy_path(source_audit)
            ),
            "attempt_ledger": ledger_path,
            "evaluation_lock": Path(f"{output}.lock"),
        })
        invalidate_score_audits(output)
    provenance = validate_upstream_audit(
        benchmark, source_input, source_audit
    )
    expected_proxy_log = Path(provenance["proxy_log_path"]).resolve()
    if profile_name == "secondary" and formal_transport_contract:
        if proxy_log is not None:
            raise ScoringError(
                "formal secondary scoring does not accept a shared --proxy-log"
            )
        if not isinstance(flex_gateway_contract, dict):
            raise ScoringError(
                "formal secondary scoring lacks a dynamic Flex gateway contract"
            )
        try:
            flex_evidence.validate_recorded_contract(flex_gateway_contract)
        except flex_evidence.EvidenceError as exc:
            raise ScoringError(str(exc)) from exc
    elif profile_name == "secondary":
        if proxy_log is None:
            proxy_log = expected_proxy_log
        else:
            proxy_log = proxy_log.expanduser().resolve()
        if proxy_log != expected_proxy_log:
            raise ScoringError(
                "secondary judge proxy log differs from the upstream shared log"
            )
    elif proxy_log is not None:
        raise ScoringError("--proxy-log is valid only for the secondary judge")
    ensure_distinct_paths({
        "evaluation_output": output,
        "source_input": source_input,
        "source_audit": source_audit,
        "source_manifest": Path(provenance["manifest_path"]),
        "upstream_proxy_log": expected_proxy_log,
        "attempt_ledger": ledger_path,
        "evaluation_lock": Path(f"{output}.lock"),
    })
    source_sha256 = sha256_file(source_input)
    selected_sha256 = sha256_json(selected)
    current_code_hashes = code_hashes()
    profile = judge_profile(benchmark, profile_name)
    if profile_name == "primary" and formal_transport_contract:
        if not isinstance(openrouter_gateway, dict):
            raise ScoringError(
                "formal primary scoring lacks OpenRouter gateway evidence"
            )
        try:
            openrouter_gateway_evidence.validate_binding(
                openrouter_gateway,
                result_root=Path(str(openrouter_gateway.get("result_root", ""))),
                base_url=str(openrouter_gateway.get("base_url", "")),
            )
        except openrouter_gateway_evidence.GatewayEvidenceError as exc:
            raise ScoringError(str(exc)) from exc
        profile["base_url"] = str(openrouter_gateway["base_url"])
        transport_contract = "marked_openrouter_gateway"
    elif profile_name == "primary":
        transport_contract = "injected_test_judge"
    else:
        if openrouter_gateway is not None:
            raise ScoringError("secondary scoring cannot bind an OpenRouter gateway")
        if formal_transport_contract:
            if not isinstance(flex_gateway_contract, dict):
                raise ScoringError("secondary scoring lacks Flex gateway evidence")
            profile["base_url"] = str(flex_gateway_contract["base_url"])
            transport_contract = "openai_gpt55_flex_gateway"
        else:
            transport_contract = "secondary_local_proxy"
    base_run_key = sha256_json({
        "benchmark": benchmark,
        "profile": profile_name,
        "source_input_sha256": source_sha256,
        "selected_input_sha256": selected_sha256,
        "source_manifest_sha256": provenance["manifest_sha256"],
        "code_hashes": current_code_hashes,
        "transport_contract": transport_contract,
        "openrouter_gateway": gateway_start_binding(openrouter_gateway),
        "openai_flex_gateway": flex_gateway_start_binding(flex_gateway_contract),
        "flex_evidence_root": flex_evidence_root,
    })
    if output.exists() and not restart:
        existing_payload = read_json(output)
        existing_meta = (
            existing_payload.get("meta") if isinstance(existing_payload, dict) else None
        )
        generation_id = (
            existing_meta.get("generation_id")
            if isinstance(existing_meta, dict) else None
        )
        if not isinstance(generation_id, str) or not generation_id:
            raise ScoringError("partial output has no durable generation id")
    else:
        generation_id = str(uuid.uuid4())
    run_key = sha256_json({
        "base_run_key": base_run_key,
        "generation_id": generation_id,
    })
    invocation_id = str(uuid.uuid4())
    append_attempt_event(ledger_path, {
        "event": "invocation_started",
        "run_key": run_key,
        "generation_id": generation_id,
        "invocation_id": invocation_id,
        "benchmark": benchmark,
        "judge_profile": profile["id"],
        "judge_provider": profile["provider"],
        "restart": restart,
    })
    meta: dict[str, Any] = {
        "schema_version": 3,
        "status": "running",
        "benchmark": benchmark,
        "scope": SCOPES[benchmark],
        "method": "NativeMem-v8.8+calendar",
        "backbone": "gpt-5.5",
        "judge_profile": profile["id"],
        "judge_provider": profile["provider"],
        "judge_requested_model": profile["requested_model"],
        "judge_base_url": profile["base_url"],
        "comparison_status": profile["comparison_status"],
        "comparable_to_published_primary": profile[
            "comparable_to_published_primary"
        ],
        "comparison_note": profile["note"],
        "metrics": metrics_for_benchmark(benchmark),
        "source_input": str(source_input.resolve()),
        "source_input_sha256": source_sha256,
        "selected_input_sha256": selected_sha256,
        "source_dataset_sha256": dataset_sha256(benchmark),
        "source_audit": provenance["audit_path"],
        "source_audit_sha256": provenance["audit_sha256"],
        "source_run_dir": provenance["run_dir"],
        "source_manifest": provenance["manifest_path"],
        "source_manifest_sha256": provenance["manifest_sha256"],
        "upstream_proxy_evidence": provenance["upstream_proxy_evidence"],
        "judge_proxy_log": str(proxy_log) if proxy_log else None,
        "transport_contract": transport_contract,
        "openrouter_gateway": openrouter_gateway,
        "openai_flex_gateway": flex_gateway_contract,
        "flex_evidence_root": flex_evidence_root,
        "code_hashes": current_code_hashes,
        "run_key": run_key,
        "generation_id": generation_id,
        "attempt_ledger": ledger_report(ledger_path, run_key),
        "created_at": utc_now(),
    }
    records = copy.deepcopy(selected)
    if output.exists() and not restart:
        records, meta = merge_partial(
            selected, read_json(output), benchmark, profile_name,
            source_sha256, selected_sha256, current_code_hashes,
            provenance, proxy_log, run_key, ledger_path, generation_id,
            openrouter_gateway, flex_gateway_contract, flex_evidence_root,
            transport_contract,
        )

    ledger_events = read_attempt_ledger(ledger_path)
    reconcile_attempt_ledger(
        ledger_events,
        run_key,
        benchmark=benchmark,
        profile_name=profile_name,
    )
    _replay_ledger_results(
        records, ledger_events, run_key, benchmark, profile_name
    )
    add_lexical_metrics(records)
    if benchmark == "locomo-cat5":
        add_cat5_refusal_diagnostics(records)
    # Any subsequent write changes the bytes authenticated by a prior audit.
    # Invalidate all known reports before the first atomic replacement, both
    # for --restart and for an ordinary resume/revalidation invocation.
    invalidate_score_audits(output)
    meta = _write_progress(output, records, meta, benchmark, "running")
    pending = [
        record for record in records
        if record.get("question_id") != "_build_stats" and "judge_score" not in record
    ]
    completed_this_invocation = 0
    current_question_id = None
    try:
        for record in pending:
            current_question_id = str(record["question_id"])
            prior_failures = _pending_failure_events(
                ledger_events, run_key, current_question_id
            )
            current_observed_event_ids: list[str] = []

            def persist_observed(event):
                persisted = append_attempt_event(ledger_path, {
                    **event,
                    "run_key": run_key,
                    "invocation_id": invocation_id,
                    "question_id": current_question_id,
                    "benchmark": benchmark,
                    "judge_profile": profile["id"],
                })
                ledger_events.append(persisted)
                current_observed_event_ids.append(persisted["event_id"])

            from src.evaluation.llm_clients import observe_attempts

            try:
                with observe_attempts(persist_observed):
                    if benchmark == "locomo":
                        score, raw, current_usage = judge_fn(
                            record["question"], record.get("gold", ""), record["answer"],
                            category=record.get("category"),
                        )
                    elif benchmark == "locomo-cat5":
                        # distractor is intentionally not part of this call.
                        score, raw, current_usage = judge_fn(
                            record["question"],
                            record["gold"],
                            record["answer"],
                            abstention=record["cat5_abstention"],
                        )
                    else:
                        score, raw, current_usage = judge_fn(
                            record.get("question_type", "multi-session"),
                            record["question"], record.get("gold", ""), record["answer"],
                            abstention=bool(record.get("abstention")),
                        )
                current_usage = normalize_judge_usage(current_usage)
                usage = merge_judge_usages(
                    *(event["usage"] for event in prior_failures), current_usage
                )
                validate_judge_result(
                    score, raw, usage, benchmark, profile_name, current_question_id
                )
            except BaseException as exc:
                failed_usage = getattr(exc, "usage", None)
                failure_event = append_attempt_event(ledger_path, {
                    "event": "judge_failure",
                    "run_key": run_key,
                    "invocation_id": invocation_id,
                    "question_id": current_question_id,
                    "benchmark": benchmark,
                    "judge_profile": profile["id"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                    "observed_event_ids": current_observed_event_ids,
                    "current_usage": (
                        normalize_judge_usage(failed_usage)
                        if isinstance(failed_usage, dict) else None
                    ),
                    "usage": (
                        normalize_judge_usage(failed_usage)
                        if isinstance(failed_usage, dict) else None
                    ),
                })
                ledger_events.append(failure_event)
                raise
            result_event = append_attempt_event(ledger_path, {
                "event": "judge_result",
                "run_key": run_key,
                "invocation_id": invocation_id,
                "question_id": current_question_id,
                "benchmark": benchmark,
                "judge_profile": profile["id"],
                "score": score,
                "raw": raw,
                "usage": usage,
                "current_usage": current_usage,
                "observed_event_ids": current_observed_event_ids,
                "consumed_failure_event_ids": [
                    event["event_id"] for event in prior_failures
                ],
            })
            ledger_events.append(result_event)
            record["judge_score"] = score
            record["judge_raw"] = raw
            record["judge_usage"] = usage
            completed_this_invocation += 1
            if completed_this_invocation % save_every == 0:
                meta = _write_progress(output, records, meta, benchmark, "running")
                print(
                    f"  judge {meta['completed_questions']}/{meta['question_count']}"
                )
        if profile_name == "secondary":
            current_question_id = "_proxy_evidence"
            if transport_contract == "openai_gpt55_flex_gateway":
                meta["judge_proxy_evidence"] = None
                meta["judge_gateway_evidence"] = None
            else:
                meta["judge_proxy_evidence"] = validate_secondary_proxy_evidence(
                    records, proxy_log
                )
                meta["judge_gateway_evidence"] = None
        else:
            meta["judge_proxy_evidence"] = None
            if transport_contract == "marked_openrouter_gateway":
                try:
                    final_binding = openrouter_gateway_evidence.finalize_binding(
                        openrouter_gateway
                    )
                    response_ids = [
                        response_id
                        for record in records
                        if record.get("question_id") != "_build_stats"
                        for response_id in normalize_judge_usage(
                            record["judge_usage"]
                        )["response_ids"]
                    ]
                    meta["openrouter_gateway"] = final_binding
                    meta["judge_gateway_evidence"] = (
                        openrouter_gateway_evidence.verify_response_ids(
                            final_binding, response_ids
                        )
                    )
                except openrouter_gateway_evidence.GatewayEvidenceError as exc:
                    raise ScoringError(str(exc)) from exc
            else:
                meta["judge_gateway_evidence"] = {
                    "status": "synthetic_injected_judge_only"
                }
    except BaseException as exc:
        meta["last_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "question_id": current_question_id,
            "at": utc_now(),
        }
        failed_usage = getattr(exc, "usage", None)
        if isinstance(failed_usage, dict):
            meta["last_error"]["request_usage"] = copy.deepcopy(failed_usage)
        _write_progress(output, records, meta, benchmark, "partial")
        raise

    append_attempt_event(ledger_path, {
        "event": "invocation_finished",
        "run_key": run_key,
        "invocation_id": invocation_id,
        "benchmark": benchmark,
        "judge_profile": profile["id"],
        "status": "complete",
    })
    meta = _write_progress(output, records, meta, benchmark, "complete")
    return {"meta": meta, "results": records}


def score_selected_records(**kwargs) -> dict[str, Any]:
    """Programmatic scoring entry point with the same lock as the CLI."""
    output = Path(kwargs["output"]).expanduser().resolve()
    with exclusive_output_lock(output):
        return _score_selected_records_unlocked(**kwargs)


def _judge_function(benchmark: str):
    from src.evaluation.judges import (
        judge_locomo,
        judge_locomo_cat5,
        judge_longmemeval,
    )

    if benchmark == "locomo":
        return judge_locomo
    if benchmark == "locomo-cat5":
        return judge_locomo_cat5
    return judge_longmemeval


def _next_secondary_flex_evidence_dir(output: Path) -> tuple[Path, int]:
    root = Path(f"{output.expanduser().resolve()}.provider-evidence")
    root.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in root.iterdir():
        match = re.fullmatch(r"invocation-(\d{4})", path.name)
        if match:
            numbers.append(int(match.group(1)))
    number = max(numbers, default=0) + 1
    return root / f"invocation-{number:04d}", number


def _attach_secondary_flex_evidence(output: Path) -> dict[str, Any] | None:
    if not output.is_file():
        return None
    payload = read_json(output)
    if not isinstance(payload, dict) or not isinstance(payload.get("meta"), dict):
        raise ScoringError("secondary output is invalid before Flex binding")
    meta = payload["meta"]
    evidence_root = Path(f"{output.resolve()}.provider-evidence")
    meta["flex_evidence_root"] = str(evidence_root)
    if meta.get("status") == "complete":
        report = validate_secondary_flex_evidence(payload.get("results", []), evidence_root)
        meta["judge_gateway_evidence"] = report
    atomic_json(output, payload)
    return meta.get("judge_gateway_evidence")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=sorted(SCOPES))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judge", required=True, choices=sorted(JUDGE_PROFILES))
    parser.add_argument("--gateway-root", type=Path)
    parser.add_argument("--allow-model-requests", action="store_true")
    parser.add_argument("--openrouter-gateway-root", type=Path)
    parser.add_argument("--openrouter-gateway-base-url")
    parser.add_argument(
        "--proxy-log", type=Path,
        help="shared GPT-5.5 proxy JSONL; secondary only (defaults to source audit)",
    )
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument(
        "--restart", action="store_true",
        help="ignore an existing output instead of resuming it",
    )
    args = parser.parse_args(argv)
    if args.benchmark in LOCOMO_BENCHMARKS:
        parser.error(
            "LoCoMo scoring is locked to scripts/eval_full.py; "
            "score_v88_gpt55_benchmarks.py may not score LoCoMo"
        )
    if not args.allow_model_requests:
        parser.error("formal scoring requires --allow-model-requests")
    source_input = args.input.expanduser().resolve()
    source_audit = args.source_audit.expanduser().resolve()
    output = args.output.expanduser().resolve()
    try:
        explicit_proxy = args.proxy_log.expanduser().resolve() if args.proxy_log else None
        ensure_distinct_paths({
            "evaluation_output": output,
            "source_input": source_input,
            "source_audit": source_audit,
            "source_manifest": source_input.parent / "run_manifest.json",
            "proxy_log": explicit_proxy or recorded_proxy_path(source_audit),
            "attempt_ledger": attempt_ledger_path(output),
            "evaluation_lock": Path(f"{output}.lock"),
        })
        if args.restart:
            invalidate_score_audits(output)
        openrouter_binding = None
        flex_invocation = None
        flex_gateway_contract = None
        if args.judge == "primary":
            if args.gateway_root is not None:
                raise ScoringError(
                    "--gateway-root is valid only for secondary GPT-5.5 judging"
                )
            if (
                args.openrouter_gateway_root is None
                or not args.openrouter_gateway_base_url
            ):
                raise ScoringError(
                    "primary formal scoring requires --openrouter-gateway-root "
                    "and --openrouter-gateway-base-url"
                )
            try:
                live_binding = openrouter_gateway_evidence.capture_binding(
                    args.openrouter_gateway_root,
                    base_url=args.openrouter_gateway_base_url,
                )
                if output.exists() and not args.restart:
                    existing = read_json(output)
                    stored = existing.get("meta", {}).get("openrouter_gateway")
                    if not isinstance(stored, dict):
                        raise ScoringError(
                            "partial primary output lacks OpenRouter gateway binding"
                        )
                    openrouter_gateway_evidence.validate_binding(
                        stored,
                        result_root=args.openrouter_gateway_root,
                        base_url=args.openrouter_gateway_base_url,
                    )
                    openrouter_binding = stored
                else:
                    openrouter_binding = live_binding
            except openrouter_gateway_evidence.GatewayEvidenceError as exc:
                raise ScoringError(str(exc)) from exc
        else:
            if args.openrouter_gateway_root or args.openrouter_gateway_base_url:
                raise ScoringError(
                    "OpenRouter gateway arguments are valid only for primary judging"
                )
            if args.gateway_root is None:
                raise ScoringError(
                    "secondary formal scoring requires --gateway-root"
                )
            if args.proxy_log is not None:
                raise ScoringError(
                    "secondary formal scoring does not accept --proxy-log"
                )
            evidence_dir, invocation_number = _next_secondary_flex_evidence_dir(
                output
            )
            try:
                flex_invocation = flex_evidence.begin_child_invocation(
                    args.gateway_root,
                    evidence_dir,
                    run_id=(
                        f"score-{args.benchmark}-secondary-"
                        f"{invocation_number:04d}-{uuid.uuid4().hex[:12]}"
                    ),
                )
            except flex_evidence.EvidenceError as exc:
                raise ScoringError(str(exc)) from exc
            flex_gateway_contract = flex_invocation.contract
        configure_judge(
            args.judge,
            openrouter_gateway=openrouter_binding,
            secondary_base_url=(
                flex_invocation.base_url if flex_invocation is not None else None
            ),
        )
        selected = select_official_records(
            _load_record_list(source_input), args.benchmark
        )
        try:
            result = score_selected_records(
                selected=selected,
                benchmark=args.benchmark,
                profile_name=args.judge,
                source_input=source_input,
                source_audit=source_audit,
                output=output,
                judge_fn=_judge_function(args.benchmark),
                proxy_log=None if args.judge == "secondary" else args.proxy_log,
                save_every=args.save_every,
                restart=args.restart,
                openrouter_gateway=openrouter_binding,
                flex_gateway_contract=flex_gateway_contract,
                formal_transport_contract=True,
            )
        finally:
            if flex_invocation is not None:
                try:
                    flex_invocation.finish()
                    _attach_secondary_flex_evidence(output)
                except flex_evidence.EvidenceError as exc:
                    raise ScoringError(str(exc)) from exc
        if args.judge == "secondary":
            result = read_json(output)
    except (ScoringError, OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result["meta"], indent=2, ensure_ascii=False))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
