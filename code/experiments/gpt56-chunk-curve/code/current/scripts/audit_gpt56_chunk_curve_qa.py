#!/usr/bin/env python3
"""Audit completed GPT-5.6 W32 screening QA artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from scripts import gpt56_chunk_curve_qa_contract as contract
from scripts import run_gpt56_chunk_curve as build_runner
from scripts import run_gpt56_chunk_curve_qa as qa_runner
from scripts import controlled_locomo_answer_contract as answer_contract
from scripts import audit_readonly_nativemem_control as readonly_auditor
from scripts import r115_beam_control_contract as beam_answer_contract
from scripts import readonly_nativemem_control as readonly_control
from src.evaluation.durable_model_ledger import read_ledger


class QAAuditError(RuntimeError):
    """A QA artifact does not satisfy the frozen screening contract."""


def _retryable_error_reason(*, status_code: object, error_text: str) -> str | None:
    if status_code != 500:
        return None
    lowered = error_text.lower()
    non_transient_markers = (
        "usage_limit_reached",
        "unauthorized",
        "authentication",
        "invalid_api_key",
        "http 401",
        "http 403",
        "http 429",
    )
    if any(marker in lowered for marker in non_transient_markers):
        return None
    if "upstream completed with empty output" in lowered:
        return contract.EMPTY_OUTPUT_RETRY_REASON
    if (
        "upstream stream failed:" in lowered
        and re.search(
            r'"code"\s*:\s*"server_error"',
            error_text,
            flags=re.IGNORECASE,
        )
        is not None
    ):
        return contract.TRANSIENT_PROVIDER_SERVER_ERROR_RETRY_REASON
    return None


def audit_question_records(
    *,
    records: Sequence[dict[str, Any]],
    qa_root: Path,
    formal: bool,
    bindings: Sequence[contract.RunBinding],
    validated_units: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Verify record identity, immutable memory, and every bound artifact hash."""

    if not records:
        raise QAAuditError("question records are empty")
    if len(bindings) != len(validated_units):
        raise QAAuditError("build bindings and validated units differ")
    binding_units: dict[str, tuple[contract.RunBinding, dict[str, Any]]] = {}
    for binding, unit in zip(bindings, validated_units, strict=True):
        if binding.run_id in binding_units:
            raise QAAuditError("build run IDs are duplicated")
        binding_units[binding.run_id] = (binding, unit)
    question_ids: set[str] = set()
    artifact_count = 0
    visible_tokens = 0
    retrieval_calls = 0
    answer_calls = 0
    for record in records:
        question_id = record.get("question_id")
        if not isinstance(question_id, str) or not question_id or question_id in question_ids:
            raise QAAuditError("question IDs are missing or duplicated")
        question_ids.add(question_id)
        bound = binding_units.get(str(record.get("run_id")))
        if bound is None:
            raise QAAuditError(f"question build binding is missing: {question_id}")
        binding, validated_unit = bound
        expected_questions = {
            str(item["question_id"]): item
            for item in validated_unit.get("questions", [])
            if isinstance(item, dict) and "question_id" in item
        }
        expected_question = expected_questions.get(
            str(record.get("original_question_id"))
        )
        if expected_question is None:
            raise QAAuditError(f"source question is missing: {question_id}")
        raw_question = str(
            expected_question.get(
                "question_text", expected_question.get("question", "")
            )
        )
        benchmark = str(record.get("benchmark"))
        model_question = raw_question
        if benchmark == "longmemeval-s":
            question_date = str(expected_question.get("question_date", ""))
            if not question_date:
                raise QAAuditError(
                    f"LongMemEval question date is missing: {question_id}"
                )
            model_question = (
                f"Current Date: {question_date}\nQuestion: {raw_question}"
            )
            if (
                record.get("question_date") != question_date
                or record.get("model_question") != model_question
            ):
                raise QAAuditError(
                    f"LongMemEval model-visible question differs: {question_id}"
                )
        if record.get("question") != raw_question:
            raise QAAuditError(f"raw evaluator question differs: {question_id}")
        result = record.get("result")
        if not isinstance(result, dict) or result.get("status") != "complete":
            raise QAAuditError(f"question result is incomplete: {question_id}")
        if result.get("question_id") != question_id:
            raise QAAuditError(f"question result identity differs: {question_id}")
        if (
            result.get("method") != contract.PROTOCOL_ID
            or result.get("condition") != "dual_source"
            or result.get("memory", {}).get("unchanged") is not True
        ):
            raise QAAuditError(f"question result contract differs: {question_id}")
        budget = result.get("budget", {})
        current_visible = budget.get("visible_tokens")
        if (
            budget.get("configured_tokens") != 20_000
            or not isinstance(current_visible, int)
            or isinstance(current_visible, bool)
            or not 0 <= current_visible <= 20_000
        ):
            raise QAAuditError(f"visible-token budget differs: {question_id}")
        visible_tokens += current_visible
        retrieval = result.get("retrieval", {})
        answer = result.get("answer", {})
        prompt = result.get("prompt", {})
        prompt_spec = contract.ANSWER_PROMPTS.get(str(record.get("benchmark")))
        if retrieval.get("model") != readonly_control.EXPECTED_MODEL:
            raise QAAuditError(f"retrieval model differs: {question_id}")
        if (
            answer.get("requested_model") != readonly_control.EXPECTED_MODEL
            or answer.get("response_model") != readonly_control.EXPECTED_MODEL
        ):
            raise QAAuditError(f"answer model differs: {question_id}")
        if (
            not isinstance(prompt_spec, dict)
            or prompt.get("kind") != prompt_spec.get("kind")
            or prompt.get("template_sha256")
            != prompt_spec.get("template_sha256")
        ):
            raise QAAuditError(f"answer prompt differs: {question_id}")
        if formal and answer.get("unsupported_parameters") not in ([], ()):
            raise QAAuditError(f"unsupported answer parameters: {question_id}")
        current_retrieval_calls = retrieval.get("model_calls")
        current_answer_calls = answer.get("logical_calls")
        if (
            not isinstance(current_retrieval_calls, int)
            or current_retrieval_calls < 1
            or current_retrieval_calls > 12
            or current_answer_calls != 1
        ):
            raise QAAuditError(f"model-call accounting differs: {question_id}")
        retrieval_calls += current_retrieval_calls
        answer_calls += current_answer_calls

        relative = qa_runner._question_root(  # noqa: SLF001
            qa_root,
            contract.RunBinding(
                run_id=str(record["run_id"]),
                row={
                    "tier": record["tier"],
                    "benchmark": record["benchmark"],
                    "unit_id": record["unit_id"],
                },
                run_dir=Path("."),
            ),
            record["original_question_id"],
        )
        attempts = [
            path
            for path in relative.glob("attempt-*/result.json")
            if path.is_file() and not path.is_symlink()
            and build_runner.read_json(path) == result
        ]
        if len(attempts) != 1:
            raise QAAuditError(f"question attempt binding differs: {question_id}")
        attempt_root = attempts[0].parent
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, dict):
            raise QAAuditError(f"question artifacts are missing: {question_id}")
        pairs = [
            (name, value)
            for name, value in artifacts.items()
            if not name.endswith("_sha256")
        ]
        for name, filename in pairs:
            expected_sha = artifacts.get(f"{name}_sha256")
            path = attempt_root / str(filename)
            if (
                Path(str(filename)).name != str(filename)
                or path.is_symlink()
                or not path.is_file()
                or not isinstance(expected_sha, str)
            ):
                raise QAAuditError(f"question artifact binding differs: {question_id}")
            if build_runner.sha256_file(path) != expected_sha:
                raise QAAuditError(f"artifact SHA-256 differs: {path}")
            artifact_count += 1
        evidence = expected_question.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        gold_source_ids = (
            readonly_auditor.expand_source_specs(evidence)
            if benchmark == "locomo" and isinstance(evidence, list)
            else []
        )
        prompt_template = (
            beam_answer_contract.ANSWER_PROMPT
            if benchmark == "beam-100k"
            else answer_contract.ANSWER_PROMPT
        )
        try:
            readonly_auditor.audit_question(
                artifact_dir=attempt_root,
                memory_root=binding.run_dir / "memory",
                method=contract.PROTOCOL_ID,
                condition="dual_source",
                question_id=question_id,
                question=model_question,
                gold_source_ids=gold_source_ids,
                source_recall_eligible=bool(gold_source_ids),
                formal=formal,
                turn_index=readonly_auditor.build_turn_index(
                    validated_unit["conversation"]
                ),
                answer_prompt_template=prompt_template,
                answer_prompt_kind=str(prompt_spec["kind"]),
                answer_prompt_template_sha256=str(
                    prompt_spec["template_sha256"]
                ),
            )
        except readonly_auditor.ReadOnlyAuditError as exc:
            raise QAAuditError(
                f"independent question audit failed: {question_id}: {exc}"
            ) from exc
    return {
        "question_count": len(records),
        "independently_reconstructed_questions": len(records),
        "artifact_count": artifact_count,
        "all_memory_unchanged": True,
        "visible_tokens": visible_tokens,
        "retrieval_model_calls": retrieval_calls,
        "answer_model_calls": answer_calls,
    }


def _attempt_request_ids(attempt_root: Path) -> set[str]:
    logical_ids: set[str] = set()
    for path in sorted(attempt_root.glob("calls/*.request.json")):
        value = _regular_json(path)
        logical_call_id = value.get("logical_call_id") if isinstance(value, dict) else None
        if isinstance(logical_call_id, str) and logical_call_id:
            logical_ids.add(logical_call_id)
    for path in sorted(attempt_root.glob("*.jsonl")):
        if path.is_symlink() or not path.is_file():
            raise QAAuditError(f"attempt ledger is unavailable: {path}")
        payload = path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            raise QAAuditError(f"attempt ledger has an incomplete line: {path}")
        for line in payload.decode("utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QAAuditError(f"attempt ledger is invalid JSON: {path}") from exc
            if not isinstance(value, dict):
                raise QAAuditError(f"attempt ledger event is not an object: {path}")
            logical_call_id = value.get("logical_call_id")
            if isinstance(logical_call_id, str) and logical_call_id:
                logical_ids.add(logical_call_id)
            run_id = value.get("run_id")
            if (
                isinstance(run_id, str)
                and run_id.endswith(":answer")
            ):
                logical_ids.add(run_id)
            nested = value.get("payload")
            nested_call_id = (
                nested.get("logical_call_id")
                if isinstance(nested, dict)
                else None
            )
            if isinstance(nested_call_id, str) and nested_call_id:
                logical_ids.add(nested_call_id)
    return logical_ids


def _attempt_manifests(
    *, qa_root: Path, qa_run_id: str
) -> dict[tuple[str, str], tuple[Path, dict[str, Any]]]:
    manifests: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(qa_root.glob("*/*/*/questions/*/attempt-*/attempt_manifest.json")):
        value = _regular_json(path)
        match = re.fullmatch(r"attempt-(\d{4})", path.parent.name)
        if not isinstance(value, dict) or match is None:
            raise QAAuditError(f"attempt manifest path or payload differs: {path}")
        attempt = int(match.group(1))
        benchmark = str(value.get("benchmark", ""))
        prompt_spec = contract.ANSWER_PROMPTS.get(benchmark)
        execution_run_id = value.get("execution_run_id")
        question_id = value.get("question_id")
        if (
            value.get("schema_version") != 1
            or not 1 <= attempt <= contract.QUESTION_MAX_ATTEMPTS
            or value.get("qa_run_id") != qa_run_id
            or value.get("attempt") != attempt
            or execution_run_id != f"{qa_run_id}:attempt-{attempt:04d}"
            or not isinstance(question_id, str)
            or not question_id
            or not isinstance(prompt_spec, dict)
            or value.get("answer_prompt_kind") != prompt_spec.get("kind")
            or value.get("answer_prompt_template_sha256")
            != prompt_spec.get("template_sha256")
        ):
            detail = (
                "attempt limit differs"
                if not 1 <= attempt <= contract.QUESTION_MAX_ATTEMPTS
                else "attempt manifest contract differs"
            )
            raise QAAuditError(f"{detail}: {path}")
        key = (str(execution_run_id), question_id)
        if key in manifests:
            raise QAAuditError("attempt manifest identity is duplicated")
        manifests[key] = (path.parent, value)
    return manifests


def _audit_retry_authorization(
    *,
    attempt_root: Path,
    manifest: dict[str, Any],
    formal: bool,
) -> dict[str, Any]:
    path = attempt_root / "retry_authorization.json"
    if path.is_symlink() or not path.is_file():
        raise QAAuditError("retry authorization is unavailable")
    recorded = _regular_json(path)
    if not isinstance(recorded, dict):
        raise QAAuditError("retry authorization is not an object")
    recorded_failure = recorded.get("failure")
    if not isinstance(recorded_failure, dict):
        raise QAAuditError("retry authorization failure evidence is missing")

    failure: dict[str, Any] | None = None
    failure_reason: str | None = None
    retrieval_path = attempt_root / "retrieval_model_ledger.jsonl"
    if retrieval_path.is_file() and not retrieval_path.is_symlink():
        candidates = []
        for event in read_ledger(retrieval_path):
            if event.get("event") != "model_call_failed":
                continue
            proxy = event.get("proxy_evidence")
            proxy_events = (
                proxy.get("events") if isinstance(proxy, dict) else None
            )
            proxy_event = (
                proxy_events[0]
                if isinstance(proxy_events, list)
                and len(proxy_events) == 1
                and isinstance(proxy_events[0], dict)
                else None
            )
            error = str(event.get("error", ""))
            status_match = re.search(r"Error code:\s*(\d+)", error)
            status_code = (
                int(status_match.group(1))
                if status_match is not None
                else proxy_event.get("http_status")
                if isinstance(proxy_event, dict)
                else recorded_failure.get("http_status")
                if not formal
                else None
            )
            reason = _retryable_error_reason(
                status_code=status_code,
                error_text=error,
            )
            if reason is not None:
                candidates.append(
                    (event, proxy_event, status_code, reason)
                )
        if len(candidates) == 1:
            event, proxy_event, status_code, failure_reason = candidates[0]
            if formal and (
                not isinstance(proxy_event, dict)
                or proxy_event.get("status") != "error"
                or proxy_event.get("http_status") != 500
                or proxy_event.get("upstream_http_attempts") is not None
                or proxy_event.get("logical_call_id")
                != event.get("logical_call_id")
            ):
                raise QAAuditError(
                    "formal retrieval retry evidence differs"
                )
            error = str(event.get("error", ""))
            failure = {
                "stage": "retrieval",
                "ledger": retrieval_path.name,
                "ledger_sha256": build_runner.sha256_file(retrieval_path),
                "logical_call_id": event.get("logical_call_id"),
                "proxy_event_id": (
                    proxy_event.get("event_id")
                    if isinstance(proxy_event, dict)
                    else None
                ),
                "http_status": status_code,
                "upstream_http_attempts": (
                    proxy_event.get("upstream_http_attempts")
                    if isinstance(proxy_event, dict)
                    else None
                ),
                "error_sha256": hashlib.sha256(
                    error.encode("utf-8")
                ).hexdigest(),
            }
    answer_path = attempt_root / "answer_ledger.jsonl"
    if failure is None and answer_path.is_file() and not answer_path.is_symlink():
        raw = [
            json.loads(line)
            for line in answer_path.read_text(encoding="utf-8").splitlines()
        ]
        run_ids = {
            str(event.get("run_id", ""))
            for event in raw
            if isinstance(event, dict)
        }
        if len(run_ids) == 1:
            events = answer_contract.audit_ledger(
                answer_path,
                expected_run_id=next(iter(run_ids)),
            )
            candidates = [
                event
                for event in events
                if event.get("event") == "physical_http_attempt_finished"
                and isinstance(event.get("payload"), dict)
                and event["payload"].get("status")
                == "retryable_empty_output"
            ]
            if len(candidates) == 1:
                payload = candidates[0]["payload"]
                error = str(payload.get("error", ""))
                if (
                    "upstream completed with empty output"
                    not in error.lower()
                    or payload.get("http_status") != 500
                    or payload.get("upstream_http_attempts") is not None
                    or (
                        formal
                        and not isinstance(payload.get("proxy_event_id"), str)
                    )
                ):
                    raise QAAuditError("answer retry evidence differs")
                failure = {
                    "stage": "answer",
                    "ledger": answer_path.name,
                    "ledger_sha256": build_runner.sha256_file(answer_path),
                    "logical_call_id": next(iter(run_ids)),
                    "proxy_event_id": payload.get("proxy_event_id"),
                    "http_status": payload.get("http_status"),
                    "upstream_http_attempts": payload.get(
                        "upstream_http_attempts"
                    ),
                    "error_sha256": hashlib.sha256(
                        error.encode("utf-8")
                    ).hexdigest(),
                }
                failure_reason = contract.EMPTY_OUTPUT_RETRY_REASON
    if failure is None:
        raise QAAuditError("retry authorization has no supported failure")
    if (
        failure_reason not in contract.RETRYABLE_QUESTION_ERRORS
        or recorded.get("reason") != failure_reason
    ):
        raise QAAuditError("retry authorization reason differs")

    attempt = manifest.get("attempt")
    qa_run_id = manifest.get("qa_run_id")
    question_id = manifest.get("question_id")
    expected_execution_run_id = (
        f"{qa_run_id}:attempt-{attempt:04d}"
        if isinstance(qa_run_id, str)
        and isinstance(attempt, int)
        and not isinstance(attempt, bool)
        else None
    )
    stage = failure.get("stage")
    logical_call_id = failure.get("logical_call_id")
    expected_prefix = (
        f"{expected_execution_run_id}:retrieval:{question_id}:"
        if stage == "retrieval"
        else f"{expected_execution_run_id}:{question_id}:"
        if stage == "answer"
        else None
    )
    if (
        not isinstance(expected_execution_run_id, str)
        or manifest.get("execution_run_id") != expected_execution_run_id
        or not isinstance(question_id, str)
        or not question_id
        or not isinstance(logical_call_id, str)
        or not isinstance(expected_prefix, str)
        or not logical_call_id.startswith(expected_prefix)
    ):
        raise QAAuditError("retry authorization failure binding differs")
    expected = {
        "schema_version": 1,
        "qa_run_id": manifest.get("qa_run_id"),
        "question_id": manifest.get("question_id"),
        "from_attempt": attempt,
        "to_attempt": attempt + 1 if isinstance(attempt, int) else None,
        "reason": failure_reason,
        "attempt_manifest_sha256": build_runner.sha256_file(
            attempt_root / "attempt_manifest.json"
        ),
        "failure": failure,
    }
    if recorded != expected:
        raise QAAuditError("retry authorization differs")
    return recorded


def _audit_attempt_chains(
    *,
    manifests: dict[tuple[str, str], tuple[Path, dict[str, Any]]],
    records: Sequence[dict[str, Any]],
    formal: bool,
) -> dict[str, Any]:
    """Verify each accepted question has one contiguous, authorized attempt chain."""

    grouped: dict[str, dict[int, tuple[tuple[str, str], Path, dict[str, Any]]]] = {}
    for key, (attempt_root, manifest) in manifests.items():
        question_id = str(manifest.get("question_id", ""))
        attempt = manifest.get("attempt")
        if not question_id or not isinstance(attempt, int) or isinstance(attempt, bool):
            raise QAAuditError("attempt manifest chain identity differs")
        attempts = grouped.setdefault(question_id, {})
        if attempt in attempts:
            raise QAAuditError("attempt number is duplicated within a question")
        attempts[attempt] = (key, attempt_root, manifest)

    accepted: dict[str, tuple[tuple[str, str], int]] = {}
    for record in records:
        question_id = str(record.get("question_id", ""))
        result = record.get("result")
        run_id = result.get("run_id") if isinstance(result, dict) else None
        match = (
            re.fullmatch(r".+:attempt-(\d{4})", run_id)
            if isinstance(run_id, str)
            else None
        )
        if not question_id or match is None or question_id in accepted:
            raise QAAuditError("accepted attempt chain identity differs")
        accepted_attempt = int(match.group(1))
        key = (run_id, question_id)
        if key not in manifests:
            raise QAAuditError("accepted attempt chain manifest is missing")
        accepted[question_id] = (key, accepted_attempt)

    if set(grouped) != set(accepted):
        raise QAAuditError("attempt chain question set differs")

    retry_bindings: list[dict[str, Any]] = []
    for question_id, attempts in grouped.items():
        accepted_key, accepted_attempt = accepted[question_id]
        if set(attempts) != set(range(1, accepted_attempt + 1)):
            raise QAAuditError("attempt chain is not contiguous")
        if attempts[accepted_attempt][0] != accepted_key:
            raise QAAuditError("accepted attempt chain binding differs")

        accepted_root = attempts[accepted_attempt][1]
        if (accepted_root / "retry_authorization.json").exists():
            raise QAAuditError("accepted attempt has a retry authorization")

        for attempt in range(1, accepted_attempt):
            _key, attempt_root, manifest = attempts[attempt]
            authorization = _audit_retry_authorization(
                attempt_root=attempt_root,
                manifest=manifest,
                formal=formal,
            )
            failure = authorization.get("failure")
            if not isinstance(failure, dict):
                raise QAAuditError("retry authorization failure evidence is missing")
            logical_call_id = failure.get("logical_call_id")
            proxy_event_id = failure.get("proxy_event_id")
            if formal and (
                not isinstance(logical_call_id, str)
                or not logical_call_id
                or not isinstance(proxy_event_id, str)
                or not proxy_event_id
            ):
                raise QAAuditError("formal retry proxy binding is missing")
            retry_bindings.append(
                {
                    "question_id": question_id,
                    "from_attempt": attempt,
                    "to_attempt": attempt + 1,
                    "stage": failure.get("stage"),
                    "logical_call_id": logical_call_id,
                    "proxy_event_id": proxy_event_id,
                }
            )

    return {
        "accepted_attempts": len(accepted),
        "authorized_retries": len(retry_bindings),
        "retry_bindings": retry_bindings,
    }


def _audit_retry_proxy_bindings(
    *,
    retry_bindings: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Bind every authorized retry to the exact failed exclusive-proxy event."""

    by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in by_id:
            raise QAAuditError("retry proxy event identity differs")
        by_id[event_id] = event

    bound_ids: list[str] = []
    for binding in retry_bindings:
        event_id = binding.get("proxy_event_id")
        event = by_id.get(event_id) if isinstance(event_id, str) else None
        if (
            not isinstance(event, dict)
            or event_id in bound_ids
            or event.get("question_id") != binding.get("question_id")
            or event.get("logical_call_id") != binding.get("logical_call_id")
            or event.get("status") != "error"
            or event.get("http_status") != 500
            or event.get("upstream_http_attempts") is not None
        ):
            raise QAAuditError("retry proxy event differs")
        bound_ids.append(event_id)

    return {
        "bound_retry_events": len(bound_ids),
        "retry_event_ids": bound_ids,
    }


def _accepted_proxy_event_ids(
    *, record: dict[str, Any], attempt_root: Path
) -> set[str]:
    result = record.get("result")
    if not isinstance(result, dict):
        raise QAAuditError("accepted question result is missing")
    events: list[Any] = []
    answer = result.get("answer")
    if isinstance(answer, dict):
        evidence = answer.get("proxy_evidence")
        if isinstance(evidence, dict) and isinstance(evidence.get("events"), list):
            events.extend(evidence["events"])
    artifacts = result.get("artifacts")
    if isinstance(artifacts, dict):
        ledger_name = artifacts.get("retrieval_model_ledger")
        if isinstance(ledger_name, str):
            ledger_path = attempt_root / ledger_name
            for ledger_event in read_ledger(ledger_path):
                if ledger_event.get("event") != "model_call_finished":
                    continue
                evidence = ledger_event.get("proxy_evidence")
                if isinstance(evidence, dict) and isinstance(
                    evidence.get("events"), list
                ):
                    events.extend(evidence["events"])
    event_ids: set[str] = set()
    for event in events:
        event_id = event.get("event_id") if isinstance(event, dict) else None
        if not isinstance(event_id, str) or not event_id or event_id in event_ids:
            raise QAAuditError("accepted proxy evidence identity differs")
        event_ids.add(event_id)
    return event_ids


def _proxy_execution_contract(
    plan: dict[str, Any],
) -> tuple[set[tuple[str, str, str]], set[str]]:
    upstream = plan.get("subscription_upstream")
    source_hashes = plan.get("source_hashes")
    if not isinstance(upstream, dict) or not isinstance(source_hashes, dict):
        raise QAAuditError("QA plan lacks the proxy execution contract")
    current = (
        upstream.get("origin"),
        upstream.get("code_sha256"),
        source_hashes.get("controlled_qa_proxy"),
    )
    origin_pattern = r"http://(?:127\.0\.0\.1|localhost|\[::1\]):\d+"
    if (
        not isinstance(current[0], str)
        or re.fullmatch(origin_pattern, current[0]) is None
        or not isinstance(current[1], str)
        or re.fullmatch(r"[0-9a-f]{64}", current[1]) is None
        or not isinstance(current[2], str)
        or re.fullmatch(r"[0-9a-f]{64}", current[2]) is None
        or current[2]
        != build_runner.sha256_file(qa_runner.CONTROLLED_QA_PROXY)
    ):
        raise QAAuditError("QA plan proxy execution contract differs")

    recovery = plan.get("execution_recovery")
    if recovery is None:
        return {current}, set()
    required_recovery_keys = {
        "schema_version",
        "posthoc",
        "accepted_proxy_generations",
        "recovered_invocations_without_wrapper_sha",
    }
    allowed_recovery_keys = required_recovery_keys | {
        "superseded_attempt_archives",
        "unbound_recovered_proxy_events",
    }
    if (
        not isinstance(recovery, dict)
        or not required_recovery_keys <= set(recovery) <= allowed_recovery_keys
        or recovery.get("schema_version") != 1
        or recovery.get("posthoc") is not True
    ):
        raise QAAuditError("proxy execution recovery contract differs")
    raw_generations = recovery.get("accepted_proxy_generations")
    raw_recovered = recovery.get("recovered_invocations_without_wrapper_sha")
    if not isinstance(raw_generations, list) or not raw_generations:
        raise QAAuditError("proxy execution recovery generations differ")
    accepted: set[tuple[str, str, str]] = set()
    for generation in raw_generations:
        if not isinstance(generation, dict) or set(generation) != {
            "origin",
            "code_sha256",
            "wrapper_sha256",
        }:
            raise QAAuditError("proxy execution recovery generation differs")
        value = (
            generation.get("origin"),
            generation.get("code_sha256"),
            generation.get("wrapper_sha256"),
        )
        if (
            not isinstance(value[0], str)
            or re.fullmatch(origin_pattern, value[0]) is None
            or not isinstance(value[1], str)
            or re.fullmatch(r"[0-9a-f]{64}", value[1]) is None
            or not isinstance(value[2], str)
            or re.fullmatch(r"[0-9a-f]{64}", value[2]) is None
            or value in accepted
        ):
            raise QAAuditError("proxy execution recovery generation differs")
        accepted.add(value)
    if current not in accepted:
        raise QAAuditError("current proxy generation is not recovery-authorized")
    if (
        not isinstance(raw_recovered, list)
        or any(
            not isinstance(name, str)
            or re.fullmatch(r"invocation-\d{4}", name) is None
            for name in raw_recovered
        )
        or len(raw_recovered) != len(set(raw_recovered))
    ):
        raise QAAuditError("recovered proxy invocation declarations differ")
    return accepted, set(raw_recovered)


def _directory_descriptor_sha256(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise QAAuditError(f"recovery archive is unavailable: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise QAAuditError(f"recovery archive contains a symlink: {path}")
        if not path.is_file():
            continue
        entries.append(
            {
                "byte_count": path.stat().st_size,
                "path": path.relative_to(root).as_posix(),
                "sha256": build_runner.sha256_file(path),
            }
        )
    if not entries:
        raise QAAuditError(f"recovery archive is empty: {root}")
    payload = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _proxy_event_ids_in_value(value: Any) -> set[str]:
    event_ids: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                key in {"event_id", "proxy_event_id"}
                and isinstance(child, str)
                and child.startswith("subscription-qa-")
            ):
                event_ids.add(child)
            event_ids.update(_proxy_event_ids_in_value(child))
    elif isinstance(value, list):
        for child in value:
            event_ids.update(_proxy_event_ids_in_value(child))
    return event_ids


def _execution_recovery_event_contract(
    *, qa_root: Path, plan: dict[str, Any]
) -> tuple[set[str], dict[str, tuple[str, str]]]:
    recovery = plan.get("execution_recovery")
    if recovery is None:
        return set(), {}
    _proxy_execution_contract(plan)
    assert isinstance(recovery, dict)
    raw_archives = recovery.get("superseded_attempt_archives", [])
    raw_unbound = recovery.get("unbound_recovered_proxy_events", [])
    if not isinstance(raw_archives, list) or not isinstance(raw_unbound, list):
        raise QAAuditError("proxy execution recovery event contract differs")
    campaign_root = qa_root.expanduser().absolute().parent.resolve()
    superseded: set[str] = set()
    for entry in raw_archives:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "descriptor_sha256",
            "proxy_event_ids",
        }:
            raise QAAuditError("superseded attempt archive contract differs")
        relative = entry.get("path")
        descriptor_sha256 = entry.get("descriptor_sha256")
        declared_event_ids = entry.get("proxy_event_ids")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(descriptor_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", descriptor_sha256) is None
            or not isinstance(declared_event_ids, list)
            or not declared_event_ids
            or any(
                not isinstance(event_id, str) or not event_id
                for event_id in declared_event_ids
            )
            or len(declared_event_ids) != len(set(declared_event_ids))
        ):
            raise QAAuditError("superseded attempt archive contract differs")
        archive = (campaign_root / relative).resolve()
        try:
            archive.relative_to(campaign_root)
        except ValueError as exc:
            raise QAAuditError(
                "superseded attempt archive escapes the campaign root"
            ) from exc
        if (
            re.fullmatch(r"attempt-\d{4}", archive.name) is None
            or _directory_descriptor_sha256(archive) != descriptor_sha256
        ):
            raise QAAuditError("superseded attempt archive descriptor differs")
        observed_event_ids: set[str] = set()
        for ledger_name in ("retrieval_model_ledger.jsonl", "answer_ledger.jsonl"):
            ledger_path = archive / ledger_name
            if not ledger_path.is_file() or ledger_path.is_symlink():
                continue
            for line in ledger_path.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise QAAuditError(
                        f"superseded attempt ledger is invalid: {ledger_path}"
                    ) from exc
                observed_event_ids.update(_proxy_event_ids_in_value(value))
        declared = set(declared_event_ids)
        if observed_event_ids != declared or superseded & declared:
            raise QAAuditError("superseded attempt archive proxy evidence differs")
        superseded.update(declared)
    unbound: dict[str, tuple[str, str]] = {}
    for entry in raw_unbound:
        if not isinstance(entry, dict) or set(entry) != {
            "event_id",
            "invocation",
            "status",
        }:
            raise QAAuditError("unbound recovery proxy event declarations differ")
        event_id = entry.get("event_id")
        invocation = entry.get("invocation")
        status = entry.get("status")
        if (
            not isinstance(event_id, str)
            or not event_id
            or not isinstance(invocation, str)
            or re.fullmatch(r"invocation-\d{4}", invocation) is None
            or status not in {"success", "error"}
            or event_id in unbound
        ):
            raise QAAuditError("unbound recovery proxy event declarations differ")
        unbound[event_id] = (invocation, status)
    if superseded & set(unbound):
        raise QAAuditError("unbound recovery proxy event declarations differ")
    return superseded, unbound


def _audit_proxy_invocation(
    *,
    invocation: Path,
    qa_run_id: str,
    plan: dict[str, Any],
) -> bool:
    accepted_generations, recovered_invocations = _proxy_execution_contract(plan)
    ready_path = invocation / "ready.json"
    start_path = invocation / "start.json"
    log_path = invocation / "requests.jsonl"
    process_log_path = invocation / "process.log"
    ready = _regular_json(ready_path)
    start = _regular_json(start_path)
    if process_log_path.is_symlink() or not process_log_path.is_file():
        raise QAAuditError(f"proxy process log is unavailable: {process_log_path}")
    if not isinstance(ready, dict) or not isinstance(start, dict):
        raise QAAuditError("proxy invocation metadata is not an object")
    pid = ready.get("pid")
    base_url = ready.get("base_url")
    health = start.get("health")
    observed_origin = ready.get("upstream")
    observed_upstream_sha = ready.get("upstream_code_sha256")
    shared_invalid = any(
        value is True
        for value in (
            ready.get("run_id") != qa_run_id,
            ready.get("log") != str(log_path.resolve()),
            not isinstance(pid, int) or isinstance(pid, bool) or pid < 1,
            not isinstance(base_url, str)
            or re.fullmatch(r"http://127\.0\.0\.1:\d+/v1", base_url) is None,
            start.get("run_id") != qa_run_id,
            start.get("pid") != pid,
            start.get("base_url") != base_url,
            start.get("upstream") != observed_origin,
            start.get("upstream_code_sha256") != observed_upstream_sha,
            start.get("ready_path") != str(ready_path.resolve()),
            start.get("log_path") != str(log_path.resolve()),
            start.get("process_log_path") != str(process_log_path.resolve()),
            start.get("started_at") != ready.get("started_at"),
            not isinstance(health, dict),
        )
    )
    if shared_invalid:
        raise QAAuditError("proxy invocation contract differs")
    assert isinstance(health, dict)
    if (
        health.get("status") != "ok"
        or health.get("run_id") != qa_run_id
        or health.get("exclusive_log") != str(log_path.resolve())
        or health.get("upstream") != observed_origin
        or health.get("upstream_code_sha256") != observed_upstream_sha
    ):
        raise QAAuditError("proxy invocation health contract differs")
    stop_path = invocation / "stop.json"
    recovery_path = invocation / "recovery.json"
    if stop_path.is_file() and not stop_path.is_symlink():
        stop = _regular_json(stop_path)
        observed_generation = (
            observed_origin,
            observed_upstream_sha,
            stop.get("wrapper_sha256") if isinstance(stop, dict) else None,
        )
        if (
            not isinstance(stop, dict)
            or any(stop.get(key) != value for key, value in start.items())
            or observed_generation not in accepted_generations
            or stop.get("returncode") not in (-15, -9)
            or not isinstance(stop.get("finished_at"), str)
        ):
            raise QAAuditError("proxy invocation contract differs")
        return False
    elif recovery_path.is_file() and not recovery_path.is_symlink():
        recovery = _regular_json(recovery_path)
        if (
            not isinstance(recovery, dict)
            or recovery.get("run_id") != qa_run_id
            or recovery.get("pid") != pid
            or recovery.get("ready_path") != str(ready_path.resolve())
            or recovery.get("ready_sha256")
            != build_runner.sha256_file(ready_path)
            or recovery.get("status")
            not in {"already_exited", "terminated_stale_proxy"}
            or invocation.name not in recovered_invocations
            or not any(
                generation[:2] == (observed_origin, observed_upstream_sha)
                for generation in accepted_generations
            )
        ):
            raise QAAuditError("proxy recovery contract differs")
        return True
    else:
        raise QAAuditError(f"proxy invocation lacks a stop record: {invocation}")


def audit_proxy_logs(
    *,
    qa_root: Path,
    qa_run_id: str,
    records: Sequence[dict[str, Any]] | None = None,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify every physical model request recorded by exclusive QA proxies."""

    proxy_root = qa_root.expanduser().absolute() / "proxy"
    logs = sorted(proxy_root.glob("invocation-*/requests.jsonl"))
    if not logs:
        raise QAAuditError("exclusive QA proxy logs are missing")
    events: list[dict[str, Any]] = []
    verified_invocations = 0
    recovered_invocations_without_wrapper_sha: set[str] = set()
    declared_recovered_invocations: set[str] = set()
    superseded_proxy_event_ids: set[str] = set()
    unbound_recovered_proxy_events: dict[str, tuple[str, str]] = {}
    event_invocations: dict[str, str] = {}
    if plan is not None and plan.get("execution_recovery") is not None:
        _accepted, declared_recovered_invocations = _proxy_execution_contract(plan)
        (
            superseded_proxy_event_ids,
            unbound_recovered_proxy_events,
        ) = _execution_recovery_event_contract(qa_root=qa_root, plan=plan)
    for log in logs:
        if log.is_symlink() or not log.is_file():
            raise QAAuditError(f"proxy log is unavailable: {log}")
        invocation = log.parent
        if plan is not None:
            recovered_without_wrapper_sha = _audit_proxy_invocation(
                invocation=invocation,
                qa_run_id=qa_run_id,
                plan=plan,
            )
            verified_invocations += 1
            if recovered_without_wrapper_sha:
                recovered_invocations_without_wrapper_sha.add(invocation.name)
        if not (
            (invocation / "stop.json").is_file()
            or (invocation / "recovery.json").is_file()
        ):
            raise QAAuditError(f"proxy invocation lacks a stop record: {invocation}")
        for line_number, line in enumerate(
            log.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QAAuditError(
                    f"proxy log line is invalid JSON: {log}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise QAAuditError(f"proxy log event is not an object: {log}")
            event_id = value.get("event_id")
            if isinstance(event_id, str):
                event_invocations[event_id] = invocation.name
            events.append(value)
    if (
        plan is not None
        and recovered_invocations_without_wrapper_sha
        != declared_recovered_invocations
    ):
        raise QAAuditError("recovered proxy invocation declarations differ")
    if not events:
        raise QAAuditError("exclusive QA proxy events are empty")
    event_ids: set[str] = set()
    response_ids: set[str] = set()
    reasoning_tokens = 0
    child_upstream_attempts = 0
    upstream_attempts = 0
    unknown_upstream_attempt_events = 0
    failed_events = 0
    for event in events:
        event_id = event.get("event_id")
        response_id = event.get("response_id")
        usage = event.get("usage")
        details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
        reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
        ignored = event.get("ignored_client_parameters")
        child_attempts = event.get("child_upstream_http_attempts")
        provider_attempts = event.get("upstream_http_attempts")
        provider_attempts_known = (
            isinstance(provider_attempts, int)
            and not isinstance(provider_attempts, bool)
        )
        provider_attempts_in_range = (
            provider_attempts_known and 1 <= provider_attempts <= 2
        )
        usage_counts = (
            [usage.get(name) for name in (
                "prompt_tokens", "completion_tokens", "total_tokens"
            )]
            if isinstance(usage, dict)
            else []
        )
        usage_invalid = (
            len(usage_counts) != 3
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in usage_counts
            )
            or (
                len(usage_counts) == 3
                and usage_counts[2] != usage_counts[0] + usage_counts[1]
            )
        )
        common_invalid = (
            event.get("run_id") != qa_run_id
            or event.get("requested_model") != readonly_control.EXPECTED_MODEL
            or event.get("client_http_attempts") != 1
            or child_attempts not in (0, 1)
            or (
                provider_attempts is not None
                and not provider_attempts_known
            )
            or (provider_attempts_known and provider_attempts < 0)
            or event.get("unsupported_parameters") != []
            or ignored not in ([], ["max_output_tokens"])
            or not isinstance(event.get("question_id"), str)
            or not isinstance(event.get("logical_call_id"), str)
            or not isinstance(event_id, str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(event.get("request_sha256")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(event.get("response_sha256")))
        )
        status = event.get("status")
        success_invalid = status == "success" and (
            event.get("http_status") != 200
            or event.get("actual_model") != readonly_control.EXPECTED_MODEL
            or event.get("error") is not None
            or child_attempts != 1
            or not provider_attempts_in_range
            or not isinstance(response_id, str)
            or reasoning != 0
            or usage_invalid
        )
        error_invalid = status == "error" and (
            not isinstance(event.get("http_status"), int)
            or event.get("http_status", 0) < 400
            or not isinstance(event.get("error"), str)
            or not event.get("error")
            or response_id is not None
            or event.get("actual_model") not in (None, readonly_control.EXPECTED_MODEL)
            or reasoning not in (None, 0)
            or (
                child_attempts == 0 and provider_attempts != 0
            )
            or (
                child_attempts == 1
                and provider_attempts is not None
                and not provider_attempts_in_range
            )
        )
        if common_invalid or status not in {"success", "error"} or success_invalid or error_invalid:
            raise QAAuditError("proxy event contract differs")
        if event_id in event_ids or (
            isinstance(response_id, str) and response_id in response_ids
        ):
            raise QAAuditError("proxy event or response ID is duplicated")
        event_ids.add(event_id)
        if isinstance(response_id, str):
            response_ids.add(response_id)
        if status == "success":
            reasoning_tokens += reasoning
        else:
            failed_events += 1
        child_upstream_attempts += child_attempts
        if provider_attempts_known:
            upstream_attempts += provider_attempts
        else:
            unknown_upstream_attempt_events += 1
    recovery_event_ids = (
        superseded_proxy_event_ids | set(unbound_recovered_proxy_events)
    )
    if not recovery_event_ids <= event_ids:
        raise QAAuditError("declared recovery proxy event is absent from logs")
    if any(
        event.get("status") != expected[1]
        or event_invocations.get(str(event.get("event_id"))) != expected[0]
        or expected[0] not in declared_recovered_invocations
        for event in events
        if (expected := unbound_recovered_proxy_events.get(str(event.get("event_id"))))
        is not None
    ):
        raise QAAuditError("unbound recovery proxy event contract differs")
    report = {
        "proxy_logs": len(logs),
        "verified_invocations": verified_invocations,
        "recovered_invocations_without_wrapper_sha_count": len(
            recovered_invocations_without_wrapper_sha
        ),
        "unbound_recovered_proxy_events": len(
            unbound_recovered_proxy_events
        ),
        "proxy_events": len(events),
        "response_ids": len(response_ids),
        "child_upstream_http_attempts": child_upstream_attempts,
        "upstream_http_attempts": (
            None if unknown_upstream_attempt_events else upstream_attempts
        ),
        "known_upstream_http_attempts": upstream_attempts,
        "unknown_upstream_attempt_events": unknown_upstream_attempt_events,
        "reasoning_tokens": reasoning_tokens,
        "failed_events": failed_events,
    }
    if records is not None:
        manifests = _attempt_manifests(qa_root=qa_root, qa_run_id=qa_run_id)
        attempt_chain_report: dict[str, Any] | None = None
        if plan is not None:
            attempt_chain_report = _audit_attempt_chains(
                manifests=manifests,
                records=records,
                formal=True,
            )
            retry_proxy_report = _audit_retry_proxy_bindings(
                retry_bindings=attempt_chain_report["retry_bindings"],
                events=events,
            )
            attempt_chain_report = {
                **attempt_chain_report,
                **retry_proxy_report,
            }
        accepted_keys: set[tuple[str, str]] = set()
        accepted_event_ids: set[str] = set()
        for record in records:
            result = record.get("result")
            if not isinstance(result, dict):
                raise QAAuditError("accepted question result is missing")
            key = (str(result.get("run_id", "")), str(record.get("question_id", "")))
            manifest_entry = manifests.get(key)
            if manifest_entry is None or key in accepted_keys:
                raise QAAuditError("accepted result attempt binding differs")
            accepted_keys.add(key)
            attempt_root, _manifest = manifest_entry
            accepted_event_ids.update(
                _accepted_proxy_event_ids(record=record, attempt_root=attempt_root)
            )
        if recovery_event_ids & accepted_event_ids:
            raise QAAuditError("recovery proxy event appears in accepted evidence")
        for key, (attempt_root, _manifest) in manifests.items():
            if key not in accepted_keys:
                result_path = attempt_root / "result.json"
                if result_path.is_file() and not result_path.is_symlink():
                    abandoned_result = _regular_json(result_path)
                    if (
                        isinstance(abandoned_result, dict)
                        and abandoned_result.get("status") == "complete"
                    ):
                        raise QAAuditError(
                            "unaccepted attempt contains a complete result"
                        )
        observed_accepted_ids: set[str] = set()
        accepted_successes = 0
        abandoned_successes = 0
        accepted_events = 0
        abandoned_events = 0
        for event in events:
            question_id = str(event["question_id"])
            logical_call_id = str(event["logical_call_id"])
            expected_completion_cap = (
                contract.ACCEPTED_ANSWER_MAX_TOKENS
                if logical_call_id.endswith(":answer")
                else 1_200
                if re.search(r":call-\d{4}$", logical_call_id)
                else None
            )
            observed_completion_cap = event.get("requested_completion_token_cap")
            if (
                observed_completion_cap not in expected_completion_cap
                if isinstance(expected_completion_cap, tuple)
                else observed_completion_cap != expected_completion_cap
            ):
                raise QAAuditError("proxy completion token cap differs")
            event_id = str(event["event_id"])
            if event_id in recovery_event_ids:
                abandoned_events += 1
                if event["status"] == "success":
                    abandoned_successes += 1
                continue
            candidates = [
                (key, entry)
                for key, entry in manifests.items()
                if key[1] == question_id
                and logical_call_id.startswith(f"{key[0]}:")
            ]
            if len(candidates) != 1:
                raise QAAuditError("proxy event attempt binding differs")
            key, (attempt_root, _manifest) = candidates[0]
            if logical_call_id not in _attempt_request_ids(attempt_root):
                raise QAAuditError("proxy event lacks attempt request evidence")
            if key in accepted_keys:
                accepted_events += 1
                observed_accepted_ids.add(event_id)
                if event_id not in accepted_event_ids:
                    raise QAAuditError(
                        "accepted proxy event is absent from result evidence"
                    )
                if event["status"] == "success":
                    accepted_successes += 1
            else:
                abandoned_events += 1
                if event_id in accepted_event_ids:
                    raise QAAuditError(
                        "abandoned proxy event appears in accepted evidence"
                    )
                if event["status"] == "success":
                    abandoned_successes += 1
        if observed_accepted_ids != accepted_event_ids:
            raise QAAuditError("accepted result proxy evidence is incomplete")
        report.update(
            {
                "accepted_events": accepted_events,
                "abandoned_events": abandoned_events,
                "accepted_successes": accepted_successes,
                "abandoned_successes": abandoned_successes,
                "all_successes_accounted": (
                    accepted_successes + abandoned_successes
                    == len(response_ids)
                ),
                "attempt_chains": attempt_chain_report,
                "superseded_proxy_events": len(superseded_proxy_event_ids),
            }
        )
        if not report["all_successes_accounted"]:
            raise QAAuditError("successful proxy event accounting differs")
    return report


def _regular_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise QAAuditError(f"required JSON artifact is unavailable: {path}")
    return build_runner.read_json(path)


def _regular_text(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise QAAuditError(f"required text artifact is unavailable: {path}")
    return path.read_text(encoding="utf-8")


def _frozen_upstream(qa_root: Path) -> tuple[str, str]:
    preregistration = _regular_json(
        qa_root.expanduser().absolute() / "preregistration.json"
    )
    upstream = (
        preregistration.get("subscription_upstream")
        if isinstance(preregistration, dict)
        else None
    )
    origin = upstream.get("origin") if isinstance(upstream, dict) else None
    code_sha256 = (
        upstream.get("code_sha256") if isinstance(upstream, dict) else None
    )
    if (
        not isinstance(origin, str)
        or re.fullmatch(r"http://(?:127\.0\.0\.1|localhost|\[::1\]):\d+", origin)
        is None
        or not isinstance(code_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", code_sha256) is None
    ):
        raise QAAuditError("preregistered upstream contract differs")
    return origin, code_sha256


def _frozen_execution_recovery(qa_root: Path) -> dict[str, Any] | None:
    preregistration = _regular_json(
        qa_root.expanduser().absolute() / "preregistration.json"
    )
    recovery = (
        preregistration.get("execution_recovery")
        if isinstance(preregistration, dict)
        else None
    )
    if recovery is None:
        return None
    if not isinstance(recovery, dict):
        raise QAAuditError("preregistered execution recovery contract differs")
    return recovery


def _audit_evaluator_inputs(
    *,
    records: Sequence[dict[str, Any]],
    qa_root: Path,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    hashes: dict[str, Any] = {}
    expected_paths: set[Path] = set()
    for tier in sorted({str(record["tier"]) for record in records}):
        selected = [record for record in records if record["tier"] == tier]
        tier_outputs: dict[str, Any] = {}
        tier_hashes: dict[str, Any] = {}
        locomo_groups: dict[int, list[dict[str, Any]]] = {}
        for record in selected:
            if record["benchmark"] == "locomo":
                sample = int(record["selection"]["sample_index"])
                locomo_groups.setdefault(sample, []).append(
                    {
                        "question_id": record["original_question_id"],
                        "question": record["question"],
                        "gold": record["gold"],
                        "category": record["category"],
                        "answer": record["answer"],
                    }
                )
        locomo_paths: list[str] = []
        locomo_hashes: list[str] = []
        for sample, expected in sorted(locomo_groups.items()):
            path = qa_root / tier / "evaluator_inputs/locomo" / f"sample{sample}_questions.json"
            if _regular_json(path) != expected:
                raise QAAuditError(f"LoCoMo evaluator input differs: {path}")
            expected_paths.add(path.resolve())
            locomo_paths.append(str(path))
            locomo_hashes.append(build_runner.sha256_file(path))
        short_present = any(
            record["benchmark"] in {"locomo", "longmemeval-s"}
            for record in selected
        )
        if short_present:
            tier_outputs["locomo"] = locomo_paths
            tier_hashes["locomo"] = locomo_hashes
            lme_path = qa_root / tier / "evaluator_inputs/longmemeval-s/hypotheses.jsonl"
            lme_rows = [
                {
                    "question_id": record["original_question_id"],
                    "hypothesis": record["answer"],
                }
                for record in selected
                if record["benchmark"] == "longmemeval-s"
            ]
            expected_text = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in lme_rows
            )
            if _regular_text(lme_path) != expected_text:
                raise QAAuditError(f"LongMemEval evaluator input differs: {lme_path}")
            expected_paths.add(lme_path.resolve())
            tier_outputs["longmemeval-s"] = str(lme_path)
            tier_hashes["longmemeval-s"] = build_runner.sha256_file(lme_path)
        beam_rows = [
            {
                "question_id": record["original_question_id"],
                "answer": record["answer"],
            }
            for record in selected
            if record["benchmark"] == "beam-100k"
        ]
        if beam_rows:
            beam_path = qa_root / tier / "evaluator_inputs/beam-100k/predictions.jsonl"
            expected_text = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in beam_rows
            )
            if _regular_text(beam_path) != expected_text:
                raise QAAuditError(f"BEAM evaluator input differs: {beam_path}")
            expected_paths.add(beam_path.resolve())
            tier_outputs["beam-100k"] = str(beam_path)
            tier_hashes["beam-100k"] = build_runner.sha256_file(beam_path)
        outputs[tier] = tier_outputs
        hashes[tier] = tier_hashes
    actual_paths = {
        path.resolve()
        for path in qa_root.glob("*/evaluator_inputs/**/*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_paths != expected_paths:
        raise QAAuditError("unexpected evaluator input files are present")
    return {"paths": outputs, "sha256": hashes}


def audit_qa_root(
    *,
    plan: dict[str, Any],
    bindings: Sequence[contract.RunBinding],
    validated_units: Sequence[dict[str, Any]],
    qa_root: Path,
    formal: bool,
) -> dict[str, Any]:
    """Audit one complete QA root against its immutable build-derived plan."""

    root = qa_root.expanduser().absolute()
    preregistration = root / "preregistration.json"
    if _regular_json(preregistration) != plan:
        raise QAAuditError("QA preregistration differs from current frozen inputs")
    completion_path = root / "completion.json"
    completion = _regular_json(completion_path)
    if (
        completion.get("status") != "complete"
        or completion.get("protocol_id") != contract.PROTOCOL_ID
        or completion.get("run_ids") != plan.get("run_ids")
        or completion.get("question_count") != plan.get("question_count")
        or completion.get("preregistration_sha256")
        != build_runner.sha256_file(preregistration)
    ):
        raise QAAuditError("QA completion contract differs")
    qa_run_id = completion.get("qa_run_id")
    if not isinstance(qa_run_id, str) or not qa_run_id:
        raise QAAuditError("QA completion run ID is invalid")
    records = qa_runner.execute_pending_questions(
        bindings=bindings,
        validated_units=validated_units,
        output_dir=root,
        qa_run_id=qa_run_id,
        completion_resource=object(),
        answer_client=object(),
        tokenizer=(answer_contract.formal_token_counter() if formal else object()),
        proxy_log=None,
        formal=formal,
    )
    if len(records) != plan.get("question_count"):
        raise QAAuditError("QA question count differs")
    question_report = audit_question_records(
        records=records,
        qa_root=root,
        formal=formal,
        bindings=bindings,
        validated_units=validated_units,
    )
    proxy_report: dict[str, Any] | None = None
    if formal:
        proxy_report = audit_proxy_logs(
            qa_root=root,
            qa_run_id=qa_run_id,
            records=records,
            plan=plan,
        )
        expected_responses = (
            question_report["retrieval_model_calls"]
            + question_report["answer_model_calls"]
        )
        if (
            proxy_report.get("accepted_successes") != expected_responses
            or proxy_report.get("all_successes_accounted") is not True
        ):
            raise QAAuditError(
                "accepted proxy successes differ from logical model calls"
            )
    evaluator = _audit_evaluator_inputs(records=records, qa_root=root)
    if (
        completion.get("evaluator_inputs") != evaluator["paths"]
        or completion.get("evaluator_input_sha256") != evaluator["sha256"]
    ):
        raise QAAuditError("QA completion evaluator binding differs")
    return {
        "schema_version": 1,
        "status": "verified_complete",
        "protocol_id": contract.PROTOCOL_ID,
        "qa_run_id": qa_run_id,
        "run_ids": plan["run_ids"],
        **question_report,
        "preregistration_sha256": build_runner.sha256_file(preregistration),
        "completion_sha256": build_runner.sha256_file(completion_path),
        "evaluator_input_sha256": evaluator["sha256"],
        "proxy": proxy_report,
    }


def audit_preflight_root(
    *,
    plan: dict[str, Any],
    bindings: Sequence[contract.RunBinding],
    validated_units: Sequence[dict[str, Any]],
    qa_root: Path,
    formal: bool,
) -> dict[str, Any]:
    """Audit the exact one-question-per-benchmark preflight scope."""

    root = qa_root.expanduser().absolute()
    preregistration = root / "preregistration.json"
    if _regular_json(preregistration) != plan:
        raise QAAuditError("QA preregistration differs from current frozen inputs")
    if (root / "completion.json").exists():
        raise QAAuditError("preflight audit requires an unfinished QA root")
    if len(bindings) != len(validated_units):
        raise QAAuditError("preflight bindings and validated units differ")
    selected_bindings: list[contract.RunBinding] = []
    selected_units: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for binding, unit in zip(bindings, validated_units, strict=True):
        key = (str(binding.row.get("tier")), str(binding.row.get("benchmark")))
        if key in seen:
            continue
        questions = unit.get("questions")
        if not isinstance(questions, list) or not questions:
            raise QAAuditError("preflight source unit has no questions")
        seen.add(key)
        selected_bindings.append(binding)
        selected_units.append({**unit, "questions": [questions[0]]})
    preflight_path = root / "preflight.json"
    preflight = _regular_json(preflight_path)
    if not isinstance(preflight, dict):
        raise QAAuditError("preflight artifact is not an object")
    qa_run_id = preflight.get("qa_run_id")
    if not isinstance(qa_run_id, str) or not qa_run_id:
        raise QAAuditError("preflight run ID is invalid")
    records = qa_runner.execute_pending_questions(
        bindings=selected_bindings,
        validated_units=selected_units,
        output_dir=root,
        qa_run_id=qa_run_id,
        completion_resource=object(),
        answer_client=object(),
        tokenizer=(answer_contract.formal_token_counter() if formal else object()),
        proxy_log=None,
        formal=formal,
    )
    question_ids = [str(record.get("question_id", "")) for record in records]
    answer_sha256 = {
        question_id: hashlib.sha256(
            str(record.get("answer", "")).encode("utf-8")
        ).hexdigest()
        for question_id, record in zip(question_ids, records, strict=True)
    }
    if (
        preflight.get("status") != "complete"
        or preflight.get("protocol_id") != contract.PROTOCOL_ID
        or preflight.get("question_count") != len(records)
        or preflight.get("question_ids") != question_ids
        or preflight.get("answer_sha256") != answer_sha256
        or preflight.get("preregistration_sha256")
        != build_runner.sha256_file(preregistration)
    ):
        raise QAAuditError("preflight completion contract differs")
    question_report = audit_question_records(
        records=records,
        qa_root=root,
        formal=formal,
        bindings=selected_bindings,
        validated_units=selected_units,
    )
    proxy_report: dict[str, Any] | None = None
    if formal:
        proxy_report = audit_proxy_logs(
            qa_root=root,
            qa_run_id=qa_run_id,
            records=records,
            plan=plan,
        )
        expected_responses = (
            question_report["retrieval_model_calls"]
            + question_report["answer_model_calls"]
        )
        if (
            proxy_report.get("accepted_successes") != expected_responses
            or proxy_report.get("all_successes_accounted") is not True
        ):
            raise QAAuditError(
                "preflight proxy successes differ from logical model calls"
            )
    evaluator_files = [
        path
        for path in root.glob("*/evaluator_inputs/**/*")
        if path.is_file() or path.is_symlink()
    ]
    if evaluator_files:
        raise QAAuditError("preflight unexpectedly contains evaluator inputs")
    return {
        "schema_version": 1,
        "status": "verified_preflight",
        "protocol_id": contract.PROTOCOL_ID,
        "qa_run_id": qa_run_id,
        "run_ids": plan["run_ids"],
        **question_report,
        "preregistration_sha256": build_runner.sha256_file(preregistration),
        "preflight_sha256": build_runner.sha256_file(preflight_path),
        "proxy": proxy_report,
    }


def audit_scope(
    *,
    matrix_path: Path,
    build_root: Path,
    qa_root: Path,
    run_ids: Sequence[str],
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Rebuild the frozen source bindings and audit one formal QA scope."""

    matrix = matrix_path.expanduser().resolve()
    bindings = contract.select_completed_runs(
        matrix_path=matrix,
        build_root=build_root.expanduser().resolve(),
        run_ids=run_ids,
    )
    units = [contract.validate_unit_binding(binding) for binding in bindings]
    qa_runner.validate_formal_scope(
        bindings=bindings,
        validated_units=units,
    )
    upstream, upstream_code_sha256 = _frozen_upstream(qa_root)
    plan = contract.build_preregistration(
        bindings=bindings,
        validated_units=units,
        matrix_path=matrix,
        upstream=upstream,
        upstream_code_sha256=upstream_code_sha256,
    )
    execution_recovery = _frozen_execution_recovery(qa_root)
    if execution_recovery is not None:
        plan["execution_recovery"] = execution_recovery
    audit_root = audit_preflight_root if preflight_only else audit_qa_root
    return audit_root(
        plan=plan,
        bindings=bindings,
        validated_units=units,
        qa_root=qa_root,
        formal=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=build_runner.DEFAULT_MATRIX)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_scope(
        matrix_path=args.matrix,
        build_root=args.build_root,
        qa_root=args.qa_root,
        run_ids=args.run_ids,
        preflight_only=args.preflight_only,
    )
    output = (
        args.output.expanduser().absolute()
        if args.output is not None
        else args.qa_root.expanduser().absolute()
        / ("preflight-audit.json" if args.preflight_only else "audit.json")
    )
    if output.is_file() and not output.is_symlink():
        if build_runner.read_json(output) != report:
            raise QAAuditError("existing QA audit report differs")
    else:
        answer_contract.atomic_json_no_clobber(output, report)
    label = "preflight" if args.preflight_only else "QA"
    print(
        f"verified {label}: {report['question_count']} questions / "
        f"{report['artifact_count']} artifacts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
