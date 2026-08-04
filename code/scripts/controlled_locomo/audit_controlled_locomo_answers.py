#!/usr/bin/env python3
"""Independently audit controlled LoCoMo GPT-5.5 answers and R004 traces."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_gpt55_locomo_baselines as baseline_auditor  # noqa: E402
import controlled_locomo_answer_contract as contract  # noqa: E402
from scripts.gateways import openai_gpt55_flex_gateway as flex_gateway  # noqa: E402
from scripts.gateways import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402
from scripts.evaluation.visible_token_audit import audit_visible_token_trace  # noqa: E402
from scripts.evaluation.visible_token_budget import TokenCounter  # noqa: E402


def _trace_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        raise contract.ControlledAnswerError("visible-token trace is incomplete")
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise contract.ControlledAnswerError(
                f"visible-token line {line_number} is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise contract.ControlledAnswerError("visible-token record is not an object")
        records.append(value)
    return records


def _resolve_inside(output_dir: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise contract.ControlledAnswerError(f"{label} path is invalid")
    raw = Path(relative)
    if raw.is_absolute():
        raise contract.ControlledAnswerError(f"{label} path must be relative")
    candidate = output_dir / raw
    contract.reject_symlink_components(candidate)
    if candidate.is_symlink():
        raise contract.ControlledAnswerError(f"{label} path is a symlink")
    path = candidate.resolve()
    try:
        path.relative_to(output_dir)
    except ValueError as exc:
        raise contract.ControlledAnswerError(f"{label} path escapes output") from exc
    contract.reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise contract.ControlledAnswerError(f"{label} is missing or not regular")
    return path


def _load_proxy_entries(output_dir: Path, run_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proxy_root = output_dir / "proxy"
    if proxy_root.exists():
        contract.reject_symlink_components(proxy_root)
        if proxy_root.is_symlink() or not proxy_root.is_dir():
            raise contract.ControlledAnswerError(
                "proxy artifact root is not a regular directory"
            )
        invocation_dirs = sorted(proxy_root.glob("invocation-*"))
    else:
        invocation_dirs = []
    manifests: list[Path] = []
    for invocation_dir in invocation_dirs:
        contract.reject_symlink_components(invocation_dir)
        if invocation_dir.is_symlink() or not invocation_dir.is_dir():
            raise contract.ControlledAnswerError(
                "proxy invocation is not a regular directory"
            )
        manifest_path = invocation_dir / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise contract.ControlledAnswerError(
                f"proxy invocation lacks a complete manifest: {invocation_dir.name}"
            )
        manifests.append(manifest_path)
    entries: list[dict[str, Any]] = []
    invocation_reports: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    for manifest_path in manifests:
        if manifest_path.is_symlink():
            raise contract.ControlledAnswerError("proxy manifest is a symlink")
        manifest = contract.read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise contract.ControlledAnswerError("proxy invocation manifest is invalid")
        resolved_proxy_paths = {
            key: _resolve_inside(output_dir, manifest.get(key), label=f"proxy {key}")
            for key in ("start", "ready", "log", "process_log")
        }
        resolved_proxy_paths["manifest"] = manifest_path
        contract.ensure_distinct_paths(resolved_proxy_paths)
        if (
            manifest.get("wrapper_sha256")
            != contract.sha256_file(ROOT / "scripts/controlled_locomo/controlled_gpt55_run_proxy.py")
            or manifest.get("base_wrapper_sha256")
            != contract.sha256_file(ROOT / "scripts/controlled_locomo/gpt55_run_proxy.py")
            or manifest.get("upstream_proxy_sha256")
            != contract.sha256_file(ROOT / "scripts/gateways/openai_gpt55_flex_gateway.py")
            or manifest.get("flex_evidence_sha256")
            != contract.sha256_file(
                ROOT / "scripts/gateways/openai_gpt55_flex_gateway_evidence.py"
            )
        ):
            raise contract.ControlledAnswerError("proxy source fingerprint differs")
        for key in ("start", "ready", "log", "process_log"):
            path = resolved_proxy_paths[key]
            if (
                manifest.get(f"{key}_sha256") != contract.sha256_file(path)
                or manifest.get(f"{key}_bytes") != path.stat().st_size
            ):
                raise contract.ControlledAnswerError(f"proxy {key} fingerprint differs")
        start = contract.read_json(resolved_proxy_paths["start"])
        if not isinstance(start, dict):
            raise contract.ControlledAnswerError("proxy start record is invalid")
        immutable_start_fields = (
            "run_id",
            "invocation_id",
            "invocation_dir",
            "started_at",
            "pid",
            "base_url",
            "ready",
            "log",
            "process_log",
            "health",
        )
        if any(start.get(key) != manifest.get(key) for key in immutable_start_fields):
            raise contract.ControlledAnswerError("proxy start linkage differs")
        if (
            start.get("run_id") != run_id
            or start.get("invocation_id") != manifest_path.parent.name
            or start.get("invocation_dir")
            != str(manifest_path.parent.relative_to(output_dir))
            or start.get("start") is not None
        ):
            raise contract.ControlledAnswerError("proxy immutable start identity differs")
        expected_artifacts = {
            "ready": manifest_path.parent / "ready.json",
            "log": manifest_path.parent / "requests.jsonl",
            "process_log": manifest_path.parent / "process.log",
        }
        if any(
            resolved_proxy_paths[key] != expected_path.resolve()
            for key, expected_path in expected_artifacts.items()
        ):
            raise contract.ControlledAnswerError("proxy artifact path linkage differs")
        status = manifest.get("status")
        if status not in (None, "interrupted_recovered"):
            raise contract.ControlledAnswerError("proxy invocation status differs")
        if status == "interrupted_recovered" and manifest.get("recovery_action") not in {
            "original_process_not_running",
            "terminated_original_process",
            "original_pid_reused",
        }:
            raise contract.ControlledAnswerError("proxy recovery action differs")
        log_path = resolved_proxy_paths["log"]
        provider_window = manifest.get("provider_window")
        if manifest.get("provider_window_error") is not None:
            raise contract.ControlledAnswerError("proxy Flex provider window failed")
        try:
            provider_contract = flex_evidence.validate_recorded_contract(
                provider_window.get("contract")
                if isinstance(provider_window, dict)
                else None
            )
        except flex_evidence.EvidenceError as exc:
            raise contract.ControlledAnswerError(
                f"proxy Flex provider contract differs: {exc}"
            ) from exc
        if (
            manifest.get("gateway_result_root") != provider_contract["result_root"]
            or manifest.get("gateway_root_marker_sha256")
            != provider_contract["root_marker_sha256"]
            or manifest.get("gateway_ready_sha256")
            != provider_contract["ready_sha256"]
            or manifest.get("gateway_max_cost_usd")
            != provider_contract["max_cost_usd"]
            or manifest.get("provider_model") != flex_gateway.PROVIDER_MODEL
            or manifest.get("service_tier") != flex_gateway.SERVICE_TIER
        ):
            raise contract.ControlledAnswerError(
                "proxy Flex result-root binding differs"
            )
        expected_upstream = provider_contract["origin"]
        ready = contract.read_json(resolved_proxy_paths["ready"])
        if not isinstance(ready, dict):
            raise contract.ControlledAnswerError("proxy ready record is invalid")
        port = ready.get("port")
        expected_base = (
            f"http://127.0.0.1:{port}/v1"
            if isinstance(port, int)
            and not isinstance(port, bool)
            and 0 < port < 65536
            else None
        )
        if (
            ready.get("run_id") != run_id
            or ready.get("pid") != manifest.get("pid")
            or ready.get("upstream") != expected_upstream
            or Path(str(ready.get("log"))).resolve() != log_path
            or ready.get("base_url") != expected_base
            or manifest.get("base_url") != expected_base
            or ready.get("base_wrapper_sha256")
            != contract.sha256_file(ROOT / "scripts/controlled_locomo/gpt55_run_proxy.py")
            or ready.get("controlled_wrapper_sha256")
            != contract.sha256_file(ROOT / "scripts/controlled_locomo/controlled_gpt55_run_proxy.py")
        ):
            raise contract.ControlledAnswerError("proxy ready linkage differs")
        health = manifest.get("health")
        upstream_health = health.get("upstream_health") if isinstance(health, dict) else None
        if (
            not isinstance(health, dict)
            or health.get("status") != "ok"
            or health.get("run_id") != run_id
            or Path(str(health.get("exclusive_log"))).resolve() != log_path
            or health.get("upstream") != expected_upstream
            or not isinstance(upstream_health, dict)
            or upstream_health.get("status") != "ok"
            or upstream_health.get("schema")
            != "openai-gpt55-flex-health/v1"
            or upstream_health.get("requested_model")
            != flex_gateway.REQUESTED_MODEL
            or upstream_health.get("provider_model")
            != flex_gateway.PROVIDER_MODEL
            or upstream_health.get("service_tier")
            != flex_gateway.SERVICE_TIER
            or not isinstance(upstream_health.get("budget"), dict)
            or upstream_health["budget"].get("max_cost_usd")
            != provider_contract["max_cost_usd"]
            or health.get("base_wrapper_sha256")
            != contract.sha256_file(ROOT / "scripts/controlled_locomo/gpt55_run_proxy.py")
            or health.get("controlled_wrapper_sha256")
            != contract.sha256_file(ROOT / "scripts/controlled_locomo/controlled_gpt55_run_proxy.py")
        ):
            raise contract.ControlledAnswerError("proxy health linkage differs")
        process_text = resolved_proxy_paths["process_log"].read_text(
            encoding="utf-8", errors="replace"
        )
        if re.search(r"traceback|proxy exception|address already in use", process_text, re.I):
            raise contract.ControlledAnswerError("proxy process log contains a fatal marker")
        count = 0
        with log_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise contract.ControlledAnswerError(
                        f"proxy log line {line_number} is invalid"
                    ) from exc
                if not isinstance(entry, dict) or entry.get("run_id") != run_id:
                    raise contract.ControlledAnswerError("proxy run linkage differs")
                event_id = entry.get("event_id")
                if not isinstance(event_id, str) or not event_id or event_id in event_ids:
                    raise contract.ControlledAnswerError("proxy event ID is invalid or duplicated")
                event_ids.add(event_id)
                question_id = entry.get("question_id")
                logical_call_id = entry.get("logical_call_id")
                if (
                    not isinstance(question_id, str)
                    or re.fullmatch(r"s\d+_q\d+", question_id) is None
                    or not isinstance(logical_call_id, str)
                    or question_id not in logical_call_id
                ):
                    raise contract.ControlledAnswerError("proxy question linkage differs")
                if entry.get("requested_model") != contract.EXPECTED_MODEL:
                    raise contract.ControlledAnswerError("proxy requested model differs")
                for field in ("request_sha256", "response_sha256"):
                    if re.fullmatch(r"[0-9a-f]{64}", str(entry.get(field))) is None:
                        raise contract.ControlledAnswerError(f"proxy {field} is invalid")
                if entry.get("client_http_attempts") != 1:
                    raise contract.ControlledAnswerError("proxy client HTTP count differs")
                unsupported = entry.get("unsupported_parameters")
                if not isinstance(unsupported, list) or not all(
                    isinstance(value, str) for value in unsupported
                ):
                    raise contract.ControlledAnswerError(
                        "proxy unsupported-parameter record differs"
                    )
                upstream = entry.get("upstream_http_attempts")
                if entry.get("status") == "success":
                    if (
                        entry.get("http_status") != 200
                        or entry.get("actual_model") != contract.EXPECTED_MODEL
                        or not isinstance(entry.get("response_id"), str)
                        or not isinstance(upstream, int)
                        or isinstance(upstream, bool)
                        or upstream < 1
                        or not isinstance(entry.get("usage"), dict)
                        or entry.get("provider_actual_model")
                        != flex_gateway.PROVIDER_MODEL
                        or entry.get("service_tier")
                        != flex_gateway.SERVICE_TIER
                        or entry.get("error") is not None
                    ):
                        raise contract.ControlledAnswerError("successful proxy entry differs")
                    gateway_request_id = entry.get("gateway_request_id")
                    if not isinstance(gateway_request_id, str) or not gateway_request_id:
                        raise contract.ControlledAnswerError(
                            "successful proxy gateway request ID is invalid"
                        )
                    for field in (
                        "gateway_request_sha256",
                        "provider_request_sha256",
                    ):
                        if re.fullmatch(
                            r"[0-9a-f]{64}", str(entry.get(field))
                        ) is None:
                            raise contract.ControlledAnswerError(
                                f"successful proxy {field} is invalid"
                            )
                elif entry.get("status") == "error":
                    if (
                        not isinstance(entry.get("http_status"), int)
                        or isinstance(entry.get("http_status"), bool)
                        or entry["http_status"] < 400
                        or not isinstance(upstream, int)
                        or isinstance(upstream, bool)
                        or upstream < 0
                        or not isinstance(entry.get("error"), str)
                        or not entry["error"]
                    ):
                        raise contract.ControlledAnswerError(
                            "failed proxy entry differs"
                        )
                else:
                    raise contract.ControlledAnswerError("proxy status differs")
                entries.append(entry)
                count += 1
        invocation_entries = entries[-count:] if count else []
        try:
            flex_report = flex_evidence.audit_window(
                provider_window,
                consumer_records=invocation_entries,
            )
        except flex_evidence.EvidenceError as exc:
            raise contract.ControlledAnswerError(
                f"proxy Flex provider window differs: {exc}"
            ) from exc
        invocation_reports.append(
            {
                "manifest": str(manifest_path.relative_to(output_dir)),
                "manifest_sha256": contract.sha256_file(manifest_path),
                "status": status or "completed",
                "recovery_action": manifest.get("recovery_action"),
                "start": manifest["start"],
                "start_sha256": manifest["start_sha256"],
                "log": manifest["log"],
                "log_sha256": manifest["log_sha256"],
                "requests": count,
                "flex_gateway": flex_report,
            }
        )
    return entries, {"invocations": invocation_reports, "entries": len(entries)}


def audit_question(
    *,
    output_dir: Path,
    input_record: Mapping[str, Any],
    raw_dataset: Sequence[Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    checkpoint_path: Path,
    proxy_entries: Sequence[Mapping[str, Any]] | None,
    require_proxy_log: bool,
) -> dict[str, Any]:
    question_id = str(input_record["question_id"])
    checkpoint = contract.read_json(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise contract.ControlledAnswerError(f"{question_id} checkpoint is invalid")
    if (
        checkpoint.get("schema_version") != contract.SCHEMA_VERSION
        or checkpoint.get("status") != "complete"
        or checkpoint.get("question_id") != question_id
    ):
        raise contract.ControlledAnswerError(f"{question_id} checkpoint contract differs")
    result_path = _resolve_inside(
        output_dir, checkpoint.get("result"), label=f"{question_id} result"
    )
    if checkpoint.get("result_sha256") != contract.sha256_file(result_path):
        raise contract.ControlledAnswerError(f"{question_id} result hash differs")
    attempt_dir = result_path.parent
    if checkpoint.get("attempt_dir") != str(attempt_dir.relative_to(output_dir)):
        raise contract.ControlledAnswerError(f"{question_id} attempt path differs")
    result = contract.read_json(result_path)
    if not isinstance(result, dict):
        raise contract.ControlledAnswerError(f"{question_id} result is invalid")
    if (
        result.get("schema_version") != contract.SCHEMA_VERSION
        or result.get("status") != "complete"
        or result.get("question_id") != question_id
        or result.get("method") != run_manifest["config"]["method"]
        or result.get("run_id") != run_manifest["run_id"]
        or result.get("input_record_sha256") != contract.canonical_hash(input_record)
        or result.get("question_sha256")
        != contract.sha256_bytes(str(input_record["question"]).encode("utf-8"))
        or result.get("preregistration_sha256")
        != run_manifest["config"]["preregistration_sha256"]
    ):
        raise contract.ControlledAnswerError(f"{question_id} result linkage differs")

    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise contract.ControlledAnswerError(f"{question_id} artifacts are missing")
    trace_path = attempt_dir / str(artifacts.get("visible_token_trace"))
    gate_manifest_path = attempt_dir / str(artifacts.get("visible_token_manifest"))
    ledger_path = attempt_dir / str(artifacts.get("question_ledger"))
    contract.ensure_distinct_paths(
        {"result": result_path, "trace": trace_path, "manifest": gate_manifest_path, "ledger": ledger_path}
    )
    for path, hash_key in (
        (trace_path, "visible_token_trace_sha256"),
        (gate_manifest_path, "visible_token_manifest_sha256"),
        (ledger_path, "question_ledger_sha256"),
    ):
        if not path.is_file() or artifacts.get(hash_key) != contract.sha256_file(path):
            raise contract.ControlledAnswerError(f"{question_id} artifact hash differs")
    gate_audit = audit_visible_token_trace(
        trace_path,
        manifest_path=gate_manifest_path,
        require_complete=True,
    )
    if gate_audit.get("audit_status") != "pass":
        raise contract.ControlledAnswerError(
            f"{question_id} R004 reconstruction failed: {gate_audit.get('errors')}"
        )
    trace = _trace_records(trace_path)
    header = trace[0]
    finalizer = trace[-1]
    deliveries = trace[1:-1]
    logical_call_id = result.get("logical_call_id")
    if (
        not isinstance(logical_call_id, str)
        or header.get("run_id") != logical_call_id
        or finalizer.get("run_id") != logical_call_id
        or header.get("metadata", {}).get("question_id") != question_id
        or header.get("metadata", {}).get("preregistration_sha256")
        != run_manifest["config"]["preregistration_sha256"]
    ):
        raise contract.ControlledAnswerError(f"{question_id} gate linkage differs")
    rendered = contract.render_memories_individually(input_record["memories"])
    if len(deliveries) != len(rendered):
        raise contract.ControlledAnswerError(f"{question_id} did not gate every memory")
    delivered_texts: list[str] = []
    for expected, delivery in zip(rendered, deliveries):
        if (
            delivery.get("record_type") != "delivery"
            or delivery.get("kind") != "tool_result"
            or delivery.get("event_id") != f"memory-{expected.rendered_rank:04d}"
            or delivery.get("raw", {}).get("text") != expected.rendered_text
            or delivery.get("metadata", {}).get("input_index") != expected.input_index
            or delivery.get("metadata", {}).get("rendered_rank") != expected.rendered_rank
            or delivery.get("cumulative_source_resolution_tokens") != 0
        ):
            raise contract.ControlledAnswerError(
                f"{question_id} gated rendering/order differs"
            )
        delivered = delivery.get("delivered")
        if delivered is not None:
            delivered_texts.append(delivered["text"])
    memory_block = "".join(delivered_texts)
    # Independent prompt reconstruction.  No dataset gold/evidence/category or
    # raw non-delivered text is used by this expression.
    reconstructed_prompt = contract.ANSWER_PROMPT.format(
        memories=memory_block,
        question=input_record["question"],
    )
    prompt_record = result.get("prompt")
    if not isinstance(prompt_record, dict):
        raise contract.ControlledAnswerError(f"{question_id} prompt record is missing")
    prompt_hash = contract.sha256_bytes(reconstructed_prompt.encode("utf-8"))
    if prompt_record.get("sha256") != prompt_hash:
        raise contract.ControlledAnswerError(f"{question_id} prompt hash differs")
    tokenizer = TokenCounter.from_identity(header["config"]["tokenizer"])
    prompt_tokens = tokenizer.count(reconstructed_prompt)
    if (
        prompt_record.get("local_tokens") != prompt_tokens
        or prompt_record.get("memory_payload_sha256")
        != contract.sha256_bytes(memory_block.encode("utf-8"))
        or prompt_record.get("local_context_check_passed") is not True
        or prompt_tokens + prompt_record.get("answer_completion_reservation_tokens", -1)
        > prompt_record.get("model_context_limit_tokens", -1)
    ):
        raise contract.ControlledAnswerError(f"{question_id} context accounting differs")

    budget = result.get("budget")
    summary = {
        "cumulative_visible_tokens": gate_audit.get("cumulative_visible_tokens"),
        "cumulative_source_resolution_tokens": gate_audit.get(
            "cumulative_source_resolution_tokens"
        ),
    }
    if not isinstance(budget, dict) or not isinstance(summary, dict):
        raise contract.ControlledAnswerError(f"{question_id} budget record is missing")
    if (
        budget.get("tokenizer") != header["config"]["tokenizer"]
        or budget.get("visible_tokens") != summary.get("cumulative_visible_tokens")
        or budget.get("source_resolution_tokens") != 0
        or summary.get("cumulative_source_resolution_tokens") != 0
    ):
        raise contract.ControlledAnswerError(f"{question_id} budget summary differs")
    policy = run_manifest["config"]["budget_policy"]
    if budget.get("policy") != policy:
        raise contract.ControlledAnswerError(f"{question_id} budget policy differs")
    if policy == contract.FULL_CONTEXT_POLICY:
        if (
            budget.get("declared_tokens") != "unbounded"
            or any(item.get("decision") != "delivered" for item in deliveries)
            or budget.get("full_context_matched_cap_claimed") is not False
            or finalizer.get("reason") != "full_context_accounted_complete"
            or budget.get("visible_tokens") != sum(
                tokenizer.count(item.rendered_text) for item in rendered
            )
        ):
            raise contract.ControlledAnswerError(
                f"{question_id} full-context accounting/truncation differs"
            )
    elif (
        budget.get("declared_tokens") != contract.EXPECTED_HARD_BUDGET
        or header["config"].get("configured_budget_tokens")
        != contract.EXPECTED_HARD_BUDGET
    ):
        raise contract.ControlledAnswerError(f"{question_id} hard cap differs")

    answer = result.get("answer")
    if not isinstance(answer, dict):
        raise contract.ControlledAnswerError(f"{question_id} answer record is missing")
    if (
        answer.get("requested_model") != contract.EXPECTED_MODEL
        or answer.get("response_model") != contract.EXPECTED_MODEL
        or not isinstance(answer.get("response_id"), str)
        or not answer["response_id"]
        or answer.get("raw_output_sha256")
        != contract.sha256_bytes(str(answer.get("raw_output", "")).encode("utf-8"))
        or answer.get("text") != contract.extract_answer(answer.get("raw_output", ""))
        or answer.get("logical_answer_calls") != 1
        or not isinstance(answer.get("client_http_attempts"), int)
        or answer["client_http_attempts"] < 1
        or not isinstance(answer.get("upstream_http_attempts"), int)
        or answer["upstream_http_attempts"] < 1
        or not isinstance(answer.get("usage"), dict)
    ):
        raise contract.ControlledAnswerError(f"{question_id} answer identity/usage differs")
    for field in (
        "exclusive_proxy_event_ids",
        "request_sha256s",
        "response_sha256s",
        "unsupported_parameters",
    ):
        if not isinstance(answer.get(field), list) or not all(
            isinstance(value, str) and value for value in answer[field]
        ):
            raise contract.ControlledAnswerError(
                f"{question_id} answer {field} differs"
            )
    if any(
        len(answer[field]) != answer["client_http_attempts"]
        for field in (
            "exclusive_proxy_event_ids",
            "request_sha256s",
            "response_sha256s",
        )
    ):
        raise contract.ControlledAnswerError(
            f"{question_id} answer physical-attempt arrays differ"
        )
    if (
        prompt_record.get("provider_reported_total_tokens")
        != answer["usage"].get("total_tokens")
        or prompt_record.get("provider_context_check_passed") is not True
        or answer["usage"].get("total_tokens", -1)
        > prompt_record["model_context_limit_tokens"]
    ):
        raise contract.ControlledAnswerError(
            f"{question_id} provider context accounting differs"
        )
    actual_usage = finalizer.get("actual_model_usage")
    if not isinstance(actual_usage, dict) or any(
        actual_usage.get(key) != value
        for key, value in {
            "requested_model": answer["requested_model"],
            "response_model": answer["response_model"],
            "response_id": answer["response_id"],
            "logical_answer_calls": answer["logical_answer_calls"],
            "client_http_attempts": answer["client_http_attempts"],
            "upstream_http_attempts": answer["upstream_http_attempts"],
            "provider_usage": answer["usage"],
            "exclusive_proxy_event_ids": answer["exclusive_proxy_event_ids"],
            "unsupported_parameters": answer["unsupported_parameters"],
        }.items()
    ):
        raise contract.ControlledAnswerError(f"{question_id} gate answer usage differs")

    ledger_records = contract.audit_ledger(ledger_path, expected_run_id=logical_call_id)
    if (
        not ledger_records
        or ledger_records[0]["event"] != "question_started"
        or ledger_records[-1]["event"] != "question_completed"
    ):
        raise contract.ControlledAnswerError(f"{question_id} ledger lifecycle differs")
    started_attempts = [
        item for item in ledger_records if item["event"] == "physical_http_attempt_started"
    ]
    finished_attempts = [
        item for item in ledger_records if item["event"] == "physical_http_attempt_finished"
    ]
    if len(started_attempts) != answer["client_http_attempts"] or len(finished_attempts) != len(
        started_attempts
    ):
        raise contract.ControlledAnswerError(f"{question_id} physical attempt ledger differs")
    if [item["payload"]["request_sha256"] for item in started_attempts] != answer.get(
        "request_sha256s"
    ):
        raise contract.ControlledAnswerError(f"{question_id} request hashes differ")
    if (
        [item["payload"].get("response_sha256") for item in finished_attempts]
        != answer["response_sha256s"]
        or [item["payload"].get("proxy_event_id") for item in finished_attempts]
        != answer["exclusive_proxy_event_ids"]
        or [item["payload"].get("status") for item in finished_attempts][-1]
        != "accepted"
        or any(
            item["payload"].get("status") != "error"
            for item in finished_attempts[:-1]
        )
        or sum(
            int(item["payload"].get("upstream_http_attempts") or 0)
            for item in finished_attempts
        )
        != answer["upstream_http_attempts"]
        or sorted(
            {
                parameter
                for item in finished_attempts
                for parameter in item["payload"].get("unsupported_parameters", [])
            }
        )
        != answer["unsupported_parameters"]
    ):
        raise contract.ControlledAnswerError(
            f"{question_id} physical attempt outcome ledger differs"
        )

    if require_proxy_log:
        if proxy_entries is None:
            raise contract.ControlledAnswerError("proxy log was required but absent")
        linked = [
            entry
            for entry in proxy_entries
            if entry.get("question_id") == question_id
            and entry.get("logical_call_id") == logical_call_id
        ]
        if len(linked) != answer["client_http_attempts"]:
            raise contract.ControlledAnswerError(f"{question_id} proxy attempt count differs")
        if [entry["request_sha256"] for entry in linked] != answer["request_sha256s"]:
            raise contract.ControlledAnswerError(f"{question_id} proxy request hashes differ")
        if [entry["response_sha256"] for entry in linked] != answer["response_sha256s"]:
            raise contract.ControlledAnswerError(f"{question_id} proxy response hashes differ")
        if [entry["event_id"] for entry in linked] != answer["exclusive_proxy_event_ids"]:
            raise contract.ControlledAnswerError(f"{question_id} proxy event IDs differ")
        successes = [entry for entry in linked if entry.get("status") == "success"]
        if len(successes) != 1:
            raise contract.ControlledAnswerError(f"{question_id} accepted proxy count differs")
        success = successes[0]
        if (
            success.get("event_id") not in answer["exclusive_proxy_event_ids"]
            or success.get("response_id") != answer["response_id"]
            or success.get("actual_model") != answer["response_model"]
            or success.get("usage") != answer["usage"]
            or sum(int(entry.get("upstream_http_attempts") or 0) for entry in linked)
            != answer["upstream_http_attempts"]
            or sorted(
                {
                    parameter
                    for entry in linked
                    for parameter in entry["unsupported_parameters"]
                }
            )
            != answer["unsupported_parameters"]
        ):
            raise contract.ControlledAnswerError(f"{question_id} accepted proxy linkage differs")

    reference = contract.canonical_scoring_reference(raw_dataset, question_id)
    scoring = result.get("scoring_reference")
    expected_scoring = {
        "category": reference["category"],
        "canonical_gold": reference["canonical_gold"],
        "canonical_gold_source": reference["canonical_gold_source"],
        "adversarial_distractor": reference["adversarial_distractor"],
        "evidence": reference["evidence"],
        "attached_after_answer": True,
        "baseline_input_gold_authoritative": False,
    }
    if scoring != expected_scoring:
        raise contract.ControlledAnswerError(f"{question_id} scoring labels differ")
    if reference["category"] == 5:
        expected_legacy_gold = (
            reference["canonical_gold"]
            if reference["canonical_gold_source"] == "raw_dataset.answer"
            else reference["adversarial_distractor"]
        )
        if (
            input_record["gold"] != expected_legacy_gold
            or (
                reference["canonical_gold_source"] == "cat5_canonical_abstention"
                and reference["canonical_gold"] != contract.CAT5_CANONICAL_ABSTENTION
            )
        ):
            raise contract.ControlledAnswerError(f"{question_id} cat5 label role differs")
    return {
        "question_id": question_id,
        "category": reference["category"],
        "logical_calls": answer["logical_answer_calls"],
        "client_http_attempts": answer["client_http_attempts"],
        "upstream_http_attempts": answer["upstream_http_attempts"],
        "visible_tokens": budget["visible_tokens"],
        "source_resolution_tokens": 0,
        "response_id": answer["response_id"],
    }


def audit_run(output_dir: Path, *, require_complete_manifest: bool = True) -> dict[str, Any]:
    output_dir = output_dir.expanduser().absolute()
    contract.reject_symlink_components(output_dir)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise contract.ControlledAnswerError("answer output directory is invalid")
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "run_manifest.json"
    manifest = contract.read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != contract.SCHEMA_VERSION:
        raise contract.ControlledAnswerError("answer run manifest differs")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise contract.ControlledAnswerError("answer run configuration is missing")
    if manifest.get("source_hashes") != {
        "runner": contract.sha256_file(ROOT / "scripts/controlled_locomo/run_controlled_locomo_answers.py"),
        "contract": contract.sha256_file(Path(contract.__file__).resolve()),
        "run_proxy": contract.sha256_file(ROOT / "scripts/controlled_locomo/controlled_gpt55_run_proxy.py"),
        "base_run_proxy": contract.sha256_file(ROOT / "scripts/controlled_locomo/gpt55_run_proxy.py"),
        "upstream_proxy": contract.sha256_file(
            ROOT / "scripts/gateways/openai_gpt55_flex_gateway.py"
        ),
        "flex_gateway_evidence": contract.sha256_file(
            ROOT / "scripts/gateways/openai_gpt55_flex_gateway_evidence.py"
        ),
        "auditor": contract.sha256_file(Path(__file__).resolve()),
        "visible_token_budget": contract.sha256_file(ROOT / "src/evaluation/visible_token_budget.py"),
        "visible_token_audit": contract.sha256_file(ROOT / "src/evaluation/visible_token_audit.py"),
        "prompts": contract.sha256_file(ROOT / "src/evaluation/prompts.py"),
    }:
        raise contract.ControlledAnswerError("answer run source hashes differ")
    configured_python = Path(config.get("python", "")).absolute()
    if (
        not configured_python.is_file()
        or config.get("python_resolved") != str(configured_python.resolve())
        or config.get("python_sha256")
        != contract.sha256_file(configured_python.resolve())
        or config.get("runner_python_resolved") != str(configured_python.resolve())
        or config.get("runner_python_sha256")
        != contract.sha256_file(configured_python.resolve())
        or Path(sys.executable).resolve() != configured_python.resolve()
    ):
        raise contract.ControlledAnswerError("answer runtime Python fingerprint differs")
    input_dir = Path(config["input_dir"]).resolve()
    preregistration = Path(config["preregistration"]).resolve()
    contract.ensure_distinct_paths(
        {"output": output_dir, "input": input_dir, "preregistration": preregistration}
    )
    protocol = contract.validate_preregistration(
        preregistration,
        expected_file_sha256=config["preregistration_sha256"],
    )
    if protocol["protocol_content_sha256"] != config["protocol_content_sha256"]:
        raise contract.ControlledAnswerError("protocol content linkage differs")
    if protocol["answerer"]["prompt_source_hashes"] != contract.prompt_source_hashes(ROOT):
        raise contract.ControlledAnswerError("prompt/gate source hashes differ")
    baseline_report = baseline_auditor.audit(input_dir)
    if baseline_report.get("status") != "passed" or baseline_report.get("method") != config["method"]:
        raise contract.ControlledAnswerError("baseline re-audit failed")
    questions_payload = contract.read_json(input_dir / "questions.json")
    questions = [
        item for item in questions_payload if item.get("question_id") != "_build_stats"
    ]
    if len(questions) != contract.EXPECTED_QUESTIONS:
        raise contract.ControlledAnswerError("answer input inventory differs")
    input_manifest = contract.read_json(input_dir / "run_manifest.json")
    dataset_path = Path(input_manifest["config"]["dataset"]).resolve()
    raw_dataset = contract.read_json(dataset_path)
    proxy_entries, proxy_report = _load_proxy_entries(output_dir, manifest["run_id"])
    completed_dir = output_dir / "completed"
    checkpoints = sorted(completed_dir.glob("s*_q*.json")) if completed_dir.exists() else []
    if len(checkpoints) != len(questions):
        raise contract.ControlledAnswerError(
            f"completed inventory differs: {len(checkpoints)} != {len(questions)}"
        )
    checkpoint_by_id = {path.stem: path for path in checkpoints}
    if set(checkpoint_by_id) != {record["question_id"] for record in questions}:
        raise contract.ControlledAnswerError("completed question IDs differ")
    reports = [
        audit_question(
            output_dir=output_dir,
            input_record=record,
            raw_dataset=raw_dataset,
            run_manifest=manifest,
            protocol=protocol,
            checkpoint_path=checkpoint_by_id[record["question_id"]],
            proxy_entries=proxy_entries,
            require_proxy_log=True,
        )
        for record in questions
    ]
    categories = Counter(report["category"] for report in reports)
    if dict(sorted(categories.items())) != contract.EXPECTED_CATEGORIES:
        raise contract.ControlledAnswerError("audited category inventory differs")
    response_ids = [report["response_id"] for report in reports]
    if len(set(response_ids)) != contract.EXPECTED_QUESTIONS:
        raise contract.ControlledAnswerError("answer response IDs are duplicated")
    logical_calls = sum(report["logical_calls"] for report in reports)
    completed_client_attempts = sum(
        report["client_http_attempts"] for report in reports
    )
    completed_upstream_attempts = sum(
        report["upstream_http_attempts"] for report in reports
    )
    if logical_calls != contract.EXPECTED_QUESTIONS:
        raise contract.ControlledAnswerError("logical answer-call inventory differs")
    completed_logical_ids = {
        contract.read_json(checkpoint_by_id[record["question_id"]])["attempt_dir"]: (
            contract.read_json(
                _resolve_inside(
                    output_dir,
                    contract.read_json(checkpoint_by_id[record["question_id"]])["result"],
                    label="completed result",
                )
            )["logical_call_id"]
        )
        for record in questions
    }
    completed_logical_id_set = set(completed_logical_ids.values())
    abandoned_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in proxy_entries:
        logical_id = str(entry["logical_call_id"])
        if logical_id in completed_logical_id_set:
            continue
        key = (str(entry["question_id"]), logical_id)
        abandoned_groups.setdefault(key, []).append(entry)
    abandoned_report: list[dict[str, Any]] = []
    for (question_id, logical_id), entries in sorted(abandoned_groups.items()):
        prefix = f"{manifest['run_id']}:{question_id}:"
        if not logical_id.startswith(prefix):
            raise contract.ControlledAnswerError("abandoned logical call ID differs")
        attempt_name = logical_id[len(prefix) :]
        if re.fullmatch(r"attempt-\d{4}", attempt_name) is None:
            raise contract.ControlledAnswerError("abandoned attempt name differs")
        attempt_dir = output_dir / "questions" / question_id / "attempts" / attempt_name
        ledger_path = attempt_dir / "question_ledger.jsonl"
        ledger = contract.audit_ledger(ledger_path, expected_run_id=logical_id)
        started = [
            item for item in ledger if item["event"] == "physical_http_attempt_started"
        ]
        finished = [
            item for item in ledger if item["event"] == "physical_http_attempt_finished"
        ]
        if (
            len(started) != len(entries)
            or len(finished) > len(started)
            or [item["payload"]["request_sha256"] for item in started]
            != [entry["request_sha256"] for entry in entries]
        ):
            raise contract.ControlledAnswerError("abandoned attempt ledger differs")
        abandoned_report.append(
            {
                "question_id": question_id,
                "logical_call_id": logical_id,
                "client_http_attempts": len(entries),
                "ledger": str(ledger_path.relative_to(output_dir)),
                "ledger_sha256": contract.sha256_file(ledger_path),
                "status": "preserved_not_scored",
            }
        )
    if any(
        not isinstance(entry.get("upstream_http_attempts"), int)
        or isinstance(entry.get("upstream_http_attempts"), bool)
        or entry["upstream_http_attempts"] < 0
        for entry in proxy_entries
    ):
        raise contract.ControlledAnswerError(
            "a proxy entry lacks an exact upstream physical-attempt count"
        )
    client_attempts = len(proxy_entries)
    upstream_attempts = sum(entry["upstream_http_attempts"] for entry in proxy_entries)
    if completed_client_attempts + sum(
        item["client_http_attempts"] for item in abandoned_report
    ) != client_attempts:
        raise contract.ControlledAnswerError("proxy entry coverage differs")
    linked_pairs = {
        (entry.get("question_id"), entry.get("logical_call_id"))
        for entry in proxy_entries
    }
    run_ledger = output_dir / "run_ledger.jsonl"
    contract.audit_ledger(run_ledger, expected_run_id=manifest["run_id"])
    report = {
        "schema_version": contract.SCHEMA_VERSION,
        "status": "passed",
        "run_id": manifest["run_id"],
        "method": config["method"],
        "budget_policy": config["budget_policy"],
        "preregistration": {
            "path": str(preregistration),
            "sha256": config["preregistration_sha256"],
            "protocol_content_sha256": config["protocol_content_sha256"],
            "formal_matrix_verified": True,
            "gate_preregistration_verified": True,
        },
        "inventory": {
            "questions": len(reports),
            "primary_cat1_4": sum(categories[value] for value in (1, 2, 3, 4)),
            "adversarial_cat5": categories[5],
            "categories": {str(key): value for key, value in sorted(categories.items())},
            "gate_traces": len(reports),
            "gate_manifests": len(reports),
            "source_resolution_tokens": sum(
                report["source_resolution_tokens"] for report in reports
            ),
        },
        "calls": {
            "successful_logical_answer_calls": logical_calls,
            "abandoned_logical_answer_calls": len(abandoned_report),
            "actual_logical_answer_calls": logical_calls + len(abandoned_report),
            "completed_question_client_http_attempts": completed_client_attempts,
            "client_http_attempts": client_attempts,
            "completed_question_upstream_http_attempts": completed_upstream_attempts,
            "upstream_http_attempts": upstream_attempts,
            "unique_response_ids": len(set(response_ids)),
            "actual_model": contract.EXPECTED_MODEL,
        },
        "proxy": {
            **proxy_report,
            "linked_question_logical_pairs": len(linked_pairs),
            "abandoned_attempts": abandoned_report,
        },
        "prompt_reconstruction": {
            "questions_verified": len(reports),
            "delivered_payload_only": True,
            "raw_non_delivered_payload_excluded": True,
            "gold_evidence_category_not_prompt_inputs": True,
        },
        "scoring_labels": {
            "raw_dataset_reconstructed": True,
            "category_5_distractor_not_gold": True,
            "category_5_separate_inventory": categories[5],
        },
        "run_ledger_sha256": contract.sha256_file(run_ledger),
    }
    complete_path = output_dir / "complete.json"
    if require_complete_manifest:
        complete = contract.read_json(complete_path)
        if not isinstance(complete, dict):
            raise contract.ControlledAnswerError("complete manifest is invalid")
        expected = {
            "schema_version": contract.SCHEMA_VERSION,
            "status": "complete",
            "run_id": manifest["run_id"],
            "inventory": report["inventory"],
            "calls": report["calls"],
            "proxy": report["proxy"],
            "run_ledger_sha256": report["run_ledger_sha256"],
            "preregistration_sha256": config["preregistration_sha256"],
        }
        for key, value in expected.items():
            if complete.get(key) != value:
                raise contract.ControlledAnswerError(
                    f"complete manifest differs in {key}"
                )
        report["complete_manifest_sha256"] = contract.sha256_file(complete_path)
    elif complete_path.exists():
        # Pre-final audit may be repeated during a resume, but a pre-existing
        # complete marker must already validate fully.
        return audit_run(output_dir, require_complete_manifest=True)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--allow-missing-complete", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_run(
        args.output_dir,
        require_complete_manifest=not args.allow_missing_complete,
    )
    if args.report is not None:
        contract.atomic_json_no_clobber(args.report.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except contract.ControlledAnswerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
