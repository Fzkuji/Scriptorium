#!/usr/bin/env python3
"""Independently audit formal and synthetic R116 shared-answer artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_readonly_nativemem_control as generic  # noqa: E402
import controlled_locomo_answer_contract as answer_contract  # noqa: E402
import r116_formal_contract as contract  # noqa: E402
import readonly_nativemem_control as readonly_control  # noqa: E402
from src.evaluation.durable_model_ledger import read_ledger  # noqa: E402
from src import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402


RUN_SCHEMA = "nativemem.r116-formal-run.v1"
CHECKPOINT_SCHEMA = "nativemem.r116-formal-question-checkpoint.v1"
COMPLETE_SCHEMA = "nativemem.r116-formal-complete.v1"
AUDIT_SCHEMA = "nativemem.r116-formal-audit.v1"


class R116FormalAuditError(RuntimeError):
    pass


def _strict_json(path: Path) -> Any:
    return answer_contract.read_json(path)


def _inside_file(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise R116FormalAuditError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise R116FormalAuditError(f"{label} path escapes output")
    candidate = root / relative
    answer_contract.reject_symlink_components(candidate)
    if candidate.is_symlink() or not candidate.is_file():
        raise R116FormalAuditError(f"{label} is not a regular file")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise R116FormalAuditError(f"{label} path escapes output") from exc
    return resolved


def _inside_directory(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise R116FormalAuditError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise R116FormalAuditError(f"{label} path escapes output")
    candidate = root / relative
    answer_contract.reject_symlink_components(candidate)
    if candidate.is_symlink() or not candidate.is_dir():
        raise R116FormalAuditError(f"{label} is not a regular directory")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise R116FormalAuditError(f"{label} path escapes output") from exc
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if not payload.endswith(b"\n"):
        raise R116FormalAuditError(f"JSONL final line is incomplete: {path}")
    records = []
    for number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise R116FormalAuditError(f"invalid JSONL line {number}: {path}") from exc
        if not isinstance(value, dict):
            raise R116FormalAuditError(f"JSONL line {number} is not an object")
        records.append(value)
    return records


def _validate_content_hash(payload: Mapping[str, Any], field: str) -> None:
    content = dict(payload)
    expected = content.pop(field, None)
    if expected != answer_contract.canonical_hash(content):
        raise R116FormalAuditError(f"{field} differs")


def _source_binding_hash(binding: Mapping[str, Any]) -> None:
    _validate_content_hash(binding, "binding_sha256")
    inventory = binding.get("inventory")
    if (
        not isinstance(inventory, list)
        or answer_contract.canonical_hash(inventory)
        != binding.get("inventory_sha256")
    ):
        raise R116FormalAuditError("source inventory hash differs")


def _memory_for_question(
    *,
    output_dir: Path,
    spec: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    formal: bool,
) -> Path:
    if formal:
        source_root = ROOT / str(source_binding["source_root"])
        return contract.source_memory_path(str(spec["benchmark"]), source_root, spec)
    entry = source_binding["inventory"][int(spec["dataset_index"])]
    memory_record = entry.get("memory")
    if not isinstance(memory_record, Mapping):
        raise R116FormalAuditError("synthetic memory descriptor is absent")
    memory = Path(str(memory_record.get("path", "")))
    answer_contract.reject_symlink_components(memory)
    if memory.is_symlink() or not memory.is_dir():
        raise R116FormalAuditError("synthetic memory root is invalid")
    try:
        memory.resolve().relative_to(output_dir.resolve())
    except ValueError as exc:
        raise R116FormalAuditError("synthetic memory escapes output") from exc
    if readonly_control.memory_descriptor(memory) != memory_record:
        raise R116FormalAuditError("synthetic memory descriptor differs")
    return memory


def _proxy_events_from_result(
    *, attempt: Path, result: Mapping[str, Any], formal: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reconcile durable ledger evidence with exact exclusive log records."""

    events: list[dict[str, Any]] = []
    response_ids: list[str] = []
    retrieval_records = read_ledger(attempt / "retrieval_model_ledger.jsonl")
    starts = {
        record["logical_call_id"]: record
        for record in retrieval_records
        if record.get("event") == "model_call_started"
    }
    finishes = {
        record["logical_call_id"]: record
        for record in retrieval_records
        if record.get("event") == "model_call_finished"
    }
    if set(starts) != set(finishes):
        raise R116FormalAuditError("retrieval model lifecycle differs")
    for logical_id, finish in finishes.items():
        response_ids.append(str(finish.get("response_id", "")))
        evidence = finish.get("proxy_evidence")
        if not isinstance(evidence, Mapping):
            raise R116FormalAuditError("retrieval proxy evidence is absent")
        if formal and evidence.get("mode") != "exclusive_proxy":
            raise R116FormalAuditError("formal retrieval proxy mode differs")
        recorded_events = evidence.get("events")
        if not isinstance(recorded_events, list):
            raise R116FormalAuditError("retrieval proxy events are invalid")
        for event in recorded_events:
            if not isinstance(event, dict):
                raise R116FormalAuditError("retrieval proxy event is not an object")
            expected = {
                "logical_call_id": logical_id,
                "question_id": result["question_id"],
                "status": "success",
                "requested_model": contract.MODEL,
                "actual_model": contract.MODEL,
                "response_id": finish["response_id"],
            }
            if (
                any(event.get(key) != value for key, value in expected.items())
                or event.get("usage") != finish.get("usage")
            ):
                raise R116FormalAuditError("retrieval proxy event linkage differs")
            events.append(event)
        if formal:
            prefix = evidence.get("log_prefix")
            if not isinstance(prefix, Mapping):
                raise R116FormalAuditError("retrieval proxy prefix is absent")
            log_path = Path(str(prefix.get("path", "")))
            live = [
                row
                for row in _read_jsonl(log_path)
                if row.get("logical_call_id") == logical_id
            ]
            if live != recorded_events:
                raise R116FormalAuditError("retrieval proxy log differs")

    answer = result.get("answer")
    if not isinstance(answer, Mapping):
        raise R116FormalAuditError("answer result is absent")
    response_ids.append(str(answer.get("response_id", "")))
    answer_evidence = answer.get("proxy_evidence")
    if not isinstance(answer_evidence, Mapping):
        raise R116FormalAuditError("answer proxy evidence is absent")
    if formal and answer_evidence.get("mode") != "exclusive_proxy":
        raise R116FormalAuditError("formal answer proxy mode differs")
    answer_events = answer_evidence.get("events")
    if not isinstance(answer_events, list):
        raise R116FormalAuditError("answer proxy events are invalid")
    logical_id = f"{result['run_id']}:{result['question_id']}:{result['condition']}:answer"
    request_hashes = answer.get("request_sha256s")
    if not isinstance(request_hashes, list):
        raise R116FormalAuditError("answer request hashes are invalid")
    for event in answer_events:
        if not isinstance(event, dict):
            raise R116FormalAuditError("answer proxy event is not an object")
        common = {
            "logical_call_id": logical_id,
            "question_id": result["question_id"],
            "requested_model": contract.MODEL,
        }
        if any(event.get(key) != value for key, value in common.items()):
            raise R116FormalAuditError("answer proxy event linkage differs")
        if event.get("status") == "success":
            if (
                event.get("actual_model") != contract.MODEL
                or event.get("response_id") != answer["response_id"]
                or event.get("usage") != answer.get("usage")
            ):
                raise R116FormalAuditError("answer success identity differs")
        elif event.get("status") == "error":
            if event.get("actual_model") not in {None, contract.MODEL}:
                raise R116FormalAuditError("answer error actual model differs")
        else:
            raise R116FormalAuditError("answer proxy event status differs")
        if event.get("request_sha256") not in request_hashes:
            raise R116FormalAuditError("answer proxy request hash differs")
        events.append(event)
    successes = [event for event in answer_events if event.get("status") == "success"]
    if len(successes) != 1:
        raise R116FormalAuditError("answer must have exactly one successful proxy event")
    if formal:
        prefix = answer_evidence.get("log_prefix")
        if not isinstance(prefix, Mapping):
            raise R116FormalAuditError("answer proxy prefix is absent")
        log_path = Path(str(prefix.get("path", "")))
        live = [
            row
            for row in _read_jsonl(log_path)
            if row.get("logical_call_id") == logical_id
        ]
        if live != answer_events:
            raise R116FormalAuditError("answer proxy log differs")
    if any(not response_id for response_id in response_ids):
        raise R116FormalAuditError("model response ID is empty")
    return events, response_ids


def audit_question_checkpoint(
    *,
    output_dir: Path,
    spec: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    formal: bool,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().absolute()
    question_root = output_dir / "questions" / str(spec["artifact_id"])
    checkpoint_path = question_root / "checkpoint.json"
    checkpoint = _strict_json(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise R116FormalAuditError("question checkpoint is not an object")
    _validate_content_hash(checkpoint, "checkpoint_content_sha256")
    identity = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "complete",
        "benchmark": spec["benchmark"],
        "question_id": spec["question_id"],
        "artifact_id": spec["artifact_id"],
        "dataset_index": spec["dataset_index"],
        "actual_model": contract.MODEL,
    }
    if any(checkpoint.get(key) != value for key, value in identity.items()):
        raise R116FormalAuditError("question checkpoint identity differs")
    input_path = _inside_file(
        output_dir, checkpoint.get("input_binding", {}).get("path"), label="input binding"
    )
    attempt = _inside_directory(
        output_dir, checkpoint.get("attempt", {}).get("path"), label="attempt"
    )
    recall_path = _inside_file(
        output_dir,
        checkpoint.get("source_recall", {}).get("path"),
        label="source recall",
    )
    audit_path = _inside_file(
        output_dir,
        checkpoint.get("question_audit", {}).get("path"),
        label="question audit",
    )
    expected_paths = {
        "input": question_root / "input_binding.json",
        "attempt": question_root / "attempt-0001",
        "recall": question_root / "source_recall.json",
        "audit": question_root / "question_audit.json",
    }
    actual_paths = {
        "input": input_path,
        "attempt": attempt,
        "recall": recall_path,
        "audit": audit_path,
    }
    if any(actual_paths[key] != value.resolve() for key, value in expected_paths.items()):
        raise R116FormalAuditError("question artifact path differs")
    for record_key, path, hash_key in (
        ("input_binding", input_path, "sha256"),
        ("source_recall", recall_path, "sha256"),
        ("question_audit", audit_path, "sha256"),
    ):
        if checkpoint[record_key].get(hash_key) != answer_contract.sha256_file(path):
            raise R116FormalAuditError(f"{record_key} hash differs")
    result_path = attempt / "result.json"
    if checkpoint["attempt"].get("result_sha256") != answer_contract.sha256_file(
        result_path
    ):
        raise R116FormalAuditError("question result hash differs")

    memory = _memory_for_question(
        output_dir=output_dir,
        spec=spec,
        source_binding=source_binding,
        formal=formal,
    )
    turn_index, session_by_turn = contract.turn_index_and_session_map(spec)
    preregistration = contract.DEFAULT_PREREGISTRATION
    expected_binding = contract.input_binding(
        preregistration=preregistration,
        source_binding=source_binding,
        spec=spec,
        memory_root=memory,
        turn_index=turn_index,
        session_by_turn=session_by_turn,
    )
    if _strict_json(input_path) != expected_binding:
        raise R116FormalAuditError("question input binding differs")
    generic_gold = (
        list(spec["gold_source_ids"])
        if spec["benchmark"] == contract.LOCOMO
        else []
    )
    generic_eligible = (
        bool(spec["source_recall_eligible"])
        if spec["benchmark"] == contract.LOCOMO
        else False
    )
    generic_report = generic.audit_question(
        artifact_dir=attempt,
        memory_root=memory,
        method=contract.METHOD,
        condition=contract.CONDITION,
        question_id=str(spec["question_id"]),
        question=str(spec["question"]),
        gold_source_ids=generic_gold,
        source_recall_eligible=generic_eligible,
        formal=formal,
    )
    if _strict_json(audit_path) != generic_report:
        raise R116FormalAuditError("recorded generic question audit differs")
    expected_recall = contract.compute_source_recall(
        artifact_dir=attempt,
        spec=spec,
        session_by_turn=session_by_turn,
    )
    recorded_recall = _strict_json(recall_path)
    if recorded_recall != expected_recall:
        raise R116FormalAuditError("benchmark source recall differs")
    contract.validate_source_recall_hash(recorded_recall)
    result = _strict_json(result_path)
    events, response_ids = _proxy_events_from_result(
        attempt=attempt, result=result, formal=formal
    )
    if checkpoint.get("response_ids") != response_ids:
        raise R116FormalAuditError("checkpoint response IDs differ")
    if len(set(response_ids)) != len(response_ids):
        raise R116FormalAuditError("question response IDs are duplicated")
    if checkpoint.get("memory_sha256") != readonly_control.memory_descriptor(memory).get(
        "sha256"
    ):
        raise R116FormalAuditError("checkpoint memory hash differs")
    return {
        "status": "passed",
        "benchmark": spec["benchmark"],
        "question_id": spec["question_id"],
        "artifact_id": spec["artifact_id"],
        "visible_tokens": generic_report["visible_tokens"],
        "source_resolution_tokens": generic_report["source_resolution_tokens"],
        "retrieval_model_calls": generic_report["retrieval_model_calls"],
        "answer_client_http_attempts": generic_report[
            "answer_client_http_attempts"
        ],
        "mapped_source_recall": recorded_recall["mapped_source_recall"],
        "source_recall_eligible": recorded_recall["source_recall_eligible"],
        "response_ids": response_ids,
        "proxy_events": events,
    }


def _audit_proxy_inventory(
    output_dir: Path,
    *,
    run_id: str,
    expected_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    proxy_root = output_dir / "proxy"
    if proxy_root.is_symlink() or not proxy_root.is_dir():
        raise R116FormalAuditError("formal proxy root is absent")
    live_events: list[dict[str, Any]] = []
    flex_reports: list[dict[str, Any]] = []
    invocations = 0
    for directory in sorted(proxy_root.glob("invocation-*")):
        if directory.is_symlink() or not directory.is_dir():
            raise R116FormalAuditError("proxy invocation directory is unsafe")
        manifest_path = directory / "manifest.json"
        manifest = _strict_json(manifest_path)
        if (
            not isinstance(manifest, dict)
            or manifest.get("run_id") != run_id
            or manifest.get("invocation_id") != directory.name
            or manifest.get("returncode") is None
        ):
            raise R116FormalAuditError("proxy invocation manifest differs")
        log = _inside_file(
            output_dir, manifest.get("log"), label="exclusive proxy log"
        )
        if (
            manifest.get("log_sha256") != answer_contract.sha256_file(log)
            or manifest.get("log_bytes") != log.stat().st_size
        ):
            raise R116FormalAuditError("exclusive proxy log binding differs")
        invocation_events = _read_jsonl(log)
        try:
            flex_reports.append(
                flex_evidence.audit_window(
                    manifest.get("provider_window"),
                    consumer_records=invocation_events,
                )
            )
        except flex_evidence.EvidenceError as exc:
            raise R116FormalAuditError(
                f"Flex provider window differs: {exc}"
            ) from exc
        live_events.extend(invocation_events)
        invocations += 1
    if invocations < 1:
        raise R116FormalAuditError("formal run has no proxy invocation")
    expected = list(expected_events)
    if Counter(str(row.get("event_id")) for row in live_events) != Counter(
        str(row.get("event_id")) for row in expected
    ):
        raise R116FormalAuditError("exclusive proxy contains orphan or missing events")
    by_id = {str(row.get("event_id")): row for row in expected}
    if any(by_id.get(str(row.get("event_id"))) != row for row in live_events):
        raise R116FormalAuditError("exclusive proxy event content differs")
    return {
        "invocations": invocations,
        "events": len(live_events),
        "upstream_http_attempts": sum(
            int(row.get("upstream_http_attempts", 0)) for row in live_events
        ),
        "flex_gateway_windows": flex_reports,
        "actual_models": sorted({row.get("actual_model") for row in live_events}),
    }


def _audit_aggregates(
    *,
    output_dir: Path,
    specs: Sequence[Mapping[str, Any]],
    complete: Mapping[str, Any],
) -> dict[str, Any]:
    expected_records: list[dict[str, Any]] = []
    expected_hypotheses: list[dict[str, str]] = []
    for spec in specs:
        question_root = output_dir / "questions" / str(spec["artifact_id"])
        result = _strict_json(question_root / "attempt-0001/result.json")
        recall = _strict_json(question_root / "source_recall.json")
        record = {
            "benchmark": spec["benchmark"],
            "question_id": spec["question_id"],
            "artifact_id": spec["artifact_id"],
            "dataset_index": spec["dataset_index"],
            "question": spec["question"],
            "answer": result["answer"]["text"],
            "answer_response_id": result["answer"]["response_id"],
            "answer_response_model": result["answer"]["response_model"],
            "answer_usage": result["answer"]["usage"],
            "retrieval_model_calls": result["retrieval"]["model_calls"],
            "visible_tokens": result["budget"]["visible_tokens"],
            "source_recall": recall,
        }
        for key in (
            "question_index",
            "category",
            "question_type",
            "abstention",
        ):
            if key in spec:
                record[key] = spec[key]
        expected_records.append(record)
        expected_hypotheses.append(
            {"question_id": spec["question_id"], "hypothesis": record["answer"]}
        )
    outputs = complete.get("outputs")
    if not isinstance(outputs, Mapping):
        raise R116FormalAuditError("aggregate output bindings are absent")
    results_path = _inside_file(
        output_dir, outputs.get("results", {}).get("path"), label="results"
    )
    hypotheses_path = _inside_file(
        output_dir,
        outputs.get("hypotheses", {}).get("path"),
        label="hypotheses",
    )
    if results_path != (output_dir / "results.json").resolve() or hypotheses_path != (
        output_dir / "hypotheses.jsonl"
    ).resolve():
        raise R116FormalAuditError("aggregate output paths differ")
    if (
        outputs["results"].get("sha256")
        != answer_contract.sha256_file(results_path)
        or outputs["hypotheses"].get("sha256")
        != answer_contract.sha256_file(hypotheses_path)
        or _strict_json(results_path)
        != {"schema_version": RUN_SCHEMA, "records": expected_records}
    ):
        raise R116FormalAuditError("aggregate results differ")
    hypotheses = _read_jsonl(hypotheses_path)
    if hypotheses != expected_hypotheses:
        raise R116FormalAuditError("aggregate hypotheses differ")
    return {
        "results_sha256": answer_contract.sha256_file(results_path),
        "hypotheses_sha256": answer_contract.sha256_file(hypotheses_path),
    }


def audit_run(output_dir: Path, *, require_complete: bool = True) -> dict[str, Any]:
    output_dir = output_dir.expanduser().absolute()
    answer_contract.reject_symlink_components(output_dir)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise R116FormalAuditError("R116 output root is invalid")
    manifest = _strict_json(output_dir / "run_manifest.json")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != RUN_SCHEMA
        or manifest.get("method") != contract.METHOD
        or manifest.get("condition") != contract.CONDITION
        or manifest.get("config", {}).get("model") != contract.MODEL
        or manifest.get("config", {}).get("budget_tokens")
        != contract.BUDGET_TOKENS
        or manifest.get("config", {}).get("max_rounds") != contract.MAX_ROUNDS
        or manifest.get("config", {}).get("model_context_limit_tokens")
        != contract.MODEL_CONTEXT_LIMIT_TOKENS
        or manifest.get("config", {}).get("answer_completion_reservation_tokens")
        != contract.ANSWER_COMPLETION_RESERVATION_TOKENS
        or manifest.get("config", {}).get("tokenizer")
        != answer_contract.formal_token_counter().identity
    ):
        raise R116FormalAuditError("R116 run manifest differs")
    prereg = contract.validate_preregistration(contract.DEFAULT_PREREGISTRATION)
    if manifest.get("preregistration") != contract.preregistration_binding(
        contract.DEFAULT_PREREGISTRATION
    ):
        raise R116FormalAuditError("run preregistration binding differs")
    mode = manifest.get("mode")
    formal = mode == "formal"
    if mode not in {"formal", "synthetic_no_network"}:
        raise R116FormalAuditError("R116 run mode differs")

    if formal:
        benchmark = str(manifest.get("benchmark"))
        if benchmark not in contract.BENCHMARKS:
            raise R116FormalAuditError("formal benchmark differs")
        if manifest.get("scope") != contract.EXPECTED_SCOPE[benchmark]:
            raise R116FormalAuditError("formal scope differs")
        source_binding = _strict_json(output_dir / "source_binding.json")
        if not isinstance(source_binding, dict):
            raise R116FormalAuditError("formal source binding is absent")
        _source_binding_hash(source_binding)
        if manifest.get("source_binding_sha256") != source_binding.get(
            "binding_sha256"
        ):
            raise R116FormalAuditError("manifest source binding differs")
        specs = contract.question_specs(benchmark)
        expected_inventory_hash = answer_contract.canonical_hash(
            [
                {
                    "question_id": spec["question_id"],
                    "artifact_id": spec["artifact_id"],
                    "question_sha256": answer_contract.sha256_bytes(
                        spec["question"].encode("utf-8")
                    ),
                    "evidence_record_sha256": spec["evidence_record_sha256"],
                }
                for spec in specs
            ]
        )
        config = manifest.get("config", {})
        python_path = Path(str(config.get("python", "")))
        try:
            provider_contract = flex_evidence.validate_recorded_contract(
                config.get("gateway_contract")
            )
        except flex_evidence.EvidenceError as exc:
            raise R116FormalAuditError(
                f"formal Flex gateway contract differs: {exc}"
            ) from exc
        if (
            manifest.get("question_inventory_sha256") != expected_inventory_hash
            or config.get("budget_policy") != "hard_cap"
            or config.get("answer_max_tokens") != contract.ANSWER_MAX_TOKENS
            or config.get("answer_retries") != contract.ANSWER_RETRIES
            or config.get("upstream") != provider_contract["origin"]
            or not python_path.is_file()
            or python_path.is_symlink()
            or config.get("python_sha256")
            != answer_contract.sha256_file(python_path)
        ):
            raise R116FormalAuditError("formal question inventory or config differs")
        expected_run_id = (
            f"r116-{benchmark}-"
            f"{str(source_binding['binding_sha256'])[:16]}-"
            f"{answer_contract.sha256_file(contract.DEFAULT_PREREGISTRATION)[:16]}"
        )
        if manifest.get("run_id") != expected_run_id:
            raise R116FormalAuditError("formal run ID differs")
        live_source_binding = contract.source_preflight(benchmark, prereg)
        if live_source_binding != source_binding:
            raise R116FormalAuditError("formal source binding differs from live audit")
        source_bindings = {benchmark: source_binding}
    else:
        benchmark = "both"
        specs = [
            contract.question_specs(contract.LOCOMO)[0],
            contract.question_specs(contract.LONGMEMEVAL)[0],
        ]
        source_bindings = {}
        for name in contract.BENCHMARKS:
            binding = _strict_json(output_dir / f"source_binding.{name}.json")
            if not isinstance(binding, dict):
                raise R116FormalAuditError("synthetic source binding is absent")
            _source_binding_hash(binding)
            source_bindings[name] = binding
        if (
            manifest.get("benchmark") != "both"
            or manifest.get("scope") != {"locomo": 1, "longmemeval-s": 1}
            or manifest.get("config", {}).get("model_requests") != 0
            or manifest.get("config", {}).get("network_requests") != 0
            or manifest.get("source_bindings")
            != {key: value["binding_sha256"] for key, value in source_bindings.items()}
        ):
            raise R116FormalAuditError("synthetic run identity differs")
    del prereg
    expected_artifacts = {str(spec["artifact_id"]) for spec in specs}
    question_root = output_dir / "questions"
    observed_artifacts = (
        {path.name for path in question_root.iterdir() if path.is_dir()}
        if question_root.is_dir()
        else set()
    )
    if observed_artifacts != expected_artifacts:
        raise R116FormalAuditError("question artifact inventory differs")
    reports = []
    all_response_ids: list[str] = []
    all_proxy_events: list[dict[str, Any]] = []
    for spec in specs:
        report = audit_question_checkpoint(
            output_dir=output_dir,
            spec=spec,
            source_binding=source_bindings[str(spec["benchmark"])],
            formal=formal,
        )
        reports.append(report)
        all_response_ids.extend(report["response_ids"])
        all_proxy_events.extend(report["proxy_events"])
    if len(set(all_response_ids)) != len(all_response_ids):
        raise R116FormalAuditError("run response IDs are duplicated")
    eligible = [report for report in reports if report["source_recall_eligible"]]
    expected_eligible = (
        contract.EXPECTED_SCOPE[benchmark]["source_recall_questions"]
        if formal
        else 2
    )
    if len(eligible) != expected_eligible:
        raise R116FormalAuditError("source-recall denominator differs")
    proxy = (
        _audit_proxy_inventory(
            output_dir,
            run_id=str(manifest["run_id"]),
            expected_events=all_proxy_events,
        )
        if formal
        else {
            "invocations": 0,
            "events": len(all_proxy_events),
            "upstream_http_attempts": 0,
            "actual_models": [contract.MODEL],
        }
    )
    aggregate_report = None
    if require_complete:
        complete = _strict_json(output_dir / "complete.json")
        if (
            not isinstance(complete, dict)
            or complete.get("schema_version") != COMPLETE_SCHEMA
            or complete.get("status") != "complete"
            or complete.get("run_id") != manifest.get("run_id")
            or complete.get("benchmark") != benchmark
            or complete.get("questions") != len(specs)
        ):
            raise R116FormalAuditError("complete manifest differs")
        if formal:
            progress_path = output_dir / "progress.json"
            progress = _strict_json(progress_path)
            if (
                progress.get("status") != "complete"
                or progress.get("completed") != len(specs)
                or progress.get("total") != len(specs)
                or complete.get("progress_sha256")
                != answer_contract.sha256_file(progress_path)
                or complete.get("source_binding_sha256")
                != source_bindings[benchmark]["binding_sha256"]
                or complete.get("preregistration_sha256")
                != answer_contract.sha256_file(contract.DEFAULT_PREREGISTRATION)
            ):
                raise R116FormalAuditError("formal completion linkage differs")
        elif (
            complete.get("model_requests") != 0
            or complete.get("network_requests") != 0
        ):
            raise R116FormalAuditError("synthetic completion accounting differs")
        aggregate_report = _audit_aggregates(
            output_dir=output_dir, specs=specs, complete=complete
        )
    recalls = [
        report["mapped_source_recall"]
        for report in eligible
        if report["mapped_source_recall"] is not None
    ]
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed",
        "mode": mode,
        "run_id": manifest["run_id"],
        "benchmark": benchmark,
        "questions": len(reports),
        "source_recall_questions": len(eligible),
        "mean_mapped_source_recall": sum(recalls) / len(recalls),
        "visible_tokens": sum(report["visible_tokens"] for report in reports),
        "source_resolution_tokens": sum(
            report["source_resolution_tokens"] for report in reports
        ),
        "retrieval_model_calls": sum(
            report["retrieval_model_calls"] for report in reports
        ),
        "answer_client_http_attempts": sum(
            report["answer_client_http_attempts"] for report in reports
        ),
        "response_ids": len(all_response_ids),
        "response_ids_unique": True,
        "model": contract.MODEL,
        "budget_tokens_per_question": contract.BUDGET_TOKENS,
        "memory_unchanged": True,
        "actual_model_evidence": proxy,
        "aggregate_outputs": aggregate_report,
        "network_requests": None if formal else 0,
        "claim_status": "formal_control" if formal else "synthetic_validation_only",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = audit_run(
        args.artifact_dir, require_complete=not args.allow_incomplete
    )
    if args.report is not None:
        answer_contract.atomic_json_replace(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        R116FormalAuditError,
        contract.R116FormalError,
        generic.ReadOnlyAuditError,
        readonly_control.ReadOnlyControlError,
        answer_contract.ControlledAnswerError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
