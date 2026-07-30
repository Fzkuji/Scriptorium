#!/usr/bin/env python3
"""Independent auditor for R207 task-evaluation artifacts.

This module intentionally does not import ``run_r207_task_eval``.  It rebuilds
question bindings, R004/M4 evidence, primary task scores, aggregate outputs,
and exclusive-proxy inventories from the frozen contract and durable files.
"""

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

import audit_readonly_nativemem_control as generic_auditor  # noqa: E402
import controlled_locomo_answer_contract as answer_contract  # noqa: E402
import r207_task_eval_contract as contract  # noqa: E402
import readonly_nativemem_control as readonly_control  # noqa: E402
from src.evaluation.durable_model_ledger import read_ledger  # noqa: E402
from src import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402


RUN_SCHEMA = "nativemem.r207-task-eval-run.v1"
CHECKPOINT_SCHEMA = "nativemem.r207-task-eval-question-checkpoint.v1"
COMPLETE_SCHEMA = "nativemem.r207-task-eval-complete.v1"
AUDIT_SCHEMA = "nativemem.r207-task-eval-audit.v1"


class R207TaskEvalAuditError(RuntimeError):
    pass


def _strict_json(path: Path) -> Any:
    return answer_contract.read_json(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise R207TaskEvalAuditError(f"JSONL has an incomplete final line: {path}")
    output: list[dict[str, Any]] = []
    for number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise R207TaskEvalAuditError(
                f"invalid JSONL line {number}: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise R207TaskEvalAuditError("JSONL record is not an object")
        output.append(value)
    return output


def _inside_file(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise R207TaskEvalAuditError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise R207TaskEvalAuditError(f"{label} path escapes output")
    path = root / relative
    answer_contract.reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise R207TaskEvalAuditError(f"{label} is not a regular file")
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise R207TaskEvalAuditError(f"{label} path escapes output") from exc
    return resolved


def _inside_directory(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise R207TaskEvalAuditError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise R207TaskEvalAuditError(f"{label} path escapes output")
    path = root / relative
    answer_contract.reject_symlink_components(path)
    if path.is_symlink() or not path.is_dir():
        raise R207TaskEvalAuditError(f"{label} is not a regular directory")
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise R207TaskEvalAuditError(f"{label} path escapes output") from exc
    return resolved


def _validate_content_hash(payload: Mapping[str, Any], field: str) -> None:
    content = dict(payload)
    expected = content.pop(field, None)
    if expected != answer_contract.canonical_hash(content):
        raise R207TaskEvalAuditError(f"{field} differs")


def _question_root(output_dir: Path, condition: str, spec: Mapping[str, Any]) -> Path:
    return (
        output_dir
        / "conditions"
        / condition
        / "questions"
        / str(spec["artifact_id"])
    )


def _expected_paths(
    output_dir: Path, condition: str, spec: Mapping[str, Any]
) -> dict[str, Path]:
    root = _question_root(output_dir, condition, spec)
    return {
        "input_binding": root / "input_binding.json",
        "stage_provenance": root / "stage_provenance.json",
        "upstream_r207_trace": root / "upstream_r207_trace.json",
        "attempt_result": root / "attempt-0001/result.json",
        "m4_stage_trace": root / "m4_stage_trace.json",
        "metrics": root / "metrics.json",
        "question_audit": root / "question_audit.json",
    }


def _audit_artifact_bindings(
    *,
    output_dir: Path,
    checkpoint: Mapping[str, Any],
    expected: Mapping[str, Path],
) -> dict[str, Path]:
    artifacts = checkpoint.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected):
        raise R207TaskEvalAuditError("checkpoint artifact inventory differs")
    paths: dict[str, Path] = {}
    for key, expected_path in expected.items():
        record = artifacts.get(key)
        if not isinstance(record, Mapping):
            raise R207TaskEvalAuditError(f"checkpoint {key} binding is invalid")
        path = _inside_file(output_dir, record.get("path"), label=key)
        if (
            path != expected_path.resolve()
            or record.get("sha256") != answer_contract.sha256_file(path)
            or record.get("bytes") != path.stat().st_size
        ):
            raise R207TaskEvalAuditError(f"checkpoint {key} binding differs")
        paths[key] = path
    return paths


def _proxy_events_from_result(
    *, attempt: Path, result: Mapping[str, Any], formal: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    response_ids: list[str] = []
    records = read_ledger(attempt / "retrieval_model_ledger.jsonl")
    starts = {
        str(row["logical_call_id"]): row
        for row in records
        if row.get("event") == "model_call_started"
    }
    finishes = {
        str(row["logical_call_id"]): row
        for row in records
        if row.get("event") == "model_call_finished"
    }
    if set(starts) != set(finishes):
        raise R207TaskEvalAuditError("retrieval model lifecycle differs")
    for logical_id, finish in finishes.items():
        response_id = str(finish.get("response_id", ""))
        response_ids.append(response_id)
        evidence = finish.get("proxy_evidence")
        if not isinstance(evidence, Mapping):
            raise R207TaskEvalAuditError("retrieval proxy evidence is absent")
        if formal and evidence.get("mode") != "exclusive_proxy":
            raise R207TaskEvalAuditError("formal retrieval proxy mode differs")
        recorded = evidence.get("events")
        if not isinstance(recorded, list) or not recorded:
            raise R207TaskEvalAuditError("retrieval proxy events are invalid")
        for event in recorded:
            if not isinstance(event, dict):
                raise R207TaskEvalAuditError("retrieval proxy event is invalid")
            expected = {
                "logical_call_id": logical_id,
                "question_id": result["question_id"],
                "status": "success",
                "requested_model": contract.MODEL,
                "actual_model": contract.MODEL,
                "response_id": response_id,
            }
            if (
                any(event.get(key) != value for key, value in expected.items())
                or event.get("usage") != finish.get("usage")
            ):
                raise R207TaskEvalAuditError("retrieval proxy linkage differs")
            events.append(event)
        if formal:
            prefix = evidence.get("log_prefix")
            if not isinstance(prefix, Mapping):
                raise R207TaskEvalAuditError("retrieval proxy prefix is absent")
            log = Path(str(prefix.get("path", "")))
            live = [
                row
                for row in _read_jsonl(log)
                if row.get("logical_call_id") == logical_id
            ]
            if live != recorded:
                raise R207TaskEvalAuditError("retrieval proxy log differs")

    answer = result.get("answer")
    if not isinstance(answer, Mapping):
        raise R207TaskEvalAuditError("answer result is absent")
    response_ids.append(str(answer.get("response_id", "")))
    evidence = answer.get("proxy_evidence")
    if not isinstance(evidence, Mapping):
        raise R207TaskEvalAuditError("answer proxy evidence is absent")
    if formal and evidence.get("mode") != "exclusive_proxy":
        raise R207TaskEvalAuditError("formal answer proxy mode differs")
    answer_events = evidence.get("events")
    if not isinstance(answer_events, list) or not answer_events:
        raise R207TaskEvalAuditError("answer proxy events are invalid")
    logical_id = (
        f"{result['run_id']}:{result['question_id']}:"
        f"{result['condition']}:answer"
    )
    request_hashes = answer.get("request_sha256s")
    if not isinstance(request_hashes, list):
        raise R207TaskEvalAuditError("answer request hashes are invalid")
    for event in answer_events:
        if not isinstance(event, dict):
            raise R207TaskEvalAuditError("answer proxy event is invalid")
        if (
            event.get("logical_call_id") != logical_id
            or event.get("question_id") != result["question_id"]
            or event.get("requested_model") != contract.MODEL
            or event.get("request_sha256") not in request_hashes
        ):
            raise R207TaskEvalAuditError("answer proxy linkage differs")
        if event.get("status") == "success":
            if (
                event.get("actual_model") != contract.MODEL
                or event.get("response_id") != answer["response_id"]
                or event.get("usage") != answer.get("usage")
            ):
                raise R207TaskEvalAuditError("answer proxy identity differs")
        elif event.get("status") != "error":
            raise R207TaskEvalAuditError("answer proxy status differs")
        events.append(event)
    successes = [row for row in answer_events if row.get("status") == "success"]
    if len(successes) != 1:
        raise R207TaskEvalAuditError("answer must have one successful proxy event")
    if formal:
        prefix = evidence.get("log_prefix")
        if not isinstance(prefix, Mapping):
            raise R207TaskEvalAuditError("answer proxy prefix is absent")
        log = Path(str(prefix.get("path", "")))
        live = [
            row
            for row in _read_jsonl(log)
            if row.get("logical_call_id") == logical_id
        ]
        if live != answer_events:
            raise R207TaskEvalAuditError("answer proxy log differs")
    if any(not value for value in response_ids):
        raise R207TaskEvalAuditError("model response ID is empty")
    return events, response_ids


def audit_question_checkpoint(
    *,
    output_dir: Path,
    source_root: Path,
    source_binding: Mapping[str, Any],
    source_context: Mapping[str, Any],
    spec: Mapping[str, Any],
    condition: str,
    formal: bool,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().absolute()
    root = _question_root(output_dir, condition, spec)
    checkpoint = _strict_json(root / "checkpoint.json")
    if not isinstance(checkpoint, dict):
        raise R207TaskEvalAuditError("question checkpoint is not an object")
    _validate_content_hash(checkpoint, "checkpoint_content_sha256")
    identity = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "complete",
        "question_id": spec["question_id"],
        "artifact_id": spec["artifact_id"],
        "dataset_index": spec["dataset_index"],
        "question_index": spec["question_index"],
        "category": spec["category"],
        "path_condition": condition,
        "generic_readonly_condition": contract.GENERIC_CONDITION,
        "actual_model": contract.MODEL,
    }
    if any(checkpoint.get(key) != value for key, value in identity.items()):
        raise R207TaskEvalAuditError("question checkpoint identity differs")
    expected_paths = _expected_paths(output_dir, condition, spec)
    paths = _audit_artifact_bindings(
        output_dir=output_dir,
        checkpoint=checkpoint,
        expected=expected_paths,
    )
    attempt = _inside_directory(
        output_dir,
        (root / "attempt-0001").relative_to(output_dir).as_posix(),
        label="question attempt",
    )
    memory = source_context["memories"][condition]
    upstream_expected = contract.trace_for_question(
        context=source_context,
        spec=spec,
        condition=condition,
        formal=formal,
    )
    upstream = _strict_json(paths["upstream_r207_trace"])
    if upstream != upstream_expected:
        raise R207TaskEvalAuditError("upstream R207 trace differs")
    provenance_expected = contract.build_stage_provenance(
        spec=spec,
        bank=source_context["bank"],
        memory=memory,
        upstream_trace=upstream,
    )
    provenance = _strict_json(paths["stage_provenance"])
    if provenance != provenance_expected:
        raise R207TaskEvalAuditError("stage provenance differs")
    input_expected = contract.input_binding(
        preregistration=contract.DEFAULT_PREREGISTRATION,
        source_binding=source_binding,
        spec=spec,
        condition=condition,
        memory=memory,
        upstream_trace=upstream,
        stage_provenance=provenance,
    )
    if _strict_json(paths["input_binding"]) != input_expected:
        raise R207TaskEvalAuditError("question input binding differs")
    turn_index = readonly_control.build_turn_index(
        contract.conversation_for_spec(spec)
    )
    generic_report = generic_auditor.audit_question(
        artifact_dir=attempt,
        memory_root=memory,
        method=contract.METHOD,
        condition=contract.GENERIC_CONDITION,
        question_id=str(spec["question_id"]),
        question=str(spec["question"]),
        gold_source_ids=list(spec["gold_source_ids"]),
        source_recall_eligible=bool(spec["source_recall_eligible"]),
        formal=formal,
        stage_provenance=provenance,
        turn_index=turn_index,
    )
    if _strict_json(paths["question_audit"]) != generic_report:
        raise R207TaskEvalAuditError("persisted generic question audit differs")
    result = _strict_json(paths["attempt_result"])
    manifest = _strict_json(output_dir / "run_manifest.json")
    if result.get("run_id") != f"{manifest['run_id']}:{condition}":
        raise R207TaskEvalAuditError("condition-specific run ID differs")
    m4_expected = contract.build_m4_trace(
        condition=condition,
        upstream_trace=upstream,
        observed_stage=result["stage_evidence"],
    )
    if _strict_json(paths["m4_stage_trace"]) != m4_expected:
        raise R207TaskEvalAuditError("M4 stage trace differs")
    metrics_expected = contract.build_question_metrics(
        attempt=attempt,
        spec=spec,
        condition=condition,
        result=result,
        m4_trace=m4_expected,
    )
    if _strict_json(paths["metrics"]) != metrics_expected:
        raise R207TaskEvalAuditError("question metrics differ")
    events, response_ids = _proxy_events_from_result(
        attempt=attempt, result=result, formal=formal
    )
    if (
        checkpoint.get("response_ids") != response_ids
        or checkpoint.get("memory_sha256")
        != metrics_expected["memory_tree"]["after_sha256"]
        or checkpoint.get("official_locomo_f1")
        != metrics_expected["official_locomo_f1"]
    ):
        raise R207TaskEvalAuditError("question checkpoint summary differs")
    return {
        "status": "passed",
        "question_id": spec["question_id"],
        "path_condition": condition,
        "source_recall_eligible": spec["source_recall_eligible"],
        "mapped_source_recall": metrics_expected["mapped_source_recall"],
        "first_relevant_file": metrics_expected["first_relevant_file"],
        "visible_tokens": metrics_expected["navigation_tokens"]["visible_tokens"],
        "source_resolution_tokens": metrics_expected["navigation_tokens"][
            "source_resolution_tokens"
        ],
        "retrieval_model_calls": metrics_expected["navigation_calls"][
            "retrieval_model_calls"
        ],
        "official_locomo_f1": metrics_expected["official_locomo_f1"],
        "memory_sha256": metrics_expected["memory_tree"]["after_sha256"],
        "m4_fields": metrics_expected["m4"]["fields"],
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
        raise R207TaskEvalAuditError("formal proxy root is absent")
    live_events: list[dict[str, Any]] = []
    flex_reports: list[dict[str, Any]] = []
    invocations = 0
    for directory in sorted(proxy_root.glob("invocation-*")):
        if directory.is_symlink() or not directory.is_dir():
            raise R207TaskEvalAuditError("proxy invocation directory is unsafe")
        manifest = _strict_json(directory / "manifest.json")
        if (
            not isinstance(manifest, dict)
            or manifest.get("run_id") != run_id
            or manifest.get("invocation_id") != directory.name
            or manifest.get("returncode") is None
        ):
            raise R207TaskEvalAuditError("proxy invocation manifest differs")
        log = _inside_file(
            output_dir, manifest.get("log"), label="exclusive proxy log"
        )
        if (
            manifest.get("log_sha256") != answer_contract.sha256_file(log)
            or manifest.get("log_bytes") != log.stat().st_size
        ):
            raise R207TaskEvalAuditError("exclusive proxy log binding differs")
        invocation_events = _read_jsonl(log)
        try:
            flex_reports.append(
                flex_evidence.audit_window(
                    manifest.get("provider_window"),
                    consumer_records=invocation_events,
                )
            )
        except flex_evidence.EvidenceError as exc:
            raise R207TaskEvalAuditError(
                f"Flex provider window differs: {exc}"
            ) from exc
        live_events.extend(invocation_events)
        invocations += 1
    if invocations < 1:
        raise R207TaskEvalAuditError("formal run has no proxy invocation")
    expected = list(expected_events)
    if Counter(str(row.get("event_id")) for row in live_events) != Counter(
        str(row.get("event_id")) for row in expected
    ):
        raise R207TaskEvalAuditError("exclusive proxy has orphan/missing events")
    by_id = {str(row.get("event_id")): row for row in expected}
    if any(by_id.get(str(row.get("event_id"))) != row for row in live_events):
        raise R207TaskEvalAuditError("exclusive proxy event content differs")
    return {
        "invocations": invocations,
        "events": len(live_events),
        "upstream_http_attempts": sum(
            int(row.get("upstream_http_attempts", 0)) for row in live_events
        ),
        "actual_models": sorted({row.get("actual_model") for row in live_events}),
        "flex_gateway_windows": flex_reports,
    }


def _expected_records(
    output_dir: Path, specs: Sequence[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    output = {condition: [] for condition in contract.PATH_CONDITIONS}
    for condition in contract.PATH_CONDITIONS:
        for spec in specs:
            root = _question_root(output_dir, condition, spec)
            result = _strict_json(root / "attempt-0001/result.json")
            metrics = _strict_json(root / "metrics.json")
            output[condition].append(
                {
                    "question_id": spec["question_id"],
                    "artifact_id": spec["artifact_id"],
                    "dataset_index": spec["dataset_index"],
                    "question_index": spec["question_index"],
                    "category": spec["category"],
                    "question": spec["question"],
                    "reference": spec["gold_answer"],
                    "answer": result["answer"]["text"],
                    "answer_response_id": result["answer"]["response_id"],
                    "official_locomo_f1": metrics["official_locomo_f1"],
                    "first_relevant_file": metrics["first_relevant_file"],
                    "mapped_source_recall": metrics["mapped_source_recall"],
                    "source_recall_eligible": metrics[
                        "source_recall_eligible"
                    ],
                    "navigation_calls": metrics["navigation_calls"],
                    "navigation_tokens": metrics["navigation_tokens"],
                    "memory_tree": metrics["memory_tree"],
                    "m4_fields": metrics["m4"]["fields"],
                }
            )
    return output


def _expected_summary(
    records: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    indexed: dict[str, dict[str, Mapping[str, Any]]] = {}
    for condition in contract.PATH_CONDITIONS:
        rows = list(records[condition])
        eligible = [row for row in rows if row["source_recall_eligible"]]
        recalls = [
            float(row["mapped_source_recall"])
            for row in eligible
            if row["mapped_source_recall"] is not None
        ]
        conditions[condition] = {
            "primary_questions": len(rows),
            "category_5_questions": 0,
            "mean_official_locomo_f1": sum(
                float(row["official_locomo_f1"]) for row in rows
            )
            / len(rows),
            "source_recall_questions": len(eligible),
            "mean_mapped_source_recall": sum(recalls) / len(recalls),
            "navigation_tool_calls": sum(
                int(row["navigation_calls"]["tool_calls"]) for row in rows
            ),
            "visible_tokens": sum(
                int(row["navigation_tokens"]["visible_tokens"]) for row in rows
            ),
        }
        indexed[condition] = {str(row["question_id"]): row for row in rows}
    left = indexed["model_directed"]
    right = indexed["deterministic_permutation"]
    if set(left) != set(right):
        raise R207TaskEvalAuditError("paired aggregate inventory differs")
    paired = [
        float(left[qid]["official_locomo_f1"])
        - float(right[qid]["official_locomo_f1"])
        for qid in sorted(left)
    ]
    return {
        "schema_version": RUN_SCHEMA,
        "primary_only": True,
        "category_5_mixed": False,
        "conditions": conditions,
        "paired_model_directed_minus_permutation_mean_f1": sum(paired)
        / len(paired),
    }


def _audit_aggregates(
    *,
    output_dir: Path,
    specs: Sequence[Mapping[str, Any]],
    complete: Mapping[str, Any],
) -> dict[str, Any]:
    outputs = complete.get("outputs")
    if not isinstance(outputs, Mapping):
        raise R207TaskEvalAuditError("aggregate output bindings are absent")
    expected_records = _expected_records(output_dir, specs)
    results_record = outputs.get("results")
    summary_record = outputs.get("summary")
    if not isinstance(results_record, Mapping) or not isinstance(
        summary_record, Mapping
    ):
        raise R207TaskEvalAuditError("aggregate file binding is invalid")
    results_path = _inside_file(
        output_dir, results_record.get("path"), label="results"
    )
    summary_path = _inside_file(
        output_dir, summary_record.get("path"), label="summary"
    )
    if (
        results_path != (output_dir / "results.json").resolve()
        or summary_path != (output_dir / "summary.json").resolve()
        or _strict_json(results_path)
        != {"schema_version": RUN_SCHEMA, "records": expected_records}
        or _strict_json(summary_path) != _expected_summary(expected_records)
    ):
        raise R207TaskEvalAuditError("aggregate result content differs")
    for record, path in (
        (results_record, results_path),
        (summary_record, summary_path),
    ):
        if (
            record.get("sha256") != answer_contract.sha256_file(path)
            or record.get("bytes") != path.stat().st_size
        ):
            raise R207TaskEvalAuditError("aggregate hash binding differs")
    hypotheses = outputs.get("hypotheses")
    if not isinstance(hypotheses, Mapping) or set(hypotheses) != set(
        contract.PATH_CONDITIONS
    ):
        raise R207TaskEvalAuditError("hypothesis inventory differs")
    for condition in contract.PATH_CONDITIONS:
        record = hypotheses[condition]
        if not isinstance(record, Mapping):
            raise R207TaskEvalAuditError("hypothesis binding is invalid")
        path = _inside_file(
            output_dir, record.get("path"), label=f"{condition} hypotheses"
        )
        expected = [
            {"question_id": row["question_id"], "hypothesis": row["answer"]}
            for row in expected_records[condition]
        ]
        if (
            path != (output_dir / "hypotheses" / f"{condition}.jsonl").resolve()
            or _read_jsonl(path) != expected
            or record.get("sha256") != answer_contract.sha256_file(path)
            or record.get("bytes") != path.stat().st_size
        ):
            raise R207TaskEvalAuditError("hypothesis artifact differs")
    return {
        "results_sha256": answer_contract.sha256_file(results_path),
        "summary_sha256": answer_contract.sha256_file(summary_path),
        "mean_f1": {
            condition: _strict_json(summary_path)["conditions"][condition][
                "mean_official_locomo_f1"
            ]
            for condition in contract.PATH_CONDITIONS
        },
    }


def audit_run(output_dir: Path, *, require_complete: bool = True) -> dict[str, Any]:
    output_dir = output_dir.expanduser().absolute()
    answer_contract.reject_symlink_components(output_dir)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise R207TaskEvalAuditError("R207 task-eval output root is invalid")
    manifest = _strict_json(output_dir / "run_manifest.json")
    if not isinstance(manifest, dict):
        raise R207TaskEvalAuditError("run manifest is invalid")
    mode = manifest.get("mode")
    formal = mode == "formal"
    if mode not in {"formal", "synthetic_no_network"}:
        raise R207TaskEvalAuditError("run mode differs")
    if (
        manifest.get("schema_version") != RUN_SCHEMA
        or manifest.get("method") != contract.METHOD
        or manifest.get("path_conditions") != list(contract.PATH_CONDITIONS)
        or manifest.get("generic_readonly_condition")
        != contract.GENERIC_CONDITION
    ):
        raise R207TaskEvalAuditError("run manifest identity differs")
    config = manifest.get("config")
    if (
        not isinstance(config, Mapping)
        or config.get("retrieval_model") != contract.MODEL
        or config.get("answer_model") != contract.MODEL
        or config.get("budget_policy") != "hard_cap"
        or config.get("budget_tokens") != contract.BUDGET_TOKENS
        or config.get("max_rounds") != contract.MAX_ROUNDS
        or config.get("model_context_limit_tokens")
        != contract.MODEL_CONTEXT_LIMIT_TOKENS
        or config.get("answer_completion_reservation_tokens")
        != contract.ANSWER_COMPLETION_RESERVATION_TOKENS
        or config.get("answer_max_tokens") != contract.ANSWER_MAX_TOKENS
        or config.get("answer_retries") != contract.ANSWER_RETRIES
        or config.get("tokenizer")
        != answer_contract.formal_token_counter().identity
    ):
        raise R207TaskEvalAuditError("run protocol differs")
    contract.validate_preregistration(contract.DEFAULT_PREREGISTRATION)
    if manifest.get("preregistration") != contract.preregistration_binding(
        contract.DEFAULT_PREREGISTRATION
    ):
        raise R207TaskEvalAuditError("run preregistration binding differs")
    source_binding = _strict_json(output_dir / "source_binding.json")
    if not isinstance(source_binding, dict):
        raise R207TaskEvalAuditError("source binding is absent")
    contract.validate_binding_hash(source_binding)
    source_root = ROOT / str(source_binding["source_root"])
    live_binding = contract.audit_r207_source(source_root, formal=formal)
    if live_binding != source_binding:
        raise R207TaskEvalAuditError("source binding differs from live audit")
    specs = contract.primary_question_specs() if formal else [contract.synthetic_spec()]
    expected_scope = {
        "samples": list(range(10)) if formal else [0],
        "questions_per_condition": len(specs),
        "question_condition_artifacts": len(specs)
        * len(contract.PATH_CONDITIONS),
        "primary_categories": [1, 2, 3, 4],
        "category_5_questions_mixed": 0,
    }
    expected_inventory = answer_contract.canonical_hash(
        [
            {
                "question_id": spec["question_id"],
                "artifact_id": spec["artifact_id"],
                "question_sha256": answer_contract.sha256_bytes(
                    str(spec["question"]).encode("utf-8")
                ),
                "gold_answer_sha256": spec["gold_answer_sha256"],
                "evidence_record_sha256": spec["evidence_record_sha256"],
            }
            for spec in specs
        ]
    )
    expected_run_id = (
        "r207-task-eval-"
        f"{str(source_binding['binding_sha256'])[:16]}-"
        f"{answer_contract.sha256_file(contract.DEFAULT_PREREGISTRATION)[:16]}"
        + ("" if formal else "-synthetic")
    )
    if (
        manifest.get("scope") != expected_scope
        or manifest.get("question_inventory_sha256") != expected_inventory
        or manifest.get("source_binding_sha256")
        != source_binding["binding_sha256"]
        or manifest.get("run_id") != expected_run_id
    ):
        raise R207TaskEvalAuditError("run scope or fingerprint differs")
    if formal:
        python = Path(str(config.get("python", "")))
        try:
            provider_contract = flex_evidence.validate_recorded_contract(
                config.get("gateway_contract")
            )
        except flex_evidence.EvidenceError as exc:
            raise R207TaskEvalAuditError(
                f"formal Flex gateway contract differs: {exc}"
            ) from exc
        if (
            config.get("upstream") != provider_contract["origin"]
            or config.get("model_requests_authorized") is not True
            or config.get("network_requests") is not None
            or python.is_symlink()
            or not python.is_file()
            or config.get("python_sha256") != answer_contract.sha256_file(python)
        ):
            raise R207TaskEvalAuditError("formal execution gate differs")
    elif (
        config.get("upstream") is not None
        or config.get("gateway_contract") is not None
        or config.get("python") is not None
        or config.get("python_sha256") is not None
        or config.get("model_requests_authorized") is not False
        or config.get("network_requests") != 0
    ):
        raise R207TaskEvalAuditError("synthetic no-network identity differs")
    contexts = {
        sample: contract.load_sample_context(
            source_root,
            source_binding,
            sample=sample,
            formal=formal,
        )
        for sample in sorted({int(spec["dataset_index"]) for spec in specs})
    }
    conditions_root = output_dir / "conditions"
    if (
        conditions_root.is_symlink()
        or not conditions_root.is_dir()
        or {path.name for path in conditions_root.iterdir()}
        != set(contract.PATH_CONDITIONS)
    ):
        raise R207TaskEvalAuditError("condition output inventory differs")
    expected_artifacts = {str(spec["artifact_id"]) for spec in specs}
    reports: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_response_ids: list[str] = []
    for condition in contract.PATH_CONDITIONS:
        question_root = conditions_root / condition / "questions"
        if (
            question_root.is_symlink()
            or not question_root.is_dir()
            or {path.name for path in question_root.iterdir()}
            != expected_artifacts
        ):
            raise R207TaskEvalAuditError("question artifact inventory differs")
        for spec in specs:
            report = audit_question_checkpoint(
                output_dir=output_dir,
                source_root=source_root,
                source_binding=source_binding,
                source_context=contexts[int(spec["dataset_index"])],
                spec=spec,
                condition=condition,
                formal=formal,
            )
            reports.append(report)
            all_events.extend(report["proxy_events"])
            all_response_ids.extend(report["response_ids"])
    if len(set(all_response_ids)) != len(all_response_ids):
        raise R207TaskEvalAuditError("response IDs are duplicated")
    expected_eligible = (
        contract.PRIMARY_SOURCE_RECALL * len(contract.PATH_CONDITIONS)
        if formal
        else 2
    )
    eligible = [report for report in reports if report["source_recall_eligible"]]
    if len(eligible) != expected_eligible:
        raise R207TaskEvalAuditError("source-recall denominator differs")
    if formal:
        proxy = _audit_proxy_inventory(
            output_dir,
            run_id=manifest["run_id"],
            expected_events=all_events,
        )
    else:
        proxy_log = output_dir / "synthetic_proxy.jsonl"
        live_events = _read_jsonl(proxy_log)
        if live_events != all_events:
            raise R207TaskEvalAuditError("synthetic proxy evidence inventory differs")
        proxy = {
            "invocations": 0,
            "events": len(live_events),
            "upstream_http_attempts": 0,
            "actual_models": [contract.MODEL],
            "synthetic_events_only": True,
        }
    aggregate = None
    if require_complete:
        complete = _strict_json(output_dir / "complete.json")
        if (
            not isinstance(complete, Mapping)
            or complete.get("schema_version") != COMPLETE_SCHEMA
            or complete.get("status") != "complete"
            or complete.get("run_id") != manifest["run_id"]
            or complete.get("questions_per_condition") != len(specs)
            or complete.get("question_condition_artifacts") != len(reports)
            or complete.get("category_5_questions_mixed") != 0
        ):
            raise R207TaskEvalAuditError("completion manifest differs")
        if formal:
            progress = _strict_json(output_dir / "progress.json")
            if (
                progress.get("status") != "complete"
                or progress.get("completed") != len(reports)
                or progress.get("total") != len(reports)
                or complete.get("source_binding_sha256")
                != source_binding["binding_sha256"]
                or complete.get("preregistration_sha256")
                != answer_contract.sha256_file(contract.DEFAULT_PREREGISTRATION)
                or complete.get("progress_sha256")
                != answer_contract.sha256_file(output_dir / "progress.json")
            ):
                raise R207TaskEvalAuditError("formal completion linkage differs")
        elif (
            complete.get("model_requests") != 0
            or complete.get("network_requests") != 0
            or complete.get("simulated_retrieval_calls") != 6
            or complete.get("simulated_answer_calls") != 2
        ):
            raise R207TaskEvalAuditError("synthetic request accounting differs")
        aggregate = _audit_aggregates(
            output_dir=output_dir, specs=specs, complete=complete
        )
    recalls = [
        float(report["mapped_source_recall"])
        for report in eligible
        if report["mapped_source_recall"] is not None
    ]
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed",
        "mode": mode,
        "run_id": manifest["run_id"],
        "path_conditions": list(contract.PATH_CONDITIONS),
        "questions_per_condition": len(specs),
        "question_condition_artifacts": len(reports),
        "primary_scores": len(reports),
        "category_5_scores_mixed": 0,
        "source_recall_questions": len(eligible),
        "mean_mapped_source_recall": sum(recalls) / len(recalls),
        "visible_tokens": sum(int(row["visible_tokens"]) for row in reports),
        "source_resolution_tokens": sum(
            int(row["source_resolution_tokens"]) for row in reports
        ),
        "retrieval_model_calls": sum(
            int(row["retrieval_model_calls"]) for row in reports
        ),
        "response_ids": len(all_response_ids),
        "response_ids_unique": True,
        "memory_unchanged": True,
        "model": contract.MODEL,
        "budget_tokens_per_question": contract.BUDGET_TOKENS,
        "actual_model_evidence": proxy,
        "aggregate_outputs": aggregate,
        "model_requests": None if formal else 0,
        "network_requests": None if formal else 0,
        "claim_status": (
            "formal_primary_task_evaluation"
            if formal
            else "synthetic_validation_only"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_run(args.artifact_dir, require_complete=True)
    except Exception as exc:  # noqa: BLE001
        report = {
            "schema_version": AUDIT_SCHEMA,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        if args.report is not None:
            answer_contract.atomic_json_replace(args.report, report)
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 1
    if args.report is not None:
        answer_contract.atomic_json_replace(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
