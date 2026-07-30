#!/usr/bin/env python3
"""Independently audit R403 checkpoint task-evaluation artifacts.

This auditor deliberately does not import the task runner.  It reconstructs
the source binding, checkpoint inventory, prefix turn index, token gate,
durable ledgers, proxy linkage, M4 stage evidence, metrics, aggregates, and
complete question inventory from immutable inputs.
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
import r403_task_eval_contract as contract  # noqa: E402
from src.evaluation.durable_model_ledger import read_ledger  # noqa: E402
from src import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402


RUN_SCHEMA = "nativemem.r403-task-eval-run.v1"
CHECKPOINT_SCHEMA = "nativemem.r403-task-eval-question-checkpoint.v1"
COMPLETE_SCHEMA = "nativemem.r403-task-eval-complete.v1"
AUDIT_SCHEMA = "nativemem.r403-task-eval-audit.v1"


class R403TaskEvalAuditError(RuntimeError):
    pass


def _strict_json(path: Path) -> Any:
    answer_contract.reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise R403TaskEvalAuditError(f"JSON artifact is not regular: {path}")
    return answer_contract.read_json(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    answer_contract.reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise R403TaskEvalAuditError(f"JSONL artifact is not regular: {path}")
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise R403TaskEvalAuditError(f"JSONL final line is incomplete: {path}")
    records = []
    for number, raw in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise R403TaskEvalAuditError(
                f"invalid JSONL line {number}: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise R403TaskEvalAuditError("JSONL record is not an object")
        records.append(value)
    return records


def _inside_file(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise R403TaskEvalAuditError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise R403TaskEvalAuditError(f"{label} path escapes output")
    candidate = root / relative
    answer_contract.reject_symlink_components(candidate)
    if candidate.is_symlink() or not candidate.is_file():
        raise R403TaskEvalAuditError(f"{label} is not a regular file")
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise R403TaskEvalAuditError(f"{label} path escapes output") from exc
    return candidate.resolve()


def _inside_directory(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise R403TaskEvalAuditError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise R403TaskEvalAuditError(f"{label} path escapes output")
    candidate = root / relative
    answer_contract.reject_symlink_components(candidate)
    if candidate.is_symlink() or not candidate.is_dir():
        raise R403TaskEvalAuditError(f"{label} is not a regular directory")
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise R403TaskEvalAuditError(f"{label} path escapes output") from exc
    return candidate.resolve()


def _validate_content_hash(payload: Mapping[str, Any], field: str) -> None:
    content = dict(payload)
    expected = content.pop(field, None)
    if expected != answer_contract.canonical_hash(content):
        raise R403TaskEvalAuditError(f"{field} differs")


def _artifact_path(
    output_dir: Path,
    checkpoint: Mapping[str, Any],
    key: str,
    expected: Path,
) -> Path:
    record = checkpoint.get("artifacts", {}).get(key)
    if not isinstance(record, Mapping):
        raise R403TaskEvalAuditError(f"checkpoint artifact is absent: {key}")
    path = _inside_file(output_dir, record.get("path"), label=key)
    if path != expected.resolve():
        raise R403TaskEvalAuditError(f"checkpoint artifact path differs: {key}")
    if (
        record.get("sha256") != answer_contract.sha256_file(path)
        or record.get("bytes") != path.stat().st_size
    ):
        raise R403TaskEvalAuditError(f"checkpoint artifact binding differs: {key}")
    return path


def _question_paths(output_dir: Path, spec: Mapping[str, Any]) -> dict[str, Path]:
    root = (
        output_dir
        / "checkpoints"
        / f"checkpoint-{int(spec['checkpoint']):03d}"
        / "questions"
        / str(spec["artifact_id"])
    )
    return {
        "root": root,
        "input": root / "input_binding.json",
        "upstream": root / "upstream_stage_evidence.json",
        "attempt": root / "attempt-0001",
        "m4": root / "m4_stage_trace.json",
        "metrics": root / "metrics.json",
        "question_audit": root / "question_audit.json",
        "checkpoint": root / "checkpoint.json",
    }


def _response_ids(attempt: Path, result: Mapping[str, Any]) -> list[str]:
    values = [
        str(row.get("response_id", ""))
        for row in read_ledger(attempt / "retrieval_model_ledger.jsonl")
        if row.get("event") == "model_call_finished"
    ]
    values.append(str(result.get("answer", {}).get("response_id", "")))
    if any(not value for value in values):
        raise R403TaskEvalAuditError("question response ID is empty")
    return values


def _question_proxy_events(
    attempt: Path, result: Mapping[str, Any], *, formal: bool
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    retrieval = read_ledger(attempt / "retrieval_model_ledger.jsonl")
    starts = {
        str(row.get("logical_call_id")): row
        for row in retrieval
        if row.get("event") == "model_call_started"
    }
    finishes = {
        str(row.get("logical_call_id")): row
        for row in retrieval
        if row.get("event") == "model_call_finished"
    }
    if not starts or set(starts) != set(finishes):
        raise R403TaskEvalAuditError("retrieval model lifecycle differs")
    for logical_id, finish in finishes.items():
        evidence = finish.get("proxy_evidence")
        if not isinstance(evidence, Mapping):
            raise R403TaskEvalAuditError("retrieval proxy evidence is absent")
        if formal and evidence.get("mode") != "exclusive_proxy":
            raise R403TaskEvalAuditError("formal retrieval proxy mode differs")
        rows = evidence.get("events")
        if not isinstance(rows, list) or not rows:
            raise R403TaskEvalAuditError("retrieval proxy event inventory differs")
        success = [row for row in rows if row.get("status") == "success"]
        if len(success) != 1:
            raise R403TaskEvalAuditError("retrieval call success count differs")
        if (
            success[0].get("logical_call_id") != logical_id
            or success[0].get("response_id") != finish.get("response_id")
            or success[0].get("actual_model") != contract.MODEL
            or success[0].get("usage") != finish.get("usage")
        ):
            raise R403TaskEvalAuditError("retrieval proxy linkage differs")
        events.extend(dict(row) for row in rows)
    answer = result.get("answer")
    if not isinstance(answer, Mapping):
        raise R403TaskEvalAuditError("answer evidence is absent")
    evidence = answer.get("proxy_evidence")
    if not isinstance(evidence, Mapping):
        raise R403TaskEvalAuditError("answer proxy evidence is absent")
    if formal and evidence.get("mode") != "exclusive_proxy":
        raise R403TaskEvalAuditError("formal answer proxy mode differs")
    rows = evidence.get("events")
    if not isinstance(rows, list) or not rows:
        raise R403TaskEvalAuditError("answer proxy event inventory differs")
    success = [row for row in rows if row.get("status") == "success"]
    if len(success) != 1:
        raise R403TaskEvalAuditError("answer proxy success count differs")
    logical_id = (
        f"{result['run_id']}:{result['question_id']}:"
        f"{contract.GENERIC_CONDITION}:answer"
    )
    if (
        success[0].get("logical_call_id") != logical_id
        or success[0].get("response_id") != answer.get("response_id")
        or success[0].get("actual_model") != contract.MODEL
        or success[0].get("usage") != answer.get("usage")
    ):
        raise R403TaskEvalAuditError("answer proxy linkage differs")
    events.extend(dict(row) for row in rows)
    event_ids = [str(row.get("event_id", "")) for row in events]
    if any(not value for value in event_ids) or len(event_ids) != len(set(event_ids)):
        raise R403TaskEvalAuditError("question proxy event IDs differ")
    return events


def audit_question_checkpoint(
    *,
    output_dir: Path,
    source_root: Path,
    source_binding: Mapping[str, Any],
    spec: Mapping[str, Any],
    formal: bool,
) -> dict[str, Any]:
    """Reconstruct and audit one complete task-evaluation question."""

    output_dir = output_dir.expanduser().absolute()
    paths = _question_paths(output_dir, spec)
    checkpoint = _strict_json(paths["checkpoint"])
    if not isinstance(checkpoint, dict):
        raise R403TaskEvalAuditError("question checkpoint is not an object")
    _validate_content_hash(checkpoint, "checkpoint_content_sha256")
    identity = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "complete",
        "question_id": spec["question_id"],
        "artifact_id": spec["artifact_id"],
        "sample": spec["sample"],
        "sample_id": spec["sample_id"],
        "checkpoint": spec["checkpoint"],
        "session_boundary": spec["session_boundary"],
        "question_index": spec["question_index"],
        "category": spec["category"],
        "cohorts": spec["cohorts"],
        "source_binding_sha256": source_binding["binding_sha256"],
        "actual_model": contract.MODEL,
    }
    if any(checkpoint.get(key) != value for key, value in identity.items()):
        raise R403TaskEvalAuditError("question checkpoint identity differs")
    input_path = _artifact_path(
        output_dir, checkpoint, "input_binding", paths["input"]
    )
    upstream_path = _artifact_path(
        output_dir,
        checkpoint,
        "upstream_stage_evidence",
        paths["upstream"],
    )
    result_path = _artifact_path(
        output_dir,
        checkpoint,
        "attempt_result",
        paths["attempt"] / "result.json",
    )
    m4_path = _artifact_path(
        output_dir, checkpoint, "m4_stage_trace", paths["m4"]
    )
    metrics_path = _artifact_path(
        output_dir, checkpoint, "metrics", paths["metrics"]
    )
    audit_path = _artifact_path(
        output_dir, checkpoint, "question_audit", paths["question_audit"]
    )
    attempt = paths["attempt"]
    answer_contract.reject_symlink_components(attempt)
    if attempt.is_symlink() or not attempt.is_dir():
        raise R403TaskEvalAuditError("question attempt is not a regular directory")

    context = contract.load_checkpoint_context(
        source_root,
        source_binding,
        sample=int(spec["sample"]),
        checkpoint=int(spec["checkpoint"]),
        formal=formal,
    )
    expected_input = contract.input_binding(
        preregistration=contract.DEFAULT_PREREGISTRATION,
        source_binding=source_binding,
        checkpoint_binding=context["binding"],
        spec=spec,
        memory=context["memory"],
        turn_index=context["turn_index"],
    )
    if _strict_json(input_path) != expected_input:
        raise R403TaskEvalAuditError("question input binding differs")
    if _strict_json(upstream_path) != spec["upstream_stage_evidence"]:
        raise R403TaskEvalAuditError("upstream stage evidence differs")

    generic_report = generic_auditor.audit_question(
        artifact_dir=attempt,
        memory_root=context["memory"],
        method=contract.METHOD,
        condition=contract.GENERIC_CONDITION,
        question_id=str(spec["question_id"]),
        question=str(spec["question"]),
        gold_source_ids=list(spec["gold_source_ids"]),
        source_recall_eligible=bool(spec["source_recall_eligible"]),
        formal=formal,
    )
    if _strict_json(audit_path) != generic_report:
        raise R403TaskEvalAuditError("persisted generic question audit differs")
    result = _strict_json(result_path)
    if (
        result.get("status") != "complete"
        or result.get("method") != contract.METHOD
        or result.get("condition") != contract.GENERIC_CONDITION
        or result.get("question_id") != spec["question_id"]
        or result.get("answer", {}).get("response_model") != contract.MODEL
        or result.get("budget", {}).get("configured_tokens")
        != contract.BUDGET_TOKENS
        or result.get("budget", {}).get("visible_tokens", contract.BUDGET_TOKENS + 1)
        > contract.BUDGET_TOKENS
        or result.get("memory", {}).get("unchanged") is not True
    ):
        raise R403TaskEvalAuditError("question result protocol differs")
    m4 = contract.build_m4_trace(
        attempt=attempt,
        spec=spec,
        turn_index=context["turn_index"],
    )
    if _strict_json(m4_path) != m4:
        raise R403TaskEvalAuditError("R403 M4 trace differs")
    metrics = contract.build_question_metrics(
        attempt=attempt,
        spec=spec,
        result=result,
        m4_trace=m4,
    )
    if _strict_json(metrics_path) != metrics:
        raise R403TaskEvalAuditError("R403 question metrics differ")
    response_ids = _response_ids(attempt, result)
    if (
        checkpoint.get("response_ids") != response_ids
        or checkpoint.get("memory_before_sha256")
        != result["memory"]["before"]["sha256"]
        or checkpoint.get("memory_after_sha256")
        != result["memory"]["after"]["sha256"]
        or checkpoint.get("memory_unchanged") is not True
        or result["memory"]["before"] != context["binding"]["memory"][
            "readonly_descriptor"
        ]
        or result["memory"]["after"] != result["memory"]["before"]
    ):
        raise R403TaskEvalAuditError("question memory or response binding differs")
    proxy_events = _question_proxy_events(attempt, result, formal=formal)
    expected_root_names = {
        "input_binding.json",
        "upstream_stage_evidence.json",
        "attempt-0001",
        "m4_stage_trace.json",
        "metrics.json",
        "question_audit.json",
        "checkpoint.json",
    }
    if {path.name for path in paths["root"].iterdir()} != expected_root_names:
        raise R403TaskEvalAuditError("question artifact inventory differs")
    return {
        "question_id": spec["question_id"],
        "artifact_id": spec["artifact_id"],
        "sample": spec["sample"],
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": answer_contract.sha256_file(paths["checkpoint"]),
        "answer": result["answer"]["text"],
        "metrics": metrics,
        "proxy_events": proxy_events,
        "response_ids": response_ids,
    }


def _artifact_binding(output_dir: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": answer_contract.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _aggregate_records(
    specs: Sequence[Mapping[str, Any]], reports: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_artifact = {str(row["artifact_id"]): row for row in reports}
    records = []
    for spec in specs:
        report = by_artifact.get(str(spec["artifact_id"]))
        if report is None:
            raise R403TaskEvalAuditError("aggregate question report is absent")
        records.append(
            {
                "artifact_id": spec["artifact_id"],
                "question_id": spec["question_id"],
                "sample": spec["sample"],
                "checkpoint": spec["checkpoint"],
                "session_boundary": spec["session_boundary"],
                "question_index": spec["question_index"],
                "question": spec["question"],
                "answer": report["answer"],
                "gold_answer": spec["gold_answer"],
                "category": spec["category"],
                "cohorts": spec["cohorts"],
                "source_recall_eligible": spec["source_recall_eligible"],
                "source_exclusion_reasons": spec["source_exclusion_reasons"],
                "metrics": report["metrics"],
            }
        )
    return records


def _aggregate_usage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {
        "logical_calls": sum(
            int(row["metrics"]["calls_tokens_latency"]["logical_calls"]["total"])
            for row in records
        ),
        "visible_tokens": sum(
            int(row["metrics"]["navigation"]["visible_tokens"]) for row in records
        ),
        "provider_tokens": {
            key: sum(
                int(
                    row["metrics"]["calls_tokens_latency"]["provider_tokens"][
                        "total"
                    ][key]
                )
                for row in records
            )
            for key in keys
        },
        "retrieval_total_latency_s": round(
            sum(
                float(
                    row["metrics"]["calls_tokens_latency"]["latency_s"][
                        "retrieval_total"
                    ]
                )
                for row in records
            ),
            6,
        ),
        "answer_proxy_latency_s": round(
            sum(
                float(
                    row["metrics"]["calls_tokens_latency"]["latency_s"][
                        "answer_proxy"
                    ]
                )
                for row in records
            ),
            6,
        ),
    }


def _audit_aggregates(
    output_dir: Path,
    *,
    specs: Sequence[Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = _aggregate_records(specs, reports)
    results_path = output_dir / "results.json"
    if _strict_json(results_path) != {"schema_version": RUN_SCHEMA, "records": records}:
        raise R403TaskEvalAuditError("R403 aggregate results differ")
    scoring_path = output_dir / "scoring_inputs.jsonl"
    expected_scoring = [
        {
            "artifact_id": row["artifact_id"],
            "question_id": row["question_id"],
            "sample": row["sample"],
            "checkpoint": row["checkpoint"],
            "question": row["question"],
            "answer": row["answer"],
            "gold_answer": row["gold_answer"],
            "category": row["category"],
            "cohorts": row["cohorts"],
            "source_recall_eligible": row["source_recall_eligible"],
            "primary_judge_status": "not_attached",
            "sensitivity_judge_status": "not_attached",
        }
        for row in records
    ]
    if _read_jsonl(scoring_path) != expected_scoring:
        raise R403TaskEvalAuditError("R403 scoring input differs")
    summaries = {}
    for checkpoint in contract.CHECKPOINTS:
        subset = [row for row in records if row["checkpoint"] == checkpoint]
        summaries[str(checkpoint)] = {
            "task_denominator": len(subset),
            "old_fact_denominator": sum(row["cohorts"]["old_fact"] for row in subset),
            "update_denominator": sum(row["cohorts"]["update"] for row in subset),
            "source_reachability_denominator": sum(
                row["cohorts"]["source"] for row in subset
            ),
            "primary_judge_status": "not_attached",
            "sensitivity_judge_status": "not_attached",
            "usage": _aggregate_usage(subset),
        }
    summary_path = output_dir / "checkpoint_summary_inputs.json"
    expected_summary = {
        "schema_version": RUN_SCHEMA,
        "denominator_policy": "R403_TASK_EVAL_PROTOCOL_FREEZE.json",
        "checkpoints": summaries,
    }
    if _strict_json(summary_path) != expected_summary:
        raise R403TaskEvalAuditError("R403 checkpoint summaries differ")
    return {
        "records": records,
        "outputs": {
            "results": _artifact_binding(output_dir, results_path),
            "scoring_inputs": _artifact_binding(output_dir, scoring_path),
            "checkpoint_summary_inputs": _artifact_binding(
                output_dir, summary_path
            ),
        },
    }


def _audit_proxy_inventory(
    output_dir: Path,
    *,
    run_id: str,
    formal: bool,
    assigned: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    assigned_by_id: dict[str, dict[str, Any]] = {}
    for raw in assigned:
        row = dict(raw)
        event_id = str(row.get("event_id", ""))
        if not event_id or event_id in assigned_by_id:
            raise R403TaskEvalAuditError("assigned proxy event IDs differ")
        if row.get("run_id") != run_id:
            raise R403TaskEvalAuditError("assigned proxy run ID differs")
        assigned_by_id[event_id] = row
    observed: list[dict[str, Any]] = []
    flex_reports: list[dict[str, Any]] = []
    invocation_count = 0
    if formal:
        proxy_root = output_dir / "proxy"
        if proxy_root.is_symlink() or not proxy_root.is_dir():
            raise R403TaskEvalAuditError("formal proxy root is absent")
        for invocation in sorted(proxy_root.glob("invocation-*")):
            if invocation.is_symlink() or not invocation.is_dir():
                raise R403TaskEvalAuditError("proxy invocation is unsafe")
            manifest = _strict_json(invocation / "manifest.json")
            if (
                manifest.get("run_id") != run_id
                or manifest.get("invocation_id") != invocation.name
                or manifest.get("returncode") is None
            ):
                raise R403TaskEvalAuditError("proxy invocation manifest differs")
            expected_names = {
                "ready.json",
                "requests.jsonl",
                "process.log",
                "start.json",
                "manifest.json",
            }
            if {path.name for path in invocation.iterdir()} != expected_names:
                raise R403TaskEvalAuditError("proxy invocation artifact inventory differs")
            log = invocation / "requests.jsonl"
            if (
                manifest.get("log_sha256") != answer_contract.sha256_file(log)
                or manifest.get("log_bytes") != log.stat().st_size
            ):
                raise R403TaskEvalAuditError("proxy invocation log binding differs")
            invocation_events = _read_jsonl(log)
            try:
                flex_reports.append(
                    flex_evidence.audit_window(
                        manifest.get("provider_window"),
                        consumer_records=invocation_events,
                    )
                )
            except flex_evidence.EvidenceError as exc:
                raise R403TaskEvalAuditError(
                    f"Flex provider window differs: {exc}"
                ) from exc
            observed.extend(invocation_events)
            invocation_count += 1
        if invocation_count < 1:
            raise R403TaskEvalAuditError("formal proxy invocation is absent")
    else:
        log = output_dir / "synthetic_proxy.jsonl"
        observed = _read_jsonl(log)
        if any(
            row.get("synthetic_no_network") is not True
            and not str(row.get("response_id", "")).startswith("fake-")
            for row in observed
        ):
            raise R403TaskEvalAuditError(
                "synthetic proxy record lacks scripted-response evidence"
            )
    observed_by_id: dict[str, dict[str, Any]] = {}
    for row in observed:
        event_id = str(row.get("event_id", ""))
        if not event_id or event_id in observed_by_id:
            raise R403TaskEvalAuditError("proxy log event IDs differ")
        if row.get("run_id") != run_id:
            raise R403TaskEvalAuditError("proxy log run ID differs")
        observed_by_id[event_id] = row
    if set(observed_by_id) != set(assigned_by_id):
        raise R403TaskEvalAuditError("proxy log has missing or unassigned events")
    for event_id, row in observed_by_id.items():
        if row != assigned_by_id[event_id]:
            raise R403TaskEvalAuditError("proxy event content differs from ledger evidence")
    return {
        "invocations": invocation_count,
        "events": len(observed),
        "flex_gateway_windows": flex_reports,
        "successful_events": sum(row.get("status") == "success" for row in observed),
        "error_events": sum(row.get("status") == "error" for row in observed),
        "event_ids_sha256": answer_contract.canonical_hash(
            sorted(observed_by_id)
        ),
    }


def _manifest_inventory(specs: Sequence[Mapping[str, Any]]) -> str:
    return answer_contract.canonical_hash(
        [
            {
                "artifact_id": spec["artifact_id"],
                "question_id": spec["question_id"],
                "sample": spec["sample"],
                "checkpoint": spec["checkpoint"],
                "question_sha256": answer_contract.sha256_bytes(
                    str(spec["question"]).encode("utf-8")
                ),
                "gold_answer_sha256": spec["gold_answer_sha256"],
                "inventory_row_sha256": spec["inventory_row_sha256"],
                "cohorts": spec["cohorts"],
            }
            for spec in specs
        ]
    )


def _audit_tree_inventory(
    output_dir: Path, specs: Sequence[Mapping[str, Any]], *, formal: bool
) -> None:
    expected_by_checkpoint = {
        checkpoint: {
            str(spec["artifact_id"])
            for spec in specs
            if int(spec["checkpoint"]) == checkpoint
        }
        for checkpoint in contract.CHECKPOINTS
    }
    checkpoint_root = output_dir / "checkpoints"
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise R403TaskEvalAuditError("task checkpoint root is absent")
    expected_dirs = {
        f"checkpoint-{checkpoint:03d}" for checkpoint in contract.CHECKPOINTS
    }
    if {path.name for path in checkpoint_root.iterdir()} != expected_dirs:
        raise R403TaskEvalAuditError("task checkpoint directory inventory differs")
    for checkpoint in contract.CHECKPOINTS:
        root = checkpoint_root / f"checkpoint-{checkpoint:03d}"
        if {path.name for path in root.iterdir()} != {"questions"}:
            raise R403TaskEvalAuditError("task checkpoint root has stale artifacts")
        questions = root / "questions"
        if {path.name for path in questions.iterdir()} != expected_by_checkpoint[
            checkpoint
        ]:
            raise R403TaskEvalAuditError("task question inventory has missing/extra rows")
    allowed = {
        ".r403-task-eval.lock",
        "run_manifest.json",
        "source_binding.json",
        "progress.json",
        "checkpoints",
        "results.json",
        "scoring_inputs.jsonl",
        "checkpoint_summary_inputs.json",
        "complete.json",
        "audit.json",
        "proxy" if formal else "synthetic_proxy.jsonl",
    }
    names = {path.name for path in output_dir.iterdir()}
    if names - allowed or (allowed - {"audit.json"}) - names:
        raise R403TaskEvalAuditError("task run root artifact inventory differs")


def audit_run(output_dir: Path, *, require_complete: bool = True) -> dict[str, Any]:
    output_dir = output_dir.expanduser().absolute()
    answer_contract.reject_symlink_components(output_dir)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise R403TaskEvalAuditError("R403 task output root is invalid")
    manifest = _strict_json(output_dir / "run_manifest.json")
    if not isinstance(manifest, dict):
        raise R403TaskEvalAuditError("run manifest is not an object")
    _validate_content_hash(manifest, "manifest_content_sha256")
    mode = manifest.get("mode")
    if mode not in {"formal", "synthetic_no_network"}:
        raise R403TaskEvalAuditError("R403 task mode differs")
    formal = mode == "formal"
    if (
        manifest.get("schema_version") != RUN_SCHEMA
        or manifest.get("status") != "answering"
        or manifest.get("method") != contract.METHOD
        or manifest.get("generic_readonly_condition")
        != contract.GENERIC_CONDITION
        or manifest.get("config", {}).get("model") != contract.MODEL
        or manifest.get("config", {}).get("budget_tokens")
        != contract.BUDGET_TOKENS
        or manifest.get("config", {}).get("max_rounds") != contract.MAX_ROUNDS
        or manifest.get("config", {}).get("future_checkpoint_access") is not False
    ):
        raise R403TaskEvalAuditError("run manifest protocol differs")
    if formal:
        try:
            provider_contract = flex_evidence.validate_recorded_contract(
                manifest.get("config", {}).get("gateway_contract")
            )
        except flex_evidence.EvidenceError as exc:
            raise R403TaskEvalAuditError(
                f"formal Flex gateway contract differs: {exc}"
            ) from exc
        if (
            manifest.get("config", {}).get("upstream")
            != provider_contract["origin"]
        ):
            raise R403TaskEvalAuditError("formal upstream differs")
    elif any(
        manifest.get("config", {}).get(key) is not None
        for key in ("upstream", "gateway_contract", "python", "python_sha256")
    ):
        raise R403TaskEvalAuditError("synthetic manifest contains formal runtime")
    contract.validate_preregistration(contract.DEFAULT_PREREGISTRATION)
    if manifest.get("preregistration") != contract.preregistration_binding(
        contract.DEFAULT_PREREGISTRATION
    ):
        raise R403TaskEvalAuditError("run preregistration binding differs")
    persisted_binding = _strict_json(output_dir / "source_binding.json")
    if not isinstance(persisted_binding, dict):
        raise R403TaskEvalAuditError("source binding is not an object")
    contract.validate_binding_hash(persisted_binding)
    source_root = ROOT / str(persisted_binding["source_root"])
    live_binding = contract.audit_growth_source(source_root, formal=formal)
    if live_binding != persisted_binding:
        raise R403TaskEvalAuditError("live R403 source binding differs")
    if manifest.get("source_binding_sha256") != persisted_binding[
        "binding_sha256"
    ]:
        raise R403TaskEvalAuditError("run/source binding differs")
    specs = contract.specs_from_binding(
        source_root, persisted_binding, formal=formal
    )
    by_checkpoint = Counter(int(spec["checkpoint"]) for spec in specs)
    expected_scope = {
        "question_checkpoint_artifacts": len(specs),
        "by_checkpoint": {
            str(checkpoint): by_checkpoint[checkpoint]
            for checkpoint in contract.CHECKPOINTS
        },
    }
    if (
        manifest.get("scope") != expected_scope
        or manifest.get("question_inventory_sha256") != _manifest_inventory(specs)
    ):
        raise R403TaskEvalAuditError("run question inventory differs")
    _audit_tree_inventory(output_dir, specs, formal=formal)
    reports = [
        audit_question_checkpoint(
            output_dir=output_dir,
            source_root=source_root,
            source_binding=persisted_binding,
            spec=spec,
            formal=formal,
        )
        for spec in specs
    ]
    proxy = _audit_proxy_inventory(
        output_dir,
        run_id=str(manifest["run_id"]),
        formal=formal,
        assigned=[
            event for report in reports for event in report["proxy_events"]
        ],
    )
    aggregates = _audit_aggregates(
        output_dir, specs=specs, reports=reports
    )
    progress = _strict_json(output_dir / "progress.json")
    expected_progress = {
        "schema_version": RUN_SCHEMA,
        "status": "complete",
        "completed": len(specs),
        "total": len(specs),
    }
    if progress != expected_progress:
        raise R403TaskEvalAuditError("R403 task progress differs")
    complete_path = output_dir / "complete.json"
    if require_complete and not complete_path.is_file():
        raise R403TaskEvalAuditError("R403 task complete marker is absent")
    complete = _strict_json(complete_path)
    expected_complete = {
        "schema_version": COMPLETE_SCHEMA,
        "status": "complete",
        "run_id": manifest["run_id"],
        "mode": mode,
        "question_checkpoint_artifacts": len(specs),
        "source_binding_sha256": persisted_binding["binding_sha256"],
        "preregistration_sha256": answer_contract.sha256_file(
            contract.DEFAULT_PREREGISTRATION
        ),
        "progress_sha256": answer_contract.sha256_file(
            output_dir / "progress.json"
        ),
        "outputs": aggregates["outputs"],
    }
    if not formal:
        expected_complete.update({"model_requests": 0, "network_requests": 0})
    if complete != expected_complete:
        raise R403TaskEvalAuditError("R403 task complete marker differs")
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "pass",
        "mode": mode,
        "run_id": manifest["run_id"],
        "source_binding_sha256": persisted_binding["binding_sha256"],
        "question_checkpoint_artifacts": len(specs),
        "by_checkpoint": {
            str(checkpoint): by_checkpoint[checkpoint]
            for checkpoint in contract.CHECKPOINTS
        },
        "source_recall_artifacts": sum(
            bool(spec["source_recall_eligible"]) for spec in specs
        ),
        "old_fact_artifacts": sum(spec["cohorts"]["old_fact"] for spec in specs),
        "update_artifacts": sum(spec["cohorts"]["update"] for spec in specs),
        "proxy": proxy,
        "aggregate_usage": _aggregate_usage(aggregates["records"]),
        "future_source_ids_observed": 0,
        "model_requests": None if formal else 0,
        "network_requests": None if formal else 0,
        "auditor_independence": {
            "imports_task_runner": False,
            "auditor_sha256": answer_contract.sha256_file(Path(__file__)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = audit_run(args.artifact, require_complete=True)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {"status": "fail", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        report_path = args.report.expanduser().absolute()
        if args.artifact.expanduser().absolute() in report_path.parents:
            raise R403TaskEvalAuditError(
                "independent audit report must be outside audited artifact"
            )
        answer_contract.atomic_json_replace(report_path, report)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
