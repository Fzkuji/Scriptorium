#!/usr/bin/env python3
"""Run R207 task evaluation on both frozen path conditions.

Preregistration, preflight, and synthetic modes are local-only.  Formal mode
has no question/sample limit flags and requires both ``--allow-model-requests``
and an independently audited all-ten R207 source.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
import run_controlled_locomo_answers as controlled_answer  # noqa: E402
from run_r116_formal import (  # noqa: E402
    _ProxyLoggingScriptedCompletionResource,
)
from src.evaluation.durable_model_ledger import read_ledger  # noqa: E402


RUN_SCHEMA = "nativemem.r207-task-eval-run.v1"
CHECKPOINT_SCHEMA = "nativemem.r207-task-eval-question-checkpoint.v1"
COMPLETE_SCHEMA = "nativemem.r207-task-eval-complete.v1"
DEFAULT_SYNTHETIC = (
    ROOT / "results/paper-experiments-20260714/r207-task-eval-synthetic-sanity"
)
DEFAULT_FORMAL = (
    ROOT / "results/paper-experiments-20260714/r207-task-eval-gpt55"
)


class R207TaskEvalRunError(RuntimeError):
    pass


def _validate_output_root(path: Path) -> Path:
    absolute = path.expanduser().absolute()
    answer_contract.reject_symlink_components(absolute)
    if absolute.is_symlink() or absolute == ROOT or ROOT not in absolute.parents:
        raise R207TaskEvalRunError("output root must be a repository descendant")
    return absolute


def _write_or_validate(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise R207TaskEvalRunError(f"existing artifact is unsafe: {path}")
        if answer_contract.read_json(path) != payload:
            raise R207TaskEvalRunError(f"existing immutable artifact differs: {path}")
        return
    answer_contract.atomic_json_no_clobber(path, payload)


def _write_text_no_clobber(path: Path, text: str) -> None:
    answer_contract.reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = text.encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _question_paths(
    output_dir: Path, condition: str, spec: Mapping[str, Any]
) -> dict[str, Path]:
    root = (
        output_dir
        / "conditions"
        / condition
        / "questions"
        / str(spec["artifact_id"])
    )
    return {
        "root": root,
        "input": root / "input_binding.json",
        "provenance": root / "stage_provenance.json",
        "upstream": root / "upstream_r207_trace.json",
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
    values.append(str(result["answer"]["response_id"]))
    if any(not value for value in values):
        raise R207TaskEvalRunError("question response ID is empty")
    return values


def execute_bound_question(
    *,
    output_dir: Path,
    run_id: str,
    preregistration: Path,
    source_root: Path,
    source_binding: Mapping[str, Any],
    source_context: Mapping[str, Any],
    spec: Mapping[str, Any],
    condition: str,
    completion_resource: Any,
    answer_client: Any,
    tokenizer: Any,
    formal: bool,
    proxy_log: Path | None,
) -> dict[str, Any]:
    """Execute one question or independently validate its checkpoint."""

    paths = _question_paths(output_dir, condition, spec)
    if paths["checkpoint"].exists():
        from audit_r207_task_eval import audit_question_checkpoint  # noqa: PLC0415

        return audit_question_checkpoint(
            output_dir=output_dir,
            source_root=source_root,
            source_binding=source_binding,
            source_context=source_context,
            spec=spec,
            condition=condition,
            formal=formal,
        )
    if paths["root"].exists() or paths["root"].is_symlink():
        raise R207TaskEvalRunError(
            "incomplete question artifact exists; use a new output root: "
            f"{paths['root']}"
        )
    memory = source_context["memories"][condition]
    upstream_trace = contract.trace_for_question(
        context=source_context,
        spec=spec,
        condition=condition,
        formal=formal,
    )
    provenance = contract.build_stage_provenance(
        spec=spec,
        bank=source_context["bank"],
        memory=memory,
        upstream_trace=upstream_trace,
    )
    binding = contract.input_binding(
        preregistration=preregistration,
        source_binding=source_binding,
        spec=spec,
        condition=condition,
        memory=memory,
        upstream_trace=upstream_trace,
        stage_provenance=provenance,
    )
    paths["root"].mkdir(parents=True)
    answer_contract.atomic_json_no_clobber(paths["input"], binding)
    answer_contract.atomic_json_no_clobber(paths["provenance"], provenance)
    answer_contract.atomic_json_no_clobber(paths["upstream"], upstream_trace)
    turn_index = readonly_control.build_turn_index(
        contract.conversation_for_spec(spec)
    )
    condition_run_id = f"{run_id}:{condition}"
    result = readonly_control.execute_question(
        artifact_dir=paths["attempt"],
        run_id=condition_run_id,
        method=contract.METHOD,
        condition=contract.GENERIC_CONDITION,
        memory_root=memory,
        turn_index=turn_index,
        question_id=str(spec["question_id"]),
        question=str(spec["question"]),
        gold_source_ids=list(spec["gold_source_ids"]),
        source_recall_eligible=bool(spec["source_recall_eligible"]),
        completion_resource=completion_resource,
        answer_client=answer_client,
        tokenizer=tokenizer,
        formal=formal,
        proxy_log=proxy_log,
        budget_tokens=contract.BUDGET_TOKENS,
        max_rounds=contract.MAX_ROUNDS,
        model_context_limit_tokens=contract.MODEL_CONTEXT_LIMIT_TOKENS,
        answer_completion_reservation_tokens=(
            contract.ANSWER_COMPLETION_RESERVATION_TOKENS
        ),
        stage_provenance=provenance,
    )
    m4_trace = contract.build_m4_trace(
        condition=condition,
        upstream_trace=upstream_trace,
        observed_stage=result["stage_evidence"],
    )
    answer_contract.atomic_json_no_clobber(paths["m4"], m4_trace)
    metrics = contract.build_question_metrics(
        attempt=paths["attempt"],
        spec=spec,
        condition=condition,
        result=result,
        m4_trace=m4_trace,
    )
    answer_contract.atomic_json_no_clobber(paths["metrics"], metrics)
    question_audit = generic_auditor.audit_question(
        artifact_dir=paths["attempt"],
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
    answer_contract.atomic_json_no_clobber(
        paths["question_audit"], question_audit
    )
    checkpoint: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "complete",
        "question_id": spec["question_id"],
        "artifact_id": spec["artifact_id"],
        "dataset_index": spec["dataset_index"],
        "question_index": spec["question_index"],
        "category": spec["category"],
        "path_condition": condition,
        "generic_readonly_condition": contract.GENERIC_CONDITION,
        "artifacts": {
            "input_binding": _artifact_binding(output_dir, paths["input"]),
            "stage_provenance": _artifact_binding(
                output_dir, paths["provenance"]
            ),
            "upstream_r207_trace": _artifact_binding(
                output_dir, paths["upstream"]
            ),
            "attempt_result": _artifact_binding(
                output_dir, paths["attempt"] / "result.json"
            ),
            "m4_stage_trace": _artifact_binding(output_dir, paths["m4"]),
            "metrics": _artifact_binding(output_dir, paths["metrics"]),
            "question_audit": _artifact_binding(
                output_dir, paths["question_audit"]
            ),
        },
        "actual_model": contract.MODEL,
        "response_ids": _response_ids(paths["attempt"], result),
        "memory_sha256": result["memory"]["after"]["sha256"],
        "official_locomo_f1": metrics["official_locomo_f1"],
    }
    checkpoint["checkpoint_content_sha256"] = answer_contract.canonical_hash(
        checkpoint
    )
    answer_contract.atomic_json_no_clobber(paths["checkpoint"], checkpoint)
    from audit_r207_task_eval import audit_question_checkpoint  # noqa: PLC0415

    return audit_question_checkpoint(
        output_dir=output_dir,
        source_root=source_root,
        source_binding=source_binding,
        source_context=source_context,
        spec=spec,
        condition=condition,
        formal=formal,
    )


def _artifact_binding(output_dir: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": answer_contract.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _question_inventory(specs: Sequence[Mapping[str, Any]]) -> str:
    return answer_contract.canonical_hash(
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


def _manifest(
    *,
    mode: str,
    preregistration: Path,
    source_binding: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    upstream: str | None,
    gateway_contract: Mapping[str, Any] | None,
    python: Path | None,
) -> dict[str, Any]:
    formal = mode == "formal"
    run_id = (
        "r207-task-eval-"
        f"{str(source_binding['binding_sha256'])[:16]}-"
        f"{answer_contract.sha256_file(preregistration)[:16]}"
        + ("" if formal else "-synthetic")
    )
    return {
        "schema_version": RUN_SCHEMA,
        "status": "answering",
        "mode": mode,
        "run_id": run_id,
        "experiment_id": "R207-task-evaluation",
        "method": contract.METHOD,
        "path_conditions": list(contract.PATH_CONDITIONS),
        "generic_readonly_condition": contract.GENERIC_CONDITION,
        "scope": {
            "samples": list(range(10)) if formal else [0],
            "questions_per_condition": len(specs),
            "question_condition_artifacts": len(specs)
            * len(contract.PATH_CONDITIONS),
            "primary_categories": [1, 2, 3, 4],
            "category_5_questions_mixed": 0,
        },
        "question_inventory_sha256": _question_inventory(specs),
        "preregistration": contract.preregistration_binding(preregistration),
        "source_binding_sha256": source_binding["binding_sha256"],
        "config": {
            "retrieval_model": contract.MODEL,
            "answer_model": contract.MODEL,
            "budget_policy": "hard_cap",
            "budget_tokens": contract.BUDGET_TOKENS,
            "max_rounds": contract.MAX_ROUNDS,
            "model_context_limit_tokens": contract.MODEL_CONTEXT_LIMIT_TOKENS,
            "answer_completion_reservation_tokens": (
                contract.ANSWER_COMPLETION_RESERVATION_TOKENS
            ),
            "answer_max_tokens": contract.ANSWER_MAX_TOKENS,
            "answer_retries": contract.ANSWER_RETRIES,
            "tokenizer": answer_contract.formal_token_counter().identity,
            "upstream": upstream,
            "gateway_contract": (
                dict(gateway_contract) if gateway_contract is not None else None
            ),
            "python": str(python) if python is not None else None,
            "python_sha256": (
                answer_contract.sha256_file(python) if python is not None else None
            ),
            "model_requests_authorized": formal,
            "network_requests": None if formal else 0,
        },
    }


def _progress(output_dir: Path, total: int, *, formal: bool) -> dict[str, Any]:
    completed = len(
        list(output_dir.glob("conditions/*/questions/*/checkpoint.json"))
    )
    return {
        "schema_version": RUN_SCHEMA,
        "status": "complete" if completed == total else "answering",
        "completed": completed,
        "total": total,
        "model_requests_authorized": formal,
    }


def _collect_contexts(
    *,
    source_root: Path,
    source_binding: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    formal: bool,
) -> dict[int, dict[str, Any]]:
    samples = sorted({int(spec["dataset_index"]) for spec in specs})
    return {
        sample: contract.load_sample_context(
            source_root,
            source_binding,
            sample=sample,
            formal=formal,
        )
        for sample in samples
    }


def _aggregate_records(
    output_dir: Path, specs: Sequence[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in contract.PATH_CONDITIONS
    }
    for condition in contract.PATH_CONDITIONS:
        for spec in specs:
            paths = _question_paths(output_dir, condition, spec)
            result = answer_contract.read_json(paths["attempt"] / "result.json")
            metrics = answer_contract.read_json(paths["metrics"])
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


def _summary(records: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    by_condition: dict[str, dict[str, Mapping[str, Any]]] = {}
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
        by_condition[condition] = {str(row["question_id"]): row for row in rows}
    left = by_condition["model_directed"]
    right = by_condition["deterministic_permutation"]
    if set(left) != set(right):
        raise R207TaskEvalRunError("paired condition inventory differs")
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


def _publish_aggregates(
    output_dir: Path, specs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    records = _aggregate_records(output_dir, specs)
    results_payload = {"schema_version": RUN_SCHEMA, "records": records}
    results_path = output_dir / "results.json"
    _write_or_validate(results_path, results_payload)
    summary_payload = _summary(records)
    summary_path = output_dir / "summary.json"
    _write_or_validate(summary_path, summary_payload)
    outputs: dict[str, Any] = {
        "results": _artifact_binding(output_dir, results_path),
        "summary": _artifact_binding(output_dir, summary_path),
        "hypotheses": {},
    }
    for condition in contract.PATH_CONDITIONS:
        path = output_dir / "hypotheses" / f"{condition}.jsonl"
        text = "".join(
            answer_contract.canonical_json(
                {
                    "question_id": row["question_id"],
                    "hypothesis": row["answer"],
                }
            )
            + "\n"
            for row in records[condition]
        )
        if path.exists():
            if path.is_symlink() or path.read_text(encoding="utf-8") != text:
                raise R207TaskEvalRunError("existing hypotheses artifact differs")
        else:
            _write_text_no_clobber(path, text)
        outputs["hypotheses"][condition] = _artifact_binding(output_dir, path)
    return outputs


def run_formal(
    *,
    output_dir: Path,
    source_root: Path,
    allow_model_requests: bool = False,
    preregistration: Path = contract.DEFAULT_PREREGISTRATION,
    preflight_path: Path = contract.DEFAULT_PREFLIGHT,
    gateway_root: Path | None = None,
    python: Path = Path(sys.executable),
) -> dict[str, Any]:
    if not allow_model_requests:
        raise R207TaskEvalRunError("formal mode requires --allow-model-requests")
    if gateway_root is None:
        raise R207TaskEvalRunError("formal mode requires --gateway-root")
    gateway_root = gateway_root.expanduser().resolve()
    output_dir = _validate_output_root(output_dir)
    source_root = source_root.expanduser().absolute()
    preflight = contract.run_preflight(
        preregistration,
        source_root=source_root,
        write_path=preflight_path,
    )
    source_binding = contract.require_ready_binding(preflight)
    specs = contract.primary_question_specs()
    python = python.expanduser().resolve()
    if python.is_symlink() or not python.is_file():
        raise R207TaskEvalRunError("formal Python executable is unavailable")
    provider_lock = None
    try:
        provider_lock = controlled_answer.flex_evidence.acquire_consumer_lock(
            gateway_root
        )
        gateway_contract = controlled_answer.flex_evidence.active_contract(
            gateway_root
        )
    except controlled_answer.flex_evidence.EvidenceError as exc:
        if provider_lock is not None:
            provider_lock.close()
        raise R207TaskEvalRunError(str(exc)) from exc
    upstream = str(gateway_contract["origin"])
    manifest = _manifest(
        mode="formal",
        preregistration=preregistration,
        source_binding=source_binding,
        specs=specs,
        upstream=upstream,
        gateway_contract=gateway_contract,
        python=python,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with provider_lock, answer_contract.FileLock(output_dir / ".r207-task-eval.lock"):
        complete_path = output_dir / "complete.json"
        if complete_path.exists():
            from audit_r207_task_eval import audit_run  # noqa: PLC0415

            return audit_run(output_dir, require_complete=True)
        _write_or_validate(output_dir / "run_manifest.json", manifest)
        _write_or_validate(output_dir / "source_binding.json", source_binding)
        contexts = _collect_contexts(
            source_root=source_root,
            source_binding=source_binding,
            specs=specs,
            formal=True,
        )
        total = len(specs) * len(contract.PATH_CONDITIONS)
        progress_path = output_dir / "progress.json"
        answer_contract.atomic_json_replace(
            progress_path, _progress(output_dir, total, formal=True)
        )
        contract.verify_source_unchanged(
            source_root=source_root, source_binding=source_binding, formal=True
        )
        controlled_answer._recover_interrupted_proxies(  # noqa: SLF001
            output_dir, manifest["run_id"]
        )
        pending: list[tuple[str, dict[str, Any]]] = []
        tokenizer = answer_contract.formal_token_counter()
        for condition in contract.PATH_CONDITIONS:
            for spec in specs:
                context = contexts[int(spec["dataset_index"])]
                if _question_paths(output_dir, condition, spec)[
                    "checkpoint"
                ].exists():
                    execute_bound_question(
                        output_dir=output_dir,
                        run_id=manifest["run_id"],
                        preregistration=preregistration,
                        source_root=source_root,
                        source_binding=source_binding,
                        source_context=context,
                        spec=spec,
                        condition=condition,
                        completion_resource=None,
                        answer_client=None,
                        tokenizer=tokenizer,
                        formal=True,
                        proxy_log=None,
                    )
                else:
                    pending.append((condition, spec))
        process = process_log = proxy_record = None
        try:
            if pending:
                process, proxy_record, process_log = controlled_answer._start_proxy(  # noqa: SLF001
                    output_dir=output_dir,
                    python=python,
                    gateway_contract=gateway_contract,
                    run_id=manifest["run_id"],
                )
                from openai import OpenAI  # noqa: PLC0415

                client = OpenAI(
                    api_key="x",
                    base_url=proxy_record["base_url"],
                    max_retries=0,
                    timeout=360.0,
                )
                completion_resource = client.chat.completions
                answer_client = controlled_answer.HttpAnswerClient(
                    base_url=proxy_record["base_url"],
                    retries=contract.ANSWER_RETRIES,
                    answer_max_tokens=contract.ANSWER_MAX_TOKENS,
                )
                proxy_log = output_dir / proxy_record["log"]
                for condition, spec in pending:
                    execute_bound_question(
                        output_dir=output_dir,
                        run_id=manifest["run_id"],
                        preregistration=preregistration,
                        source_root=source_root,
                        source_binding=source_binding,
                        source_context=contexts[int(spec["dataset_index"])],
                        spec=spec,
                        condition=condition,
                        completion_resource=completion_resource,
                        answer_client=answer_client,
                        tokenizer=tokenizer,
                        formal=True,
                        proxy_log=proxy_log,
                    )
                    answer_contract.atomic_json_replace(
                        progress_path, _progress(output_dir, total, formal=True)
                    )
        finally:
            if process is not None and proxy_record is not None and process_log is not None:
                controlled_answer._stop_proxy(  # noqa: SLF001
                    process, proxy_record, process_log, output_dir
                )
        contract.verify_source_unchanged(
            source_root=source_root, source_binding=source_binding, formal=True
        )
        progress = _progress(output_dir, total, formal=True)
        if progress["completed"] != total:
            raise R207TaskEvalRunError("formal R207 task scope is incomplete")
        answer_contract.atomic_json_replace(progress_path, progress)
        outputs = _publish_aggregates(output_dir, specs)
        complete = {
            "schema_version": COMPLETE_SCHEMA,
            "status": "complete",
            "run_id": manifest["run_id"],
            "questions_per_condition": len(specs),
            "question_condition_artifacts": total,
            "category_5_questions_mixed": 0,
            "source_binding_sha256": source_binding["binding_sha256"],
            "preregistration_sha256": answer_contract.sha256_file(
                preregistration
            ),
            "progress_sha256": answer_contract.sha256_file(progress_path),
            "outputs": outputs,
        }
        answer_contract.atomic_json_no_clobber(complete_path, complete)
        from audit_r207_task_eval import audit_run  # noqa: PLC0415

        report = audit_run(output_dir, require_complete=True)
        answer_contract.atomic_json_replace(output_dir / "audit.json", report)
        return report


def run_synthetic(
    output_dir: Path = DEFAULT_SYNTHETIC,
    *,
    source_root: Path = contract.DEFAULT_SYNTHETIC_SOURCE,
    preregistration: Path = contract.DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    contract.validate_preregistration(preregistration)
    output_dir = _validate_output_root(output_dir)
    source_root = source_root.expanduser().absolute()
    source_binding = contract.audit_r207_source(source_root, formal=False)
    spec = contract.synthetic_spec()
    specs = [spec]
    manifest = _manifest(
        mode="synthetic_no_network",
        preregistration=preregistration,
        source_binding=source_binding,
        specs=specs,
        upstream=None,
        gateway_contract=None,
        python=None,
    )
    if (output_dir / "complete.json").exists():
        from audit_r207_task_eval import audit_run  # noqa: PLC0415

        return audit_run(output_dir, require_complete=True)
    if output_dir.exists():
        raise R207TaskEvalRunError(
            "existing synthetic output is incomplete; use a new output root"
        )
    output_dir.mkdir(parents=True)
    with answer_contract.FileLock(output_dir / ".r207-task-eval.lock"):
        answer_contract.atomic_json_no_clobber(
            output_dir / "run_manifest.json", manifest
        )
        answer_contract.atomic_json_no_clobber(
            output_dir / "source_binding.json", source_binding
        )
        context = contract.load_sample_context(
            source_root, source_binding, sample=0, formal=False
        )
        proxy_log = output_dir / "synthetic_proxy.jsonl"
        answer_client = controlled_answer.FakeAnswerClient(
            proxy_log=proxy_log,
            run_id=manifest["run_id"],
            response_text="<answer>a dog</answer>",
        )
        retrieval = _ProxyLoggingScriptedCompletionResource(
            [
                [
                    {
                        "name": "read_memory_file",
                        "arguments": {"path": "topics/A/pets.md"},
                    }
                ],
                [
                    {
                        "name": "resolve_sources",
                        "arguments": {"source_ids": ["D1:1"]},
                    }
                ],
                [],
                [
                    {
                        "name": "read_memory_file",
                        "arguments": {"path": "topics/B/home.md"},
                    }
                ],
                [
                    {
                        "name": "resolve_sources",
                        "arguments": {"source_ids": ["D1:1"]},
                    }
                ],
                [],
            ],
            proxy_log=proxy_log,
            run_id=manifest["run_id"],
        )
        tokenizer = answer_contract.formal_token_counter()
        for condition in contract.PATH_CONDITIONS:
            execute_bound_question(
                output_dir=output_dir,
                run_id=manifest["run_id"],
                preregistration=preregistration,
                source_root=source_root,
                source_binding=source_binding,
                source_context=context,
                spec=spec,
                condition=condition,
                completion_resource=retrieval,
                answer_client=answer_client,
                tokenizer=tokenizer,
                formal=False,
                proxy_log=proxy_log,
            )
        outputs = _publish_aggregates(output_dir, specs)
        complete = {
            "schema_version": COMPLETE_SCHEMA,
            "status": "complete",
            "run_id": manifest["run_id"],
            "questions_per_condition": 1,
            "question_condition_artifacts": 2,
            "category_5_questions_mixed": 0,
            "model_requests": 0,
            "network_requests": 0,
            "simulated_retrieval_calls": retrieval.calls,
            "simulated_answer_calls": len(answer_client.prompts),
            "outputs": outputs,
        }
        answer_contract.atomic_json_no_clobber(
            output_dir / "complete.json", complete
        )
        from audit_r207_task_eval import audit_run  # noqa: PLC0415

        report = audit_run(output_dir, require_complete=True)
        answer_contract.atomic_json_replace(output_dir / "audit.json", report)
        return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preregister", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--synthetic-sanity", action="store_true")
    modes.add_argument("--formal", action="store_true")
    parser.add_argument("--preregistration", type=Path, default=contract.DEFAULT_PREREGISTRATION)
    parser.add_argument("--preflight-report", type=Path, default=contract.DEFAULT_PREFLIGHT)
    parser.add_argument("--r207-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-model-requests", action="store_true")
    parser.add_argument("--gateway-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args(argv)
    if args.allow_model_requests and not args.formal:
        parser.error("--allow-model-requests is valid only with --formal")
    if args.formal and not args.allow_model_requests:
        parser.error("formal mode requires --allow-model-requests")
    if args.formal and args.r207_root is None:
        parser.error("formal mode requires explicit --r207-root")
    if args.formal and args.output_dir is None:
        parser.error("formal mode requires --output-dir")
    if args.formal and args.gateway_root is None:
        parser.error("formal mode requires --gateway-root")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preregister:
        payload = contract.freeze_preregistration(args.preregistration)
    elif args.preflight:
        payload = contract.run_preflight(
            args.preregistration,
            source_root=args.r207_root or contract.DEFAULT_R207_ROOT,
            write_path=args.preflight_report,
        )
    elif args.synthetic_sanity:
        payload = run_synthetic(
            args.output_dir or DEFAULT_SYNTHETIC,
            source_root=args.r207_root or contract.DEFAULT_SYNTHETIC_SOURCE,
            preregistration=args.preregistration,
        )
    else:
        payload = run_formal(
            output_dir=args.output_dir or DEFAULT_FORMAL,
            source_root=args.r207_root,
            allow_model_requests=args.allow_model_requests,
            preregistration=args.preregistration,
            preflight_path=args.preflight_report,
            gateway_root=args.gateway_root,
            python=args.python,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
