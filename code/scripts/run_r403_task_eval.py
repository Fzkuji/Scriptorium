#!/usr/bin/env python3
"""Run fixed GPT-5.5 retrieval and answering over R403 checkpoints.

Formal mode has no sample or question limit flags.  It requires an explicit
authorization gate, an independently audited all-ten R403 growth source, and
an upstream endpoint fixed to loopback.  Synthetic mode uses scripted local
resources and sends no network or model request.
"""

from __future__ import annotations

import argparse
import json
import os
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
import readonly_nativemem_control as readonly_control  # noqa: E402
import run_controlled_locomo_answers as controlled_answer  # noqa: E402
from src.evaluation.durable_model_ledger import read_ledger  # noqa: E402


RUN_SCHEMA = "nativemem.r403-task-eval-run.v1"
CHECKPOINT_SCHEMA = "nativemem.r403-task-eval-question-checkpoint.v1"
COMPLETE_SCHEMA = "nativemem.r403-task-eval-complete.v1"


class R403TaskEvalRunError(RuntimeError):
    pass


class _ProxyLoggingScriptedCompletionResource(
    readonly_control.ScriptedCompletionResource
):
    """Scripted retrieval resource with synthetic exclusive-proxy records."""

    def __init__(
        self,
        scripts: Sequence[Sequence[Mapping[str, Any]]],
        *,
        proxy_log: Path,
        run_id: str,
    ):
        super().__init__(scripts)
        self.proxy_log = proxy_log
        self.run_id = run_id

    def create(self, **kwargs: Any) -> Any:
        response = super().create(**kwargs)
        headers = kwargs.get("extra_headers", {})
        logical_id = str(headers.get("X-Controlled-Logical-Call-ID", ""))
        question_id = str(headers.get("X-Controlled-Question-ID", ""))
        if not logical_id or not question_id:
            raise R403TaskEvalRunError("synthetic retrieval linkage is absent")
        request_bytes = json.dumps(
            kwargs,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        response_bytes = json.dumps(
            response.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        record = {
            "run_id": self.run_id,
            "started_at": answer_contract.utc_now(),
            "finished_at": answer_contract.utc_now(),
            "status": "success",
            "http_status": 200,
            "requested_model": contract.MODEL,
            "actual_model": response.model,
            "response_id": response.id,
            "event_id": f"synthetic-r403-retrieval-{self.calls:05d}",
            "question_id": question_id,
            "logical_call_id": logical_id,
            "request_sha256": answer_contract.sha256_bytes(request_bytes),
            "response_sha256": answer_contract.sha256_bytes(response_bytes),
            "client_http_attempts": 0,
            "upstream_http_attempts": 0,
            "unsupported_parameters": [],
            "usage": response.usage,
            "error": None,
            "synthetic_no_network": True,
        }
        with self.proxy_log.open("a", encoding="utf-8") as handle:
            handle.write(answer_contract.canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return response


def _validate_output_root(path: Path) -> Path:
    absolute = path.expanduser().absolute()
    answer_contract.reject_symlink_components(absolute)
    if absolute.is_symlink() or absolute == ROOT or ROOT not in absolute.parents:
        raise R403TaskEvalRunError("output root must be a repository descendant")
    return absolute


def _write_or_validate(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise R403TaskEvalRunError(f"existing artifact is unsafe: {path}")
        if answer_contract.read_json(path) != payload:
            raise R403TaskEvalRunError(f"existing immutable artifact differs: {path}")
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


def _artifact_binding(output_dir: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": answer_contract.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _response_ids(attempt: Path, result: Mapping[str, Any]) -> list[str]:
    values = [
        str(row.get("response_id", ""))
        for row in read_ledger(attempt / "retrieval_model_ledger.jsonl")
        if row.get("event") == "model_call_finished"
    ]
    values.append(str(result["answer"]["response_id"]))
    if any(not value for value in values):
        raise R403TaskEvalRunError("R403 task response ID is empty")
    return values


def execute_bound_question(
    *,
    output_dir: Path,
    run_id: str,
    preregistration: Path,
    source_root: Path,
    source_binding: Mapping[str, Any],
    context: Mapping[str, Any],
    spec: Mapping[str, Any],
    completion_resource: Any,
    answer_client: Any,
    tokenizer: Any,
    formal: bool,
    proxy_log: Path | None,
) -> dict[str, Any]:
    """Execute one question or audit its already complete checkpoint."""

    paths = _question_paths(output_dir, spec)
    if paths["checkpoint"].exists():
        from audit_r403_task_eval import audit_question_checkpoint  # noqa: PLC0415

        return audit_question_checkpoint(
            output_dir=output_dir,
            source_root=source_root,
            source_binding=source_binding,
            spec=spec,
            formal=formal,
        )
    if paths["root"].exists() or paths["root"].is_symlink():
        raise R403TaskEvalRunError(
            "incomplete R403 question artifact exists; use a new output root: "
            f"{paths['root']}"
        )
    binding = contract.input_binding(
        preregistration=preregistration,
        source_binding=source_binding,
        checkpoint_binding=context["binding"],
        spec=spec,
        memory=context["memory"],
        turn_index=context["turn_index"],
    )
    paths["root"].mkdir(parents=True)
    answer_contract.atomic_json_no_clobber(paths["input"], binding)
    answer_contract.atomic_json_no_clobber(
        paths["upstream"], spec["upstream_stage_evidence"]
    )
    # The same LoCoMo question can be evaluated at several checkpoints.  A
    # checkpoint-specific run identity keeps durable logical-call IDs unique
    # without changing the frozen benchmark question ID.
    question_run_id = (
        f"{run_id}:sample-{int(spec['sample']):02d}:"
        f"checkpoint-{int(spec['checkpoint']):03d}"
    )
    result = readonly_control.execute_question(
        artifact_dir=paths["attempt"],
        run_id=question_run_id,
        method=contract.METHOD,
        condition=contract.GENERIC_CONDITION,
        memory_root=context["memory"],
        turn_index=context["turn_index"],
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
    )
    m4_trace = contract.build_m4_trace(
        attempt=paths["attempt"],
        spec=spec,
        turn_index=context["turn_index"],
    )
    answer_contract.atomic_json_no_clobber(paths["m4"], m4_trace)
    metrics = contract.build_question_metrics(
        attempt=paths["attempt"],
        spec=spec,
        result=result,
        m4_trace=m4_trace,
    )
    answer_contract.atomic_json_no_clobber(paths["metrics"], metrics)
    question_audit = generic_auditor.audit_question(
        artifact_dir=paths["attempt"],
        memory_root=context["memory"],
        method=contract.METHOD,
        condition=contract.GENERIC_CONDITION,
        question_id=str(spec["question_id"]),
        question=str(spec["question"]),
        gold_source_ids=list(spec["gold_source_ids"]),
        source_recall_eligible=bool(spec["source_recall_eligible"]),
        formal=formal,
    )
    answer_contract.atomic_json_no_clobber(
        paths["question_audit"], question_audit
    )
    checkpoint: dict[str, Any] = {
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
        "artifacts": {
            "input_binding": _artifact_binding(output_dir, paths["input"]),
            "upstream_stage_evidence": _artifact_binding(
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
        "memory_before_sha256": result["memory"]["before"]["sha256"],
        "memory_after_sha256": result["memory"]["after"]["sha256"],
        "memory_unchanged": result["memory"]["unchanged"],
    }
    checkpoint["checkpoint_content_sha256"] = answer_contract.canonical_hash(
        checkpoint
    )
    answer_contract.atomic_json_no_clobber(paths["checkpoint"], checkpoint)
    from audit_r403_task_eval import audit_question_checkpoint  # noqa: PLC0415

    return audit_question_checkpoint(
        output_dir=output_dir,
        source_root=source_root,
        source_binding=source_binding,
        spec=spec,
        formal=formal,
    )


def _question_inventory(specs: Sequence[Mapping[str, Any]]) -> str:
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
    run_id = (
        f"r403-task-{mode}-"
        f"{str(source_binding['binding_sha256'])[:16]}-"
        f"{answer_contract.sha256_file(preregistration)[:16]}"
    )
    by_checkpoint = Counter(int(spec["checkpoint"]) for spec in specs)
    payload = {
        "schema_version": RUN_SCHEMA,
        "status": "answering",
        "mode": mode,
        "run_id": run_id,
        "method": contract.METHOD,
        "generic_readonly_condition": contract.GENERIC_CONDITION,
        "source_binding_sha256": source_binding["binding_sha256"],
        "preregistration": contract.preregistration_binding(preregistration),
        "question_inventory_sha256": _question_inventory(specs),
        "scope": {
            "question_checkpoint_artifacts": len(specs),
            "by_checkpoint": {
                str(checkpoint): by_checkpoint[checkpoint]
                for checkpoint in contract.CHECKPOINTS
            },
        },
        "config": {
            "model": contract.MODEL,
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
            "future_checkpoint_access": False,
        },
    }
    payload["manifest_content_sha256"] = answer_contract.canonical_hash(payload)
    return payload


def _progress(output_dir: Path, specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = sum(
        _question_paths(output_dir, spec)["checkpoint"].is_file() for spec in specs
    )
    return {
        "schema_version": RUN_SCHEMA,
        "status": "complete" if complete == len(specs) else "answering",
        "completed": complete,
        "total": len(specs),
    }


def _aggregate_records(
    output_dir: Path, specs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    records = []
    for spec in specs:
        paths = _question_paths(output_dir, spec)
        result = answer_contract.read_json(paths["attempt"] / "result.json")
        metrics = answer_contract.read_json(paths["metrics"])
        records.append(
            {
                "artifact_id": spec["artifact_id"],
                "question_id": spec["question_id"],
                "sample": spec["sample"],
                "checkpoint": spec["checkpoint"],
                "session_boundary": spec["session_boundary"],
                "question_index": spec["question_index"],
                "question": spec["question"],
                "answer": result["answer"]["text"],
                "gold_answer": spec["gold_answer"],
                "category": spec["category"],
                "cohorts": spec["cohorts"],
                "source_recall_eligible": spec["source_recall_eligible"],
                "source_exclusion_reasons": spec["source_exclusion_reasons"],
                "metrics": metrics,
            }
        )
    return records


def _aggregate_usage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usage_keys = ("prompt_tokens", "completion_tokens", "total_tokens")
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
            for key in usage_keys
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


def _publish_aggregates(
    output_dir: Path, specs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    records = _aggregate_records(output_dir, specs)
    results_path = output_dir / "results.json"
    _write_or_validate(
        results_path,
        {"schema_version": RUN_SCHEMA, "records": records},
    )
    scoring_path = output_dir / "scoring_inputs.jsonl"
    scoring_text = "".join(
        answer_contract.canonical_json(
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
        )
        + "\n"
        for row in records
    )
    if scoring_path.exists():
        if scoring_path.is_symlink() or scoring_path.read_text(
            encoding="utf-8"
        ) != scoring_text:
            raise R403TaskEvalRunError("existing R403 scoring input differs")
    else:
        _write_text_no_clobber(scoring_path, scoring_text)
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
    _write_or_validate(
        summary_path,
        {
            "schema_version": RUN_SCHEMA,
            "denominator_policy": "R403_TASK_EVAL_PROTOCOL_FREEZE.json",
            "checkpoints": summaries,
        },
    )
    return {
        "results": _artifact_binding(output_dir, results_path),
        "scoring_inputs": _artifact_binding(output_dir, scoring_path),
        "checkpoint_summary_inputs": _artifact_binding(output_dir, summary_path),
    }


def run_formal(
    *,
    source_root: Path,
    output_dir: Path = contract.DEFAULT_FORMAL,
    preregistration: Path = contract.DEFAULT_PREREGISTRATION,
    preflight_path: Path = contract.DEFAULT_PREFLIGHT,
    gateway_root: Path | None = None,
    python: Path = Path(sys.executable),
    allow_model_requests: bool = False,
) -> dict[str, Any]:
    if not allow_model_requests:
        raise R403TaskEvalRunError("formal R403 task execution is not authorized")
    if gateway_root is None:
        raise R403TaskEvalRunError("formal R403 task requires --gateway-root")
    gateway_root = gateway_root.expanduser().resolve()
    output_dir = _validate_output_root(output_dir)
    source_root = source_root.expanduser().absolute()
    preflight = contract.run_preflight(
        preregistration,
        source_root=source_root,
        write_path=preflight_path,
    )
    source_binding = contract.require_ready_binding(preflight)
    specs = contract.specs_from_binding(
        source_root, source_binding, formal=True
    )
    python = python.expanduser().resolve()
    if python.is_symlink() or not python.is_file():
        raise R403TaskEvalRunError("formal Python executable is unavailable")
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
        raise R403TaskEvalRunError(str(exc)) from exc
    upstream = str(gateway_contract["origin"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with provider_lock, answer_contract.FileLock(output_dir / ".r403-task-eval.lock"):
        complete_path = output_dir / "complete.json"
        if complete_path.exists():
            from audit_r403_task_eval import audit_run  # noqa: PLC0415

            return audit_run(output_dir, require_complete=True)
        manifest = _manifest(
            mode="formal",
            preregistration=preregistration,
            source_binding=source_binding,
            specs=specs,
            upstream=upstream,
            gateway_contract=gateway_contract,
            python=python,
        )
        _write_or_validate(output_dir / "run_manifest.json", manifest)
        _write_or_validate(output_dir / "source_binding.json", source_binding)
        progress_path = output_dir / "progress.json"
        answer_contract.atomic_json_replace(
            progress_path, _progress(output_dir, specs)
        )
        contract.verify_source_unchanged(
            source_root=source_root, source_binding=source_binding, formal=True
        )
        controlled_answer._recover_interrupted_proxies(  # noqa: SLF001
            output_dir, manifest["run_id"]
        )
        tokenizer = answer_contract.formal_token_counter()
        pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for spec in specs:
            context = contract.load_checkpoint_context(
                source_root,
                source_binding,
                sample=int(spec["sample"]),
                checkpoint=int(spec["checkpoint"]),
                formal=True,
            )
            if _question_paths(output_dir, spec)["checkpoint"].exists():
                execute_bound_question(
                    output_dir=output_dir,
                    run_id=manifest["run_id"],
                    preregistration=preregistration,
                    source_root=source_root,
                    source_binding=source_binding,
                    context=context,
                    spec=spec,
                    completion_resource=None,
                    answer_client=None,
                    tokenizer=tokenizer,
                    formal=True,
                    proxy_log=None,
                )
            else:
                pending.append((dict(spec), context))
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
                for spec, context in pending:
                    execute_bound_question(
                        output_dir=output_dir,
                        run_id=manifest["run_id"],
                        preregistration=preregistration,
                        source_root=source_root,
                        source_binding=source_binding,
                        context=context,
                        spec=spec,
                        completion_resource=completion_resource,
                        answer_client=answer_client,
                        tokenizer=tokenizer,
                        formal=True,
                        proxy_log=proxy_log,
                    )
                    answer_contract.atomic_json_replace(
                        progress_path, _progress(output_dir, specs)
                    )
        finally:
            if (
                process is not None
                and proxy_record is not None
                and process_log is not None
            ):
                controlled_answer._stop_proxy(  # noqa: SLF001
                    process, proxy_record, process_log, output_dir
                )
        contract.verify_source_unchanged(
            source_root=source_root, source_binding=source_binding, formal=True
        )
        progress = _progress(output_dir, specs)
        if progress["completed"] != len(specs):
            raise R403TaskEvalRunError("formal R403 task scope is incomplete")
        answer_contract.atomic_json_replace(progress_path, progress)
        outputs = _publish_aggregates(output_dir, specs)
        complete = {
            "schema_version": COMPLETE_SCHEMA,
            "status": "complete",
            "run_id": manifest["run_id"],
            "mode": "formal",
            "question_checkpoint_artifacts": len(specs),
            "source_binding_sha256": source_binding["binding_sha256"],
            "preregistration_sha256": answer_contract.sha256_file(
                preregistration
            ),
            "progress_sha256": answer_contract.sha256_file(progress_path),
            "outputs": outputs,
        }
        answer_contract.atomic_json_no_clobber(complete_path, complete)
        from audit_r403_task_eval import audit_run  # noqa: PLC0415

        report = audit_run(output_dir, require_complete=True)
        answer_contract.atomic_json_replace(output_dir / "audit.json", report)
        return report


def _source_paths_for_spec(
    context: Mapping[str, Any], spec: Mapping[str, Any]
) -> list[str]:
    gold = set(str(value) for value in spec["gold_source_ids"])
    paths: list[str] = []
    for path in sorted(context["memory"].rglob("*.md")):
        relative = path.relative_to(context["memory"]).as_posix()
        ids = set(
            readonly_control.source_ids_in_text(path.read_text(encoding="utf-8"))
        )
        if ids & gold:
            paths.append(relative)
    return paths


def _synthetic_scripts(
    specs: Sequence[Mapping[str, Any]],
    contexts: Mapping[tuple[int, int], Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    scripts: list[list[dict[str, Any]]] = []
    for spec in specs:
        context = contexts[(int(spec["sample"]), int(spec["checkpoint"]))]
        paths = _source_paths_for_spec(context, spec)
        if paths:
            for path in paths:
                scripts.append(
                    [{"name": "read_memory_file", "arguments": {"path": path}}]
                )
        else:
            scripts.append([{"name": "list_memory_files", "arguments": {}}])
        if spec["gold_source_ids"]:
            scripts.append(
                [
                    {
                        "name": "resolve_sources",
                        "arguments": {"source_ids": spec["gold_source_ids"]},
                    }
                ]
            )
        scripts.append([])
    return scripts


def run_synthetic(
    *,
    source_root: Path,
    output_dir: Path = contract.DEFAULT_SYNTHETIC,
    preregistration: Path = contract.DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    contract.validate_preregistration(preregistration)
    source_root = source_root.expanduser().absolute()
    source_binding = contract.audit_growth_source(source_root, formal=False)
    specs = contract.specs_from_binding(
        source_root, source_binding, formal=False
    )
    output_dir = _validate_output_root(output_dir)
    if (output_dir / "complete.json").exists():
        from audit_r403_task_eval import audit_run  # noqa: PLC0415

        return audit_run(output_dir, require_complete=True)
    if output_dir.exists():
        raise R403TaskEvalRunError(
            "existing synthetic R403 output is incomplete; use a new root"
        )
    output_dir.mkdir(parents=True)
    with answer_contract.FileLock(output_dir / ".r403-task-eval.lock"):
        manifest = _manifest(
            mode="synthetic_no_network",
            preregistration=preregistration,
            source_binding=source_binding,
            specs=specs,
            upstream=None,
            gateway_contract=None,
            python=None,
        )
        answer_contract.atomic_json_no_clobber(
            output_dir / "run_manifest.json", manifest
        )
        answer_contract.atomic_json_no_clobber(
            output_dir / "source_binding.json", source_binding
        )
        contexts = {
            (sample, checkpoint): contract.load_checkpoint_context(
                source_root,
                source_binding,
                sample=sample,
                checkpoint=checkpoint,
                formal=False,
            )
            for sample in [0]
            for checkpoint in contract.CHECKPOINTS
        }
        proxy_log = output_dir / "synthetic_proxy.jsonl"
        retrieval = _ProxyLoggingScriptedCompletionResource(
            _synthetic_scripts(specs, contexts),
            proxy_log=proxy_log,
            run_id=manifest["run_id"],
        )
        answer_client = controlled_answer.FakeAnswerClient(
            proxy_log=proxy_log,
            run_id=manifest["run_id"],
            response_text="<answer>synthetic checkpoint answer</answer>",
        )
        tokenizer = answer_contract.formal_token_counter()
        for spec in specs:
            execute_bound_question(
                output_dir=output_dir,
                run_id=manifest["run_id"],
                preregistration=preregistration,
                source_root=source_root,
                source_binding=source_binding,
                context=contexts[(int(spec["sample"]), int(spec["checkpoint"]))],
                spec=spec,
                completion_resource=retrieval,
                answer_client=answer_client,
                tokenizer=tokenizer,
                formal=False,
                proxy_log=proxy_log,
            )
        progress_path = output_dir / "progress.json"
        answer_contract.atomic_json_no_clobber(
            progress_path, _progress(output_dir, specs)
        )
        outputs = _publish_aggregates(output_dir, specs)
        complete = {
            "schema_version": COMPLETE_SCHEMA,
            "status": "complete",
            "run_id": manifest["run_id"],
            "mode": "synthetic_no_network",
            "question_checkpoint_artifacts": len(specs),
            "source_binding_sha256": source_binding["binding_sha256"],
            "preregistration_sha256": answer_contract.sha256_file(
                preregistration
            ),
            "progress_sha256": answer_contract.sha256_file(progress_path),
            "model_requests": 0,
            "network_requests": 0,
            "outputs": outputs,
        }
        answer_contract.atomic_json_no_clobber(
            output_dir / "complete.json", complete
        )
        from audit_r403_task_eval import audit_run  # noqa: PLC0415

        report = audit_run(output_dir, require_complete=True)
        answer_contract.atomic_json_no_clobber(output_dir / "audit.json", report)
        return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=contract.DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=contract.DEFAULT_FORMAL)
    parser.add_argument(
        "--preregistration", type=Path, default=contract.DEFAULT_PREREGISTRATION
    )
    parser.add_argument("--preflight-report", type=Path, default=contract.DEFAULT_PREFLIGHT)
    parser.add_argument("--gateway-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--freeze-preregistration", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--synthetic-sanity", action="store_true")
    parser.add_argument("--allow-model-requests", action="store_true")
    args = parser.parse_args()
    modes = sum(
        bool(value)
        for value in (
            args.freeze_preregistration,
            args.preflight,
            args.synthetic_sanity,
        )
    )
    if modes > 1:
        parser.error("select only one local-only mode")
    if args.freeze_preregistration:
        contract.freeze_preregistration(args.preregistration)
        print(args.preregistration)
        return 0
    if args.preflight:
        report = contract.run_preflight(
            args.preregistration,
            source_root=args.source_root,
            write_path=args.preflight_report,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["status"] == "ready" else 2
    if args.synthetic_sanity:
        if args.allow_model_requests:
            parser.error("synthetic mode forbids --allow-model-requests")
        report = run_synthetic(
            source_root=args.source_root,
            output_dir=args.output_dir,
            preregistration=args.preregistration,
        )
    else:
        if not args.allow_model_requests:
            parser.error("formal execution requires --allow-model-requests")
        if args.gateway_root is None:
            parser.error("formal execution requires --gateway-root")
        report = run_formal(
            source_root=args.source_root,
            output_dir=args.output_dir,
            preregistration=args.preregistration,
            preflight_path=args.preflight_report,
            gateway_root=args.gateway_root,
            python=args.python,
            allow_model_requests=True,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
