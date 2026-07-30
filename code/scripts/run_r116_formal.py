#!/usr/bin/env python3
"""Execute formal LoCoMo or LongMemEval-S R116 shared-answer controls.

Formal mode has no scope-limit flags.  It starts an exclusive controlled proxy,
uses the frozen GPT-5.5 retrieval and answer boundaries, and resumes only from
independently auditable complete per-question checkpoints.  Preflight and
synthetic modes never send model or network requests.
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
import r116_formal_contract as contract  # noqa: E402
import readonly_nativemem_control as readonly_control  # noqa: E402
import run_controlled_locomo_answers as controlled_answer  # noqa: E402
from src.evaluation.durable_model_ledger import read_ledger  # noqa: E402


RUN_SCHEMA = "nativemem.r116-formal-run.v1"
CHECKPOINT_SCHEMA = "nativemem.r116-formal-question-checkpoint.v1"
COMPLETE_SCHEMA = "nativemem.r116-formal-complete.v1"
DEFAULT_SYNTHETIC = (
    ROOT / "results/paper-experiments-20260714/r116-formal-synthetic-sanity"
)


class R116RunError(RuntimeError):
    pass


class _ProxyLoggingScriptedCompletionResource(
    readonly_control.ScriptedCompletionResource
):
    """Fake retrieval resource with append-only simulated proxy evidence."""

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
            raise R116RunError("synthetic retrieval lacks controlled call linkage")
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
            "event_id": f"synthetic-retrieval-{self.calls:04d}",
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
    if absolute.is_symlink():
        raise R116RunError("R116 output root must not be a symlink")
    return absolute


def _write_or_validate(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise R116RunError(f"existing artifact is unsafe: {path}")
        if answer_contract.read_json(path) != payload:
            raise R116RunError(f"existing immutable artifact differs: {path}")
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


def _generic_gold(spec: Mapping[str, Any]) -> tuple[list[str], bool]:
    if spec["benchmark"] == contract.LOCOMO:
        return list(spec["gold_source_ids"]), bool(spec["source_recall_eligible"])
    # LongMemEval R002 labels are session IDs.  The generic read-only control
    # intentionally accepts only turn-level Dn:m labels, so the wrapper audits
    # session recall in source_recall.json instead of inventing turn labels.
    return [], False


def _question_paths(output_dir: Path, spec: Mapping[str, Any]) -> dict[str, Path]:
    root = output_dir / "questions" / str(spec["artifact_id"])
    return {
        "root": root,
        "input": root / "input_binding.json",
        "attempt": root / "attempt-0001",
        "source_recall": root / "source_recall.json",
        "question_audit": root / "question_audit.json",
        "checkpoint": root / "checkpoint.json",
    }


def execute_bound_question(
    *,
    output_dir: Path,
    run_id: str,
    preregistration: Path,
    source_binding: Mapping[str, Any],
    spec: Mapping[str, Any],
    memory_root: Path,
    completion_resource: Any,
    answer_client: Any,
    tokenizer: Any,
    formal: bool,
    proxy_log: Path | None,
) -> dict[str, Any]:
    """Execute exactly one new question or validate its complete checkpoint."""

    paths = _question_paths(output_dir, spec)
    if paths["checkpoint"].exists():
        from audit_r116_formal import audit_question_checkpoint  # noqa: PLC0415

        return audit_question_checkpoint(
            output_dir=output_dir,
            spec=spec,
            source_binding=source_binding,
            formal=formal,
        )
    if paths["root"].exists() or paths["root"].is_symlink():
        raise R116RunError(
            f"incomplete question artifact exists; use a new output root: {paths['root']}"
        )
    turn_index, session_by_turn = contract.turn_index_and_session_map(spec)
    binding = contract.input_binding(
        preregistration=preregistration,
        source_binding=source_binding,
        spec=spec,
        memory_root=memory_root,
        turn_index=turn_index,
        session_by_turn=session_by_turn,
    )
    paths["root"].mkdir(parents=True)
    answer_contract.atomic_json_no_clobber(paths["input"], binding)
    generic_gold, generic_eligible = _generic_gold(spec)
    result = readonly_control.execute_question(
        artifact_dir=paths["attempt"],
        run_id=run_id,
        method=contract.METHOD,
        condition=contract.CONDITION,
        memory_root=memory_root,
        turn_index=turn_index,
        question_id=str(spec["question_id"]),
        question=str(spec["question"]),
        gold_source_ids=generic_gold,
        source_recall_eligible=generic_eligible,
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
    recall = contract.compute_source_recall(
        artifact_dir=paths["attempt"],
        spec=spec,
        session_by_turn=session_by_turn,
    )
    answer_contract.atomic_json_no_clobber(paths["source_recall"], recall)
    question_audit = generic_auditor.audit_question(
        artifact_dir=paths["attempt"],
        memory_root=memory_root,
        method=contract.METHOD,
        condition=contract.CONDITION,
        question_id=str(spec["question_id"]),
        question=str(spec["question"]),
        gold_source_ids=generic_gold,
        source_recall_eligible=generic_eligible,
        formal=formal,
    )
    answer_contract.atomic_json_no_clobber(
        paths["question_audit"], question_audit
    )
    response_ids = [
        record["response_id"]
        for record in read_ledger(paths["attempt"] / "retrieval_model_ledger.jsonl")
        if record.get("event") == "model_call_finished"
    ]
    response_ids.append(result["answer"]["response_id"])
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "complete",
        "benchmark": spec["benchmark"],
        "question_id": spec["question_id"],
        "artifact_id": spec["artifact_id"],
        "dataset_index": spec["dataset_index"],
        "input_binding": {
            "path": paths["input"].relative_to(output_dir).as_posix(),
            "sha256": answer_contract.sha256_file(paths["input"]),
        },
        "attempt": {
            "path": paths["attempt"].relative_to(output_dir).as_posix(),
            "result_sha256": answer_contract.sha256_file(
                paths["attempt"] / "result.json"
            ),
        },
        "source_recall": {
            "path": paths["source_recall"].relative_to(output_dir).as_posix(),
            "sha256": answer_contract.sha256_file(paths["source_recall"]),
        },
        "question_audit": {
            "path": paths["question_audit"].relative_to(output_dir).as_posix(),
            "sha256": answer_contract.sha256_file(paths["question_audit"]),
        },
        "actual_model": contract.MODEL,
        "response_ids": response_ids,
        "memory_sha256": result["memory"]["after"]["sha256"],
    }
    checkpoint["checkpoint_content_sha256"] = answer_contract.canonical_hash(
        checkpoint
    )
    answer_contract.atomic_json_no_clobber(paths["checkpoint"], checkpoint)
    from audit_r116_formal import audit_question_checkpoint  # noqa: PLC0415

    return audit_question_checkpoint(
        output_dir=output_dir,
        spec=spec,
        source_binding=source_binding,
        formal=formal,
    )


def _formal_manifest(
    *,
    benchmark: str,
    output_dir: Path,
    preregistration: Path,
    source_binding: Mapping[str, Any],
    gateway_contract: Mapping[str, Any],
    python: Path,
) -> dict[str, Any]:
    specs = contract.question_specs(benchmark)
    python = python.expanduser().resolve()
    if python.is_symlink() or not python.is_file():
        raise R116RunError("formal Python executable is unavailable")
    return {
        "schema_version": RUN_SCHEMA,
        "status": "answering",
        "mode": "formal",
        "run_id": (
            f"r116-{benchmark}-"
            f"{str(source_binding['binding_sha256'])[:16]}-"
            f"{answer_contract.sha256_file(preregistration)[:16]}"
        ),
        "benchmark": benchmark,
        "method": contract.METHOD,
        "condition": contract.CONDITION,
        "scope": contract.EXPECTED_SCOPE[benchmark],
        "question_inventory_sha256": answer_contract.canonical_hash(
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
        ),
        "preregistration": contract.preregistration_binding(preregistration),
        "source_binding_sha256": source_binding["binding_sha256"],
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
            "gateway_contract": dict(gateway_contract),
            "upstream": gateway_contract["origin"],
            "python": str(python),
            "python_sha256": answer_contract.sha256_file(python),
        },
    }


def _progress(output_dir: Path, *, benchmark: str, total: int) -> dict[str, Any]:
    completed = len(list((output_dir / "questions").glob("*/checkpoint.json")))
    return {
        "schema_version": RUN_SCHEMA,
        "status": "complete" if completed == total else "answering",
        "benchmark": benchmark,
        "completed": completed,
        "total": total,
        "model_requests_authorized": True,
    }


def _aggregate_records(
    output_dir: Path, specs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in specs:
        paths = _question_paths(output_dir, spec)
        result = answer_contract.read_json(paths["attempt"] / "result.json")
        recall = answer_contract.read_json(paths["source_recall"])
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
        records.append(record)
    return records


def _publish_aggregates(
    output_dir: Path, specs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    records = _aggregate_records(output_dir, specs)
    results_path = output_dir / "results.json"
    _write_or_validate(results_path, {"schema_version": RUN_SCHEMA, "records": records})
    hypotheses_path = output_dir / "hypotheses.jsonl"
    hypotheses_text = "".join(
        answer_contract.canonical_json(
            {
                "question_id": record["question_id"],
                "hypothesis": record["answer"],
            }
        )
        + "\n"
        for record in records
    )
    if hypotheses_path.exists():
        if hypotheses_path.is_symlink() or hypotheses_path.read_text(
            encoding="utf-8"
        ) != hypotheses_text:
            raise R116RunError("existing hypotheses artifact differs")
    else:
        _write_text_no_clobber(hypotheses_path, hypotheses_text)
    return {
        "results": {
            "path": results_path.relative_to(output_dir).as_posix(),
            "sha256": answer_contract.sha256_file(results_path),
        },
        "hypotheses": {
            "path": hypotheses_path.relative_to(output_dir).as_posix(),
            "sha256": answer_contract.sha256_file(hypotheses_path),
        },
    }


def run_formal(
    *,
    benchmark: str,
    output_dir: Path,
    preregistration: Path = contract.DEFAULT_PREREGISTRATION,
    preflight_path: Path = contract.DEFAULT_PREFLIGHT,
    gateway_root: Path | None = None,
    python: Path = Path(sys.executable),
    allow_model_requests: bool = False,
) -> dict[str, Any]:
    if not allow_model_requests:
        raise R116RunError("formal R116 requires --allow-model-requests")
    if gateway_root is None:
        raise R116RunError("formal R116 requires --gateway-root")
    if benchmark not in contract.BENCHMARKS:
        raise R116RunError(f"unknown benchmark: {benchmark}")
    gateway_root = gateway_root.expanduser().resolve()
    output_dir = _validate_output_root(output_dir)
    preflight = contract.run_preflight(
        preregistration, write_path=preflight_path
    )
    source_binding = contract.require_ready_binding(preflight, benchmark)
    specs = contract.question_specs(benchmark)
    source_root = ROOT / str(source_binding["source_root"])
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
        raise R116RunError(str(exc)) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    with provider_lock, answer_contract.FileLock(output_dir / ".r116-formal.lock"):
        complete_path = output_dir / "complete.json"
        if complete_path.exists():
            from audit_r116_formal import audit_run  # noqa: PLC0415

            return audit_run(output_dir, require_complete=True)
        manifest = _formal_manifest(
            benchmark=benchmark,
            output_dir=output_dir,
            preregistration=preregistration,
            source_binding=source_binding,
            gateway_contract=gateway_contract,
            python=python,
        )
        _write_or_validate(output_dir / "run_manifest.json", manifest)
        _write_or_validate(output_dir / "source_binding.json", source_binding)
        progress_path = output_dir / "progress.json"
        answer_contract.atomic_json_replace(
            progress_path,
            _progress(output_dir, benchmark=benchmark, total=len(specs)),
        )
        contract.verify_source_unchanged(
            source_binding=source_binding, benchmark=benchmark
        )
        controlled_answer._recover_interrupted_proxies(  # noqa: SLF001
            output_dir, manifest["run_id"]
        )
        tokenizer = answer_contract.formal_token_counter()
        pending: list[tuple[dict[str, Any], Path]] = []
        for spec in specs:
            memory_root = contract.source_memory_path(benchmark, source_root, spec)
            if _question_paths(output_dir, spec)["checkpoint"].exists():
                execute_bound_question(
                    output_dir=output_dir,
                    run_id=manifest["run_id"],
                    preregistration=preregistration,
                    source_binding=source_binding,
                    spec=spec,
                    memory_root=memory_root,
                    completion_resource=None,
                    answer_client=None,
                    tokenizer=tokenizer,
                    formal=True,
                    proxy_log=None,
                )
            else:
                pending.append((spec, memory_root))
        process = process_log = proxy_record = None
        try:
            if pending:
                process, proxy_record, process_log = controlled_answer._start_proxy(  # noqa: SLF001
                    output_dir=output_dir,
                    python=python.expanduser().resolve(),
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
                for spec, memory_root in pending:
                    execute_bound_question(
                        output_dir=output_dir,
                        run_id=manifest["run_id"],
                        preregistration=preregistration,
                        source_binding=source_binding,
                        spec=spec,
                        memory_root=memory_root,
                        completion_resource=completion_resource,
                        answer_client=answer_client,
                        tokenizer=tokenizer,
                        formal=True,
                        proxy_log=proxy_log,
                    )
                    answer_contract.atomic_json_replace(
                        progress_path,
                        _progress(
                            output_dir, benchmark=benchmark, total=len(specs)
                        ),
                    )
        finally:
            if process is not None and proxy_record is not None and process_log is not None:
                controlled_answer._stop_proxy(  # noqa: SLF001
                    process, proxy_record, process_log, output_dir
                )
        contract.verify_source_unchanged(
            source_binding=source_binding, benchmark=benchmark
        )
        progress = _progress(output_dir, benchmark=benchmark, total=len(specs))
        if progress["completed"] != len(specs):
            raise R116RunError("formal R116 scope is incomplete")
        answer_contract.atomic_json_replace(progress_path, progress)
        outputs = _publish_aggregates(output_dir, specs)
        complete = {
            "schema_version": COMPLETE_SCHEMA,
            "status": "complete",
            "run_id": manifest["run_id"],
            "benchmark": benchmark,
            "questions": len(specs),
            "source_binding_sha256": source_binding["binding_sha256"],
            "preregistration_sha256": answer_contract.sha256_file(preregistration),
            "progress_sha256": answer_contract.sha256_file(progress_path),
            "outputs": outputs,
        }
        answer_contract.atomic_json_no_clobber(complete_path, complete)
        from audit_r116_formal import audit_run  # noqa: PLC0415

        report = audit_run(output_dir, require_complete=True)
        answer_contract.atomic_json_replace(output_dir / "audit.json", report)
        return report


def _synthetic_source_binding(
    *, benchmark: str, memory: Path, preregistration: Path
) -> dict[str, Any]:
    entry_key = "sample_index" if benchmark == contract.LOCOMO else "dataset_index"
    entry = {
        entry_key: 0,
        "question_id": (
            None if benchmark == contract.LOCOMO else contract.question_specs(benchmark)[0]["question_id"]
        ),
        "memory": readonly_control.memory_descriptor(memory),
        "synthetic": True,
    }
    binding = {
        "benchmark": benchmark,
        "source_root": memory.parent.relative_to(ROOT).as_posix()
        if memory.is_relative_to(ROOT)
        else str(memory.parent),
        "run_manifest": {"synthetic": True},
        "independent_audit": {"synthetic": True},
        "inventory": [entry],
        "inventory_sha256": answer_contract.canonical_hash([entry]),
        "questions": 1,
        "preregistration_sha256": answer_contract.sha256_file(preregistration),
    }
    binding["binding_sha256"] = answer_contract.canonical_hash(binding)
    return binding


def run_synthetic(
    output_dir: Path = DEFAULT_SYNTHETIC,
    *,
    preregistration: Path = contract.DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    """Exercise both benchmark modes with fake local model resources."""

    contract.validate_preregistration(preregistration)
    output_dir = _validate_output_root(output_dir)
    if (output_dir / "complete.json").exists():
        from audit_r116_formal import audit_run  # noqa: PLC0415

        return audit_run(output_dir, require_complete=True)
    if output_dir.exists():
        raise R116RunError("existing synthetic output is incomplete; use a new root")
    output_dir.mkdir(parents=True)
    with answer_contract.FileLock(output_dir / ".r116-formal.lock"):
        specs = {
            benchmark: contract.question_specs(benchmark)[0]
            for benchmark in contract.BENCHMARKS
        }
        locomo_source_id = str(specs[contract.LOCOMO]["gold_source_ids"][0])
        lme_spec = specs[contract.LONGMEMEVAL]
        _turns, lme_session_map = contract.turn_index_and_session_map(lme_spec)
        lme_source_id = next(
            turn
            for turn, session in lme_session_map.items()
            if session == lme_spec["gold_source_ids"][0]
        )
        memories: dict[str, Path] = {}
        for benchmark, source_id in (
            (contract.LOCOMO, locomo_source_id),
            (contract.LONGMEMEVAL, lme_source_id),
        ):
            memory = output_dir / "synthetic_sources" / benchmark / "memory"
            _write_text_no_clobber(
                memory / "topics/fact.md",
                f"Synthetic evidence for contract validation [{source_id}].\n",
            )
            _write_text_no_clobber(
                memory / "timeline/2025/01/01.md",
                f"Synthetic timeline evidence [{source_id}].\n",
            )
            memories[benchmark] = memory
        bindings = {
            benchmark: _synthetic_source_binding(
                benchmark=benchmark,
                memory=memories[benchmark],
                preregistration=preregistration,
            )
            for benchmark in contract.BENCHMARKS
        }
        manifest = {
            "schema_version": RUN_SCHEMA,
            "status": "answering",
            "mode": "synthetic_no_network",
            "run_id": "r116-formal-synthetic-v1",
            "benchmark": "both",
            "method": contract.METHOD,
            "condition": contract.CONDITION,
            "scope": {"locomo": 1, "longmemeval-s": 1},
            "preregistration": contract.preregistration_binding(preregistration),
            "source_bindings": {
                key: value["binding_sha256"] for key, value in bindings.items()
            },
            "config": {
                "model": contract.MODEL,
                "budget_tokens": contract.BUDGET_TOKENS,
                "max_rounds": contract.MAX_ROUNDS,
                "model_context_limit_tokens": contract.MODEL_CONTEXT_LIMIT_TOKENS,
                "answer_completion_reservation_tokens": (
                    contract.ANSWER_COMPLETION_RESERVATION_TOKENS
                ),
                "tokenizer": answer_contract.formal_token_counter().identity,
                "model_requests": 0,
                "network_requests": 0,
            },
        }
        answer_contract.atomic_json_no_clobber(
            output_dir / "run_manifest.json", manifest
        )
        for benchmark, binding in bindings.items():
            answer_contract.atomic_json_no_clobber(
                output_dir / f"source_binding.{benchmark}.json", binding
            )
        proxy_log = output_dir / "synthetic_proxy.jsonl"
        tokenizer = answer_contract.formal_token_counter()
        answer_client = controlled_answer.FakeAnswerClient(
            proxy_log=proxy_log,
            run_id=manifest["run_id"],
            response_text="<answer>synthetic</answer>",
        )
        retrieval_scripts: list[list[dict[str, Any]]] = []
        for source_id in (locomo_source_id, lme_source_id):
            retrieval_scripts.extend(
                [
                    [
                        {
                            "name": "read_memory_file",
                            "arguments": {"path": "topics/fact.md"},
                        }
                    ],
                    [
                        {
                            "name": "resolve_sources",
                            "arguments": {"source_ids": [source_id]},
                        }
                    ],
                    [],
                ]
            )
        retrieval = _ProxyLoggingScriptedCompletionResource(
            retrieval_scripts,
            proxy_log=proxy_log,
            run_id=manifest["run_id"],
        )
        for benchmark in contract.BENCHMARKS:
            spec = specs[benchmark]
            execute_bound_question(
                output_dir=output_dir,
                run_id=manifest["run_id"],
                preregistration=preregistration,
                source_binding=bindings[benchmark],
                spec=spec,
                memory_root=memories[benchmark],
                completion_resource=retrieval,
                answer_client=answer_client,
                tokenizer=tokenizer,
                formal=False,
                proxy_log=proxy_log,
            )
        outputs = _publish_aggregates(
            output_dir,
            [specs[contract.LOCOMO], specs[contract.LONGMEMEVAL]],
        )
        complete = {
            "schema_version": COMPLETE_SCHEMA,
            "status": "complete",
            "run_id": manifest["run_id"],
            "benchmark": "both",
            "questions": 2,
            "model_requests": 0,
            "network_requests": 0,
            "outputs": outputs,
        }
        answer_contract.atomic_json_no_clobber(
            output_dir / "complete.json", complete
        )
        from audit_r116_formal import audit_run  # noqa: PLC0415

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
    parser.add_argument("--benchmark", choices=contract.BENCHMARKS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=contract.DEFAULT_PREREGISTRATION,
    )
    parser.add_argument(
        "--preflight-report", type=Path, default=contract.DEFAULT_PREFLIGHT
    )
    parser.add_argument("--allow-model-requests", action="store_true")
    parser.add_argument("--gateway-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args(argv)
    if args.formal:
        if not args.allow_model_requests:
            parser.error("formal R116 requires explicit --allow-model-requests")
        if args.benchmark is None or args.output_dir is None:
            parser.error("formal R116 requires --benchmark and --output-dir")
        if args.gateway_root is None:
            parser.error("formal R116 requires --gateway-root")
    else:
        if args.allow_model_requests:
            parser.error("non-formal R116 modes forbid --allow-model-requests")
        if args.benchmark is not None:
            parser.error("--benchmark is valid only with --formal")
        if args.synthetic_sanity and args.output_dir is None:
            args.output_dir = DEFAULT_SYNTHETIC
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preregister:
        result = contract.freeze_preregistration(args.preregistration)
    elif args.preflight:
        result = contract.run_preflight(
            args.preregistration, write_path=args.preflight_report
        )
    elif args.synthetic_sanity:
        result = run_synthetic(
            args.output_dir, preregistration=args.preregistration
        )
    else:
        result = run_formal(
            benchmark=args.benchmark,
            output_dir=args.output_dir,
            preregistration=args.preregistration,
            preflight_path=args.preflight_report,
            gateway_root=args.gateway_root,
            python=args.python,
            allow_model_requests=args.allow_model_requests,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        R116RunError,
        contract.R116FormalError,
        readonly_control.ReadOnlyControlError,
        generic_auditor.ReadOnlyAuditError,
        answer_contract.ControlledAnswerError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
