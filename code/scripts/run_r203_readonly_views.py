#!/usr/bin/env python3
"""R203 read-only NativeMem view/source conditions.

The synthetic mode materializes four byte-identical condition copies and
executes dual+source, topic+source, timeline+source, and dual-no-source without
network access. Formal execution remains disabled until the frozen all-ten
NativeMem memory artifact and its source mapping pass their upstream audits.
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
import r203_formal_contract as formal_contract  # noqa: E402
import readonly_nativemem_control as control  # noqa: E402
import run_controlled_locomo_answers as controlled_answer  # noqa: E402
from src.evaluation.durable_model_ledger import (  # noqa: E402
    DurableLedgerError,
    read_ledger,
)


RUN_SCHEMA = "nativemem.r203-readonly-views-run.v1"
FORMAL_RUN_SCHEMA = "nativemem.r203-formal-run.v2"
FORMAL_CHECKPOINT_SCHEMA = "nativemem.r203-formal-question-checkpoint.v1"
FORMAL_COMPLETE_SCHEMA = "nativemem.r203-formal-complete.v1"
METHOD = "r203_readonly_views"
DISPLAY_NAMES = {
    "dual_source": "dual+source",
    "topic_source": "topic+source",
    "timeline_source": "timeline+source",
    "dual_no_source": "dual-no-source",
}


class R203Error(RuntimeError):
    pass


def _source_hashes() -> dict[str, str]:
    paths = {
        "scripts/r203_formal_contract.py": Path(formal_contract.__file__).resolve(),
        "scripts/readonly_nativemem_control.py": Path(control.__file__).resolve(),
        "scripts/audit_readonly_nativemem_control.py": Path(
            generic_auditor.__file__
        ).resolve(),
        "scripts/run_r203_readonly_views.py": Path(__file__).resolve(),
        "scripts/audit_r203_readonly_views.py": (
            ROOT / "scripts/audit_r203_readonly_views.py"
        ),
        "scripts/controlled_locomo_answer_contract.py": Path(
            answer_contract.__file__
        ).resolve(),
        "scripts/run_controlled_locomo_answers.py": Path(
            controlled_answer.__file__
        ).resolve(),
        "src/evaluation/durable_model_ledger.py": (
            ROOT / "src/evaluation/durable_model_ledger.py"
        ),
        "src/evaluation/visible_token_budget.py": (
            ROOT / "src/evaluation/visible_token_budget.py"
        ),
        "src/evaluation/visible_token_audit.py": (
            ROOT / "src/evaluation/visible_token_audit.py"
        ),
    }
    return {name: answer_contract.sha256_file(path) for name, path in paths.items()}


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


def _fixture(output_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source = output_dir / "memory_source"
    _write_text_no_clobber(
        source / "topics/travel.md",
        "## Travel\n[2025-02-03] Mina visited Kyoto [D1:1].\n",
    )
    _write_text_no_clobber(
        source / "timeline/2025/02/03.md",
        "[2025-02-03] Mina visited Kyoto · [D1:1] → topics/travel.md\n",
    )
    conversation = {
        "speaker_a": "Mina",
        "speaker_b": "Noah",
        "session_1_date_time": "3 February 2025",
        "session_1": [
            {
                "speaker": "Mina",
                "dia_id": "D1:1",
                "text": "I visited Kyoto.",
            },
            {
                "speaker": "Noah",
                "dia_id": "D1:2",
                "text": "How was the trip?",
            },
        ],
    }
    question = {
        "question_id": "s0_q0",
        "question": "Which city did Mina visit?",
        "gold_source_ids": ["D1:1"],
        "source_recall_eligible": True,
    }
    return source, conversation, question


def _retrieval_script(condition: str) -> list[list[dict[str, Any]]]:
    path = (
        "timeline/2025/02/03.md"
        if condition == "timeline_source"
        else "topics/travel.md"
    )
    script: list[list[dict[str, Any]]] = [
        [{"name": "read_memory_file", "arguments": {"path": path}}]
    ]
    if control.CONDITION_SOURCE_ENABLED[condition]:
        script.append(
            [
                {
                    "name": "resolve_sources",
                    "arguments": {"source_ids": ["D1:1"]},
                }
            ]
        )
    script.append([])
    return script


def _existing_report(output_dir: Path) -> dict[str, Any] | None:
    if not (output_dir / "complete.json").exists():
        return None
    from audit_r203_readonly_views import audit_run  # noqa: PLC0415

    return audit_run(output_dir)


def run_synthetic(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().absolute()
    answer_contract.reject_symlink_components(output_dir)
    if output_dir.is_symlink():
        raise R203Error("output directory must not be a symlink")
    if output_dir.exists():
        report = _existing_report(output_dir)
        if report is None:
            raise R203Error("existing R203 output is incomplete; use a new root")
        return report
    output_dir.mkdir(parents=True)
    with answer_contract.FileLock(output_dir / ".r203.lock"):
        source, conversation, question = _fixture(output_dir)
        source_descriptor = control.memory_descriptor(source)
        run_id = f"r203-synthetic-{source_descriptor['sha256'][:16]}"
        tokenizer = answer_contract.formal_token_counter()
        stage_provenance = control.build_zero_maintenance_stage_provenance(
            memory_root=source,
            question_id=question["question_id"],
            gold_source_ids=question["gold_source_ids"],
            mapping_complete=question["source_recall_eligible"],
        )
        stage_provenance_path = output_dir / "stage_evidence_provenance.json"
        answer_contract.atomic_json_no_clobber(
            stage_provenance_path, stage_provenance
        )
        copies: dict[str, dict[str, Any]] = {}
        for condition in control.CONDITIONS:
            destination = output_dir / "conditions" / condition / "memory"
            copies[condition] = control.copy_memory_tree(source, destination)
        copy_hashes = {value["sha256"] for value in copies.values()}
        if copy_hashes != {source_descriptor["sha256"]}:
            raise R203Error("R203 condition copies are not byte-identical")
        manifest = {
            "schema_version": RUN_SCHEMA,
            "status": "running",
            "mode": "synthetic_no_network",
            "run_id": run_id,
            "created_at": answer_contract.utc_now(),
            "method": METHOD,
            "conditions": [
                {"id": condition, "display_name": DISPLAY_NAMES[condition]}
                for condition in control.CONDITIONS
            ],
            "scope": {
                "samples": [0],
                "questions_per_condition": 1,
                "formal_all_ten_required": True,
                "formal": False,
            },
            "config": {
                "model": control.EXPECTED_MODEL,
                "answerer": control.EXPECTED_MODEL,
                "budget_policy": "hard_cap",
                "budget_tokens": control.EXPECTED_BUDGET,
                "tokenizer": tokenizer.identity,
                "max_rounds": 12,
                "retrieval_read_only": True,
                "network_requests": 0,
            },
            "source_hashes": _source_hashes(),
            "memory_source": {
                "path": str(source.relative_to(output_dir)),
                "descriptor": source_descriptor,
            },
            "condition_copies": {
                condition: {
                    "path": str(
                        (output_dir / "conditions" / condition / "memory").relative_to(
                            output_dir
                        )
                    ),
                    "descriptor": descriptor,
                }
                for condition, descriptor in copies.items()
            },
            "question": question,
            "conversation": conversation,
            "stage_evidence_provenance": {
                "path": stage_provenance_path.name,
                "sha256": answer_contract.sha256_file(stage_provenance_path),
                "record_sha256": answer_contract.canonical_hash(stage_provenance),
                "formal_requirement": (
                    "bind R002 mapping plus the independently audited final "
                    "NativeMem tree as canonical R203 input, with zero R203 "
                    "maintenance and no NativeMem builder pre-maintenance claim"
                ),
            },
            "formal_blocker": (
                "new completed and independently audited all-ten GPT-5.5 Flex "
                "NativeMem source is not yet available"
            ),
        }
        answer_contract.atomic_json_no_clobber(output_dir / "run_manifest.json", manifest)
        reports: dict[str, dict[str, Any]] = {}
        for condition in control.CONDITIONS:
            condition_root = output_dir / "conditions" / condition
            proxy_log = condition_root / "synthetic_proxy.jsonl"
            answer_client = controlled_answer.FakeAnswerClient(
                proxy_log=proxy_log,
                run_id=run_id,
                response_text="<answer>Kyoto</answer>",
            )
            retrieval = control.ScriptedCompletionResource(
                _retrieval_script(condition)
            )
            artifact_dir = condition_root / "questions/s0_q0/attempt-0001"
            result = control.execute_question(
                artifact_dir=artifact_dir,
                run_id=run_id,
                method=METHOD,
                condition=condition,
                memory_root=condition_root / "memory",
                turn_index=control.build_turn_index(conversation),
                question_id=question["question_id"],
                question=question["question"],
                gold_source_ids=question["gold_source_ids"],
                source_recall_eligible=question["source_recall_eligible"],
                completion_resource=retrieval,
                answer_client=answer_client,
                tokenizer=tokenizer,
                formal=False,
                proxy_log=proxy_log,
                stage_provenance=stage_provenance,
            )
            checkpoint = {
                "schema_version": RUN_SCHEMA,
                "status": "complete",
                "condition": condition,
                "question_id": question["question_id"],
                "artifact_dir": str(artifact_dir.relative_to(output_dir)),
                "result_sha256": answer_contract.sha256_file(
                    artifact_dir / "result.json"
                ),
            }
            answer_contract.atomic_json_no_clobber(
                condition_root / "completed/s0_q0.json", checkpoint
            )
            report = generic_auditor.audit_question(
                artifact_dir=artifact_dir,
                memory_root=condition_root / "memory",
                method=METHOD,
                condition=condition,
                question_id=question["question_id"],
                question=question["question"],
                gold_source_ids=question["gold_source_ids"],
                source_recall_eligible=question["source_recall_eligible"],
                formal=False,
                stage_provenance=stage_provenance,
                turn_index=control.build_turn_index(conversation),
            )
            if result["answer"]["text"] != "Kyoto":
                raise R203Error("synthetic fixed answer differs")
            reports[condition] = report
        complete = {
            "schema_version": RUN_SCHEMA,
            "status": "complete",
            "run_id": run_id,
            "completed_at": answer_contract.utc_now(),
            "network_requests": 0,
            "condition_count": 4,
            "questions_per_condition": 1,
            "memory_source_sha256": source_descriptor["sha256"],
            "stage_evidence_provenance_sha256": answer_contract.sha256_file(
                stage_provenance_path
            ),
            "condition_audits": reports,
        }
        answer_contract.atomic_json_no_clobber(output_dir / "complete.json", complete)
        from audit_r203_readonly_views import audit_run  # noqa: PLC0415

        final_report = audit_run(output_dir)
        answer_contract.atomic_json_replace(output_dir / "audit.json", final_report)
        return final_report


def _validate_output_root(path: Path) -> Path:
    absolute = path.expanduser().absolute()
    answer_contract.reject_symlink_components(absolute)
    if absolute.is_symlink():
        raise R203Error("R203 output root must not be a symlink")
    return absolute


def _write_or_validate(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise R203Error(f"existing immutable artifact is unsafe: {path}")
        if answer_contract.read_json(path) != payload:
            raise R203Error(f"existing immutable artifact differs: {path}")
        return
    answer_contract.atomic_json_no_clobber(path, payload)


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
        "stage": root / "stage_provenance.json",
        "attempt": root / "attempt-0001",
        "question_audit": root / "question_audit.json",
        "checkpoint": root / "checkpoint.json",
    }


def _response_ids(attempt: Path, result: Mapping[str, Any]) -> list[str]:
    ids = [
        str(record["response_id"])
        for record in read_ledger(attempt / "retrieval_model_ledger.jsonl")
        if record.get("event") == "model_call_finished"
    ]
    ids.append(str(result["answer"]["response_id"]))
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise R203Error("question response IDs are empty or duplicated")
    return ids


def _copy_condition_memories(
    *, output_dir: Path, source_binding: Mapping[str, Any]
) -> dict[str, Any]:
    path = output_dir / "condition_copy_inventory.json"
    if path.exists():
        record = answer_contract.read_json(path)
        if not isinstance(record, dict):
            raise R203Error("condition-copy inventory is invalid")
        formal_contract.validate_content_hash(record, "inventory_content_sha256")
        rows = record.get("rows")
        if not isinstance(rows, list):
            raise R203Error("condition-copy rows are absent")
        formal_contract.validate_copy_inventory(
            output_dir=output_dir,
            source_binding=source_binding,
            inventory=rows,
        )
        return record
    if (output_dir / "conditions").exists():
        raise R203Error(
            "incomplete condition-copy initialization exists; use a new output root"
        )
    formal_contract.verify_source_unchanged(source_binding)
    for sample in formal_contract.SAMPLES:
        source = formal_contract.source_memory_path(source_binding, sample)
        for condition in formal_contract.CONDITIONS:
            destination = formal_contract.condition_memory_path(
                output_dir, condition, sample
            )
            control.copy_memory_tree(source, destination)
    rows = formal_contract.copy_inventory(output_dir)
    formal_contract.validate_copy_inventory(
        output_dir=output_dir,
        source_binding=source_binding,
        inventory=rows,
    )
    record = {
        "schema_version": "nativemem.r203-condition-copy-inventory.v1",
        "initialization": "all_40_copies_completed_before_formal_questions",
        "samples": formal_contract.SAMPLES,
        "conditions": list(formal_contract.CONDITIONS),
        "rows": rows,
        "byte_identical_to_sample_source": True,
    }
    record["inventory_content_sha256"] = answer_contract.canonical_hash(record)
    answer_contract.atomic_json_no_clobber(path, record)
    return record


def _formal_manifest(
    *,
    preregistration: Path,
    preflight: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    copy_record: Mapping[str, Any],
    gateway_contract: Mapping[str, Any],
    python: Path,
) -> dict[str, Any]:
    specs = formal_contract.question_specs()
    python = python.expanduser().resolve()
    if python.is_symlink() or not python.is_file():
        raise R203Error("formal Python executable is unavailable")
    return {
        "schema_version": FORMAL_RUN_SCHEMA,
        "status": "answering",
        "mode": "formal",
        "run_id": (
            "r203-formal-"
            f"{str(source_binding['binding_sha256'])[:16]}-"
            f"{answer_contract.sha256_file(preregistration)[:16]}"
        ),
        "method": METHOD,
        "conditions": list(formal_contract.CONDITIONS),
        "scope": {
            "samples": formal_contract.SAMPLES,
            "primary_questions_per_condition": formal_contract.PRIMARY_QUESTIONS,
            "condition_artifacts": formal_contract.TOTAL_ARTIFACTS,
            "source_recall_denominator_per_condition": (
                formal_contract.SOURCE_RECALL_DENOMINATOR
            ),
            "category_counts": {
                str(key): value
                for key, value in formal_contract.CATEGORY_COUNTS.items()
            },
        },
        "question_inventory_sha256": formal_contract.question_inventory_sha256(
            specs
        ),
        "preregistration": formal_contract.preregistration_binding(
            preregistration
        ),
        "preflight_content_sha256": preflight["preflight_content_sha256"],
        "source_binding_sha256": source_binding["binding_sha256"],
        "condition_copy_inventory_sha256": copy_record[
            "inventory_content_sha256"
        ],
        "config": {
            "retrieval_model": formal_contract.MODEL,
            "answer_model": formal_contract.MODEL,
            "budget_policy": "hard_cap",
            "budget_tokens": formal_contract.BUDGET_TOKENS,
            "max_rounds": formal_contract.MAX_ROUNDS,
            "model_context_limit_tokens": (
                formal_contract.MODEL_CONTEXT_LIMIT_TOKENS
            ),
            "answer_completion_reservation_tokens": (
                formal_contract.ANSWER_COMPLETION_RESERVATION_TOKENS
            ),
            "answer_max_tokens": formal_contract.ANSWER_MAX_TOKENS,
            "answer_retries": formal_contract.ANSWER_RETRIES,
            "tokenizer": answer_contract.formal_token_counter().identity,
            "retrieval_read_only": True,
            "gateway_contract": dict(gateway_contract),
            "upstream": gateway_contract["origin"],
            "python": str(python),
            "python_sha256": answer_contract.sha256_file(python),
        },
        "code_source_hashes": formal_contract.source_hashes(),
        "provenance_claim_boundary": source_binding[
            "provenance_claim_boundary"
        ],
    }


def _progress(output_dir: Path) -> dict[str, Any]:
    completed = len(
        list(output_dir.glob("conditions/*/questions/*/checkpoint.json"))
    )
    return {
        "schema_version": FORMAL_RUN_SCHEMA,
        "status": (
            "complete"
            if completed == formal_contract.TOTAL_ARTIFACTS
            else "answering"
        ),
        "completed": completed,
        "total": formal_contract.TOTAL_ARTIFACTS,
        "model_requests_authorized": True,
    }


def _assert_closed_proxy_inventory(output_dir: Path, run_id: str) -> None:
    proxy_root = output_dir / "proxy"
    checkpoints = sorted(
        output_dir.glob("conditions/*/questions/*/checkpoint.json")
    )
    if not proxy_root.exists():
        if checkpoints:
            raise R203Error("formal checkpoints exist without a proxy inventory")
        return
    controlled_answer._recover_interrupted_proxies(output_dir, run_id)  # noqa: SLF001
    live_records: list[dict[str, Any]] = []
    for directory in sorted(proxy_root.glob("invocation-*")):
        manifest_path = directory / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise R203Error("orphan proxy invocation remains after recovery")
        manifest = answer_contract.read_json(manifest_path)
        if (
            not isinstance(manifest, dict)
            or manifest.get("run_id") != run_id
            or manifest.get("provider_window_error") is not None
            or not isinstance(manifest.get("provider_window"), dict)
        ):
            raise R203Error("proxy invocation is not safely closed")
        log = output_dir / str(manifest.get("log", ""))
        if log.is_symlink() or not log.is_file():
            raise R203Error("closed proxy log is absent or unsafe")
        records = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        try:
            controlled_answer.flex_evidence.audit_window(
                manifest["provider_window"], consumer_records=records
            )
        except controlled_answer.flex_evidence.EvidenceError as exc:
            raise R203Error(f"closed proxy provider window differs: {exc}") from exc
        live_records.extend(records)
    from audit_r203_readonly_views import _result_proxy_events  # noqa: PLC0415

    expected_records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        question_root = checkpoint.parent
        result = answer_contract.read_json(
            question_root / "attempt-0001/result.json"
        )
        events, _response_ids = _result_proxy_events(
            attempt=question_root / "attempt-0001", result=result
        )
        expected_records.extend(events)
    live_by_id = {
        str(record.get("event_id")): record for record in live_records
    }
    expected_by_id = {
        str(record.get("event_id")): record for record in expected_records
    }
    if (
        len(live_by_id) != len(live_records)
        or len(expected_by_id) != len(expected_records)
        or live_by_id != expected_by_id
    ):
        raise R203Error(
            "closed proxy inventory contains an orphan, missing, or changed event"
        )


def _audit_existing_formal_checkpoints(
    *,
    output_dir: Path,
    specs: Sequence[Mapping[str, Any]],
    source_binding: Mapping[str, Any],
    copy_record: Mapping[str, Any],
) -> set[tuple[str, str]]:
    """Audit every existing question before allowing any additional request."""

    by_artifact = {str(spec["artifact_id"]): spec for spec in specs}
    completed: set[tuple[str, str]] = set()
    from audit_r203_readonly_views import (  # noqa: PLC0415
        audit_formal_question_checkpoint,
    )

    for condition in formal_contract.CONDITIONS:
        question_root = output_dir / "conditions" / condition / "questions"
        if not question_root.exists():
            continue
        if question_root.is_symlink() or not question_root.is_dir():
            raise R203Error("formal question root is unsafe")
        for root in sorted(question_root.iterdir()):
            if root.is_symlink() or not root.is_dir():
                raise R203Error("formal question artifact is unsafe")
            spec = by_artifact.get(root.name)
            if spec is None:
                raise R203Error(f"unknown formal question artifact: {root}")
            if not (root / "checkpoint.json").is_file():
                raise R203Error(
                    "incomplete formal question artifact exists; use a new "
                    f"output root: {root}"
                )
            audit_formal_question_checkpoint(
                output_dir=output_dir,
                condition=condition,
                spec=spec,
                source_binding=source_binding,
                copy_record=copy_record,
            )
            completed.add((condition, root.name))
    return completed


def execute_formal_question(
    *,
    output_dir: Path,
    run_id: str,
    preregistration: Path,
    source_binding: Mapping[str, Any],
    copy_record: Mapping[str, Any],
    condition: str,
    spec: Mapping[str, Any],
    completion_resource: Any,
    answer_client: Any,
    tokenizer: Any,
    proxy_log: Path | None,
) -> dict[str, Any]:
    paths = _question_paths(output_dir, condition, spec)
    memory_root = formal_contract.condition_memory_path(
        output_dir, condition, int(spec["dataset_index"])
    )
    if paths["checkpoint"].exists():
        from audit_r203_readonly_views import (  # noqa: PLC0415
            audit_formal_question_checkpoint,
        )

        return audit_formal_question_checkpoint(
            output_dir=output_dir,
            condition=condition,
            spec=spec,
            source_binding=source_binding,
            copy_record=copy_record,
        )
    if paths["root"].exists() or paths["root"].is_symlink():
        raise R203Error(
            "incomplete formal question artifact exists; use a new output root: "
            f"{paths['root']}"
        )
    _conversation, turn_index = formal_contract.conversation_and_turn_index(spec)
    stage = formal_contract.build_stage_provenance(
        memory_root=memory_root,
        spec=spec,
        source_binding=source_binding,
    )
    binding = formal_contract.input_binding(
        preregistration=preregistration,
        source_binding=source_binding,
        copy_inventory_sha256=copy_record["inventory_content_sha256"],
        condition=condition,
        spec=spec,
        memory_root=memory_root,
        turn_index=turn_index,
        stage_provenance=stage,
    )
    paths["root"].mkdir(parents=True)
    answer_contract.atomic_json_no_clobber(paths["stage"], stage)
    answer_contract.atomic_json_no_clobber(paths["input"], binding)
    result = control.execute_question(
        artifact_dir=paths["attempt"],
        run_id=run_id,
        method=METHOD,
        condition=condition,
        memory_root=memory_root,
        turn_index=turn_index,
        question_id=str(spec["question_id"]),
        question=str(spec["question"]),
        gold_source_ids=list(spec["gold_source_ids"]),
        source_recall_eligible=bool(spec["source_recall_eligible"]),
        completion_resource=completion_resource,
        answer_client=answer_client,
        tokenizer=tokenizer,
        formal=True,
        proxy_log=proxy_log,
        budget_tokens=formal_contract.BUDGET_TOKENS,
        max_rounds=formal_contract.MAX_ROUNDS,
        model_context_limit_tokens=formal_contract.MODEL_CONTEXT_LIMIT_TOKENS,
        answer_completion_reservation_tokens=(
            formal_contract.ANSWER_COMPLETION_RESERVATION_TOKENS
        ),
        stage_provenance=stage,
    )
    question_audit = generic_auditor.audit_question(
        artifact_dir=paths["attempt"],
        memory_root=memory_root,
        method=METHOD,
        condition=condition,
        question_id=str(spec["question_id"]),
        question=str(spec["question"]),
        gold_source_ids=list(spec["gold_source_ids"]),
        source_recall_eligible=bool(spec["source_recall_eligible"]),
        formal=True,
        stage_provenance=stage,
        turn_index=turn_index,
    )
    answer_contract.atomic_json_no_clobber(
        paths["question_audit"], question_audit
    )
    checkpoint = {
        "schema_version": FORMAL_CHECKPOINT_SCHEMA,
        "status": "complete",
        "condition": condition,
        "question_id": spec["question_id"],
        "artifact_id": spec["artifact_id"],
        "dataset_index": spec["dataset_index"],
        "question_index": spec["question_index"],
        "category": spec["category"],
        "input_binding": {
            "path": paths["input"].relative_to(output_dir).as_posix(),
            "sha256": answer_contract.sha256_file(paths["input"]),
        },
        "stage_provenance": {
            "path": paths["stage"].relative_to(output_dir).as_posix(),
            "sha256": answer_contract.sha256_file(paths["stage"]),
        },
        "attempt": {
            "path": paths["attempt"].relative_to(output_dir).as_posix(),
            "result_sha256": answer_contract.sha256_file(
                paths["attempt"] / "result.json"
            ),
        },
        "question_audit": {
            "path": paths["question_audit"].relative_to(output_dir).as_posix(),
            "sha256": answer_contract.sha256_file(paths["question_audit"]),
        },
        "actual_model": formal_contract.MODEL,
        "response_ids": _response_ids(paths["attempt"], result),
        "memory_sha256": result["memory"]["after"]["sha256"],
        "source_recall_eligible": spec["source_recall_eligible"],
    }
    checkpoint["checkpoint_content_sha256"] = answer_contract.canonical_hash(
        checkpoint
    )
    answer_contract.atomic_json_no_clobber(paths["checkpoint"], checkpoint)
    from audit_r203_readonly_views import (  # noqa: PLC0415
        audit_formal_question_checkpoint,
    )

    return audit_formal_question_checkpoint(
        output_dir=output_dir,
        condition=condition,
        spec=spec,
        source_binding=source_binding,
        copy_record=copy_record,
    )


def _publish_formal_outputs(
    output_dir: Path, specs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    hypotheses: dict[str, list[dict[str, str]]] = {
        condition: [] for condition in formal_contract.CONDITIONS
    }
    for condition in formal_contract.CONDITIONS:
        for spec in specs:
            paths = _question_paths(output_dir, condition, spec)
            result = answer_contract.read_json(paths["attempt"] / "result.json")
            record = {
                "condition": condition,
                "question_id": spec["question_id"],
                "artifact_id": spec["artifact_id"],
                "dataset_index": spec["dataset_index"],
                "question_index": spec["question_index"],
                "category": spec["category"],
                "question": spec["question"],
                "answer": result["answer"]["text"],
                "answer_response_id": result["answer"]["response_id"],
                "answer_response_model": result["answer"]["response_model"],
                "answer_usage": result["answer"]["usage"],
                "retrieval_model_calls": result["retrieval"]["model_calls"],
                "visible_tokens": result["budget"]["visible_tokens"],
                "source_resolution_tokens": result["budget"][
                    "source_resolution_tokens"
                ],
                "diagnostics": result["diagnostics"],
                "stage_evidence": result["stage_evidence"],
            }
            records.append(record)
            hypotheses[condition].append(
                {
                    "question_id": str(spec["question_id"]),
                    "hypothesis": str(result["answer"]["text"]),
                }
            )
    results_path = output_dir / "results.json"
    _write_or_validate(
        results_path,
        {"schema_version": FORMAL_RUN_SCHEMA, "records": records},
    )
    outputs: dict[str, Any] = {
        "results": {
            "path": results_path.relative_to(output_dir).as_posix(),
            "sha256": answer_contract.sha256_file(results_path),
        },
        "hypotheses": {},
    }
    for condition, rows in hypotheses.items():
        path = output_dir / "hypotheses" / f"{condition}.jsonl"
        text = "".join(
            answer_contract.canonical_json(row) + "\n" for row in rows
        )
        if path.exists():
            if path.is_symlink() or path.read_text(encoding="utf-8") != text:
                raise R203Error("existing formal hypotheses artifact differs")
        else:
            _write_text_no_clobber(path, text)
        outputs["hypotheses"][condition] = {
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": answer_contract.sha256_file(path),
        }
    return outputs


def run_formal(
    *,
    output_dir: Path,
    preregistration: Path = formal_contract.DEFAULT_PREREGISTRATION,
    preflight_path: Path = formal_contract.DEFAULT_PREFLIGHT,
    source_root: Path = formal_contract.DEFAULT_SOURCE_ROOT,
    gateway_root: Path | None = None,
    python: Path = Path(sys.executable),
    allow_model_requests: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    if not allow_model_requests:
        raise R203Error("formal R203 requires --allow-model-requests")
    if gateway_root is None:
        raise R203Error("formal R203 requires --gateway-root")
    if preregistration.expanduser().resolve() != (
        formal_contract.DEFAULT_PREREGISTRATION.resolve()
    ):
        raise R203Error("formal R203 requires the registered default preregistration")
    output_dir = _validate_output_root(output_dir)
    if output_dir.exists() and not resume:
        raise R203Error("formal R203 output exists; pass --resume or use a new root")
    preflight = formal_contract.run_preflight(
        preregistration,
        source_root=source_root,
        write_path=preflight_path,
    )
    source_binding = formal_contract.require_ready_binding(preflight)
    provider_lock = None
    try:
        provider_lock = controlled_answer.flex_evidence.acquire_consumer_lock(
            gateway_root.expanduser().resolve()
        )
        gateway_contract = controlled_answer.flex_evidence.active_contract(
            gateway_root.expanduser().resolve()
        )
    except controlled_answer.flex_evidence.EvidenceError as exc:
        if provider_lock is not None:
            provider_lock.close()
        raise R203Error(str(exc)) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    with provider_lock, answer_contract.FileLock(output_dir / ".r203-formal.lock"):
        if (output_dir / "complete.json").exists():
            from audit_r203_readonly_views import audit_run  # noqa: PLC0415

            return audit_run(output_dir)
        copy_record = _copy_condition_memories(
            output_dir=output_dir, source_binding=source_binding
        )
        manifest = _formal_manifest(
            preregistration=preregistration,
            preflight=preflight,
            source_binding=source_binding,
            copy_record=copy_record,
            gateway_contract=gateway_contract,
            python=python,
        )
        _write_or_validate(output_dir / "run_manifest.json", manifest)
        _write_or_validate(output_dir / "source_binding.json", source_binding)
        progress_path = output_dir / "progress.json"
        answer_contract.atomic_json_replace(progress_path, _progress(output_dir))
        formal_contract.verify_source_unchanged(source_binding)
        _assert_closed_proxy_inventory(output_dir, manifest["run_id"])
        specs = formal_contract.question_specs()
        completed = _audit_existing_formal_checkpoints(
            output_dir=output_dir,
            specs=specs,
            source_binding=source_binding,
            copy_record=copy_record,
        )
        tokenizer = answer_contract.formal_token_counter()
        for condition in formal_contract.CONDITIONS:
            for sample in formal_contract.SAMPLES:
                batch = [
                    spec
                    for spec in specs
                    if int(spec["dataset_index"]) == sample
                ]
                pending: list[Mapping[str, Any]] = []
                for spec in batch:
                    if (condition, str(spec["artifact_id"])) not in completed:
                        pending.append(spec)
                process = process_log = proxy_record = None
                try:
                    if pending:
                        process, proxy_record, process_log = (
                            controlled_answer._start_proxy(  # noqa: SLF001
                                output_dir=output_dir,
                                python=python.expanduser().resolve(),
                                gateway_contract=gateway_contract,
                                run_id=manifest["run_id"],
                            )
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
                            retries=formal_contract.ANSWER_RETRIES,
                            answer_max_tokens=formal_contract.ANSWER_MAX_TOKENS,
                        )
                        proxy_log = output_dir / proxy_record["log"]
                        for spec in pending:
                            execute_formal_question(
                                output_dir=output_dir,
                                run_id=manifest["run_id"],
                                preregistration=preregistration,
                                source_binding=source_binding,
                                copy_record=copy_record,
                                condition=condition,
                                spec=spec,
                                completion_resource=completion_resource,
                                answer_client=answer_client,
                                tokenizer=tokenizer,
                                proxy_log=proxy_log,
                            )
                            answer_contract.atomic_json_replace(
                                progress_path, _progress(output_dir)
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
                _assert_closed_proxy_inventory(output_dir, manifest["run_id"])
        formal_contract.verify_source_unchanged(source_binding)
        formal_contract.validate_copy_inventory(
            output_dir=output_dir,
            source_binding=source_binding,
            inventory=copy_record["rows"],
        )
        progress = _progress(output_dir)
        if progress["completed"] != formal_contract.TOTAL_ARTIFACTS:
            raise R203Error("formal R203 artifact scope is incomplete")
        answer_contract.atomic_json_replace(progress_path, progress)
        outputs = _publish_formal_outputs(output_dir, specs)
        complete = {
            "schema_version": FORMAL_COMPLETE_SCHEMA,
            "status": "complete",
            "run_id": manifest["run_id"],
            "questions_per_condition": formal_contract.PRIMARY_QUESTIONS,
            "condition_artifacts": formal_contract.TOTAL_ARTIFACTS,
            "source_recall_denominator_per_condition": (
                formal_contract.SOURCE_RECALL_DENOMINATOR
            ),
            "source_binding_sha256": source_binding["binding_sha256"],
            "condition_copy_inventory_sha256": copy_record[
                "inventory_content_sha256"
            ],
            "preregistration_sha256": answer_contract.sha256_file(
                preregistration
            ),
            "progress_sha256": answer_contract.sha256_file(progress_path),
            "outputs": outputs,
        }
        answer_contract.atomic_json_no_clobber(
            output_dir / "complete.json", complete
        )
        from audit_r203_readonly_views import audit_run  # noqa: PLC0415

        report = audit_run(output_dir)
        answer_contract.atomic_json_replace(output_dir / "audit.json", report)
        return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--synthetic-sanity", action="store_true")
    modes.add_argument("--formal", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=formal_contract.DEFAULT_PREREGISTRATION,
    )
    parser.add_argument(
        "--preflight-report",
        type=Path,
        default=formal_contract.DEFAULT_PREFLIGHT,
    )
    parser.add_argument(
        "--source-root", type=Path, default=formal_contract.DEFAULT_SOURCE_ROOT
    )
    parser.add_argument("--gateway-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--allow-model-requests", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.formal:
        if args.output_dir is None:
            parser.error("formal R203 requires --output-dir")
        if not args.allow_model_requests:
            parser.error("formal R203 requires explicit --allow-model-requests")
        if args.gateway_root is None:
            parser.error("formal R203 requires --gateway-root")
    else:
        if args.allow_model_requests:
            parser.error("non-formal R203 modes forbid --allow-model-requests")
        if args.gateway_root is not None:
            parser.error("--gateway-root is valid only with --formal")
        if args.resume:
            parser.error("--resume is valid only with --formal")
        if args.synthetic_sanity and args.output_dir is None:
            parser.error("synthetic R203 requires --output-dir")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight:
        result = formal_contract.run_preflight(
            args.preregistration,
            source_root=args.source_root,
            write_path=args.preflight_report,
        )
    elif args.synthetic_sanity:
        result = run_synthetic(args.output_dir)
    else:
        result = run_formal(
            output_dir=args.output_dir,
            preregistration=args.preregistration,
            preflight_path=args.preflight_report,
            source_root=args.source_root,
            gateway_root=args.gateway_root,
            python=args.python,
            allow_model_requests=args.allow_model_requests,
            resume=args.resume,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        R203Error,
        formal_contract.R203FormalError,
        control.ReadOnlyControlError,
        answer_contract.ControlledAnswerError,
        DurableLedgerError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
