#!/usr/bin/env python3
"""R116 NativeMem shared-answer control; synthetic implementation gate.

Formal execution remains disabled until the frozen all-ten NativeMem LoCoMo
memory artifact is complete and independently audited.  ``--synthetic-sanity``
exercises the exact 20K gate, read-only retrieval, durable model ledger, proxy
prefix evidence, fixed answerer, checkpoint, and independent auditor without a
network request.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_readonly_nativemem_control as generic_auditor  # noqa: E402
import controlled_locomo_answer_contract as answer_contract  # noqa: E402
import readonly_nativemem_control as control  # noqa: E402
import run_controlled_locomo_answers as controlled_answer  # noqa: E402
from src.evaluation.durable_model_ledger import DurableLedgerError  # noqa: E402


RUN_SCHEMA = "nativemem.r116-shared-answer-run.v1"
METHOD = "nativemem_shared_answer"
FROZEN_NATIVE_HASHES = {
    "src/nativemem.py": "305de2a10dae40631fd3e376be576cd37afcca37fe11aa739e13ed60efb46d82",
    "src/v8_memory.py": "7847825c7819c90270950dbbba8aa52fa3d93118240f25d6300c121e90b22c8b",
    "src/adapters/run_nativemem.py": "b5e9388660f6665d47149763437bd9044f7945eb4fdc460a9da1d68a88ba7031",
    "src/chatgpt_proxy.py": "a0fba59861f99e5bb9134e4359297b26b9f45240be57906cc2fdfea64aef0321",
    "scripts/run_v88_gpt55_locomo.py": "3a9ddeaf851fd4cc76343fc11b38e6bd5629a42baed5c3f35a83be3ced9cdfc7",
}


class R116Error(RuntimeError):
    pass


def _source_hashes() -> dict[str, str]:
    paths = {
        **{relative: ROOT / relative for relative in FROZEN_NATIVE_HASHES},
        "scripts/readonly_nativemem_control.py": Path(control.__file__).resolve(),
        "scripts/audit_readonly_nativemem_control.py": Path(
            generic_auditor.__file__
        ).resolve(),
        "scripts/run_r116_nativemem_shared_answer.py": Path(__file__).resolve(),
        "scripts/audit_r116_nativemem_shared_answer.py": (
            ROOT / "scripts/audit_r116_nativemem_shared_answer.py"
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
    hashes = {name: answer_contract.sha256_file(path) for name, path in paths.items()}
    for relative, expected in FROZEN_NATIVE_HASHES.items():
        if hashes[relative] != expected:
            raise R116Error(f"frozen NativeMem source differs: {relative}")
    return hashes


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


def _synthetic_fixture(output_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    memory = output_dir / "memory_source"
    _write_text_no_clobber(
        memory / "topics/pets.md",
        "## Pets\n[2025-01-01] Ari adopted a cat named Pixel [D1:1].\n",
    )
    _write_text_no_clobber(
        memory / "timeline/2025/01/01.md",
        "[2025-01-01] Ari adopted Pixel · [D1:1] → topics/pets.md\n",
    )
    conversation = {
        "speaker_a": "Ari",
        "speaker_b": "Bo",
        "session_1_date_time": "1 January 2025",
        "session_1": [
            {
                "speaker": "Ari",
                "dia_id": "D1:1",
                "text": "I adopted a cat named Pixel.",
            },
            {
                "speaker": "Bo",
                "dia_id": "D1:2",
                "text": "Pixel is a good name.",
            },
        ],
    }
    question = {
        "question_id": "s0_q0",
        "question": "What is the cat's name?",
        "gold_source_ids": ["D1:1"],
        "source_recall_eligible": True,
    }
    return memory, conversation, question


def _existing_report(output_dir: Path) -> dict[str, Any] | None:
    complete = output_dir / "complete.json"
    if not complete.exists():
        return None
    from audit_r116_nativemem_shared_answer import audit_run  # noqa: PLC0415

    return audit_run(output_dir)


def run_synthetic(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().absolute()
    answer_contract.reject_symlink_components(output_dir)
    if output_dir.is_symlink():
        raise R116Error("output directory must not be a symlink")
    if output_dir.exists():
        report = _existing_report(output_dir)
        if report is None:
            raise R116Error("existing R116 output is incomplete; use a new root")
        return report
    output_dir.mkdir(parents=True)
    with answer_contract.FileLock(output_dir / ".r116.lock"):
        source_hashes = _source_hashes()
        memory, conversation, question = _synthetic_fixture(output_dir)
        memory_identity = control.memory_descriptor(memory)
        run_id = f"r116-synthetic-{memory_identity['sha256'][:16]}"
        manifest = {
            "schema_version": RUN_SCHEMA,
            "status": "running",
            "mode": "synthetic_no_network",
            "run_id": run_id,
            "created_at": answer_contract.utc_now(),
            "method": METHOD,
            "condition": "dual_source",
            "scope": {
                "samples": [0],
                "questions": 1,
                "formal": False,
            },
            "config": {
                "model": control.EXPECTED_MODEL,
                "budget_policy": "hard_cap",
                "budget_tokens": control.EXPECTED_BUDGET,
                "tokenizer": answer_contract.formal_token_counter().identity,
                "max_rounds": 12,
                "network_requests": 0,
            },
            "source_hashes": source_hashes,
            "memory_source": {
                "path": str(memory.relative_to(output_dir)),
                "descriptor": memory_identity,
                "frozen_main_artifact_used": False,
            },
            "formal_blocker": (
                "frozen all-ten NativeMem LoCoMo memory artifact is not complete "
                "and independently audited"
            ),
            "question": question,
            "conversation": conversation,
        }
        answer_contract.atomic_json_no_clobber(output_dir / "run_manifest.json", manifest)
        proxy_log = output_dir / "synthetic_proxy.jsonl"
        answer_client = controlled_answer.FakeAnswerClient(
            proxy_log=proxy_log,
            run_id=run_id,
            response_text="<answer>Pixel</answer>",
        )
        retrieval = control.ScriptedCompletionResource(
            [
                [
                    {
                        "name": "read_memory_file",
                        "arguments": {"path": "topics/pets.md"},
                    }
                ],
                [
                    {
                        "name": "resolve_sources",
                        "arguments": {"source_ids": ["D1:1"]},
                    }
                ],
                [],
            ]
        )
        artifact_dir = output_dir / "questions/s0_q0/attempt-0001"
        result = control.execute_question(
            artifact_dir=artifact_dir,
            run_id=run_id,
            method=METHOD,
            condition="dual_source",
            memory_root=memory,
            turn_index=control.build_turn_index(conversation),
            question_id=question["question_id"],
            question=question["question"],
            gold_source_ids=question["gold_source_ids"],
            source_recall_eligible=question["source_recall_eligible"],
            completion_resource=retrieval,
            answer_client=answer_client,
            tokenizer=answer_contract.formal_token_counter(),
            formal=False,
            proxy_log=proxy_log,
        )
        checkpoint = {
            "schema_version": RUN_SCHEMA,
            "status": "complete",
            "question_id": question["question_id"],
            "artifact_dir": str(artifact_dir.relative_to(output_dir)),
            "result_sha256": answer_contract.sha256_file(
                artifact_dir / "result.json"
            ),
        }
        answer_contract.atomic_json_no_clobber(
            output_dir / "completed/s0_q0.json", checkpoint
        )
        question_report = generic_auditor.audit_question(
            artifact_dir=artifact_dir,
            memory_root=memory,
            method=METHOD,
            condition="dual_source",
            question_id=question["question_id"],
            question=question["question"],
            gold_source_ids=question["gold_source_ids"],
            source_recall_eligible=question["source_recall_eligible"],
            formal=False,
        )
        complete = {
            "schema_version": RUN_SCHEMA,
            "status": "complete",
            "run_id": run_id,
            "completed_at": answer_contract.utc_now(),
            "network_requests": 0,
            "questions": 1,
            "result_sha256": checkpoint["result_sha256"],
            "question_audit": question_report,
            "visible_tokens": result["budget"]["visible_tokens"],
            "source_resolution_tokens": result["budget"][
                "source_resolution_tokens"
            ],
        }
        answer_contract.atomic_json_no_clobber(output_dir / "complete.json", complete)
        from audit_r116_nativemem_shared_answer import audit_run  # noqa: PLC0415

        report = audit_run(output_dir)
        answer_contract.atomic_json_replace(output_dir / "audit.json", report)
        return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic-sanity", action="store_true")
    parser.add_argument("--allow-model-requests", action="store_true")
    args = parser.parse_args(argv)
    if not args.synthetic_sanity:
        parser.error(
            "formal R116 remains disabled until the frozen NativeMem memory "
            "artifact passes its independent audit"
        )
    if args.allow_model_requests:
        parser.error("synthetic R116 forbids --allow-model-requests")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_synthetic(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        R116Error,
        control.ReadOnlyControlError,
        answer_contract.ControlledAnswerError,
        DurableLedgerError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
