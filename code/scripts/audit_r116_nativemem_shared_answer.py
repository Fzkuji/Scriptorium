#!/usr/bin/env python3
"""Independently audit an R116 NativeMem shared-answer artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_readonly_nativemem_control as generic  # noqa: E402
import controlled_locomo_answer_contract as answer_contract  # noqa: E402
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
SOURCE_PATHS = (
    *FROZEN_NATIVE_HASHES,
    "scripts/readonly_nativemem_control.py",
    "scripts/audit_readonly_nativemem_control.py",
    "scripts/run_r116_nativemem_shared_answer.py",
    "scripts/audit_r116_nativemem_shared_answer.py",
    "scripts/controlled_locomo_answer_contract.py",
    "scripts/run_controlled_locomo_answers.py",
    "src/evaluation/durable_model_ledger.py",
    "src/evaluation/visible_token_budget.py",
    "src/evaluation/visible_token_audit.py",
)


class R116AuditError(RuntimeError):
    pass


def _inside_directory(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise R116AuditError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise R116AuditError(f"{label} path escapes output")
    candidate = root / relative
    answer_contract.reject_symlink_components(candidate)
    if candidate.is_symlink() or not candidate.is_dir():
        raise R116AuditError(f"{label} is not a regular directory")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise R116AuditError(f"{label} path escapes output") from exc
    return resolved


def audit_run(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().absolute()
    answer_contract.reject_symlink_components(output_dir)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise R116AuditError("R116 output root is invalid")
    output_dir = output_dir.resolve()
    manifest = answer_contract.read_json(output_dir / "run_manifest.json")
    complete = answer_contract.read_json(output_dir / "complete.json")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != RUN_SCHEMA
        or manifest.get("mode") != "synthetic_no_network"
        or manifest.get("method") != METHOD
        or manifest.get("condition") != "dual_source"
        or manifest.get("scope")
        != {"samples": [0], "questions": 1, "formal": False}
        or manifest.get("config", {}).get("budget_tokens") != 20_000
        or manifest.get("config", {}).get("model") != "gpt-5.5"
        or manifest.get("config", {}).get("network_requests") != 0
    ):
        raise R116AuditError("R116 run manifest differs")
    generic.audit_current_source_hashes(
        manifest.get("source_hashes"),
        expected_paths=SOURCE_PATHS,
        frozen_hashes=FROZEN_NATIVE_HASHES,
    )
    if (
        not isinstance(complete, dict)
        or complete.get("schema_version") != RUN_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("run_id") != manifest.get("run_id")
        or complete.get("network_requests") != 0
        or complete.get("questions") != 1
    ):
        raise R116AuditError("R116 complete manifest differs")
    memory_record = manifest.get("memory_source")
    if not isinstance(memory_record, dict):
        raise R116AuditError("R116 memory record is absent")
    memory_relative = memory_record.get("path")
    if not isinstance(memory_relative, str):
        raise R116AuditError("R116 memory path is absent")
    if memory_relative != "memory_source":
        raise R116AuditError("R116 memory path differs")
    memory = _inside_directory(output_dir, memory_relative, label="R116 memory")
    live_memory = generic.snapshot_memory_path(memory).descriptor
    if live_memory != memory_record.get("descriptor"):
        raise R116AuditError("R116 source-memory descriptor differs")
    question = manifest.get("question")
    if (
        not isinstance(question, dict)
        or question.get("question_id") != "s0_q0"
        or not isinstance(question.get("question"), str)
        or not isinstance(question.get("gold_source_ids"), list)
        or not all(
            isinstance(value, str) and value for value in question["gold_source_ids"]
        )
        or not isinstance(question.get("source_recall_eligible"), bool)
    ):
        raise R116AuditError("R116 question is invalid")
    checkpoint = answer_contract.read_json(output_dir / "completed/s0_q0.json")
    if not isinstance(checkpoint, dict) or checkpoint.get("status") != "complete":
        raise R116AuditError("R116 checkpoint differs")
    if checkpoint.get("artifact_dir") != "questions/s0_q0/attempt-0001":
        raise R116AuditError("R116 checkpoint artifact path differs")
    artifact_dir = _inside_directory(
        output_dir, checkpoint.get("artifact_dir"), label="R116 question artifact"
    )
    result_path = artifact_dir / "result.json"
    result = answer_contract.read_json(result_path)
    if (
        checkpoint.get("result_sha256")
        != answer_contract.sha256_file(result_path)
        or checkpoint.get("result_sha256") != complete.get("result_sha256")
    ):
        raise R116AuditError("R116 checkpoint result hash differs")
    if (
        not isinstance(result, dict)
        or result.get("budget", {}).get("tokenizer")
        != manifest.get("config", {}).get("tokenizer")
    ):
        raise R116AuditError("R116 tokenizer linkage differs")
    question_report = generic.audit_question(
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
    if question_report != complete.get("question_audit"):
        raise R116AuditError("R116 recorded question audit differs")
    if (
        question_report["mapped_source_recall"] != 1.0
        or question_report["first_relevant_file"] != "topics/pets.md"
        or question_report["source_resolution_tokens"] <= 0
    ):
        raise R116AuditError("R116 synthetic diagnostics differ")
    return {
        "schema_version": "nativemem.r116-shared-answer-audit.v1",
        "status": "passed",
        "mode": "synthetic_no_network",
        "run_id": manifest["run_id"],
        "network_requests": 0,
        "questions": 1,
        "budget_tokens": 20_000,
        "model": "gpt-5.5",
        "memory_unchanged": True,
        "question_audit": question_report,
        "formal_status": "blocked_waiting_frozen_nativemem_memory",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(audit_run(args.artifact_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        R116AuditError,
        generic.ReadOnlyAuditError,
        answer_contract.ControlledAnswerError,
        DurableLedgerError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
