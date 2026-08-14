from __future__ import annotations

import fcntl
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_r403_task_eval as auditor  # noqa: E402
import controlled_locomo_answer_contract as answer_contract  # noqa: E402
import r403_task_eval_contract as contract  # noqa: E402
import readonly_nativemem_control as readonly_control  # noqa: E402
import run_r403_task_eval as runner  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_artifacts() -> tuple[Path, Path, Path]:
    results = ROOT / "results"
    with tempfile.TemporaryDirectory(
        prefix="test-r403-task-eval-", dir=results
    ) as raw:
        root = Path(raw)
        source = root / "source"
        output = root / "task"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "run_r403_incremental_growth.py"),
                "--synthetic-sanity",
                "--output-dir",
                str(source),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = runner.run_synthetic(source_root=source, output_dir=output)
        assert report["status"] == "pass"
        yield root, source, output


def _copy_output(tmp_path: Path, source: Path) -> Path:
    destination = tmp_path / "copied-output"
    shutil.copytree(source, destination)
    return destination


def _json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_protocol_freeze_and_preregistration_are_live() -> None:
    protocol = answer_contract.read_json(contract.PROTOCOL_FREEZE)
    assert protocol["status"] == "frozen_before_task_eval_implementation"
    assert "all 1540" in protocol["denominators"]["task_accuracy_100"]
    prereg = contract.validate_preregistration()
    assert prereg["scope"]["primary_questions_at_100_percent"] == 1540
    assert prereg["scope"]["primary_source_recall_at_100_percent"] == 1533
    assert prereg["retrieval_and_answer"]["visible_token_budget_tokens"] == 20_000
    assert prereg["retrieval_and_answer"]["max_retrieval_rounds"] == 12
    assert prereg["formal_execution_gate"]["scope_limit_flags_available"] is False
    assert prereg["formal_execution_gate"][
        "gateway_origin_derived_from_validated_root"
    ] is True
    assert prereg["formal_execution_gate"]["exact_provider_window_required"] is True


def test_synthetic_four_checkpoint_scope_and_independent_audit(
    synthetic_artifacts: tuple[Path, Path, Path],
) -> None:
    _root, _source, output = synthetic_artifacts
    report = auditor.audit_run(output, require_complete=True)
    assert report["status"] == "pass"
    assert report["mode"] == "synthetic_no_network"
    assert report["by_checkpoint"] == {"10": 1, "25": 1, "50": 2, "100": 4}
    assert report["question_checkpoint_artifacts"] == 8
    assert report["source_recall_artifacts"] == 7
    assert report["old_fact_artifacts"] == 4
    assert report["update_artifacts"] == 1
    assert report["future_source_ids_observed"] == 0
    assert report["model_requests"] == 0
    assert report["network_requests"] == 0
    assert report["auditor_independence"]["imports_task_runner"] is False


def test_denominators_follow_frozen_inventory(
    synthetic_artifacts: tuple[Path, Path, Path],
) -> None:
    _root, _source, output = synthetic_artifacts
    summary = answer_contract.read_json(output / "checkpoint_summary_inputs.json")
    rows = summary["checkpoints"]
    assert {key: value["task_denominator"] for key, value in rows.items()} == {
        "10": 1,
        "25": 1,
        "50": 2,
        "100": 4,
    }
    assert rows["100"]["source_reachability_denominator"] == 3
    assert rows["50"]["update_denominator"] == 1
    scoring = auditor._read_jsonl(output / "scoring_inputs.jsonl")  # noqa: SLF001
    excluded = [
        row for row in scoring
        if row["checkpoint"] == 100 and not row["source_recall_eligible"]
    ]
    assert len(excluded) == 1
    assert excluded[0]["cohorts"]["main_qa_100"] is True
    assert excluded[0]["cohorts"]["source"] is False


def test_checkpoint_prefix_prevents_future_turn_resolution(
    synthetic_artifacts: tuple[Path, Path, Path],
) -> None:
    _root, source, output = synthetic_artifacts
    binding = answer_contract.read_json(output / "source_binding.json")
    context = contract.load_checkpoint_context(
        source, binding, sample=0, checkpoint=10, formal=False
    )
    assert set(context["turn_index"]) == {"D1:1"}
    assert "D2:1" not in context["turn_index"]
    text, valid = readonly_control.resolve_sources(
        ["D1:1", "D2:1", "D10:1"], context["turn_index"]
    )
    assert valid == ["D1:1"]
    assert "D2:1" not in text and "D10:1" not in text


def test_future_delivery_is_rejected(
    tmp_path: Path,
    synthetic_artifacts: tuple[Path, Path, Path],
) -> None:
    _root, source, output = synthetic_artifacts
    binding = answer_contract.read_json(output / "source_binding.json")
    specs = contract.specs_from_binding(source, binding, formal=False)
    spec = next(row for row in specs if row["checkpoint"] == 10)
    context = contract.load_checkpoint_context(
        source, binding, sample=0, checkpoint=10, formal=False
    )
    original = (
        output / "checkpoints/checkpoint-010/questions"
        / spec["artifact_id"] / "attempt-0001"
    )
    copied = tmp_path / "attempt"
    shutil.copytree(original, copied)
    trace = copied / "visible_tokens.jsonl"
    records = auditor._read_jsonl(trace)  # noqa: SLF001
    delivery = next(row for row in records if row.get("record_type") == "delivery")
    delivery["delivered"]["text"] = "future evidence [D10:1]"
    trace.write_text(
        "".join(answer_contract.canonical_json(row) + "\n" for row in records),
        encoding="utf-8",
    )
    with pytest.raises(contract.R403TaskEvalError, match="future source"):
        contract.build_m4_trace(
            attempt=copied,
            spec=spec,
            turn_index=context["turn_index"],
        )


def test_wrong_source_resolution_does_not_count_as_correct_reach(
    tmp_path: Path,
    synthetic_artifacts: tuple[Path, Path, Path],
) -> None:
    _root, source, output = synthetic_artifacts
    binding = answer_contract.read_json(output / "source_binding.json")
    specs = contract.specs_from_binding(source, binding, formal=False)
    spec = next(row for row in specs if row["checkpoint"] == 50 and row["question_index"] == 1)
    context = contract.load_checkpoint_context(
        source, binding, sample=0, checkpoint=50, formal=False
    )
    original = (
        output / "checkpoints/checkpoint-050/questions"
        / spec["artifact_id"] / "attempt-0001"
    )
    copied = tmp_path / "attempt"
    shutil.copytree(original, copied)
    trace = copied / "visible_tokens.jsonl"
    records = auditor._read_jsonl(trace)  # noqa: SLF001
    for row in records:
        if row.get("record_type") == "delivery" and row.get("kind") == "source_resolution":
            row["delivered"]["text"] = "[D2:1] wrong checkpoint evidence"
    trace.write_text(
        "".join(answer_contract.canonical_json(row) + "\n" for row in records),
        encoding="utf-8",
    )
    rebuilt = contract.build_m4_trace(
        attempt=copied,
        spec=spec,
        turn_index=context["turn_index"],
    )
    assert rebuilt["m4_fields"]["source_resolution_returned_gold_content"] is False


def test_tampered_metrics_fail_independent_audit(
    tmp_path: Path,
    synthetic_artifacts: tuple[Path, Path, Path],
) -> None:
    _root, _source, output = synthetic_artifacts
    copied = _copy_output(tmp_path, output)
    metrics_path = next(copied.glob("checkpoints/*/questions/*/metrics.json"))
    metrics = answer_contract.read_json(metrics_path)
    metrics["navigation"]["visible_tokens"] += 1
    _json_write(metrics_path, metrics)
    with pytest.raises(Exception):
        auditor.audit_run(copied, require_complete=True)


def test_partial_question_fails_closed(
    tmp_path: Path,
    synthetic_artifacts: tuple[Path, Path, Path],
) -> None:
    _root, _source, output = synthetic_artifacts
    copied = _copy_output(tmp_path, output)
    next(copied.glob("checkpoints/*/questions/*/checkpoint.json")).unlink()
    with pytest.raises(auditor.R403TaskEvalAuditError):
        auditor.audit_run(copied, require_complete=True)


def test_extra_conflicting_question_is_rejected(
    tmp_path: Path,
    synthetic_artifacts: tuple[Path, Path, Path],
) -> None:
    _root, _source, output = synthetic_artifacts
    copied = _copy_output(tmp_path, output)
    questions = copied / "checkpoints/checkpoint-010/questions"
    shutil.copytree(next(questions.iterdir()), questions / "unexpected-question")
    with pytest.raises(
        auditor.R403TaskEvalAuditError, match="missing/extra"
    ):
        auditor.audit_run(copied, require_complete=True)


def test_unassigned_proxy_event_is_rejected(
    tmp_path: Path,
    synthetic_artifacts: tuple[Path, Path, Path],
) -> None:
    _root, _source, output = synthetic_artifacts
    copied = _copy_output(tmp_path, output)
    proxy = copied / "synthetic_proxy.jsonl"
    records = auditor._read_jsonl(proxy)  # noqa: SLF001
    extra = dict(records[0])
    extra["event_id"] = "synthetic-unassigned-event"
    with proxy.open("a", encoding="utf-8") as handle:
        handle.write(answer_contract.canonical_json(extra) + "\n")
    with pytest.raises(auditor.R403TaskEvalAuditError, match="unassigned"):
        auditor.audit_run(copied, require_complete=True)


def test_visible_budget_tamper_is_rejected(
    tmp_path: Path,
    synthetic_artifacts: tuple[Path, Path, Path],
) -> None:
    _root, _source, output = synthetic_artifacts
    copied = _copy_output(tmp_path, output)
    manifest_path = next(
        copied.glob("checkpoints/*/questions/*/attempt-0001/visible_tokens.manifest.json")
    )
    manifest = answer_contract.read_json(manifest_path)
    manifest["configured_budget_tokens"] = 19_999
    _json_write(manifest_path, manifest)
    with pytest.raises(Exception):
        auditor.audit_run(copied, require_complete=True)


def test_formal_gate_and_gateway_root_are_fail_closed(
    tmp_path: Path, fake_flex_provider
) -> None:
    with pytest.raises(runner.R403TaskEvalRunError, match="not authorized"):
        runner.run_formal(
            source_root=tmp_path / "missing-source",
            output_dir=tmp_path / "output",
            allow_model_requests=False,
        )
    with pytest.raises(runner.R403TaskEvalRunError, match="gateway-root"):
        runner.run_formal(
            source_root=tmp_path / "missing-source",
            output_dir=tmp_path / "output",
            allow_model_requests=True,
        )
    provider_contract = runner.controlled_answer.flex_evidence.active_contract(
        fake_flex_provider.root
    )
    assert provider_contract["origin"].startswith("http://127.0.0.1:")
    assert provider_contract["origin"] != "http://127.0.0.1:8199"


def test_formal_preflight_rejects_synthetic_source_without_requests(
    tmp_path: Path,
    synthetic_artifacts: tuple[Path, Path, Path],
) -> None:
    _root, source, _output = synthetic_artifacts
    report = contract.run_preflight(
        source_root=source,
        write_path=tmp_path / "preflight.json",
    )
    assert report["status"] == "blocked"
    assert report["model_requests"] == 0
    assert report["network_requests"] == 0


def test_active_growth_lock_is_rejected(
    synthetic_artifacts: tuple[Path, Path, Path],
) -> None:
    _root, source, _output = synthetic_artifacts
    contract.BUILD_ACTIVE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with contract.BUILD_ACTIVE_LOCK.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(contract.R403TaskEvalError, match="actively locked"):
                contract.audit_growth_source(source, formal=False)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    contract.BUILD_ACTIVE_LOCK.unlink(missing_ok=True)


def test_growth_source_tamper_is_rejected(
    tmp_path: Path,
    synthetic_artifacts: tuple[Path, Path, Path],
) -> None:
    _root, source, _output = synthetic_artifacts
    copied = tmp_path / "source"
    shutil.copytree(source, copied)
    inventory = copied / (
        "samples/sample-0/checkpoints/checkpoint-010/question_inventory.jsonl"
    )
    with inventory.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(Exception):
        contract.audit_growth_source(copied, formal=False)


def test_complete_resume_is_no_clobber(
    synthetic_artifacts: tuple[Path, Path, Path],
) -> None:
    _root, source, output = synthetic_artifacts
    before = answer_contract.sha256_file(output / "complete.json")
    report = runner.run_synthetic(source_root=source, output_dir=output)
    assert report["status"] == "pass"
    assert answer_contract.sha256_file(output / "complete.json") == before


def test_auditor_source_does_not_import_task_runner() -> None:
    source = (SCRIPTS / "audit_r403_task_eval.py").read_text(encoding="utf-8")
    assert "import run_r403_task_eval" not in source
    assert "from run_r403_task_eval" not in source


def test_formal_budget_override_is_rejected_before_model_call(
    tmp_path: Path,
    synthetic_artifacts: tuple[Path, Path, Path],
) -> None:
    _root, source, output = synthetic_artifacts
    binding = answer_contract.read_json(output / "source_binding.json")
    context = contract.load_checkpoint_context(
        source, binding, sample=0, checkpoint=10, formal=False
    )
    with pytest.raises(
        readonly_control.ReadOnlyControlError, match="budget must be exactly"
    ):
        readonly_control.execute_question(
            artifact_dir=tmp_path / "attempt",
            run_id="budget-rejection",
            method=contract.METHOD,
            condition=contract.GENERIC_CONDITION,
            memory_root=context["memory"],
            turn_index=context["turn_index"],
            question_id="budget:q000",
            question="What happened?",
            gold_source_ids=["D1:1"],
            source_recall_eligible=True,
            completion_resource=None,
            answer_client=None,
            tokenizer=answer_contract.formal_token_counter(),
            formal=True,
            proxy_log=None,
            budget_tokens=19_999,
        )
