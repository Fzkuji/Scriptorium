from __future__ import annotations

import json
import shutil
import socket
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_r207_task_eval as auditor  # noqa: E402
import r207_task_eval_contract as contract  # noqa: E402
import run_r207_task_eval as runner  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_run() -> tuple[Path, dict[str, object]]:
    root = Path(
        tempfile.mkdtemp(prefix="r207-task-eval-test-", dir=ROOT / "results")
    )
    output = root / "artifact"
    try:
        report = runner.run_synthetic(
            output,
            source_root=contract.DEFAULT_SYNTHETIC_SOURCE,
        )
        yield output, report
    finally:
        shutil.rmtree(root)


def test_primary_scope_and_exact_preregistration() -> None:
    specs = contract.primary_question_specs()
    assert len(specs) == 1540
    assert sum(bool(spec["source_recall_eligible"]) for spec in specs) == 1533
    assert {int(spec["category"]) for spec in specs} == {1, 2, 3, 4}
    prereg = contract.validate_preregistration()
    assert prereg == contract.build_preregistration()
    assert prereg["scope"]["category_5_questions"] == 446
    assert prereg["scope"]["primary_question_condition_artifacts"] == 3080
    assert prereg["formal_execution_gate"] == {
        "all_ten_source_required": True,
        "allow_model_requests_flag_required": True,
        "audited_R207_source_required": True,
        "gateway_root_flag_required": True,
        "gateway_origin_derived_from_validated_root": True,
        "arbitrary_upstream_forbidden": True,
        "exclusive_consumer_lock_required": True,
        "exact_provider_window_required": True,
        "provider_model": "gpt-5.5-2026-04-23",
        "service_tier": "flex",
        "exclusive_controlled_proxy_required": True,
        "scope_limit_flags_available": False,
    }


def test_current_formal_preflight_is_fail_closed_and_offline(tmp_path: Path) -> None:
    report = contract.run_preflight(
        write_path=tmp_path / "preflight.json",
        source_root=contract.DEFAULT_R207_ROOT,
    )
    assert report["status"] == "blocked"
    assert report["formal_source"] is None
    assert report["model_requests"] == 0
    assert report["network_requests"] == 0
    assert "not a regular directory" in report["errors"][0]


def test_cli_requires_explicit_formal_gates(fake_flex_provider) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--formal", "--r207-root", "missing"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--formal", "--allow-model-requests"])
    parsed = runner.parse_args(
        [
            "--formal",
            "--allow-model-requests",
            "--r207-root",
            "missing",
            "--output-dir",
            "out",
            "--gateway-root",
            str(fake_flex_provider.root),
        ]
    )
    assert parsed.allow_model_requests is True


def test_formal_rejects_synthetic_r207_before_output(
    tmp_path: Path, fake_flex_provider
) -> None:
    fixture_dir = ROOT / "results" / f"r207-formal-reject-{tmp_path.name}"
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    fixture_dir.mkdir(parents=True)
    output = fixture_dir / "output"
    live_preregistration = fixture_dir / "live-r207-task-preregistration.json"
    live_preregistration.write_text(
        json.dumps(contract.build_preregistration(), indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        with pytest.raises(contract.R207TaskEvalError, match="all-ten"):
            runner.run_formal(
                output_dir=output,
                source_root=contract.DEFAULT_SYNTHETIC_SOURCE,
                allow_model_requests=True,
                gateway_root=fake_flex_provider.root,
                preregistration=live_preregistration,
                preflight_path=fixture_dir / "formal-preflight.json",
            )
        assert not output.exists()
    finally:
        if fixture_dir.exists():
            shutil.rmtree(fixture_dir)


def test_synthetic_dual_condition_no_network(
    synthetic_run: tuple[Path, dict[str, object]],
) -> None:
    output, report = synthetic_run
    assert report["status"] == "passed"
    assert report["path_conditions"] == list(contract.PATH_CONDITIONS)
    assert report["question_condition_artifacts"] == 2
    assert report["primary_scores"] == 2
    assert report["category_5_scores_mixed"] == 0
    assert report["source_recall_questions"] == 2
    assert report["mean_mapped_source_recall"] == 1.0
    assert report["retrieval_model_calls"] == 6
    assert report["response_ids"] == 8
    assert report["response_ids_unique"] is True
    assert report["memory_unchanged"] is True
    assert report["model_requests"] == 0
    assert report["network_requests"] == 0
    assert report["actual_model_evidence"]["events"] == 8
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["primary_only"] is True
    assert summary["category_5_mixed"] is False
    for condition in contract.PATH_CONDITIONS:
        row = summary["conditions"][condition]
        assert row["primary_questions"] == 1
        assert row["category_5_questions"] == 0
        assert row["mean_official_locomo_f1"] == 1.0


def test_independent_auditor_reconstructs_every_question_output(
    synthetic_run: tuple[Path, dict[str, object]],
) -> None:
    output, expected = synthetic_run
    assert auditor.audit_run(output) == expected
    for condition in contract.PATH_CONDITIONS:
        root = output / "conditions" / condition / "questions" / "s00-q000"
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["first_relevant_file"] in {
            "topics/A/pets.md",
            "topics/B/home.md",
        }
        assert metrics["mapped_source_recall"] == 1.0
        assert metrics["navigation_calls"] == {
            "read_calls": 1,
            "retrieval_model_calls": 3,
            "tool_calls": 2,
        }
        assert metrics["memory_tree"]["unchanged"] is True
        assert metrics["r004"]["visible_token_trace_sha256"]
        assert metrics["m4"]["fields"] == {
            "gold_source_in_canonical_entries": True,
            "gold_source_mapping_complete": True,
            "gold_source_path_valid": True,
            "gold_source_survived_maintenance": True,
            "retrieval_reached_gold_source": True,
            "source_resolution_returned_gold_content": True,
        }
        assert metrics["durable_ledgers"]["retrieval"]["sha256"]
        assert metrics["durable_ledgers"]["answer"]["sha256"]
        assert metrics["exclusive_proxy_evidence"]["retrieval_event_ids"]
        assert metrics["exclusive_proxy_evidence"]["answer_event_ids"]


def test_resume_audits_complete_checkpoints_without_new_events(
    synthetic_run: tuple[Path, dict[str, object]],
) -> None:
    output, expected = synthetic_run
    proxy = output / "synthetic_proxy.jsonl"
    before = proxy.read_bytes()
    second = runner.run_synthetic(
        output,
        source_root=contract.DEFAULT_SYNTHETIC_SOURCE,
    )
    assert second == expected
    assert proxy.read_bytes() == before


def test_incomplete_synthetic_root_fails_closed(tmp_path: Path) -> None:
    output = ROOT / "results" / f"r207-incomplete-{tmp_path.name}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "partial.txt").write_text("partial\n", encoding="utf-8")
    try:
        with pytest.raises(runner.R207TaskEvalRunError, match="incomplete"):
            runner.run_synthetic(
                output,
                source_root=contract.DEFAULT_SYNTHETIC_SOURCE,
            )
    finally:
        shutil.rmtree(output)


def test_tampered_metric_is_rejected(
    synthetic_run: tuple[Path, dict[str, object]], tmp_path: Path
) -> None:
    source, _report = synthetic_run
    copied = tmp_path / "tampered"
    shutil.copytree(source, copied)
    path = (
        copied
        / "conditions/model_directed/questions/s00-q000/metrics.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mapped_source_recall"] = 0.0
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(auditor.R207TaskEvalAuditError):
        auditor.audit_run(copied)


def test_synthetic_execution_never_opens_a_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_connect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("synthetic task evaluation attempted network access")

    monkeypatch.setattr(socket.socket, "connect", reject_connect)
    root = Path(
        tempfile.mkdtemp(prefix="r207-task-no-network-", dir=ROOT / "results")
    )
    try:
        report = runner.run_synthetic(
            root / "artifact",
            source_root=contract.DEFAULT_SYNTHETIC_SOURCE,
        )
        assert report["network_requests"] == 0
    finally:
        shutil.rmtree(root)


def test_auditor_does_not_import_task_runner() -> None:
    source = (SCRIPTS / "audit_r207_task_eval.py").read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith('"')
    )
    assert "import run_r207_task_eval" not in executable
    assert "from run_r207_task_eval" not in executable
