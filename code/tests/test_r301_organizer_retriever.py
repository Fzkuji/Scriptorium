from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


pytest.importorskip("tiktoken")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import audit_r301_organizer_retriever as auditor  # noqa: E402
import run_r301_organizer_retriever as runner  # noqa: E402


def temporary_result_root(label: str) -> Path:
    return ROOT / "results" / f"r301-test-{label}-{uuid.uuid4().hex[:10]}"


def run_synthetic(output: Path) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_r301_organizer_retriever.py"),
            "--synthetic-sanity",
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_preregistration_binds_frozen_m4_rule() -> None:
    prereg = runner.validate_preregistration(runner.DEFAULT_PREREG)
    assert prereg["m4_comparison_preregistration"]["sha256"] == (
        runner.EXPECTED_M4_COMPARISON_SHA256
    )
    assert prereg["interaction"]["formula"] == (
        "(O55-R55 - O4o-R55) - (O55-R4o - O4o-R4o)"
    )


def test_placement_contract_rejects_reordering_and_unsafe_paths() -> None:
    _samples, banks, _questions = runner._synthetic_inputs()
    bank = banks[0]
    valid = {
        "placements": [
            {"entry_id": entry["entry_id"], "topic_path": f"topic/{index}"}
            for index, entry in enumerate(bank["entries"])
        ]
    }
    assert runner.validate_placements(valid, bank) == valid["placements"]
    reordered = {"placements": list(reversed(valid["placements"]))}
    with pytest.raises(runner.R301Error, match="reordered"):
        runner.validate_placements(reordered, bank)
    unsafe = json.loads(json.dumps(valid))
    unsafe["placements"][0]["topic_path"] = "../escape"
    with pytest.raises(runner.R301Error, match="unsafe"):
        runner.validate_placements(unsafe, bank)


def test_interaction_rule_uses_equal_replicates_and_conversation_clusters() -> None:
    values = {
        "O55-R55": 1.0,
        "O4o-R55": 0.0,
        "O55-R4o": 0.0,
        "O4o-R4o": 1.0,
    }
    records = []
    for sample in ("c0", "c1"):
        for question in ("q0", "q1"):
            for cell, value in values.items():
                for replicate in runner.REPLICATES:
                    records.append(
                        {
                            "sample_id": sample,
                            "question_id": question,
                            "cell": cell,
                            "fixed_answerer_f1_set": value,
                            "replicate": replicate,
                        }
                    )
    analysis = runner.analyze_interaction(records, formal=True)
    assert analysis["interaction"]["point_estimate"] == 2.0
    assert analysis["interaction"]["confidence_interval_95"] == {
        "lower": 2.0,
        "upper": 2.0,
    }
    assert analysis["decision"]["claim_status"] == (
        "model_specific_organization_supported"
    )


def test_no_network_full_matrix_audit_and_resume() -> None:
    output = temporary_result_root("full")
    try:
        report = run_synthetic(output)
        assert report["status"] == "pass"
        assert report["network_requests"] == 0
        assert report["organizer_artifact_count"] == 6
        assert report["question_result_count"] == 12
        assert report["model_call_count"] == 42
        ledger_hash = runner.sha256_file(output / "operations.jsonl")
        second = run_synthetic(output)
        assert second["run_fingerprint"] == report["run_fingerprint"]
        assert runner.sha256_file(output / "operations.jsonl") == ledger_hash
        assert auditor.audit_run(output)["status"] == "pass"
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_independent_auditor_rejects_question_tampering() -> None:
    output = temporary_result_root("tamper")
    try:
        run_synthetic(output)
        inventory = auditor.load_jsonl(output / "analysis/question_results.jsonl")
        result_path = output / inventory[0]["result_path"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["fixed_answerer_f1_set"] = 0.123
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        with pytest.raises(auditor.AuditFailure):
            auditor.audit_run(output)
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_formal_execution_is_explicitly_gated() -> None:
    output = temporary_result_root("formal-gate")
    args = runner.parse_args(["--output-dir", str(output), "--samples", "0-9"])
    try:
        with pytest.raises(runner.R301Error, match="allow-model-requests"):
            runner.run(args)
        assert not output.exists()
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_stale_proxy_recovery_is_no_clobber_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy_dir = tmp_path / "proxy" / "gpt4o"
    proxy_dir.mkdir(parents=True)
    log = proxy_dir / "launch-001.jsonl"
    ready = proxy_dir / "launch-001.ready.json"
    log.write_text("", encoding="utf-8")
    ready.write_text(
        json.dumps(
            {
                "run_id": "test-run:gpt4o:launch-001",
                "pid": 987654321,
                "log": str(log),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner.ManagedProxy,
        "process_command",
        staticmethod(lambda _pid: None),
    )

    runner.ManagedProxy.recover_stale(ready)
    stopped = runner.ManagedProxy.stopped_path(ready)
    first_sha = runner.sha256_file(stopped)
    assert runner.read_json(stopped)["status"] == "already_exited"

    runner.ManagedProxy.recover_stale(ready)
    assert runner.sha256_file(stopped) == first_sha


def test_independent_auditor_binds_formal_proxy_launches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg = runner.validate_preregistration(runner.DEFAULT_PREREG)
    fingerprint = "a" * 64
    config = {
        "gpt55_gateway": {
            "schema": runner.flex_evidence.CONTRACT_SCHEMA,
            "origin": "http://127.0.0.1:19055",
        },
        "gpt55_upstream": "http://127.0.0.1:19055",
        "gpt4o_upstream": "https://openrouter.ai/api/v1",
        "gpt4o_api_key_env": "OPENROUTER_API_KEY",
    }
    expected_logs = {}
    for index, (label, profile, upstream, api_key_env) in enumerate(
        (
            (
                "gpt55",
                prereg["answerer"],
                config["gpt55_upstream"],
                None,
            ),
            (
                "gpt4o",
                prereg["organizers"]["O4o"],
                config["gpt4o_upstream"],
                config["gpt4o_api_key_env"],
            ),
        ),
        start=1,
    ):
        directory = tmp_path / "proxy" / label
        directory.mkdir(parents=True)
        log = directory / "launch-001.jsonl"
        ready = directory / "launch-001.ready.json"
        stopped = directory / "launch-001.stopped.json"
        log.write_text("", encoding="utf-8")
        port = 19000 + index
        ready_payload = {
            "run_id": f"{fingerprint}:{label}:launch-001",
            "pid": 999000 + index,
            "port": port,
            "base_url": f"http://127.0.0.1:{port}/v1",
            "upstream": upstream,
            "log": str(log),
        }
        if label == "gpt55":
            ready_payload.update(
                {
                    "base_wrapper_sha256": runner.sha256_file(
                        ROOT / "scripts/gpt55_run_proxy.py"
                    ),
                    "controlled_wrapper_sha256": runner.sha256_file(
                        runner.GPT55_PROXY_SCRIPT
                    ),
                }
            )
        else:
            ready_payload.update(
                {
                    "schema": "nativemem.r301-exclusive-proxy-ready.v1",
                    "api_key_env": api_key_env,
                    "api_key_recorded": False,
                    "expected_requested_model": profile["requested_model"],
                    "accepted_actual_model_regex": profile[
                        "accepted_actual_model_regex"
                    ],
                    "wrapper_sha256": runner.sha256_file(runner.PROXY_SCRIPT),
                }
            )
        ready.write_text(
            json.dumps(ready_payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        window_start = directory / "launch-001.flex-window-start.json"
        if label == "gpt55":
            window_start.write_text(
                json.dumps({"contract": config["gpt55_gateway"]}) + "\n",
                encoding="utf-8",
            )
        stopped.write_text(
            json.dumps(
                {
                    "schema": "nativemem.r301-exclusive-proxy-stop.v1",
                    "status": "normal_stop",
                    "run_id": ready_payload["run_id"],
                    "pid": ready_payload["pid"],
                    "ready": str(ready),
                    "ready_sha256": runner.sha256_file(ready),
                    "log": str(log),
                    "log_sha256": runner.sha256_file(log),
                    "gateway_contract": (
                        config["gpt55_gateway"] if label == "gpt55" else None
                    ),
                    "provider_window_start": (
                        str(window_start) if label == "gpt55" else None
                    ),
                    "provider_window_start_sha256": (
                        runner.sha256_file(window_start)
                        if label == "gpt55"
                        else None
                    ),
                    "provider_window": (
                        {"schema": runner.flex_evidence.WINDOW_SCHEMA}
                        if label == "gpt55"
                        else None
                    ),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expected_logs[log.resolve()] = {
            "run_id": ready_payload["run_id"],
            "label": label,
        }
    monkeypatch.setattr(
        auditor.flex_evidence,
        "validate_recorded_contract",
        lambda record: record,
    )
    monkeypatch.setattr(
        auditor.flex_evidence,
        "audit_window",
        lambda window, consumer_records: {"status": "passed", "requests": 0},
    )
    monkeypatch.setattr(
        auditor.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr=""
        ),
    )

    assert auditor.audit_proxy_lifecycle(
        tmp_path,
        prereg,
        run_fingerprint=fingerprint,
        config=config,
    ) == (expected_logs, [{"status": "passed", "requests": 0}])
