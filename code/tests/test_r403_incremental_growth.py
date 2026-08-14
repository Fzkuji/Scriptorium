from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from scripts import audit_r403_incremental_growth as audit_mod
from scripts import run_r403_incremental_growth as run_mod
from src.evaluation.durable_model_ledger import HashChainLedger


ROOT = Path(__file__).resolve().parents[1]


def run_synthetic(output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/run_r403_incremental_growth.py",
            "--synthetic-sanity",
            "--output-dir",
            str(output),
            *extra,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.fixture(scope="module")
def synthetic_artifact() -> Path:
    (ROOT / "results").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r403-test-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        completed = run_synthetic(output)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        yield output


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_core_source_gate_is_narrow_and_frozen() -> None:
    assert run_mod.FROZEN_SOURCE_HASHES == audit_mod.FROZEN_SOURCE_HASHES
    assert set(run_mod.OBSERVED_BENCHMARK_RUNNERS).isdisjoint(run_mod.SOURCE_FILES)
    for relative, expected in run_mod.FROZEN_SOURCE_HASHES.items():
        assert run_mod.file_sha256(ROOT / relative) == expected
    sources = run_mod.source_hashes()
    assert set(sources) == set(run_mod.SOURCE_FILES)
    assert all(relative not in sources for relative in run_mod.OBSERVED_BENCHMARK_RUNNERS)


def test_synthetic_run_passes_independent_audit(
    synthetic_artifact: Path,
) -> None:
    report = audit_mod.audit(synthetic_artifact)
    assert report["status"] == "pass"
    assert report["mode"] == "synthetic_sanity"
    assert report["sample_count"] == 1
    assert report["model_call_count"] == 39
    assert report["failed_model_calls"] == 0
    assert report["operation_count"] == 34
    assert report["proxy"] == {
        "events": 0,
        "client_http_attempts": 0,
        "upstream_http_attempts": 0,
    }
    assert not (ROOT / "results" / ".r403-active-build.lock").exists()


def test_checkpoint_cohorts_and_costs_are_bound(
    synthetic_artifact: Path,
) -> None:
    sample = synthetic_artifact / "samples" / "sample-0"
    ledgers = read_jsonl(sample / "operations.jsonl")
    response_ids = [item["response_id"] for item in ledgers
                    if item["event"] == "model_call_finished"]
    assert len(response_ids) == len(set(response_ids))
    for percent in (10, 25, 50, 100):
        checkpoint = sample / "checkpoints" / f"checkpoint-{percent:03d}"
        manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
        inventory = read_jsonl(checkpoint / "question_inventory.jsonl")
        assert manifest["continuing_state_finalized"] is False
        assert manifest["checkpoint_copy_finalized"] is True
        assert manifest["cost"]["checkpoint_finalization_model"][
            "logical_model_calls"
        ] >= 1
        assert manifest["cost"]["cumulative_continuing_build_model"][
            "logical_model_calls"
        ] >= percent // 10
        assert all(item["main_qa_100_cohort"] is (percent == 100)
                   for item in inventory)

    at_10 = read_jsonl(
        sample / "checkpoints/checkpoint-010/question_inventory.jsonl"
    )
    at_25 = read_jsonl(
        sample / "checkpoints/checkpoint-025/question_inventory.jsonl"
    )
    at_50 = read_jsonl(
        sample / "checkpoints/checkpoint-050/question_inventory.jsonl"
    )
    at_100 = read_jsonl(
        sample / "checkpoints/checkpoint-100/question_inventory.jsonl"
    )
    assert at_10[0]["growth_eligible"] is True
    assert at_25[0]["old_fact_cohort"] is True
    assert at_50[1]["update_cohort"] is True
    assert at_100[2]["growth_eligible"] is True
    assert at_100[3]["growth_eligible"] is False
    assert at_100[3]["source_cohort_eligible"] is False
    assert at_100[3]["main_qa_100_cohort"] is True
    assert len(at_100) == 4
    assert at_100[0]["trace_id"] == (
        f"r403:{at_100[0]['question_id']}:checkpoint-100"
    )
    stages = at_100[0]["stage_evidence"]
    assert all(stages[name]["value"] is True for name in (
        "mapping_complete", "canonical_entry_source_exists",
        "maintenance_survival", "path_validity",
    ))
    assert all(
        stages[name] == {
            "status": "not_observed",
            "value": None,
            "reason": "task_results_not_attached",
        }
        for name in ("retrieval_reach", "source_resolution")
    )
    assert stages["m4_fields"] == {
        "gold_source_mapping_complete": True,
        "gold_source_in_canonical_entries": True,
        "gold_source_survived_maintenance": True,
        "gold_source_path_valid": True,
        "retrieval_reached_gold_source": None,
        "source_resolution_returned_gold_content": None,
    }
    assert all(stages["trace_ids"][name] for name in (
        "mapping", "canonical_entries", "maintenance", "paths",
    ))
    assert stages["trace_ids"]["retrieval"] == []
    assert stages["trace_ids"]["source_resolution"] == []
    excluded = at_100[3]["stage_evidence"]
    assert excluded["mapping_complete"]["value"] is False
    assert excluded["evidence_status"] == "excluded_incomplete_mapping"
    assert excluded["canonical_entry_source_exists"] == {
        "status": "not_applicable",
        "value": None,
        "reason": "source_recall_excluded",
        "present_source_ids": [],
        "missing_source_ids": [],
    }


def test_only_checkpoint_operations_finalize_copies(
    synthetic_artifact: Path,
) -> None:
    ledger = read_jsonl(
        synthetic_artifact / "samples/sample-0/operations.jsonl"
    )
    for item in ledger:
        if item["event"] == "operation_started":
            if item["kind"] == "session_model_maintenance":
                assert item["metadata"]["final"] is False
            if item["kind"] == "finalized_checkpoint_copy":
                assert item["metadata"]["continuing_state_finalized"] is False
                assert item["metadata"]["checkpoint_copy_finalized"] is True
        if (item["event"] == "operation_committed"
                and item["kind"] == "finalized_checkpoint_copy"):
            assert item["continuing_tree_before"] == item["continuing_tree_after"]


def test_safe_stop_resume_does_not_repeat_committed_operations() -> None:
    with tempfile.TemporaryDirectory(prefix="r403-resume-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        stopped = run_synthetic(output, "--stop-after-operations", "5")
        assert stopped.returncode == 3, stopped.stdout + stopped.stderr
        before = read_jsonl(output / "samples/sample-0/operations.jsonl")
        committed_before = [item["operation_id"] for item in before
                            if item["event"] == "operation_committed"]
        assert len(committed_before) == 5
        resumed = run_synthetic(output, "--resume")
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        after = read_jsonl(output / "samples/sample-0/operations.jsonl")
        committed_after = [item["operation_id"] for item in after
                           if item["event"] == "operation_committed"]
        assert committed_after[:5] == committed_before
        assert len(committed_after) == len(set(committed_after)) == 34
        assert audit_mod.audit(output)["status"] == "pass"
        assert not (ROOT / "results" / ".r403-active-build.lock").exists()


def test_resume_recovers_durable_unlinked_sample_manifest(
    synthetic_artifact: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="r403-promote-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        shutil.copytree(synthetic_artifact, output)
        manifest_path = output / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "partial_safe_stop"
        manifest["samples"]["0"] = {"status": "running"}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        resumed = run_synthetic(output, "--resume")
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert audit_mod.audit(output)["status"] == "pass"


def test_resume_rejects_orphan_operation_without_resending() -> None:
    with tempfile.TemporaryDirectory(prefix="r403-orphan-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        stopped = run_synthetic(output, "--stop-after-operations", "5")
        assert stopped.returncode == 3
        manifest = json.loads((output / "run_manifest.json").read_text())
        sample = output / "samples/sample-0"
        ledger = HashChainLedger(
            sample / "operations.jsonl",
            run_id=f"{manifest['run_fingerprint']}:sample-0",
        )
        ledger.append(
            "operation_started",
            operation_id="sample-00/orphan-operation",
            kind="test-orphan",
            operation_input_sha256="a" * 64,
            metadata={},
            continuing_tree_before=run_mod.tree_descriptor(
                sample / "continuing_memory"
            ),
        )
        resumed = run_synthetic(output, "--resume")
        assert resumed.returncode != 0
        assert "unresolved ledger state" in resumed.stderr
        assert not (ROOT / "results" / ".r403-active-build.lock").exists()


@pytest.mark.parametrize(
    "tamper",
    ["ledger", "response", "inventory", "config", "symlink", "hardlink"],
)
def test_auditor_detects_tampering(
    synthetic_artifact: Path, tamper: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="r403-tamper-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        shutil.copytree(synthetic_artifact, output)
        sample = output / "samples/sample-0"
        if tamper == "ledger":
            path = sample / "operations.jsonl"
            lines = path.read_text().splitlines()
            value = json.loads(lines[0])
            value["kind"] = "tampered"
            lines[0] = json.dumps(value, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n")
        elif tamper == "response":
            next((sample / "calls").glob("*.response.json")).unlink()
        elif tamper == "inventory":
            path = sample / "checkpoints/checkpoint-100/question_inventory.jsonl"
            lines = path.read_text().splitlines()
            value = json.loads(lines[0])
            value["main_qa_100_cohort"] = False
            lines[0] = json.dumps(value, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n")
        elif tamper == "config":
            path = output / "run_manifest.json"
            value = json.loads(path.read_text())
            value["config"]["environment"]["NATIVEMEM_CHUNK_TURNS"] = "7"
            path.write_text(json.dumps(value, indent=2) + "\n")
        elif tamper == "symlink":
            (output / "rogue").symlink_to(output / "run_manifest.json")
        else:
            os.link(output / "run_manifest.json", output / "rogue")
        with pytest.raises(audit_mod.AuditFailure):
            audit_mod.audit(output)


def test_unrelated_runner_observation_is_not_fingerprint_bound(
    synthetic_artifact: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="r403-observed-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        shutil.copytree(synthetic_artifact, output)
        path = output / "run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["environment_observations"][
            "scripts/run_v88_gpt55_beam.py"
        ] = "f" * 64
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        assert audit_mod.audit(output)["status"] == "pass"


def test_core_source_manifest_tamper_is_rejected(
    synthetic_artifact: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="r403-core-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        shutil.copytree(synthetic_artifact, output)
        path = output / "run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["source_hashes"]["src/v8_memory.py"] = "f" * 64
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        with pytest.raises(audit_mod.AuditFailure, match="source changed"):
            audit_mod.audit(output)


def test_cli_rejects_v9_and_partial_formal_sample_set() -> None:
    with tempfile.TemporaryDirectory(prefix="r403-cli-", dir=ROOT / "results") as raw:
        output = Path(raw) / "v9"
        environment = os.environ.copy()
        environment["NATIVEMEM_V9_PIPELINE"] = "two_tier"
        rejected = subprocess.run(
            [sys.executable, "scripts/run_r403_incremental_growth.py",
             "--synthetic-sanity", "--output-dir", str(output)],
            cwd=ROOT, env=environment, text=True, capture_output=True, check=False,
        )
        assert rejected.returncode != 0
        assert "two_tier is forbidden" in rejected.stderr
        assert not output.exists()

        formal = subprocess.run(
            [sys.executable, "scripts/run_r403_incremental_growth.py",
             "--allow-model-requests", "--samples", "0", "--output-dir",
             str(Path(raw) / "formal")],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        assert formal.returncode != 0
        assert "requires all samples 0-9" in formal.stderr
        assert not (Path(raw) / "formal").exists()


@pytest.mark.parametrize(
    "extra, expected",
    [
        (["--allow-model-requests"], "synthetic R403 forbids --allow-model-requests"),
        (["--gateway-root", "/private/tmp/unused-r403-gateway"], "synthetic R403 forbids --gateway-root"),
    ],
)
def test_synthetic_cli_rejects_formal_transport_flags(extra, expected) -> None:
    with tempfile.TemporaryDirectory(prefix="r403-gate-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        completed = run_synthetic(output, *extra)
        assert completed.returncode != 0
        assert expected in completed.stderr
        assert not output.exists()


def test_formal_proxy_auditor_reconciles_prefixes_and_response_meta(
    tmp_path: Path,
    fake_flex_provider,
) -> None:
    root = tmp_path / "artifact"
    proxy = root / "proxy"
    calls = root / "samples/sample-0/calls"
    proxy.mkdir(parents=True)
    calls.mkdir(parents=True)
    fingerprint = "a" * 64
    provider = fake_flex_provider.close_window("r403-proxy-audit")
    provider_response = provider["response"]
    assert provider_response is not None
    (root / "run_manifest.json").write_text(json.dumps({
        "run_fingerprint": fingerprint,
        "config": {"gateway_contract": provider["contract"]},
    }))
    logical_id = "run:sample-00/session-001/segment-001:call-0001"
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    event = {
        "run_id": f"{fingerprint}:launch-001",
        "status": "success",
        "http_status": 200,
        "requested_model": "gpt-5.5",
        "actual_model": "gpt-5.5",
        "response_id": provider_response["id"],
        "event_id": "event-1",
        "question_id": None,
        "logical_call_id": logical_id,
        "request_sha256": "b" * 64,
        "response_sha256": "c" * 64,
        "client_http_attempts": 1,
        "upstream_http_attempts": 2,
        "unsupported_parameters": ["max_output_tokens"],
        "usage": usage,
        "error": None,
    }
    event.update(provider["consumer_records"][0])
    log = proxy / "launch-001.jsonl"
    log.write_text(json.dumps(event, separators=(",", ":")) + "\n")
    ready = {
        "run_id": f"{fingerprint}:launch-001",
        "log": str(log.resolve()),
        "base_wrapper_sha256": audit_mod.file_sha256(
            ROOT / "scripts/gpt55_run_proxy.py"
        ),
        "controlled_wrapper_sha256": audit_mod.file_sha256(
            ROOT / "scripts/controlled_gpt55_run_proxy.py"
        ),
        "gateway_contract": provider["contract"],
        "provider_window": provider["window"],
    }
    (proxy / "launch-001.ready.json").write_text(json.dumps(ready))
    response_relative = "calls/response.response.json"
    (root / "samples/sample-0" / response_relative).write_text(json.dumps({
        "response": {
            "id": provider_response["id"],
            "model": "gpt-5.5",
            "usage": usage,
            "exclusive_proxy_meta": {
                "event_id": "event-1",
                "run_id": f"{fingerprint}:launch-001",
                "question_id": None,
                "logical_call_id": logical_id,
                "request_sha256": "b" * 64,
                "client_http_attempts": 1,
                "upstream_http_attempts": 2,
                "unsupported_parameters": ["max_output_tokens"],
            },
        }
    }))
    raw_log = log.read_bytes()
    start = {
        "operation_id": "sample-00/session-001/segment-001",
        "proxy_log_start": {
            "path": str(log.resolve()), "byte_offset": 0,
            "prefix_sha256": hashlib.sha256(b"").hexdigest(),
        },
    }
    terminal = {
        "event": "model_call_finished",
        "operation_id": start["operation_id"],
        "logical_call_id": logical_id,
        "response_path": response_relative,
        "response_id": provider_response["id"],
        "response_model": "gpt-5.5",
        "usage": usage,
        "proxy_evidence": {
            "mode": "exclusive_proxy",
            "events": [event],
            "client_http_attempts": 1,
            "upstream_http_attempts": 2,
            "unsupported_parameters": ["max_output_tokens"],
            "log_prefix": {
                "path": str(log.resolve()), "byte_offset": len(raw_log),
                "prefix_sha256": hashlib.sha256(raw_log).hexdigest(),
            },
        },
    }
    report = audit_mod.audit_proxy(
        root,
        mode="formal",
        starts_by_sample=[{logical_id: start}],
        terminals_by_sample=[{logical_id: terminal}],
    )
    assert report["events"] == 1
    assert report["client_http_attempts"] == 1
    assert report["upstream_http_attempts"] == 2
    assert report["flex_gateway_windows"][0]["status"] == "passed"

    unassigned = dict(event)
    unassigned["event_id"] = "event-2"
    unassigned["logical_call_id"] = "unassigned"
    with log.open("a") as handle:
        handle.write(json.dumps(unassigned, separators=(",", ":")) + "\n")
    with pytest.raises(
        audit_mod.AuditFailure,
        match="Flex provider window|unassigned proxy",
    ):
        audit_mod.audit_proxy(
            root,
            mode="formal",
            starts_by_sample=[{logical_id: start}],
            terminals_by_sample=[{logical_id: terminal}],
        )
