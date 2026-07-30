from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import audit_r207_path_control as audit_mod
from scripts import run_r207_path_control as run_mod
from src.evaluation.durable_model_ledger import HashChainLedger


ROOT = Path(__file__).resolve().parents[1]


def run_synthetic(output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/run_r207_path_control.py",
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


def test_preregistered_frozen_hashes_match_current_sources() -> None:
    assert run_mod.FROZEN_SOURCE_HASHES == audit_mod.FROZEN_SOURCE_HASHES
    assert set(run_mod.OBSERVED_BENCHMARK_RUNNERS).isdisjoint(
        run_mod.CORE_SOURCES
    )
    assert set(run_mod.CORE_SOURCES) == set(audit_mod.CORE_SOURCES)
    for relative, expected in run_mod.FROZEN_SOURCE_HASHES.items():
        assert run_mod.file_sha256(ROOT / relative) == expected
    assert set(run_mod.source_hashes()) == set(run_mod.CORE_SOURCES)


def event(topic: str = "person/pets") -> dict[str, object]:
    return {
        "when": "2023-05-07",
        "summary": "A adopted a dog",
        "summary_inline": "A adopted a dog [D1:1]",
        "dia_ids": ["D1:1"],
        "topic": topic,
    }


def fake_adapter(
    batches: list[list[dict[str, object]]],
    writes: list[object],
    write_error: Exception | None = None,
) -> SimpleNamespace:
    queue = [copy.deepcopy(batch) for batch in batches]

    def distill(*args, **kwargs):
        del args, kwargs
        return queue.pop(0)

    def write(memory_dir, events):
        del memory_dir
        writes.append(events)
        if write_error is not None:
            raise write_error

    v8 = SimpleNamespace(
        distill_events=distill,
        _sanitize_topic=lambda value: value,
    )
    return SimpleNamespace(
        v8_memory=v8,
        distill_events=distill,
        write_events=write,
    )


def test_capture_hook_preserves_original_write_and_separates_topic() -> None:
    writes: list[object] = []
    adapter = fake_adapter([[event()]], writes)
    original_distill = adapter.v8_memory.distill_events
    original_write = adapter.write_events
    recorder = run_mod.CaptureRecorder(0, lambda value: value, {"D1:1"})
    with run_mod.AdapterCaptureHook(adapter, recorder):
        events = adapter.v8_memory.distill_events(
            [("A", "A adopted a dog")], "2023-05-07", ["D1:1"]
        )
        adapter.write_events("unused", events)
    assert writes == [events]
    assert writes[0] is events
    assert adapter.v8_memory.distill_events is original_distill
    assert adapter.write_events is original_write
    assert "topic" not in recorder.entries[0]
    assert "topic_path" not in recorder.entries[0]
    assert recorder.original_placements == [{
        "entry_id": recorder.entries[0]["entry_id"],
        "original_topic": "person/pets",
        "topic_path": "person/pets",
    }]
    assert recorder.calls[0]["write_observed"] is True


def test_capture_distinguishes_equal_payload_objects_and_accepts_empty() -> None:
    writes: list[object] = []
    adapter = fake_adapter([[event()], [event()], []], writes)
    recorder = run_mod.CaptureRecorder(0, lambda value: value, {"D1:1"})
    with run_mod.AdapterCaptureHook(adapter, recorder):
        first = adapter.v8_memory.distill_events([], "2023-05-07", ["D1:1"])
        adapter.write_events("unused", first)
        second = adapter.v8_memory.distill_events([], "2023-05-07", ["D1:1"])
        assert first == second and first is not second
        adapter.write_events("unused", second)
        empty = adapter.v8_memory.distill_events([], "2023-05-07", ["D1:1"])
        assert empty == []
    assert len(writes) == 2
    assert len(recorder.entries) == 2
    assert recorder.entries[0]["entry_id"] != recorder.entries[1]["entry_id"]
    assert [call["write_observed"] for call in recorder.calls] == [True, True, False]


def test_capture_rejects_uncaptured_or_unwritten_batches() -> None:
    adapter = fake_adapter([[event()]], [])
    recorder = run_mod.CaptureRecorder(0, lambda value: value, {"D1:1"})
    with pytest.raises(RuntimeError, match="did not receive an observed"):
        with run_mod.AdapterCaptureHook(adapter, recorder):
            adapter.write_events("unused", [event()])

    adapter = fake_adapter([[event()]], [])
    recorder = run_mod.CaptureRecorder(0, lambda value: value, {"D1:1"})
    with pytest.raises(RuntimeError, match="never written"):
        with run_mod.AdapterCaptureHook(adapter, recorder):
            adapter.v8_memory.distill_events([], "2023-05-07", ["D1:1"])


def test_capture_rejects_reused_event_list_before_write() -> None:
    shared = [event()]

    def distill(*args, **kwargs):
        del args, kwargs
        return shared

    adapter = SimpleNamespace(
        v8_memory=SimpleNamespace(
            distill_events=distill, _sanitize_topic=lambda value: value
        ),
        distill_events=distill,
        write_events=lambda memory_dir, events: None,
    )
    recorder = run_mod.CaptureRecorder(0, lambda value: value, {"D1:1"})
    with pytest.raises(RuntimeError, match="reused an event-list object"):
        with run_mod.AdapterCaptureHook(adapter, recorder):
            adapter.v8_memory.distill_events([], "2023-05-07", ["D1:1"])
            adapter.v8_memory.distill_events([], "2023-05-07", ["D1:1"])


def test_capture_restores_hook_when_original_write_raises() -> None:
    adapter = fake_adapter([[event()]], [], RuntimeError("write failed"))
    original_distill = adapter.v8_memory.distill_events
    original_alias = adapter.distill_events
    original_write = adapter.write_events
    recorder = run_mod.CaptureRecorder(0, lambda value: value, {"D1:1"})
    with pytest.raises(RuntimeError, match="write failed"):
        with run_mod.AdapterCaptureHook(adapter, recorder):
            events = adapter.v8_memory.distill_events(
                [], "2023-05-07", ["D1:1"]
            )
            adapter.write_events("unused", events)
    assert adapter.v8_memory.distill_events is original_distill
    assert adapter.distill_events is original_alias
    assert adapter.write_events is original_write


def test_permutation_preserves_exact_path_contract() -> None:
    placements = [
        {"entry_id": "e1", "topic_path": "A/x"},
        {"entry_id": "e2", "topic_path": "A/x"},
        {"entry_id": "e3", "topic_path": "B/long-name"},
        {"entry_id": "e4", "topic_path": "C/y/z"},
    ]
    permuted, metadata = run_mod.deterministic_permutation(placements)
    bank = {"sample": 0, "entries_sha256": "a" * 64}
    left = run_mod.placement_payload("model_directed", bank, placements)
    right = run_mod.placement_payload(
        "deterministic_permutation", bank, permuted, metadata
    )
    run_mod.assert_placement_match(left, right)
    assert left["metrics"] == right["metrics"]
    assert sorted(item["topic_path"] for item in placements) == sorted(
        item["topic_path"] for item in permuted
    )
    assert metadata["fixed_points"] == min(
        sum(
            placements[index]["topic_path"]
            == placements[(index + offset) % len(placements)]["topic_path"]
            for index in range(len(placements))
        )
        for offset in range(1, len(placements))
    )


@pytest.fixture(scope="module")
def synthetic_artifact() -> Path:
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r207-test-", dir=results) as raw_parent:
        artifact = Path(raw_parent) / "artifact"
        completed = run_synthetic(artifact)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        yield artifact


def test_synthetic_cli_has_no_model_calls_and_passes_audit(
    synthetic_artifact: Path,
) -> None:
    report = audit_mod.audit(synthetic_artifact)
    assert report["status"] == "pass"
    assert report["sample_count"] == 1
    sample = synthetic_artifact / "samples" / "sample-0"
    bank = json.loads((sample / "canonical_entry_bank.json").read_text())
    original = json.loads((sample / "original_placement.json").read_text())
    model_calls = json.loads((sample / "model_calls.json").read_text())
    model = json.loads((sample / "placements/model_directed.json").read_text())
    control = json.loads(
        (sample / "placements/deterministic_permutation.json").read_text()
    )
    assert model_calls["call_count"] == 0
    assert model_calls["schema"] == "nativemem.r207-model-calls.v2"
    assert model_calls["ledger_event_count"] == 2
    assert model_calls["cost"]["logical_model_calls"] == 0
    assert report["question_trace_count"] == 0
    assert (sample / "question_traces.jsonl").read_text() == ""
    operations = [
        json.loads(line)
        for line in (sample / "operations.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in operations] == [
        "operation_started", "operation_committed",
    ]
    assert not (ROOT / "results" / ".r207-active-build.lock").exists()
    assert all("topic" not in entry and "topic_path" not in entry
               for entry in bank["entries"])
    assert all(item["original_topic"] for item in original["placements"])
    assert model["metrics"] == control["metrics"]
    assert model["metrics"]["path_string_total_bytes"] == control["metrics"][
        "path_string_total_bytes"
    ]


def test_cli_rejects_v9_pipeline_environment() -> None:
    with tempfile.TemporaryDirectory(prefix="r207-v9-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        environment = os.environ.copy()
        environment["NATIVEMEM_V9_PIPELINE"] = "two_tier"
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_r207_path_control.py",
                "--synthetic-sanity",
                "--output-dir",
                str(output),
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode != 0
        assert "two_tier is forbidden" in completed.stderr
        assert not output.exists()


@pytest.mark.parametrize(
    "tamper", ["edit", "delete", "duplicate", "path", "trace"]
)
def test_auditor_detects_content_tampering(
    synthetic_artifact: Path, tamper: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="r207-tamper-", dir=ROOT / "results") as raw:
        artifact = Path(raw) / "copy"
        shutil.copytree(synthetic_artifact, artifact)
        sample = artifact / "samples" / "sample-0"
        if tamper == "edit":
            path = sample / "canonical_entry_bank.json"
            payload = json.loads(path.read_text())
            payload["entries"][0]["summary"] = "edited"
            path.write_text(json.dumps(payload), encoding="utf-8")
        elif tamper == "delete":
            (sample / "placements/model_directed.json").unlink()
        elif tamper == "duplicate":
            source = next((sample / "conditions/model_directed").rglob("*.md"))
            shutil.copy2(source, source.parent / "duplicate.md")
        elif tamper == "path":
            path = sample / "placements/model_directed.json"
            payload = json.loads(path.read_text())
            payload["placements"][0]["topic_path"] = "tampered/path"
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            (sample / "question_traces.jsonl").write_text("{}\n")
        with pytest.raises(audit_mod.AuditFailure):
            audit_mod.audit(artifact)


def test_auditor_rejects_symlink_hardlink_and_stale_artifact(
    synthetic_artifact: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="r207-nodes-", dir=ROOT / "results") as raw:
        parent = Path(raw)
        for mode in ("symlink", "hardlink", "stale"):
            artifact = parent / mode
            shutil.copytree(synthetic_artifact, artifact)
            rogue = artifact / f"rogue-{mode}"
            source = artifact / "run_manifest.json"
            if mode == "symlink":
                rogue.symlink_to(source)
            elif mode == "hardlink":
                os.link(source, rogue)
            else:
                rogue.write_text("stale", encoding="utf-8")
            with pytest.raises(audit_mod.AuditFailure):
                audit_mod.audit(artifact)


def test_auditor_rejects_unicode_case_topic_collision() -> None:
    placements = [
        {"entry_id": "e1", "topic_path": "Topic/A"},
        {"entry_id": "e2", "topic_path": "topic/a"},
    ]
    payload = {
        "schema": audit_mod.PLACEMENT_SCHEMA,
        "condition": "model_directed",
        "sample": 0,
        "bank_entries_sha256": "b" * 64,
        "placements_sha256": audit_mod.value_sha256(placements),
        "metrics": audit_mod.placement_metrics(placements),
        "placements": placements,
    }
    with pytest.raises(audit_mod.AuditFailure, match="colliding topic paths"):
        audit_mod.audit_placement(
            payload, "model_directed", 0, "b" * 64, {"e1", "e2"}
        )


def test_resume_accepts_complete_artifact_without_new_work(
    synthetic_artifact: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="r207-complete-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        shutil.copytree(synthetic_artifact, output)
        completed = run_synthetic(output, "--resume")
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "already complete" in completed.stdout
        assert audit_mod.audit(output)["status"] == "pass"
        assert not (ROOT / "results" / ".r207-active-build.lock").exists()


@pytest.mark.parametrize("layout", ["final", "staging"])
def test_resume_recovers_durable_unlinked_sample(
    synthetic_artifact: Path, layout: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="r207-recover-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        shutil.copytree(synthetic_artifact, output)
        manifest_path = output / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "running"
        manifest["samples"]["0"] = {"status": "running"}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if layout == "staging":
            (output / "samples/sample-0").rename(
                output / "samples/.sample-0.staging-999999"
            )
        completed = run_synthetic(output, "--resume")
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert (output / "samples/sample-0").is_dir()
        assert not list((output / "samples").glob(".*.staging-*"))
        assert audit_mod.audit(output)["status"] == "pass"


def test_resume_rejects_orphan_staging_without_repeating_work(
    synthetic_artifact: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="r207-orphan-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        shutil.copytree(synthetic_artifact, output)
        manifest_path = output / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "running"
        manifest["samples"]["0"] = {"status": "running"}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        staging = output / "samples/.sample-0.staging-999999"
        (output / "samples/sample-0").rename(staging)
        ledger = HashChainLedger(
            staging / "operations.jsonl",
            run_id=f"{manifest['run_fingerprint']}:sample-0",
        )
        ledger.append(
            "operation_started",
            operation_id="sample-00/orphan",
            kind="test-orphan",
            operation_input_sha256="a" * 64,
            metadata={},
            continuing_tree_before=run_mod.tree_descriptor(
                staging / "captured_final_build"
            ),
        )
        completed = run_synthetic(output, "--resume")
        assert completed.returncode != 0
        assert "unresolved ledger state" in completed.stderr
        assert "use a new output root" in completed.stderr
        assert staging.is_dir()
        assert not (output / "samples/sample-0").exists()
        assert not (ROOT / "results" / ".r207-active-build.lock").exists()


def test_environment_observation_is_not_fingerprint_bound(
    synthetic_artifact: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="r207-observed-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        shutil.copytree(synthetic_artifact, output)
        manifest_path = output / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["environment_observations"][
            "scripts/run_v88_gpt55_beam.py"
        ] = "f" * 64
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        assert audit_mod.audit(output)["status"] == "pass"


def test_core_source_manifest_tamper_is_rejected(
    synthetic_artifact: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="r207-core-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        shutil.copytree(synthetic_artifact, output)
        manifest_path = output / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["core_source_hashes"]["src/v8_memory.py"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        with pytest.raises(audit_mod.AuditFailure, match="core source changed"):
            audit_mod.audit(output)


def test_cli_rejects_partial_formal_sample_set() -> None:
    with tempfile.TemporaryDirectory(prefix="r207-formal-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_r207_path_control.py",
                "--allow-model-requests",
                "--samples",
                "0",
                "--output-dir",
                str(output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode != 0
        assert "requires all samples 0-9" in completed.stderr
        assert not output.exists()


@pytest.mark.parametrize(
    "extra, expected",
    [
        (["--allow-model-requests"], "synthetic R207 forbids --allow-model-requests"),
        (["--gateway-root", "/private/tmp/unused-r207-gateway"], "synthetic R207 forbids --gateway-root"),
    ],
)
def test_synthetic_cli_rejects_formal_transport_flags(extra, expected) -> None:
    with tempfile.TemporaryDirectory(prefix="r207-gate-", dir=ROOT / "results") as raw:
        output = Path(raw) / "artifact"
        completed = run_synthetic(output, *extra)
        assert completed.returncode != 0
        assert expected in completed.stderr
        assert not output.exists()


def test_formal_proxy_auditor_reconciles_prefix_and_response_meta(
    tmp_path: Path,
    fake_flex_provider,
) -> None:
    root = tmp_path / "artifact"
    proxy = root / "proxy"
    calls = root / "samples/sample-0/calls"
    proxy.mkdir(parents=True)
    calls.mkdir(parents=True)
    fingerprint = "a" * 64
    provider = fake_flex_provider.close_window("r207-proxy-audit")
    provider_response = provider["response"]
    assert provider_response is not None
    (root / "run_manifest.json").write_text(json.dumps({
        "run_fingerprint": fingerprint,
        "config": {"gateway_contract": provider["contract"]},
    }))
    logical_id = f"{fingerprint}:sample-0:sample-00/capture-build:call-0001"
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
    (proxy / "launch-001.ready.json").write_text(json.dumps({
        "run_id": f"{fingerprint}:launch-001",
        "log": str(log.resolve()),
        "gateway_contract": provider["contract"],
        "provider_window": provider["window"],
        "base_wrapper_sha256": audit_mod.file_sha256(
            ROOT / "scripts/gpt55_run_proxy.py"
        ),
        "controlled_wrapper_sha256": audit_mod.file_sha256(
            ROOT / "scripts/controlled_gpt55_run_proxy.py"
        ),
    }))
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
        "operation_id": "sample-00/capture-build",
        "proxy_log_start": {
            "path": str(log.resolve()),
            "byte_offset": 0,
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
                "path": str(log.resolve()),
                "byte_offset": len(raw_log),
                "prefix_sha256": hashlib.sha256(raw_log).hexdigest(),
            },
        },
    }
    report = audit_mod.audit_proxy(
        root,
        mode="locomo",
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
            mode="locomo",
            starts_by_sample=[{logical_id: start}],
            terminals_by_sample=[{logical_id: terminal}],
        )


def test_question_trace_stages_are_observed_or_explicitly_unobserved(
    tmp_path: Path,
) -> None:
    build = tmp_path / "captured_final_build"
    conditions = tmp_path / "conditions"
    build.mkdir()
    (build / "memory.md").write_text("fact [D1:1]\n")
    for condition in run_mod.CONDITIONS:
        target = conditions / condition / "topic"
        target.mkdir(parents=True)
        (target / "events.md").write_text("fact [D1:1]\n")
    mapping = {
        "sample_index": 0,
        "sample_id": "conv-test",
        "question_index": 0,
        "question_id": "locomo:conv-test:q000",
        "normalized_source_ids": ["D1:1"],
        "source_recall_eligible": True,
        "source_recall_exclusion_reasons": [],
    }
    bank = {"entries": [{"dia_ids": ["D1:1"]}]}
    records = run_mod.question_trace_records(
        sample=0,
        sample_id="conv-test",
        mappings=[mapping],
        bank=bank,
        build_dir=build,
        conditions_dir=conditions,
    )
    assert records == audit_mod.expected_trace_records(
        sample=0,
        sample_id="conv-test",
        mappings=[mapping],
        entries=bank["entries"],
        sample_dir=tmp_path,
    )
    assert len(records) == 2
    for record in records:
        stages = record["stage_evidence"]
        assert record["trace_id"] == (
            f"r207:locomo:conv-test:q000:{record['condition']}"
        )
        assert all(stages[name]["value"] is True for name in (
            "mapping_complete", "canonical_entry_source_exists",
            "maintenance_survival", "path_validity",
        ))
        assert all(
            stages[name] == {
                "status": "not_observed",
                "value": None,
                "reason": "retrieval_results_not_attached",
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
