import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import scripts.evaluation.visible_token_budget as budget_module
from scripts.evaluation.visible_token_audit import _reconstruct_truncation
from scripts.evaluation.visible_token_audit import audit_visible_token_trace
from scripts.evaluation.visible_token_budget import (
    TokenCounter,
    VisibleTokenBudgetGate,
    copy_snapshot,
    snapshot_memory_path,
)


ROOT = Path(__file__).resolve().parents[1]


def _memory_snapshots(tmp_path):
    before = tmp_path / "memory-before"
    after = tmp_path / "memory-after"
    before.mkdir()
    after.mkdir()
    (before / "state.txt").write_text("before\n", encoding="utf-8")
    (after / "state.txt").write_text("after\n", encoding="utf-8")
    return before, after


def _finalized_gate(tmp_path, *, budget=12, overflow_policy="truncate"):
    before, after = _memory_snapshots(tmp_path)
    trace = tmp_path / "trace.jsonl"
    gate = VisibleTokenBudgetGate(
        trace_path=trace,
        run_id="test-run",
        configured_budget_tokens=budget,
        tokenizer=TokenCounter.utf8_bytes(requested_model="test-model"),
        memory_before=snapshot_memory_path(before),
        overflow_policy=overflow_policy,
    )
    return gate, trace, before, after


def test_gate_records_and_auditor_reconstructs_all_accounting(tmp_path):
    gate, trace, before, after = _finalized_gate(tmp_path, budget=12)
    first = gate.deliver_tool_result(
        event_id="tool-1",
        raw_text="abc",
        tool_name="read",
        tool_call_id="call-1",
    )
    source = gate.deliver_source_resolution(
        event_id="source-1",
        raw_text="dé",
        source_ids=["s1:t2"],
    )
    overflow = gate.deliver_tool_result(
        event_id="tool-2",
        raw_text="0123456789",
        tool_name="read",
    )
    rejected = gate.deliver_tool_result(
        event_id="tool-3",
        raw_text="not visible",
        tool_name="read",
    )
    manifest = gate.finalize(
        memory_after=snapshot_memory_path(after),
        actual_model_usage={
            "logical_calls": 1,
            "physical_attempts": 1,
            "responses": [{"actual_model": "provider/model", "response_id": "r1"}],
        },
    )

    assert first.delivered_text == "abc"
    assert source.delivered_tokens == 3
    assert overflow.decision == "truncated"
    assert overflow.delivered_text == "012345"
    assert rejected.decision == "rejected_budget_exhausted"
    assert rejected.delivered_text is None
    assert manifest["summary"]["cumulative_visible_tokens"] == 12
    assert manifest["summary"]["cumulative_source_resolution_tokens"] == 3
    assert manifest["memory_before_sha256"] != manifest["memory_after_sha256"]

    report = audit_visible_token_trace(
        trace,
        memory_before_path=before,
        memory_after_path=after,
    )
    assert report["audit_status"] == "pass", report["errors"]
    assert report["delivery_record_count"] == 4
    assert report["provider_exact"] is False


def test_reject_policy_ends_retrieval_without_spending_remainder(tmp_path):
    gate, trace, before, after = _finalized_gate(
        tmp_path,
        budget=4,
        overflow_policy="reject",
    )
    overflow = gate.deliver(
        event_id="oversize",
        kind="tool_result",
        raw_text="12345",
    )
    later = gate.deliver(
        event_id="later",
        kind="tool_result",
        raw_text="1",
    )
    gate.finalize(
        memory_after=snapshot_memory_path(after),
        actual_model_usage={"logical_calls": 0, "physical_attempts": 0},
    )

    assert overflow.decision == "rejected_overflow"
    assert later.decision == "rejected_budget_exhausted"
    assert gate.cumulative_visible_tokens == 0
    report = audit_visible_token_trace(
        trace,
        memory_before_path=before,
        memory_after_path=after,
    )
    assert report["audit_status"] == "pass", report["errors"]
    assert report["remaining_tokens"] == 4
    assert report["exhausted"] is True


def test_multibyte_prefix_that_cannot_fit_is_rejected(tmp_path):
    gate, trace, before, after = _finalized_gate(tmp_path, budget=1)
    result = gate.deliver(
        event_id="unicode",
        kind="tool_result",
        raw_text="é",
    )
    gate.finalize(
        memory_after=snapshot_memory_path(after),
        actual_model_usage={"logical_calls": 0, "physical_attempts": 0},
    )
    assert result.decision == "rejected_truncation_empty"
    assert result.delivered_text is None
    assert audit_visible_token_trace(
        trace,
        memory_before_path=before,
        memory_after_path=after,
    )["audit_status"] == "pass"


def test_fake_tokenizer_rewinds_until_decode_is_source_prefix():
    def encode(text):
        mapping = {
            "": [],
            "a": [1],
            "abc": [1, 2, 3],
            "�": [9, 9, 9],
        }
        return mapping[text]

    def decode(token_ids):
        mapping = {
            (): "",
            (1,): "a",
            (1, 2): "�",
            (1, 2, 3): "abc",
        }
        return mapping[tuple(token_ids)]

    tokenizer = TokenCounter(
        identity={
            "implementation": "boundary-test",
            "implementation_version": "1",
            "encoding_name": "boundary-test",
            "requested_model": "test",
            "resolution": "test",
            "provider_exact": False,
        },
        encoder=encode,
        decoder=decode,
    )
    assert tokenizer.truncate("abc", 2) == "a"
    assert _reconstruct_truncation("abc", 2, tokenizer) == "a"


def test_auditor_detects_delivered_text_tampering(tmp_path):
    gate, trace, _, after = _finalized_gate(tmp_path, budget=10)
    gate.deliver(
        event_id="tool",
        kind="tool_result",
        raw_text="abc",
    )
    gate.finalize(
        memory_after=snapshot_memory_path(after),
        actual_model_usage={"logical_calls": 0, "physical_attempts": 0},
    )
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    records[1]["delivered"]["text"] = "abd"
    trace.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    report = audit_visible_token_trace(trace)
    assert report["audit_status"] == "fail"
    assert any("trace_sha256" in error for error in report["errors"])
    assert any("record_hash" in error for error in report["errors"])
    assert any("delivered text differs" in error for error in report["errors"])


def test_auditor_recomputes_tokens_instead_of_trusting_recorded_count(tmp_path):
    gate, trace, _, after = _finalized_gate(tmp_path, budget=10)
    gate.deliver(
        event_id="tool",
        kind="tool_result",
        raw_text="abc",
    )
    gate.finalize(
        memory_after=snapshot_memory_path(after),
        actual_model_usage={"logical_calls": 0, "physical_attempts": 0},
    )
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    records[1]["raw"]["tokens"] = 1
    trace.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    report = audit_visible_token_trace(trace)
    assert report["audit_status"] == "fail"
    assert any("raw: tokens mismatch" in error for error in report["errors"])


def test_auditor_detects_recorded_payload_hash_tampering(tmp_path):
    gate, trace, _, after = _finalized_gate(tmp_path, budget=10)
    gate.deliver(
        event_id="tool",
        kind="tool_result",
        raw_text="abc",
    )
    gate.finalize(
        memory_after=snapshot_memory_path(after),
        actual_model_usage={"logical_calls": 0, "physical_attempts": 0},
    )
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    records[1]["delivered"]["sha256"] = "0" * 64
    trace.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    report = audit_visible_token_trace(trace)
    assert report["audit_status"] == "fail"
    assert any("delivered: sha256 mismatch" in error for error in report["errors"])
    assert any("record_hash mismatch" in error for error in report["errors"])


def test_auditor_detects_live_memory_mismatch(tmp_path):
    gate, trace, before, after = _finalized_gate(tmp_path, budget=10)
    gate.finalize(
        memory_after=snapshot_memory_path(after),
        actual_model_usage={"logical_calls": 0, "physical_attempts": 0},
    )
    (after / "state.txt").write_text("tampered\n", encoding="utf-8")

    report = audit_visible_token_trace(
        trace,
        memory_before_path=before,
        memory_after_path=after,
    )
    assert report["audit_status"] == "fail"
    assert any("memory_after: live sha256 mismatch" in error for error in report["errors"])


def test_incomplete_trace_cannot_pass_audit(tmp_path):
    gate, trace, _, _ = _finalized_gate(tmp_path, budget=10)
    gate.deliver(
        event_id="tool",
        kind="tool_result",
        raw_text="abc",
    )
    gate.close_incomplete()
    report = audit_visible_token_trace(trace)
    assert report["audit_status"] == "fail"
    assert "complete manifest is missing" in report["errors"]
    assert "required finalizer is missing" in report["errors"]


def test_event_ids_are_unique_and_delivery_stops_after_finalize(tmp_path):
    gate, _, _, after = _finalized_gate(tmp_path, budget=10)
    gate.deliver(event_id="same", kind="tool_result", raw_text="a")
    with pytest.raises(ValueError, match="unique"):
        gate.deliver(event_id="same", kind="tool_result", raw_text="b")
    gate.finalize(
        memory_after=snapshot_memory_path(after),
        actual_model_usage={"logical_calls": 0, "physical_attempts": 0},
    )
    with pytest.raises(RuntimeError, match="finalized"):
        gate.deliver(event_id="new", kind="tool_result", raw_text="c")


def test_byte_fallback_identity_is_explicitly_non_provider_exact():
    tokenizer = TokenCounter.utf8_bytes(
        requested_model="gpt-5.5",
        fallback_reason="test",
    )
    assert tokenizer.count("Aé") == 3
    assert tokenizer.identity["resolution"] == "explicit_byte_fallback"
    assert tokenizer.identity["fallback_reason"] == "test"
    assert tokenizer.identity["provider_exact"] is False
    assert "not a provider" in tokenizer.identity["counting_note"]


def test_sanity_command_generates_audited_minimal_artifacts(tmp_path):
    output = tmp_path / "sanity"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "token_budget" / "run_visible_token_budget_sanity.py"),
            "--output-dir",
            str(output),
            "--budget-tokens",
            "32",
            "--model",
            "gpt-5.5",
            "--allow-byte-fallback",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    summary = json.loads(result.stdout)
    assert summary["audit_status"] == "pass"
    assert summary["decisions"] == [
        "delivered",
        "delivered",
        "truncated",
        "rejected_budget_exhausted",
    ]
    assert summary["source_resolution_tokens"] > 0
    assert summary["cumulative_visible_tokens"] == 32
    assert json.loads((output / "audit.json").read_text())["audit_status"] == "pass"


def test_manifest_cannot_alias_derived_lock_path(tmp_path):
    before, _ = _memory_snapshots(tmp_path)
    trace = tmp_path / "trace.jsonl"
    lock = trace.with_suffix(trace.suffix + ".lock")
    with pytest.raises(ValueError, match="path collision"):
        VisibleTokenBudgetGate(
            trace_path=trace,
            manifest_path=lock,
            run_id="collision",
            configured_budget_tokens=10,
            tokenizer=TokenCounter.utf8_bytes(requested_model="test"),
            memory_before=snapshot_memory_path(before),
        )
    assert not trace.exists()
    assert not lock.exists()


def test_output_identity_rejects_unicode_case_alias(tmp_path):
    before, _ = _memory_snapshots(tmp_path)
    with pytest.raises(ValueError, match="path collision"):
        VisibleTokenBudgetGate(
            trace_path=tmp_path / "Café.jsonl",
            manifest_path=tmp_path / "café.jsonl",
            run_id="unicode-collision",
            configured_budget_tokens=10,
            tokenizer=TokenCounter.utf8_bytes(requested_model="test"),
            memory_before=snapshot_memory_path(before),
        )


def test_output_parent_symlink_is_rejected(tmp_path):
    before, _ = _memory_snapshots(tmp_path)
    real_parent = tmp_path / "real-output"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias-output"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink component"):
        VisibleTokenBudgetGate(
            trace_path=alias_parent / "trace.jsonl",
            run_id="symlink-parent",
            configured_budget_tokens=10,
            tokenizer=TokenCounter.utf8_bytes(requested_model="test"),
            memory_before=snapshot_memory_path(before),
        )


def test_trace_open_failure_releases_owned_lock(tmp_path, monkeypatch):
    before, _ = _memory_snapshots(tmp_path)
    trace = tmp_path / "trace.jsonl"
    lock = trace.with_suffix(trace.suffix + ".lock")
    original_open = Path.open

    def failing_open(path, *args, **kwargs):
        if path == trace and args and args[0] == "x":
            raise OSError("injected trace open failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="injected"):
        VisibleTokenBudgetGate(
            trace_path=trace,
            run_id="open-failure",
            configured_budget_tokens=10,
            tokenizer=TokenCounter.utf8_bytes(requested_model="test"),
            memory_before=snapshot_memory_path(before),
        )
    assert not lock.exists()
    assert not trace.exists()


def test_manifest_write_failure_releases_lock_and_remains_incomplete(
    tmp_path,
    monkeypatch,
):
    gate, trace, _, after = _finalized_gate(tmp_path, budget=10)
    lock = trace.with_suffix(trace.suffix + ".lock")

    def fail_manifest(*args, **kwargs):
        raise OSError("injected manifest failure")

    monkeypatch.setattr(budget_module, "_atomic_write_json", fail_manifest)
    with pytest.raises(OSError, match="injected manifest"):
        gate.finalize(
            memory_after=snapshot_memory_path(after),
            actual_model_usage={"logical_calls": 0, "physical_attempts": 0},
        )
    assert not lock.exists()
    assert not trace.with_suffix(trace.suffix + ".manifest.json").exists()
    report = audit_visible_token_trace(trace)
    assert report["audit_status"] == "fail"
    assert "complete manifest is missing" in report["errors"]


def test_manifest_publication_does_not_overwrite_racing_path(tmp_path):
    gate, trace, _, after = _finalized_gate(tmp_path, budget=10)
    manifest = trace.with_suffix(trace.suffix + ".manifest.json")
    manifest.write_text("external\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        gate.finalize(
            memory_after=snapshot_memory_path(after),
            actual_model_usage={"logical_calls": 0, "physical_attempts": 0},
        )
    assert manifest.read_text(encoding="utf-8") == "external\n"
    assert not trace.with_suffix(trace.suffix + ".lock").exists()


def test_copy_snapshot_rejects_source_tree_symlink_before_copy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (source / "external-link").symlink_to(outside)
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="contains symlink"):
        copy_snapshot(source, destination)
    assert not destination.exists()


def test_published_private_tmp_sanity_command_shape_runs():
    private_tmp = Path("/private/tmp")
    if not private_tmp.is_dir():
        pytest.skip("/private/tmp is not available on this platform")
    with tempfile.TemporaryDirectory(prefix="r004-command-", dir=private_tmp) as parent:
        output = Path(parent) / "sanity"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "token_budget" / "run_visible_token_budget_sanity.py"),
                "--output-dir",
                str(output),
                "--budget-tokens",
                "32",
                "--model",
                "gpt-5.5",
                "--allow-byte-fallback",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        assert json.loads(result.stdout)["audit_status"] == "pass"


def test_cli_help_examples_use_private_tmp_only():
    for script in (
        "run_visible_token_budget_sanity.py",
        "audit_visible_token_budget.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "token_budget" / script), "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "/private/tmp/nativemem-r004" in result.stdout
        assert " --output-dir /tmp/" not in result.stdout
