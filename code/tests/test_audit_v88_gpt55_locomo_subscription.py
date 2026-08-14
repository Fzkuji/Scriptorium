import copy
import json
from pathlib import Path

import pytest

from scripts import audit_v88_gpt55_locomo_subscription as auditor
from src.adapters.question_checkpoint import memory_sha256


def _import_fixture(tmp_path):
    source_root = tmp_path / "source"
    run_dir = tmp_path / "run"
    source_memory = source_root / "memory_sample0"
    imported_memory = run_dir / "memory_sample0"
    source_memory.mkdir(parents=True)
    imported_memory.mkdir(parents=True)
    (source_memory / "event.md").write_text("same memory", encoding="utf-8")
    (imported_memory / "event.md").write_text("same memory", encoding="utf-8")
    memory_hash = memory_sha256(source_memory)
    source_config = {
        "samples": list(range(10)),
        "model": "gpt-5.5",
        "provider": "chatgpt_pro_subscription",
        "proxy_base_url": "http://127.0.0.1:8199/v1",
        "proxy_request_log": str((source_root / "proxy_requests.jsonl").resolve()),
        "reasoning_effort": "none",
        "chunk_turns": 6,
        "segment": "fixed",
        "single_model_retrieve_answer": True,
        "calendar": True,
        "sample_workers": 1,
        "request_concurrency": 1,
        "retries": 1,
        "max_sessions": None,
        "questions_limit": None,
    }
    source_hashes = {
        name: "2" * 64 for name in auditor.SOURCE_IMPORT_REQUIRED_HASHES
    }
    source_hashes["benchmarks/locomo/data/locomo10.json"] = (
        auditor.common.sha256_file(auditor.common.DATA)
    )
    fingerprint = auditor.runner.fingerprint(source_config, source_hashes)
    source_manifest = {
        "benchmark": "locomo",
        "method": "NativeMem-v8.8+calendar",
        "backbone": "gpt-5.5",
        "provider": "chatgpt_pro_subscription",
        "formal_flex_result": False,
        "fingerprint": fingerprint,
        "output_dir": str(source_root.resolve()),
        "config": source_config,
        "source_hashes": source_hashes,
        "samples": {
            "0": {
                "status": "failed",
                "finished_at": "2026-07-14T00:01:00+00:00",
                "error": "returncode=-15,validation=missing",
            }
        },
        "status": "failed",
    }
    (source_root / "run_manifest.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )
    (source_root / "sample0.log").write_text(
        "\n".join([
            "[2026-07-14T00:00:00+00:00] attempt=1",
            f"[nativemem] building memory -> {source_memory.resolve()}",
            "[nativemem] build done: 1s",
            "  q0: 1 memories, 1 steps, 1s",
        ]) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "config": {
            "imported_sample0_memory": {
                "source_result_root": str(source_root.resolve()),
                "source_memory": str(source_memory.resolve()),
                "memory_sha256": memory_hash,
                "source_fingerprint": fingerprint,
            }
        }
    }
    current = {
        "question_id": "_build_stats",
        "build_time_s": 0.0,
        "num_memories": 1,
        "notes": f"reused existing memory dir: {imported_memory.resolve()}",
        "build_calls": None,
        "build_tokens_in": None,
        "build_tokens_out": None,
        "build_llm_time_s": None,
    }
    return run_dir, manifest, current


def test_imported_sample0_provenance_keeps_unavailable_counts_null(tmp_path):
    run_dir, manifest, current = _import_fixture(tmp_path)

    recovered = auditor.authenticate_imported_sample0_build(
        run_dir, manifest, current, {"memory_reused": True}
    )

    assert recovered["build_calls"] is None
    assert recovered["build_tokens_in"] is None
    accounting = recovered["build_accounting"]
    assert accounting["status"] == auditor.IMPORTED_BUILD_ACCOUNTING_STATUS
    assert accounting["build_calls"] is None
    assert accounting["source_manifest"]["status"] == "failed"
    assert accounting["source_memory"]["sha256"] == (
        accounting["imported_memory"]["sha256"]
    )


def test_imported_sample0_rejects_changed_memory_and_claimed_counts(tmp_path):
    run_dir, manifest, current = _import_fixture(tmp_path)
    (run_dir / "memory_sample0" / "event.md").write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(auditor.AuditError, match="hash differs"):
        auditor.authenticate_imported_sample0_build(
            run_dir, manifest, current, {"memory_reused": True}
        )

    run_dir, manifest, current = _import_fixture(tmp_path / "second")
    claimed = copy.deepcopy(current)
    claimed["build_calls"] = 1
    with pytest.raises(auditor.AuditError, match="unexpectedly claims"):
        auditor.authenticate_imported_sample0_build(
            run_dir, manifest, claimed, {"memory_reused": True}
        )


def test_imported_sample0_requires_recomputed_source_fingerprint_and_build_done(
    tmp_path,
):
    run_dir, manifest, current = _import_fixture(tmp_path)
    source_root = Path(
        manifest["config"]["imported_sample0_memory"]["source_result_root"]
    )
    source_manifest_path = source_root / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["config"]["chunk_turns"] = 7
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    with pytest.raises(auditor.AuditError, match="fingerprint does not recompute"):
        auditor.authenticate_imported_sample0_build(
            run_dir, manifest, current, {"memory_reused": True}
        )

    run_dir, manifest, current = _import_fixture(tmp_path / "missing-log")
    source_root = Path(
        manifest["config"]["imported_sample0_memory"]["source_result_root"]
    )
    (source_root / "sample0.log").unlink()
    with pytest.raises(auditor.AuditError, match="build log is missing"):
        auditor.authenticate_imported_sample0_build(
            run_dir, manifest, current, {"memory_reused": True}
        )


def test_imported_sample0_rejects_invalid_fingerprint_and_memory_count(tmp_path):
    run_dir, manifest, current = _import_fixture(tmp_path)
    manifest["config"]["imported_sample0_memory"]["source_fingerprint"] = "not-a-sha"
    with pytest.raises(auditor.AuditError, match="source fingerprint is invalid"):
        auditor.authenticate_imported_sample0_build(
            run_dir, manifest, current, {"memory_reused": True}
        )

    run_dir, manifest, current = _import_fixture(tmp_path / "count")
    current["num_memories"] = 999
    with pytest.raises(auditor.AuditError, match="num_memories differs"):
        auditor.authenticate_imported_sample0_build(
            run_dir, manifest, current, {"memory_reused": True}
        )


def test_normal_build_cannot_claim_imported_accounting(tmp_path):
    current = {
        "question_id": "_build_stats",
        "notes": "built fresh: 1 events, model=gpt-5.5",
        "build_calls": 1,
        "build_tokens_in": 1,
        "num_memories": 1,
        "build_accounting": {
            "status": auditor.IMPORTED_BUILD_ACCOUNTING_STATUS,
        },
    }
    with pytest.raises(auditor.AuditError, match="normal build carries"):
        auditor.resolve_build_record(
            tmp_path,
            {"config": {"imported_sample0_memory": None}},
            0,
            current,
            {},
        )


def _success_proxy_entry():
    return {
        "timestamp": "2026-07-14T00:01:00+00:00",
        "status": "success",
        "requested_model": "gpt-5.5",
        "actual_model": "gpt-5.5",
        "requested_reasoning_effort": "none",
        "actual_reasoning_effort": "none",
        "response_id": "response-1",
        "attempts": 1,
        "unsupported_parameters": [],
        "ignored_client_parameters": ["max_output_tokens"],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 2,
            "total_tokens": 13,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _error_proxy_entry():
    return {
        "timestamp": "2026-07-14T00:02:00+00:00",
        "status": "error",
        "requested_model": "gpt-5.5",
        "requested_reasoning_effort": "none",
        "attempts": 1,
        "unsupported_parameters": [],
        "ignored_client_parameters": ["max_output_tokens"],
        "error": "ReadTimeout",
        "http_status": None,
        "http_request_id": None,
        "actual_reasoning_effort": None,
        "reasoning_tokens": None,
        "reasoning_tokens_present": False,
    }


def _proxy_manifest():
    return {
        "created_at": "2026-07-14T00:00:00+00:00",
        "finished_at": "2026-07-14T00:03:00+00:00",
        "proxy_health": {"max_attempts": 2},
        "proxy_summary": {
            "successes": 1,
            "errors": 1,
            "prompt_tokens": 11,
            "cached_prompt_tokens": 3,
            "completion_tokens": 2,
            "reasoning_tokens": 0,
            "thinking_off_verified": True,
        },
    }


def _write_proxy_entries(tmp_path, entries):
    (tmp_path / "proxy_requests.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )


def test_proxy_audit_requires_explicit_success_reasoning_and_validates_errors(
    tmp_path,
):
    manifest = _proxy_manifest()
    success = _success_proxy_entry()
    error = _error_proxy_entry()
    _write_proxy_entries(tmp_path, [success, error])

    report = auditor.audit_proxy_window(tmp_path, manifest)
    assert report["thinking_off_evidence_scope"] == (
        "all_successful_outputs_explicit"
    )
    assert report["errors_without_observed_reasoning"] == 1
    assert report["ignored_client_parameters"] == ["max_output_tokens"]
    assert report["requested_output_limit_enforced"] is False

    missing_reasoning = copy.deepcopy(success)
    del missing_reasoning["usage"]["completion_tokens_details"]["reasoning_tokens"]
    _write_proxy_entries(tmp_path, [missing_reasoning, error])
    with pytest.raises(auditor.AuditError, match="success 0 differs"):
        auditor.audit_proxy_window(tmp_path, manifest)

    wrong_error = copy.deepcopy(error)
    wrong_error["requested_model"] = "another-model"
    _write_proxy_entries(tmp_path, [success, wrong_error])
    with pytest.raises(auditor.AuditError, match="request identity differs"):
        auditor.audit_proxy_window(tmp_path, manifest)

    incomplete_stream = copy.deepcopy(error)
    incomplete_stream["http_status"] = 200
    incomplete_stream["error"] = "ChunkedEncodingError: Response ended prematurely"
    _write_proxy_entries(tmp_path, [success, incomplete_stream])
    auditor.audit_proxy_window(tmp_path, manifest)

    invalid_success_status = copy.deepcopy(incomplete_stream)
    invalid_success_status["error"] = "ReadTimeout: response ended prematurely"
    _write_proxy_entries(tmp_path, [success, invalid_success_status])
    with pytest.raises(auditor.AuditError, match="error 1 evidence differs"):
        auditor.audit_proxy_window(tmp_path, manifest)


def test_proxy_audit_rejects_attempts_above_health_policy(tmp_path):
    success = _success_proxy_entry()
    success["attempts"] = 3
    _write_proxy_entries(tmp_path, [success, _error_proxy_entry()])
    with pytest.raises(auditor.AuditError, match="attempts differ"):
        auditor.audit_proxy_window(tmp_path, _proxy_manifest())


def test_manifest_requires_connect_timeout(tmp_path):
    run_dir = tmp_path.resolve()
    config = {
        "samples": list(range(10)),
        "model": "gpt-5.5",
        "provider": "chatgpt_pro_subscription",
        "proxy_base_url": "http://127.0.0.1:8199/v1",
        "proxy_request_log": str(run_dir / "proxy_requests.jsonl"),
        "reasoning_effort": "none",
        "chunk_turns": 6,
        "segment": "fixed",
        "single_model_retrieve_answer": True,
        "calendar": True,
        "sample_workers": 2,
        "request_concurrency": 1,
        "proxy_concurrency": 2,
        "retries": 2,
        "sdk_max_retries": 0,
        "sdk_http_timeout_s": 600,
        "imported_sample0_memory": None,
        "max_sessions": None,
        "questions_limit": None,
    }
    sources = auditor.runner.source_hashes()
    manifest = {
        "schema_version": 1,
        "benchmark": "locomo",
        "method": "NativeMem-v8.8+calendar",
        "backbone": "gpt-5.5",
        "provider": "chatgpt_pro_subscription",
        "formal_flex_result": False,
        "thinking_off": True,
        "output_dir": str(run_dir),
        "status": "complete",
        "config": config,
        "source_hashes": sources,
        "fingerprint": auditor.runner.fingerprint(config, sources),
        "proxy_health": {
            "status": "ok",
            "auth_readable": True,
            "max_concurrency": 2,
            "max_attempts": 2,
            "read_timeout_s": 150.0,
            "requested_reasoning_effort": "none",
            "request_log": str(run_dir / "proxy_requests.jsonl"),
            "code_sha256": sources["src/chatgpt_proxy.py"],
        },
    }
    with pytest.raises(auditor.AuditError, match="health contract differs"):
        auditor.validate_manifest(run_dir, manifest)
    manifest["proxy_health"]["connect_timeout_s"] = 20.0
    auditor.validate_manifest(run_dir, manifest)


def test_failed_cli_removes_stale_subscription_audit(tmp_path):
    audit_path = tmp_path / "audit.json"
    audit_path.write_text('{"status":"passed"}', encoding="utf-8")
    with pytest.raises(SystemExit):
        auditor.main([str(tmp_path)])
    assert not audit_path.exists()
