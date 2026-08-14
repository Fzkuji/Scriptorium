import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "audit_v88_gpt55_locomo.py"
SPEC = importlib.util.spec_from_file_location("audit_v88_gpt55_locomo", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_expected_records_match_official_full_counts():
    expected = MOD.expected_records()
    assert len(expected) == 1986
    counts = {}
    for record in expected.values():
        counts[record["category"]] = counts.get(record["category"], 0) + 1
    assert counts == MOD.EXPECTED_CATEGORY_COUNTS


def test_parse_time_rejects_invalid_timestamp():
    with pytest.raises(MOD.AuditError, match="invalid timestamp"):
        MOD.parse_time("not-a-date")


def test_reused_memory_requires_and_recovers_prior_build_accounting(tmp_path):
    partials = tmp_path / "partials"
    partials.mkdir()
    prior = [{
        "question_id": "_build_stats",
        "notes": "built fresh: 10 events, model=gpt-5.5",
        "build_calls": 3,
        "build_tokens_in": 100,
        "num_memories": 2,
    }]
    path = partials / "sample0_questions.json.20260714"
    path.write_text(json.dumps(prior))
    current = {
        "question_id": "_build_stats",
        "notes": f"reused existing memory dir: {tmp_path / 'memory_sample0'}",
        "num_memories": 4,
    }

    recovered = MOD.resolve_build_record(
        tmp_path, 0, current, {"memory_reused": True}
    )

    assert recovered["build_calls"] == 3
    assert recovered["num_memories"] == 4
    assert recovered["recovered_from"].startswith("partials/")

    with pytest.raises(MOD.AuditError, match="manifest evidence"):
        MOD.resolve_build_record(tmp_path, 0, current, {"memory_reused": False})


def test_source_hashes_must_be_nonempty_and_cover_runner_inventory(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(MOD, "ROOT", tmp_path)
    with pytest.raises(MOD.AuditError, match="non-empty"):
        MOD.validate_source_hashes({"source_hashes": {}})

    partial = {next(iter(MOD.REQUIRED_SOURCE_PATHS)): "0" * 64}
    with pytest.raises(MOD.AuditError, match="omits required files"):
        MOD.validate_source_hashes({"source_hashes": partial})

    source_hashes = {}
    for relative in MOD.REQUIRED_SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        source_hashes[relative] = MOD.sha256_file(path)
    MOD.validate_source_hashes({"source_hashes": source_hashes})


def test_proxy_window_ignores_partial_tail_and_rejects_bad_complete_line(tmp_path):
    path = tmp_path / "proxy.jsonl"
    entry = {
        "timestamp": "2026-07-14T00:00:00+00:00",
        "status": "success",
        "actual_model": "gpt-5.5",
    }
    complete = json.dumps(entry).encode("utf-8") + b"\n"
    path.write_bytes(complete + b'{"error":"\xe4')
    assert MOD.load_proxy_window(
        path,
        "2026-07-13T00:00:00+00:00",
        "2026-07-15T00:00:00+00:00",
    ) == [entry]

    path.write_bytes(complete + b'{"broken":\n')
    with pytest.raises(MOD.AuditError, match="invalid proxy log line 2"):
        MOD.load_proxy_window(
            path,
            "2026-07-13T00:00:00+00:00",
            "2026-07-15T00:00:00+00:00",
        )


def test_failed_cli_removes_stale_passed_audit(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text('{"status":"passed"}', encoding="utf-8")

    with pytest.raises(SystemExit):
        MOD.main([str(tmp_path)])

    assert not audit.exists()
