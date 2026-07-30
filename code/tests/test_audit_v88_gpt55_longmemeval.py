import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "audit_v88_gpt55_longmemeval.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_v88_gpt55_longmemeval", SCRIPT
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_official_longmemeval_s_inventory():
    records = MOD.expected_records()
    assert len(records) == 500
    assert sum(str(record["question_id"]).endswith("_abs") for record in records) == 30


def test_parse_time_rejects_invalid_value():
    with pytest.raises(MOD.AuditError, match="invalid timestamp"):
        MOD.parse_time("invalid")


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


def _experiment_metadata():
    config = {**MOD.FROZEN_CONFIG, "NATIVEMEM_V8_CONCURRENCY": "2"}
    return {
        "method": MOD.EXPECTED_METHOD,
        "models": MOD.EXPECTED_MODELS,
        "config": config,
        "code": {"git_commit": "test", "runner_sha256": "hash"},
        "request_audit": {"proxy_request_log": "/tmp/proxy.jsonl"},
    }


def test_frozen_config_covers_every_method_parameter():
    metadata = _experiment_metadata()
    assert MOD.validate_frozen_config(metadata["config"]) == metadata["config"]

    changed = dict(metadata["config"])
    changed["NATIVEMEM_V8_VERIFY"] = "off"
    with pytest.raises(MOD.AuditError, match="configuration mismatch"):
        MOD.validate_frozen_config(changed)

    missing = dict(metadata["config"])
    missing.pop("NATIVEMEM_TOPK")
    with pytest.raises(MOD.AuditError, match="config keys differ"):
        MOD.validate_frozen_config(missing)


def test_checkpoint_metadata_and_paths_are_bound_to_run(tmp_path):
    metadata = _experiment_metadata()
    item_dir = tmp_path / "items" / "0000_question"
    memory_dir = item_dir / "memory"
    memory_dir.mkdir(parents=True)
    checkpoint_path = item_dir / "checkpoint.json"
    checkpoint_path.write_text("{}", encoding="utf-8")
    checkpoint = {
        **metadata,
        "paths": {
            "item_dir": str(item_dir),
            "memory_dir": str(memory_dir),
            "checkpoint": str(checkpoint_path),
        },
    }

    assert MOD.validate_checkpoint_identity(
        checkpoint, metadata, checkpoint_path, tmp_path, 0
    ) == memory_dir

    changed = json.loads(json.dumps(checkpoint))
    changed["config"]["NATIVEMEM_TOPK"] = "10"
    with pytest.raises(MOD.AuditError, match="config differs"):
        MOD.validate_checkpoint_identity(
            changed, metadata, checkpoint_path, tmp_path, 0
        )

    memory_dir.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    memory_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(MOD.AuditError, match="memory directory.*symbolic-link"):
        MOD.validate_checkpoint_identity(
            checkpoint, metadata, checkpoint_path, tmp_path, 0
        )


def test_items_checkpoint_and_sibling_memory_symlinks_are_rejected(tmp_path):
    metadata = _experiment_metadata()
    outside = tmp_path / "outside-items"
    item_dir = outside / "0000_question"
    memory_dir = item_dir / "memory"
    memory_dir.mkdir(parents=True)
    checkpoint_path = item_dir / "checkpoint.json"
    checkpoint_path.write_text("{}", encoding="utf-8")
    (tmp_path / "items").symlink_to(outside, target_is_directory=True)
    checkpoint = {
        **metadata,
        "paths": {
            "item_dir": str(item_dir),
            "memory_dir": str(memory_dir),
            "checkpoint": str(checkpoint_path),
        },
    }
    with pytest.raises(MOD.AuditError, match="items directory.*symbolic-link"):
        MOD.validate_checkpoint_identity(
            checkpoint, metadata, tmp_path / "items/0000_question/checkpoint.json",
            tmp_path, 0,
        )

    (tmp_path / "items").unlink()
    own_item = tmp_path / "items/0000_question"
    sibling_memory = tmp_path / "items/0001_question/memory"
    sibling_memory.mkdir(parents=True)
    own_item.mkdir(parents=True)
    own_checkpoint = own_item / "checkpoint.json"
    own_checkpoint.write_text("{}", encoding="utf-8")
    (own_item / "memory").symlink_to(sibling_memory, target_is_directory=True)
    checkpoint["paths"] = {
        "item_dir": str(own_item),
        "memory_dir": str(own_item / "memory"),
        "checkpoint": str(own_checkpoint),
    }
    with pytest.raises(MOD.AuditError, match="memory directory.*symbolic-link"):
        MOD.validate_checkpoint_identity(
            checkpoint, metadata, own_checkpoint, tmp_path, 0
        )

    (own_item / "memory").unlink()
    real_checkpoint = own_item / "real-checkpoint.json"
    real_checkpoint.write_text("{}", encoding="utf-8")
    own_checkpoint.unlink()
    own_checkpoint.symlink_to(real_checkpoint)
    with pytest.raises(MOD.AuditError, match="checkpoint.*symbolic-link"):
        MOD.validate_checkpoint_identity(
            checkpoint, metadata, own_checkpoint, tmp_path, 0
        )


def test_aggregate_results_must_exactly_match_checkpoints(tmp_path):
    metadata = _experiment_metadata()
    checkpoint = {
        "dataset_index": 0,
        "question_id": "q0",
        "question_type": "multi-session",
        "question": "question",
        "gold": "gold",
        "answer": {"hypothesis": "answer"},
        "build": {"status": "complete"},
        "retrieval": {"status": "complete"},
        "models": metadata["models"],
        "config": metadata["config"],
    }
    expected_result = {
        "dataset_index": 0,
        "question_id": "q0",
        "question_type": "multi-session",
        "question": "question",
        "gold": "gold",
        "hypothesis": "answer",
        "build": checkpoint["build"],
        "retrieval": checkpoint["retrieval"],
        "models": checkpoint["models"],
        "config": checkpoint["config"],
    }
    (tmp_path / "results.json").write_text(
        json.dumps([expected_result]), encoding="utf-8"
    )
    (tmp_path / "hypotheses.jsonl").write_text(
        json.dumps({"question_id": "q0", "hypothesis": "answer"}) + "\n",
        encoding="utf-8",
    )
    qa_records = [{"question_id": "q0", "answer": "answer"}]
    MOD.validate_aggregate_outputs(tmp_path, [checkpoint], qa_records)

    expected_result["hypothesis"] = "stale"
    (tmp_path / "results.json").write_text(
        json.dumps([expected_result]), encoding="utf-8"
    )
    with pytest.raises(MOD.AuditError, match="results.json differs"):
        MOD.validate_aggregate_outputs(tmp_path, [checkpoint], qa_records)


def test_failed_cli_removes_stale_passed_audit(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text('{"status":"passed"}', encoding="utf-8")

    with pytest.raises(SystemExit):
        MOD.main([str(tmp_path)])

    assert not audit.exists()
