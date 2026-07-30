import fcntl
import json
from pathlib import Path

import pytest

from scripts import merge_v88_gpt55_longmemeval as MOD



def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def make_dataset(path, total=4):
    data = []
    for index in range(total):
        sessions = [
            [{"role": "user", "content": f"item {index} session {session}"}]
            for session in range(20)
        ]
        data.append({
            "question_id": f"q{index}",
            "question_type": "temporal-reasoning",
            "question": f"question {index}?",
            "answer": f"answer {index}",
            "question_date": "2023/01/21",
            "answer_session_ids": [f"s{index}-0"],
            "haystack_sessions": sessions,
            "haystack_dates": [
                f"2023/01/{session + 1:02d}" for session in range(20)
            ],
            "haystack_session_ids": [
                f"s{index}-{session}" for session in range(20)
            ],
        })
    write_json(path, data)
    return data


def code_inventory():
    return {
        "git_commit": MOD.current_git_head(),
        **{
            key: MOD.sha256_file(path)
            for key, path in MOD.CODE_PATHS.items()
        },
    }


def common_manifest(
    run_dir, data_path, total, invocation, fake_flex_provider
):
    manifest = {
        "schema_version": MOD.runner.SCHEMA_VERSION,
        "benchmark": "LongMemEval-S",
        "dataset_path": str(data_path.resolve()),
        "dataset_sha256": MOD.sha256_file(data_path),
        "dataset_items": total,
        "method": {
            "name": "NativeMem",
            "version": "v8.8+calendar",
            "calendar": True,
            "single_model_retrieve_answer": True,
        },
        "models": {
            "builder": "gpt-5.5",
            "retriever": "gpt-5.5",
            "answerer": "gpt-5.5",
            "provider": "openai_api_flex_via_exclusive_child_proxy",
            "gateway_root": str(fake_flex_provider.root),
        },
        "config": MOD.runner.method_config(2),
        "code": code_inventory(),
        "request_audit": {
            "mode": "bounded_gateway_segments_with_exclusive_child_proxy",
            "gateway_root": str(fake_flex_provider.root),
            "explicit_model_request_authorization": True,
        },
        "output_dir": str(run_dir),
        "created_at": "2026-07-14T00:00:00+00:00",
        "completed": 1,
        "checkpoint_counts": {"complete": 1},
        "status": "partial",
        "invocations": [invocation],
    }
    fake_flex_provider.attach(run_dir, manifest, f"lme-{run_dir.name}")
    return manifest


def make_complete_item(run_dir, manifest, reference, index):
    paths = MOD.runner.item_paths(run_dir, index, reference["question_id"])
    memory_file = paths["memory_dir"] / "topics" / "memory.md"
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    memory_file.write_text(f"memory {index}\n", encoding="utf-8")
    stats = MOD.runner.memory_stats(paths["memory_dir"])
    checkpoint = {
        "schema_version": MOD.runner.SCHEMA_VERSION,
        "status": "complete",
        "dataset_index": index,
        "question_id": reference["question_id"],
        "question_type": reference["question_type"],
        "question": reference["question"],
        "gold": reference["answer"],
        "question_date": "2023-01-21",
        "question_date_raw": reference["question_date"],
        "answer_session_ids": reference["answer_session_ids"],
        "input": {
            "sessions": len(reference["haystack_sessions"]),
            "turns": sum(len(value) for value in reference["haystack_sessions"]),
            "empty_source_turns": 0,
            "source_session_ids": reference["haystack_session_ids"],
            "source_dates": reference["haystack_dates"],
        },
        "method": manifest["method"],
        "models": manifest["models"],
        "config": manifest["config"],
        "code": manifest["code"],
        "request_audit": manifest["request_audit"],
        "paths": {
            "item_dir": str(paths["item_dir"].resolve()),
            "memory_dir": str(paths["memory_dir"].resolve()),
            "checkpoint": str(paths["checkpoint"].resolve()),
        },
        "build": {
            "status": "complete",
            "events": 1,
            "usage": {"calls": 1, "tokens_in": 10, "tokens_out": 2},
            **stats,
        },
        "retrieval": {
            "status": "complete",
            "steps": 1,
            "usage": {"calls": 1, "tokens_in": 5, "tokens_out": 1},
        },
        "answer": {
            "status": "complete",
            "hypothesis": f"answer {index}",
            "model": "gpt-5.5",
        },
    }
    write_json(paths["checkpoint"], checkpoint)


def make_partial_item(run_dir, manifest, reference, index):
    paths = MOD.runner.item_paths(run_dir, index, reference["question_id"])
    paths["item_dir"].mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": MOD.runner.SCHEMA_VERSION,
        "status": "building",
        "dataset_index": index,
        "question_id": reference["question_id"],
        "method": manifest["method"],
        "models": manifest["models"],
        "config": manifest["config"],
        "code": manifest["code"],
        "request_audit": manifest["request_audit"],
        "paths": {
            "item_dir": str(paths["item_dir"].resolve()),
            "memory_dir": str(paths["memory_dir"].resolve()),
            "checkpoint": str(paths["checkpoint"].resolve()),
        },
        "build": {"status": "running"},
    }
    write_json(paths["checkpoint"], checkpoint)


def invocation(start, limit, end):
    return {
        "started_at": "2026-07-14T00:00:00+00:00",
        "start": start,
        "limit": limit,
        "indices": [start, end],
        "resume": False,
    }


def make_four_shards(tmp_path, fake_flex_provider):
    data_path = (tmp_path / "data.json").resolve()
    data = make_dataset(data_path)
    definitions = [
        (0, invocation(0, None, 3), 1),
        (1, invocation(1, 1, 1), None),
        (2, invocation(2, None, 3), 3),
        (3, invocation(3, 1, 3), None),
    ]
    specs = []
    for index, declared, partial_index in definitions:
        run_dir = (tmp_path / f"shard{index}").resolve()
        run_dir.mkdir()
        (run_dir / ".launcher.lock").touch()
        manifest = common_manifest(
            run_dir, data_path, len(data), declared, fake_flex_provider
        )
        make_complete_item(run_dir, manifest, data[index], index)
        if partial_index is not None:
            make_partial_item(run_dir, manifest, data[partial_index], partial_index)
            manifest["checkpoint_counts"]["building"] = 1
        write_json(run_dir / "run_manifest.json", manifest)
        specs.append(MOD.ShardSpec(run_dir, index, 1))
    return data_path, specs


def test_merge_four_responsibility_ranges_rewrites_paths_and_is_idempotent(
        tmp_path, monkeypatch, fake_flex_provider
):
    data_path, specs = make_four_shards(tmp_path, fake_flex_provider)
    output_dir = (tmp_path / "merged").resolve()

    def fail_if_backend_is_loaded():
        raise AssertionError("merger must not initialize the model backend")

    monkeypatch.setattr(MOD.runner, "load_backend", fail_if_backend_is_loaded)
    report = MOD.merge_shards(
        specs, output_dir, data_path, expected_total=4
    )
    assert report["status"] == "complete"
    assert report["copied"] == 4 and report["reused"] == 0
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["completed"] == 4
    assert manifest["checkpoint_counts"] == {"complete": 4}
    assert manifest["status"] == "complete"
    assert manifest["merge"]["no_model_requests"] is True
    assert manifest["merge"]["coverage"] == {
        "start": 0, "limit": 4, "indices": [0, 3]
    }
    ignored = [
        item
        for source in manifest["merge"]["sources"]
        for item in source["ignored_partials"]
    ]
    assert [(item["dataset_index"], item["status"], item["copied"])
            for item in ignored] == [(1, "building", False), (3, "building", False)]
    assert len(json.loads((output_dir / "results.json").read_text())) == 4
    assert len((output_dir / "hypotheses.jsonl").read_text().splitlines()) == 4
    for index in range(4):
        checkpoint_path = next(
            (output_dir / "items").glob(f"{index:04d}_*/checkpoint.json")
        )
        checkpoint = json.loads(checkpoint_path.read_text())
        assert checkpoint["paths"]["checkpoint"] == str(checkpoint_path.resolve())
        assert str(output_dir) in checkpoint["paths"]["memory_dir"]

    original_manifest = (output_dir / "run_manifest.json").read_bytes()
    original_results = (output_dir / "results.json").read_bytes()
    resumed = MOD.merge_shards(
        specs, output_dir, data_path, resume=True, expected_total=4
    )
    assert resumed["copied"] == 0 and resumed["reused"] == 4
    assert (output_dir / "run_manifest.json").read_bytes() == original_manifest
    assert (output_dir / "results.json").read_bytes() == original_results


def test_resume_recovers_a_missing_destination_item(tmp_path, fake_flex_provider):
    data_path, specs = make_four_shards(tmp_path, fake_flex_provider)
    output_dir = (tmp_path / "merged").resolve()
    MOD.merge_shards(specs, output_dir, data_path, expected_total=4)
    missing = next((output_dir / "items").glob("0002_*"))
    for path in sorted(missing.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    missing.rmdir()
    (output_dir / "results.json").unlink()

    report = MOD.merge_shards(
        specs, output_dir, data_path, resume=True, expected_total=4
    )
    assert report["copied"] == 1 and report["reused"] == 3
    assert len(json.loads((output_dir / "results.json").read_text())) == 4


def test_resume_rejects_untracked_root_and_items_artifacts(
    tmp_path, fake_flex_provider
):
    data_path, specs = make_four_shards(tmp_path, fake_flex_provider)
    output_dir = (tmp_path / "merged").resolve()
    MOD.merge_shards(specs, output_dir, data_path, expected_total=4)
    (output_dir / "unexpected-untracked.bin").write_bytes(b"unexpected")
    with pytest.raises(MOD.MergeError, match="unexpected root artifacts"):
        MOD.merge_shards(
            specs, output_dir, data_path, resume=True, expected_total=4
        )

    (output_dir / "unexpected-untracked.bin").unlink()
    (output_dir / "items" / ".unexpected").write_text("unexpected", encoding="utf-8")
    with pytest.raises(MOD.MergeError, match="unexpected artifacts"):
        MOD.merge_shards(
            specs, output_dir, data_path, resume=True, expected_total=4
        )


def test_existing_output_requires_resume(tmp_path, fake_flex_provider):
    data_path, specs = make_four_shards(tmp_path, fake_flex_provider)
    output_dir = (tmp_path / "merged").resolve()
    MOD.merge_shards(specs, output_dir, data_path, expected_total=4)
    with pytest.raises(MOD.MergeError, match="already exists"):
        MOD.merge_shards(specs, output_dir, data_path, expected_total=4)


def test_completed_item_outside_responsibility_is_rejected(
    tmp_path, fake_flex_provider
):
    data_path, specs = make_four_shards(tmp_path, fake_flex_provider)
    source = specs[0].run_dir
    partial_path = next((source / "items").glob("0001_*/checkpoint.json"))
    partial = json.loads(partial_path.read_text())
    partial["status"] = "complete"
    write_json(partial_path, partial)
    with pytest.raises(MOD.MergeError, match="completed item 1 outside"):
        MOD.merge_shards(
            specs, tmp_path / "merged", data_path, expected_total=4
        )


def test_partial_directory_without_checkpoint_is_recorded_and_not_copied(
    tmp_path, fake_flex_provider
):
    data_path, specs = make_four_shards(tmp_path, fake_flex_provider)
    source = specs[0].run_dir
    partial_path = next((source / "items").glob("0001_*/checkpoint.json"))
    partial_path.unlink()
    output_dir = (tmp_path / "merged").resolve()
    MOD.merge_shards(specs, output_dir, data_path, expected_total=4)
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    ignored = manifest["merge"]["sources"][0]["ignored_partials"]
    assert ignored[0]["dataset_index"] == 1
    assert ignored[0]["status"] == "missing_checkpoint"
    assert ignored[0]["checkpoint"] is None
    assert len(list((output_dir / "items").glob("0001_*"))) == 1


def test_declared_invocation_bounds_are_validated(tmp_path, fake_flex_provider):
    data_path, specs = make_four_shards(tmp_path, fake_flex_provider)
    manifest_path = specs[1].run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["invocations"][0]["indices"] = [1, 2]
    write_json(manifest_path, manifest)
    with pytest.raises(MOD.MergeError, match="indices differ"):
        MOD.merge_shards(
            specs, tmp_path / "merged", data_path, expected_total=4
        )


def test_responsibility_ranges_must_cover_without_overlap(
    tmp_path, fake_flex_provider
):
    data_path, specs = make_four_shards(tmp_path, fake_flex_provider)
    bad = [*specs[:2], MOD.ShardSpec(specs[2].run_dir, 1, 2), specs[3]]
    with pytest.raises(MOD.MergeError, match="overlap"):
        MOD.merge_shards(
            bad, tmp_path / "merged", data_path, expected_total=4
        )


def test_active_source_launcher_lock_is_rejected(tmp_path, fake_flex_provider):
    data_path, specs = make_four_shards(tmp_path, fake_flex_provider)
    lock = (specs[0].run_dir / ".launcher.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(MOD.MergeError, match="active or locked"):
            MOD.merge_shards(
                specs, tmp_path / "merged", data_path, expected_total=4
            )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def test_parse_repeated_input_spec_uses_inclusive_range():
    parsed = MOD.parse_input_spec("/tmp/lme-shard:125-249")
    assert parsed == MOD.ShardSpec(Path("/tmp/lme-shard"), 125, 125)
    with pytest.raises(Exception, match="DIR:START-END"):
        MOD.parse_input_spec("/tmp/lme-shard")
