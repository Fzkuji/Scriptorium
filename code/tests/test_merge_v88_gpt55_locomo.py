import fcntl
import json
import os
from pathlib import Path

import pytest

from scripts import merge_v88_gpt55_locomo as MOD


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def proxy_entry(path):
    entry = {
        "timestamp": "2026-07-14T00:05:00+00:00",
        "status": "success",
        "actual_model": "gpt-5.5",
        "attempts": 1,
        "unsupported_parameters": [],
    }
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")


def config(selection, gateway_root):
    return {
        "samples": list(selection),
        "model": "gpt-5.5",
        "provider": "openai_api_flex_via_exclusive_child_proxy",
        "gateway_root": str(gateway_root),
        "explicit_model_request_authorization": True,
        "chunk_turns": 6,
        "segment": "fixed",
        "single_model_retrieve_answer": True,
        "calendar": True,
        "sample_workers": 1,
        "request_concurrency": 2,
        "max_sessions": None,
        "questions_limit": None,
    }


def make_complete_sample(run_dir, sample):
    data = json.loads(MOD.runner.DATA.read_text())[sample]
    memory = run_dir / f"memory_sample{sample}"
    memory.mkdir(parents=True)
    (memory / "memory.md").write_text(
        f"memory for {sample}; operational path={memory}\n", encoding="utf-8"
    )
    output = run_dir / f"sample{sample}_questions.json"
    records = [{
        "question_id": "_build_stats",
        "notes": f"built fresh: 1 events, model=gpt-5.5; memory={memory}",
        "build_calls": 1,
        "build_tokens_in": 10,
        "build_tokens_out": 2,
        "num_memories": 1,
    }]
    for question_index, qa in enumerate(data["qa"]):
        records.append({
            "question_id": f"s{sample}_q{question_index}",
            "question": qa["question"],
            "gold": str(qa.get("answer", qa.get("adversarial_answer", ""))),
            "category": int(qa["category"]),
            "memories": [],
            "retrieval": {
                "latency_s": 0.1,
                "k": 20,
                "steps": 1,
                "calls": 1,
                "tokens_in": 5,
                "tokens_out": 1,
            },
            "answer": f"answer {sample}-{question_index}",
        })
    write_json(output, records)
    log = run_dir / f"sample{sample}.log"
    log.write_text(f"completed at {run_dir}\n", encoding="utf-8")
    return {
        "status": "complete",
        "finished_at": f"2026-07-14T00:{sample + 1:02d}:00+00:00",
        "attempt": 1,
        "validation": "complete",
        "output": str(output),
        "log": str(log),
        "artifact": MOD.runner.artifact_metadata(run_dir, sample),
        "memory_reused": False,
    }


def make_manifest(
    run_dir, selection, responsibility, fake_flex_provider, status
):
    run_dir.mkdir()
    (run_dir / ".launcher.lock").touch()
    samples = {
        str(sample): make_complete_sample(run_dir, sample)
        for sample in responsibility
    }
    cfg = config(selection, fake_flex_provider.root)
    sources = MOD.runner.source_hashes()
    manifest = {
        "schema_version": 1,
        "benchmark": "locomo",
        "method": "NativeMem-v8.8+calendar",
        "backbone": "gpt-5.5",
        "git_commit": MOD.current_git_head(),
        "created_at": "2026-07-14T00:00:00+00:00",
        "finished_at": "2026-07-14T00:20:00+00:00",
        "output_dir": str(run_dir),
        "config": cfg,
        "source_hashes": sources,
        "fingerprint": MOD.runner.fingerprint(cfg, sources),
        "samples": samples,
        "status": status,
        "failed_samples": [],
    }
    fake_flex_provider.attach(
        run_dir, manifest, f"locomo-{run_dir.name}"
    )
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def make_sources(tmp_path, fake_flex_provider):
    original = (tmp_path / "original").resolve()
    shard = (tmp_path / "shard5-9").resolve()
    first = make_manifest(
        original, range(10), range(5), fake_flex_provider, status="running"
    )
    partial_memory = original / "memory_sample5"
    partial_memory.mkdir()
    (partial_memory / "partial.md").write_text("partial\n", encoding="utf-8")
    (original / "sample5.log").write_text("partial log\n", encoding="utf-8")
    first["samples"]["5"] = {
        "status": "running",
        "started_at": "2026-07-14T00:10:00+00:00",
    }
    write_json(original / "run_manifest.json", first)
    make_manifest(
        shard, range(5, 10), range(5, 10), fake_flex_provider,
        status="complete",
    )
    specs = [
        MOD.ShardSpec(original, tuple(range(5))),
        MOD.ShardSpec(shard, tuple(range(5, 10))),
    ]
    return specs


@pytest.fixture
def sources(tmp_path, fake_flex_provider):
    return make_sources(tmp_path, fake_flex_provider)


def test_merge_rewrites_paths_records_partial_and_passes_strict_auditor(
    tmp_path, sources, monkeypatch
):
    output = (tmp_path / "canonical").resolve()

    def model_call_is_forbidden(*_args, **_kwargs):
        raise AssertionError("merger must not initialize an experiment environment")

    monkeypatch.setattr(MOD.runner, "experiment_env", model_call_is_forbidden)
    report = MOD.merge_shards(sources, output)

    assert report["copied"] == 10
    assert report["no_model_requests"] is True
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["config"]["samples"] == list(range(10))
    assert manifest["fingerprint"] == MOD.runner.fingerprint(
        manifest["config"], manifest["source_hashes"]
    )
    assert manifest["merge"]["no_model_requests"] is True
    assert manifest["merge"]["sources"][0]["original_selection"] == list(range(10))
    assert manifest["merge"]["sources"][0]["responsibility"] == list(range(5))
    ignored = manifest["merge"]["sources"][0]["ignored_partials"]
    assert ignored[0]["sample"] == 5
    assert ignored[0]["status"] == "running"
    assert ignored[0]["copied"] is False
    source_text = str(sources[0].run_dir)
    assert source_text not in (output / "sample0_questions.json").read_text()
    assert source_text not in (output / "sample0.log").read_text()
    assert source_text not in (output / "memory_sample0" / "memory.md").read_text()
    combined, audit_report = MOD.auditor.audit(output)
    assert len(combined) == 1996
    assert audit_report["status"] == "passed"


def test_resume_is_a_read_only_idempotent_validation(tmp_path, sources):
    output = (tmp_path / "canonical").resolve()
    MOD.merge_shards(sources, output)
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*") if path.is_file()
    }
    report = MOD.merge_shards(sources, output, resume=True)
    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*") if path.is_file()
    }
    assert report["copied"] == 0 and report["reused"] == 10
    assert after == before


def test_resume_rejects_untracked_root_artifact(tmp_path, sources):
    output = (tmp_path / "canonical").resolve()
    MOD.merge_shards(sources, output)
    (output / "unexpected-untracked.bin").write_bytes(b"unexpected")

    with pytest.raises(MOD.MergeError, match="unexpected root artifacts"):
        MOD.merge_shards(sources, output, resume=True)


def test_resume_rejects_manifest_state_tampering(tmp_path, sources):
    output = (tmp_path / "canonical").resolve()
    MOD.merge_shards(sources, output)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["samples"]["0"]["log"] = "/tmp/unrelated.log"
    write_json(manifest_path, manifest)
    with pytest.raises(MOD.MergeError, match="source provenance"):
        MOD.merge_shards(sources, output, resume=True)


def test_existing_output_requires_resume(tmp_path, sources):
    output = (tmp_path / "canonical").resolve()
    MOD.merge_shards(sources, output)
    with pytest.raises(MOD.MergeError, match="already exists"):
        MOD.merge_shards(sources, output)


def test_complete_sample_outside_responsibility_is_rejected(tmp_path, sources):
    manifest_path = sources[0].run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["samples"]["5"] = {"status": "complete"}
    write_json(manifest_path, manifest)
    with pytest.raises(MOD.MergeError, match="completed sample 5 outside"):
        MOD.merge_shards(sources, tmp_path / "canonical")


def test_config_except_selection_must_match(tmp_path, sources):
    manifest_path = sources[1].run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["request_concurrency"] = 3
    manifest["fingerprint"] = MOD.runner.fingerprint(
        manifest["config"], manifest["source_hashes"]
    )
    write_json(manifest_path, manifest)
    with pytest.raises(MOD.MergeError, match="config_except_samples"):
        MOD.merge_shards(sources, tmp_path / "canonical")


def test_active_source_launcher_is_rejected(tmp_path, sources):
    lock = (sources[0].run_dir / ".launcher.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(MOD.MergeError, match="launcher is active"):
            MOD.merge_shards(sources, tmp_path / "canonical")
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def test_symlink_in_responsible_memory_is_rejected(tmp_path, sources):
    memory = sources[0].run_dir / "memory_sample0"
    os.symlink(memory / "memory.md", memory / "alias.md")
    with pytest.raises(MOD.MergeError, match="symbolic links"):
        MOD.merge_shards(sources, tmp_path / "canonical")


def test_symlink_source_root_is_rejected(tmp_path, sources):
    source_link = tmp_path / "source-link"
    source_link.symlink_to(sources[0].run_dir, target_is_directory=True)
    linked_specs = [MOD.ShardSpec(source_link, sources[0].samples), sources[1]]
    with pytest.raises(MOD.MergeError, match="source directory cannot"):
        MOD.merge_shards(linked_specs, tmp_path / "canonical")


def test_failed_strict_audit_leaves_no_destination(tmp_path, sources, monkeypatch):
    output = (tmp_path / "canonical").resolve()

    def reject(_run_dir):
        raise MOD.auditor.AuditError("synthetic audit failure")

    monkeypatch.setattr(MOD.auditor, "audit", reject)
    with pytest.raises(MOD.MergeError, match="synthetic audit failure"):
        MOD.merge_shards(sources, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".canonical.merge-*"))


def test_responsibilities_and_input_specs_are_exact(tmp_path, sources):
    assert MOD.parse_input_spec("/tmp/original:0-2,4") == MOD.ShardSpec(
        Path("/tmp/original"), (0, 1, 2, 4)
    )
    with pytest.raises(Exception, match="DIR:SAMPLESPEC"):
        MOD.parse_input_spec("/tmp/original")
    overlap = [
        MOD.ShardSpec(sources[0].run_dir, tuple(range(6))),
        MOD.ShardSpec(sources[1].run_dir, tuple(range(5, 10))),
    ]
    with pytest.raises(MOD.MergeError, match="overlap"):
        MOD.merge_shards(overlap, tmp_path / "canonical")
    gap = [
        MOD.ShardSpec(sources[0].run_dir, tuple(range(4))),
        MOD.ShardSpec(sources[1].run_dir, tuple(range(5, 10))),
    ]
    with pytest.raises(MOD.MergeError, match="cover samples"):
        MOD.merge_shards(gap, tmp_path / "canonical")
