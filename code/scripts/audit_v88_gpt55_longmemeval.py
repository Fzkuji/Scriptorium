#!/usr/bin/env python3
"""Collect and strictly audit a full LongMemEval-S GPT-5.5 run."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from contextlib import contextmanager


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402

DATA = ROOT / "benchmarks" / "longmemeval" / "data" / "longmemeval_s_cleaned.json"
EXPECTED_TYPES = {
    "multi-session": 133,
    "temporal-reasoning": 133,
    "knowledge-update": 78,
    "single-session-user": 70,
    "single-session-assistant": 56,
    "single-session-preference": 30,
}
EXPECTED_METHOD = {
    "name": "NativeMem",
    "version": "v8.8+calendar",
    "calendar": True,
    "single_model_retrieve_answer": True,
}
EXPECTED_MODELS = {
    "builder": "gpt-5.5",
    "retriever": "gpt-5.5",
    "answerer": "gpt-5.5",
    "provider": "openai_api_flex_via_exclusive_child_proxy",
}
FROZEN_CONFIG = {
    "NATIVEMEM_PROMPT": "v8",
    "NATIVEMEM_STORE_MODE": "oneshot",
    "NATIVEMEM_V8_SINGLE": "1",
    "NATIVEMEM_CHUNK_TURNS": "6",
    "NATIVEMEM_V8_SEGMENT": "fixed",
    "NATIVEMEM_V8_TIDY": "on",
    "NATIVEMEM_V8_SECTIONS": "on",
    "NATIVEMEM_V8_ARTICLE": "off",
    "NATIVEMEM_V8_VERIFY": "on",
    "NATIVEMEM_V8_MERGE_LINES": "on",
    "NATIVEMEM_V8_TIDY_COMBINED": "off",
    "NATIVEMEM_V8_MAX_TOPICS": "30",
    "NATIVEMEM_TOPK": "20",
    "NATIVEMEM_V8_MAX_ROUNDS": "12",
    "NATIVEMEM_V8_MAX_TOKENS": "1200",
    "NATIVEMEM_V8_READ_CONTEXT": "1",
    "NATIVEMEM_V8_MAP": "dir",
    "NATIVEMEM_V8_MAP_INLINE": "8",
    "NATIVEMEM_V8_MAX_DEPTH": "4",
    "NATIVEMEM_V8_REWRITE_MIN": "8",
    "NATIVEMEM_V8_SEGMENT_MAX": "10",
}


class AuditError(RuntimeError):
    pass


def reject_symlink_components(path: Path, label: str) -> Path:
    """Reject a symlink at any existing component; never resolve a trust root."""
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise AuditError(f"{label} contains a symbolic-link component: {current}")
    return absolute


def memory_tree_sha256(memory_dir: Path) -> str:
    entries = []
    for path in sorted(memory_dir.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise AuditError(f"memory tree contains a symbolic link: {path}")
        relative = path.relative_to(memory_dir).as_posix()
        if path.is_dir():
            entries.append({"path": relative, "type": "directory"})
        elif path.is_file():
            entries.append({
                "path": relative,
                "type": "file",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        else:
            raise AuditError(f"memory tree contains a non-regular entry: {path}")
    payload = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def stable_run_lock(run_dir: Path):
    lock_path = run_dir / ".launcher.lock"
    if not lock_path.is_file() or lock_path.is_symlink():
        raise AuditError(f"runner lock file is missing or unsafe: {lock_path}")
    handle = lock_path.open("r+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AuditError(f"LongMemEval runner is active in {run_dir}") from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time(value: object) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise AuditError(f"invalid timestamp: {value!r}") from exc


def expected_records(data_path: Path = DATA) -> list[dict[str, Any]]:
    data = read_json(data_path)
    if not isinstance(data, list) or len(data) != 500:
        raise AuditError(f"LongMemEval-S source has {len(data)} items, expected 500")
    ids = [str(item.get("question_id", "")) for item in data]
    if len(set(ids)) != 500 or any(not question_id for question_id in ids):
        raise AuditError("LongMemEval-S question ids are missing or duplicated")
    types = Counter(str(item.get("question_type")) for item in data)
    if dict(types) != EXPECTED_TYPES:
        raise AuditError(f"LongMemEval-S type counts differ: {dict(types)}")
    if sum(question_id.endswith("_abs") for question_id in ids) != 30:
        raise AuditError("LongMemEval-S must contain 30 abstention questions")
    return data


def validate_frozen_config(config: object) -> dict[str, str]:
    if not isinstance(config, dict):
        raise AuditError("manifest config is not an object")
    expected_keys = set(FROZEN_CONFIG) | {"NATIVEMEM_V8_CONCURRENCY"}
    if set(config) != expected_keys:
        raise AuditError(
            "method config keys differ: "
            f"missing={sorted(expected_keys - set(config))}, "
            f"unexpected={sorted(set(config) - expected_keys)}"
        )
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in FROZEN_CONFIG.items()
        if config.get(key) != value
    }
    if mismatches:
        raise AuditError(f"method configuration mismatch: {mismatches}")
    concurrency = config.get("NATIVEMEM_V8_CONCURRENCY")
    try:
        parsed_concurrency = int(concurrency)
    except (TypeError, ValueError) as exc:
        raise AuditError("method concurrency is not an integer string") from exc
    if str(parsed_concurrency) != concurrency or parsed_concurrency <= 0:
        raise AuditError("method concurrency must be a canonical positive integer")
    return dict(config)


def audit_provider_evidence(
    run_dir: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    provider = manifest.get("provider_evidence")
    request_audit = manifest.get("request_audit")
    if (
        not isinstance(provider, dict)
        or provider.get("schema") != "openai-gpt55-flex-invocations/v1"
        or provider.get("active_run_id") is not None
        or not isinstance(request_audit, dict)
        or request_audit.get("gateway_root") != provider.get("gateway_root")
    ):
        raise AuditError("manifest Flex provider evidence is missing or active")
    invocations = provider.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise AuditError("manifest has no closed Flex provider invocation")
    recorded_output = Path(str(manifest.get("output_dir", run_dir))).resolve()
    evidence_root = (run_dir / "provider_evidence").resolve()

    def local_path(value: str) -> Path:
        path = Path(value).resolve()
        try:
            relative = path.relative_to(recorded_output)
        except ValueError:
            return path
        return (run_dir / relative).resolve()

    def localize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: localize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [localize(item) for item in value]
        if isinstance(value, str) and value.startswith(f"{recorded_output}{os.sep}"):
            return str(local_path(value))
        return value

    reports: list[dict[str, Any]] = []
    consumers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, record in enumerate(invocations):
        if not isinstance(record, dict) or not isinstance(record.get("record_path"), str):
            raise AuditError(f"provider invocation {position} is invalid")
        record_path = local_path(record["record_path"])
        if not record_path.is_relative_to(evidence_root):
            raise AuditError("provider invocation record escapes result evidence root")
        stored = read_json(record_path)
        if record.get("record_sha256") != sha256_file(record_path):
            raise AuditError("provider invocation record hash differs")
        if stored != {
            key: value for key, value in record.items()
            if key not in {"record_path", "record_sha256"}
        }:
            raise AuditError("provider invocation manifest copy differs")
        localized = localize(stored)
        try:
            reports.append(flex_evidence.audit_invocation(localized))
            records = flex_evidence.load_child_proxy_records(
                Path(localized["consumer_log"]["path"]),
                expected_run_id=str(localized["run_id"]),
            )
        except flex_evidence.EvidenceError as exc:
            raise AuditError(f"provider invocation {position} failed: {exc}") from exc
        for item in records:
            request_id = str(item["gateway_request_id"])
            if request_id in seen:
                raise AuditError("gateway request ID appears in multiple invocations")
            seen.add(request_id)
        consumers.extend(records)
    return reports, consumers


def validate_checkpoint_identity(
    checkpoint: dict[str, Any], manifest: dict[str, Any],
    checkpoint_path: Path, run_dir: Path, index: int,
) -> Path:
    """Bind immutable checkpoint metadata and paths to this exact run."""
    for key in ("method", "models", "config", "code", "request_audit"):
        if checkpoint.get(key) != manifest.get(key):
            raise AuditError(f"item {index} {key} differs from run manifest")

    run_dir = reject_symlink_components(run_dir, "run directory")
    items_dir = reject_symlink_components(run_dir / "items", "items directory")
    actual_checkpoint = reject_symlink_components(
        checkpoint_path, f"item {index} checkpoint"
    )
    item_dir = actual_checkpoint.parent
    if item_dir.parent != items_dir:
        raise AuditError(f"item {index} checkpoint is outside run items")
    if actual_checkpoint.name != "checkpoint.json":
        raise AuditError(f"item {index} checkpoint filename is invalid")
    if not item_dir.is_dir() or not items_dir.is_dir():
        raise AuditError(f"item {index} item/items directory is missing")
    if len(item_dir.relative_to(items_dir).parts) != 1:
        raise AuditError(f"item {index} checkpoint has an unexpected directory depth")
    memory_dir = reject_symlink_components(
        item_dir / "memory", f"item {index} memory directory"
    )
    if memory_dir.parent != item_dir:
        raise AuditError(f"item {index} memory directory is outside its item")
    memory_tree_sha256(memory_dir)

    expected_paths = {
        "item_dir": str(item_dir.resolve()),
        "memory_dir": str(memory_dir.resolve()),
        "checkpoint": str(actual_checkpoint.resolve()),
    }
    paths = checkpoint.get("paths")
    if not isinstance(paths, dict) or set(paths) != set(expected_paths):
        raise AuditError(f"item {index} paths metadata is incomplete or unexpected")
    for key, expected in expected_paths.items():
        value = paths.get(key)
        if not isinstance(value, str) or value != expected:
            raise AuditError(f"item {index} {key} path differs from actual artifact")
    return memory_dir


def checkpoint_to_records(
    checkpoint: dict[str, Any], reference: dict[str, Any], index: int, *,
    manifest: dict[str, Any], checkpoint_path: Path, run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    question_id = str(reference["question_id"])
    if checkpoint.get("status") != "complete":
        raise AuditError(f"item {index} is not complete")
    if checkpoint.get("dataset_index") != index:
        raise AuditError(f"item {index} dataset index mismatch")
    if str(checkpoint.get("question_id")) != question_id:
        raise AuditError(f"item {index} question id mismatch")
    exact_fields = {
        "question_type": reference["question_type"],
        "question": reference["question"],
        "gold": reference["answer"],
        "question_date_raw": reference["question_date"],
        "answer_session_ids": reference["answer_session_ids"],
    }
    for key, expected in exact_fields.items():
        if checkpoint.get(key) != expected:
            raise AuditError(f"item {index} {key} mismatch")
    input_meta = checkpoint.get("input", {})
    if input_meta.get("source_session_ids") != reference["haystack_session_ids"]:
        raise AuditError(f"item {index} source session ids mismatch")
    if input_meta.get("source_dates") != reference["haystack_dates"]:
        raise AuditError(f"item {index} source dates mismatch")
    expected_turns = sum(len(session) for session in reference["haystack_sessions"])
    if input_meta.get("sessions") != len(reference["haystack_sessions"]):
        raise AuditError(f"item {index} source session count mismatch")
    if input_meta.get("turns") != expected_turns:
        raise AuditError(f"item {index} source turn count mismatch")

    memory_dir = validate_checkpoint_identity(
        checkpoint, manifest, checkpoint_path, run_dir, index
    )
    models = checkpoint.get("models", {})
    if any(models.get(role) != "gpt-5.5" for role in ("builder", "retriever", "answerer")):
        raise AuditError(f"item {index} does not record GPT-5.5 for all model roles")
    build = checkpoint.get("build", {})
    retrieval = checkpoint.get("retrieval", {})
    answer = checkpoint.get("answer", {})
    if build.get("status") != "complete" or retrieval.get("status") != "complete":
        raise AuditError(f"item {index} has an incomplete build/retrieval stage")
    if answer.get("status") != "complete" or answer.get("model") != "gpt-5.5":
        raise AuditError(f"item {index} has invalid answer metadata")
    hypothesis = str(answer.get("hypothesis", "")).strip()
    if not hypothesis:
        raise AuditError(f"item {index} has an empty answer")
    build_usage = build.get("usage", {})
    retrieval_usage = retrieval.get("usage", {})
    for phase, usage in (("build", build_usage), ("retrieval", retrieval_usage)):
        if int(usage.get("calls", 0) or 0) <= 0:
            raise AuditError(f"item {index} has no {phase} calls")
        if int(usage.get("tokens_in", 0) or 0) <= 0:
            raise AuditError(f"item {index} has no {phase} input tokens")
    if int(build.get("events", 0) or 0) <= 0:
        raise AuditError(f"item {index} has no memory events")
    if int(retrieval.get("steps", 0) or 0) <= 0:
        raise AuditError(f"item {index} has no retrieval steps")
    if not memory_dir.is_dir() or not any(memory_dir.rglob("*.md")):
        raise AuditError(f"item {index} memory directory is missing or empty")

    build_record = {
        "question_id": "_build_stats",
        "dataset_index": index,
        "source_question_id": question_id,
        "build_time_s": build.get("wall_time_s", 0),
        "build_calls": build_usage.get("calls", 0),
        "build_tokens_in": build_usage.get("tokens_in", 0),
        "build_tokens_out": build_usage.get("tokens_out", 0),
        "build_llm_time_s": build_usage.get("llm_time_s", 0),
        "num_memories": build.get("events", 0),
        "notes": "LongMemEval-S independent history; model=gpt-5.5",
    }
    qa_record = {
        "question_id": question_id,
        "dataset_index": index,
        "question": reference["question"],
        "gold": reference["answer"],
        "question_type": reference["question_type"],
        "abstention": question_id.endswith("_abs"),
        "answer": hypothesis,
        "retrieval": {
            "latency_s": retrieval.get("wall_time_s", 0),
            "steps": retrieval.get("steps", 0),
            "calls": retrieval_usage.get("calls", 0),
            "tokens_in": retrieval_usage.get("tokens_in", 0),
            "tokens_out": retrieval_usage.get("tokens_out", 0),
        },
    }
    return build_record, qa_record


def load_proxy_window(path: Path, start_value: object, end_value: object):
    start, end = parse_time(start_value), parse_time(end_value)
    selected = []
    payload = path.read_bytes()
    final_newline = payload.rfind(b"\n")
    complete = payload[:final_newline] if final_newline >= 0 else b""
    for line_number, raw_line in enumerate(complete.split(b"\n"), start=1):
        if not raw_line.strip():
            continue
        try:
            line = raw_line.decode("utf-8")
            entry = json.loads(line)
            timestamp = parse_time(entry["timestamp"])
        except Exception as exc:  # noqa: BLE001
            raise AuditError(f"invalid proxy log line {line_number}") from exc
        if start <= timestamp <= end:
            selected.append(entry)
    return selected


def validate_aggregate_outputs(
    run_dir: Path, checkpoints: list[dict[str, Any]],
    qa_records: list[dict[str, Any]],
) -> None:
    expected_results = [{
        "dataset_index": item["dataset_index"],
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "gold": item["gold"],
        "hypothesis": item["answer"]["hypothesis"],
        "build": item["build"],
        "retrieval": item["retrieval"],
        "models": item["models"],
        "config": item["config"],
    } for item in checkpoints]
    official_results = read_json(run_dir / "results.json")
    if official_results != expected_results:
        raise AuditError("results.json differs from durable checkpoints")

    official_hypotheses = [
        json.loads(line)
        for line in (run_dir / "hypotheses.jsonl").read_text().splitlines()
        if line.strip()
    ]
    expected_hypotheses = [
        {"question_id": record["question_id"], "hypothesis": record["answer"]}
        for record in qa_records
    ]
    if official_hypotheses != expected_hypotheses:
        raise AuditError("official hypotheses.jsonl differs from checkpoints")


def _audit_unlocked(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = reject_symlink_components(run_dir, "run directory")
    if not run_dir.is_dir():
        raise AuditError(f"run directory is missing: {run_dir}")
    reject_symlink_components(run_dir / "items", "items directory")
    manifest_path = run_dir / "run_manifest.json"
    references = expected_records()
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("completed") != 500:
        raise AuditError("run manifest does not report 500 completed items")
    if manifest.get("benchmark") != "LongMemEval-S":
        raise AuditError("manifest benchmark is not LongMemEval-S")
    if manifest.get("method") != EXPECTED_METHOD:
        raise AuditError("manifest method metadata differs from v8.8+calendar")
    models = manifest.get("models", {})
    if (
        not isinstance(models, dict)
        or {key: models.get(key) for key in EXPECTED_MODELS} != EXPECTED_MODELS
        or set(models) != {*EXPECTED_MODELS, "gateway_root"}
        or not isinstance(models.get("gateway_root"), str)
    ):
        raise AuditError("manifest model metadata differs from frozen GPT-5.5 setup")
    validate_frozen_config(manifest.get("config"))
    if manifest.get("dataset_items") != 500:
        raise AuditError("manifest dataset size is not 500")
    if manifest.get("dataset_sha256") != sha256_file(DATA):
        raise AuditError("manifest dataset hash differs from LongMemEval-S source")

    code_paths = {
        "runner_sha256": ROOT / "scripts" / "run_v88_gpt55_longmemeval.py",
        "nativemem_sha256": ROOT / "src" / "nativemem.py",
        "v8_memory_sha256": ROOT / "src" / "v8_memory.py",
        "adapter_sha256": ROOT / "src" / "adapters" / "run_nativemem.py",
        "proxy_sha256": ROOT / "src" / "chatgpt_proxy.py",
        "flex_gateway_sha256": ROOT / "src" / "openai_gpt55_flex_gateway.py",
        "flex_evidence_sha256": (
            ROOT / "src" / "openai_gpt55_flex_gateway_evidence.py"
        ),
        "child_proxy_sha256": ROOT / "scripts" / "gpt55_run_proxy.py",
    }
    code = manifest.get("code")
    if not isinstance(code, dict) or set(code) != {"git_commit", *code_paths}:
        raise AuditError("manifest code inventory differs from the runner schema")
    for key, path in code_paths.items():
        if code.get(key) != sha256_file(path):
            raise AuditError(f"source hash mismatch: {key}")
    request_audit = manifest.get("request_audit")
    if (
        not isinstance(request_audit, dict)
        or set(request_audit)
        != {"mode", "gateway_root", "explicit_model_request_authorization"}
        or request_audit.get("mode")
        != "bounded_gateway_segments_with_exclusive_child_proxy"
        or request_audit.get("gateway_root") != models.get("gateway_root")
        or request_audit.get("explicit_model_request_authorization") is not True
    ):
        raise AuditError("manifest request_audit metadata is invalid")

    records = []
    qa_records = []
    build_records = []
    checkpoints = []
    memory_hashes = []
    for index, reference in enumerate(references):
        candidates = list((run_dir / "items").glob(f"{index:04d}_*/checkpoint.json"))
        if len(candidates) != 1:
            raise AuditError(f"item {index} has {len(candidates)} checkpoints")
        checkpoint = read_json(candidates[0])
        if not isinstance(checkpoint, dict):
            raise AuditError(f"item {index} checkpoint is not an object")
        build_record, qa_record = checkpoint_to_records(
            checkpoint, reference, index, manifest=manifest,
            checkpoint_path=candidates[0], run_dir=run_dir,
        )
        checkpoints.append(checkpoint)
        memory_hashes.append({
            "dataset_index": index,
            "sha256": memory_tree_sha256(candidates[0].parent / "memory"),
        })
        build_records.append(build_record)
        qa_records.append(qa_record)
        records.extend((build_record, qa_record))

    qids = [record["question_id"] for record in qa_records]
    if len(set(qids)) != 500:
        raise AuditError("collected question ids are duplicated")
    types = Counter(record["question_type"] for record in qa_records)
    if dict(types) != EXPECTED_TYPES:
        raise AuditError(f"collected question type counts differ: {dict(types)}")
    if sum(record["abstention"] for record in qa_records) != 30:
        raise AuditError("collected abstention count is not 30")

    validate_aggregate_outputs(run_dir, checkpoints, qa_records)

    provider_reports, proxy_entries = audit_provider_evidence(run_dir, manifest)
    if not proxy_entries:
        raise AuditError("Flex provider evidence has no successful requests")
    recorded_calls = sum(int(record["build_calls"]) for record in build_records)
    recorded_calls += sum(int(record["retrieval"]["calls"]) for record in qa_records)

    report = {
        "schema_version": 1,
        "status": "passed",
        "benchmark": "LongMemEval-S",
        "method": "NativeMem-v8.8+calendar",
        "model": "gpt-5.5",
        "run_dir": str(run_dir),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "questions": 500,
        "abstention_questions": 30,
        "question_type_counts": dict(types),
        "empty_answers": 0,
        "source_hashes_match": True,
        "recorded_model_calls": recorded_calls,
        "memory_tree_inventory_sha256": hashlib.sha256(json.dumps(
            memory_hashes, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "provider_evidence": {
            "evidence_scope": "bounded_gateway_segments",
            "exact_run_response_id_linkage": True,
            "consumer_provider_correspondence": True,
            "invocations": provider_reports,
            "gateway_request_ids": [
                entry["gateway_request_id"] for entry in proxy_entries
            ],
            "response_ids": [entry["response_id"] for entry in proxy_entries],
            "run_call_count_comparison": {
                "recorded": recorded_calls,
                "provider": len(proxy_entries),
            },
            "global_actual_model_consistency": True,
            "content_hash_linkage": True,
            "logical_proxy_requests": len(proxy_entries),
            "entries": len(proxy_entries),
            "successes": len(proxy_entries),
            "errors": 0,
            "provider_models": sorted(
                {entry.get("provider_actual_model") for entry in proxy_entries}
            ),
            "service_tiers": sorted(
                {entry.get("service_tier") for entry in proxy_entries}
            ),
        },
        "evaluation_input_sha256": None,
        "scoring_input": None,
    }
    return records, report


def audit(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = reject_symlink_components(run_dir, "run directory")
    with stable_run_lock(run_dir):
        return _audit_unlocked(run_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = reject_symlink_components(args.run_dir, "run directory")
    audit_path = run_dir / "audit.json"
    audit_path.unlink(missing_ok=True)
    try:
        with stable_run_lock(run_dir):
            records, report = _audit_unlocked(run_dir)
            output = run_dir / "evaluation_input.json"
            atomic_json(output, records)
            report["evaluation_input_sha256"] = sha256_file(output)
            report["scoring_input"] = {
                "path": str(output),
                "sha256": report["evaluation_input_sha256"],
            }
            atomic_json(audit_path, report)
    except (AuditError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
