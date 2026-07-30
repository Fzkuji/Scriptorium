#!/usr/bin/env python3
"""Collect and strictly audit a full v8.8+calendar GPT-5.5 LoCoMo run."""

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

DATA = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
EXPECTED_SAMPLES = list(range(10))
EXPECTED_CATEGORY_COUNTS = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}
REQUIRED_SOURCE_PATHS = {
    "src/nativemem.py",
    "src/v8_memory.py",
    "src/adapters/run_nativemem.py",
    "src/chatgpt_proxy.py",
    "src/openai_gpt55_flex_gateway.py",
    "src/openai_gpt55_flex_gateway_evidence.py",
    "scripts/gpt55_run_proxy.py",
    "benchmarks/locomo/data/locomo10.json",
    "scripts/run_v88_gpt55_locomo.py",
}


class AuditError(RuntimeError):
    pass


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
            raise AuditError(f"LoCoMo runner is active in {run_dir}") from exc
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_time(value: object) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise AuditError(f"invalid timestamp: {value!r}") from exc


def load_proxy_window(
    path: Path, started_at: object, finished_at: object
) -> list[dict[str, Any]]:
    start = parse_time(started_at)
    finish = parse_time(finished_at)
    entries = []
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
            raise AuditError(
                f"invalid proxy log line {line_number}: {type(exc).__name__}"
            ) from exc
        if start <= timestamp <= finish:
            entries.append(entry)
    return entries


def expected_records() -> dict[str, dict[str, Any]]:
    dataset = read_json(DATA)
    if not isinstance(dataset, list) or len(dataset) != 10:
        raise AuditError("LoCoMo source must contain exactly 10 samples")
    result = {}
    for sample, conversation in enumerate(dataset):
        for question_index, qa in enumerate(conversation.get("qa", [])):
            question_id = f"s{sample}_q{question_index}"
            result[question_id] = {
                "question": qa["question"],
                "gold": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                "category": int(qa["category"]),
            }
    if len(result) != 1986:
        raise AuditError(f"LoCoMo source has {len(result)} questions, expected 1986")
    return result


def resolve_build_record(
    run_dir: Path, sample: int, current: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    """Recover build accounting from a pre-retry artifact when memory was reused."""
    notes = str(current.get("notes", ""))
    if "model=gpt-5.5" in notes:
        return dict(current)
    if not notes.startswith("reused existing memory dir:"):
        raise AuditError(f"sample {sample} build does not record model=gpt-5.5")
    if state.get("memory_reused") is not True:
        raise AuditError(f"sample {sample} claims reused memory without manifest evidence")
    pattern = f"sample{sample}_questions.json.*"
    candidates = sorted((run_dir / "partials").glob(pattern), reverse=True)
    for candidate in candidates:
        try:
            records = read_json(candidate)
        except Exception:  # noqa: BLE001
            continue
        builds = [
            record for record in records
            if isinstance(record, dict) and record.get("question_id") == "_build_stats"
        ] if isinstance(records, list) else []
        if len(builds) != 1 or "model=gpt-5.5" not in str(builds[0].get("notes", "")):
            continue
        recovered = dict(builds[0])
        recovered["num_memories"] = current.get(
            "num_memories", recovered.get("num_memories")
        )
        recovered["recovered_from"] = str(candidate.relative_to(run_dir))
        return recovered
    raise AuditError(
        f"sample {sample} reused memory but has no prior GPT-5.5 build accounting"
    )


def validate_source_hashes(manifest: dict[str, Any]) -> None:
    """Require and verify every source file frozen by the LoCoMo runner."""
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise AuditError("manifest source_hashes must be a non-empty object")
    missing_source_hashes = sorted(REQUIRED_SOURCE_PATHS - set(source_hashes))
    if missing_source_hashes:
        raise AuditError(
            "manifest source_hashes omits required files: "
            f"{missing_source_hashes}"
        )
    source_mismatches = {}
    for relative, recorded_hash in source_hashes.items():
        if not isinstance(relative, str) or not isinstance(recorded_hash, str):
            raise AuditError("manifest source_hashes contains a non-string entry")
        source = ROOT / relative
        current = sha256_file(source) if source.is_file() else None
        if current != recorded_hash:
            source_mismatches[relative] = {
                "recorded": recorded_hash,
                "current": current,
            }
    if source_mismatches:
        raise AuditError(f"source files changed during run: {source_mismatches}")


def audit_provider_evidence(
    run_dir: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    provider = manifest.get("provider_evidence")
    if (
        not isinstance(provider, dict)
        or provider.get("schema") != "openai-gpt55-flex-invocations/v1"
        or provider.get("active_run_id") is not None
    ):
        raise AuditError("manifest Flex provider evidence is missing or active")
    if provider.get("gateway_root") != manifest.get("config", {}).get("gateway_root"):
        raise AuditError("manifest Flex gateway root differs from method config")
    invocations = provider.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise AuditError("manifest has no closed Flex provider invocation")
    reports: list[dict[str, Any]] = []
    consumers: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    evidence_root = (run_dir / "provider_evidence").resolve()
    recorded_output = Path(str(manifest.get("output_dir", run_dir))).resolve()

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

    for position, record in enumerate(invocations):
        if not isinstance(record, dict):
            raise AuditError(f"provider invocation {position} is not an object")
        record_path_value = record.get("record_path")
        if not isinstance(record_path_value, str):
            raise AuditError(f"provider invocation {position} has no record path")
        record_path = local_path(record_path_value)
        if not record_path.is_relative_to(evidence_root):
            raise AuditError("provider invocation record escapes result evidence root")
        stored = read_json(record_path)
        if record.get("record_sha256") != sha256_file(record_path):
            raise AuditError("provider invocation record hash differs")
        expected_record = {
            key: value for key, value in record.items()
            if key not in {"record_path", "record_sha256"}
        }
        if stored != expected_record:
            raise AuditError("provider invocation manifest copy differs")
        try:
            localized = localize(stored)
            report = flex_evidence.audit_invocation(localized)
            child_records = flex_evidence.load_child_proxy_records(
                Path(localized["consumer_log"]["path"]),
                expected_run_id=str(stored["run_id"]),
            )
        except flex_evidence.EvidenceError as exc:
            raise AuditError(f"provider invocation {position} failed: {exc}") from exc
        for item in child_records:
            request_id = str(item["gateway_request_id"])
            if request_id in seen_ids:
                raise AuditError("gateway request ID appears in multiple invocations")
            seen_ids.add(request_id)
        reports.append(report)
        consumers.extend(child_records)
    return reports, consumers


def _audit_unlocked(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise AuditError(f"run status is {manifest.get('status')!r}, not 'complete'")
    if manifest.get("benchmark") != "locomo":
        raise AuditError("manifest benchmark is not locomo")
    if manifest.get("method") != "NativeMem-v8.8+calendar":
        raise AuditError("manifest method is not NativeMem-v8.8+calendar")
    if manifest.get("backbone") != "gpt-5.5":
        raise AuditError("manifest backbone is not gpt-5.5")

    config = manifest.get("config", {})
    required_config = {
        "samples": EXPECTED_SAMPLES,
        "model": "gpt-5.5",
        "provider": "openai_api_flex_via_exclusive_child_proxy",
        "explicit_model_request_authorization": True,
        "chunk_turns": 6,
        "segment": "fixed",
        "single_model_retrieve_answer": True,
        "calendar": True,
    }
    mismatched_config = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in required_config.items() if config.get(key) != value
    }
    if mismatched_config:
        raise AuditError(f"method configuration mismatch: {mismatched_config}")
    if config.get("max_sessions") is not None or config.get("questions_limit") is not None:
        raise AuditError("full run cannot set max_sessions or questions_limit")

    expected = expected_records()
    combined: list[dict[str, Any]] = []
    build_records = []
    observed: dict[str, dict[str, Any]] = {}
    sample_audit = {}
    for sample in EXPECTED_SAMPLES:
        state = manifest.get("samples", {}).get(str(sample), {})
        if state.get("status") != "complete":
            raise AuditError(f"sample {sample} is not complete in manifest")
        output = run_dir / f"sample{sample}_questions.json"
        records = read_json(output)
        if not isinstance(records, list):
            raise AuditError(f"sample {sample} output is not a list")
        builds = [r for r in records if r.get("question_id") == "_build_stats"]
        questions = [r for r in records if r.get("question_id") != "_build_stats"]
        if len(builds) != 1:
            raise AuditError(f"sample {sample} has {len(builds)} build records")
        build = resolve_build_record(run_dir, sample, builds[0], state)
        build["sample"] = sample
        if int(build.get("build_calls", 0) or 0) <= 0:
            raise AuditError(f"sample {sample} has no recorded build calls")
        if int(build.get("build_tokens_in", 0) or 0) <= 0:
            raise AuditError(f"sample {sample} has no recorded build input tokens")
        if int(build.get("num_memories", 0) or 0) <= 0:
            raise AuditError(f"sample {sample} has no memory events")
        build_records.append(build)
        combined.append(build)

        for record in questions:
            question_id = str(record.get("question_id", ""))
            if question_id in observed:
                raise AuditError(f"duplicate question id: {question_id}")
            if question_id not in expected:
                raise AuditError(f"unknown question id: {question_id}")
            reference = expected[question_id]
            for key in ("question", "gold", "category"):
                if record.get(key) != reference[key]:
                    raise AuditError(f"{question_id} has mismatched {key}")
            if not str(record.get("answer", "")).strip():
                raise AuditError(f"{question_id} has an empty answer")
            retrieval = record.get("retrieval", {})
            if int(retrieval.get("calls", 0) or 0) <= 0:
                raise AuditError(f"{question_id} has no retrieval calls")
            if int(retrieval.get("steps", 0) or 0) <= 0:
                raise AuditError(f"{question_id} has no retrieval steps")
            if int(retrieval.get("tokens_in", 0) or 0) <= 0:
                raise AuditError(f"{question_id} has no retrieval input tokens")
            observed[question_id] = record
            combined.append(record)

        artifact = state.get("artifact", {})
        if artifact.get("json_sha256") != sha256_file(output):
            raise AuditError(f"sample {sample} output hash differs from manifest")
        if int(artifact.get("empty_answers", -1)) != 0:
            raise AuditError(f"sample {sample} manifest reports empty answers")
        sample_audit[str(sample)] = {
            "questions": len(questions),
            "output_sha256": sha256_file(output),
            "build_calls": build["build_calls"],
            "retrieval_calls": sum(
                int(record["retrieval"]["calls"]) for record in questions
            ),
        }

    missing = sorted(set(expected) - set(observed))
    if missing:
        raise AuditError(f"missing {len(missing)} question ids; first={missing[0]}")
    if len(observed) != 1986:
        raise AuditError(f"collected {len(observed)} questions, expected 1986")
    categories = Counter(int(record["category"]) for record in observed.values())
    if dict(sorted(categories.items())) != EXPECTED_CATEGORY_COUNTS:
        raise AuditError(f"category counts mismatch: {dict(categories)}")

    validate_source_hashes(manifest)

    provider_reports, proxy_entries = audit_provider_evidence(run_dir, manifest)
    if not proxy_entries:
        raise AuditError("Flex provider evidence has no successful requests")
    recorded_calls = sum(int(record["build_calls"]) for record in build_records)
    recorded_calls += sum(
        int(record["retrieval"]["calls"]) for record in observed.values()
    )

    report = {
        "schema_version": 1,
        "status": "passed",
        "benchmark": "LoCoMo",
        "method": "NativeMem-v8.8+calendar",
        "model": "gpt-5.5",
        "run_dir": str(run_dir),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "questions": len(observed),
        "cat1_4": sum(categories[key] for key in (1, 2, 3, 4)),
        "category_counts": dict(sorted(categories.items())),
        "empty_answers": 0,
        "samples": sample_audit,
        "source_hashes_match": True,
        "recorded_model_calls": recorded_calls,
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
        "combined_sha256": None,
        "scoring_input": None,
    }
    return combined, report


def audit(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    with stable_run_lock(run_dir):
        return _audit_unlocked(run_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    audit_path = run_dir / "audit.json"
    audit_path.unlink(missing_ok=True)
    try:
        with stable_run_lock(run_dir):
            combined, report = _audit_unlocked(run_dir)
            combined_path = run_dir / "questions_all.json"
            atomic_json(combined_path, combined)
            report["combined_sha256"] = sha256_file(combined_path)
            report["scoring_input"] = {
                "path": str(combined_path),
                "sha256": report["combined_sha256"],
            }
            atomic_json(audit_path, report)
    except (AuditError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
