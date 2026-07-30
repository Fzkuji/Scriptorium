#!/usr/bin/env python3
"""Strictly merge completed LoCoMo responsibility sets without model calls.

The two launchers used for the formal run may declare overlapping selections
(the original launcher declared 0--9 before it was stopped after sample 4).
The responsibility supplied to this program is therefore separate from the
selection recorded by each launcher.  A responsible sample must be complete;
any complete sample outside responsibility is rejected.  Incomplete artifacts
outside responsibility are recorded but never copied.

For a new destination, all artifacts are copied into a sibling temporary
directory, operational paths are rewritten, the ordinary strict LoCoMo auditor
is executed in-process, and the directory is installed with one ``os.replace``.
The module never imports or initializes a model backend.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_v88_gpt55_locomo as auditor  # noqa: E402
from scripts import run_v88_gpt55_locomo as runner  # noqa: E402


EXPECTED_SAMPLES = tuple(range(10))
EXPECTED_CONFIG_KEYS = {
    "samples",
    "model",
    "provider",
    "gateway_root",
    "explicit_model_request_authorization",
    "chunk_turns",
    "segment",
    "single_model_retrieve_answer",
    "calendar",
    "sample_workers",
    "request_concurrency",
    "max_sessions",
    "questions_limit",
}
EXPECTED_CONFIG_VALUES = {
    "model": "gpt-5.5",
    "provider": "openai_api_flex_via_exclusive_child_proxy",
    "explicit_model_request_authorization": True,
    "chunk_turns": 6,
    "segment": "fixed",
    "single_model_retrieve_answer": True,
    "calendar": True,
    "max_sessions": None,
    "questions_limit": None,
}


class MergeError(RuntimeError):
    """The source shards or destination do not satisfy the merge contract."""


@dataclass(frozen=True)
class ShardSpec:
    run_dir: Path
    samples: tuple[int, ...]


@dataclass(frozen=True)
class ValidatedSample:
    sample: int
    source_dir: Path
    state: dict[str, Any]
    output: Path
    log: Path
    memory: Path
    partials: tuple[Path, ...]
    output_sha256: str
    log_sha256: str
    memory_sha256: str
    partials_sha256: str
    questions: int
    finished_at: str


@dataclass(frozen=True)
class ValidatedShard:
    spec: ShardSpec
    manifest: dict[str, Any]
    manifest_sha256: str
    original_selection: tuple[int, ...]
    samples: tuple[ValidatedSample, ...]
    ignored_partials: tuple[dict[str, Any], ...]
    inventory_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeError(f"cannot read valid JSON from {path}: {exc}") from exc


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_timestamp(value: object, label: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value.strip():
        raise MergeError(f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MergeError(f"{label} is not an ISO timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise MergeError(f"{label} must include a timezone")
    return value, parsed


def current_git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise MergeError(f"cannot determine current git commit: {exc}") from exc


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_no_symlinks(path: Path) -> None:
    if path.is_symlink():
        raise MergeError(f"symbolic links are not accepted: {path}")
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_symlink():
                raise MergeError(f"symbolic links are not accepted: {child}")
            if not child.is_dir() and not child.is_file():
                raise MergeError(f"non-regular filesystem entry: {child}")
    elif not path.is_file():
        raise MergeError(f"artifact is not a regular file or directory: {path}")


def tree_sha256(path: Path) -> str:
    validate_no_symlinks(path)
    records: list[dict[str, Any]] = []
    if path.is_file():
        return sha256_json({
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        relative = child.relative_to(path).as_posix()
        if child.is_dir():
            records.append({"path": relative, "type": "directory"})
        else:
            records.append({
                "path": relative,
                "type": "file",
                "bytes": child.stat().st_size,
                "sha256": sha256_file(child),
            })
    return sha256_json(records)


def merge_payload_tree_inventory(run_dir: Path) -> dict[str, Any]:
    """Hash every merge-owned node, excluding self/derived audit artifacts."""
    excluded = {"run_manifest.json", "questions_all.json", "audit.json"}
    records: list[dict[str, Any]] = []
    seen_paths: dict[str, str] = {}
    seen_inodes: dict[tuple[int, int], str] = {}
    for path in sorted(run_dir.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(run_dir).as_posix()
        if relative.split("/", 1)[0] in excluded:
            continue
        collision = unicodedata.normalize("NFC", relative).casefold()
        previous = seen_paths.get(collision)
        if previous is not None and previous != relative:
            raise MergeError(f"Unicode/case path collision: {previous} / {relative}")
        seen_paths[collision] = relative
        if path.is_symlink():
            raise MergeError(f"symbolic links are not accepted: {path}")
        if path.is_dir():
            records.append({"path": relative, "type": "directory"})
            continue
        if not path.is_file():
            raise MergeError(f"non-regular filesystem entry: {path}")
        stat = path.stat(follow_symlinks=False)
        inode = (stat.st_dev, stat.st_ino)
        if stat.st_nlink != 1 or inode in seen_inodes:
            raise MergeError(f"hard links are not accepted: {path}")
        seen_inodes[inode] = relative
        records.append({
            "path": relative,
            "type": "file",
            "bytes": stat.st_size,
            "sha256": sha256_file(path),
        })
    return {
        "excluded_named_artifacts": sorted(excluded),
        "nodes": len(records),
        "sha256": sha256_json(records),
    }


def parse_samples(value: str) -> tuple[int, ...]:
    try:
        return tuple(runner.parse_samples(value))
    except (argparse.ArgumentTypeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_input_spec(value: str) -> ShardSpec:
    try:
        directory, sample_spec = value.rsplit(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("input must use DIR:SAMPLESPEC") from exc
    if not directory.strip() or not sample_spec.strip():
        raise argparse.ArgumentTypeError("input must use DIR:SAMPLESPEC")
    return ShardSpec(Path(directory), parse_samples(sample_spec))


def validate_specs(specs: list[ShardSpec]) -> None:
    if len(specs) < 2:
        raise MergeError(f"at least two input specs are required, found {len(specs)}")
    coverage: set[int] = set()
    for spec in specs:
        responsibility = set(spec.samples)
        overlap = coverage & responsibility
        if overlap:
            raise MergeError(
                f"responsibilities overlap at samples {sorted(overlap)}"
            )
        coverage.update(responsibility)
    expected = set(EXPECTED_SAMPLES)
    if coverage != expected:
        raise MergeError(
            "responsibilities must cover samples 0..9 exactly: "
            f"missing={sorted(expected - coverage)}, "
            f"unexpected={sorted(coverage - expected)}"
        )


def validate_path_layout(specs: list[ShardSpec], output_dir: Path) -> None:
    sources = [spec.run_dir.resolve() for spec in specs]
    if len(set(sources)) != len(sources):
        raise MergeError("source directories must be distinct")
    for source in sources:
        if source == output_dir:
            raise MergeError("output directory must differ from every source")
        if is_within(source, output_dir) or is_within(output_dir, source):
            raise MergeError("output and source directories cannot be nested")


@contextmanager
def directory_lock(run_dir: Path, *, exclusive: bool) -> Iterator[TextIO]:
    lock_path = run_dir / ".launcher.lock"
    if lock_path.is_symlink() or not lock_path.is_file():
        raise MergeError(f"launcher lock is missing or invalid: {lock_path}")
    handle = lock_path.open("a+", encoding="utf-8")
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise MergeError(f"source or destination launcher is active: {run_dir}") from exc
    try:
        yield handle
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def merge_lock(output_dir: Path) -> Iterator[TextIO]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.merge.lock"
    if lock_path.is_symlink():
        raise MergeError(f"merge lock cannot be a symbolic link: {lock_path}")
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise MergeError(f"another merger is using {output_dir}") from exc
    try:
        yield handle
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def resolve_recorded_path(value: object, expected: Path, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise MergeError(f"{label} path is missing")
    recorded = Path(value)
    if not recorded.is_absolute():
        recorded = ROOT / recorded
    if recorded.resolve() != expected.resolve():
        raise MergeError(
            f"{label} path does not bind to its source artifact: {value!r}"
        )


def validate_manifest_common(
    manifest: dict[str, Any], spec: ShardSpec
) -> tuple[int, ...]:
    if manifest.get("schema_version") != 1:
        raise MergeError(f"{spec.run_dir} has an unsupported manifest schema")
    expected_top = {
        "benchmark": "locomo",
        "method": "NativeMem-v8.8+calendar",
        "backbone": "gpt-5.5",
    }
    for key, value in expected_top.items():
        if manifest.get(key) != value:
            raise MergeError(f"{spec.run_dir} manifest {key} differs from {value!r}")
    if manifest.get("git_commit") != current_git_head():
        raise MergeError(f"{spec.run_dir} git commit differs from current checkout")
    parse_timestamp(manifest.get("created_at"), f"{spec.run_dir} created_at")
    resolve_recorded_path(
        manifest.get("output_dir"), spec.run_dir, f"{spec.run_dir} output_dir"
    )

    config = manifest.get("config")
    if not isinstance(config, dict) or set(config) != EXPECTED_CONFIG_KEYS:
        raise MergeError(f"{spec.run_dir} configuration keys differ from the runner")
    for key, value in EXPECTED_CONFIG_VALUES.items():
        if config.get(key) != value:
            raise MergeError(f"{spec.run_dir} configuration differs at {key}")
    for key in ("sample_workers", "request_concurrency"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MergeError(f"{spec.run_dir} has invalid {key}")
    selection = config.get("samples")
    if (
        not isinstance(selection, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in selection)
        or sorted(set(selection)) != selection
        or any(item not in EXPECTED_SAMPLES for item in selection)
    ):
        raise MergeError(f"{spec.run_dir} has an invalid original sample selection")
    if not set(spec.samples).issubset(selection):
        raise MergeError(
            f"{spec.run_dir} responsibility is not contained in its original selection"
        )
    try:
        auditor.audit_provider_evidence(spec.run_dir, manifest)
    except (auditor.AuditError, OSError, json.JSONDecodeError) as exc:
        raise MergeError(
            f"{spec.run_dir} has invalid Flex provider evidence: {exc}"
        ) from exc

    sources = manifest.get("source_hashes")
    expected_sources = runner.source_hashes()
    if sources != expected_sources:
        raise MergeError(f"{spec.run_dir} source hashes differ from current frozen sources")
    if manifest.get("fingerprint") != runner.fingerprint(config, sources):
        raise MergeError(f"{spec.run_dir} runner fingerprint is invalid")
    samples = manifest.get("samples")
    if not isinstance(samples, dict):
        raise MergeError(f"{spec.run_dir} samples state is not an object")
    return tuple(selection)


def sample_partials(run_dir: Path, sample: int) -> tuple[Path, ...]:
    partials_dir = run_dir / "partials"
    if not partials_dir.exists():
        return ()
    validate_no_symlinks(partials_dir)
    prefixes = (
        f"sample{sample}_questions.json.",
        f"sample{sample}.log.",
        f"memory_sample{sample}.",
    )
    return tuple(
        path for path in sorted(partials_dir.iterdir(), key=lambda item: item.name)
        if path.name.startswith(prefixes)
    )


def partials_inventory(paths: tuple[Path, ...]) -> str:
    records = [{"name": path.name, "sha256": tree_sha256(path)} for path in paths]
    return sha256_json(records)


def strict_sample_records(
    output: Path, sample: int, state: dict[str, Any], run_dir: Path
) -> tuple[int, dict[str, Any]]:
    valid, reason = runner.validate_sample(
        output, sample, questions_limit=None, expected_model="gpt-5.5"
    )
    if not valid:
        raise MergeError(f"sample {sample} output is incomplete: {reason}")
    records = read_json(output)
    builds = [record for record in records if record.get("question_id") == "_build_stats"]
    questions = [record for record in records if record.get("question_id") != "_build_stats"]
    try:
        build = auditor.resolve_build_record(run_dir, sample, builds[0], state)
    except auditor.AuditError as exc:
        raise MergeError(str(exc)) from exc
    for key in ("build_calls", "build_tokens_in", "num_memories"):
        if int(build.get(key, 0) or 0) <= 0:
            raise MergeError(f"sample {sample} build has invalid {key}")

    expected = auditor.expected_records()
    expected_ids = {
        question_id for question_id in expected if question_id.startswith(f"s{sample}_")
    }
    observed: set[str] = set()
    for record in questions:
        question_id = str(record.get("question_id", ""))
        if question_id in observed or question_id not in expected_ids:
            raise MergeError(f"sample {sample} has invalid question id {question_id!r}")
        reference = expected[question_id]
        for key in ("question", "gold", "category"):
            if record.get(key) != reference[key]:
                raise MergeError(f"{question_id} has mismatched {key}")
        if not str(record.get("answer", "")).strip():
            raise MergeError(f"{question_id} has an empty answer")
        retrieval = record.get("retrieval")
        if not isinstance(retrieval, dict):
            raise MergeError(f"{question_id} retrieval metadata is missing")
        for key in ("calls", "steps", "tokens_in"):
            if int(retrieval.get(key, 0) or 0) <= 0:
                raise MergeError(f"{question_id} retrieval has invalid {key}")
        observed.add(question_id)
    if observed != expected_ids:
        raise MergeError(f"sample {sample} question coverage differs from the dataset")
    return len(questions), build


def validate_complete_sample(
    spec: ShardSpec, manifest: dict[str, Any], sample: int
) -> ValidatedSample:
    state = manifest["samples"].get(str(sample))
    if not isinstance(state, dict) or state.get("status") != "complete":
        raise MergeError(f"{spec.run_dir} responsible sample {sample} is not complete")
    output = spec.run_dir / f"sample{sample}_questions.json"
    log = spec.run_dir / f"sample{sample}.log"
    memory = spec.run_dir / f"memory_sample{sample}"
    for path in (output, log, memory):
        if not is_within(path, spec.run_dir):
            raise MergeError(f"sample {sample} artifact escapes its source directory")
        validate_no_symlinks(path)
    if output.stat().st_size <= 0 or log.stat().st_size <= 0:
        raise MergeError(f"sample {sample} output or log is empty")
    if not any(memory.rglob("*.md")):
        raise MergeError(f"sample {sample} memory directory has no markdown files")
    resolve_recorded_path(state.get("output"), output, f"sample {sample} output")
    resolve_recorded_path(state.get("log"), log, f"sample {sample} log")
    questions, _build = strict_sample_records(output, sample, state, spec.run_dir)
    artifact = state.get("artifact")
    expected_artifact = runner.artifact_metadata(spec.run_dir, sample)
    if not isinstance(artifact, dict) or artifact != expected_artifact:
        raise MergeError(f"sample {sample} manifest artifact metadata differs")
    partials = sample_partials(spec.run_dir, sample)
    if state.get("memory_reused") is True and not any(
        path.name.startswith(f"sample{sample}_questions.json.") for path in partials
    ):
        raise MergeError(f"sample {sample} reused memory without retained build output")
    finished_value = state.get("finished_at", state.get("validated_at"))
    finished_at, _ = parse_timestamp(
        finished_value, f"sample {sample} completion timestamp"
    )
    return ValidatedSample(
        sample=sample,
        source_dir=spec.run_dir,
        state=copy.deepcopy(state),
        output=output,
        log=log,
        memory=memory,
        partials=partials,
        output_sha256=sha256_file(output),
        log_sha256=sha256_file(log),
        memory_sha256=tree_sha256(memory),
        partials_sha256=partials_inventory(partials),
        questions=questions,
        finished_at=finished_at,
    )


def ignored_artifact(path: Path, kind: str) -> dict[str, Any]:
    validate_no_symlinks(path)
    return {
        "kind": kind,
        "path": str(path),
        "sha256": tree_sha256(path),
        "copied": False,
    }


def validate_outside_responsibility(
    spec: ShardSpec, manifest: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    ignored: list[dict[str, Any]] = []
    responsible = set(spec.samples)
    for sample in EXPECTED_SAMPLES:
        if sample in responsible:
            continue
        state = manifest["samples"].get(str(sample))
        if isinstance(state, dict) and state.get("status") == "complete":
            raise MergeError(
                f"{spec.run_dir} has completed sample {sample} outside responsibility"
            )
        output = spec.run_dir / f"sample{sample}_questions.json"
        if output.exists():
            validate_no_symlinks(output)
            complete, _ = runner.validate_sample(
                output, sample, questions_limit=None, expected_model="gpt-5.5"
            )
            if complete:
                raise MergeError(
                    f"{spec.run_dir} has complete output for sample {sample} "
                    "outside responsibility"
                )
        artifacts = []
        for path, kind in (
            (output, "questions"),
            (spec.run_dir / f"sample{sample}.log", "log"),
            (spec.run_dir / f"memory_sample{sample}", "memory"),
        ):
            if path.exists():
                artifacts.append(ignored_artifact(path, kind))
        partials = sample_partials(spec.run_dir, sample)
        artifacts.extend(ignored_artifact(path, "archived_partial") for path in partials)
        if state is not None or artifacts:
            ignored.append({
                "sample": sample,
                "status": state.get("status") if isinstance(state, dict) else None,
                "declared_in_original_selection": sample in manifest["config"]["samples"],
                "artifacts": artifacts,
                "copied": False,
            })
    return tuple(ignored)


def validate_shard(spec: ShardSpec) -> ValidatedShard:
    manifest_path = spec.run_dir / "run_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise MergeError(f"source manifest is missing or invalid: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise MergeError(f"source manifest is not an object: {manifest_path}")
    original_selection = validate_manifest_common(manifest, spec)
    samples = tuple(
        validate_complete_sample(spec, manifest, sample) for sample in spec.samples
    )
    ignored = validate_outside_responsibility(spec, manifest)
    inventory = [{
        "sample": item.sample,
        "output_sha256": item.output_sha256,
        "log_sha256": item.log_sha256,
        "memory_sha256": item.memory_sha256,
        "partials_sha256": item.partials_sha256,
    } for item in samples]
    return ValidatedShard(
        spec=spec,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        original_selection=original_selection,
        samples=samples,
        ignored_partials=ignored,
        inventory_sha256=sha256_json(inventory),
    )


def config_without_selection(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "samples"}


def validate_shard_compatibility(shards: list[ValidatedShard]) -> None:
    reference = shards[0].manifest
    for shard in shards[1:]:
        manifest = shard.manifest
        mismatches = []
        for key in ("schema_version", "benchmark", "method", "backbone", "git_commit"):
            if manifest.get(key) != reference.get(key):
                mismatches.append(key)
        if manifest.get("source_hashes") != reference.get("source_hashes"):
            mismatches.append("source_hashes")
        if config_without_selection(manifest["config"]) != config_without_selection(
            reference["config"]
        ):
            mismatches.append("config_except_samples")
        if mismatches:
            raise MergeError(
                f"source manifests are incompatible in {', '.join(mismatches)}"
            )


def source_descriptor(shard: ValidatedShard) -> dict[str, Any]:
    return {
        "run_dir": str(shard.spec.run_dir),
        "manifest_sha256": shard.manifest_sha256,
        "original_selection": list(shard.original_selection),
        "responsibility": list(shard.spec.samples),
        "responsible_samples": len(shard.samples),
        "inventory_sha256": shard.inventory_sha256,
        "reported_status": shard.manifest.get("status"),
        "ignored_partials": list(shard.ignored_partials),
    }


def path_replacements(source: Path, destination: Path) -> tuple[tuple[str, str], ...]:
    values = [(str(source), str(destination))]
    try:
        values.append((str(source.relative_to(ROOT)), str(destination.relative_to(ROOT))))
    except ValueError:
        pass
    return tuple(sorted(set(values), key=lambda item: len(item[0]), reverse=True))


def rewrite_value(value: Any, replacements: tuple[tuple[str, str], ...]) -> Any:
    if isinstance(value, dict):
        return {key: rewrite_value(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_value(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    return value


def rewrite_text_files(path: Path, replacements: tuple[tuple[str, str], ...]) -> None:
    files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    for file_path in files:
        payload = file_path.read_bytes()
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        rewritten = text
        for old, new in replacements:
            rewritten = rewritten.replace(old, new)
        if rewritten != text:
            file_path.write_text(rewritten, encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def destination_state(
    sample: ValidatedSample, artifact_dir: Path, output_dir: Path
) -> dict[str, Any]:
    replacements = path_replacements(sample.source_dir, output_dir)
    output = artifact_dir / f"sample{sample.sample}_questions.json"
    log = artifact_dir / f"sample{sample.sample}.log"
    memory = artifact_dir / f"memory_sample{sample.sample}"
    state = rewrite_value(copy.deepcopy(sample.state), replacements)
    state["status"] = "complete"
    state["output"] = display_path(output_dir / output.name)
    state["log"] = display_path(output_dir / log.name)
    state["memory"] = display_path(output_dir / memory.name)
    state["artifact"] = {
        "json_sha256": sha256_file(output),
        "questions": sample.questions,
        "empty_answers": 0,
        "memory_markdown_files": sum(1 for path in memory.rglob("*.md")),
    }
    state["merge_source"] = {
        "run_dir": str(sample.source_dir),
        "sample": sample.sample,
        "source_output_sha256": sample.output_sha256,
        "source_log_sha256": sample.log_sha256,
        "source_memory_sha256": sample.memory_sha256,
        "source_partials_sha256": sample.partials_sha256,
    }
    return state


def copy_sample(sample: ValidatedSample, staging: Path, output_dir: Path) -> dict[str, Any]:
    replacements = path_replacements(sample.source_dir, output_dir)
    output = staging / f"sample{sample.sample}_questions.json"
    log = staging / f"sample{sample.sample}.log"
    memory = staging / f"memory_sample{sample.sample}"
    records = rewrite_value(read_json(sample.output), replacements)
    atomic_json(output, records)
    shutil.copy2(sample.log, log)
    shutil.copytree(sample.memory, memory, copy_function=shutil.copy2)
    rewrite_text_files(log, replacements)
    rewrite_text_files(memory, replacements)

    partials_dir = staging / "partials"
    for source in sample.partials:
        target = partials_dir / source.name
        partials_dir.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, copy_function=shutil.copy2)
        else:
            shutil.copy2(source, target)
        rewrite_text_files(target, replacements)

    return destination_state(sample, staging, output_dir)


def destination_inventory(run_dir: Path) -> dict[str, Any]:
    samples = {}
    for sample in EXPECTED_SAMPLES:
        output = run_dir / f"sample{sample}_questions.json"
        log = run_dir / f"sample{sample}.log"
        memory = run_dir / f"memory_sample{sample}"
        for path in (output, log, memory):
            validate_no_symlinks(path)
        partials = sample_partials(run_dir, sample)
        samples[str(sample)] = {
            "output_sha256": sha256_file(output),
            "log_sha256": sha256_file(log),
            "memory_sha256": tree_sha256(memory),
            "partials_sha256": partials_inventory(partials),
        }
    payload_tree = merge_payload_tree_inventory(run_dir)
    return {
        "samples": samples,
        "sha256": sha256_json(samples),
        "merge_payload_tree": payload_tree,
    }


def validate_exact_destination_names(run_dir: Path) -> None:
    expected_outputs = {f"sample{sample}_questions.json" for sample in EXPECTED_SAMPLES}
    expected_logs = {f"sample{sample}.log" for sample in EXPECTED_SAMPLES}
    expected_memories = {f"memory_sample{sample}" for sample in EXPECTED_SAMPLES}
    observed_outputs = {path.name for path in run_dir.glob("sample*_questions.json")}
    observed_logs = {path.name for path in run_dir.glob("sample*.log")}
    observed_memories = {path.name for path in run_dir.glob("memory_sample*")}
    if observed_outputs != expected_outputs:
        raise MergeError("destination question-file inventory is not exactly samples 0..9")
    if observed_logs != expected_logs:
        raise MergeError("destination log inventory is not exactly samples 0..9")
    if observed_memories != expected_memories:
        raise MergeError("destination memory inventory is not exactly samples 0..9")
    allowed_root = {
        ".launcher.lock",
        "run_manifest.json",
        "questions_all.json",
        "audit.json",
        "partials",
        "provider_evidence",
        *expected_outputs,
        *expected_logs,
        *expected_memories,
    }
    unexpected_root = sorted(
        path.name for path in run_dir.iterdir() if path.name not in allowed_root
    )
    if unexpected_root:
        raise MergeError(
            f"destination contains unexpected root artifacts: {unexpected_root}"
        )


def build_manifest(
    shards: list[ValidatedShard], output_dir: Path, states: dict[str, Any],
    inventory: dict[str, Any], provider_evidence: dict[str, Any],
    *, merged_at: str | None = None,
) -> dict[str, Any]:
    descriptors = [source_descriptor(shard) for shard in shards]
    created_values = [
        parse_timestamp(shard.manifest["created_at"], "source created_at")
        for shard in shards
    ]
    finished_values = [
        parse_timestamp(sample.finished_at, f"sample {sample.sample} finished_at")
        for shard in shards for sample in shard.samples
    ]
    created_at = min(created_values, key=lambda item: item[1])[0]
    finished_at = max(finished_values, key=lambda item: item[1])[0]
    config = copy.deepcopy(shards[0].manifest["config"])
    config["samples"] = list(EXPECTED_SAMPLES)
    sources = copy.deepcopy(shards[0].manifest["source_hashes"])
    merged_at = merged_at or utc_now()
    return {
        "schema_version": 1,
        "benchmark": "locomo",
        "method": "NativeMem-v8.8+calendar",
        "backbone": "gpt-5.5",
        "git_commit": shards[0].manifest["git_commit"],
        "created_at": created_at,
        "finished_at": finished_at,
        "output_dir": str(output_dir),
        "config": config,
        "source_hashes": sources,
        "fingerprint": runner.fingerprint(config, sources),
        "samples": states,
        "status": "complete",
        "failed_samples": [],
        "provider_evidence": provider_evidence,
        "merge": {
            "schema_version": 1,
            "tool": Path(__file__).name,
            "tool_sha256": sha256_file(Path(__file__)),
            "merged_at": merged_at,
            "no_model_requests": True,
            "coverage": {"samples": list(EXPECTED_SAMPLES)},
            "source_fingerprint": sha256_json(descriptors),
            "sources": descriptors,
            "destination_inventory": inventory,
        },
}


def copy_provider_evidence(
    shards: list[ValidatedShard], staging: Path, output_dir: Path
) -> dict[str, Any]:
    invocations: list[dict[str, Any]] = []
    gateway_roots = {
        str(shard.manifest["provider_evidence"]["gateway_root"])
        for shard in shards
    }
    if len(gateway_roots) != 1:
        raise MergeError("source shards use different Flex gateway roots")
    for shard_index, shard in enumerate(shards):
        source_provider = shard.manifest["provider_evidence"]
        for invocation_index, source_record in enumerate(
            source_provider["invocations"]
        ):
            source_record_path = Path(source_record["record_path"]).resolve()
            source_dir = source_record_path.parent
            relative = Path(f"shard-{shard_index:03d}") / (
                f"invocation-{invocation_index:03d}"
            )
            target_staging = staging / "provider_evidence" / relative
            target_final = output_dir / "provider_evidence" / relative
            shutil.copytree(source_dir, target_staging, copy_function=shutil.copy2)
            stored = read_json(source_record_path)
            replacements = path_replacements(source_dir, target_final)
            rewritten = rewrite_value(stored, replacements)
            window_path = target_final / "gateway_window.json"
            consumer_path = target_final / "child_proxy_requests.jsonl"
            rewritten["window"]["path"] = str(window_path)
            rewritten["consumer_log"]["path"] = str(consumer_path)
            ready_staging = target_staging / "child_proxy_ready.json"
            ready_final = target_final / "child_proxy_ready.json"
            ready_record = rewrite_value(read_json(ready_staging), replacements)
            atomic_json(ready_staging, ready_record)
            rewritten["child_ready"]["path"] = str(ready_final)
            rewritten["child_ready"]["sha256"] = sha256_file(ready_staging)
            record_path = target_staging / "invocation.json"
            atomic_json(record_path, rewritten)
            manifest_record = copy.deepcopy(rewritten)
            manifest_record["record_path"] = str(target_final / "invocation.json")
            manifest_record["record_sha256"] = sha256_file(record_path)
            invocations.append(manifest_record)
    return {
        "schema": "openai-gpt55-flex-invocations/v1",
        "gateway_root": gateway_roots.pop(),
        "active_run_id": None,
        "invocations": invocations,
    }


def run_strict_audit(run_dir: Path, *, launcher_lock_held: bool = False) -> None:
    try:
        if launcher_lock_held:
            audit_unlocked = getattr(auditor, "_audit_unlocked", None)
            if not callable(audit_unlocked):
                raise MergeError(
                    "strict auditor does not expose its lock-aware audit implementation"
                )
            combined, report = audit_unlocked(run_dir)
        else:
            combined, report = auditor.audit(run_dir)
    except (auditor.AuditError, OSError, json.JSONDecodeError) as exc:
        raise MergeError(f"strict LoCoMo audit failed: {exc}") from exc
    if len(combined) != 1996 or report.get("status") != "passed":
        raise MergeError("strict LoCoMo audit returned an incomplete result")


def validate_existing_destination(
    output_dir: Path, shards: list[ValidatedShard]
) -> dict[str, Any]:
    validate_no_symlinks(output_dir)
    validate_exact_destination_names(output_dir)
    manifest_path = output_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise MergeError("existing destination manifest is not an object")
    merge = manifest.get("merge")
    descriptors = [source_descriptor(shard) for shard in shards]
    if not isinstance(merge, dict):
        raise MergeError("existing destination has no merge provenance")
    expected_merge_fields = {
        "tool": Path(__file__).name,
        "tool_sha256": sha256_file(Path(__file__)),
        "no_model_requests": True,
        "source_fingerprint": sha256_json(descriptors),
        "sources": descriptors,
        "coverage": {"samples": list(EXPECTED_SAMPLES)},
    }
    for key, value in expected_merge_fields.items():
        if merge.get(key) != value:
            raise MergeError(f"existing destination merge provenance differs at {key}")
    expected_inventory = destination_inventory(output_dir)
    if merge.get("destination_inventory") != expected_inventory:
        raise MergeError("existing destination artifact inventory differs")
    merged_at, _ = parse_timestamp(merge.get("merged_at"), "merge merged_at")
    states = {
        str(sample.sample): destination_state(sample, output_dir, output_dir)
        for shard in shards for sample in shard.samples
    }
    expected_manifest = build_manifest(
        shards,
        output_dir,
        states,
        expected_inventory,
        copy.deepcopy(manifest.get("provider_evidence")),
        merged_at=merged_at,
    )
    if manifest != expected_manifest:
        raise MergeError("existing destination manifest differs from source provenance")
    run_strict_audit(output_dir, launcher_lock_held=True)
    return manifest


def merge_shards(
    specs: list[ShardSpec], output_dir: Path, *, resume: bool = False
) -> dict[str, Any]:
    normalized: list[ShardSpec] = []
    for spec in specs:
        raw_source = spec.run_dir.expanduser()
        if raw_source.is_symlink():
            raise MergeError(f"source directory cannot be a symbolic link: {raw_source}")
        normalized.append(ShardSpec(raw_source.resolve(), tuple(spec.samples)))
    normalized_specs = sorted(
        normalized, key=lambda item: (item.samples, str(item.run_dir))
    )
    raw_output = output_dir.expanduser()
    if raw_output.is_symlink():
        raise MergeError(f"output directory cannot be a symbolic link: {raw_output}")
    output_dir = raw_output.resolve()
    validate_specs(normalized_specs)
    validate_path_layout(normalized_specs, output_dir)

    with merge_lock(output_dir), ExitStack() as stack:
        if output_dir.exists() and not resume:
            raise MergeError(
                f"output directory already exists; pass --resume to validate it: {output_dir}"
            )
        for spec in sorted(normalized_specs, key=lambda item: str(item.run_dir)):
            if spec.run_dir.is_symlink() or not spec.run_dir.is_dir():
                raise MergeError(f"source directory is missing or a symlink: {spec.run_dir}")
            stack.enter_context(directory_lock(spec.run_dir, exclusive=False))
        shards = [validate_shard(spec) for spec in normalized_specs]
        validate_shard_compatibility(shards)

        if output_dir.exists():
            stack.enter_context(directory_lock(output_dir, exclusive=True))
            manifest = validate_existing_destination(output_dir, shards)
            return {
                "status": "complete",
                "output_dir": str(output_dir),
                "samples": len(EXPECTED_SAMPLES),
                "copied": 0,
                "reused": len(EXPECTED_SAMPLES),
                "manifest_sha256": sha256_file(output_dir / "run_manifest.json"),
                "source_fingerprint": manifest["merge"]["source_fingerprint"],
                "no_model_requests": True,
            }

        staging = Path(tempfile.mkdtemp(
            prefix=f".{output_dir.name}.merge-", dir=output_dir.parent
        ))
        try:
            (staging / ".launcher.lock").touch()
            states: dict[str, Any] = {}
            for shard in shards:
                for sample in shard.samples:
                    states[str(sample.sample)] = copy_sample(sample, staging, output_dir)
            if sorted(map(int, states)) != list(EXPECTED_SAMPLES):
                raise MergeError("copied sample states do not cover samples 0..9")
            provider_evidence = copy_provider_evidence(shards, staging, output_dir)
            validate_no_symlinks(staging)
            validate_exact_destination_names(staging)
            inventory = destination_inventory(staging)
            manifest = build_manifest(
                shards, output_dir, states, inventory, provider_evidence
            )
            atomic_json(staging / "run_manifest.json", manifest)
            run_strict_audit(staging)
            os.replace(staging, output_dir)
            parent_fd = os.open(output_dir.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    return {
        "status": "complete",
        "output_dir": str(output_dir),
        "samples": len(EXPECTED_SAMPLES),
        "copied": len(EXPECTED_SAMPLES),
        "reused": 0,
        "manifest_sha256": sha256_file(output_dir / "run_manifest.json"),
        "source_fingerprint": manifest["merge"]["source_fingerprint"],
        "no_model_requests": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly merge LoCoMo sample responsibilities without model calls"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="DIR:SAMPLESPEC",
        help="source directory and responsibility (for example DIR:0-4); repeat",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        specs = [parse_input_spec(value) for value in args.input]
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    try:
        report = merge_shards(specs, args.output_dir, resume=args.resume)
    except MergeError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
