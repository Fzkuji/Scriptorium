#!/usr/bin/env python3
"""Merge completed LongMemEval-S runner shards without model requests.

The merger treats item checkpoints and memory directories as the durable
source of truth.  It validates both source manifests, requires exact assigned
index coverage, copies each item through a temporary directory, rewrites the
absolute checkpoint paths, and regenerates the two runner aggregate files.

The script never imports or initializes the NativeMem backend.  Its only
dependency on the runner module is the runner's offline schema and validation
helpers.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
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

from scripts import run_v88_gpt55_longmemeval as runner  # noqa: E402
from scripts import audit_v88_gpt55_longmemeval as auditor  # noqa: E402


DEFAULT_DATA = runner.DEFAULT_DATA
EXPECTED_ITEMS = runner.EXPECTED_LONGMEMEVAL_SIZE
COMPATIBILITY_KEYS = (
    "schema_version",
    "benchmark",
    "dataset_path",
    "dataset_sha256",
    "dataset_items",
    "method",
    "models",
    "config",
    "code",
    "request_audit",
)
CODE_PATHS = {
    "runner_sha256": ROOT / "scripts" / "run_v88_gpt55_longmemeval.py",
    "nativemem_sha256": ROOT / "src" / "nativemem.py",
    "v8_memory_sha256": ROOT / "src" / "v8_memory.py",
    "adapter_sha256": ROOT / "src" / "adapters" / "run_nativemem.py",
    "proxy_sha256": ROOT / "src" / "chatgpt_proxy.py",
    "flex_gateway_sha256": ROOT / "src" / "openai_gpt55_flex_gateway.py",
    "flex_evidence_sha256": ROOT / "src" / "openai_gpt55_flex_gateway_evidence.py",
    "child_proxy_sha256": ROOT / "scripts" / "gpt55_run_proxy.py",
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


class MergeError(RuntimeError):
    """A source shard or destination is incompatible with a strict merge."""


def reject_symlink_components(path: Path, label: str) -> Path:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise MergeError(f"{label} contains a symbolic-link component: {current}")
    return absolute


@dataclass(frozen=True)
class ShardSpec:
    run_dir: Path
    start: int
    limit: int

    @property
    def stop(self) -> int:
        return self.start + self.limit

    @property
    def indices(self) -> range:
        return range(self.start, self.stop)


@dataclass(frozen=True)
class ValidatedItem:
    index: int
    question_id: str
    item_dir: Path
    checkpoint: dict[str, Any]
    checkpoint_sha256: str
    payload_sha256: str


@dataclass(frozen=True)
class ValidatedShard:
    spec: ShardSpec
    manifest: dict[str, Any]
    manifest_sha256: str
    declared_selections: tuple[dict[str, Any], ...]
    items: tuple[ValidatedItem, ...]
    ignored_partials: tuple[dict[str, Any], ...]
    inventory_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeError(f"cannot read valid JSON from {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def merge_payload_tree_inventory(output_dir: Path) -> dict[str, Any]:
    """Hash all merge-owned nodes while excluding self/derived audit files."""
    excluded = {"run_manifest.json", "evaluation_input.json", "audit.json"}
    records: list[dict[str, Any]] = []
    seen_paths: dict[str, str] = {}
    seen_inodes: dict[tuple[int, int], str] = {}
    for path in sorted(output_dir.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(output_dir).as_posix()
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


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def current_git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise MergeError(f"cannot determine current git commit: {exc}") from exc


def parse_timestamp(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MergeError(f"{label} is missing")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MergeError(f"{label} is not an ISO timestamp: {value!r}") from exc
    return value


def is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def validate_path_layout(specs: list[ShardSpec], output_dir: Path) -> None:
    resolved_sources = [spec.run_dir.resolve() for spec in specs]
    if len(set(resolved_sources)) != len(resolved_sources):
        raise MergeError("source shard directories must be distinct")
    resolved_output = output_dir.resolve()
    for source in resolved_sources:
        if resolved_output == source:
            raise MergeError("output directory must differ from source shards")
        if is_within(resolved_output, source) or is_within(source, resolved_output):
            raise MergeError("output and source shard directories cannot be nested")


@contextmanager
def directory_lock(
    run_dir: Path, *, shared: bool, create_dir: bool = False
) -> Iterator[TextIO]:
    if create_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".launcher.lock"
    try:
        handle = lock_path.open("a+", encoding="utf-8")
    except OSError as exc:
        raise MergeError(f"cannot open lock {lock_path}: {exc}") from exc
    operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    try:
        fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        role = "source shard" if shared else "output directory"
        raise MergeError(f"{role} is active or locked: {run_dir}") from exc
    try:
        yield handle
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def validate_specs(specs: list[ShardSpec], total: int) -> None:
    if len(specs) < 2:
        raise MergeError(f"at least two shard specs are required, found {len(specs)}")
    for spec in specs:
        if spec.start < 0 or spec.limit <= 0 or spec.stop > total:
            raise MergeError(
                f"invalid shard selection {spec.start}:{spec.limit} for {total} items"
            )
    coverage: set[int] = set()
    for spec in specs:
        selection = set(spec.indices)
        overlap = coverage & selection
        if overlap:
            raise MergeError(
                f"shard responsibilities overlap at indices {sorted(overlap)[:10]}"
            )
        coverage.update(selection)
    expected = set(range(total))
    if coverage != expected:
        missing = sorted(expected - coverage)
        unexpected = sorted(coverage - expected)
        raise MergeError(
            "shard selections do not cover the dataset exactly: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )


def validate_declared_selections(
    manifest: dict[str, Any], spec: ShardSpec, total: int
) -> tuple[dict[str, Any], ...]:
    invocations = manifest.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise MergeError(f"{spec.run_dir} manifest has no invocations")
    normalized: list[dict[str, Any]] = []
    assigned = set(spec.indices)
    matching = False
    for position, invocation in enumerate(invocations):
        if not isinstance(invocation, dict):
            raise MergeError(
                f"{spec.run_dir} invocation {position} is not an object"
            )
        start = invocation.get("start")
        limit = invocation.get("limit")
        if isinstance(start, bool) or not isinstance(start, int):
            raise MergeError(
                f"{spec.run_dir} invocation {position} has invalid start"
            )
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        ):
            raise MergeError(
                f"{spec.run_dir} invocation {position} has invalid limit"
            )
        try:
            indices = runner.select_indices(total, start, limit)
        except runner.DataValidationError as exc:
            raise MergeError(
                f"{spec.run_dir} invocation {position} is invalid: {exc}"
            ) from exc
        declared_bounds = invocation.get("indices")
        expected_bounds = [indices[0], indices[-1]]
        if declared_bounds != expected_bounds:
            raise MergeError(
                f"{spec.run_dir} invocation {position} indices differ: "
                f"expected {expected_bounds}, found {declared_bounds!r}"
            )
        normalized.append({
            "start": start,
            "limit": limit,
            "indices": expected_bounds,
            "resume": bool(invocation.get("resume", False)),
        })
        if start == spec.start and assigned.issubset(indices):
            matching = True
    if not matching:
        raise MergeError(
            f"{spec.run_dir} has no invocation covering assigned selection "
            f"{spec.start}:{spec.limit} from the same start index"
        )
    return tuple(normalized)


def validate_common_manifest(
    manifest: dict[str, Any], spec: ShardSpec, data_path: Path,
    dataset_sha256: str, total: int,
) -> None:
    if manifest.get("schema_version") != runner.SCHEMA_VERSION:
        raise MergeError(f"{spec.run_dir} has an unsupported manifest schema")
    if manifest.get("benchmark") != "LongMemEval-S":
        raise MergeError(f"{spec.run_dir} is not a LongMemEval-S run")
    if manifest.get("dataset_items") != total:
        raise MergeError(f"{spec.run_dir} dataset_items differs from {total}")
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise MergeError(f"{spec.run_dir} dataset hash differs from --data")
    if manifest.get("dataset_path") != str(data_path):
        raise MergeError(f"{spec.run_dir} dataset path differs from --data")
    parse_timestamp(manifest.get("created_at"), f"{spec.run_dir} created_at")

    code = manifest.get("code")
    expected_code_keys = {"git_commit", *CODE_PATHS}
    if not isinstance(code, dict) or set(code) != expected_code_keys:
        raise MergeError(f"{spec.run_dir} code inventory differs from runner schema")
    if code.get("git_commit") != current_git_head():
        raise MergeError(f"{spec.run_dir} git commit differs from current checkout")
    for key, path in CODE_PATHS.items():
        actual = sha256_file(path)
        if code.get(key) != actual:
            raise MergeError(
                f"{spec.run_dir} source hash {key} differs from {path}"
            )

    if manifest.get("method") != EXPECTED_METHOD:
        raise MergeError(f"{spec.run_dir} method differs from v8.8+calendar")
    models = manifest.get("models")
    if (
        not isinstance(models, dict)
        or {key: models.get(key) for key in EXPECTED_MODELS} != EXPECTED_MODELS
        or set(models) != {*EXPECTED_MODELS, "gateway_root"}
    ):
        raise MergeError(f"{spec.run_dir} model configuration differs from GPT-5.5")
    config = manifest.get("config")
    expected_config_keys = set(runner.METHOD_ENV) | {"NATIVEMEM_V8_CONCURRENCY"}
    if not isinstance(config, dict) or set(config) != expected_config_keys:
        raise MergeError(f"{spec.run_dir} method configuration keys differ")
    for key, value in runner.METHOD_ENV.items():
        if config.get(key) != value:
            raise MergeError(f"{spec.run_dir} method configuration differs at {key}")
    concurrency = config.get("NATIVEMEM_V8_CONCURRENCY")
    try:
        parsed_concurrency = int(concurrency)
    except (TypeError, ValueError) as exc:
        raise MergeError(f"{spec.run_dir} concurrency is invalid") from exc
    if str(parsed_concurrency) != concurrency or parsed_concurrency <= 0:
        raise MergeError(f"{spec.run_dir} concurrency is not canonical and positive")
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
        raise MergeError(f"{spec.run_dir} request audit metadata is invalid")
    try:
        auditor.audit_provider_evidence(spec.run_dir, manifest)
    except (auditor.AuditError, OSError, json.JSONDecodeError) as exc:
        raise MergeError(
            f"{spec.run_dir} has invalid Flex provider evidence: {exc}"
        ) from exc


def validate_no_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise MergeError(f"symbolic links are not accepted: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise MergeError(f"symbolic links are not accepted: {path}")
        if not path.is_dir() and not path.is_file():
            raise MergeError(f"non-regular filesystem entry is not accepted: {path}")


def payload_sha256(item_dir: Path) -> str:
    """Hash every item artifact except checkpoint.json and its absolute paths."""
    entries: list[dict[str, Any]] = []
    for path in sorted(item_dir.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(item_dir).as_posix()
        if relative == "checkpoint.json":
            continue
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
            raise MergeError(f"unsupported item artifact: {path}")
    return sha256_json(entries)


def exact_checkpoint_fields(
    checkpoint: dict[str, Any], reference: dict[str, Any], index: int
) -> None:
    expected = {
        "question_type": reference["question_type"],
        "question": reference["question"],
        "gold": reference["answer"],
        "question_date_raw": reference["question_date"],
        "answer_session_ids": reference["answer_session_ids"],
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise MergeError(f"item {index} {key} differs from the dataset")
    input_meta = checkpoint.get("input")
    if not isinstance(input_meta, dict):
        raise MergeError(f"item {index} input metadata is missing")
    sessions = reference["haystack_sessions"]
    empty_turns = sum(
        not str(turn.get("content", "")).strip()
        for session in sessions for turn in session
    )
    input_expected = {
        "sessions": len(sessions),
        "turns": sum(len(session) for session in sessions),
        "empty_source_turns": empty_turns,
        "source_session_ids": reference["haystack_session_ids"],
        "source_dates": reference["haystack_dates"],
    }
    for key, value in input_expected.items():
        if input_meta.get(key) != value:
            raise MergeError(f"item {index} input.{key} differs from the dataset")


def validate_item(
    checkpoint_path: Path, manifest: dict[str, Any], reference: dict[str, Any],
    index: int, source_dir: Path,
) -> ValidatedItem:
    expected_item_dir = runner.item_paths(
        source_dir, index, reference["question_id"]
    )["item_dir"].resolve()
    item_dir = checkpoint_path.parent.resolve()
    if item_dir != expected_item_dir:
        raise MergeError(
            f"item {index} directory differs from runner naming: {item_dir}"
        )
    validate_no_symlinks(item_dir)
    checkpoint = read_json(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise MergeError(f"item {index} checkpoint is not an object")
    if checkpoint.get("schema_version") != runner.SCHEMA_VERSION:
        raise MergeError(f"item {index} checkpoint schema differs")
    if checkpoint.get("status") != "complete":
        raise MergeError(f"item {index} status is not complete")
    if checkpoint.get("dataset_index") != index:
        raise MergeError(f"item {index} dataset_index differs")
    question_id = str(reference["question_id"])
    if str(checkpoint.get("question_id")) != question_id:
        raise MergeError(f"item {index} question_id differs from the dataset")
    exact_checkpoint_fields(checkpoint, reference, index)
    for key in ("method", "models", "config", "code", "request_audit"):
        if checkpoint.get(key) != manifest.get(key):
            raise MergeError(f"item {index} {key} differs from its run manifest")

    paths = checkpoint.get("paths")
    expected_paths = {
        "item_dir": str(item_dir),
        "memory_dir": str(item_dir / "memory"),
        "checkpoint": str((item_dir / "checkpoint.json").resolve()),
    }
    if not isinstance(paths, dict) or paths != expected_paths:
        raise MergeError(f"item {index} paths do not bind to its source directory")
    memory_dir = item_dir / "memory"
    complete, reason = runner.checkpoint_is_complete(
        checkpoint, reference, index, memory_dir
    )
    if not complete:
        raise MergeError(f"item {index} is not durable: {reason}")

    build = checkpoint.get("build")
    retrieval = checkpoint.get("retrieval")
    answer = checkpoint.get("answer")
    if not all(isinstance(value, dict) for value in (build, retrieval, answer)):
        raise MergeError(f"item {index} stage metadata is incomplete")
    if answer.get("status") != "complete":
        raise MergeError(f"item {index} answer status is not complete")
    if answer.get("model") != manifest["models"]["answerer"]:
        raise MergeError(f"item {index} answer model differs from the manifest")
    for phase, usage in (
        ("build", build.get("usage")),
        ("retrieval", retrieval.get("usage")),
    ):
        if not isinstance(usage, dict):
            raise MergeError(f"item {index} {phase} usage is missing")
        if int(usage.get("calls", 0) or 0) <= 0:
            raise MergeError(f"item {index} {phase} has no recorded model calls")
        if int(usage.get("tokens_in", 0) or 0) <= 0:
            raise MergeError(f"item {index} {phase} has no recorded input tokens")
    if int(build.get("events", 0) or 0) <= 0:
        raise MergeError(f"item {index} build has no events")
    if int(retrieval.get("steps", 0) or 0) <= 0:
        raise MergeError(f"item {index} retrieval has no model steps")
    stats = runner.memory_stats(memory_dir)
    if build.get("markdown_files") != stats["markdown_files"]:
        raise MergeError(f"item {index} markdown file count differs")
    if build.get("bytes") != stats["bytes"]:
        raise MergeError(f"item {index} memory byte count differs")

    return ValidatedItem(
        index=index,
        question_id=question_id,
        item_dir=item_dir,
        checkpoint=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint_path),
        payload_sha256=payload_sha256(item_dir),
    )


def inventory_checkpoint_paths(
    spec: ShardSpec, manifest: dict[str, Any], references: list[dict[str, Any]],
    declared_selections: tuple[dict[str, Any], ...],
) -> tuple[dict[int, Path], tuple[dict[str, Any], ...]]:
    items_dir = spec.run_dir / "items"
    if not items_dir.is_dir():
        raise MergeError(f"source shard has no items directory: {items_dir}")
    item_dirs = [path for path in items_dir.iterdir() if path.is_dir()]
    expected = set(spec.indices)
    declared_indices: set[int] = set()
    for selection in declared_selections:
        declared_indices.update(
            runner.select_indices(
                len(references), selection["start"], selection["limit"]
            )
        )
    ignored: list[dict[str, Any]] = []
    missing_checkpoint_indices: set[int] = set()
    checkpoint_paths = []
    for item_dir in item_dirs:
        checkpoint_path = item_dir / "checkpoint.json"
        if checkpoint_path.is_file():
            checkpoint_paths.append(checkpoint_path)
            continue
        match = re.match(r"^(\d{4})_", item_dir.name)
        if not match:
            raise MergeError(
                f"source item directory has no checkpoint or valid index: {item_dir}"
            )
        index = int(match.group(1))
        if index < 0 or index >= len(references):
            raise MergeError(f"source item directory index is invalid: {item_dir}")
        expected_item_dir = runner.item_paths(
            spec.run_dir, index, references[index]["question_id"]
        )["item_dir"].resolve()
        if item_dir.resolve() != expected_item_dir:
            raise MergeError(
                f"source item directory differs from runner naming: {item_dir}"
            )
        if index in expected:
            raise MergeError(f"responsibility item {index} has no checkpoint")
        if index not in declared_indices:
            raise MergeError(
                f"source item {index} is outside all declared invocations"
            )
        if index in missing_checkpoint_indices:
            raise MergeError(f"source shard has duplicate partial item index {index}")
        validate_no_symlinks(item_dir)
        missing_checkpoint_indices.add(index)
        ignored.append({
            "dataset_index": index,
            "question_id": str(references[index]["question_id"]),
            "status": "missing_checkpoint",
            "checkpoint": None,
            "checkpoint_sha256": None,
            "payload_sha256": payload_sha256(item_dir),
            "copied": False,
        })
    inventory: dict[int, Path] = {}
    for checkpoint_path in checkpoint_paths:
        checkpoint = read_json(checkpoint_path)
        if not isinstance(checkpoint, dict):
            raise MergeError(f"checkpoint is not an object: {checkpoint_path}")
        index = checkpoint.get("dataset_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise MergeError(f"checkpoint has invalid dataset_index: {checkpoint_path}")
        if index in inventory:
            raise MergeError(f"source shard has duplicate item index {index}")
        if index in missing_checkpoint_indices:
            raise MergeError(f"source shard has duplicate item index {index}")
        inventory[index] = checkpoint_path
    actual = set(inventory)
    missing = expected - actual
    if missing:
        raise MergeError(
            f"{spec.run_dir} checkpoint coverage differs from assigned selection: "
            f"missing={sorted(missing)[:10]}"
        )
    for index in sorted(actual - expected):
        checkpoint_path = inventory[index]
        checkpoint = read_json(checkpoint_path)
        if index < 0 or index >= len(references):
            raise MergeError(
                f"{spec.run_dir} has an out-of-dataset checkpoint index {index}"
            )
        if index not in declared_indices:
            raise MergeError(
                f"{spec.run_dir} item {index} is outside all declared invocations"
            )
        if checkpoint.get("status") == "complete":
            raise MergeError(
                f"{spec.run_dir} has a completed item {index} outside its "
                "responsibility range"
            )
        reference = references[index]
        expected_item_dir = runner.item_paths(
            spec.run_dir, index, reference["question_id"]
        )["item_dir"].resolve()
        item_dir = checkpoint_path.parent.resolve()
        if item_dir != expected_item_dir:
            raise MergeError(
                f"ignored partial item {index} directory differs from runner naming"
            )
        validate_no_symlinks(item_dir)
        if checkpoint.get("dataset_index") != index:
            raise MergeError(f"ignored partial item {index} index differs")
        if str(checkpoint.get("question_id")) != str(reference["question_id"]):
            raise MergeError(f"ignored partial item {index} question id differs")
        for key in ("method", "models", "config", "code", "request_audit"):
            if checkpoint.get(key) != manifest.get(key):
                raise MergeError(
                    f"ignored partial item {index} {key} differs from manifest"
                )
        expected_paths = {
            "item_dir": str(item_dir),
            "memory_dir": str(item_dir / "memory"),
            "checkpoint": str((item_dir / "checkpoint.json").resolve()),
        }
        if checkpoint.get("paths") != expected_paths:
            raise MergeError(f"ignored partial item {index} paths differ")
        ignored.append({
            "dataset_index": index,
            "question_id": str(reference["question_id"]),
            "status": checkpoint.get("status"),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "copied": False,
        })
    ignored.sort(key=lambda value: int(value["dataset_index"]))
    return inventory, tuple(ignored)


def validate_shard(
    spec: ShardSpec, references: list[dict[str, Any]], data_path: Path,
    dataset_sha256: str,
) -> ValidatedShard:
    reject_symlink_components(spec.run_dir, "source run directory")
    reject_symlink_components(spec.run_dir / "items", "source items directory")
    manifest_path = spec.run_dir / "run_manifest.json"
    reject_symlink_components(manifest_path, "source manifest")
    if not manifest_path.is_file():
        raise MergeError(f"source manifest is missing: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise MergeError(f"source manifest is not an object: {manifest_path}")
    validate_common_manifest(
        manifest, spec, data_path, dataset_sha256, len(references)
    )
    declared = validate_declared_selections(manifest, spec, len(references))
    inventory, ignored_partials = inventory_checkpoint_paths(
        spec, manifest, references, declared
    )
    items = tuple(
        validate_item(
            inventory[index], manifest, references[index], index, spec.run_dir
        )
        for index in spec.indices
    )
    inventory_records = [{
        "dataset_index": item.index,
        "question_id": item.question_id,
        "checkpoint_sha256": item.checkpoint_sha256,
        "payload_sha256": item.payload_sha256,
    } for item in items]
    completed = manifest.get("completed")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise MergeError(f"{spec.run_dir} manifest completed count is invalid")
    if completed > len(items):
        raise MergeError(
            f"{spec.run_dir} manifest reports more completed items than exist"
        )
    return ValidatedShard(
        spec=spec,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        declared_selections=declared,
        items=items,
        ignored_partials=ignored_partials,
        inventory_sha256=sha256_json(inventory_records),
    )


def validate_shard_compatibility(shards: list[ValidatedShard]) -> None:
    reference = shards[0].manifest
    for shard in shards[1:]:
        mismatches = [
            key for key in COMPATIBILITY_KEYS
            if shard.manifest.get(key) != reference.get(key)
        ]
        if mismatches:
            raise MergeError(
                f"source shard manifests differ in {', '.join(mismatches)}"
            )


def rewritten_checkpoint(
    item: ValidatedItem, output_dir: Path
) -> tuple[dict[str, Any], Path]:
    target_dir = runner.item_paths(
        output_dir, item.index, item.question_id
    )["item_dir"].resolve()
    checkpoint = copy.deepcopy(item.checkpoint)
    checkpoint["paths"] = {
        "item_dir": str(target_dir),
        "memory_dir": str(target_dir / "memory"),
        "checkpoint": str(target_dir / "checkpoint.json"),
    }
    return checkpoint, target_dir


def clean_stale_temporaries(items_dir: Path) -> None:
    for path in items_dir.glob(".merge-item-*"):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def copy_or_validate_item(
    item: ValidatedItem, output_dir: Path, resume: bool
) -> bool:
    expected_checkpoint, target_dir = rewritten_checkpoint(item, output_dir)
    if target_dir.exists():
        if not resume:
            raise MergeError(f"destination item already exists: {target_dir}")
        validate_no_symlinks(target_dir)
        target_checkpoint = read_json(target_dir / "checkpoint.json")
        if target_checkpoint != expected_checkpoint:
            raise MergeError(
                f"existing destination checkpoint differs for item {item.index}"
            )
        if payload_sha256(target_dir) != item.payload_sha256:
            raise MergeError(
                f"existing destination payload differs for item {item.index}"
            )
        return False

    items_dir = target_dir.parent
    temporary = Path(tempfile.mkdtemp(prefix=".merge-item-", dir=items_dir))
    try:
        shutil.copytree(item.item_dir, temporary, dirs_exist_ok=True, copy_function=shutil.copy2)
        atomic_json(temporary / "checkpoint.json", expected_checkpoint)
        validate_no_symlinks(temporary)
        if payload_sha256(temporary) != item.payload_sha256:
            raise MergeError(f"copied payload differs for item {item.index}")
        if read_json(temporary / "checkpoint.json") != expected_checkpoint:
            raise MergeError(f"rewritten checkpoint differs for item {item.index}")
        os.replace(temporary, target_dir)
        directory_fd = os.open(items_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return True


def validate_destination_inventory(output_dir: Path, expected_total: int) -> None:
    items_root = output_dir / "items"
    children = list(items_root.iterdir())
    invalid_children = sorted(
        path.name
        for path in children
        if path.is_symlink() or not path.is_dir() or path.name.startswith(".")
    )
    if invalid_children:
        raise MergeError(
            f"destination items contain unexpected artifacts: {invalid_children}"
        )
    item_dirs = children
    indices: list[int] = []
    for item_dir in item_dirs:
        checkpoint = read_json(item_dir / "checkpoint.json")
        if not isinstance(checkpoint, dict):
            raise MergeError(f"destination checkpoint is invalid: {item_dir}")
        index = checkpoint.get("dataset_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise MergeError(f"destination checkpoint index is invalid: {item_dir}")
        expected_paths = {
            "item_dir": str(item_dir.resolve()),
            "memory_dir": str((item_dir / "memory").resolve()),
            "checkpoint": str((item_dir / "checkpoint.json").resolve()),
        }
        if checkpoint.get("paths") != expected_paths:
            raise MergeError(f"destination item {index} paths were not rewritten")
        if not runner.has_durable_complete_payload(checkpoint):
            raise MergeError(f"destination item {index} is not durable")
        indices.append(index)
    if sorted(indices) != list(range(expected_total)):
        raise MergeError("destination item inventory does not cover the dataset exactly")


def validate_destination_root_layout(output_dir: Path) -> None:
    allowed = {
        ".launcher.lock",
        "items",
        "results.json",
        "hypotheses.jsonl",
        "run_manifest.json",
        "evaluation_input.json",
        "audit.json",
        "provider_evidence",
    }
    unexpected = sorted(
        path.name for path in output_dir.iterdir() if path.name not in allowed
    )
    if unexpected:
        raise MergeError(
            f"destination contains unexpected root artifacts: {unexpected}"
        )


def aggregate_records(
    items: list[ValidatedItem], output_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    checkpoints = []
    for item in sorted(items, key=lambda value: value.index):
        _expected, target_dir = rewritten_checkpoint(item, output_dir)
        checkpoint = read_json(target_dir / "checkpoint.json")
        checkpoints.append(checkpoint)
    results = [{
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
    hypotheses = "".join(
        json.dumps({
            "question_id": item["question_id"],
            "hypothesis": item["answer"]["hypothesis"],
        }, ensure_ascii=False) + "\n"
        for item in checkpoints
    )
    return results, hypotheses


def source_descriptor(shard: ValidatedShard) -> dict[str, Any]:
    return {
        "run_dir": str(shard.spec.run_dir),
        "manifest_sha256": shard.manifest_sha256,
        "assigned_selection": {
            "start": shard.spec.start,
            "limit": shard.spec.limit,
            "indices": [shard.spec.start, shard.spec.stop - 1],
        },
        "declared_invocations": list(shard.declared_selections),
        "items": len(shard.items),
        "inventory_sha256": shard.inventory_sha256,
        "reported_completed": shard.manifest.get("completed"),
        "reported_status": shard.manifest.get("status"),
        "ignored_partials": list(shard.ignored_partials),
    }


def build_manifest(
    shards: list[ValidatedShard], output_dir: Path, expected_total: int,
    results_path: Path, hypotheses_path: Path, inventory: dict[str, Any],
    provider_evidence: dict[str, Any],
) -> dict[str, Any]:
    existing_path = output_dir / "run_manifest.json"
    source_descriptors = [source_descriptor(shard) for shard in shards]
    source_fingerprint = sha256_json(source_descriptors)
    existing: dict[str, Any] | None = None
    if existing_path.exists():
        candidate = read_json(existing_path)
        if not isinstance(candidate, dict):
            raise MergeError("existing destination manifest is not an object")
        merge = candidate.get("merge")
        if (
            not isinstance(merge, dict)
            or merge.get("source_fingerprint") != source_fingerprint
        ):
            raise MergeError("existing destination manifest has different sources")
        for key in COMPATIBILITY_KEYS:
            if candidate.get(key) != shards[0].manifest.get(key):
                raise MergeError(
                    f"existing destination manifest differs in {key}"
                )
        existing = candidate

    merged_at = (
        existing["merge"]["merged_at"] if existing is not None else utc_now()
    )
    base = copy.deepcopy(existing) if existing is not None else {
        key: copy.deepcopy(shards[0].manifest[key])
        for key in COMPATIBILITY_KEYS
    }
    if existing is None:
        created_times = [
            parse_timestamp(
                shard.manifest.get("created_at"),
                f"{shard.spec.run_dir} created_at",
            )
            for shard in shards
        ]
        base["created_at"] = min(created_times)
        base["invocations"] = [{
            "started_at": merged_at,
            "start": 0,
            "limit": expected_total,
            "indices": [0, expected_total - 1],
            "resume": False,
            "mode": "merge_completed_shards",
            "model_requests": 0,
        }]
        base["last_invocation_finished_at"] = merged_at
        base["last_invocation_failures"] = []
        base["updated_at"] = merged_at
        base["output_dir"] = str(output_dir)
    base.update({
        "completed": expected_total,
        "checkpoint_counts": {"complete": expected_total},
        "status": "complete",
        "provider_evidence": provider_evidence,
    })
    base["merge"] = {
        "schema_version": 1,
        "tool": Path(__file__).name,
        "tool_sha256": sha256_file(Path(__file__)),
        "merged_at": merged_at,
        "no_model_requests": True,
        "coverage": {
            "start": 0,
            "limit": expected_total,
            "indices": [0, expected_total - 1],
        },
        "source_fingerprint": source_fingerprint,
        "sources": source_descriptors,
        "aggregates": {
            "results_json": {
                "path": str(results_path),
                "sha256": sha256_file(results_path),
            },
            "hypotheses_jsonl": {
                "path": str(hypotheses_path),
                "sha256": sha256_file(hypotheses_path),
            },
        },
        "destination_inventory": inventory,
    }
    return base


def copy_provider_evidence(
    shards: list[ValidatedShard], output_dir: Path
) -> dict[str, Any]:
    invocations: list[dict[str, Any]] = []
    gateway_roots = {
        str(shard.manifest["provider_evidence"]["gateway_root"])
        for shard in shards
    }
    if len(gateway_roots) != 1:
        raise MergeError("source shards use different Flex gateway roots")
    root = output_dir / "provider_evidence"
    if root.exists():
        raise MergeError("destination provider_evidence already exists")
    for shard_index, shard in enumerate(shards):
        for invocation_index, source_record in enumerate(
            shard.manifest["provider_evidence"]["invocations"]
        ):
            source_record_path = Path(source_record["record_path"]).resolve()
            source_dir = source_record_path.parent
            target = root / f"shard-{shard_index:03d}" / (
                f"invocation-{invocation_index:03d}"
            )
            shutil.copytree(source_dir, target, copy_function=shutil.copy2)
            stored = read_json(source_record_path)

            def rewrite(value: Any) -> Any:
                if isinstance(value, dict):
                    return {key: rewrite(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [rewrite(item) for item in value]
                if isinstance(value, str) and value.startswith(f"{source_dir}{os.sep}"):
                    return f"{target}{value[len(str(source_dir)):]}"
                return value

            rewritten = rewrite(stored)
            rewritten["window"]["path"] = str(target / "gateway_window.json")
            rewritten["consumer_log"]["path"] = str(
                target / "child_proxy_requests.jsonl"
            )
            ready_path = target / "child_proxy_ready.json"
            ready_record = rewrite(read_json(ready_path))
            atomic_json(ready_path, ready_record)
            rewritten["child_ready"]["path"] = str(ready_path)
            rewritten["child_ready"]["sha256"] = sha256_file(ready_path)
            record_path = target / "invocation.json"
            atomic_json(record_path, rewritten)
            manifest_record = copy.deepcopy(rewritten)
            manifest_record["record_path"] = str(record_path)
            manifest_record["record_sha256"] = sha256_file(record_path)
            invocations.append(manifest_record)
    return {
        "schema": "openai-gpt55-flex-invocations/v1",
        "gateway_root": gateway_roots.pop(),
        "active_run_id": None,
        "invocations": invocations,
    }


def merge_shards(
    specs: list[ShardSpec], output_dir: Path, data_path: Path, *,
    resume: bool = False, expected_total: int = EXPECTED_ITEMS,
) -> dict[str, Any]:
    specs = sorted([
        ShardSpec(
            reject_symlink_components(
                spec.run_dir, "source run directory"
            ).resolve(),
            spec.start,
            spec.limit,
        )
        for spec in specs
    ], key=lambda value: (value.start, value.stop, str(value.run_dir)))
    output_dir = reject_symlink_components(
        output_dir, "output directory"
    ).resolve()
    data_path = data_path.expanduser().resolve()
    validate_specs(specs, expected_total)
    validate_path_layout(specs, output_dir)
    if not data_path.is_file():
        raise MergeError(f"dataset does not exist: {data_path}")
    try:
        references = runner.load_dataset(data_path, expected_count=expected_total)
        runner.validate_longmemeval_s(references)
    except runner.DataValidationError as exc:
        raise MergeError(str(exc)) from exc
    dataset_sha256 = sha256_file(data_path)

    if output_dir.exists() and not resume:
        raise MergeError(
            f"output directory already exists; use --resume only for this merge: "
            f"{output_dir}"
        )
    copied = 0
    reused = 0
    with ExitStack() as stack:
        for spec in sorted(specs, key=lambda value: str(value.run_dir)):
            if not spec.run_dir.is_dir():
                raise MergeError(f"source shard does not exist: {spec.run_dir}")
            stack.enter_context(directory_lock(spec.run_dir, shared=True))

        shards = [
            validate_shard(spec, references, data_path, dataset_sha256)
            for spec in specs
        ]
        validate_shard_compatibility(shards)

        created_output = False
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=False)
                created_output = True
            except FileExistsError as exc:
                raise MergeError(f"output directory was created concurrently: {output_dir}") from exc
        if created_output or resume:
            stack.enter_context(directory_lock(output_dir, shared=False))
        validate_destination_root_layout(output_dir)
        items_dir = output_dir / "items"
        items_dir.mkdir(parents=True, exist_ok=True)
        clean_stale_temporaries(items_dir)

        all_items = [item for shard in shards for item in shard.items]
        for item in sorted(all_items, key=lambda value: value.index):
            if copy_or_validate_item(item, output_dir, resume=resume):
                copied += 1
            else:
                reused += 1
        validate_destination_inventory(output_dir, expected_total)

        results, hypotheses = aggregate_records(all_items, output_dir)
        results_path = output_dir / "results.json"
        hypotheses_path = output_dir / "hypotheses.jsonl"
        atomic_json(results_path, results)
        atomic_text(hypotheses_path, hypotheses)
        existing_manifest_path = output_dir / "run_manifest.json"
        if (output_dir / "provider_evidence").exists():
            if not resume or not existing_manifest_path.is_file():
                raise MergeError(
                    "destination provider_evidence exists without a resumable manifest"
                )
            previous_manifest = read_json(existing_manifest_path)
            provider_evidence = copy.deepcopy(
                previous_manifest.get("provider_evidence")
            )
            try:
                auditor.audit_provider_evidence(output_dir, previous_manifest)
            except (auditor.AuditError, OSError, json.JSONDecodeError) as exc:
                raise MergeError(
                    f"destination Flex provider evidence is invalid: {exc}"
                ) from exc
        else:
            provider_evidence = copy_provider_evidence(shards, output_dir)
        validate_destination_root_layout(output_dir)
        inventory = merge_payload_tree_inventory(output_dir)
        manifest = build_manifest(
            shards,
            output_dir,
            expected_total,
            results_path,
            hypotheses_path,
            inventory,
            provider_evidence,
        )
        atomic_json(output_dir / "run_manifest.json", manifest)
        if expected_total == EXPECTED_ITEMS and data_path == DEFAULT_DATA.resolve():
            try:
                records, report = auditor._audit_unlocked(output_dir)
            except (auditor.AuditError, OSError, json.JSONDecodeError) as exc:
                raise MergeError(f"strict LongMemEval audit failed: {exc}") from exc
            if len(records) != 1000 or report.get("status") != "passed":
                raise MergeError(
                    "strict LongMemEval audit returned an incomplete result"
                )
        else:
            try:
                auditor.audit_provider_evidence(output_dir, manifest)
            except (auditor.AuditError, OSError, json.JSONDecodeError) as exc:
                raise MergeError(
                    f"synthetic Flex provider evidence failed: {exc}"
                ) from exc

    return {
        "status": "complete",
        "output_dir": str(output_dir),
        "items": expected_total,
        "copied": copied,
        "reused": reused,
        "results_sha256": sha256_file(output_dir / "results.json"),
        "hypotheses_sha256": sha256_file(output_dir / "hypotheses.jsonl"),
        "manifest_sha256": sha256_file(output_dir / "run_manifest.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly merge completed LongMemEval-S responsibility ranges "
            "without model requests"
        )
    )
    parser.add_argument(
        "--input", action="append", required=True, metavar="DIR:START-END",
        help=(
            "source run directory and inclusive responsibility range; repeat "
            "for every shard"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--resume", action="store_true")
    return parser


def parse_input_spec(value: str) -> ShardSpec:
    try:
        directory, interval = value.rsplit(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "input must use DIR:START-END"
        ) from exc
    match = re.fullmatch(r"(\d+)-(\d+)", interval.strip())
    if not directory.strip() or not match:
        raise argparse.ArgumentTypeError("input must use DIR:START-END")
    start, end = map(int, match.groups())
    if end < start:
        raise argparse.ArgumentTypeError("input END must be at least START")
    return ShardSpec(Path(directory), start, end - start + 1)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        specs = [parse_input_spec(value) for value in args.input]
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    try:
        report = merge_shards(
            specs, args.output_dir, args.data, resume=args.resume,
            expected_total=EXPECTED_ITEMS,
        )
    except MergeError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
