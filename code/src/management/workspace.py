"""Transactional editable memory workspace."""

import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..markdown import parse_topic_tree, topic_prose
from .block_views import BlockViewsMixin
from .config import MemoryConfig
from .event_writing import EventWritingMixin
from .patching import apply_patch
from .source_archive import SourceArchiveMixin
from .topic_normalization import TopicNormalizationMixin
from .topic_reconciliation import TopicReconciliationMixin
from .transaction import (
    TransactionError,
    TransactionLimits,
    TransactionResult,
    committed_baseline,
    git_commit_state,
    install_state,
    parse_sources,
    resolve_source_labels,
    source_records,
    workspace_revision,
    workspace_write_lock,
)
from ..workspace_layout import TEMPORARY_PREFIX, is_internal_path, runtime_dir


class MemoryWorkspace(
    TopicNormalizationMixin,
    TopicReconciliationMixin,
    SourceArchiveMixin,
    EventWritingMixin,
    BlockViewsMixin,
):
    def __init__(
        self,
        memory_dir: str | Path,
        reconciler: Any | None = None,
        *,
        config: MemoryConfig | None = None,
    ):
        self.memory_dir = Path(memory_dir).resolve()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.stage_dir = Path(tempfile.mkdtemp(prefix=f"{TEMPORARY_PREFIX}topics-"))
        self.pending: dict[str, dict[str, Any]] = {}
        self.reconciler = reconciler
        self.config = config or MemoryConfig()
        self.committed = False
        self.last_changed_topics: list[str] = []
        self.last_created_blocks = 0
        self._refresh_stage()

    def _refresh_stage(self) -> None:
        shutil.rmtree(self.stage_dir, ignore_errors=True)
        self.stage_dir.mkdir()
        for name in ("topics", "timeline", "sources"):
            source = self.memory_dir / name
            if source.exists():
                shutil.copytree(source, self.stage_dir / name)
        (self.stage_dir / "topics").mkdir(exist_ok=True)
        recent = self.memory_dir / "recent_events.jsonl"
        if recent.exists():
            shutil.copy2(recent, self.stage_dir / recent.name)
        relations = self.memory_dir / "relations.json"
        if relations.exists():
            shutil.copy2(relations, self.stage_dir / relations.name)
        core = self.memory_dir / "core.md"
        if core.exists():
            shutil.copy2(core, self.stage_dir / core.name)
        runtime = runtime_dir(self.memory_dir) / "runtime.json"
        staged_runtime = self.stage_dir / runtime.parent.name / "runtime.json"
        staged_runtime.parent.mkdir(parents=True, exist_ok=True)
        if runtime.exists():
            shutil.copy2(runtime, staged_runtime)

    def shell(
        self, command: str, *, allow_correction: bool = False
    ) -> subprocess.CompletedProcess[str]:
        self.last_changed_topics = []
        self.last_created_blocks = 0
        before = self._workspace_fingerprint()
        before_topics = self._topic_fingerprints(self.stage_dir / "topics")
        before_sources = self._tree_fingerprint(self.stage_dir / "sources")
        before_units = parse_topic_tree(self.stage_dir / "topics")
        before_block_ids = {unit.memory_id for unit in before_units}
        core = self.stage_dir / "core.md"
        if core.is_file():
            before_block_ids.update(re.findall(
                r"(?m)\^([A-Za-z0-9-]+)\s*$",
                core.read_text(encoding="utf-8"),
            ))
        before_prose = topic_prose(self.stage_dir / "topics")
        result = subprocess.run(
            command,
            cwd=self.stage_dir,
            shell=True,
            text=True,
            capture_output=True,
            timeout=120,
        )
        changed = self._workspace_fingerprint() != before
        if result.returncode != 0 and changed:
            self._refresh_stage()
        elif result.returncode == 0 and changed:
            try:
                if self._tree_fingerprint(self.stage_dir / "sources") != before_sources:
                    raise ValueError("Source Memory is append-only")
                self._normalize_topic_edits(before_block_ids)
                self._validate_topic_contract(before_units)
                self._reconcile_topic_edit(
                    before_units, before_prose, allow_correction=allow_correction
                )
                self._synchronize()
                after_units = parse_topic_tree(self.stage_dir / "topics")
                after_topics = self._topic_fingerprints(self.stage_dir / "topics")
                self.last_changed_topics = [
                    "topics/" + path
                    for path in sorted(set(before_topics) | set(after_topics))
                    if before_topics.get(path) != after_topics.get(path)
                ]
                self.last_created_blocks = len(
                    {unit.memory_id for unit in after_units}
                    - {unit.memory_id for unit in before_units}
                )
            except Exception:
                self._refresh_stage()
                raise
        return result

    def revision(self) -> str:
        """Fingerprint of the committed workspace, for optimistic concurrency."""
        return workspace_revision(self.memory_dir)

    def update(
        self,
        *,
        base_revision: str,
        patch: str,
        sources: Any = None,
        commit_message: str | None = None,
        git_commit: str = "auto",
        limits: TransactionLimits | None = None,
    ) -> TransactionResult:
        """Apply sources and a topic patch as one atomic transaction.

        Sources are archived into the stage rather than ``memory_dir`` so that
        evidence and the topics citing it become visible in the same install.
        """
        if git_commit not in ("auto", "on", "off"):
            raise TransactionError(
                "INVALID_ARGUMENT", "git_commit must be auto, on or off"
            )
        limits = limits or TransactionLimits()
        if len(patch.encode("utf-8")) > limits.max_patch_bytes:
            raise TransactionError(
                "INVALID_ARGUMENT",
                f"patch exceeds {limits.max_patch_bytes} bytes",
            )
        message = (commit_message or "Update memory").replace("\n", " ").strip()
        message = "".join(
            char for char in message if char.isprintable()
        )[:limits.max_commit_message_chars] or "Update memory"
        parsed_sources = parse_sources(sources, limits)

        with workspace_write_lock(self.memory_dir):
            current = workspace_revision(self.memory_dir)
            if base_revision != current:
                raise TransactionError(
                    "CONCURRENT_UPDATE",
                    "workspace changed since base_revision was read",
                    details={"base_revision": base_revision, "revision": current},
                )
            self._refresh_stage()
            before_files = self._committed_files()
            before_units, before_block_ids = committed_baseline(self)
            try:
                records = source_records(parsed_sources)
                mapping = {
                    item.label: record.source_id
                    for item, record in zip(parsed_sources, records)
                }
                if records:
                    self.archive_source_records(records, root=self.stage_dir)
                resolved = resolve_source_labels(patch, mapping)
                changed = apply_patch(self.stage_dir, resolved)
                install_state(self, before_units, before_block_ids)
            except TransactionError:
                self._refresh_stage()
                raise
            except Exception as exc:
                self._refresh_stage()
                raise TransactionError(
                    "INVALID_TOPIC_FORMAT", str(exc)
                ) from exc

            after_files = self._committed_files()
            changed_files = sorted(
                set(changed)
                | {
                    path
                    for path in set(before_files) | set(after_files)
                    if before_files.get(path) != after_files.get(path)
                }
            )
            result = TransactionResult(
                revision=workspace_revision(self.memory_dir),
                source_ids=mapping,
                block_ids=dict(getattr(self, "last_block_id_map", {}) or {}),
                evidence_ids=dict(
                    getattr(self, "last_evidence_id_map", {}) or {}
                ),
                changed_files=changed_files,
            )
            if git_commit == "off":
                return result
            try:
                commit = git_commit_state(self.memory_dir, message)
            except Exception as exc:
                if git_commit == "on":
                    raise TransactionError(
                        "GIT_COMMIT_FAILED",
                        # Files are already installed; a failed commit does not
                        # undo them and must not be reported as a rollback.
                        f"memory committed but git commit failed: {exc}",
                        details={"memory_committed": True, "git_committed": False},
                    ) from exc
                return result
            if commit is None and git_commit == "on":
                raise TransactionError(
                    "GIT_COMMIT_FAILED",
                    "memory committed but workspace is not a git repository",
                    details={"memory_committed": True, "git_committed": False},
                )
            result.git_commit = commit
            result.git_committed = commit is not None
            result.revision = workspace_revision(self.memory_dir)
            return result

    def _committed_files(self) -> dict[str, str]:
        root = self.memory_dir
        result = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if is_internal_path(relative):
                continue
            result[relative.as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        return result

    def _workspace_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(self.stage_dir.rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(self.stage_dir).as_posix().encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def structure(self) -> str:
        paths = [
            path.relative_to(self.stage_dir).as_posix()
            for path in sorted(self.stage_dir.rglob("*"))
            if path.is_file() and not is_internal_path(
                path.relative_to(self.stage_dir)
            )
        ]
        return "\n".join(paths) or "(empty workspace)"
