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
from .source_archive import SourceArchiveMixin
from .topic_normalization import TopicNormalizationMixin
from .topic_reconciliation import TopicReconciliationMixin


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
        self.stage_dir = Path(tempfile.mkdtemp(prefix="nativemem-topics-"))
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
        runtime = self.memory_dir / ".nativemem" / "runtime.json"
        if runtime.exists():
            staged_runtime = self.stage_dir / ".nativemem" / "runtime.json"
            staged_runtime.parent.mkdir(parents=True, exist_ok=True)
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
            if path.is_file() and ".nativemem" not in path.relative_to(
                self.stage_dir
            ).parts
        ]
        return "\n".join(paths) or "(empty workspace)"
