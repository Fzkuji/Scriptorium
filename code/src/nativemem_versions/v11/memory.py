"""V11 agentic memory writer and manager."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
import sys
import unicodedata
from urllib.parse import quote
from pathlib import Path
from typing import Any

import tiktoken

from .topic_markdown import parse_topic_tree, topic_prose
from .reconciliation import (
    ReconciliationError,
    ReconciliationResult,
    apply_reconciliation,
    classify_topic_diff,
)
from .runtime_state import SourceRecord


SYSTEM_PROMPT = """You manage a long-term memory workspace.

Inspect the supplied workspace structure and existing documents before making changes. Use the shell to read and organize topic files.

Preserve concrete facts, dates, source references, and the complete history of events and state changes. Do not replace historical events with only the latest state.

Use `save_memory` to store events in appropriate topic files and heading paths. The code synchronizes timeline, recent events, sources, and links."""

WRITER_TASK = """Integrate the following conversation session into the memory workspace.

Review the supplied workspace structure and relevant existing documents. Save all useful information with `save_memory`, reusing and reorganizing existing files and headings where appropriate.

Preserve the session as a dated historical observation.

Use the shell to update `core.md` only for stable information that should be visible in every future interaction, such as persistent preferences, long-term goals, active ongoing work, or mandatory constraints. Keep source references in Core Memory.

Observation date:
{observation_date}

Conversation:
{conversation}"""

WRITER_BATCH_TASK = """Integrate the following conversation sessions into the memory workspace.

Review the supplied workspace structure and relevant existing documents. Save all useful information with `save_memory`, reusing and reorganizing existing files and headings where appropriate.

Preserve the sessions as dated historical observations.

Use the shell to update `core.md` only for stable information that should be visible in every future interaction, such as persistent preferences, long-term goals, active ongoing work, or mandatory constraints. Keep source references in Core Memory.

Sessions:
{sessions}"""

MANAGER_TASK = """Organize the topic files into a coherent structure.

Use the supplied workspace structure and shell. Split or merge existing topic files and headings when appropriate. Preserve every `memory-event` marker and the complete dated history."""

LOCAL_MANAGER_TASK = """Organize only the following recently updated topic files and their local structure.

Limit this maintenance pass to these topic files. Merge redundant headings, split or combine local files when useful, and repair their local links. Do not reorganize unrelated topics. Preserve every `memory-event` marker, source reference, and the complete dated history.

Touched topic files:
{topic_paths}"""

VERIFICATION_PROBE_TASK = """Select one concrete factual detail from this session that should be recoverable from long-term memory.

Output only JSON with this shape:
{{"question":"a natural factual question","expected_answer":"the source-grounded answer","refs":["D1:1"]}}

Observation date:
{observation_date}

Conversation:
{conversation}"""

VERIFICATION_RETRIEVAL_TASK = """Answer this question using the supplied memory workspace.

Inspect whichever memory views, files, and sections you consider appropriate. Do not assume where the answer should be stored.

Question: {question}

After inspection, output exactly one <answer>...</answer> block."""

VERIFICATION_REPAIR_TASK = """Repair the memory workspace so that the question can be answered through the memory organization.

Inspect the source records, the existing memory, and the retrieval trace. Make any changes you consider useful. You may add, revise, move, merge, reorder, or remove memory content while preserving valid historical information and source grounding.

Question: {question}
Expected source-grounded answer: {expected_answer}
Source references: {refs}
Previous retrieval answer: {retrieved_answer}
Previous retrieval trace:
{trace}"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command in the memory workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save events into topic files; code synchronizes the other memory views.",
            "parameters": {
                "type": "object",
                "properties": {
                    "events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "when": {"type": "string"},
                                "content": {"type": "string"},
                                "topic_path": {
                                    "type": "string",
                                    "description": (
                                        "Markdown path relative to the topics root; "
                                        "do not include a leading topics/ directory"
                                    ),
                                },
                                "headings": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                    "maxItems": 6,
                                },
                                "refs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["when", "content", "refs", "topic_path", "headings"],
                        },
                    }
                },
                "required": ["events"],
            },
        },
    },
]


class MemoryWorkspace:
    def __init__(self, memory_dir: str | Path, reconciler: Any | None = None):
        self.memory_dir = Path(memory_dir).resolve()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.stage_dir = Path(tempfile.mkdtemp(prefix="v11-topics-"))
        self.pending: dict[str, dict[str, Any]] = {}
        self.reconciler = reconciler
        self.committed = False
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
        core = self.memory_dir / "core.md"
        if core.exists():
            shutil.copy2(core, self.stage_dir / core.name)

    def shell(
        self, command: str, *, allow_correction: bool = False
    ) -> subprocess.CompletedProcess[str]:
        before = self._workspace_fingerprint()
        before_sources = self._tree_fingerprint(self.stage_dir / "sources")
        before_units = parse_topic_tree(self.stage_dir / "topics")
        before_prose = topic_prose(self.stage_dir / "topics")
        result = subprocess.run(
            command,
            cwd=self.stage_dir,
            shell=True,
            text=True,
            capture_output=True,
            timeout=120,
        )
        if result.returncode == 0 and self._workspace_fingerprint() != before:
            try:
                if self._tree_fingerprint(self.stage_dir / "sources") != before_sources:
                    raise ValueError("Source Memory is append-only")
                self._reconcile_topic_edit(
                    before_units, before_prose, allow_correction=allow_correction
                )
                self._synchronize()
            except Exception:
                self._refresh_stage()
                raise
        return result

    @staticmethod
    def _tree_fingerprint(root: Path) -> str:
        digest = hashlib.sha256()
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    digest.update(path.relative_to(root).as_posix().encode())
                    digest.update(path.read_bytes())
        return digest.hexdigest()

    def _reconcile_topic_edit(
        self,
        before_units: list[Any],
        before_prose: str,
        *,
        allow_correction: bool = False,
    ) -> None:
        after_units = parse_topic_tree(self.stage_dir / "topics", strict=False)
        diff = classify_topic_diff(before_units, after_units)
        prose_changed = topic_prose(self.stage_dir / "topics") != before_prose
        if (
            not diff.reconciler_required
            and not prose_changed
        ) or self.reconciler is None:
            return
        affected = set(diff.changed_ids) | set(diff.removed_ids)
        old_units = [unit for unit in before_units if unit.memory_id in affected]
        edited_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((self.stage_dir / "topics").rglob("*.md"))
        )
        candidate_sources = {
            ref
            for unit in before_units + after_units
            for ref in unit.source_refs
        }
        for source in (self.stage_dir / "sources").glob("D*.md"):
            for conversation, turn in re.findall(
                r'<a id="d(\d+)-(\d+)"></a>',
                source.read_text(encoding="utf-8"),
            ):
                candidate_sources.add(f"D{conversation}:{turn}")
        for source in (self.stage_dir / "sources").rglob("*.md"):
            candidate_sources.update(re.findall(
                r"<!-- source-id:([^>]+) -->",
                source.read_text(encoding="utf-8"),
            ))
        for attempt in range(2):
            result = self.reconciler(edited_text, old_units, candidate_sources)
            try:
                apply_reconciliation(
                    edited_text,
                    old_units,
                    result,
                    candidate_sources=candidate_sources,
                    allow_correction=allow_correction,
                )
                break
            except ReconciliationError:
                if attempt == 1:
                    raise
        self._materialize_reconciliation(result, old_units)

    def _materialize_reconciliation(
        self, result: Any, old_units: list[Any]
    ) -> None:
        topics = self.stage_dir / "topics"
        old_by_id = {unit.memory_id: unit for unit in old_units}
        affected = set(result.matches) | set(result.deleted_ids)
        citation = re.compile(
            r"\[\^(" + "|".join(re.escape(value) for value in affected) + r")\]"
        ) if affected else None
        definition = re.compile(
            r"^\[\^(" + "|".join(re.escape(value) for value in affected) + r")\]:"
        ) if affected else None
        for path in topics.rglob("*.md"):
            rendered = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if definition and definition.match(line):
                    continue
                rendered.append(citation.sub("", line) if citation else line)
            path.write_text("\n".join(rendered).rstrip() + "\n", encoding="utf-8")

        placements: dict[str, list[tuple[str, str | None, tuple[str, ...]]]] = {}
        for memory_id, content in result.matches.items():
            unit = old_by_id[memory_id]
            placements.setdefault(content, []).append(
                (memory_id, unit.when, unit.source_refs)
            )
        for content, when, refs in result.creates:
            payload = json.dumps(
                [when, content, list(refs)],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            memory_id = "mem_" + hashlib.sha256(payload.encode()).hexdigest()[:16]
            placements.setdefault(content, []).append((memory_id, when, refs))

        definitions: dict[Path, list[str]] = {}
        for content, rows in placements.items():
            candidates = []
            for path in topics.rglob("*.md"):
                body = "\n".join(
                    line for line in path.read_text(encoding="utf-8").splitlines()
                    if not line.startswith("[^mem_")
                )
                if body.count(content) == 1:
                    candidates.append(path)
            if len(candidates) != 1:
                raise ValueError(
                    f"reconciled quote must occur in one Topic: {content!r}"
                )
            path = candidates[0]
            text = path.read_text(encoding="utf-8")
            citations = "".join(f"[^{memory_id}]" for memory_id, _when, _refs in rows)
            text = text.replace(content, content + citations, 1)
            path.write_text(text, encoding="utf-8")
            relative_topic = Path("topics") / path.relative_to(topics)
            for memory_id, when, refs in rows:
                links = " · ".join(
                    self._source_link(relative_topic, ref) for ref in refs
                )
                definitions.setdefault(path, []).append(
                    f"[^{memory_id}]: {when or 'undated'} · Sources: {links}"
                )
        for path, rows in definitions.items():
            text = path.read_text(encoding="utf-8").rstrip()
            path.write_text(f"{text}\n\n" + "\n".join(rows) + "\n", encoding="utf-8")

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
            if path.is_file()
        ]
        return "\n".join(paths) or "(empty workspace)"

    def archive_sessions(self, sessions: list[dict[str, Any]]) -> None:
        grouped: dict[str, list[tuple[str, str, str]]] = {}
        for session in sessions:
            date = str(session.get("observation_date", ""))
            for (speaker, content), ref in zip(
                session.get("turns", []), session.get("refs", [])
            ):
                match = re.fullmatch(r"D(\d+):(\d+)", str(ref))
                if not match:
                    raise ValueError("refs must be complete source references like D18:11")
                grouped.setdefault(f"D{match.group(1)}", []).append(
                    (str(ref), date, f"{speaker}: {content}")
                )
        source_dir = self.memory_dir / "sources"
        source_dir.mkdir(exist_ok=True)
        for conversation, rows in grouped.items():
            path = source_dir / f"{conversation}.md"
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            known = set(re.findall(r'<a id="(d\d+-\d+)"></a>', existing))
            additions = []
            for ref, date, content in rows:
                anchor = ref.lower().replace(":", "-")
                if anchor not in known:
                    additions.append(
                        f'<a id="{anchor}"></a>\n[{date}] {content} [{ref}]'
                    )
                    known.add(anchor)
            if additions:
                body = existing.rstrip()
                path.write_text(
                    (body + "\n\n" if body else f"# {conversation}\n\n")
                    + "\n\n".join(additions) + "\n",
                    encoding="utf-8",
                )
        self._refresh_stage()

    @staticmethod
    def _provider_source_location(ref: str) -> tuple[Path, str] | None:
        parts = ref.split("/", 2)
        if len(parts) != 3 or any(not part for part in parts):
            return None
        provider, thread_id, _message_id = parts
        path = Path("sources") / quote(provider, safe="-_.") / (
            quote(thread_id, safe="-_.") + ".md"
        )
        anchor = "source-" + hashlib.sha256(ref.encode()).hexdigest()[:16]
        return path, anchor

    def archive_source_records(self, records: list[SourceRecord]) -> list[str]:
        grouped: dict[Path, list[SourceRecord]] = {}
        for record in sorted(records, key=lambda value: (
            value.provider, value.thread_id, value.ordinal
        )):
            location = self._provider_source_location(record.source_id)
            if location is None:
                raise ValueError("provider source IDs require provider/thread/message")
            grouped.setdefault(location[0], []).append(record)
        refs = []
        for relative, rows in grouped.items():
            path = self.memory_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            known = set(re.findall(r"<!-- source-id:([^>]+) -->", text))
            additions = []
            for record in rows:
                refs.append(record.source_id)
                if record.source_id in known:
                    continue
                _target, anchor = self._provider_source_location(record.source_id)
                additions.extend([
                    f'<a id="{anchor}"></a>',
                    f"<!-- source-id:{record.source_id} -->",
                    f"[{record.timestamp or ''}] {record.role}: {record.content}",
                    "",
                ])
                known.add(record.source_id)
            if additions:
                body = text.rstrip()
                path.write_text(
                    (body + "\n\n" if body else f"# {rows[0].thread_id}\n\n")
                    + "\n".join(additions).rstrip() + "\n",
                    encoding="utf-8",
                )
        self._refresh_stage()
        return refs

    @staticmethod
    def _validate_event(event: dict[str, Any]) -> dict[str, Any]:
        when = str(event.get("when", "")).strip()
        content = str(event.get("content", "")).strip()
        refs = event.get("refs")
        topic_path = str(event.get("topic_path", "")).strip()
        headings = event.get("headings")
        if not when or not content or not isinstance(refs, list) or not refs:
            raise ValueError("each event requires when, content, and refs")
        if when != "undated" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", when):
            raise ValueError("when must use YYYY-MM-DD or undated")
        expanded_refs = []
        for raw_ref in refs:
            ref = str(raw_ref).strip()
            single = re.fullmatch(r"D(\d+):(\d+)", ref)
            span = re.fullmatch(r"D(\d+):(\d+)-(?:D\1:)?(\d+)", ref)
            if single:
                expanded_refs.append(ref)
            elif span:
                conversation, first, last = map(int, span.groups())
                step = 1 if first <= last else -1
                expanded_refs.extend(
                    f"D{conversation}:{turn}"
                    for turn in range(first, last + step, step)
                )
            else:
                if MemoryWorkspace._provider_source_location(ref) is not None:
                    expanded_refs.append(ref)
                else:
                    raise ValueError("refs must be complete source references like D18:11")
        relative = Path(topic_path)
        while relative.parts and relative.parts[0] == "topics":
            relative = Path(*relative.parts[1:])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".md":
            raise ValueError("topic_path must be a relative Markdown path")
        if not isinstance(headings, list) or not 1 <= len(headings) <= 6:
            raise ValueError("headings must contain one to six levels")
        clean_headings = [str(value).strip() for value in headings]
        if any(not value or "\n" in value for value in clean_headings):
            raise ValueError("headings must be non-empty single lines")
        refs = list(dict.fromkeys(expanded_refs))
        payload = json.dumps(
            [when, content, refs], ensure_ascii=False, separators=(",", ":")
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return {
            "event_id": "ev_" + digest,
            "memory_id": "mem_" + digest,
            "when": when,
            "content": content,
            "refs": refs,
            "topic_path": relative.as_posix(),
            "headings": clean_headings,
        }

    @staticmethod
    def _append_event(path: Path, event: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if f"<!-- memory-event:{event['event_id']} -->" in text:
            return
        lines = text.rstrip().splitlines() if text.strip() else []
        wanted = event["headings"]
        wanted_keys = [
            unicodedata.normalize("NFKC", " ".join(heading.split())).casefold()
            for heading in wanted
        ]
        stack: list[str] = []
        best_depth = 0
        best_index = None
        best_level = 0
        for index, line in enumerate(lines):
            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if not match:
                continue
            level = len(match.group(1))
            stack = stack[:level - 1] + [match.group(2)]
            keys = [
                unicodedata.normalize(
                    "NFKC", " ".join(heading.split())
                ).casefold()
                for heading in stack
            ]
            if keys == wanted_keys[:len(keys)] and len(keys) > best_depth:
                best_depth = len(keys)
                best_index = index
                best_level = level
            if keys == wanted_keys:
                break

        insertion = len(lines)
        if best_index is not None:
            for later in range(best_index + 1, len(lines)):
                next_heading = re.match(r"^(#{1,6})\s+", lines[later])
                if next_heading and len(next_heading.group(1)) <= best_level:
                    insertion = later
                    break

        missing = wanted[best_depth:]
        if missing:
            headings = [
                f"{'#' * level} {heading}"
                for level, heading in enumerate(missing, best_depth + 1)
            ]
            if insertion and lines[insertion - 1].strip():
                headings.insert(0, "")
            lines[insertion:insertion] = headings
            insertion += len(headings)
        memory_id = event.get("memory_id") or "mem_" + event["event_id"].removeprefix("ev_")
        block = [
            "",
            f"<a id=\"event-{event['event_id']}\"></a>",
            f"<!-- memory-event:{event['event_id']} -->",
            f"<!-- memory-links:{event['event_id']} -->",
            f"{event['content']}[^{memory_id}]",
            "",
            f"[^{memory_id}]: {event['when']} · Sources: "
            + " · ".join(
                MemoryWorkspace._source_link(
                    Path("topics") / event["topic_path"], ref
                )
                for ref in event["refs"]
            ),
        ]
        lines[insertion:insertion] = block
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def save_memory(self, events: list[dict[str, Any]]) -> str:
        rows = []
        for event in events:
            row = self._validate_event(event)
            rows.append(row)
            self.pending[row["event_id"]] = row
            self._append_event(
                self.stage_dir / "topics" / row["topic_path"], row
            )
        self._synchronize()
        noun = "event" if len(rows) == 1 else "events"
        return f"saved {len(rows)} {noun}"

    @staticmethod
    def _timeline_catalog(memory_dir: Path) -> dict[str, dict[str, Any]]:
        catalog: dict[str, dict[str, Any]] = {}
        timeline = memory_dir / "timeline"
        if not timeline.exists():
            return catalog
        prefix = "<!-- memory-record:"
        for path in timeline.rglob("*.md"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith(prefix) and line.endswith(" -->"):
                    row = json.loads(line[len(prefix):-4])
                    catalog[row["event_id"]] = row
        return catalog

    @staticmethod
    def _topic_locations(topics: Path) -> dict[str, dict[str, Any]]:
        locations: dict[str, dict[str, Any]] = {}
        for path in topics.rglob("*.md"):
            stack: list[str] = []
            lines = path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
                if heading:
                    level = len(heading.group(1))
                    stack = stack[:level - 1] + [heading.group(2)]
                    continue
                marker = re.fullmatch(r"<!-- memory-event:(ev_[0-9a-f]{16}) -->", line)
                if marker:
                    event_id = marker.group(1)
                    if event_id in locations:
                        raise ValueError(f"duplicate event_id in topics: {event_id}")
                    locations[event_id] = {
                        "topic_path": "topics/" + path.relative_to(topics).as_posix(),
                        "headings": list(stack),
                    }
                    for candidate in lines[index + 1:]:
                        event_line = re.match(
                            r"^\[(\d{4}-\d{2}-\d{2})\]\s+(.+?)\s*$",
                            candidate,
                        )
                        if event_line:
                            locations[event_id].update({
                                "when": event_line.group(1),
                                "content": event_line.group(2),
                            })
                            break
                        footnote_line = re.match(
                            r"^(.+?)\[\^(mem_[A-Za-z0-9_-]+)\]\s*$",
                            candidate,
                        )
                        if footnote_line:
                            definition = re.compile(
                                rf"^\[\^{re.escape(footnote_line.group(2))}\]:\s*"
                                r"(\d{4}-\d{2}-\d{2}|undated)\s+·\s+Sources:"
                            )
                            when = next(
                                (
                                    match.group(1)
                                    for value in lines
                                    if (match := definition.match(value))
                                ),
                                None,
                            )
                            if when is None:
                                raise ValueError(
                                    f"missing footnote definition: {footnote_line.group(2)}"
                                )
                            locations[event_id].update({
                                "when": when,
                                "content": footnote_line.group(1),
                            })
                            break
                        if re.match(r"^(#{1,6})\s+", candidate) or re.fullmatch(
                            r"<!-- memory-event:(ev_[0-9a-f]{16}) -->", candidate
                        ):
                            break
                    if "content" not in locations[event_id]:
                        raise ValueError(f"missing event content in topics: {event_id}")
        return locations

    @staticmethod
    def _source_link(topic_path: Path, ref: str) -> str:
        legacy = re.fullmatch(r"D(\d+):(\d+)", ref)
        if legacy:
            conversation, turn = legacy.groups()
            target = Path("sources") / f"D{conversation}.md"
            anchor = f"d{conversation}-{turn}"
        else:
            location = MemoryWorkspace._provider_source_location(ref)
            if location is None:
                raise ValueError(f"invalid source reference: {ref}")
            target, anchor = location
        relative = os.path.relpath(target, topic_path.parent).replace(os.sep, "/")
        return f"[{ref}]({relative}#{anchor})"

    def _rewrite_topic_links(
        self, catalog: dict[str, dict[str, Any]], locations: dict[str, dict[str, Any]]
    ) -> None:
        topics = self.stage_dir / "topics"
        for path in topics.rglob("*.md"):
            relative_topic = Path("topics") / path.relative_to(topics)
            lines = path.read_text(encoding="utf-8").splitlines()
            rendered = []
            for line in lines:
                definition = re.fullmatch(
                    r"\[\^(mem_([0-9a-f]{16}))\]:\s*.*", line
                )
                if definition:
                    event_id = "ev_" + definition.group(2)
                    event = catalog[event_id]
                    rendered.append(
                        f"[^{definition.group(1)}]: {event['when']} · Sources: "
                        + " · ".join(
                            self._source_link(relative_topic, ref)
                            for ref in event["refs"]
                        )
                    )
                    continue
                match = re.fullmatch(r"<!-- memory-links:(ev_[0-9a-f]{16}) -->.*", line)
                if not match:
                    rendered.append(line)
                    continue
                event = catalog[match.group(1)]
                timeline = Path("timeline") / Path(*event["when"].split("-"))
                timeline = timeline.with_suffix(".md")
                relative_timeline = os.path.relpath(
                    timeline, relative_topic.parent
                ).replace(os.sep, "/")
                links = [
                    f"[Timeline]({relative_timeline}#event-{event['event_id']})"
                ] + [self._source_link(relative_topic, ref) for ref in event["refs"]]
                rendered.append(
                    f"<!-- memory-links:{event['event_id']} --> " + " ".join(links)
                )
            path.write_text("\n".join(rendered) + "\n", encoding="utf-8")

    def _write_timeline(
        self, catalog: dict[str, dict[str, Any]], locations: dict[str, dict[str, Any]]
    ) -> Path:
        timeline = Path(tempfile.mkdtemp(prefix="v11-timeline-"))
        by_date: dict[str, list[dict[str, Any]]] = {}
        for event in catalog.values():
            by_date.setdefault(event["when"], []).append(event)
        for when, events in by_date.items():
            path = timeline / Path(*when.split("-"))
            path = path.with_suffix(".md")
            path.parent.mkdir(parents=True, exist_ok=True)
            blocks = [f"# {when}"]
            for event in sorted(events, key=lambda row: row["event_id"]):
                location = locations[event["event_id"]]
                topic_target = Path(location["topic_path"])
                topic_link = os.path.relpath(
                    topic_target, Path("timeline") / path.relative_to(timeline).parent
                ).replace(os.sep, "/")
                source_links = []
                for ref in event["refs"]:
                    source_links.append(
                        self._source_link(
                            Path("timeline") / path.relative_to(timeline), ref
                        )
                    )
                record = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                blocks.extend([
                    "",
                    f'<a id="event-{event["event_id"]}"></a>',
                    f"<!-- memory-record:{record} -->",
                    f"[{when}] {event['content']} "
                    f"[Topic]({topic_link}#event-{event['event_id']}) "
                    + " ".join(source_links),
                ])
            path.write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")
        return timeline

    def _synchronize(self) -> str:
        core = self.stage_dir / "core.md"
        if core.exists():
            limit = int(os.environ.get("NATIVEMEM_V11_CORE_MAX_TOKENS", "2000"))
            if limit < 0:
                raise ValueError("NATIVEMEM_V11_CORE_MAX_TOKENS must be non-negative")
            token_count = len(
                tiktoken.get_encoding("o200k_base").encode(
                    core.read_text(encoding="utf-8")
                )
            )
            if token_count > limit:
                raise ValueError(
                    f"Core Memory exceeds {limit} tokens: {token_count}"
                )

        parsed_units = parse_topic_tree(self.stage_dir / "topics")
        authoritative_topics = bool(parsed_units) or any(
            re.search(r"\[\^mem_[A-Za-z0-9_-]+\]", path.read_text(encoding="utf-8"))
            for path in (self.memory_dir / "topics").rglob("*.md")
        )
        if authoritative_topics:
            catalog = {
                "ev_" + unit.memory_id.removeprefix("mem_"): {
                    "event_id": "ev_" + unit.memory_id.removeprefix("mem_"),
                    "memory_id": unit.memory_id,
                    "when": unit.when or "undated",
                    "content": unit.content,
                    "refs": list(unit.source_refs),
                }
                for unit in parsed_units
            }
        else:
            # One-way compatibility for workspaces created before footnote IDs.
            catalog = self._timeline_catalog(self.memory_dir)
            catalog.update(self.pending)
        if authoritative_topics:
            locations = {
                "ev_" + unit.memory_id.removeprefix("mem_"): {
                    "topic_path": "topics/" + unit.topic_path,
                    "headings": list(unit.headings),
                    "when": unit.when or "undated",
                    "content": unit.content,
                }
                for unit in parsed_units
            }
        else:
            locations = self._topic_locations(self.stage_dir / "topics")
        if set(locations) != set(catalog):
            missing = sorted(set(catalog) - set(locations))
            unknown = sorted(set(locations) - set(catalog))
            raise ValueError(f"topic event mismatch; missing={missing}, unknown={unknown}")
        for event_id, location in locations.items():
            catalog[event_id]["when"] = location["when"]
            catalog[event_id]["content"] = location["content"]
        for event in catalog.values():
            for ref in event["refs"]:
                legacy = re.fullmatch(r"D(\d+):(\d+)", ref)
                if legacy:
                    conversation, turn = legacy.groups()
                    source = self.memory_dir / "sources" / f"D{conversation}.md"
                    anchor = f'<a id="d{conversation}-{turn}"></a>'
                else:
                    location = self._provider_source_location(ref)
                    if location is None:
                        raise ValueError(f"invalid source reference: {ref}")
                    relative, source_anchor = location
                    source = self.memory_dir / relative
                    anchor = f'<a id="{source_anchor}"></a>'
                if not source.exists() or anchor not in source.read_text(encoding="utf-8"):
                    raise ValueError(f"missing source reference: {ref}")
        self._rewrite_topic_links(catalog, locations)
        timeline = self._write_timeline(catalog, locations)

        recent_path = self.memory_dir / "recent_events.jsonl"
        recent = []
        if recent_path.exists():
            recent = [json.loads(line) for line in recent_path.read_text().splitlines() if line]
        recent = [row for row in recent if row["event_id"] in catalog]
        known_recent = {row["event_id"] for row in recent}
        recent.extend(row for key, row in catalog.items() if key not in known_recent)
        limit = int(os.environ.get("NATIVEMEM_V11_RECENT_LIMIT", "50"))
        if limit < 0:
            raise ValueError("NATIVEMEM_V11_RECENT_LIMIT must be non-negative")
        recent = recent[-max(0, limit):] if limit else []
        for row in recent:
            row.update(catalog[row["event_id"]])
            row.update(locations[row["event_id"]])
            row["timeline_path"] = "timeline/" + row["when"].replace("-", "/") + ".md"

        new_topics = self.stage_dir / "topics"
        old_topics = self.memory_dir / "topics"
        old_timeline = self.memory_dir / "timeline"
        old_core = self.memory_dir / "core.md"
        new_core = self.stage_dir / "core.md"
        backup = self.memory_dir / ".v11-backup"
        if backup.exists():
            shutil.rmtree(backup)
        backup.mkdir()
        moved_topics = moved_timeline = moved_core = False
        installed_topics = installed_timeline = installed_core = False
        try:
            if old_topics.exists():
                os.replace(old_topics, backup / "topics")
                moved_topics = True
            if old_timeline.exists():
                os.replace(old_timeline, backup / "timeline")
                moved_timeline = True
            if old_core.exists():
                os.replace(old_core, backup / "core.md")
                moved_core = True
            os.replace(new_topics, old_topics)
            installed_topics = True
            os.replace(timeline, old_timeline)
            installed_timeline = True
            if new_core.exists():
                os.replace(new_core, old_core)
                installed_core = True
            temp_recent = self.memory_dir / ".recent_events.tmp"
            temp_recent.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in recent),
                encoding="utf-8",
            )
            os.replace(temp_recent, recent_path)
        except Exception:
            if installed_topics and old_topics.exists():
                shutil.rmtree(old_topics)
            if moved_topics and (backup / "topics").exists():
                os.replace(backup / "topics", old_topics)
            if installed_timeline and old_timeline.exists():
                shutil.rmtree(old_timeline)
            if moved_timeline and (backup / "timeline").exists():
                os.replace(backup / "timeline", old_timeline)
            if installed_core and old_core.exists():
                old_core.unlink()
            if moved_core and (backup / "core.md").exists():
                os.replace(backup / "core.md", old_core)
            raise
        finally:
            shutil.rmtree(backup, ignore_errors=True)
        self.pending.clear()
        self.committed = True
        self._refresh_stage()
        return f"committed {len(catalog)} events"


def render_conversation(turns: list[tuple[str, str]], refs: list[str]) -> str:
    return "\n".join(
        f"[{ref}] {speaker}: {text}"
        for (speaker, text), ref in zip(turns, refs)
    )


def _compact_tool_history(messages: list[Any]) -> None:
    """Retain full output for only the latest tool-call round."""
    def role(message: Any) -> object:
        return message.get("role") if isinstance(message, dict) else getattr(message, "role", None)

    last_assistant = max(
        (i for i, message in enumerate(messages) if role(message) == "assistant"),
        default=-1,
    )
    for message in messages[:last_assistant]:
        if role(message) == "tool" and len(message.get("content", "")) > 1000:
            message["content"] = "[previous tool output omitted]"


def _chat_completion_with_retry(create: Any, **kwargs: Any) -> Any:
    """Retry transient provider failures without discarding agent state."""
    attempt = 0
    while True:
        try:
            return create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            temporary_quota = status == 403 and (
                "insufficient_user_quota" in str(exc)
                or "用户额度不足" in str(exc)
            )
            transient = (
                isinstance(exc, (ConnectionError, TimeoutError))
                or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
                or status == 429
                or temporary_quota
                or isinstance(status, int) and 500 <= status < 600
            )
            if not transient:
                raise
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", {}) or {}
            try:
                delay = float(headers.get("retry-after", 0))
            except (TypeError, ValueError):
                delay = 0
            retry_delay = delay or (30 if temporary_quota else min(2 ** attempt, 120))
            if os.environ.get("NATIVEMEM_RETRY_LOG") == "1":
                print(
                    f"V11 provider retry attempt={attempt + 1} status={status} "
                    f"delay={retry_delay} error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(retry_delay)
            attempt += 1


def _provider_options() -> dict[str, Any]:
    options: dict[str, Any] = {}
    reasoning_effort = os.environ.get("NATIVEMEM_REASONING_EFFORT")
    if reasoning_effort:
        options["reasoning_effort"] = reasoning_effort
    thinking = os.environ.get("NATIVEMEM_THINKING")
    if thinking:
        options["extra_body"] = {"thinking": {"type": thinking}}
    return options


def _make_reconciler(client: Any, model: str, usage_logger: Any | None):
    def reconcile(edited_text, old_units, candidate_sources):
        payload = {
            "edited_topic_text": edited_text,
            "old_memories": [
                {
                    "memory_id": unit.memory_id,
                    "content": unit.content,
                    "when": unit.when,
                    "source_refs": list(unit.source_refs),
                }
                for unit in old_units
            ],
            "candidate_sources": sorted(candidate_sources),
        }
        response = _chat_completion_with_retry(
            client.chat.completions.create,
            model=model,
            messages=[{
                "role": "system",
                "content": (
                    "Reconcile edited Topic Markdown with existing memory IDs. "
                    "Every returned text value must be an exact quote from edited_topic_text. "
                    "Use only candidate_sources. Output JSON with matches, creates, "
                    "deleted_ids, and organizational_quotes. creates contains objects "
                    "with content, when (YYYY-MM-DD or null), and source_refs."
                ),
            }, {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            }],
            temperature=0.0,
            **_provider_options(),
        )
        if usage_logger is not None:
            usage_logger(response)
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ReconciliationError("Reconciler did not return JSON")
        value = json.loads(match.group(0))
        creates = tuple(
            (
                str(row["content"]),
                None if row.get("when") is None else str(row["when"]),
                tuple(str(ref) for ref in row.get("source_refs", [])),
            )
            for row in value.get("creates", [])
        )
        return ReconciliationResult(
            matches={str(key): str(quote) for key, quote in value.get("matches", {}).items()},
            creates=creates,
            deleted_ids=tuple(str(value) for value in value.get("deleted_ids", [])),
            organizational_quotes=tuple(
                str(value) for value in value.get("organizational_quotes", [])
            ),
        )
    return reconcile


def _run_agent(
    memory_dir: str | Path,
    *,
    client: Any,
    model: str,
    task: str,
    source_sessions: list[dict[str, Any]] | None = None,
    usage_logger: Any | None = None,
    final_output: list[str] | None = None,
    max_rounds: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if max_rounds is None:
        max_rounds = int(os.environ.get("NATIVEMEM_V11_AGENT_MAX_ROUNDS", "12"))
    if max_rounds < 1:
        raise ValueError("V11 agent max rounds must be positive")
    available_tools = TOOLS if tools is None else tools
    allowed_tools = {tool["function"]["name"] for tool in available_tools}
    workspace = MemoryWorkspace(
        memory_dir,
        reconciler=_make_reconciler(client, model, usage_logger),
    )
    if source_sessions:
        workspace.archive_sessions(source_sessions)
        workspace._refresh_stage()
    task = f"{task}\n\nCurrent workspace structure:\n{workspace.structure()}"
    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    audit: list[dict[str, Any]] = []
    completed = False
    for round_no in range(max_rounds):
        _compact_tool_history(messages)
        request = {
            "model": model,
            "messages": messages,
            "tools": available_tools,
            "temperature": 0.1,
        }
        request.update(_provider_options())
        response = _chat_completion_with_retry(
            client.chat.completions.create,
            **request,
        )
        if usage_logger is not None:
            usage_logger(response)
        message = response.choices[0].message
        messages.append(message)
        calls = message.tool_calls or []
        if not calls:
            if final_output is not None:
                final_output.append(message.content or "")
            completed = True
            break
        for call in calls:
            try:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise ValueError("invalid JSON arguments") from exc
                if call.function.name not in allowed_tools:
                    raise ValueError(f"tool is not allowed in this phase: {call.function.name}")
                if call.function.name == "shell":
                    result = workspace.shell(args["command"])
                    output = result.stdout + result.stderr
                    record = {"round": round_no, "tool": "shell",
                              "command": args["command"],
                              "returncode": result.returncode, "output": output}
                elif call.function.name == "save_memory":
                    output = workspace.save_memory(args["events"])
                    record = {"round": round_no, "tool": "save_memory",
                              "count": len(args["events"]), "output": output,
                              "topic_paths": sorted({
                                  "topics/" + MemoryWorkspace._validate_event(event)[
                                      "topic_path"
                                  ]
                                  for event in args["events"]
                              })}
                else:
                    raise ValueError(f"unknown tool: {call.function.name}")
                record["status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                output = f"Tool error: {exc}"
                record = {"round": round_no, "tool": call.function.name,
                          "status": "error", "output": output}
            audit.append(record)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": output[-100_000:]})
    if not completed:
        audit.append({
            "tool": "agent",
            "status": "stopped",
            "reason": "round_limit",
            "rounds": max_rounds,
        })
    shutil.rmtree(workspace.stage_dir, ignore_errors=True)
    return audit


def _json_response(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    usage_logger: Any | None,
) -> dict[str, Any]:
    response = _chat_completion_with_retry(
        client.chat.completions.create,
        model=model,
        messages=messages,
        max_tokens=500,
        temperature=0.0,
        **_provider_options(),
    )
    if usage_logger is not None:
        usage_logger(response)
    content = response.choices[0].message.content or ""
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("verification model did not return JSON")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("verification response must be a JSON object")
    return value


def _verification_retrieve(
    memory_dir: str | Path,
    *,
    client: Any,
    model: str,
    question: str,
    usage_logger: Any | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v11-verify-read-") as temporary:
        copied = Path(temporary) / "memory"
        shutil.copytree(Path(memory_dir), copied)
        final: list[str] = []
        audit = _run_agent(
            copied,
            client=client,
            model=model,
            task=VERIFICATION_RETRIEVAL_TASK.format(question=question),
            usage_logger=usage_logger,
            final_output=final,
            tools=TOOLS[:1],
        )
    text = final[-1] if final else ""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("verification retrieval did not return an answer block")
    return {"question": question, "answer": match.group(1).strip(), "trace": audit}


def _verification_answer_supported(
    *,
    client: Any,
    model: str,
    question: str,
    expected_answer: str,
    retrieved_answer: str,
    usage_logger: Any | None,
) -> bool:
    result = _json_response(
        client=client,
        model=model,
        usage_logger=usage_logger,
        messages=[{
            "role": "user",
            "content": (
                "Decide whether the retrieved answer correctly answers the question "
                "according to the expected source-grounded answer. Output only "
                '{"supported":true} or {"supported":false}.\n\n'
                f"Question: {question}\n"
                f"Expected answer: {expected_answer}\n"
                f"Retrieved answer: {retrieved_answer}"
            ),
        }],
    )
    if not isinstance(result.get("supported"), bool):
        raise ValueError("verification answer check must return supported boolean")
    return result["supported"]


def verify_session(
    memory_dir: str | Path,
    *,
    client: Any,
    model: str,
    observation_date: str,
    turns: list[tuple[str, str]],
    refs: list[str],
    usage_logger: Any | None = None,
) -> dict[str, Any]:
    probe = _json_response(
        client=client,
        model=model,
        usage_logger=usage_logger,
        messages=[{
            "role": "user",
            "content": VERIFICATION_PROBE_TASK.format(
                observation_date=observation_date,
                conversation=render_conversation(turns, refs),
            ),
        }],
    )
    question = str(probe.get("question", "")).strip()
    expected_answer = str(probe.get("expected_answer", "")).strip()
    probe_refs = [str(ref) for ref in probe.get("refs", [])]
    if (
        not question
        or not expected_answer
        or not probe_refs
        or any(ref not in refs for ref in probe_refs)
    ):
        raise ValueError("verification probe is incomplete or cites another session")
    probe = {
        "question": question,
        "expected_answer": expected_answer,
        "refs": probe_refs,
    }

    initial = _verification_retrieve(
        memory_dir,
        client=client,
        model=model,
        question=question,
        usage_logger=usage_logger,
    )
    initial["supported"] = _verification_answer_supported(
        client=client,
        model=model,
        question=question,
        expected_answer=expected_answer,
        retrieved_answer=initial["answer"],
        usage_logger=usage_logger,
    )
    if initial["supported"]:
        return {
            "probe": probe,
            "initial": initial,
            "repaired": False,
            "repair_trace": [],
            "post_repair": None,
        }

    repair_trace = _run_agent(
        memory_dir,
        client=client,
        model=model,
        task=VERIFICATION_REPAIR_TASK.format(
            question=question,
            expected_answer=expected_answer,
            refs=", ".join(probe_refs),
            retrieved_answer=initial["answer"],
            trace=json.dumps(initial["trace"], ensure_ascii=False, indent=2),
        ),
        usage_logger=usage_logger,
    )
    post_repair = _verification_retrieve(
        memory_dir,
        client=client,
        model=model,
        question=question,
        usage_logger=usage_logger,
    )
    post_repair["supported"] = _verification_answer_supported(
        client=client,
        model=model,
        question=question,
        expected_answer=expected_answer,
        retrieved_answer=post_repair["answer"],
        usage_logger=usage_logger,
    )
    return {
        "probe": probe,
        "initial": initial,
        "repaired": True,
        "repair_trace": repair_trace,
        "post_repair": post_repair,
    }


def write_session(
    memory_dir: str | Path,
    *,
    client: Any,
    model: str,
    observation_date: str,
    turns: list[tuple[str, str]],
    refs: list[str],
    usage_logger: Any | None = None,
) -> list[dict[str, Any]]:
    task = WRITER_TASK.format(
        observation_date=observation_date,
        conversation=render_conversation(turns, refs),
    )
    return _run_agent(memory_dir, client=client, model=model, task=task,
                      source_sessions=[{
                          "observation_date": observation_date,
                          "turns": turns,
                          "refs": refs,
                      }],
                      usage_logger=usage_logger)


def write_sessions(
    memory_dir: str | Path,
    *,
    client: Any,
    model: str,
    sessions: list[dict[str, Any]],
    usage_logger: Any | None = None,
) -> list[dict[str, Any]]:
    rendered = []
    for number, session in enumerate(sessions, start=1):
        rendered.append(
            f"## Session {number}\nObservation date: "
            f"{session['observation_date']}\n\n"
            f"{render_conversation(session['turns'], session['refs'])}"
        )
    task = WRITER_BATCH_TASK.format(sessions="\n\n".join(rendered))
    return _run_agent(
        memory_dir,
        client=client,
        model=model,
        task=task,
        source_sessions=sessions,
        usage_logger=usage_logger,
    )


def manage_memory(
    memory_dir: str | Path,
    *,
    client: Any,
    model: str,
    usage_logger: Any | None = None,
) -> list[dict[str, Any]]:
    max_rounds = int(os.environ.get("NATIVEMEM_V11_MANAGER_MAX_ROUNDS", "8"))
    return _run_agent(memory_dir, client=client, model=model, task=MANAGER_TASK,
                      usage_logger=usage_logger, max_rounds=max_rounds,
                      tools=TOOLS[:1])


def organize_topics(
    memory_dir: str | Path,
    *,
    client: Any,
    model: str,
    touched: set[str] | None = None,
    final: bool = False,
    usage_logger: Any | None = None,
) -> list[dict[str, Any]]:
    """Compatibility entry point for existing manager callers."""
    if final or touched is None:
        return manage_memory(
            memory_dir, client=client, model=model, usage_logger=usage_logger
        )
    normalized = sorted({
        Path(path).as_posix()
        for path in touched
        if Path(path).as_posix().startswith("topics/")
    })
    if not normalized:
        return []
    return _run_agent(
        memory_dir,
        client=client,
        model=model,
        task=LOCAL_MANAGER_TASK.format(topic_paths="\n".join(normalized)),
        usage_logger=usage_logger,
        max_rounds=int(os.environ.get("NATIVEMEM_V11_MANAGER_MAX_ROUNDS", "8")),
        tools=TOOLS[:1],
    )
