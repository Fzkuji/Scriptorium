#!/usr/bin/env python3
"""Lossless, model-directed organizer pilot for one copied memory library."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from openai import OpenAI


REF_RE = re.compile(r"D\d+:\d+")


@dataclass(frozen=True)
class TopicSnapshot:
    entries: Counter[str]
    references: Counter[str]


def _entry_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def snapshot_topics(root: Path) -> TopicSnapshot:
    entries: Counter[str] = Counter()
    refs: Counter[str] = Counter()
    for path in sorted((root / "topics").rglob("*.md")):
        for line in _entry_lines(path):
            entries[line] += 1
            refs.update(REF_RE.findall(line))
    return TopicSnapshot(entries, refs)


def validate_topics(before: TopicSnapshot, root: Path) -> dict[str, Any]:
    after = snapshot_topics(root)
    missing = before.entries - after.entries
    added = after.entries - before.entries
    missing_refs = before.references - after.references
    empty = [str(p.relative_to(root / "topics")) for p in (root / "topics").rglob("*.md")
             if not _entry_lines(p)]
    return {
        "valid": not missing and not added and not missing_refs and not empty,
        "entries": after.entries,
        "missing_entries": dict(missing),
        "added_entries": dict(added),
        "missing_references": dict(missing_refs),
        "empty_files": sorted(empty),
    }


class OrganizerWorkspace:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.topics = (self.root / "topics").resolve()

    def _path(self, relative: str) -> Path:
        path = (self.topics / relative).resolve()
        try:
            path.relative_to(self.topics)
        except ValueError as exc:
            raise ValueError("path must remain inside topics") from exc
        if path.suffix != ".md":
            raise ValueError("path must name a Markdown file inside topics")
        return path

    @staticmethod
    def _write(path: Path, lines: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines).rstrip() + "\n")
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def list_tree(self) -> str:
        rows = []
        for path in sorted(self.topics.rglob("*.md")):
            rows.append(f"{path.relative_to(self.topics)} [{len(_entry_lines(path))} entries]")
        return "\n".join(rows)

    def read_file(self, path: str) -> str:
        target = self._path(path)
        return target.read_text(encoding="utf-8")[:50000]

    def search_memory(self, query: str) -> str:
        hits = []
        needle = query.casefold()
        for path in sorted(self.topics.rglob("*.md")):
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if needle in line.casefold():
                    hits.append(f"{path.relative_to(self.topics)}:{n}:{line}")
        return "\n".join(hits[:300]) or "(no matches)"

    def merge_files(self, target: str, sources: list[str]) -> str:
        dst = self._path(target)
        lines = dst.read_text(encoding="utf-8").splitlines() if dst.exists() else []
        for source in sources:
            src = self._path(source)
            if src == dst or not src.exists():
                continue
            lines.extend(src.read_text(encoding="utf-8").splitlines())
            src.unlink()
        self._write(dst, lines)
        return f"merged into {target}"

    def rename_file(self, source: str, target: str) -> str:
        src, dst = self._path(source), self._path(target)
        if dst.exists():
            return self.merge_files(target, [source])
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)
        return f"renamed {source} to {target}"

    def move_entries(self, source: str, target: str, entries: list[str]) -> str:
        src, dst = self._path(source), self._path(target)
        source_lines = src.read_text(encoding="utf-8").splitlines()
        requested = Counter(entries)
        present = Counter(line for line in source_lines if line in requested)
        if present != requested:
            raise ValueError("every requested entry must exactly match a source line")
        remaining = list(source_lines)
        for entry in entries:
            remaining.remove(entry)
        destination = dst.read_text(encoding="utf-8").splitlines() if dst.exists() else []
        destination.extend(entries)
        self._write(dst, destination)
        if any(line.strip() and not line.lstrip().startswith("#") for line in remaining):
            self._write(src, remaining)
        else:
            src.unlink()
        return f"moved {len(entries)} entries"

    def delete_empty_file(self, path: str) -> str:
        target = self._path(path)
        if target.exists() and _entry_lines(target):
            raise ValueError("file still contains memory entries")
        target.unlink(missing_ok=True)
        return f"deleted {path}"


TOOLS = [{"type": "function", "function": {"name": name, "description": desc,
          "parameters": params}} for name, desc, params in [
    ("list_tree", "List every topic document and its entry count.", {"type":"object","properties":{}}),
    ("read_file", "Read one topic document.", {"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
    ("search_memory", "Search all topic documents for text.", {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
    ("merge_files", "Move complete source documents into one target.", {"type":"object","properties":{"target":{"type":"string"},"sources":{"type":"array","items":{"type":"string"}}},"required":["target","sources"]}),
    ("rename_file", "Rename one Markdown document. Both paths must end in .md. Use only when its current semantic topic is wrong, never for cosmetic spelling, capitalization, spaces, or underscores.", {"type":"object","properties":{"source":{"type":"string"},"target":{"type":"string"}},"required":["source","target"]}),
    ("delete_empty_file", "Delete a document with no memory entries.", {"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
    ("finish", "Finish when the topic library is coherently organized.", {"type":"object","properties":{"summary":{"type":"string"}}}),
]]


def structure(root: Path) -> dict[str, Any]:
    counts = [len(_entry_lines(p)) for p in (root / "topics").rglob("*.md")]
    return {"files": len(counts), "entries": sum(counts), "single_entry_files": sum(x == 1 for x in counts),
            "min": min(counts, default=0), "median": sorted(counts)[len(counts)//2] if counts else 0,
            "max": max(counts, default=0)}


def organize(root: Path, base_url: str, model: str, trace_path: Path, max_rounds: int) -> None:
    ws = OrganizerWorkspace(root)
    client = OpenAI(base_url=base_url, api_key="local-openrouter-gateway")
    messages: list[Any] = [{"role":"system","content":(
        "You organize a personal memory topics directory. Inspect its tree and content, then organize related information into coherent documents. "
        "The current problem is fragmentation: many documents contain only isolated facts even though related facts belong together. Read relevant documents and combine complete documents about the same subject. "
        "Do not spend time beautifying names, capitalization, spaces, underscores, or directory names. Rename a .md document only when its semantic topic is wrong. "
        "Preserve every memory line exactly, including dates and D source references. Do not rewrite facts. Distinct dates and state changes must remain. Call finish when satisfied.")},
        {"role":"user","content":"Inspect and organize this topics library."}]
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as trace:
        for round_no in range(max_rounds):
            response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS, max_tokens=4000, temperature=0.1)
            msg = response.choices[0].message
            messages.append(msg)
            calls = msg.tool_calls or []
            if not calls:
                break
            for call in calls:
                args = json.loads(call.function.arguments or "{}")
                name = call.function.name
                if name == "finish":
                    trace.write(json.dumps({"round":round_no,"tool":name,"args":args}, ensure_ascii=False)+"\n")
                    return
                try:
                    result = getattr(ws, name)(**args)
                except Exception as exc:  # noqa: BLE001
                    result = f"Error: {type(exc).__name__}: {exc}"
                trace.write(json.dumps({"round":round_no,"tool":name,"args":args,"result":result[:1000]}, ensure_ascii=False)+"\n")
                trace.flush()
                messages.append({"role":"tool","tool_call_id":call.id,"content":result})


def _jsonable(value: Any) -> Any:
    if isinstance(value, Counter): return dict(value)
    if isinstance(value, dict): return {k:_jsonable(v) for k,v in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-memory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:63059/v1")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--max-rounds", type=int, default=80)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    memory = args.output_dir / "memory"
    before_path = args.output_dir / "before_snapshot.json"
    if not args.validate_only:
        if args.output_dir.exists():
            raise SystemExit("output directory already exists")
        shutil.copytree(args.source_memory, memory)
        before = snapshot_topics(memory)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        before_path.write_text(json.dumps(_jsonable(before.__dict__), indent=2), encoding="utf-8")
        (args.output_dir / "before_structure.json").write_text(json.dumps(structure(memory), indent=2), encoding="utf-8")
        organize(memory, args.base_url, args.model, args.output_dir / "organizer_trace.jsonl", args.max_rounds)
    raw = json.loads(before_path.read_text(encoding="utf-8"))
    before = TopicSnapshot(Counter(raw["entries"]), Counter(raw["references"]))
    report = validate_topics(before, memory)
    (args.output_dir / "validation.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    (args.output_dir / "after_structure.json").write_text(json.dumps(structure(memory), indent=2), encoding="utf-8")
    if not report["valid"]:
        raise SystemExit("organized memory failed lossless validation")


if __name__ == "__main__":
    main()
