"""Parse and render authoritative NativeMem Topic Markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


MEMORY_ID = r"mem_[A-Za-z0-9_-]+"
_CITATION_GROUP = re.compile(rf"(?:\[\^(?P<id>{MEMORY_ID})\])+")
_SINGLE_CITATION = re.compile(rf"\[\^(?P<id>{MEMORY_ID})\]")
_DEFINITION = re.compile(
    rf"^\[\^(?P<id>{MEMORY_ID})\]:\s*(?P<when>\d{{4}}-\d{{2}}-\d{{2}}|undated)"
    r"\s*·\s*Sources:\s*(?P<sources>.+?)\s*$"
)
_LINK = re.compile(r"\[([^]]+)\]\(([^)]+)\)")


class TopicFormatError(ValueError):
    """Raised when Topic Markdown cannot be mapped to stable memory units."""


@dataclass(frozen=True)
class MemoryUnit:
    memory_id: str
    content: str
    when: str | None
    source_refs: tuple[str, ...]
    source_links: tuple[str, ...]
    topic_path: str
    headings: tuple[str, ...]
    created_order: int


def _definitions(lines: list[str]) -> dict[str, tuple[str | None, tuple[str, ...], tuple[str, ...]]]:
    definitions = {}
    for line in lines:
        match = _DEFINITION.match(line)
        if not match:
            continue
        memory_id = match.group("id")
        if memory_id in definitions:
            raise TopicFormatError(f"duplicate footnote definition: {memory_id}")
        links = _LINK.findall(match.group("sources"))
        if not links:
            raise TopicFormatError(f"memory source links required: {memory_id}")
        definitions[memory_id] = (
            None if match.group("when") == "undated" else match.group("when"),
            tuple(label for label, _target in links),
            tuple(target for _label, target in links),
        )
    return definitions


def _paragraphs(lines: list[str]):
    headings: list[str] = []
    paragraph: list[str] = []
    paragraph_headings: tuple[str, ...] = ()
    in_fence = False

    def flush():
        nonlocal paragraph
        if paragraph:
            value = "\n".join(paragraph).strip()
            paragraph = []
            if value:
                return value, paragraph_headings
        return None

    for line in lines + [""]:
        if line.lstrip().startswith("```"):
            result = flush()
            if result:
                yield result
            in_fence = not in_fence
            continue
        if in_fence or _DEFINITION.match(line) or re.match(r"\s*(?:<!--|<a\s)", line):
            result = flush()
            if result:
                yield result
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            result = flush()
            if result:
                yield result
            level = len(heading.group(1))
            headings = headings[: level - 1] + [heading.group(2)]
            continue
        if not line.strip():
            result = flush()
            if result:
                yield result
            continue
        if not paragraph:
            paragraph_headings = tuple(headings)
        paragraph.append(line)


def parse_topic_tree(topics: Path, *, strict: bool = True) -> list[MemoryUnit]:
    """Return memory units in their current Markdown occurrence order."""
    topics = Path(topics)
    units: list[MemoryUnit] = []
    seen: set[str] = set()
    for path in sorted(topics.rglob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        definitions = _definitions(lines)
        used_definitions: set[str] = set()
        for paragraph, headings in _paragraphs(lines):
            cursor = 0
            for group in _CITATION_GROUP.finditer(paragraph):
                content = paragraph[cursor:group.start()].strip()
                ids = [m.group("id") for m in _SINGLE_CITATION.finditer(group.group(0))]
                if not content:
                    raise TopicFormatError(f"missing memory content before {ids[0]}")
                for memory_id in ids:
                    if memory_id in seen:
                        raise TopicFormatError(f"duplicate memory_id: {memory_id}")
                    if memory_id not in definitions:
                        raise TopicFormatError(f"undefined footnote: {memory_id}")
                    when, refs, links = definitions[memory_id]
                    units.append(MemoryUnit(
                        memory_id=memory_id,
                        content=content,
                        when=when,
                        source_refs=refs,
                        source_links=links,
                        topic_path=path.relative_to(topics).as_posix(),
                        headings=headings,
                        created_order=len(units),
                    ))
                    seen.add(memory_id)
                    used_definitions.add(memory_id)
                cursor = group.end()
        unused = set(definitions) - used_definitions
        if strict and unused:
            raise TopicFormatError(f"unused footnote definition: {sorted(unused)[0]}")
    return units


def topic_prose(topics: Path) -> str:
    """Return normalized non-structural Topic prose, including unbound text."""
    values = []
    for path in sorted(Path(topics).rglob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for paragraph, _headings in _paragraphs(lines):
            value = _SINGLE_CITATION.sub("", paragraph)
            values.append(" ".join(value.split()))
    return "\n".join(values)


def append_memory_unit(path: Path, unit: MemoryUnit) -> None:
    """Append one unit under its heading path without rewriting existing prose."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if f"[^{unit.memory_id}]" in text:
        return
    lines = text.rstrip().splitlines()
    existing_headings = {
        (len(match.group(1)), match.group(2))
        for line in lines
        if (match := re.match(r"^(#{1,6})\s+(.+?)\s*$", line))
    }
    for level, heading in enumerate(unit.headings, start=1):
        if (level, heading) not in existing_headings:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(f"{'#' * level} {heading}")
            existing_headings.add((level, heading))
    if lines and lines[-1].strip():
        lines.append("")
    lines.append(f"{unit.content}[^{unit.memory_id}]")
    links = " · ".join(
        f"[{label}]({target})"
        for label, target in zip(unit.source_refs, unit.source_links)
    )
    when = unit.when or "undated"
    lines.extend(["", f"[^{unit.memory_id}]: {when} · Sources: {links}"])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
