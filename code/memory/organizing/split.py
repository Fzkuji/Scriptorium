"""Splitting a topic file that outgrew its subject, deterministically.

A conversation about one deep subject — a codebase, a case, a project —
puts everything in one file, and nothing in `tidy` has a rule for that: it
merges duplicates and prunes dead structure, but nothing there shortens a
file that is one subject told at length. Left alone, such a file grows
until its own timeline is unreadable — ordering and counting questions
against it go unanswered because the paragraphs that would answer them are
scattered across one undifferentiated wall of text.

The rule is size, not subject count: past a threshold, a topic file becomes
a directory of its own headings — `topics/projects/tracker.md` becoming
`topics/projects/tracker/{database,deployment,testing}.md`. This is the
same move `reorganize`'s model pass is asked to make for a file that "now
covers two subjects" (see `memory/prompts/organize.py`), just triggered by
size instead of judgement — and a rule that can be computed needs no model
call to apply it: every paragraph already carries the heading path a plain
scan can group by.

    from memory.organizing.split import split
    report = split(workspace)
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from ..markdown.syntax import SINGLE_CITATION, definition_match, paragraphs
from ..workspace import MemoryWorkspace
from .tidy import _is_fact

_Entry = tuple[str, tuple[str, ...]]


def _slug(heading: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-") or "section"
    slug, suffix = base, 2
    while slug in used:
        slug = f"{base}-{suffix}"
        suffix += 1
    used.add(slug)
    return slug


def _sections(entries: list[_Entry]) -> dict[str, list[_Entry]] | None:
    """Group paragraphs by the innermost heading each one sits under.

    Grouping by the innermost heading, not the full path, is what lets a
    file with an outer title over its sections (`# Tracker` above
    `## Database`, `## Deployment`, ...) split on the sections rather than
    the title every paragraph shares — and what lets a file that never uses
    a level-1 heading at all split correctly too: `memory/markdown/syntax.py`
    `paragraphs()`'s heading stack only ever truncates to `level - 1`
    entries, so a file built entirely from level-2 headings (a real shape:
    `beam5-v2/.../budget_tracker.md`) leaves its first heading pinned at
    position 0 of every later paragraph's heading tuple as if every section
    were nested under it, though none of them are. The innermost element is
    right regardless: it is always whichever heading a paragraph is
    actually under.

    `None` means no clean split exists: some paragraph carries no heading at
    all, or two paragraphs share an innermost heading name while sitting
    under different full heading paths — the same name reused for two
    different sections is not one section, and there is no single file to
    put both in.
    """
    if any(not headings for _body, headings in entries):
        return None
    order: list[str] = []
    grouped: dict[str, list[_Entry]] = {}
    paths: dict[str, tuple[str, ...]] = {}
    for body, headings in entries:
        key = headings[-1]
        if key not in grouped:
            order.append(key)
            grouped[key] = []
            paths[key] = headings
        elif paths[key] != headings:
            return None
        grouped[key].append((body, headings))
    return {key: grouped[key] for key in order} if len(grouped) > 1 else None


def _definition_lines(lines: list[str]) -> dict[str, str]:
    return {
        match.group("id"): line
        for line in lines
        if (match := definition_match(line))
    }


def _render_section(
    heading: str, entries: list[_Entry], definitions: dict[str, str]
) -> str:
    """A new file's whole content: the heading, its paragraphs verbatim, and
    the footnote definitions they cite.

    Paragraphs are relocated exactly as written rather than rewritten from
    parsed fields, so a merged paragraph's suffix run of several block IDs
    (`tidy._merge_duplicates`'s doing) moves as one unit without this having
    to know that shape exists. Definitions carry stale relative paths into
    their new file; `MemoryWorkspace.commit_edits` recomputes them for every
    file it installs (`workspace/block_views.py::_rewrite_block_links`), so
    this does not have to.
    """
    rendered = [f"## {heading}", ""]
    cited: list[str] = []
    for body, _headings in entries:
        rendered.append(body)
        rendered.append("")
        for citation_id in SINGLE_CITATION.findall(body):
            if citation_id not in cited:
                cited.append(citation_id)
    for citation_id in cited:
        definition = definitions.get(citation_id)
        if definition is not None:
            rendered.append(definition)
            rendered.append("")
    return "\n".join(rendered).rstrip() + "\n"


def _plan(
    path: Path, threshold: int, min_section: int
) -> dict[str, str] | None:
    """This file's sections rendered as a split would write them, or None.

    None covers every reason not to split: under the size threshold, a
    `<stem>/` directory already sitting where the split would put one, a
    paragraph `_sections` cannot place (see there), only one section once
    grouped, or a section too small to be worth its own file. That last
    check applies to the whole file, not section by section: a file with
    one undersized section among nine large ones still fully splits (a
    topic that happens to be quiet is still a topic), but a file that would
    mostly come out as slivers does not split at all — turning it into
    fifteen one-paragraph files is worse than the file it started as, so
    nothing between "all of it" and "none of it" is offered.
    """
    original = path.read_text(encoding="utf-8")
    if len(original.encode("utf-8")) < threshold:
        return None
    if path.with_suffix("").exists():
        return None
    lines = original.split("\n")
    entries = list(paragraphs(lines))
    if not entries or any(not _is_fact(body) for body, _headings in entries):
        return None
    grouped = _sections(entries)
    if grouped is None:
        return None
    definitions = _definition_lines(lines)
    sections = {
        heading: _render_section(heading, rows, definitions)
        for heading, rows in grouped.items()
    }
    if any(
        len(text.encode("utf-8")) < min_section for text in sections.values()
    ):
        return None
    return sections


def split(
    memory_dir: str | Path, *, threshold: int = 16_384, min_section: int = 500
) -> dict[str, Any]:
    """Turn every topic file past `threshold` bytes into a directory of its
    own headings, transactionally: a split that would leave the workspace
    invalid changes nothing, the same guarantee `tidy` gives its own edits.

    16 KB and 500 B are measured, not guessed. Every topic file across
    `comp-workspaces/` — the small, hand-inspectable fixtures this project
    already uses for other real-workspace calibration — is under 8 KB and
    stays a plain file. A real 33 KB single-file project log
    (`beam5-v2/c1/.../budget_tracker.md`, ten headings, one topic told at
    length) fully splits into ten files, none under about 560 B; a 29 KB
    file with only one heading does not split at all, because there is
    nothing to split it by; a 25 KB file with a near-empty title heading
    over eleven real sections also splits cleanly, the title heading itself
    contributing no section (nothing sits under it before the first real
    one). 500 B sits below that 560 B smallest real section with headroom,
    while still ruling out a file whose sections would mostly be a
    paragraph or two.
    """
    workspace = MemoryWorkspace(memory_dir)
    try:
        root = workspace.stage_dir / "topics"
        if not root.is_dir():
            return {"files": []}
        before_units, before_block_ids, before_topics, before_sources = (
            workspace.baseline(strict=False)
        )
        moved: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*.md")):
            sections = _plan(path, threshold, min_section)
            if sections is None:
                continue
            directory = path.with_suffix("")
            directory.mkdir()
            used_slugs: set[str] = set()
            to: list[str] = []
            for heading, content in sections.items():
                target = directory / f"{_slug(heading, used_slugs)}.md"
                target.write_text(content, encoding="utf-8")
                to.append(target.relative_to(workspace.stage_dir).as_posix())
            path.unlink()
            moved.append({
                "from": path.relative_to(workspace.stage_dir).as_posix(),
                "to": sorted(to),
            })
        if not moved:
            return {"files": []}
        try:
            workspace.commit_edits(
                before_units, before_block_ids, before_topics, before_sources
            )
        except Exception as exc:
            # commit_edits already rolled the stage back; a split that would
            # have produced a contract-invalid workspace changes nothing,
            # the same way a rejected tidy pass does.
            return {"files": [], "error": str(exc)}
        return {"files": moved}
    finally:
        shutil.rmtree(workspace.stage_dir, ignore_errors=True)
