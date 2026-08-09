"""Maintenance that needs no judgement, and so needs no model.

Writing only ever makes files longer. Left alone a workspace becomes one
enormous file per subject with the same fact recorded four times in slightly
different words, which is the shape that makes counting and ordering
questions unanswerable.

A model was doing this pass. Most of what it did was mechanical — the same
sentence twice is the same sentence, a footnote nobody cites is dead, a
heading with nothing under it is noise — and a weak model does mechanical
work badly and expensively. What is left for judgement is which of two
differently-worded facts to keep, and this pass does not touch that: it
merges only what is textually the same.

`merge_candidates` still touches no judgement: it shortlists the pairs close
enough in wording that judging them is worth a model's turn, and leaves the
judging itself to `reorganize`. `merge_blocks` is the fold that judgement
leads to, written once so `reorganize`'s tool and `_merge_duplicates` below
call the same code for "make these two paragraphs one" rather than agreeing
to keep two implementations in step.

    from memory.organizing.tidy import tidy, merge_candidates
    report = tidy(workspace)
    candidates = merge_candidates(workspace)
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from ..markdown import parse_topic_tree
from ..markdown.syntax import BLOCK_SUFFIX, SINGLE_CITATION, definition_match
from ..workspace import MemoryWorkspace

# Prose with its citations, block IDs and punctuation taken off, so "Dave
# moved to Pudong." and "Dave moved to Pudong" are recognised as one fact.
_CITATION = re.compile(r"\[\^[^\]]+\]")
_BLOCKS = re.compile(r"(?:\s+\^[A-Za-z0-9-]+)+\s*$")


def _essence(line: str) -> str:
    body = _BLOCKS.sub("", _CITATION.sub("", line))
    return re.sub(r"[^a-z0-9 ]+", "", body.lower()).strip()


def _is_fact(line: str) -> bool:
    return bool(line.strip()) and not line.startswith("[^") and bool(
        BLOCK_SUFFIX.search(line)
    )


@dataclass
class TidyReport:
    """What changed, per file, so a pass that did nothing says so."""

    merged: int = 0
    pruned_notes: int = 0
    pruned_headings: int = 0
    files: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def changed(self) -> bool:
        return bool(self.files)

    def as_dict(self) -> dict[str, Any]:
        result = {
            "merged_duplicates": self.merged,
            "pruned_footnotes": self.pruned_notes,
            "pruned_headings": self.pruned_headings,
            "files": self.files,
        }
        if self.error is not None:
            result["error"] = self.error
        return result


def _merge_lines(lines: list[str], keep: int, drop: int) -> None:
    """Fold `lines[drop]` into `lines[keep]`: its ID and citations move onto
    `keep`, its own wording is gone.

    The format keeps a merged paragraph's absorbed IDs as a run of suffixes,
    the last being its identity, so links into any of them still resolve.
    Shared by `_merge_duplicates`, which picks `keep`/`drop` by matching
    text, and `merge_blocks`, which picks them by a model's verdict on two
    paragraphs that do not match — the fold itself does not care which.
    """
    block = BLOCK_SUFFIX.search(lines[keep])
    # A caller only ever passes an index `_is_fact` already matched against
    # this same pattern, so this cannot be None; computed once and reused
    # rather than trusting that invariant a second time in the same
    # expression.
    dup_block = BLOCK_SUFFIX.search(lines[drop])
    absorbed = re.findall(r"\^([A-Za-z0-9-]+)", lines[drop][dup_block.start():])
    citations = "".join(
        f"[^{name}]" for name in SINGLE_CITATION.findall(lines[drop])
        if f"[^{name}]" not in lines[keep]
    )
    head = lines[keep][:block.start()]
    tail = lines[keep][block.start():]
    # Absorbed IDs go before the identity, which stays last.
    existing = re.findall(r"\^([A-Za-z0-9-]+)", tail)
    every = [*absorbed, *existing]
    lines[keep] = head + citations + "".join(f" ^{name}" for name in every)


def _merge_duplicates(lines: list[str], report: TidyReport) -> list[str]:
    """One paragraph per fact, carrying every block ID that said it."""
    first: dict[str, int] = {}
    drop: set[int] = set()
    for index, line in enumerate(lines):
        if not _is_fact(line):
            continue
        key = _essence(line)
        if not key:
            continue
        if key not in first:
            first[key] = index
            continue
        _merge_lines(lines, first[key], index)
        drop.add(index)
        report.merged += 1
    return [line for index, line in enumerate(lines) if index not in drop]


def _find_block_line(lines: list[str], block_id: str) -> int | None:
    """The index of the paragraph whose current suffix run carries this ID.

    A merge can leave an ID anywhere in the run, not only last, so this
    checks every ID a line carries rather than only the one `BLOCK_SUFFIX`
    itself captures (that group is the run's last ID: its identity).
    """
    for index, line in enumerate(lines):
        block = BLOCK_SUFFIX.search(line)
        if block and block_id in re.findall(
            r"\^([A-Za-z0-9-]+)", line[block.start():]
        ):
            return index
    return None


def merge_blocks(path: Path, a: str, b: str) -> None:
    """Fold the paragraphs named by two block IDs, in one file, into one.

    The mechanical half of a merge verdict on a `merge_candidates` pair:
    which two paragraphs is the judgement, folding them is `_merge_lines`,
    the same code `_merge_duplicates` uses for a pair a string comparison
    already caught. Whichever paragraph is earlier in the file keeps its
    wording, matching that policy exactly rather than inventing a second one
    for a judged pair — the caller names two IDs, not a winner.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    index_a = _find_block_line(lines, a)
    index_b = _find_block_line(lines, b)
    if index_a is None:
        raise ValueError(f"no such block in {path.name}: {a}")
    if index_b is None:
        raise ValueError(f"no such block in {path.name}: {b}")
    if index_a == index_b:
        raise ValueError(f"{a} and {b} are already one paragraph")
    keep, gone = sorted((index_a, index_b))
    _merge_lines(lines, keep, gone)
    del lines[gone]
    path.write_text("\n".join(_collapse(lines)) + "\n", encoding="utf-8")


def _prune_notes(lines: list[str], report: TidyReport) -> list[str]:
    """Footnote definitions nothing cites are dead weight the parser rejects."""
    cited = {
        name for line in lines if not line.startswith("[^")
        for name in SINGLE_CITATION.findall(line)
    }
    kept = []
    for line in lines:
        found = definition_match(line)
        if found and found.group("id") not in cited:
            report.pruned_notes += 1
            continue
        kept.append(line)
    return kept


def _prune_headings(lines: list[str], report: TidyReport) -> list[str]:
    """A heading with no fact under it survived a merge and says nothing now."""
    kept: list[str] = []
    for index, line in enumerate(lines):
        if not re.match(r"^#{2,6} ", line):
            kept.append(line)
            continue
        empty = True
        for following in lines[index + 1:]:
            if re.match(r"^#{1,6} ", following):
                break
            if following.strip() and not following.startswith("[^"):
                empty = False
                break
        if empty:
            report.pruned_headings += 1
            continue
        kept.append(line)
    return kept


def _collapse(lines: list[str]) -> list[str]:
    """At most one blank line between paragraphs."""
    out: list[str] = []
    for line in lines:
        if not line.strip() and out and not out[-1].strip():
            continue
        out.append(line)
    while out and not out[-1].strip():
        out.pop()
    return out


def tidy(memory_dir: str | Path) -> dict[str, Any]:
    """Deduplicate, prune and normalise every topic file, transactionally.

    Runs the same commit path a model's edit would, so a pass that would
    produce a file the contract rejects changes nothing and reports why.

    The baseline snapshot is taken non-strict: this pass exists to repair a
    workspace, including one that does not currently meet the topic contract
    (an uncited footnote left behind some other way, say), and a strict
    snapshot would refuse to even look at the file that needs the repair.
    """
    workspace = MemoryWorkspace(memory_dir)
    try:
        report = TidyReport()
        root = workspace.stage_dir / "topics"
        if not root.is_dir():
            return report.as_dict()

        before_units, before_block_ids, before_topics, before_sources = (
            workspace.baseline(strict=False)
        )
        for path in sorted(root.rglob("*.md")):
            original = path.read_text(encoding="utf-8")
            lines = original.split("\n")
            lines = _merge_duplicates(lines, report)
            lines = _prune_notes(lines, report)
            lines = _prune_headings(lines, report)
            rewritten = "\n".join(_collapse(lines)) + "\n"
            if rewritten != original:
                path.write_text(rewritten, encoding="utf-8")
                report.files.append(
                    path.relative_to(workspace.stage_dir).as_posix()
                )
        if not report.changed:
            return report.as_dict()
        try:
            workspace.commit_edits(
                before_units, before_block_ids, before_topics, before_sources
            )
        except Exception as exc:
            # commit_edits has already rolled the stage back to what is on
            # disk and re-raised; a pass that would have produced a
            # contract-invalid file changes nothing, the same way a rejected
            # shell or edit_file call does (workspace/tool_support.record).
            return TidyReport(error=str(exc)).as_dict()
        return report.as_dict()
    finally:
        # Never installed on the model-pass path either: run_pass tears its
        # workspace's stage down the same way once the turn is over.
        shutil.rmtree(workspace.stage_dir, ignore_errors=True)


_WORD = re.compile(r"[a-z0-9]+")


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity over words: cheap, no dependency, and enough to
    shortlist pairs for a model to judge — not to decide for it."""
    left = set(_WORD.findall(a.lower()))
    right = set(_WORD.findall(b.lower()))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def merge_candidates(
    memory_dir: str | Path, *, threshold: float = 0.28
) -> list[dict[str, Any]]:
    """Pairs, within one topic file, worded differently but likely one fact.

    `_merge_duplicates` above only catches a paragraph textually identical
    to another; two records of the same fact in different words survive it
    as two paragraphs, and whether they mean the same thing is a judgement
    this module does not have. This shortlists the pairs close enough in
    wording to be worth putting to a model — run after `tidy`, so a pair a
    string comparison already merged is one paragraph by the time this
    looks, not a pair.

    0.28 is measured, not guessed: run against a real workspace
    (`comp-workspaces/final1`), it keeps exactly the four pairs that are
    actually one fact restated in different words — including the clearest
    one in that workspace, two Melanie paragraphs that both just say
    cherishing family time makes her happy — and drops the pairs that only
    share a topic without being one fact, such as Caroline researching
    adoption agencies (0.257 similarity against the paragraph about her
    later being thrilled to raise a child, itself a different fact: the
    search for an agency and her feelings once she found one). A denser,
    differently shaped file (a 70-paragraph software project log, sharing
    vocabulary like the author's name and framework across unrelated facts)
    proposes one pair at this threshold out of 2,415 possible, not a flood.
    """
    topics = Path(memory_dir) / "topics"
    if not topics.is_dir():
        return []
    by_file: dict[str, list[Any]] = {}
    for unit in parse_topic_tree(topics):
        if unit.content.strip():
            by_file.setdefault(unit.topic_path, []).append(unit)
    candidates = []
    for path, units in by_file.items():
        for left, right in combinations(units, 2):
            if left.content == right.content:
                # Not two paragraphs: one paragraph a prior merge already
                # gave two IDs to (`_merge_duplicates` keeps every absorbed
                # ID resolvable, per-ID, to the same content). Two genuinely
                # distinct paragraphs cannot reach exact content equality
                # and survive `tidy` — its own dedup is essence equality,
                # a looser match than this — so this can only be that case.
                continue
            similarity = _token_overlap(left.content, right.content)
            if similarity >= threshold:
                candidates.append({
                    "path": f"topics/{path}",
                    "block_ids": (left.memory_id, right.memory_id),
                    "texts": (left.content, right.content),
                    "similarity": round(similarity, 3),
                })
    candidates.sort(key=lambda row: row["similarity"], reverse=True)
    return candidates
