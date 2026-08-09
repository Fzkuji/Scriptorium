"""tidy() against real workspaces: it had never been run against one before.

Every fixture goes through `MemoryWorkspace.archive_sessions` for its source
refs and is hand-written for its topic file, then re-parsed with the real
parser after `tidy` runs — reading the code is not enough to know that the
suffix run it writes is the one `memory/markdown/parser.py` reads back, or
that a rejected commit really leaves disk untouched.

Footnote labels here are already-stable-shaped (`e-fact-a`, not `e1`): a
bare `e<digits>` is a writer placeholder the commit path rewrites to a
fresh stable ID, which would make the fixture's own citation text change
out from under the assertions for reasons that have nothing to do with tidy.
"""

import importlib
from pathlib import Path

from memory.markdown import parse_topic_tree
from memory.organizing.tidy import merge_candidates, tidy
from memory.workspace import MemoryWorkspace

# `memory.organizing`'s __init__ does `from .tidy import tidy`, which rebinds
# the package attribute `organizing.tidy` to the function, shadowing the
# submodule of the same name — so `import memory.organizing.tidy as x` would
# hand back the function too. importlib goes by the registered module name
# instead of that attribute chain, so it reaches the actual submodule.
tidy_module = importlib.import_module("memory.organizing.tidy")


def _seed(memory_dir: Path, refs: list[str]) -> None:
    """Archive one source turn per ref, so a footnote citing it is valid."""
    MemoryWorkspace(memory_dir).archive_sessions([{
        "observation_date": "2026-01-01",
        "turns": [("user", f"about {ref}") for ref in refs],
        "refs": refs,
    }])


def _topic(memory_dir: Path, relative: str, text: str) -> Path:
    path = memory_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_identical_facts_merge_and_every_absorbed_id_still_resolves(tmp_path):
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1", "D1:2"])
    _topic(memory_dir, "topics/people/dave.md", (
        "# Dave\n\n"
        "Dave moved to Pudong.[^e-fact-a] ^blockidaaa\n\n"
        "Dave moved to Pudong.[^e-fact-b] ^blockidbbb\n\n"
        "[^e-fact-a]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
        "[^e-fact-b]: Time: `2026-01-02`; Sources: [D1:2](../../sources/D1.md#d1-2)\n"
    ))

    report = tidy(memory_dir)

    assert report == {
        "merged_duplicates": 1,
        "pruned_footnotes": 0,
        "pruned_headings": 0,
        "files": ["topics/people/dave.md"],
    }
    # The merged paragraph parses, and both of the original IDs it absorbed
    # still resolve to a unit — a link into either one still finds the fact.
    units = parse_topic_tree(memory_dir / "topics")
    assert {unit.memory_id for unit in units} == {"blockidaaa", "blockidbbb"}
    assert {unit.content for unit in units} == {"Dave moved to Pudong."}
    # Both citations survived the merge, so both footnotes stay defined.
    text = (memory_dir / "topics/people/dave.md").read_text()
    assert "[^e-fact-a]:" in text and "[^e-fact-b]:" in text


def test_differently_worded_facts_are_not_merged(tmp_path):
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1", "D1:2"])
    _topic(memory_dir, "topics/people/dave.md", (
        "# Dave\n\n"
        "Dave moved to Pudong.[^e-fact-a] ^blockidaaa\n\n"
        "Dave relocated to Shanghai.[^e-fact-b] ^blockidbbb\n\n"
        "[^e-fact-a]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
        "[^e-fact-b]: Time: `2026-01-02`; Sources: [D1:2](../../sources/D1.md#d1-2)\n"
    ))

    report = tidy(memory_dir)

    assert report["merged_duplicates"] == 0
    assert not report["files"]
    units = parse_topic_tree(memory_dir / "topics")
    assert {unit.memory_id for unit in units} == {"blockidaaa", "blockidbbb"}


def test_uncited_footnote_is_pruned_even_though_it_makes_the_file_unparseable_strictly(
    tmp_path,
):
    """The file this pass has to fix does not meet the contract yet, or
    there would be nothing for it to fix: the baseline it snapshots has to
    tolerate that, not refuse to look."""
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1"])
    _topic(memory_dir, "topics/people/dave.md", (
        "# Dave\n\n"
        "Dave moved to Pudong.[^e-fact-a] ^blockidaaa\n\n"
        "[^e-fact-a]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
        "[^e-stale]: Time: `2025-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
    ))

    report = tidy(memory_dir)

    assert report["pruned_footnotes"] == 1
    assert "error" not in report
    text = (memory_dir / "topics/people/dave.md").read_text()
    assert "[^e-stale]:" not in text
    assert "Dave moved to Pudong." in text
    # And the result now meets the contract strictly, footnote and all.
    units = parse_topic_tree(memory_dir / "topics")
    assert units[0].memory_id == "blockidaaa"


def test_a_heading_with_live_facts_survives_when_footnotes_collect_at_the_foot(
    tmp_path,
):
    """The real file shape `writing.render.insert` produces: every heading's
    own paragraph sits directly under it, and every footnote collects at the
    very end, under whatever heading is textually last."""
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1", "D1:2"])
    _topic(memory_dir, "topics/people/dave.md", (
        "# Dave\n\n"
        "## Move history\n\n"
        "Dave moved to Pudong.[^e-fact-a] ^blockidaaa\n\n"
        "## Employment\n\n"
        "Dave works at Acme.[^e-fact-b] ^blockidbbb\n\n"
        "[^e-fact-a]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
        "[^e-fact-b]: Time: `2026-01-02`; Sources: [D1:2](../../sources/D1.md#d1-2)\n"
    ))

    report = tidy(memory_dir)

    assert report["pruned_headings"] == 0
    text = (memory_dir / "topics/people/dave.md").read_text()
    assert "## Move history" in text
    assert "## Employment" in text


def test_a_heading_emptied_by_a_merge_is_pruned(tmp_path):
    """The case `_prune_headings` exists for: a merge drops the only
    paragraph a heading had, and nothing but the heading line is left."""
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1", "D1:2"])
    _topic(memory_dir, "topics/people/dave.md", (
        "# Dave\n\n"
        "## First telling\n\n"
        "Dave moved to Pudong.[^e-fact-a] ^blockidaaa\n\n"
        "## Second telling\n\n"
        "Dave moved to Pudong.[^e-fact-b] ^blockidbbb\n\n"
        "[^e-fact-a]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
        "[^e-fact-b]: Time: `2026-01-02`; Sources: [D1:2](../../sources/D1.md#d1-2)\n"
    ))

    report = tidy(memory_dir)

    assert report["merged_duplicates"] == 1
    assert report["pruned_headings"] == 1
    text = (memory_dir / "topics/people/dave.md").read_text()
    assert "## First telling" in text
    assert "## Second telling" not in text


def test_a_tidy_that_would_break_the_contract_leaves_the_workspace_untouched(
    tmp_path, monkeypatch
):
    """However tidy's own logic might go wrong, breaking the contract must
    not reach disk: this drives that path directly, the way a bug in
    `_merge_duplicates` that drops a block ID outright would."""
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1"])
    _topic(memory_dir, "topics/people/dave.md", (
        "# Dave\n\n"
        "Dave moved to Pudong.[^e-fact-a] ^blockidaaa\n\n"
        "[^e-fact-a]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
    ))
    before = (memory_dir / "topics/people/dave.md").read_bytes()

    monkeypatch.setattr(
        tidy_module, "_merge_duplicates",
        lambda lines, report: [line for line in lines if "blockidaaa" not in line],
    )

    report = tidy(memory_dir)

    assert "error" in report
    assert report["files"] == []
    after = (memory_dir / "topics/people/dave.md").read_bytes()
    assert after == before


def test_a_pass_that_changes_nothing_reports_no_files(tmp_path):
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1"])
    _topic(memory_dir, "topics/people/dave.md", (
        "# Dave\n\n"
        "Dave moved to Pudong.[^e-fact-a] ^blockidaaa\n\n"
        "[^e-fact-a]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
    ))
    before = (memory_dir / "topics/people/dave.md").read_bytes()

    report = tidy(memory_dir)

    assert report == {
        "merged_duplicates": 0,
        "pruned_footnotes": 0,
        "pruned_headings": 0,
        "files": [],
    }
    assert (memory_dir / "topics/people/dave.md").read_bytes() == before


def test_a_workspace_with_no_topics_directory_is_a_no_op(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    assert tidy(memory_dir) == {
        "merged_duplicates": 0,
        "pruned_footnotes": 0,
        "pruned_headings": 0,
        "files": [],
    }


def test_two_differently_worded_records_of_one_fact_are_proposed_as_a_candidate(
    tmp_path,
):
    """The pair `_merge_duplicates` cannot catch, worded from a real
    duplicate found in `comp-workspaces/final1/topics/people/melanie.md`
    (two paragraphs that both just say cherishing family time brings her
    happiness) — the case `merge_candidates` exists for."""
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1", "D1:2"])
    _topic(memory_dir, "topics/people/dave.md", (
        "# Dave\n\n"
        "Dave cherishes time with his family, as it is when he really "
        "feels alive and happy.[^e-fact-a] ^blockidaaa\n\n"
        "Dave feels that cherishing moments with his family is important "
        "and brings him happiness.[^e-fact-b] ^blockidbbb\n\n"
        "[^e-fact-a]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
        "[^e-fact-b]: Time: `2026-01-02`; Sources: [D1:2](../../sources/D1.md#d1-2)\n"
    ))

    candidates = merge_candidates(memory_dir)

    assert len(candidates) == 1
    row = candidates[0]
    assert row["path"] == "topics/people/dave.md"
    assert set(row["block_ids"]) == {"blockidaaa", "blockidbbb"}
    assert row["similarity"] >= 0.28


def test_two_genuinely_different_facts_are_not_proposed(tmp_path):
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1", "D1:2"])
    _topic(memory_dir, "topics/people/dave.md", (
        "# Dave\n\n"
        "Dave moved to Pudong.[^e-fact-a] ^blockidaaa\n\n"
        "Dave adopted a rescue dog named Biscuit.[^e-fact-b] ^blockidbbb\n\n"
        "[^e-fact-a]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
        "[^e-fact-b]: Time: `2026-01-02`; Sources: [D1:2](../../sources/D1.md#d1-2)\n"
    ))

    assert merge_candidates(memory_dir) == []


def test_identical_facts_already_merged_by_tidy_are_not_proposed_again(tmp_path):
    """`merge_candidates` runs after `tidy` in `reorganize`; this drives
    that ordering directly rather than assuming it. Once `tidy` has folded
    the identical pair into one paragraph, there is only one paragraph left
    — not a pair a string comparison already settled, but no pair at all."""
    memory_dir = tmp_path / "memory"
    _seed(memory_dir, ["D1:1", "D1:2"])
    _topic(memory_dir, "topics/people/dave.md", (
        "# Dave\n\n"
        "Dave moved to Pudong.[^e-fact-a] ^blockidaaa\n\n"
        "Dave moved to Pudong.[^e-fact-b] ^blockidbbb\n\n"
        "[^e-fact-a]: Time: `2026-01-01`; Sources: [D1:1](../../sources/D1.md#d1-1)\n"
        "[^e-fact-b]: Time: `2026-01-02`; Sources: [D1:2](../../sources/D1.md#d1-2)\n"
    ))

    tidy(memory_dir)

    assert merge_candidates(memory_dir) == []


def test_merge_candidates_on_a_workspace_with_no_topics_directory_is_empty(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    assert merge_candidates(memory_dir) == []
