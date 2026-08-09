"""remember, update and forget, exercised as the tools a writer calls.

`remember` and `update` both render through `memory.writing.render`; these
tests run the tool calls end to end so a rendering bug shows up as a broken
paragraph in a real file, not just as a wrong string on the render module.
"""

import asyncio
from pathlib import Path

from memory.markdown.syntax import BLOCK_SUFFIX
from memory.workspace.staging import MemoryWorkspace
from memory.writing.tools import writing_tools


def call(tool, **arguments) -> str:
    reply = asyncio.run(tool.handler(arguments))
    return reply["content"][0]["text"]


def verbs(tmp_path: Path):
    (tmp_path / "topics").mkdir()
    (tmp_path / "sources").mkdir()
    (tmp_path / "core.md").write_text("# Core\n", encoding="utf-8")
    space = MemoryWorkspace(tmp_path)
    # A footnote has to cite evidence that exists, so archive one first.
    space.archive_sessions([{
        "observation_date": "2026-08-09",
        "turns": [("user", "I have moved to Pudong.")],
        "refs": ["D1:1"],
    }])
    given = writing_tools(space, [])
    return space, {definition.name: definition for definition in given}


def test_remember_writes_the_markup_the_contract_needs(tmp_path: Path):
    """A weak model cannot hand-write block IDs and footnotes reliably.

    Twenty turns of `memory block ID required` is what that looks like, so
    the fact arrives as prose and the runtime supplies the markup.
    """
    space, given = verbs(tmp_path)

    output = call(
        given["remember"],
        subject="Dave", kind="person",
        fact="Dave moved to Pudong.",
        sources=["D1:1"],
    )

    written = (tmp_path / "topics/people/dave.md").read_text()
    assert "wrote" in output
    assert "Dave moved to Pudong." in written
    assert "[^e-" in written and "Sources:" in written
    # The Runtime assigns identity; nothing the model wrote survives as one.
    assert "new-block" not in written


def test_remember_needs_the_evidence_it_is_citing(tmp_path: Path):
    space, given = verbs(tmp_path)

    output = call(
        given["remember"],
        subject="Dave", kind="person",
        fact="Dave moved to Pudong.",
        sources=[],
    )

    assert "sources is required" in output
    assert not (tmp_path / "topics/people/dave.md").exists()


def test_a_second_fact_joins_the_file_under_its_heading(tmp_path: Path):
    space, given = verbs(tmp_path)
    call(
        given["remember"], subject="Dave", kind="person",
        fact="Dave moved to Pudong.", sources=["D1:1"],
    )

    call(
        given["remember"], subject="Dave", kind="person",
        fact="Dave paints on weekends.", sources=["D1:1"], heading="Hobbies",
    )

    written = (tmp_path / "topics/people/dave.md").read_text()
    assert "## Hobbies" in written
    assert written.count("[^e-") == 4  # two citations, two definitions
    assert "Dave moved to Pudong." in written


def test_the_subject_decides_the_file_not_the_model(tmp_path: Path):
    """Where a person's facts live is a rule, not a judgement.

    Left to the model it comes out `people/caroline.md` one pass and
    `relationship/caroline-and-melanie.md` the next, and one person ends up
    as two subjects.
    """
    space, given = verbs(tmp_path)

    call(
        given["remember"], subject="Dave Chen", kind="person",
        fact="Dave moved to Pudong.", sources=["D1:1"],
    )

    assert (tmp_path / "topics/people/dave-chen.md").is_file()


def test_update_links_the_new_fact_to_the_one_it_replaces(tmp_path: Path):
    space, given = verbs(tmp_path)
    call(
        given["remember"], subject="Dave", kind="person",
        fact="Dave lives in Pudong.", sources=["D1:1"],
    )

    output = call(
        given["update"], subject="Dave", kind="person",
        replaces="Dave lives in Pudong",
        fact="Dave lives in Hongqiao.",
        sources=["D1:1"],
    )

    written = (tmp_path / "topics/people/dave.md").read_text()
    assert "wrote" in output
    assert "Dave lives in Hongqiao." in written
    # The link target gets rewritten by the contract's own normalization;
    # what this verb owns is the "(replaces ...)" clause and the anchor.
    assert "(replaces [the earlier note](" in written
    assert "#^" in written
    # What was true before stays on the page; only the new paragraph links.
    assert "Dave lives in Pudong." in written


def test_update_needs_a_phrase_that_matches_one_existing_fact(tmp_path: Path):
    space, given = verbs(tmp_path)
    call(
        given["remember"], subject="Dave", kind="person",
        fact="Dave lives in Pudong.", sources=["D1:1"],
    )

    output = call(
        given["update"], subject="Dave", kind="person",
        replaces="Dave lives in Narnia",
        fact="Dave lives in Hongqiao.",
        sources=["D1:1"],
    )

    assert "no recorded fact contains" in output
    assert "Dave lives in Hongqiao." not in (
        tmp_path / "topics/people/dave.md"
    ).read_text()


def test_forget_withdraws_the_wording_but_keeps_the_block_id(tmp_path: Path):
    space, given = verbs(tmp_path)
    call(
        given["remember"], subject="Dave", kind="person",
        fact="Dave has a cat.", sources=["D1:1"],
    )
    before = (tmp_path / "topics/people/dave.md").read_text()
    paragraph = next(
        line for line in before.splitlines() if "Dave has a cat" in line
    )
    block_id = BLOCK_SUFFIX.search(paragraph).group("id")

    output = call(
        given["forget"], subject="Dave", kind="person", fact="Dave has a cat"
    )

    written = (tmp_path / "topics/people/dave.md").read_text()
    assert "withdrawn" in output
    assert "Dave has a cat." not in written
    assert "A fact recorded here was withdrawn." in written
    # Identity survives so links through it still resolve.
    assert f"^{block_id}" in written
