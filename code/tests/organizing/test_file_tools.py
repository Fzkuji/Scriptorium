"""File tools for a runtime that brings none of its own.

Claude Code has Read, Write and Edit built in; an OpenAI-compatible model has
only what it is handed. Without these it is told by the prompts to use a Write
tool it does not have, falls back to shell redirection, and spends its whole
turn budget being refused.
"""

import asyncio
from pathlib import Path

import pytest

from memory.organizing.tools import organizing_tools
from memory.workspace.staging import MemoryWorkspace


def call(tool, **arguments) -> str:
    reply = asyncio.run(tool.handler(arguments))
    return reply["content"][0]["text"]


TOPIC = (
    "# Dave\n\n"
    "Dave moved to Pudong.[^new-evidence-move] ^new-block-move\n\n"
    "[^new-evidence-move]: Time: `2026-08-09`; Sources: D1:1\n"
)


def tools(tmp_path: Path):
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
    given = organizing_tools(space, [], file_tools=True)
    return space, {definition.name: definition for definition in given}


def test_a_runtime_with_its_own_file_tools_is_given_only_the_shell(tmp_path: Path):
    (tmp_path / "topics").mkdir()
    (tmp_path / "core.md").write_text("# Core\n", encoding="utf-8")
    space = MemoryWorkspace(tmp_path)

    names = [d.name for d in organizing_tools(space, [])]

    assert names == ["shell"]


def test_writing_a_topic_installs_it(tmp_path: Path):
    space, given = tools(tmp_path)

    output = call(
        given["write_file"],
        path="topics/people/dave.md",
        content=TOPIC,
    )

    assert "wrote" in output
    assert "Dave moved to Pudong" in (
        tmp_path / "topics/people/dave.md"
    ).read_text()


def test_a_path_outside_the_workspace_is_refused(tmp_path: Path):
    space, given = tools(tmp_path)

    output = call(
        given["write_file"], path="../escape.md", content="# Nope\n"
    )

    assert "escapes the workspace" in output
    assert not (tmp_path.parent / "escape.md").exists()


def test_the_same_failing_call_is_named_as_a_repeat(tmp_path: Path):
    """A model that cannot see why it was refused will try again.

    Twenty identical rejected edits is how a turn budget disappears, so the
    second one has to say that it is the second one.
    """
    space, given = tools(tmp_path)

    first = call(
        given["edit_file"], path="topics/absent.md",
        old_text="a", new_text="b",
    )
    second = call(
        given["edit_file"], path="topics/absent.md",
        old_text="a", new_text="b",
    )

    assert "attempt" not in first
    assert "attempt 2 at the same call" in second


def test_a_rejected_edit_leaves_the_workspace_alone(tmp_path: Path):
    space, given = tools(tmp_path)
    call(given["write_file"], path="topics/people/dave.md", content=TOPIC)
    before = (tmp_path / "topics/people/dave.md").read_text()

    output = call(
        given["write_file"],
        path="topics/people/dave.md",
        content="# Dave\n\nA paragraph citing a footnote nobody defined.[^e9]\n",
    )

    assert "Error" in output or "error" in output.lower()
    assert (tmp_path / "topics/people/dave.md").read_text() == before


def test_reading_back_a_file_the_model_just_wrote(tmp_path: Path):
    space, given = tools(tmp_path)
    call(given["write_file"], path="topics/people/dave.md", content=TOPIC)

    output = call(given["read_file"], path="topics/people/dave.md")

    assert "Dave moved to Pudong" in output


def test_the_write_stage_gets_only_its_three_verbs(tmp_path: Path):
    """Writing states facts; it does not rearrange files.

    A shell and a whole-file writer in this pass are latitude a weak model
    spends on rejected edits rather than on recording the conversation.
    """
    space, _ = tools(tmp_path)

    writing = organizing_tools(space, [], file_tools=True, stage="write")
    organizing = organizing_tools(space, [], file_tools=True, stage="organize")

    assert [d.name for d in writing] == ["remember", "update", "forget"]
    assert "edit_file" in [d.name for d in organizing]
