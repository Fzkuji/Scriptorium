"""Topic reorganization: the model-driven half of the organize stage.

Writing's API (`write_sessions` and what it renders) lives in
`memory.writing.session`, running the shared agent pass for that stage.
`reorganize` runs the same shared pass for this one, but only after `tidy`
and `split`: there is no point asking a model to think about duplicates a
string comparison can merge or a file only its size decided to break up, so
the two deterministic passes always go first and the model is left with
only what actually takes judgement — moving what's left where it belongs,
and weighing the handful of worded-differently pairs `tidy.merge_candidates`
found for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import MemoryConfig
from ..prompts import ORGANIZE_MEMORY
from ..workspace import MemoryWorkspace
from ..workspace.agent_pass import run_pass
from ..workspace.tool_support import edit_tools
from .split import split
from .tidy import _find_block_line, merge_blocks, merge_candidates, tidy
from .tools import _repair_guidance, organizing_tools


def _merge_candidate_section(candidates: list[dict[str, Any]]) -> str:
    """The prompt text asking the model to judge `merge_candidates`' pairs.

    Empty when there are none, so a pass with nothing to weigh does not ask
    the model to weigh nothing — and, since `_tools_for` below only adds the
    merge tool alongside this text, does not hand out a tool with nothing
    to call it on either.
    """
    if not candidates:
        return ""
    pairs = "\n".join(
        f'- {row["path"]}: ^{row["block_ids"][0]} "{row["texts"][0]}" '
        f'vs ^{row["block_ids"][1]} "{row["texts"][1]}"'
        for row in candidates
    )
    return (
        "\nSome paragraphs below say the same fact in different words. Call "
        "merge_paragraphs(a, b) with their block IDs for a pair that does; "
        "leave a pair that doesn't as it is, no call needed.\n\n" + pairs + "\n"
    )


def _shared_file(topics: Path, a: str, b: str) -> str:
    """The one topic file whose current text carries both block IDs.

    Raised as `ValueError` rather than returning `None`: `merge_paragraphs`
    below runs this inside the same try `edit_tools.record` already wraps
    every other organizing edit in, so it becomes the model's next
    correction the same way a rejected `edit_file` call does.
    """
    for candidate in sorted(topics.rglob("*.md")):
        lines = candidate.read_text(encoding="utf-8").split("\n")
        if (
            _find_block_line(lines, a) is not None
            and _find_block_line(lines, b) is not None
        ):
            return candidate.relative_to(topics.parent).as_posix()
    raise ValueError(f"no single topic file has both blocks: {a}, {b}")


def _merge_tool(workspace: MemoryWorkspace, audit: list[dict[str, Any]]):
    """The one tool this pass adds beyond `organizing_tools`.

    Folding two paragraphs into one is mechanical once the pair is chosen —
    `tidy.merge_blocks` does it, the same code `tidy` itself uses to fold an
    exact duplicate — so the model only ever names a pair, never hand-writes
    a block-ID suffix run. A tool call is also validated by its schema
    before it runs; free text asking the model to describe its verdicts and
    parsing that text back out is not, and the mandated model already fails
    the topic format often enough hand-writing far simpler edits (see
    `memory/prompts/organize.py`'s neighbour, `SYSTEM_PROMPT`). Defined here
    rather than folded into `organizing_tools` because a merge tool is only
    ever useful alongside `merge_candidates`' pairs, and every other
    organizing pass runs without either.
    """
    from claude_agent_sdk import tool

    apply_edit, record = edit_tools(workspace, audit, _repair_guidance)

    @tool(
        "merge_paragraphs",
        (
            "Fold two paragraphs that state one fact in different words "
            "into one paragraph. Both block IDs still resolve afterwards. "
            "Call this only for a pair you judge to be the same fact; a "
            "pair that isn't needs no call."
        ),
        {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )
    async def merge_paragraphs(arguments: dict[str, Any]) -> dict[str, Any]:
        def run() -> str:
            a, b = str(arguments.get("a", "")), str(arguments.get("b", ""))
            path = _shared_file(workspace.stage_dir / "topics", a, b)

            def change(target: Path) -> None:
                merge_blocks(target, a, b)

            return apply_edit(path, change)
        return record("merge_paragraphs", arguments, run)

    return merge_paragraphs


def _tools_for(agent: Any, candidates: list[dict[str, Any]]):
    """The tool list this pass hands the model, given the workspace it edits.

    The merge tool is only added when there are candidates to weigh, so a
    pass over an already-tidy workspace hands out exactly what it did
    before this existed.
    """
    def build(workspace: MemoryWorkspace, audit: list[dict[str, Any]]):
        tools = organizing_tools(
            workspace, audit,
            file_tools=not getattr(agent, "has_file_tools", True),
            stage="organize",
        )
        if candidates:
            tools = [*tools, _merge_tool(workspace, audit)]
        return tools
    return build


def _scoped_paths(touched: set[str], moved: dict[str, list[str]]) -> list[str]:
    """The topic paths `touched` names, with any `split` just replaced with
    a directory substituted for the new files that now hold its content."""
    expanded: set[str] = set()
    for path in touched:
        posix = Path(path).as_posix()
        if not posix.startswith("topics/"):
            continue
        expanded.update(moved.get(posix, [posix]))
    return sorted(expanded)


def reorganize(
    memory_dir: str | Path,
    *,
    agent: Any,
    touched: set[str] | None = None,
    usage_logger: Any | None = None,
    config: MemoryConfig | None = None,
) -> dict[str, Any]:
    """Tidy, split, then reorganize Topic files. ``touched`` scopes the model pass.

    Passing ``touched=None`` reorganizes every Topic file, which is the
    end-of-build pass; `tidy` and `split` always run over the whole tree
    regardless, since neither spends a model turn and scoping either would
    only save time the model pass is what actually spends.

    Returns what each part did: `{"tidy": ..., "split": ..., "organize":
    ...}`. The three are independent outcomes — a workspace can be fully
    tidy and already right-sized with nothing left for the model, and that
    is not a failure of either deterministic pass.
    """
    config = config or MemoryConfig()
    tidy_report = tidy(memory_dir)
    split_report = split(memory_dir)
    moved = {row["from"]: row["to"] for row in split_report.get("files", [])}

    root = Path(memory_dir) / "topics"
    if touched is None:
        paths = sorted(
            path.relative_to(memory_dir).as_posix()
            for path in root.rglob("*.md")
        )
    else:
        paths = _scoped_paths(touched, moved)
    if not paths:
        return {"tidy": tidy_report, "split": split_report, "organize": []}
    candidates = [
        row for row in merge_candidates(memory_dir) if row["path"] in paths
    ]
    audit = run_pass(
        memory_dir,
        agent=agent,
        task=ORGANIZE_MEMORY.format(
            topic_paths="\n".join(paths),
            merge_candidates=_merge_candidate_section(candidates),
        ),
        usage_logger=usage_logger,
        config=config,
        stage="organize",
        tools=_tools_for(agent, candidates),
        guidance=_repair_guidance,
    )
    return {"tidy": tidy_report, "split": split_report, "organize": audit}
