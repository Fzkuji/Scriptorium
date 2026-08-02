"""Visible memory files and tool availability for each retrieval condition."""

from pathlib import Path
from typing import Any

from .schemas import CONDITION_VIEWS, TOOL_DEFINITIONS


def memory_files(
    memory_dir: Path,
    condition: str = "native",
    *,
    include_recent: bool = True,
) -> list[Path]:
    root = memory_dir.resolve()
    views = CONDITION_VIEWS.get(condition)
    if condition != "native" and views is None:
        raise ValueError(f"unknown V11 condition: {condition}")
    result = [
        path
        for path in root.rglob("*.md")
        if path.is_file()
        and not path.is_symlink()
        and (
            views is None
            or path.relative_to(root).parts[0] in views
            or path.relative_to(root).as_posix() == "core.md"
        )
    ]
    recent = root / "recent_events.jsonl"
    if (
        recent.is_file()
        and not recent.is_symlink()
        and include_recent
        and (views is None or "recent" in views)
    ):
        result.append(recent)
    return sorted(
        result, key=lambda path: path.relative_to(root).as_posix()
    )


def read_memory_file(
    memory_dir: Path,
    raw_path: object,
    condition: str = "native",
    *,
    include_recent: bool = True,
) -> str:
    root = memory_dir.resolve()
    relative = Path(str(raw_path or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("memory path escapes workspace")
    path = (root / relative).resolve()
    path.relative_to(root)
    allowed = path in memory_files(
        root, condition, include_recent=include_recent
    )
    if path.is_symlink() or not path.is_file() or not allowed:
        raise ValueError("memory path is not an allowed memory file")
    return path.read_text(encoding="utf-8")


def tools_for(condition: str) -> list[dict[str, Any]]:
    if condition != "native" and condition not in CONDITION_VIEWS:
        raise ValueError(f"unknown V11 condition: {condition}")
    return [
        tool
        for tool in TOOL_DEFINITIONS
        if (condition == "native" or tool["function"]["name"] != "bash")
        and not (
            condition == "timeline_source"
            and tool["function"]["name"]
            in {"bm25_search", "embedding_search"}
        )
    ]
