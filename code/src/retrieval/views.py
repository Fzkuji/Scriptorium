"""Visible memory files and tool availability for each retrieval condition."""

from pathlib import Path
from typing import Any

from .schemas import (
    CONDITION_VIEWS,
    TOOL_DEFINITIONS,
    normalize_memory_components,
)


def memory_files(
    memory_dir: Path,
    condition: str = "native",
    *,
    include_recent: bool = True,
    components: tuple[str, ...] | str | None = None,
) -> list[Path]:
    root = memory_dir.resolve()
    views = CONDITION_VIEWS.get(condition)
    if condition != "native" and views is None:
        raise ValueError(f"unknown Scriptorium condition: {condition}")
    selected = normalize_memory_components(components)
    if selected is not None and condition != "native":
        raise ValueError(
            "explicit memory components require condition='native'"
        )
    selected_set = set(selected or ())

    def visible_markdown(path: Path) -> bool:
        relative = path.relative_to(root)
        if selected is not None:
            if relative.as_posix() == "core.md":
                return "core" in selected_set
            return bool(relative.parts and relative.parts[0] in selected_set)
        return (
            views is None
            or relative.parts[0] in views
            or relative.as_posix() == "core.md"
        )

    result = [
        path
        for path in root.rglob("*.md")
        if path.is_file()
        and not path.is_symlink()
        and visible_markdown(path)
    ]
    recent = root / "recent_events.jsonl"
    if (
        recent.is_file()
        and not recent.is_symlink()
        and include_recent
        and (
            (selected is not None and "recent" in selected_set)
            or (selected is None and (views is None or "recent" in views))
        )
    ):
        result.append(recent)
    relations = root / "relations.json"
    if (
        selected is not None
        and "relations" in selected_set
        and relations.is_file()
        and not relations.is_symlink()
    ):
        result.append(relations)
    return sorted(
        result, key=lambda path: path.relative_to(root).as_posix()
    )


def read_memory_file(
    memory_dir: Path,
    raw_path: object,
    condition: str = "native",
    *,
    include_recent: bool = True,
    components: tuple[str, ...] | str | None = None,
    offset: object = 1,
    limit: object | None = None,
) -> str:
    root = memory_dir.resolve()
    relative = Path(str(raw_path or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("memory path escapes workspace")
    path = (root / relative).resolve()
    path.relative_to(root)
    allowed = path in memory_files(
        root,
        condition,
        include_recent=include_recent,
        components=components,
    )
    if path.is_symlink() or not path.is_file() or not allowed:
        raise ValueError("memory path is not an allowed memory file")
    start = int(offset) - 1
    count = None if limit is None else int(limit)
    if start < 0 or (count is not None and count < 1):
        raise ValueError("offset and limit must be positive")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(lines[start:] if count is None else lines[start:start + count])


def tools_for(
    condition: str,
    *,
    components: tuple[str, ...] | str | None = None,
) -> list[dict[str, Any]]:
    if condition != "native" and condition not in CONDITION_VIEWS:
        raise ValueError(f"unknown Scriptorium condition: {condition}")
    selected = normalize_memory_components(components)
    if selected is not None and condition != "native":
        raise ValueError(
            "explicit memory components require condition='native'"
        )
    searchable = selected is None or bool({"topics", "sources"} & set(selected))
    return [
        tool
        for tool in TOOL_DEFINITIONS
        if (
            (condition == "native" and selected is None)
            or tool["function"]["name"] != "bash"
        )
        and not (
            (condition == "timeline_source" or not searchable)
            and tool["function"]["name"]
            in {"bm25_search", "embedding_search"}
        )
    ]
