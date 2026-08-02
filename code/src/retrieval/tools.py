"""Dispatch NativeMem retrieval tool calls."""

from pathlib import Path
from typing import Any

from .bm25 import MemoryBM25Index
from .bm25 import render_search_results as render_bm25_results
from .embedding import MemoryEmbeddingIndex
from .embedding import render_search_results as render_embedding_results

from .shell import normalize_workspace_command, validate_read_only_command
from .views import read_memory_file


def execute_tool_call(
    runtime: Any,
    name: str,
    args: dict[str, Any],
    *,
    memory_dir: Path,
    files: list[Path],
    condition: str,
    include_recent: bool,
    indexes: dict[str, Any],
) -> tuple[str, bool, bool | None]:
    if name == "bash":
        command = normalize_workspace_command(
            args.get("command", ""), memory_dir
        )
        accepted, reason = validate_read_only_command(command)
        if not accepted:
            return f"Command rejected: {reason}", False, False
        output = runtime.execute_tool(
            "bash",
            {**args, "command": command},
            str(memory_dir),
            hide_raw=True,
        )
        return output, True, True
    if name == "list_memory_files":
        prefix = str(args.get("prefix", ""))
        return "\n".join(
            path.relative_to(memory_dir).as_posix()
            for path in files
            if path.relative_to(memory_dir).as_posix().startswith(prefix)
        ), True, None
    if name == "read_memory_file":
        return read_memory_file(
            memory_dir,
            args.get("path"),
            condition,
            include_recent=include_recent,
        ), True, None
    if name in {"bm25_search", "embedding_search"}:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("search query is empty")
        top_k = max(1, min(int(args.get("top_k", 10)), 10))
        if name == "bm25_search":
            if "bm25" not in indexes:
                indexes["bm25"] = MemoryBM25Index(
                    memory_dir, persist=False
                )
            results = indexes["bm25"].search(
                query,
                top_k=top_k,
                path_prefix=args.get("path_prefix") or None,
                date_from=args.get("date_from") or None,
                date_to=args.get("date_to") or None,
            )
            return render_bm25_results(results), True, None
        if "embedding" not in indexes:
            indexes["embedding"] = MemoryEmbeddingIndex(memory_dir)
        results = indexes["embedding"].search(
            query,
            top_k=top_k,
            date_from=args.get("date_from") or None,
            date_to=args.get("date_to") or None,
        )
        return render_embedding_results(results), True, None
    raise ValueError(f"unknown tool: {name}")
