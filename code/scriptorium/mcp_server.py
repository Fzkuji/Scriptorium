"""stdio MCP server exposing one memory workspace.

Adapter only: schema conversion, boundary checks and stable JSON. Every
capability comes from the same core modules the experiment path uses. No
shell, no subprocess, no second model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.management import MemoryWorkspace
from src.management.transaction import TransactionError
from src.retrieval import inspect

TOOL_NAMES = (
    "memory_status",
    "memory_list",
    "memory_read",
    "memory_grep",
    "memory_search",
    "memory_update",
)


def _ok(data: dict[str, Any], revision: str) -> str:
    return json.dumps(
        {"ok": True, "data": data, "revision": revision},
        ensure_ascii=False,
    )


def _error(exc: TransactionError) -> str:
    payload: dict[str, Any] = {"code": exc.code, "message": exc.message}
    if exc.path:
        payload["path"] = exc.path
    if exc.details:
        payload["details"] = exc.details
    return json.dumps({"ok": False, "error": payload}, ensure_ascii=False)


def _guard(call):
    """Return a stable JSON envelope instead of a traceback or a stage path."""
    try:
        return call()
    except TransactionError as exc:
        return _error(exc)
    except Exception as exc:  # noqa: BLE001 - boundary of the tool surface
        return _error(TransactionError("INTERNAL_ERROR", str(exc)))


def build_server(workspace_dir: Path, *, git_commit: str = "auto") -> Any:
    from mcp.server.fastmcp import FastMCP

    root = Path(workspace_dir).resolve()
    if not root.is_dir():
        raise ValueError(
            f"workspace is not a directory: {root} (run 'scriptorium init' first)"
        )
    server = FastMCP("scriptorium")

    @server.tool(
        name="memory_status",
        description=(
            "Report memory workspace counts and the current revision. "
            "Call this first: memory_update requires the revision."
        ),
    )
    def memory_status() -> str:
        def run():
            data = inspect.status(root)
            return _ok(data, data["revision"])
        return _guard(run)

    @server.tool(
        name="memory_list",
        description=(
            "List memory files. Use prefix='topics/' for authored memory; "
            "timeline/, recent_events.jsonl and relations.json are derived."
        ),
    )
    def memory_list(
        prefix: str = "",
        include_derived: bool = True,
        limit: int = 200,
    ) -> str:
        def run():
            data = inspect.list_files(
                root,
                prefix=prefix,
                include_derived=include_derived,
                limit=limit,
            )
            return _ok(data, inspect.status(root)["revision"])
        return _guard(run)

    @server.tool(
        name="memory_read",
        description=(
            "Read one memory file. Use at most one of heading, block_id, or "
            "offset/limit. block_id returns the block with the footnotes it "
            "cites. sources/** is readable but never writable."
        ),
    )
    def memory_read(
        path: str,
        heading: str | None = None,
        block_id: str | None = None,
        offset: int = 1,
        limit: int | None = None,
    ) -> str:
        def run():
            data = inspect.read_file(
                root,
                path,
                heading=heading,
                block_id=block_id,
                offset=offset,
                limit=limit,
            )
            return _ok(data, inspect.status(root)["revision"])
        return _guard(run)

    @server.tool(
        name="memory_grep",
        description=(
            "Literal or regular-expression search over memory text. "
            "Use for exact strings; use memory_search for meaning."
        ),
    )
    def memory_grep(
        query: str,
        prefix: str = "",
        case_sensitive: bool = False,
        literal: bool = True,
        limit: int = 50,
    ) -> str:
        def run():
            data = inspect.grep(
                root,
                query,
                prefix=prefix,
                case_sensitive=case_sensitive,
                literal=literal,
                limit=limit,
            )
            return _ok(data, inspect.status(root)["revision"])
        return _guard(run)

    @server.tool(
        name="memory_search",
        description=(
            "Ranked retrieval over memory blocks and sources. method is "
            "'bm25' or 'embedding'; embedding returns EMBEDDING_UNAVAILABLE "
            "when no backend is installed rather than silently using bm25."
        ),
    )
    def memory_search(
        query: str,
        method: str = "bm25",
        top_k: int = 8,
        path_prefix: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> str:
        def run():
            data = inspect.search(
                root,
                query,
                method=method,
                top_k=top_k,
                path_prefix=path_prefix,
                date_from=date_from,
                date_to=date_to,
            )
            return _ok(data, inspect.status(root)["revision"])
        return _guard(run)

    @server.tool(
        name="memory_update",
        description=(
            "Commit new evidence and a topic edit as one transaction. "
            "base_revision comes from memory_status or any read. patch is a "
            "unified diff touching only topics/**/*.md and core.md. Cite new "
            "sources by their transaction-local label (new-source-<name>) in "
            "the footnote Sources: field; new blocks use ^new-block-<name>. "
            "The runtime assigns stable IDs and rebuilds derived views."
        ),
    )
    def memory_update(
        base_revision: str,
        patch: str,
        sources: list[dict[str, Any]] | None = None,
        commit_message: str | None = None,
    ) -> str:
        def run():
            workspace = MemoryWorkspace(root)
            result = workspace.update(
                base_revision=base_revision,
                patch=patch,
                sources=sources,
                commit_message=commit_message,
                git_commit=git_commit,
            )
            return _ok(
                {
                    "source_ids": result.source_ids,
                    "block_ids": result.block_ids,
                    "evidence_ids": result.evidence_ids,
                    "changed_files": result.changed_files,
                    "memory_committed": result.memory_committed,
                    "git_committed": result.git_committed,
                    "git_commit": result.git_commit,
                },
                result.revision,
            )
        return _guard(run)

    return server


def run(workspace_dir: Path, *, git_commit: str = "auto") -> None:
    build_server(workspace_dir, git_commit=git_commit).run(transport="stdio")
