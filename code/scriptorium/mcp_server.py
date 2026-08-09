"""stdio MCP server exposing memory workspaces.

Adapter only: schema conversion, boundary checks and stable JSON. Every
capability comes from the same core modules the experiment path uses. No
shell, no subprocess, no second model.

One workspace is the common case and behaves exactly as it always has. Given
more than one, they are read as a single layered memory: reads span every
layer, each path is qualified with its layer (`global:topics/api.md`), and a
write lands in the layer named by its `layer` argument — the first workspace
by default. What you learn about the person belongs in the wide layer; what
this project decided belongs in its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memory.management import MemoryWorkspace
from memory.management.transaction import TransactionError
from memory.retrieval.layers import Layer, LayeredMemory

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


def _as_layers(workspaces: Path | list[tuple[str, Path]]) -> list[Layer]:
    if isinstance(workspaces, (str, Path)):
        workspaces = [("memory", Path(workspaces))]
    layers = []
    for name, path in workspaces:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(
                f"workspace is not a directory: {root} "
                f"(run 'scriptorium init {root}' first)"
            )
        layers.append(Layer(name=name, root=root))
    return layers


def build_server(
    workspaces: Path | list[tuple[str, Path]], *, git_commit: str = "auto"
) -> Any:
    from mcp.server.fastmcp import FastMCP

    memory = LayeredMemory(_as_layers(workspaces))
    server = FastMCP("scriptorium")

    if memory.single:
        layer_note = ""
        update_layer_note = ""
    else:
        names = ", ".join(layer.name for layer in memory.layers)
        layer_note = (
            f" Memory has layers ({names}); paths are prefixed with their "
            f"layer, like '{memory.layers[-1].name}:topics/x.md'."
        )
        update_layer_note = (
            f" layer chooses which workspace receives the write ({names}; "
            f"default {memory.default.name}). Facts about the person belong "
            f"in the widest layer; facts about this project in its own."
        )

    @server.tool(
        name="memory_status",
        description=(
            "Report memory workspace counts and the current revision. "
            "Call this first: memory_update requires the revision." + layer_note
        ),
    )
    def memory_status() -> str:
        def run():
            data = memory.status()
            return _ok(data, data["revision"])
        return _guard(run)

    @server.tool(
        name="memory_list",
        description=(
            "List memory files. Use prefix='topics/' for authored memory; "
            "timeline/, recent_events.jsonl and relations.json are derived."
            + layer_note
        ),
    )
    def memory_list(
        prefix: str = "",
        include_derived: bool = True,
        limit: int = 200,
    ) -> str:
        def run():
            data = memory.list_files(
                prefix=prefix,
                include_derived=include_derived,
                limit=limit,
            )
            return _ok(data, memory.revision())
        return _guard(run)

    @server.tool(
        name="memory_read",
        description=(
            "Read one memory file. Use at most one of heading, block_id, or "
            "offset/limit. block_id returns the block with the footnotes it "
            "cites. sources/** is readable but never writable." + layer_note
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
            data = memory.read_file(
                path,
                heading=heading,
                block_id=block_id,
                offset=offset,
                limit=limit,
            )
            return _ok(data, memory.revision())
        return _guard(run)

    @server.tool(
        name="memory_grep",
        description=(
            "Literal or regular-expression search over memory text. "
            "Use for exact strings; use memory_search for meaning." + layer_note
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
            data = memory.grep(
                query,
                prefix=prefix,
                case_sensitive=case_sensitive,
                literal=literal,
                limit=limit,
            )
            return _ok(data, memory.revision())
        return _guard(run)

    @server.tool(
        name="memory_search",
        description=(
            "Ranked retrieval over memory blocks and sources. method is "
            "'bm25' or 'embedding'; embedding returns EMBEDDING_UNAVAILABLE "
            "when no backend is installed rather than silently using bm25."
            + layer_note
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
            data = memory.search(
                query,
                method=method,
                top_k=top_k,
                path_prefix=path_prefix,
                date_from=date_from,
                date_to=date_to,
            )
            return _ok(data, memory.revision())
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
            + update_layer_note
        ),
    )
    def memory_update(
        base_revision: str,
        patch: str,
        sources: list[dict[str, Any]] | None = None,
        commit_message: str | None = None,
        layer: str | None = None,
    ) -> str:
        def run():
            target = memory.resolve(layer) if layer else memory.default
            workspace = MemoryWorkspace(target.root)
            result = workspace.update(
                base_revision=memory.layer_revision(base_revision, target),
                patch=patch,
                sources=sources,
                commit_message=commit_message,
                git_commit=git_commit,
            )
            return _ok(
                {
                    "layer": target.name,
                    "source_ids": result.source_ids,
                    "block_ids": result.block_ids,
                    "evidence_ids": result.evidence_ids,
                    "changed_files": [
                        memory.qualify(target, changed)
                        for changed in result.changed_files
                    ],
                    "memory_committed": result.memory_committed,
                    "git_committed": result.git_committed,
                    "git_commit": result.git_commit,
                },
                memory.revision(),
            )
        return _guard(run)

    return server


def run(
    workspaces: Path | list[tuple[str, Path]], *, git_commit: str = "auto"
) -> None:
    build_server(workspaces, git_commit=git_commit).run(transport="stdio")
