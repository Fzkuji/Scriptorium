"""agent-memory command line: init, validate, mcp."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from src.management.transaction import TransactionError, workspace_revision
from src.retrieval import inspect


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-memory",
        description="Model-managed Markdown files as external agent memory.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser(
        "init", help="create a memory workspace, or check an existing one"
    )
    init.add_argument("workspace", help="directory to initialize")

    validate = commands.add_parser(
        "validate", help="parse topics and rebuild derived views in a scratch copy"
    )
    validate.add_argument("--workspace", required=True)

    mcp = commands.add_parser("mcp", help="run the stdio MCP server")
    mcp.add_argument("--workspace", required=True)
    mcp.add_argument(
        "--git-commit",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "auto commits when the workspace is a git repository; "
            "on requires git; off never commits"
        ),
    )
    return parser


def command_init(workspace: str) -> int:
    root = Path(workspace).expanduser().resolve()
    existing = root.is_dir() and any(
        (root / name).exists()
        for name in ("core.md", "topics", "sources")
    )
    root.mkdir(parents=True, exist_ok=True)
    if existing:
        # Never overwrite memory that is already there.
        print(f"workspace already initialized: {root}")
        print(json.dumps(inspect.status(root), ensure_ascii=False, indent=2))
        return 0
    (root / "topics").mkdir(exist_ok=True)
    (root / "sources").mkdir(exist_ok=True)
    core = root / "core.md"
    if not core.exists():
        core.write_text("# Core\n", encoding="utf-8")
    runtime = root / ".nativemem"
    runtime.mkdir(exist_ok=True)
    runtime_json = runtime / "runtime.json"
    if not runtime_json.exists():
        runtime_json.write_text("{}\n", encoding="utf-8")
    print(f"initialized workspace: {root}")
    print(f"revision: {workspace_revision(root)}")
    return 0


def command_validate(workspace: str) -> int:
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        print(f"workspace is not a directory: {root}", file=sys.stderr)
        return 2
    scratch = Path(tempfile.mkdtemp(prefix="agent-memory-validate-"))
    try:
        # Rebuild into a scratch copy so a successful check never writes.
        copy = scratch / "memory"
        shutil.copytree(root, copy, symlinks=True)
        from src.management import MemoryWorkspace
        from src.management.transaction import committed_baseline, install_state

        space = MemoryWorkspace(copy)
        before_units, before_block_ids = committed_baseline(space)
        install_state(space, before_units, before_block_ids)
    except TransactionError as exc:
        print(json.dumps({"ok": False, "error": {
            "code": exc.code, "message": exc.message, "path": exc.path,
        }}, ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        print(json.dumps({"ok": False, "error": {
            "code": "INVALID_TOPIC_FORMAT", "message": str(exc),
        }}, ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(json.dumps(
        {"ok": True, "data": inspect.status(root)}, ensure_ascii=False, indent=2
    ))
    return 0


def command_mcp(workspace: str, git_commit: str) -> int:
    from .mcp_server import run

    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        print(
            f"workspace is not a directory: {root}\n"
            f"run: agent-memory init {root}",
            file=sys.stderr,
        )
        return 2
    run(root, git_commit=git_commit)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        return command_init(args.workspace)
    if args.command == "validate":
        return command_validate(args.workspace)
    return command_mcp(args.workspace, args.git_commit)


if __name__ == "__main__":
    raise SystemExit(main())
