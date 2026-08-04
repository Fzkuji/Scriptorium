"""scriptorium command line: init, validate, mcp."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from src.management.transaction import TransactionError, workspace_revision
from src.retrieval import inspect


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scriptorium",
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
    mcp.add_argument(
        "--workspace",
        action="append",
        required=True,
        metavar="[NAME=]PATH",
        help=(
            "memory workspace; repeat for layers. With several, each needs a "
            "name and the first receives writes by default: "
            "--workspace project=.memory --workspace global=~/memory"
        ),
    )
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


def ensure_workspace(root: Path, *, self_ignore: bool = False) -> bool:
    """Create the workspace skeleton if it is not there. True when created.

    Never overwrites memory that already exists. With self_ignore, an
    auto-created workspace also carries a `.gitignore` of `*` (the venv
    convention), so personal memory landing inside a repository is never
    accidentally committed to it.
    """
    existing = root.is_dir() and any(
        (root / name).exists()
        for name in ("core.md", "topics", "sources")
    )
    root.mkdir(parents=True, exist_ok=True)
    if existing:
        return False
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
    if self_ignore:
        ignore = root / ".gitignore"
        if not ignore.exists():
            ignore.write_text("*\n", encoding="utf-8")
    return True


def project_root(start: Path) -> Path:
    """Nearest ancestor holding `.git`, else start itself.

    MCP servers inherit the directory the session opened in, which may be a
    subdirectory of the repository; one repository must map to one project
    memory no matter where inside it the session started.
    """
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def command_init(workspace: str) -> int:
    root = Path(workspace).expanduser().resolve()
    if not ensure_workspace(root):
        print(f"workspace already initialized: {root}")
        print(json.dumps(inspect.status(root), ensure_ascii=False, indent=2))
        return 0
    print(f"initialized workspace: {root}")
    print(f"revision: {workspace_revision(root)}")
    return 0


def command_validate(workspace: str) -> int:
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        print(f"workspace is not a directory: {root}", file=sys.stderr)
        return 2
    scratch = Path(tempfile.mkdtemp(prefix="scriptorium-validate-"))
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


LAYER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def parse_workspaces(values: list[str]) -> list[tuple[str, Path]]:
    """`NAME=PATH` entries into named layers; a lone bare path stays plain.

    Order matters: the first workspace receives writes that name no layer,
    so put the narrowest (the project) first. A name is a plain identifier;
    anything else is read as a path, so a filename containing `=` still
    works in the single-workspace form.
    """
    layered = len(values) > 1
    workspaces: list[tuple[str, Path]] = []
    for value in values:
        name, sep, path = value.partition("=")
        if sep and LAYER_NAME.fullmatch(name):
            workspaces.append((name, Path(path).expanduser()))
        elif layered:
            raise ValueError(
                f"with several workspaces each needs a name: NAME=PATH, got {value!r}"
            )
        else:
            workspaces.append(("memory", Path(value).expanduser()))
    return workspaces


def prepare_layers(
    workspace_values: list[str], cwd: Path
) -> list[tuple[str, Path]]:
    """Resolve, and create when missing, every workspace for the server.

    A relative path resolves against the project root of `cwd`, so the first
    session opened anywhere in a repository brings that repository's memory
    into being. A workspace that cannot be created is dropped with a note —
    an unwritable volume costs that layer, not the session.
    """
    base = project_root(cwd)
    layers: list[tuple[str, Path]] = []
    for name, path in parse_workspaces(workspace_values):
        root = (path if path.is_absolute() else base / path).resolve()
        try:
            if ensure_workspace(root, self_ignore=True):
                print(f"initialized {name} workspace: {root}", file=sys.stderr)
        except OSError as exc:
            print(f"skipping {name} workspace: {exc}", file=sys.stderr)
            continue
        layers.append((name, root))
    return layers


def command_mcp(workspace_values: list[str], git_commit: str) -> int:
    from .mcp_server import run

    try:
        layers = prepare_layers(workspace_values, Path.cwd())
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not layers:
        print("no workspace could be prepared", file=sys.stderr)
        return 2
    if len(layers) == 1:
        run(layers[0][1], git_commit=git_commit)
    else:
        run(layers, git_commit=git_commit)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        return command_init(args.workspace)
    if args.command == "validate":
        return command_validate(args.workspace)
    return command_mcp(list(args.workspace), args.git_commit)


if __name__ == "__main__":
    raise SystemExit(main())
