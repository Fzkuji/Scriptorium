"""scriptorium command line: init, validate, ingest, mcp."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from memory.workspace.transaction import TransactionError, workspace_revision
from memory.workspace.layout import RUNTIME_DIR, has_runtime_dir
from memory.retrieval import inspect


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

    ingest = commands.add_parser(
        "ingest",
        help="write a session transcript into memory once it is long enough",
    )
    ingest.add_argument("--transcript", required=True)
    ingest.add_argument("--workspace", default="~/.scriptorium/memory")
    ingest.add_argument(
        "--token-threshold",
        type=int,
        default=16_000,
        help="hold new turns until this many tokens have accumulated",
    )
    ingest.add_argument(
        "--model", help="defaults to whatever model your own CLI uses"
    )
    ingest.add_argument("--claude-cli")
    ingest.add_argument(
        "--force",
        action="store_true",
        help="write what has accumulated without waiting for the threshold",
    )

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
    existing = root.is_dir() and (
        any((root / name).exists() for name in ("core.md", "topics", "sources"))
        or has_runtime_dir(root)
    )
    root.mkdir(parents=True, exist_ok=True)
    if existing:
        return False
    (root / "topics").mkdir(exist_ok=True)
    (root / "sources").mkdir(exist_ok=True)
    core = root / "core.md"
    if not core.exists():
        core.write_text("# Core\n", encoding="utf-8")
    runtime = root / RUNTIME_DIR
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


def _first_batch(
    records: list[Any], token_counter: Any, threshold: int
) -> list[Any]:
    """The leading records that together reach the threshold.

    Includes the record that crosses it, so the batch is always at or above
    the threshold and the runtime's own check agrees it is time to write. A
    backlog shorter than the threshold comes back whole and is held.
    """
    total = 0
    for index, record in enumerate(records):
        total += token_counter(record.content)
        if total >= threshold:
            return records[:index + 1]
    return records


def command_ingest(
    transcript: str,
    workspace: str,
    *,
    token_threshold: int,
    model: str | None = None,
    cli_path: str | None = None,
    force: bool = False,
) -> int:
    """Write a session's new turns into memory, once there are enough.

    Called after every assistant turn, so the common outcome is to do
    nothing: the cursor says what has already been written and the
    threshold holds the rest back until a batch is worth an agent call.
    """
    # Imported here rather than at module scope: the SDK is a heavy import
    # and the other subcommands never need it.
    from memory.agent_runtime import ClaudeCodeAgent, ClaudeCodeConfig
    from memory.ingestion import read_transcript
    from memory.management import MemoryWorkspace, organize_topics
    from memory.management.agent import _run_agent
    from memory.writing.session import render_writer_task
    from memory.workspace.transaction import workspace_write_lock
    from memory.runtime.online import OnlineMemoryRuntime
    from memory.runtime.tokenization import TokenCounter

    root = Path(workspace).expanduser().resolve()
    ensure_workspace(root, self_ignore=True)
    try:
        records = read_transcript(transcript)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not records:
        print("nothing to ingest")
        return 0

    agent = ClaudeCodeAgent(
        ClaudeCodeConfig.inherited(model=model, cli_path=cli_path)
    )

    def writer(space: Any, batch: tuple[Any, ...]) -> None:
        observed = next(
            (
                record.timestamp[:10]
                for record in reversed(batch) if record.timestamp
            ),
            "undated",
        )
        _run_agent(
            space.memory_dir,
            agent=agent,
            task=render_writer_task([{
                "observation_date": observed,
                "turns": [
                    (record.role, record.content) for record in batch
                ],
                "refs": [record.source_id for record in batch],
            }]),
            stage="write",
        )

    def organizer(space: Any) -> None:
        organize_topics(space.memory_dir, agent=agent)

    runtime = OnlineMemoryRuntime(
        root,
        token_counter=TokenCounter.resolve(
            requested_model=model or "claude"
        ).count,
        token_threshold=token_threshold,
    )
    pending = runtime.pending(records)
    if not pending:
        print("nothing new in this transcript")
        return 0
    # The threshold is how much is worth writing, not how much to write at
    # once. A session that has been running all day arrives with hundreds of
    # thousands of tokens of backlog, and handing that to one agent call
    # would exceed its context. Send one batch and let the next turn's hook
    # take the rest.
    pending = _first_batch(pending, runtime.token_counter, token_threshold)

    try:
        # Short wait on purpose. Another session writing right now is the
        # normal case, not a failure, and this runs again after every turn.
        with workspace_write_lock(root, timeout_s=1.0):
            wrote = runtime.process(
                pending,
                writer,
                local_manager=organizer,
                global_manager=organizer,
                force=force,
            )
    except TransactionError as exc:
        if exc.code == "CONCURRENT_UPDATE":
            print("skipped: workspace busy")
            return 0
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 1

    if not wrote:
        print(f"holding {len(pending)} turn(s) below the threshold")
        return 0
    print(f"wrote {len(pending)} turn(s)")
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
        from memory.management import MemoryWorkspace
        from memory.workspace.transaction import committed_baseline, install_state

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
    a value whose text before `=` is not one is read as a path.
    """
    layered = len(values) > 1
    workspaces: list[tuple[str, Path]] = []
    for value in values:
        name, sep, path = value.partition("=")
        if sep and LAYER_NAME.fullmatch(name):
            if not path.strip():
                # An unset variable expands to nothing. Left alone, the layer
                # would resolve to the project root and be initialized there.
                raise ValueError(f"workspace {name} has an empty path: {value!r}")
            workspaces.append((name, Path(path).expanduser()))
        elif layered:
            raise ValueError(
                f"with several workspaces each needs a name: NAME=PATH, got {value!r}"
            )
        elif value.strip():
            workspaces.append(("memory", Path(value).expanduser()))
        else:
            raise ValueError("workspace path is empty")
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
    run(layers, git_commit=git_commit)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        return command_init(args.workspace)
    if args.command == "validate":
        return command_validate(args.workspace)
    if args.command == "ingest":
        return command_ingest(
            args.transcript,
            args.workspace,
            token_threshold=args.token_threshold,
            model=args.model,
            cli_path=args.claude_cli,
            force=args.force,
        )
    return command_mcp(list(args.workspace), args.git_commit)


if __name__ == "__main__":
    raise SystemExit(main())
