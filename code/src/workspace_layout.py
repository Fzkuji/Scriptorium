"""The names the runtime owns inside a memory workspace.

A workspace holds the memory — `core.md`, `topics/`, `sources/` and the
derived views — and beside it a small runtime area: cursors, the write lock,
staged backups, retrieval caches. The runtime area is not memory. It is hidden
from listings, left out of the revision, and never writable by a patch.

Workspaces built before the project took its current name carry the runtime
directory under its former name. They keep it: a stored run's hash covers
every byte of its workspace, so renaming a directory inside one would
invalidate the record it was published with. Anything opened for writing uses
whichever name that workspace already has, and a new workspace gets the
current one.
"""

from __future__ import annotations

from pathlib import Path

RUNTIME_DIR = ".scriptorium"
LEGACY_RUNTIME_DIRS = (".nativemem",)
RUNTIME_DIR_NAMES = (RUNTIME_DIR, *LEGACY_RUNTIME_DIRS)

TEMPORARY_PREFIX = "scriptorium-"


def is_runtime_name(name: str) -> bool:
    """True for the runtime directory and anything it stages beside itself."""
    return any(
        name == known or name.startswith(f"{known}-")
        for known in RUNTIME_DIR_NAMES
    )


def is_runtime_path(relative: Path) -> bool:
    """True for a workspace-relative path that belongs to the runtime."""
    parts = relative.parts
    if not parts:
        return False
    return is_runtime_name(parts[0]) or is_runtime_name(relative.name)


def runtime_dir(memory_dir: Path | str) -> Path:
    """This workspace's runtime directory, keeping the name it already has."""
    root = Path(memory_dir)
    for legacy in LEGACY_RUNTIME_DIRS:
        if (root / legacy).is_dir():
            return root / legacy
    return root / RUNTIME_DIR
