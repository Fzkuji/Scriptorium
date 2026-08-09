"""Cross-platform resolution for the model-visible POSIX shell contract."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def is_wsl_drvfs_path(
    path: str | Path, *, mounts: Path = Path("/proc/mounts")
) -> bool:
    """Return whether *path* is on Windows DrvFs mounted inside WSL."""
    if os.name == "nt" or not mounts.is_file():
        return False
    resolved = Path(path).resolve()
    best_match: tuple[int, str, str] | None = None
    try:
        lines = mounts.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    for line in lines:
        fields = line.split()
        if len(fields) < 4:
            continue
        mount_point = Path(
            fields[1].replace(r"\040", " ").replace(r"\134", "\\")
        )
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        match = (len(mount_point.parts), fields[2], fields[3])
        if best_match is None or match[0] > best_match[0]:
            best_match = match
    if best_match is None:
        return False
    _, fs_type, options = best_match
    return fs_type == "9p" and any(
        option.startswith("aname=drvfs") for option in options.split(",")
    )


def resolve_posix_bash() -> str:
    configured = os.environ.get("SCRIPTORIUM_BASH")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
        raise RuntimeError(f"SCRIPTORIUM_BASH does not exist: {path}")
    if os.name != "nt":
        return shutil.which("bash") or "/bin/bash"
    git = shutil.which("git")
    candidates: list[Path] = []
    if git:
        git_path = Path(git).resolve()
        candidates.extend([
            git_path.parent.parent / "bin" / "bash.exe",
            git_path.parent.parent / "usr" / "bin" / "bash.exe",
        ])
    candidates.extend([
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
    ])
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(
        "Git Bash is required for POSIX tool commands on Windows; "
        "set SCRIPTORIUM_BASH to bash.exe"
    )
