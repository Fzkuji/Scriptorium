"""Resolve host-specific execution details without changing experiment semantics."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .shell_backend import is_wsl_drvfs_path


PLATFORM_PROFILES = ("auto", "wsl", "macos")


def detect_platform(
    *,
    os_name: str | None = None,
    sys_platform: str | None = None,
    proc_version: Path = Path("/proc/version"),
) -> str:
    """Return a stable host label used for audit and profile validation."""
    os_name = os.name if os_name is None else os_name
    sys_platform = sys.platform if sys_platform is None else sys_platform
    if os_name == "nt":
        return "windows"
    if sys_platform == "darwin":
        return "macos"
    if sys_platform.startswith("linux"):
        try:
            version = proc_version.read_text(
                encoding="utf-8", errors="replace"
            ).lower()
        except OSError:
            version = ""
        return "wsl" if "microsoft" in version else "linux"
    return sys_platform or os_name


@dataclass(frozen=True)
class ExecutionProfile:
    requested: str
    detected: str
    shell_backend: str
    output_storage: str

    def to_manifest(self) -> dict[str, str]:
        return asdict(self)


def _classify_output_storage(path: Path, *, detected: str) -> str:
    if detected == "wsl":
        return "wsl-drvfs" if is_wsl_drvfs_path(path) else "wsl-linux"
    if detected == "macos":
        # Python's standard library does not expose the APFS mount type
        # portably. Keep the host label auditable without guessing it.
        return "macos-local-or-mounted"
    if detected == "windows":
        return "windows"
    return "posix"


def resolve_execution_profile(
    requested: str,
    *,
    shell_backend: str,
    output_dir: Path,
    detected: str | None = None,
) -> ExecutionProfile:
    """Validate a platform profile and resolve its non-semantic defaults."""
    if requested not in PLATFORM_PROFILES:
        raise ValueError(
            "platform_profile must be auto, wsl, or macos"
        )
    detected = detect_platform() if detected is None else detected
    if requested != "auto" and requested != detected:
        raise ValueError(
            f"platform profile {requested!r} does not match host {detected!r}"
        )
    resolved_shell = shell_backend
    if resolved_shell == "auto" and detected in {"wsl", "macos"}:
        resolved_shell = "posix-bash"
    return ExecutionProfile(
        requested=requested,
        detected=detected,
        shell_backend=resolved_shell,
        output_storage=_classify_output_storage(output_dir, detected=detected),
    )


__all__ = [
    "ExecutionProfile",
    "PLATFORM_PROFILES",
    "detect_platform",
    "resolve_execution_profile",
]
