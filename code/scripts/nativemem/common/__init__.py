"""Shared utilities for NativeMem experiment entrypoints."""

from . import run_config
from .io import (
    atomic_json,
    read_json,
    sha256_file,
    source_tree_sha256,
    tree_sha256,
    utc_now,
)


def stop_on_signal(_signum: object, _frame: object) -> None:
    raise KeyboardInterrupt


__all__ = [
    "atomic_json",
    "read_json",
    "run_config",
    "sha256_file",
    "source_tree_sha256",
    "stop_on_signal",
    "tree_sha256",
    "utc_now",
]
