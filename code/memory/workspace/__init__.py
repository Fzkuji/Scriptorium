"""The substrate the three phases sit on.

Writing, organizing and retrieval all reach the same files through the same
transaction: a staged copy of the workspace, validated against the topic
contract, installed whole or not at all. Nothing here decides what memory
should say — it decides what a change to memory has to satisfy before it
lands.
"""

from .layout import (
    LEGACY_RUNTIME_DIRS,
    RUNTIME_DIR,
    RUNTIME_DIR_NAMES,
    STATE_FILE,
    TEMPORARY_PREFIX,
    VERSION_CONTROL_DIRS,
    has_runtime_dir,
    is_internal_path,
    is_runtime_name,
    is_state_file,
    runtime_dir,
)
from .staging import MemoryWorkspace
from .transaction import (
    TransactionError,
    committed_baseline,
    install_state,
    workspace_revision,
    workspace_write_lock,
)

__all__ = [
    "LEGACY_RUNTIME_DIRS",
    "MemoryWorkspace",
    "RUNTIME_DIR",
    "RUNTIME_DIR_NAMES",
    "STATE_FILE",
    "TEMPORARY_PREFIX",
    "TransactionError",
    "VERSION_CONTROL_DIRS",
    "committed_baseline",
    "has_runtime_dir",
    "install_state",
    "is_internal_path",
    "is_runtime_name",
    "is_state_file",
    "runtime_dir",
    "workspace_revision",
    "workspace_write_lock",
]
