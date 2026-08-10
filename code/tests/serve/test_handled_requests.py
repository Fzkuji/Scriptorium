"""A chunk that arrives twice is written once.

The platform resends a chunk whenever it did not hear the answer to the
first attempt. Reading it again costs a second model pass and files the same
facts beside the ones already there, and a retry as slow as the call it
retries misses the same deadline again.
"""
from __future__ import annotations

from pathlib import Path

from memory.workspace import MemoryWorkspace
from memory.workspace.layout import is_internal_path, runtime_dir
from scriptorium_serve.server import (
    _already_written,
    _handled_requests,
    _mark_written,
)


def test_a_request_is_unknown_until_it_is_recorded(tmp_path: Path) -> None:
    assert not _already_written(tmp_path, "req-1")

    _mark_written(tmp_path, "req-1")

    assert _already_written(tmp_path, "req-1")
    assert not _already_written(tmp_path, "req-2"), "only the one recorded"


def test_ids_sharing_a_prefix_stay_apart(tmp_path: Path) -> None:
    _mark_written(tmp_path, "req-1")

    assert not _already_written(tmp_path, "req-12")
    assert not _already_written(tmp_path, "req")


def test_the_record_is_not_part_of_the_memory(tmp_path: Path) -> None:
    """It must not make a stage look dirty, or every commit installs nothing.

    ``stage_is_dirty`` compares the workspace against the stage, and the
    stage is built from the memory directories only. A file counted as memory
    but never staged would report a difference on every single write.
    """
    workspace = MemoryWorkspace(tmp_path, config=None)
    workspace._refresh_stage()
    assert not workspace.stage_is_dirty()

    _mark_written(tmp_path, "req-1")

    assert is_internal_path(_handled_requests(tmp_path).relative_to(tmp_path))
    assert _handled_requests(tmp_path).parent == runtime_dir(tmp_path)
    assert not workspace.stage_is_dirty(), "recording a request is not an edit"
