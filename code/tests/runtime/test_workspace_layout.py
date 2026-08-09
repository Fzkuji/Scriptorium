"""The runtime area's names, current and inherited."""

import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))

from memory.management.transaction import workspace_revision  # noqa: E402
from memory.retrieval import inspect  # noqa: E402
from memory.runtime.state import RuntimeStateStore  # noqa: E402
from memory.workspace_layout import (  # noqa: E402
    LEGACY_RUNTIME_DIRS, RUNTIME_DIR, is_internal_path, is_runtime_name,
    is_state_file, runtime_dir,
)

LEGACY = LEGACY_RUNTIME_DIRS[0]


def make_workspace(root: Path, runtime_name: str) -> Path:
    (root / "topics").mkdir(parents=True, exist_ok=True)
    (root / "sources").mkdir(parents=True, exist_ok=True)
    (root / "core.md").write_text("# Core\n", encoding="utf-8")
    runtime = root / runtime_name
    runtime.mkdir(exist_ok=True)
    (runtime / "runtime.json").write_text("{}\n", encoding="utf-8")
    return root


def test_a_new_workspace_uses_the_current_name(tmp_path: Path):
    assert runtime_dir(tmp_path) == tmp_path / RUNTIME_DIR


def test_a_workspace_keeps_the_runtime_name_it_already_has(tmp_path: Path):
    make_workspace(tmp_path, LEGACY)

    assert runtime_dir(tmp_path) == tmp_path / LEGACY
    # Writing must land in the directory that is there, not beside it.
    store = RuntimeStateStore(tmp_path)
    state = store.load()
    store.save(state)
    assert not (tmp_path / RUNTIME_DIR).exists()
    assert json.loads((tmp_path / LEGACY / "runtime.json").read_text())


def test_both_names_are_hidden_from_listings(tmp_path: Path):
    for name in (RUNTIME_DIR, LEGACY):
        root = make_workspace(tmp_path / name.lstrip("."), name)
        (root / f"{name}-bm25.json").write_text("{}", encoding="utf-8")

        listed = {entry["path"] for entry in inspect.list_files(root)["files"]}

        assert listed == {"core.md"}


def test_the_cursor_file_still_counts_toward_the_revision(tmp_path: Path):
    root = make_workspace(tmp_path, RUNTIME_DIR)
    before = workspace_revision(root)

    (root / RUNTIME_DIR / "runtime.json").write_text(
        '{"cursors": {"t": 1}}\n', encoding="utf-8"
    )
    moved = workspace_revision(root)

    # A cache written by a read must not look like a write.
    (root / f"{RUNTIME_DIR}-bm25.json").write_text("{}", encoding="utf-8")

    assert moved != before
    assert workspace_revision(root) == moved


def test_runtime_names_cover_the_directory_and_what_it_stages(tmp_path: Path):
    assert is_runtime_name(RUNTIME_DIR)
    assert is_runtime_name(f"{RUNTIME_DIR}-block-backup")
    assert is_runtime_name(f"{LEGACY}-bm25.json")
    assert not is_runtime_name("topics")
    assert not is_runtime_name(".scriptoriumish")

    assert is_internal_path(Path(RUNTIME_DIR) / "runtime.json")
    assert is_internal_path(Path(f"{RUNTIME_DIR}-bm25.json"))
    assert is_internal_path(Path(".git") / "HEAD")
    assert not is_internal_path(Path("topics") / "api.md")
    # A file inside topics/ is authored memory whatever it is named.
    assert not is_internal_path(Path("topics") / f"{RUNTIME_DIR}-note.md")

    assert is_state_file(Path(RUNTIME_DIR) / "runtime.json")
    assert not is_state_file(
        Path(f"{RUNTIME_DIR}-block-backup") / RUNTIME_DIR / "runtime.json"
    )
