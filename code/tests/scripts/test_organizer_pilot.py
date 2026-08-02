from collections import Counter
from pathlib import Path

import pytest

from scripts.nativemem.organizer_pilot import (
    OrganizerWorkspace,
    snapshot_topics,
    validate_topics,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_workspace_moves_and_merges_without_changing_entries(tmp_path):
    _write(tmp_path / "topics/London/travel.md", "## Trip\n[2023-01-01] London visit [D1:1]\n")
    _write(tmp_path / "topics/UK/London.md", "[2023-02-01] London hotel [D2:2]\n")
    before = snapshot_topics(tmp_path)
    ws = OrganizerWorkspace(tmp_path)

    ws.move_entries("London/travel.md", "places/London.md", ["[2023-01-01] London visit [D1:1]"])
    ws.merge_files("places/London.md", ["UK/London.md"])

    report = validate_topics(before, tmp_path)
    assert report["valid"] is True
    assert report["entries"] == Counter({
        "[2023-01-01] London visit [D1:1]": 1,
        "[2023-02-01] London hotel [D2:2]": 1,
    })
    assert not (tmp_path / "topics/UK/London.md").exists()


def test_workspace_rejects_escape_and_validation_detects_loss(tmp_path):
    _write(tmp_path / "topics/a.md", "[2023-01-01] Fact [D1:1]\n")
    before = snapshot_topics(tmp_path)
    ws = OrganizerWorkspace(tmp_path)
    with pytest.raises(ValueError, match="topics"):
        ws.read_file("../outside.md")

    (tmp_path / "topics/a.md").unlink()
    report = validate_topics(before, tmp_path)
    assert report["valid"] is False
    assert report["missing_entries"] == {"[2023-01-01] Fact [D1:1]": 1}
