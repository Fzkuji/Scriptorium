import importlib.util
import os
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts/runners/run_longmemeval.py"
SPEC = importlib.util.spec_from_file_location("run_longmemeval_script", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_private_key_file_is_readable_on_current_platform(tmp_path):
    key = tmp_path / "provider-api-key.txt"
    key.write_text("secret-value\n", encoding="utf-8")
    if os.name != "nt":
        key.chmod(0o600)

    assert MOD._read_private_key(key) == "secret-value"


def test_private_key_rejects_open_mode_on_posix_filesystem(tmp_path, monkeypatch):
    key = tmp_path / "provider-api-key.txt"
    key.write_text("secret-value\n", encoding="utf-8")
    key.chmod(0o644)
    monkeypatch.setattr(MOD, "_has_meaningful_posix_permissions", lambda _path: True)

    with pytest.raises(ValueError, match="permissions must be 600"):
        MOD._read_private_key(key)


def test_private_key_ignores_synthetic_drvfs_mode(tmp_path, monkeypatch):
    key = tmp_path / "provider-api-key.txt"
    key.write_text("secret-value\n", encoding="utf-8")
    key.chmod(0o777)
    monkeypatch.setattr(MOD, "_has_meaningful_posix_permissions", lambda _path: False)

    assert MOD._read_private_key(key) == "secret-value"


def test_stop_requested_for_item_follows_build_checkpoint_sentinel(tmp_path):
    checkpoint = tmp_path / "item" / "build-checkpoint.json"
    checkpoint.parent.mkdir()
    payload = {"paths": {"build_checkpoint": str(checkpoint)}}

    assert MOD._stop_requested_for_item(payload) is False
    (checkpoint.parent / ".stop-after-current-batch").write_text(
        "stop\n", encoding="utf-8"
    )
    assert MOD._stop_requested_for_item(payload) is True
    assert MOD._stop_requested_for_item(None) is False
    assert MOD._should_stop_worker(
        is_paused=True, checkpoint=None
    ) is True
    assert MOD._should_stop_worker(
        is_paused=False, checkpoint=payload
    ) is True


@pytest.mark.skipif(os.name == "nt", reason="WSL mount parsing is Linux-only")
def test_detects_real_wsl_drvfs_mount_option(tmp_path):
    mounts = tmp_path / "mounts"
    mounts.write_text(
        "E:\\134 /mnt/e 9p "
        "rw,aname=drvfs;path=E:\\;uid=1000;gid=1000,cache=5 0 0\n",
        encoding="utf-8",
    )
    assert MOD._has_meaningful_posix_permissions(
        MOD.Path("/mnt/e/key"), mounts=mounts
    ) is False


def test_round_robin_indices_balance_question_types():
    data = [
        {"question_type": "a"},
        {"question_type": "a"},
        {"question_type": "b"},
        {"question_type": "c"},
        {"question_type": "b"},
        {"question_type": "a"},
    ]

    assert MOD.round_robin_indices(data) == [0, 2, 3, 1, 4, 5]


def test_question_type_indices_use_positions_within_type_and_reverse():
    data = [
        {"question_type": "a"},
        {"question_type": "b"},
        {"question_type": "a"},
        {"question_type": "a"},
    ]

    assert MOD.question_type_indices(data, "a", 1, 2, reverse=False) == [2, 3]
    assert MOD.question_type_indices(data, "a", 0, 2, reverse=True) == [3, 2]


def test_shared_claim_is_atomic_and_records_terminal_state(tmp_path):
    db = tmp_path / "queue.sqlite3"
    MOD.initialize_queue(db, [7])
    assert MOD.claim_item(db, 7, "worker-a")
    assert not MOD.claim_item(db, 7, "worker-b")

    MOD.finish_claim(db, 7)

    assert MOD.queue_states(db) == {7: 2}
