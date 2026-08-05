import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/runners/run_longmemeval.py"
SPEC = importlib.util.spec_from_file_location("run_longmemeval_script", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


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
