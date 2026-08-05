import json
from pathlib import Path


def test_ablation_loads_locomo_without_historical_runner(tmp_path: Path):
    from scripts.runners.ablation.outputs import load_unit

    data = tmp_path / "locomo.json"
    data.write_text(json.dumps([{
        "sample_id": "conv-test",
        "conversation": {
            "session_1": [{
                "speaker": "user",
                "text": "Remember this.",
                "dia_id": "D1:1",
            }],
            "session_1_date_time": "2026-08-02",
        },
        "qa": [{
            "question": "What should be remembered?",
            "answer": "this",
            "category": 1,
            "evidence": ["D1:1"],
        }],
    }]), encoding="utf-8")
    row = {
        "benchmark": "locomo",
        "selection": {"sample_index": 0, "sample_id": "conv-test"},
        "data_path": str(data),
    }

    conversation, questions, identity = load_unit(row)

    assert conversation["session_1"][0]["text"] == "Remember this."
    assert questions[0]["gold"] == "this"
    assert identity == {"sample_id": "conv-test"}


def test_longmemeval_selection_uses_current_support_module():
    from scripts.runners.longmemeval.selection import question_type_indices

    data = [
        {"question_type": "single-session-user"},
        {"question_type": "single-session-user"},
    ]

    assert question_type_indices(
        data, "single-session-user", 0, 1, reverse=False
    ) == [0]
