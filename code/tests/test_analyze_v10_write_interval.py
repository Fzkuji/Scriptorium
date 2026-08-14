import json

import pytest

from scripts import analyze_v10_write_interval as analyze


def test_analyze_build_reports_gold_evidence_coverage(tmp_path):
    memory_dir = tmp_path / "memory"
    topics = memory_dir / "topics"
    topics.mkdir(parents=True)
    (topics / "pets.md").write_text(
        "# pets\n"
        "[2023-05-07] User adopted Poppy. · [D1:1]\n"
        "[2023-05-07] User adopted Poppy. · [D1:1]\n"
        "[2023-05-07] Poppy likes parks. · [D1:3-4]\n"
    )
    dataset = [
        {
            "qa": [
                {"category": 1, "evidence": ["D1:1"]},
                {"category": 2, "evidence": ["D1:2", "D1:3"]},
                {"category": 5, "evidence": ["D1:4"]},
            ]
        }
    ]
    build_record = tmp_path / "build.json"
    build_record.write_text(
        json.dumps(
            [
                {
                    "question_id": "_build_stats",
                    "method": "NativeMem-v10",
                    "sample": 0,
                    "builder_model": "test-model",
                    "memory_dir": str(memory_dir),
                    "v10_config": {
                        "write_turns": 6,
                        "context_mode": "events",
                        "context_items": 20,
                        "summary_max_words": 180,
                        "tidy_every_sessions": 1,
                        "session_tidy_passes": 1,
                        "final_tidy_passes": 1,
                    },
                    "build_calls": 7,
                    "build_tokens_in": 100,
                    "build_tokens_out": 20,
                    "build_time_s": 3.5,
                    "build_phase_usage": {
                        "v8_distill": {
                            "calls": 5,
                            "tokens_in": 80,
                            "tokens_out": 15,
                        }
                    },
                }
            ]
        )
    )

    row = analyze.analyze_build(build_record, dataset, max_sessions_override=1)

    assert row["questions_with_evidence"] == 2
    assert row["evidence_question_any_recall"] == 1.0
    assert row["evidence_question_all_recall"] == 0.5
    assert row["unique_gold_evidence_ids"] == 3
    assert row["unique_gold_evidence_id_recall"] == 2 / 3
    assert row["questions_with_evidence_in_build_scope"] == 2
    assert row["unique_gold_evidence_id_recall_in_build_scope"] == 2 / 3
    assert row["event_lines"] == 3
    assert row["unique_event_lines_normalized"] == 2
    assert row["exact_normalized_duplicate_fraction"] == pytest.approx(1 / 3)
    assert row["build_calls"] == 7
    assert row["summary_max_words"] == 180
    assert row["build_phase_usage"]["v8_distill"]["calls"] == 5
