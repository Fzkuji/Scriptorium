"""Run summary reads both the legacy and the current result layouts."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "analyze_run.py"

EVAL = {
    "records": [
        {"judge_score": 1, "answer": "Tokyo", "category": 2},
        {"judge_score": 0, "answer": "not mentioned", "category": 1},
    ]
}


def run(path: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def test_reads_current_build_and_performance(tmp_path: Path):
    (tmp_path / "eval_full.json").write_text(json.dumps(EVAL))
    (tmp_path / "build.json").write_text(json.dumps({
        "wall_time_s": 6000, "calls": 469, "output_tokens": 650_000,
        "event_count": 155,
        "memory": {"files": 87, "topic_files": 3, "timeline_files": 50,
                   "source_files": 30},
    }))
    (tmp_path / "performance.json").write_text(json.dumps({
        "usage": {"totals": {
            "estimated_cost_usd": 0.8355,
            "input_tokens": 4_111_302,
            "output_tokens": 928_213,
            "cache_read_tokens": 12_288,
        }}
    }))

    out = run(tmp_path)

    assert "100min 469调用" in out
    assert "events=155" in out
    assert "87文件" in out
    # The trustworthy figure, not the SDK's Anthropic-priced one.
    assert "estimated=$0.8355" in out


def test_still_reads_the_legacy_build_stats_record(tmp_path: Path):
    (tmp_path / "eval_full.json").write_text(json.dumps(EVAL))
    (tmp_path / "sample0_questions.json").write_text(json.dumps([{
        "question_id": "_build_stats",
        "build_time_s": 600, "build_calls": 365,
        "build_tokens_out": 40_000, "notes": "built fresh",
    }]))

    out = run(tmp_path)

    assert "10min 365调用" in out
    assert "built fresh" in out


def test_missing_build_artifacts_do_not_crash(tmp_path: Path):
    (tmp_path / "eval_full.json").write_text(json.dumps(EVAL))

    out = run(tmp_path)

    assert "n=2" in out


def test_reads_longmemeval_results(tmp_path: Path):
    """LongMemEval reports avg_score/results, not overall/records."""
    (tmp_path / "eval_30.json").write_text(json.dumps({
        "avg_score": 73.33, "found": 29,
        "results": [
            {"question_type": "temporal-reasoning", "score": 1},
            {"question_type": "multi-session", "score": 0},
        ],
    }))

    out = run(tmp_path)

    assert "avg=73.3" in out
    assert "found=29" in out
    assert "temporal-reasoning" in out


def test_locomo_and_longmemeval_can_coexist(tmp_path: Path):
    (tmp_path / "eval_full.json").write_text(json.dumps(EVAL))
    (tmp_path / "eval_30.json").write_text(json.dumps({
        "avg_score": 50.0, "found": 1,
        "results": [{"question_type": "x", "score": 1}],
    }))

    out = run(tmp_path)

    assert "n=2" in out
    assert "avg=50.0" in out
