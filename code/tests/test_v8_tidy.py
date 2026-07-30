import os
import src.v8_memory as V8


def test_dedup_removes_exact_duplicate_lines(tmp_path):
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when": "2023-05-07", "summary": "A", "dia_ids": ["D1:1"], "topic": "t"}])
    V8.write_events(d, [{"when": "2023-05-07", "summary": "A", "dia_ids": ["D1:1"], "topic": "t"}])
    removed = V8.dedup_topic_files(d)
    body = open(os.path.join(d, "topics", "t.md")).read()
    assert body.count("A ·") == 1
    assert removed >= 1


def test_topic_merge_candidates_finds_near_synonyms(tmp_path):
    d = str(tmp_path / "mem")
    for t in ["mental-health-career", "mental-health-work", "art"]:
        V8.write_events(d, [{"when": "2023-05-07", "summary": "x", "dia_ids": ["D1:1"], "topic": t}])
    groups = V8._topic_merge_candidates(d)
    flat = [set(g) for g in groups]
    assert any({"mental-health-career", "mental-health-work"} <= g for g in flat)


def test_consolidate_merges_per_model_verdict(tmp_path, monkeypatch):
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when": "2023-05-07", "summary": "P", "dia_ids": ["D1:1"], "topic": "mental-health-career"}])
    V8.write_events(d, [{"when": "2023-05-08", "summary": "Q", "dia_ids": ["D1:2"], "topic": "mental-health-work"}])
    import json as _j
    monkeypatch.setattr(V8.client.chat.completions, "create",
        lambda **k: type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content":
          _j.dumps({"merges": [{"from": ["mental-health-career", "mental-health-work"],
                               "into": "mental-health-career", "merge": True}]})})()})()],
          "usage": type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()})())
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)
    V8.consolidate_topic_files(d)
    assert not os.path.exists(os.path.join(d, "topics", "mental-health-work.md"))
    merged = open(os.path.join(d, "topics", "mental-health-career.md")).read()
    assert "P" in merged and "Q" in merged
