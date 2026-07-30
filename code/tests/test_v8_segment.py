import json
import src.v8_memory as V8
import src.adapters.run_nativemem as R


class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 10})()


def _install_fake(monkeypatch, content):
    def fake_create(**kwargs):
        return _FakeResp(content)
    monkeypatch.setattr(V8.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)


def _mk(n):
    turns = [(f"S{i}", f"t{i}") for i in range(1, n + 1)]
    dia_ids = [f"D1:{i}" for i in range(1, n + 1)]
    return turns, dia_ids


def test_segment_normal(monkeypatch):
    turns, dia_ids = _mk(6)
    _install_fake(monkeypatch, json.dumps({"segments": [[1, 2, 3], [4, 5, 6]]}))
    out = V8.segment_session_by_topic(turns, dia_ids)
    assert len(out) == 2
    assert out[0] == ([turns[0], turns[1], turns[2]], ["D1:1", "D1:2", "D1:3"])
    assert out[1] == ([turns[3], turns[4], turns[5]], ["D1:4", "D1:5", "D1:6"])


def test_segment_cap_hard_split(monkeypatch):
    # 一段 12 句 > cap 10 → 硬切成 10 + 2
    turns, dia_ids = _mk(12)
    _install_fake(monkeypatch, json.dumps({"segments": [list(range(1, 13))]}))
    out = V8.segment_session_by_topic(turns, dia_ids)
    assert [len(t) for t, _ in out] == [10, 2]
    # 覆盖完整、不乱序
    flat = [d for _, ids in out for d in ids]
    assert flat == dia_ids


def test_segment_missing_number_falls_back(monkeypatch):
    turns, dia_ids = _mk(6)
    # 漏了 5
    _install_fake(monkeypatch, json.dumps({"segments": [[1, 2, 3], [4, 6]]}))
    out = V8.segment_session_by_topic(turns, dia_ids)
    # 回退固定 6 句切块 → 单块
    assert out == [(turns, dia_ids)]


def test_segment_duplicate_number_falls_back(monkeypatch):
    turns, dia_ids = _mk(6)
    _install_fake(monkeypatch, json.dumps({"segments": [[1, 2, 3], [3, 4, 5, 6]]}))
    out = V8.segment_session_by_topic(turns, dia_ids)
    assert out == [(turns, dia_ids)]


def test_segment_non_consecutive_falls_back(monkeypatch):
    turns, dia_ids = _mk(6)
    # 段内非连续递增 [1,3,2]
    _install_fake(monkeypatch, json.dumps({"segments": [[1, 3, 2], [4, 5, 6]]}))
    out = V8.segment_session_by_topic(turns, dia_ids)
    assert out == [(turns, dia_ids)]


def test_segment_bad_json_falls_back(monkeypatch):
    turns, dia_ids = _mk(6)
    _install_fake(monkeypatch, "not json at all")
    out = V8.segment_session_by_topic(turns, dia_ids, max_retry=1)
    assert out == [(turns, dia_ids)]


def test_segment_fallback_respects_chunk_env(monkeypatch):
    # 回退用 NATIVEMEM_CHUNK_TURNS 决定固定块大小
    monkeypatch.setenv("NATIVEMEM_CHUNK_TURNS", "3")
    turns, dia_ids = _mk(6)
    _install_fake(monkeypatch, "garbage")
    out = V8.segment_session_by_topic(turns, dia_ids, max_retry=1)
    assert [len(t) for t, _ in out] == [3, 3]


def test_segment_cap_env(monkeypatch):
    monkeypatch.setenv("NATIVEMEM_V8_SEGMENT_MAX", "4")
    turns, dia_ids = _mk(6)
    _install_fake(monkeypatch, json.dumps({"segments": [[1, 2, 3, 4, 5, 6]]}))
    out = V8.segment_session_by_topic(turns, dia_ids)
    assert [len(t) for t, _ in out] == [4, 2]


def test_segment_empty(monkeypatch):
    # 无 turn → 空，且不打 API
    called = {"n": 0}

    def fake_create(**kwargs):
        called["n"] += 1
        return _FakeResp("{}")
    monkeypatch.setattr(V8.client.chat.completions, "create", fake_create)
    out = V8.segment_session_by_topic([], [])
    assert out == []
    assert called["n"] == 0


def _build_conv():
    return {
        "session_1": [
            {"speaker": "A", "dia_id": f"D1:{i}", "text": f"t{i}"}
            for i in range(1, 5)
        ],
        "session_1_date_time": "2023-05-07",
    }


def test_build_fixed_default_skips_segment(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v8")
    called = {"n": 0}
    monkeypatch.setattr(V8, "segment_session_by_topic",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or [])
    monkeypatch.setattr(V8, "distill_events", lambda *a, **k: [])
    R.build_memory(_build_conv(), str(tmp_path / "m"))
    assert called["n"] == 0


def test_build_topic_calls_segment(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v8")
    monkeypatch.setenv("NATIVEMEM_V8_SEGMENT", "topic")
    seen = {"turns": None}

    def fake_segment(turns, dia_ids, **k):
        seen["turns"] = list(turns)
        return [(turns, dia_ids)]
    monkeypatch.setattr(V8, "segment_session_by_topic", fake_segment)
    monkeypatch.setattr(V8, "distill_events", lambda *a, **k: [])
    R.build_memory(_build_conv(), str(tmp_path / "m"))
    # 整个 session 铺平后交给 segment：4 个 turn 一次性传入
    assert len(seen["turns"]) == 4
