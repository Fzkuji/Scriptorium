import json
import src.v8_memory as V8

class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 10})()

def _install_fake(monkeypatch, content):
    def fake_create(**kwargs):
        return _FakeResp(content)
    monkeypatch.setattr(V8.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)

def test_distill_events_parses_json(monkeypatch):
    payload = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "去支持小组", "refs": [1], "topic": "support"},
    ]})
    _install_fake(monkeypatch, payload)
    evs = V8.distill_events([("Caroline", "I went to a support group")],
                            "2023-05-07", ["D1:3"])
    assert len(evs) == 1
    assert evs[0]["summary"] == "去支持小组"
    assert evs[0]["dia_ids"] == ["D1:3"]

def test_distill_events_backfills_missing_when(monkeypatch):
    payload = json.dumps({"events": [{"summary": "x", "refs": [1], "topic": "t"}]})
    _install_fake(monkeypatch, payload)
    evs = V8.distill_events([("A", "x")], "2023-05-07", ["D1:1"])
    assert evs[0]["when"] == "2023-05-07"

def test_distill_events_bad_json_returns_empty(monkeypatch):
    _install_fake(monkeypatch, "not json at all")
    evs = V8.distill_events([("A", "x")], "2023-05-07", ["D1:1"], max_retry=1)
    assert evs == []

def test_distill_refs_bare_number_strings(monkeypatch):
    # refs 是模型给出的行号（字符串也接受），代码经 _refs_to_dia_ids 转成 dia_id
    payload = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "x", "refs": ["1", "3"], "topic": "t"},
    ]})
    _install_fake(monkeypatch, payload)
    evs = V8.distill_events([("A", "a"), ("B", "b"), ("C", "c")],
                            "2023-05-07", ["D1:3", "D1:4", "D1:5"])
    assert evs[0]["dia_ids"] == ["D1:3", "D1:5"]

def test_distill_drops_invalid_dia_ids(monkeypatch):
    # 越界行号（对话只有 1 个 turn）→ 转不出 dia_id → 丢弃
    payload = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "x", "refs": [99], "topic": "t"},
    ]})
    _install_fake(monkeypatch, payload)
    evs = V8.distill_events([("A", "x")], "2023-05-07", ["D1:3"])
    assert evs == []

def test_distill_keeps_correct_dia_ids(monkeypatch):
    payload = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "x", "refs": [1], "topic": "t"},
    ]})
    _install_fake(monkeypatch, payload)
    evs = V8.distill_events([("A", "x"), ("B", "y")], "2023-05-07", ["D1:3", "D1:4"])
    assert evs[0]["dia_ids"] == ["D1:3"]

def test_distill_refs_multiple_refs_one_event(monkeypatch):
    # 多个行号引用同一事件，经 _refs_to_dia_ids 各自转出
    payload = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "x", "refs": [1, 3], "topic": "t"},
    ]})
    _install_fake(monkeypatch, payload)
    evs = V8.distill_events([("A", "a"), ("B", "b"), ("C", "c")],
                            "2023-05-07", ["D1:3", "D1:4", "D1:5"])
    assert evs[0]["dia_ids"] == ["D1:3", "D1:5"]

def test_distill_events_scalar_json_returns_empty(monkeypatch):
    # Test that bare scalar JSON (null, true, false, number) returns [] without raising
    _install_fake(monkeypatch, "null")
    evs = V8.distill_events([("A", "x")], "2023-05-07", ["D1:1"], max_retry=1)
    assert evs == []

    # Also test a number
    _install_fake(monkeypatch, "42")
    evs = V8.distill_events([("A", "x")], "2023-05-07", ["D1:1"], max_retry=1)
    assert evs == []


def test_distill_rejects_relative_time_in_when(monkeypatch):
    # 模型把相对时间词塞进 when（"friday"）而非 YYYY-MM-DD —— 必须回填 obs_date，
    # 不能让它进 when（否则 timeline_path 的 when.split("-") 会崩）。
    import json as _json
    payload = _json.dumps({"events": [
        {"when": "friday", "summary": "Melanie signed up for pottery on friday",
         "refs": [1], "topic": "Melanie-pottery"}]})
    _install_fake(monkeypatch, payload)
    turns = [("Melanie", "signed up for pottery class on friday")]
    evs = V8.distill_events(turns, "2023-07-15", ["D1:1"])
    assert len(evs) == 1
    assert evs[0]["when"] == "2023-07-15"          # 非法 when 回填 obs_date
    assert "friday" in evs[0]["summary"].lower()   # 相对时间词留在 summary 里
