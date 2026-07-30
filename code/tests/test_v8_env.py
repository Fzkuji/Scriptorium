import os
import src.adapters.run_nativemem as R
import src.v8_memory as V8


def _seed(d):
    V8.write_events(d, [
        {"when": "2023-05-07", "summary": "A", "dia_ids": ["D1:1"], "topic": "t"},
    ])


class _FakeMsg:
    def __init__(self):
        self.content = ""
        self.tool_calls = None


class _FakeResp:
    def __init__(self):
        self.choices = [type("C", (), {"message": _FakeMsg()})()]
        self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()


def _install_recording_client(monkeypatch):
    """fake client that records create() kwargs and returns a no-tool-call msg
    (so the collect loop exits after 1 round)."""
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return _FakeResp()
    monkeypatch.setattr(R.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(R, "log_usage", lambda *a, **k: None)
    return calls


def test_max_tokens_default(tmp_path, monkeypatch):
    d = str(tmp_path / "mem"); _seed(d)
    calls = _install_recording_client(monkeypatch)
    R._collect_v8("q?", d, {})
    assert calls[0]["max_tokens"] == 1200   # 默认值 = 现值


def test_max_tokens_from_env(tmp_path, monkeypatch):
    d = str(tmp_path / "mem"); _seed(d)
    monkeypatch.setenv("NATIVEMEM_V8_MAX_TOKENS", "77")
    calls = _install_recording_client(monkeypatch)
    R._collect_v8("q?", d, {})
    assert calls[0]["max_tokens"] == 77


def test_max_rounds_from_env(tmp_path, monkeypatch):
    # fake client 每次都返回一个 read_original tool_call → 永不自然收敛，
    # 循环轮数只受 max_rounds 限制。设 env=3 → 恰好 3 轮。
    d = str(tmp_path / "mem"); _seed(d)
    monkeypatch.setenv("NATIVEMEM_V8_MAX_ROUNDS", "3")

    class _TC:
        id = "1"
        function = type("F", (), {"name": "read_original",
                                  "arguments": '{"dia_ids": ["D1:1"]}'})()

    class _MsgTC:
        content = ""
        tool_calls = [_TC()]

    def fake_create(**kwargs):
        return type("R", (), {"choices": [type("C", (), {"message": _MsgTC()})()],
                              "usage": type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()})()
    monkeypatch.setattr(R.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(R, "log_usage", lambda *a, **k: None)
    _mems, steps = R._collect_v8("q?", d, {"D1:1": {"speaker": "C", "text": "x", "order": 0, "date": "2023-05-07"}})
    assert steps == 3


def test_read_context_from_env(tmp_path, monkeypatch):
    # NATIVEMEM_V8_READ_CONTEXT 传给 read_turns 的 context 参数。
    d = str(tmp_path / "mem"); _seed(d)
    monkeypatch.setenv("NATIVEMEM_V8_READ_CONTEXT", "5")
    seen = []
    real_read = R.read_turns

    def spy_read(turn_index, dia_ids, context=1):
        seen.append(context)
        return real_read(turn_index, dia_ids, context=context)
    monkeypatch.setattr(R, "read_turns", spy_read)

    class _TC:
        id = "1"
        function = type("F", (), {"name": "read_original",
                                  "arguments": '{"dia_ids": ["D1:1"]}'})()

    responses = [
        type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "", "tool_calls": [_TC()]})()})()],
                       "usage": type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()})(),
        type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "", "tool_calls": None})()})()],
                       "usage": type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()})(),
    ]
    it = iter(responses)
    monkeypatch.setattr(R.client.chat.completions, "create", lambda **k: next(it))
    monkeypatch.setattr(R, "log_usage", lambda *a, **k: None)
    R._collect_v8("q?", d, {"D1:1": {"speaker": "C", "text": "x", "order": 0, "date": "2023-05-07"}})
    assert 5 in seen
