import json
import src.v8_memory as V8


class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()


def _capture(monkeypatch, payload):
    captured = {}

    def fake_create(**kwargs):
        captured.setdefault("messages", []).append(kwargs["messages"])
        return _FakeResp(payload)
    monkeypatch.setattr(V8.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)
    return captured


_PAYLOAD = json.dumps({"events": [
    {"when": "2023-05-07", "summary": "x", "refs": [1], "topic": "t"}]})


def test_distill_includes_recent_context(monkeypatch):
    cap = _capture(monkeypatch, _PAYLOAD)
    V8.distill_events([("A", "x")], "2023-05-07", ["D1:1"],
                      recent=["Caroline adopted a dog", "Melanie painted a lake"])
    sys_prompt = cap["messages"][0][0]["content"]
    assert "Caroline adopted a dog" in sys_prompt
    assert "Melanie painted a lake" in sys_prompt
    assert "别重复" in sys_prompt          # 勿重复指令
    assert "消解" in sys_prompt            # 指代消解指令


def test_distill_no_recent_no_block(monkeypatch):
    cap = _capture(monkeypatch, _PAYLOAD)
    V8.distill_events([("A", "x")], "2023-05-07", ["D1:1"])
    sys_prompt = cap["messages"][0][0]["content"]
    assert "别重复" not in sys_prompt


def test_build_threads_rolling_context(tmp_path, monkeypatch):
    # build 主循环把本 session 已提炼句子的尾部传给下一个 chunk 的 distill
    import src.adapters.run_nativemem as R
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v8")
    monkeypatch.setenv("NATIVEMEM_CHUNK_TURNS", "1")
    conv = {"session_1": [
        {"speaker": "C", "dia_id": "D1:1", "text": "a"},
        {"speaker": "C", "dia_id": "D1:2", "text": "b"},
    ], "session_1_date_time": "2023-05-07"}
    seen_recent = []

    def fake_distill(turns, obs, dids, known_topics=None, recent=None, **k):
        seen_recent.append(list(recent or []))
        return [{"when": "2023-05-07", "summary": f"sum-{dids[0]}",
                 "dia_ids": list(dids), "topic": "t"}]
    monkeypatch.setattr(V8, "distill_events", fake_distill)
    R.build_memory(conv, str(tmp_path / "mem"))
    assert seen_recent[0] == []                    # 首 chunk 无上下文
    assert seen_recent[1] == ["sum-D1:1"]          # 第二 chunk 带上第一块产出
