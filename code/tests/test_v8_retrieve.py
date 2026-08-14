import os
import src.adapters.run_nativemem as R
import src.v8_memory as V8

def test_collect_v8_returns_original_via_dia_id(tmp_path, monkeypatch):
    d = str(tmp_path / "mem")
    # 预置双视图记忆：一条事件行带 [D1:3]
    V8.write_events(d, [{"when": "2023-05-07", "summary": "去支持小组",
                         "dia_ids": ["D1:3"], "topic": "support"}])
    conv = {"session_1": [
        {"speaker": "Caroline", "dia_id": "D1:3", "text": "I went to a LGBTQ support group"}]}
    idx = V8.build_turn_index(conv)

    # fake 模型：第一轮吐一个 grep tool_call，第二轮调 read_original 回原文，第三轮收尾
    calls = {"n": 0}
    class _Msg:
        def __init__(self, content, tool_calls=None):
            self.content = content; self.tool_calls = tool_calls
    class _Resp:
        def __init__(self, msg): self.choices=[type("C",(),{"message":msg})()]; self.usage=type("U",(),{"prompt_tokens":1,"completion_tokens":1})()
    def fake_create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            tc = type("T", (), {"id":"1","function":type("F",(),{"name":"bash","arguments":'{"cmd":"grep -r 支持小组 topics/"}'})()})()
            return _Resp(_Msg("", [tc]))
        if calls["n"] == 2:
            tc = type("T", (), {"id":"2","function":type("F",(),{
                "name":"read_original","arguments":'{"dia_ids": ["D1:3"]}'})()})()
            return _Resp(_Msg("", [tc]))  # 模型调 read_original 指认 D1:3
        return _Resp(_Msg("done"))  # 第三轮无 tool_call 收尾
    monkeypatch.setattr(R.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(R, "log_usage", lambda *a, **k: None)

    mems, steps = R._collect_v8("去过支持小组吗", d, idx)
    joined = " ".join(m.get("text","") for m in mems)
    assert "LGBTQ support group" in joined  # 回原文取到了 D1:3 的原始句子
