import json
import src.v8_memory as V8
from src.v8_memory import _inline_refs


LM = {1: "D1:1", 3: "D1:3", 5: "D1:5"}


# ---- _inline_refs：句内 [n] 标记 → 真实 dia_id ----

def test_inline_marker_converted_at_mention_point():
    plain, inline, extra = _inline_refs(
        "decided to adopt [1], visited an agency [3]", LM)
    assert inline == "decided to adopt [D1:1], visited an agency [D1:3]"
    assert plain == "decided to adopt, visited an agency"
    assert extra == ["D1:1", "D1:3"]


def test_inline_marker_group():
    _plain, inline, extra = _inline_refs("chose it for its values [1,3]", LM)
    assert inline == "chose it for its values [D1:1,3]"
    assert extra == ["D1:1", "D1:3"]


def test_inline_invalid_number_dropped():
    plain, inline, extra = _inline_refs("went hiking [9]", LM)
    # 无效行号：标记整个移除，回落为无标记
    assert "[" not in inline
    assert plain == "went hiking"
    assert extra == []


def test_inline_no_marker_passthrough():
    plain, inline, extra = _inline_refs("went hiking", LM)
    assert plain == "went hiking" == inline
    assert extra == []


# ---- _parse_distill_response：summary 拆纯文本 + summary_inline ----

class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()


def _fake(monkeypatch, content):
    monkeypatch.setattr(V8.client.chat.completions, "create",
                        lambda **k: _FakeResp(content))
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)


def test_distill_summary_split_plain_and_inline(monkeypatch):
    payload = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "Caroline adopted a dog [1]",
         "refs": [1], "topic": "Caroline/pets"}]})
    _fake(monkeypatch, payload)
    evs = V8.distill_events([("Caroline", "I adopted a dog")],
                            "2023-05-07", ["D1:3"])
    assert evs[0]["summary"] == "Caroline adopted a dog"          # 纯文本
    assert evs[0]["summary_inline"] == "Caroline adopted a dog [D1:3]"
    assert evs[0]["dia_ids"] == ["D1:3"]


def test_distill_intext_marker_unions_into_dia_ids(monkeypatch):
    # refs=[1] 但句内还引了 [2]：dia_ids 取并集，timeline 锚不缺
    payload = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "went to Paris [1], loved it [2]",
         "refs": [1], "topic": "travel"}]})
    _fake(monkeypatch, payload)
    evs = V8.distill_events([("A", "a"), ("B", "b")],
                            "2023-05-07", ["D1:1", "D1:2"])
    assert evs[0]["dia_ids"] == ["D1:1", "D1:2"]
    assert evs[0]["summary_inline"] == "went to Paris [D1:1], loved it [D1:2]"


def test_distill_no_marker_falls_back_to_trailing_refs(monkeypatch):
    payload = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "went hiking",
         "refs": [1], "topic": "hobbies"}]})
    _fake(monkeypatch, payload)
    evs = V8.distill_events([("A", "I went hiking")], "2023-05-07", ["D1:3"])
    # 句内无标记：回退为句末追加（与老 topics 行体一致）
    assert evs[0]["summary_inline"] == "went hiking · [D1:3]"


def test_distill_prompt_teaches_inline_and_absolute_dates():
    p = V8._V8_DISTILL_PROMPT
    assert "提及处" in p or "写到哪引到哪" in p    # 内联标记指令
    assert "绝对日期" in p                          # 相对时间换算指令
