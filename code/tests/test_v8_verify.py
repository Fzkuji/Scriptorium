import json
import src.v8_memory as V8
from src.v8_memory import verify_event_coverage


# ---- verify_event_coverage：抓原文有、摘要没覆盖的专名/数字 ----

def test_verify_catches_missing_proper_noun():
    turns = [("Caroline", "I'm reading Becoming Nicole this week")]
    events = [{"summary": "Caroline is reading a book"}]   # 丢了书名
    missing = verify_event_coverage(turns, events)
    assert any("Becoming Nicole" in m or "Nicole" in m for m in missing)


def test_verify_empty_when_fully_covered():
    turns = [("Caroline", "I went to Paris in 2019")]
    events = [{"summary": "Caroline went to Paris in 2019"}]
    assert verify_event_coverage(turns, events) == []


def test_verify_catches_missing_number():
    turns = [("Melanie", "My daughter turned 7 last month")]
    events = [{"summary": "Melanie's daughter had a birthday"}]   # 丢了 7
    missing = verify_event_coverage(turns, events)
    assert "7" in missing


def test_verify_ignores_speaker_names():
    # 说话人名（turn 的 speaker）不该被当成缺失的专名
    turns = [("Caroline", "I like tea")]
    events = [{"summary": "she likes tea"}]
    missing = verify_event_coverage(turns, events)
    assert "Caroline" not in missing


def test_verify_case_insensitive_coverage():
    turns = [("A", "I visited BERLIN")]
    events = [{"summary": "visited berlin"}]   # 大小写不同也算覆盖
    assert verify_event_coverage(turns, events) == []


# ---- distill_events 补抽轮次：首轮漏了专名，第二轮补回并合并 ----

class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 10})()


def test_distill_second_pass_backfills_missing(monkeypatch):
    round1 = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "Caroline is reading a book",
         "refs": [1], "topic": "Caroline/reading"}]})
    round2 = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "Caroline is reading Becoming Nicole",
         "refs": [1], "topic": "Caroline/reading"}]})
    seq = iter([round1, round2])

    calls = {"n": 0}
    def fake_create(**kwargs):
        calls["n"] += 1
        return _FakeResp(next(seq))
    monkeypatch.setattr(V8.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)
    monkeypatch.setenv("NATIVEMEM_V8_VERIFY", "on")

    evs = V8.distill_events([("Caroline", "I'm reading Becoming Nicole")],
                            "2023-05-07", ["D1:3"])
    # 两轮都调了模型（首轮漏书名 → 触发补抽）
    assert calls["n"] == 2
    # 合并了两轮结果
    joined = " ".join(e["summary"] for e in evs)
    assert "Becoming Nicole" in joined


def test_distill_verify_off_skips_second_pass(monkeypatch):
    round1 = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "Caroline is reading a book",
         "refs": [1], "topic": "Caroline/reading"}]})
    calls = {"n": 0}
    def fake_create(**kwargs):
        calls["n"] += 1
        return _FakeResp(round1)
    monkeypatch.setattr(V8.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)
    monkeypatch.setenv("NATIVEMEM_V8_VERIFY", "off")

    evs = V8.distill_events([("Caroline", "I'm reading Becoming Nicole")],
                            "2023-05-07", ["D1:3"])
    assert calls["n"] == 1   # off：不补抽
    assert len(evs) == 1
