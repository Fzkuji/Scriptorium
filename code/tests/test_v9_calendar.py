"""v9 日历条注入:calendar_strip 纯日历事实 + 转写 prompt 对照查表规则。
设计红线:函数输出不许出现任何英文相对时间短语(那是语言规则,归模型)。"""
import src.v8_memory as V8


def test_sunday_strip_rows():
    s = V8.calendar_strip("2023-07-23")
    assert "对话日期：2023-07-23（星期日/Sun）" in s
    assert "2023-07-21 星期五/Fri" in s          # 验尸案例:上周五该是 07-21
    assert "2023-07-22 星期六/Sat" in s
    assert "上一个自然周：2023-07-10（星期一）~ 2023-07-16（星期日）" in s
    assert "上月：2023-06" in s and "去年：2022" in s


def test_no_relative_phrases_in_strip():
    s = V8.calendar_strip("2023-07-23").lower()
    for banned in ["yesterday", "tomorrow", "last friday", "last week",
                   "days ago", "weekend"]:
        assert banned not in s


def test_cross_month_and_year():
    s = V8.calendar_strip("2024-01-05")
    assert "2023-12-31" in s and "2023-12-20" in s   # 三周日历跨回去年 12 月
    assert "上月：2023-12" in s and "去年：2023" in s
    s2 = V8.calendar_strip("2023-08-03")
    assert "2023-07-31" in s2                        # 跨月


def test_weekday_correctness_monday():
    s = V8.calendar_strip("2023-07-24")              # 周一
    assert "对话日期：2023-07-24（星期一/Mon）" in s
    assert "2023-07-23 星期日/Sun" in s
    assert "上一个自然周：2023-07-17（星期一）~ 2023-07-23（星期日）" in s


def test_illegal_dates_empty():
    assert V8.calendar_strip("2023-13-01") == ""
    assert V8.calendar_strip("2023-02-30") == ""
    assert V8.calendar_strip("") == ""
    assert V8.calendar_strip(None) == ""


def test_prompt_has_calendar_slot_and_rules():
    p = V8._V9_TRANSCRIBE_PROMPT
    assert "{calendar}" in p
    assert "对照上面的日历" in p and "禁止脱离日历心算" in p
    assert "保留原词，禁止造日期" in p and "并存" in p


def test_transcribe_injects_calendar(monkeypatch):
    captured = {}

    def fake(messages, phase="", max_retry=6, model=None):
        captured["sys"] = messages[0]["content"]
        return '{"notes":[{"line":1,"note":"n1"},{"line":2,"note":"n2"}]}'

    monkeypatch.setattr(V8, "_distill_call", fake)
    V8.transcribe_session([("A", "hi"), ("B", "yo")], ["D1:1", "D1:2"],
                          "2023-07-23")
    assert "2023-07-21 星期五/Fri" in captured["sys"]


def test_transcribe_illegal_date_uses_fallback(monkeypatch):
    captured = {}

    def fake(messages, phase="", max_retry=6, model=None):
        captured["sys"] = messages[0]["content"]
        return '{"notes":[{"line":1,"note":"n1"}]}'

    monkeypatch.setattr(V8, "_distill_call", fake)
    V8.transcribe_session([("A", "hi")], ["D1:1"], "未知日期")
    assert "保留原词" in captured["sys"]


def test_v8_distill_injects_calendar(monkeypatch):
    captured = {}

    def fake(messages, phase="", max_retry=6, model=None):
        captured["sys"] = messages[0]["content"]
        return '{"events":[]}'

    monkeypatch.setattr(V8, "_distill_call", fake)
    V8.distill_events([("A", "I went hiking last Friday")], "2023-07-23",
                      ["D1:1"])
    assert "2023-07-21 星期五/Fri" in captured["sys"]
    assert "日期换算的唯一权威" in captured["sys"]
    assert "禁止脱离日历心算" in captured["sys"]
    assert "日期和原相对时间词必须并存" in captured["sys"]


def test_v8_distill_illegal_date_uses_fallback(monkeypatch):
    captured = {}

    def fake(messages, phase="", max_retry=6, model=None):
        captured["sys"] = messages[0]["content"]
        return '{"events":[]}'

    monkeypatch.setattr(V8, "_distill_call", fake)
    V8.distill_events([("A", "recently")], "未知日期", ["D1:1"])
    assert "本次无日历" in captured["sys"]
    assert "禁止造日期" in captured["sys"]


def test_per_turn_mode(monkeypatch):
    monkeypatch.setenv("NATIVEMEM_V9_SCRIBE_MODE", "per_turn")
    calls = []

    def fake(messages, phase="", max_retry=6, model=None):
        calls.append(messages[1]["content"])
        return '{"notes":[{"line":1,"note":"note-" + str(len(calls))}]}'.replace(
            '" + str(len(calls)) + "', str(len(calls)))

    monkeypatch.setattr(V8, "_distill_call", fake)
    out = V8.transcribe_session([("A", "hi"), ("B", "I adopted a dog"), ("A", "cool")],
                                ["D1:1", "D1:2", "D1:3"], "2023-07-23")
    assert len(calls) == 3                       # 一句一次调用
    assert "（上下文）A: hi" in calls[1]          # 第二句带第一句当上下文
    assert "（上下文）" not in calls[0]           # 第一句无上下文
    assert [d for d, _ in out] == ["D1:1", "D1:2", "D1:3"]


def test_per_turn_fallback_keeps_sentence(monkeypatch):
    monkeypatch.setenv("NATIVEMEM_V9_SCRIBE_MODE", "per_turn")
    monkeypatch.setattr(V8, "_distill_call", lambda *a, **k: "垃圾输出")
    out = V8.transcribe_session([("A", "hello there")], ["D1:1"], "2023-07-23")
    assert out == [("D1:1", "A: hello there")]   # 解析不出→原句兜底不丢句
