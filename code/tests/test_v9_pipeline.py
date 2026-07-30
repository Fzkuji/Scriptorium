"""v9.0 两级建库流水线测试（transcribe 逐句转写 + compose 谱曲 + two_tier 接线）。
全程 mock LLM，默认 off 路径不调新函数，一律不打真实 API。"""
import json
import src.v8_memory as V8
import src.adapters.run_nativemem as R


class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 10})()


def _install_fake(monkeypatch, content):
    """单一固定回复（content 可为 str，或 callable(**kwargs)->str 做多轮不同回复）。"""
    def fake_create(**kwargs):
        c = content(**kwargs) if callable(content) else content
        return _FakeResp(c)
    monkeypatch.setattr(V8.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)


def _mk(n):
    turns = [(f"S{i}", f"line {i}") for i in range(1, n + 1)]
    dia_ids = [f"D1:{i}" for i in range(1, n + 1)]
    return turns, dia_ids


# ---------------- 第一级：transcribe_session ----------------

def test_transcribe_normal(monkeypatch):
    turns, dia_ids = _mk(3)
    notes = {"notes": [{"line": 1, "note": "note one"},
                       {"line": 2, "note": "note two"},
                       {"line": 3, "note": "note three"}]}
    _install_fake(monkeypatch, json.dumps(notes))
    out = V8.transcribe_session(turns, dia_ids, "2023-05-07")
    assert out == [("D1:1", "note one"), ("D1:2", "note two"), ("D1:3", "note three")]


def test_transcribe_missing_line_refilled(monkeypatch):
    # 首轮漏第 2 句，补转写轮补上
    turns, dia_ids = _mk(3)
    calls = {"n": 0}

    def reply(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:                       # 首轮缺 line 2
            return json.dumps({"notes": [{"line": 1, "note": "n1"},
                                         {"line": 3, "note": "n3"}]})
        return json.dumps({"notes": [{"line": 2, "note": "n2-refill"}]})
    _install_fake(monkeypatch, reply)
    out = V8.transcribe_session(turns, dia_ids, "2023-05-07")
    assert out == [("D1:1", "n1"), ("D1:2", "n2-refill"), ("D1:3", "n3")]
    assert calls["n"] == 2                         # 首轮 + 一轮补转


def test_transcribe_two_rounds_still_missing_falls_back_to_raw(monkeypatch):
    # 两轮补转仍缺 line 2 → 用原句原文 "S2: line 2" 兜底，绝不丢句
    turns, dia_ids = _mk(3)

    def reply(**kwargs):
        return json.dumps({"notes": [{"line": 1, "note": "n1"},
                                     {"line": 3, "note": "n3"}]})
    _install_fake(monkeypatch, reply)
    out = V8.transcribe_session(turns, dia_ids, "2023-05-07")
    assert out == [("D1:1", "n1"), ("D1:2", "S2: line 2"), ("D1:3", "n3")]
    assert len(out) == 3


def test_transcribe_bad_json_all_fallback(monkeypatch):
    # JSON 全坏，两轮都补不出 → 每句都用原句兜底，长度恒等
    turns, dia_ids = _mk(2)
    _install_fake(monkeypatch, "not json")
    out = V8.transcribe_session(turns, dia_ids, "2023-05-07", max_retry=1)
    assert out == [("D1:1", "S1: line 1"), ("D1:2", "S2: line 2")]


def test_transcribe_batches_over_limit(monkeypatch):
    # 句数 > NATIVEMEM_V9_SCRIBE_BATCH → 机械分批，每批一次调用
    monkeypatch.setenv("NATIVEMEM_V9_SCRIBE_BATCH", "2")
    turns, dia_ids = _mk(5)                        # 2 + 2 + 1 = 3 批
    calls = {"n": 0}

    def reply(**kwargs):
        calls["n"] += 1
        # 每批都完整回全（批内编号从 1 起），无缺行不触发补转
        user = kwargs["messages"][-1]["content"]
        nlines = sum(1 for ln in user.split("\n") if ln.strip())
        return json.dumps({"notes": [{"line": i, "note": f"b{calls['n']}-{i}"}
                                     for i in range(1, nlines + 1)]})
    _install_fake(monkeypatch, reply)
    out = V8.transcribe_session(turns, dia_ids, "2023-05-07")
    assert [d for d, _ in out] == dia_ids          # 顺序、覆盖完整
    assert len(out) == 5
    assert calls["n"] == 3                         # 3 批各一次首轮调用


def test_transcribe_one_sentence_multiple_notes(monkeypatch):
    # v9.0c：一句话陈述多个独立事实 → 同一 line 出多条笔记，展开成多个 (dia_id, note)
    turns, dia_ids = _mk(2)
    notes = {"notes": [{"line": 1, "note": "adopted a dog"},
                       {"line": 1, "note": "it means a lot to her"},
                       {"line": 1, "note": "reminds her of her late father"},
                       {"line": 2, "note": "n2"}]}
    _install_fake(monkeypatch, json.dumps(notes))
    out = V8.transcribe_session(turns, dia_ids, "2023-05-07")
    # 同一 dia_id D1:1 展开成 3 条，保序；D1:2 一条
    assert out == [("D1:1", "adopted a dog"),
                   ("D1:1", "it means a lot to her"),
                   ("D1:1", "reminds her of her late father"),
                   ("D1:2", "n2")]


def test_transcribe_at_least_one_per_line_no_false_missing(monkeypatch):
    # 覆盖判定是"每号至少一条"：line 1 拿了 2 条、line 2 拿了 1 条 → 无缺行，不触发补转
    turns, dia_ids = _mk(2)
    calls = {"n": 0}

    def reply(**kwargs):
        calls["n"] += 1
        return json.dumps({"notes": [{"line": 1, "note": "a1"},
                                     {"line": 1, "note": "a2"},
                                     {"line": 2, "note": "b1"}]})
    _install_fake(monkeypatch, reply)
    out = V8.transcribe_session(turns, dia_ids, "2023-05-07")
    assert calls["n"] == 1                          # 每号都有 → 不补转
    assert out == [("D1:1", "a1"), ("D1:1", "a2"), ("D1:2", "b1")]


def test_transcribe_prompt_keeps_soft_facts_and_anchors():
    # prompt 明确点名要保住的三类"排位靠后"事实与"一句多笔记"
    p = V8._V9_TRANSCRIBE_PROMPT
    assert "几个独立事实" in p and "几条笔记" in p     # 一句多笔记
    assert "第二/第三个事实" in p                     # 并列软事实
    assert "after the road trip" in p                # 事件锚点短语原样保留
    assert "引号内的原话" in p


def test_transcribe_empty_no_api(monkeypatch):
    called = {"n": 0}

    def fake_create(**kwargs):
        called["n"] += 1
        return _FakeResp("{}")
    monkeypatch.setattr(V8.client.chat.completions, "create", fake_create)
    assert V8.transcribe_session([], [], "2023-05-07") == []
    assert called["n"] == 0


# ---------------- 第二级：compose_events ----------------

def test_compose_normal_ledger_skip_and_two_events(monkeypatch):
    # 逐条记账：skip 一条 + 两个原子事实各成事件；编号经 line_map 映射回 dia_id
    monkeypatch.setenv("NATIVEMEM_V8_VERIFY", "off")
    notes = [("D2:3", "Caroline visited an adoption agency"),
             ("D2:4", "Melanie greeted Caroline"),
             ("D2:5", "Caroline liked its LGBTQ-friendly values"),
             ("D2:6", "Caroline bought a book")]
    payload = {
        "notes": [{"n": 2, "do": "skip"},          # 寒暄
                  {"n": 1, "do": "E1"}, {"n": 3, "do": "E1"},
                  {"n": 4, "do": "E2"}],
        "events": [
            {"id": "E1", "when": "2023-05-07",
             "summary": "Caroline visited an adoption agency [1] and liked its "
                        "LGBTQ-friendly values [3]", "topic": "Caroline/adoption"},
            {"id": "E2", "when": "2023-05-07",
             "summary": "Caroline bought a book [4]", "topic": "Caroline/reading"}]}
    _install_fake(monkeypatch, json.dumps(payload))
    evs = V8.compose_events(notes, "2023-05-07")
    assert len(evs) == 2
    by_topic = {e["topic"]: e for e in evs}
    e1 = by_topic["Caroline/adoption"]
    assert e1["when"] == "2023-05-07"
    assert e1["dia_ids"] == ["D2:3", "D2:5"]        # 认领编号 1→D2:3, 3→D2:5
    assert "[D2:3]" in e1["summary_inline"] and "[D2:5]" in e1["summary_inline"]
    assert "[1]" not in e1["summary"]               # plain 去标记
    assert by_topic["Caroline/reading"]["dia_ids"] == ["D2:6"]


def test_compose_dia_ids_union_of_claims(monkeypatch):
    # dia_ids = 认领并集：认领了 1、2、3，但 summary 只内联了 2 → dia_ids 仍含全部三条
    monkeypatch.setenv("NATIVEMEM_V8_VERIFY", "off")
    notes = [("D1:1", "Caroline chose an agency"),
             ("D1:2", "Caroline liked its values"),
             ("D1:3", "Caroline filled a form")]
    payload = {
        "notes": [{"n": 1, "do": "E1"}, {"n": 2, "do": "E1"}, {"n": 3, "do": "E1"}],
        "events": [{"id": "E1", "when": "2023-05-07",
                    "summary": "Caroline started adopting [2]",  # 只内联了 2
                    "topic": "Caroline/adoption"}]}
    _install_fake(monkeypatch, json.dumps(payload))
    evs = V8.compose_events(notes, "2023-05-07")
    assert len(evs) == 1
    assert evs[0]["dia_ids"] == ["D1:1", "D1:2", "D1:3"]   # 认领并集，比内联更全


def test_compose_smalltalk_all_skip(monkeypatch):
    # 全寒暄 → notes 全 skip，events 空 → 不入库
    monkeypatch.setenv("NATIVEMEM_V8_VERIFY", "off")
    notes = [("D1:1", "Melanie greeted Caroline"), ("D1:2", "Caroline said thanks")]
    _install_fake(monkeypatch, json.dumps(
        {"notes": [{"n": 1, "do": "skip"}, {"n": 2, "do": "skip"}], "events": []}))
    assert V8.compose_events(notes, "2023-05-07") == []


def test_compose_event_without_claim_dropped(monkeypatch):
    # 事件无人认领（notes 里没有 do 指向它）→ 丢弃
    monkeypatch.setenv("NATIVEMEM_V8_VERIFY", "off")
    notes = [("D1:1", "Caroline adopted a cat named Whiskers")]
    payload = {"notes": [{"n": 1, "do": "skip"}],
               "events": [{"id": "E1", "when": "2023-05-07",
                           "summary": "Caroline adopted a cat [1]",
                           "topic": "Caroline/pets"}]}   # E1 无人 do
    _install_fake(monkeypatch, json.dumps(payload))
    assert V8.compose_events(notes, "2023-05-07") == []


def test_compose_do_to_missing_event_tolerated(monkeypatch):
    # do 指向不存在的事件 id → 该编号算没交代；关 verify 时不触发修补，仅存在的事件入库
    monkeypatch.setenv("NATIVEMEM_V8_VERIFY", "off")
    notes = [("D1:1", "Caroline visited Paris"), ("D1:2", "Caroline bought a book")]
    calls = {"n": 0}

    def reply(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:                       # 编号 1 的 do 指向不存在的 E9
            return json.dumps({
                "notes": [{"n": 1, "do": "E9"}, {"n": 2, "do": "E2"}],
                "events": [{"id": "E2", "when": "2023-05-07",
                            "summary": "Caroline bought a book [2]",
                            "topic": "Caroline/reading"}]})
        # 修补轮：把编号 1 补进新事件
        return json.dumps({
            "notes": [{"n": 1, "do": "E3"}],
            "events": [{"id": "E3", "when": "2023-05-07",
                        "summary": "Caroline visited Paris [1]",
                        "topic": "Caroline/travel"}]})
    _install_fake(monkeypatch, reply)
    evs = V8.compose_events(notes, "2023-05-07")
    assert calls["n"] == 2                          # 首轮 + 记账修补一轮
    topics = {e["topic"] for e in evs}
    assert topics == {"Caroline/reading", "Caroline/travel"}
    assert any(e["dia_ids"] == ["D1:1"] for e in evs)


def test_compose_missing_number_triggers_repair(monkeypatch):
    # 首轮漏了编号 2（notes 只交代了 1）→ 记账修补一轮把 2 补进事件
    monkeypatch.setenv("NATIVEMEM_V8_VERIFY", "off")
    notes = [("D1:1", "Caroline visited Paris"), ("D1:2", "Caroline bought a book")]
    calls = {"n": 0}

    def reply(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:                       # notes 只有编号 1，缺 2
            return json.dumps({
                "notes": [{"n": 1, "do": "E1"}],
                "events": [{"id": "E1", "when": "2023-05-07",
                            "summary": "Caroline visited Paris [1]",
                            "topic": "Caroline/travel"}]})
        # 修补：编号 2 → 新事件 E2
        assert "2" in kwargs["messages"][-1]["content"]   # 缺号发回去了
        return json.dumps({
            "notes": [{"n": 2, "do": "E2"}],
            "events": [{"id": "E2", "when": "2023-05-07",
                        "summary": "Caroline bought a book [2]",
                        "topic": "Caroline/reading"}]})
    _install_fake(monkeypatch, reply)
    evs = V8.compose_events(notes, "2023-05-07")
    assert calls["n"] == 2
    assert {e["topic"] for e in evs} == {"Caroline/travel", "Caroline/reading"}


def test_compose_repair_still_missing_treated_as_skip(monkeypatch):
    # 修补后编号 2 仍没交代 → 按 skip 处理，不崩、不入库，只留 warning
    monkeypatch.setenv("NATIVEMEM_V8_VERIFY", "off")
    notes = [("D1:1", "Caroline visited Paris"), ("D1:2", "Caroline bought a book")]

    def reply(**kwargs):
        # 首轮与修补轮都只交代编号 1，编号 2 始终缺
        return json.dumps({
            "notes": [{"n": 1, "do": "E1"}],
            "events": [{"id": "E1", "when": "2023-05-07",
                        "summary": "Caroline visited Paris [1]",
                        "topic": "Caroline/travel"}]})
    _install_fake(monkeypatch, reply)
    evs = V8.compose_events(notes, "2023-05-07")
    assert len(evs) == 1                            # 只有 E1，编号 2 按 skip 落地
    assert evs[0]["dia_ids"] == ["D1:1"]


def test_compose_verify_refill(monkeypatch):
    # 记账齐全但漏了专名 Paris → verify_event_coverage 触发补谱一轮（记账修补不触发）
    notes = [("D1:1", "Caroline visited Paris"),
             ("D1:2", "Caroline bought a book")]
    calls = {"n": 0}

    def reply(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:                       # 记账齐全（1、2 都 skip 掉 Paris 那条内容）
            return json.dumps({
                "notes": [{"n": 1, "do": "skip"}, {"n": 2, "do": "E1"}],
                "events": [{"id": "E1", "when": "2023-05-07",
                            "summary": "Caroline bought a book [2]",
                            "topic": "Caroline/reading"}]})
        return json.dumps({                         # 补谱：Paris
            "notes": [{"n": 1, "do": "E2"}],
            "events": [{"id": "E2", "when": "2023-05-07",
                        "summary": "Caroline visited Paris [1]",
                        "topic": "Caroline/travel"}]})
    _install_fake(monkeypatch, reply)
    evs = V8.compose_events(notes, "2023-05-07")
    assert calls["n"] == 2                          # 首轮记账齐 → 跳修补；verify 补谱一轮
    topics = {e["topic"] for e in evs}
    assert topics == {"Caroline/reading", "Caroline/travel"}
    assert any(e["dia_ids"] == ["D1:1"] for e in evs)


def test_compose_bad_json_empty(monkeypatch):
    monkeypatch.setenv("NATIVEMEM_V8_VERIFY", "off")
    _install_fake(monkeypatch, "not json")
    assert V8.compose_events([("D1:1", "x")], "2023-05-07", max_retry=1) == []


def test_compose_when_backfilled_from_note_date(monkeypatch):
    # v9.0c：模型 when 给了观测日，但认领笔记文本里有绝对日期 → when 回填到笔记里的日期
    monkeypatch.setenv("NATIVEMEM_V8_VERIFY", "off")
    notes = [("D1:1", "Caroline went hiking on 2023-05-06"),
             ("D1:2", "Caroline also visited a museum on 2023-05-04")]
    payload = {
        "notes": [{"n": 1, "do": "E1"}, {"n": 2, "do": "E1"}],
        "events": [{"id": "E1", "when": "2023-05-07",   # 模型只给了观测日
                    "summary": "Caroline went hiking [1] and visited a museum [2]",
                    "topic": "Caroline/outings"}]}
    _install_fake(monkeypatch, json.dumps(payload))
    evs = V8.compose_events(notes, "2023-05-07")
    assert len(evs) == 1
    assert evs[0]["when"] == "2023-05-04"           # 认领笔记里最早的绝对日期

def test_compose_when_keeps_model_absolute_date(monkeypatch):
    # 模型自己给了 ≠观测日的绝对日期 → 照用，不被回填覆盖
    monkeypatch.setenv("NATIVEMEM_V8_VERIFY", "off")
    notes = [("D1:1", "Caroline went hiking on 2023-05-06")]
    payload = {"notes": [{"n": 1, "do": "E1"}],
               "events": [{"id": "E1", "when": "2023-05-06",
                           "summary": "Caroline went hiking [1]",
                           "topic": "Caroline/outings"}]}
    _install_fake(monkeypatch, json.dumps(payload))
    evs = V8.compose_events(notes, "2023-05-07")
    assert evs[0]["when"] == "2023-05-06"

def test_compose_when_year_only_not_backfilled(monkeypatch):
    # 笔记里只有年份 "in 2022"（非 YYYY-MM-DD）→ 代码不回填，仍是观测日（靠 prompt 管年份）
    monkeypatch.setenv("NATIVEMEM_V8_VERIFY", "off")
    notes = [("D1:1", "Caroline moved to Berlin in 2022")]
    payload = {"notes": [{"n": 1, "do": "E1"}],
               "events": [{"id": "E1", "when": "2023-05-07",
                           "summary": "Caroline moved to Berlin [1]",
                           "topic": "Caroline/relocation"}]}
    _install_fake(monkeypatch, json.dumps(payload))
    evs = V8.compose_events(notes, "2023-05-07")
    assert evs[0]["when"] == "2023-05-07"

def test_compose_prompt_atomic_event_discipline():
    # prompt 关键词断言：一个原子事实 / 同一事实的重复陈述 / 禁止胖事件 / 一句短句
    p = V8._V9_COMPOSE_PROMPT
    assert "一个事件 = 一个原子事实" in p
    assert "同一事实的重复陈述" in p
    assert "禁止把一段连续对话里的多个不同事实并进一个事件" in p
    assert "一句短句" in p and "禁止从句堆叠" in p
    # when 规则：优先取认领笔记里的绝对日期
    assert "when 用那个日期" in p

def test_compose_empty_notes_no_api(monkeypatch):
    called = {"n": 0}

    def fake_create(**kwargs):
        called["n"] += 1
        return _FakeResp("{}")
    monkeypatch.setattr(V8.client.chat.completions, "create", fake_create)
    assert V8.compose_events([], "2023-05-07") == []
    assert called["n"] == 0


# ---------------- 接线：two_tier 开关 ----------------

def _build_conv():
    return {
        "session_1": [
            {"speaker": "A", "dia_id": f"D1:{i}", "text": f"t{i}"}
            for i in range(1, 5)
        ],
        "session_1_date_time": "2023-05-07",
    }


def test_build_default_skips_v9(tmp_path, monkeypatch):
    # 默认（无 NATIVEMEM_V9_PIPELINE）→ 不调 transcribe/compose，走既有 distill
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v8")
    called = {"t": 0, "c": 0}
    monkeypatch.setattr(V8, "transcribe_session",
                        lambda *a, **k: called.__setitem__("t", called["t"] + 1) or [])
    monkeypatch.setattr(V8, "compose_events",
                        lambda *a, **k: called.__setitem__("c", called["c"] + 1) or [])
    monkeypatch.setattr(V8, "distill_events", lambda *a, **k: [])
    R.build_memory(_build_conv(), str(tmp_path / "m"))
    assert called == {"t": 0, "c": 0}


def test_build_two_tier_calls_new_pipeline(tmp_path, monkeypatch):
    # =two_tier → transcribe → compose 被调，且不走 distill；events 落盘
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v8")
    monkeypatch.setenv("NATIVEMEM_V9_PIPELINE", "two_tier")
    seen = {"n_turns": None, "distill": 0}

    def fake_transcribe(turns, dia_ids, obs, **k):
        seen["n_turns"] = len(turns)
        return [(d, f"note {d}") for d in dia_ids]

    def fake_compose(notes, obs, **k):
        assert len(notes) == 4                     # 整 session 笔记一次性见齐
        return [{"when": "2023-05-07", "summary": "s [D1:1]",
                 "summary_inline": "s [D1:1]", "dia_ids": ["D1:1"],
                 "topic": "misc"}]
    monkeypatch.setattr(V8, "transcribe_session", fake_transcribe)
    monkeypatch.setattr(V8, "compose_events", fake_compose)
    monkeypatch.setattr(V8, "distill_events",
                        lambda *a, **k: seen.__setitem__("distill", seen["distill"] + 1) or [])
    R.build_memory(_build_conv(), str(tmp_path / "m"))
    assert seen["n_turns"] == 4                     # 整 session 铺平后交给 transcribe
    assert seen["distill"] == 0                     # 新路径不碰 distill
    # events 落盘到 timeline
    import os
    assert os.path.exists(str(tmp_path / "m" / "timeline" / "2023" / "05" / "07.md"))
