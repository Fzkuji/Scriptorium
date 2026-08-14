import os
import src.adapters.run_nativemem as R
import src.v8_memory as V8

def test_collect_v8_uses_read_original_tool(tmp_path, monkeypatch):
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when": "2023-05-07", "summary": "去支持小组",
                         "dia_ids": ["D1:3"], "topic": "support"}])
    conv = {"session_1": [
        {"speaker": "Caroline", "dia_id": "D1:3", "text": "I went to a LGBTQ support group"}]}
    idx = V8.build_turn_index(conv)

    calls = {"n": 0}
    class _Msg:
        def __init__(self, content, tool_calls=None):
            self.content = content; self.tool_calls = tool_calls
    class _Resp:
        def __init__(self, msg):
            self.choices = [type("C", (), {"message": msg})()]
            self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()
    def fake_create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            # 模型调 read_original，必填 dia_ids
            tc = type("T", (), {"id": "1", "function": type("F", (), {
                "name": "read_original",
                "arguments": '{"dia_ids": ["D1:3"]}'})()})()
            return _Resp(_Msg("", [tc]))
        return _Resp(_Msg("done"))  # 第二轮无 tool_call 收尾
    monkeypatch.setattr(R.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(R, "log_usage", lambda *a, **k: None)

    mems, steps = R._collect_v8("去过支持小组吗", d, idx)
    joined = " ".join(m.get("text", "") for m in mems)
    assert "LGBTQ support group" in joined   # read_original 工具回原文取到 D1:3


def _mk_tc(tc_id, name, arguments):
    return type("T", (), {"id": tc_id, "function": type("F", (), {
        "name": name, "arguments": arguments})()})()


class _Msg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Resp:
    def __init__(self, msg):
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()


def test_collect_v8_handles_two_tool_calls_in_one_round(tmp_path, monkeypatch):
    # 一轮里模型同时吐两个 tool_call（一个 bash grep + 一个 read_original）。
    # API 契约要求：每个 tool_call_id 都要有对应的 tool 消息回应，且顺序、
    # id 要对上，否则下一轮请求会被 API 拒绝（tool_call 没有匹配 response）。
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when": "2023-05-07", "summary": "去支持小组",
                         "dia_ids": ["D1:3"], "topic": "support"}])
    conv = {"session_1": [
        {"speaker": "Caroline", "dia_id": "D1:3", "text": "I went to a LGBTQ support group"}]}
    idx = V8.build_turn_index(conv)

    calls = {"n": 0}

    def fake_create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            tc_bash = _mk_tc("1", "bash", '{"command": "grep -r 支持小组 topics/"}')
            tc_read = _mk_tc("2", "read_original", '{"dia_ids": ["D1:3"]}')
            return _Resp(_Msg("", [tc_bash, tc_read]))
        # 第二轮校验上一轮两个 tool_call 都被正确回应了
        msgs = kw["messages"]
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert {m["tool_call_id"] for m in tool_msgs} == {"1", "2"}
        return _Resp(_Msg("done"))  # 收尾，无 tool_call

    monkeypatch.setattr(R.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(R, "log_usage", lambda *a, **k: None)

    mems, steps = R._collect_v8("去过支持小组吗", d, idx)
    # 两个 tool_call 都被回答了，循环没有因为 API 契约错误而中断
    assert calls["n"] == 2
    joined = " ".join(m.get("text", "") for m in mems)
    assert "LGBTQ support group" in joined


def test_collect_v8_coerces_bare_string_dia_ids(tmp_path, monkeypatch):
    # 模型把 dia_ids 写成裸字符串 "D1:3" 而不是 ["D1:3"]（常见的模型输出松散）。
    # 代码必须把它包成单元素列表，而不是逐字符遍历字符串（那样会把 "D1:3"
    # 拆成 'D','1',':','3' 四个无效 id，检索不到任何原文）。
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when": "2023-05-07", "summary": "去支持小组",
                         "dia_ids": ["D1:3"], "topic": "support"}])
    conv = {"session_1": [
        {"speaker": "Caroline", "dia_id": "D1:3", "text": "I went to a LGBTQ support group"}]}
    idx = V8.build_turn_index(conv)

    calls = {"n": 0}

    def fake_create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            tc = _mk_tc("1", "read_original", '{"dia_ids": "D1:3"}')  # 字符串，非列表
            return _Resp(_Msg("", [tc]))
        return _Resp(_Msg("done"))

    monkeypatch.setattr(R.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(R, "log_usage", lambda *a, **k: None)

    mems, steps = R._collect_v8("去过支持小组吗", d, idx)
    joined = " ".join(m.get("text", "") for m in mems)
    assert "LGBTQ support group" in joined  # 字符串被 coerce 成 [str]，命中 D1:3


def test_collect_v8_empty_and_invalid_dia_ids_no_crash(tmp_path, monkeypatch):
    # 空列表和不存在的 dia_id 都不应崩，也不应产出记忆；循环要正常收尾。
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when": "2023-05-07", "summary": "去支持小组",
                         "dia_ids": ["D1:3"], "topic": "support"}])
    conv = {"session_1": [
        {"speaker": "Caroline", "dia_id": "D1:3", "text": "I went to a LGBTQ support group"}]}
    idx = V8.build_turn_index(conv)

    calls = {"n": 0}

    def fake_create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            tc = _mk_tc("1", "read_original", '{"dia_ids": []}')  # 空列表
            return _Resp(_Msg("", [tc]))
        if calls["n"] == 2:
            tc = _mk_tc("2", "read_original", '{"dia_ids": ["D9:99"]}')  # 不存在
            return _Resp(_Msg("", [tc]))
        return _Resp(_Msg("done"))

    monkeypatch.setattr(R.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(R, "log_usage", lambda *a, **k: None)

    mems, steps = R._collect_v8("去过支持小组吗", d, idx)
    assert mems == []           # 空/无效 dia_ids 都不产出记忆
    assert calls["n"] == 3      # 循环正常走完三轮（两次工具调用 + 收尾）后结束
    assert steps == 3


def test_collect_v8_grep_miss_no_read_original_returns_empty(tmp_path, monkeypatch):
    # 模型只 grep 且 grep 不命中（输出里没有任何 dia_id），也从不调 read_original
    # —— memories 必须是空列表，steps 受 max_rounds 限制（不无限循环）。
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when": "2023-05-07", "summary": "support group",
                         "dia_ids": ["D1:3"], "topic": "support"}])
    conv = {"session_1": [
        {"speaker": "Caroline", "dia_id": "D1:3", "text": "I went to a LGBTQ support group"}]}
    idx = V8.build_turn_index(conv)

    calls = {"n": 0}

    def fake_create(**kw):
        calls["n"] += 1
        # grep 一个记忆库里不存在的词 → 输出为空 → 没有 dia_id 可兜底
        tc = _mk_tc(str(calls["n"]), "bash", '{"command": "grep -r NONEXISTENTZZZ topics/"}')
        return _Resp(_Msg("", [tc]))

    monkeypatch.setattr(R.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(R, "log_usage", lambda *a, **k: None)

    mems, steps = R._collect_v8("q", d, idx, max_rounds=4)
    assert mems == []
    assert steps == 4           # 被 max_rounds 卡住，没有无限循环


def test_collect_v8_grep_hit_fallback_without_read_original(tmp_path, monkeypatch):
    # 兜底：模型 grep 命中了事件行（输出含 [D1:3]）但忘了调 read_original 就收尾。
    # 代码应从 grep 输出里扫到 dia_id，兜底回原文，不让已找到的结果白丢。
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when": "2023-05-07", "summary": "support group",
                         "dia_ids": ["D1:3"], "topic": "support"}])
    conv = {"session_1": [
        {"speaker": "Caroline", "dia_id": "D1:3", "text": "I went to a LGBTQ support group"}]}
    idx = V8.build_turn_index(conv)

    calls = {"n": 0}

    def fake_create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            # grep 命中英文事件行（其输出含 · [D1:3]）
            tc = _mk_tc("1", "bash", '{"command": "grep -rn support ."}')
            return _Resp(_Msg("", [tc]))
        return _Resp(_Msg("done"))  # 收尾，从不调 read_original

    monkeypatch.setattr(R.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(R, "log_usage", lambda *a, **k: None)

    mems, steps = R._collect_v8("q", d, idx)
    joined = " ".join(m.get("text", "") for m in mems)
    assert "LGBTQ support group" in joined  # 兜底：grep 命中的 D1:3 被回原文




def test_v8_event_lines_lookup(tmp_path):
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when": "2023-05-07", "summary": "went to LGBTQ support group",
                         "dia_ids": ["D1:3"], "topic": "Caroline-support"}])
    line = R._v8_event_lines(d, ["D1:3"])
    assert "went to LGBTQ support group" in line   # 精准事件行摘要
    assert "D1:3" in line
    assert R._v8_event_lines(d, ["D9:99"]) == ""


def test_v8_event_lines_matches_article_sentence(tmp_path):
    # 文章化后引用内联在自由文体句子里（无行首日期前缀）——也要能反查命中
    d = str(tmp_path / "mem")
    import os as _os
    _os.makedirs(_os.path.join(d, "topics"), exist_ok=True)
    with open(_os.path.join(d, "topics", "adoption.md"), "w") as f:
        f.write("## Caroline's adoption\n"
                "On 2023-05-25, Caroline decided to adopt [D2:3], "
                "visited an agency [D2:8].\n")
    line = R._v8_event_lines(d, ["D2:8"])
    assert "visited an agency" in line


def test_collect_v8_includes_event_line_not_just_original(tmp_path, monkeypatch):
    # 方案A：answerer 收到的 memory 里既有精准事件行摘要，又有原文
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when": "2023-05-07", "summary": "went to LGBTQ support group",
                         "dia_ids": ["D1:3"], "topic": "Caroline-support"}])
    conv = {"session_1": [{"speaker": "Caroline", "dia_id": "D1:3",
                           "text": "I went there yesterday it was amazing"}]}
    idx = V8.build_turn_index(conv)

    def fake_create(**kw):
        if not any(m.get("role") == "tool" for m in kw["messages"]):
            return _Resp(_Msg("", [_mk_tc("1", "read_original", '{"dia_ids": ["D1:3"]}')]))
        return _Resp(_Msg("done"))
    monkeypatch.setattr(R.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(R, "log_usage", lambda *a, **k: None)

    mems, _ = R._collect_v8("when support group", d, idx)
    joined = " ".join(m.get("text", "") for m in mems)
    assert "went to LGBTQ support group" in joined   # 事件行摘要在
    assert "I went there yesterday" in joined          # 原文也在


def test_collect_and_answer_v8_single_model(tmp_path, monkeypatch):
    # 方案B：单模型检索完直接给 <answer>，返回 (memories, steps, answer)
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when": "2023-05-07", "summary": "went to LGBTQ support group",
                         "dia_ids": ["D1:3"], "topic": "Caroline-support"}])
    conv = {"session_1": [{"speaker": "Caroline", "dia_id": "D1:3",
                           "text": "I went there on May 7"}]}
    idx = V8.build_turn_index(conv)

    def fake_create(**kw):
        if not any(m.get("role") == "tool" for m in kw["messages"]):
            return _Resp(_Msg("", [_mk_tc("1", "read_original", '{"dia_ids": ["D1:3"]}')]))
        return _Resp(_Msg("Based on memory <answer>May 7, 2023</answer>"))
    monkeypatch.setattr(R.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(R, "log_usage", lambda *a, **k: None)

    mems, steps, answer = R._collect_and_answer_v8("when support group", d, idx)
    assert answer == "May 7, 2023"          # 从 <answer> 抽取
    joined = " ".join(m.get("text", "") for m in mems)
    assert "went to LGBTQ support group" in joined   # 检索到的也在


def test_collect_and_answer_forces_tools_disabled_finalization(tmp_path, monkeypatch):
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when": "2023-05-07", "summary": "went hiking",
                         "dia_ids": ["D1:1"], "topic": "hiking"}])
    conv = {"session_1": [{"speaker": "A", "dia_id": "D1:1",
                            "text": "I went hiking on May 7"}]}
    idx = V8.build_turn_index(conv)
    calls = []

    def fake_create(**kw):
        calls.append(kw)
        if len(calls) <= 2:
            return _Resp(_Msg("", [_mk_tc(
                str(len(calls)), "read_original", '{"dia_ids": ["D1:1"]}')]))
        assert "tools" not in kw
        return _Resp(_Msg("<answer>May 7, 2023</answer>"))

    monkeypatch.setattr(R.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(R, "log_usage", lambda *a, **k: None)
    _mems, steps, answer = R._collect_and_answer_v8(
        "when hiking", d, idx, max_rounds=2)
    assert steps == 3
    assert answer == "May 7, 2023"
