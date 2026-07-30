import json
import src.v8_memory as V8

def test_number_chunk_adds_line_numbers_and_maps():
    turns = [("Caroline", "hi"), ("Melanie", "hello"), ("Caroline", "went to group")]
    numbered, line_map = V8._number_chunk(turns, ["D1:1", "D1:2", "D1:3"])
    assert "[1] Caroline: hi" in numbered
    assert "[3] Caroline: went to group" in numbered
    assert line_map == {1: "D1:1", 2: "D1:2", 3: "D1:3"}

def test_number_chunk_multiline_turn_stays_aligned():
    # 中间 turn（Melanie）的发言内含单个换行。结构化编号下，一个 turn 恒为
    # 一个编号块，turn 内部换行不会顶掉后续编号。
    turns = [("Caroline", "hi"), ("Melanie", "first line\nsecond line"),
             ("Caroline", "bye")]
    numbered, line_map = V8._number_chunk(turns, ["D1:1", "D1:2", "D1:3"])
    assert line_map == {1: "D1:1", 2: "D1:2", 3: "D1:3"}  # 3 个 turn
    assert "[3] Caroline: bye" in numbered  # 多行 turn 没有把后续编号顶掉

def test_number_chunk_internal_blank_line_turn_stays_aligned():
    # C1 回归：某个 turn 的 text 自身含【空行】。旧实现把拼接串按 "\n\n+"
    # 切块，无法区分 turn 分隔空行与 turn 内部空行，会把这个 turn 切成 2 块，
    # 使其后所有块号相对 dia_ids 整体错位——"line two" 会错误映射到 D1:3，
    # 而 "goodbye"/D1:3 丢失。结构化编号下一个 turn 恒为一块，对齐由构造保证。
    turns = [("A", "hello"), ("B", "line one\n\nline two"), ("A", "goodbye")]
    numbered, line_map = V8._number_chunk(turns, ["D1:1", "D1:2", "D1:3"])
    # 恰好 3 条映射，1↔D1:1 / 2↔D1:2 / 3↔D1:3
    assert line_map == {1: "D1:1", 2: "D1:2", 3: "D1:3"}
    # "line two" 属于第 2 块（D1:2），不是单独的第 3 块
    assert "[2] B: line one\n\nline two" in numbered
    # "goodbye" 仍在第 3 块（D1:3），没有被顶掉/丢失
    assert "[3] A: goodbye" in numbered
    # 显式：确保没有出现把 line two 单独编成一块顶掉 goodbye 的错位
    assert "[3] A: goodbye" in numbered and "[4]" not in numbered

def test_refs_to_dia_ids_maps_and_drops_invalid():
    lm = {1: "D1:1", 2: "D1:2", 3: "D1:3"}
    assert V8._refs_to_dia_ids([1, 3], lm) == ["D1:1", "D1:3"]
    assert V8._refs_to_dia_ids([3, 3, 2], lm) == ["D1:3", "D1:2"]  # 去重、保序
    assert V8._refs_to_dia_ids([99], lm) == []                     # 越界号丢弃
    assert V8._refs_to_dia_ids(["2"], lm) == ["D1:2"]              # 字符串号也接受

class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 10})()

def _install_fake(monkeypatch, content):
    monkeypatch.setattr(V8.client.chat.completions, "create", lambda **k: _FakeResp(content))
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)

def test_distill_uses_refs_and_code_fills_dia_ids(monkeypatch):
    payload = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "去支持小组", "refs": [3], "topic": "support"}]})
    _install_fake(monkeypatch, payload)
    evs = V8.distill_events([("Caroline", "hi"), ("Melanie", "hello"),
                             ("Caroline", "went to group")],
                            "2023-05-07", ["D1:1", "D1:2", "D1:3"])
    assert len(evs) == 1
    assert evs[0]["dia_ids"] == ["D1:3"]   # 代码从 refs=[3] 转出

def test_distill_drops_event_with_empty_refs(monkeypatch):
    payload = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "无引用事件", "refs": [], "topic": "t"}]})
    _install_fake(monkeypatch, payload)
    evs = V8.distill_events([("Caroline", "hi")], "2023-05-07", ["D1:1"])
    assert evs == []   # refs 空 → 丢弃
