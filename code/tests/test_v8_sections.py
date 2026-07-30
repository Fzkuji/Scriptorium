import os
import src.v8_memory as V8


class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()


def _fake(monkeypatch, content):
    monkeypatch.setattr(V8.client.chat.completions, "create",
                        lambda **k: _FakeResp(content))
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)


def _seed(d, name="Caroline/adoption", n=9):
    """写 n 条原子行到一个 topic 文件（无 `## ` 分节，全是未归节行）。"""
    for i in range(n):
        V8.write_events(d, [
            {"when": f"2023-05-{7+i:02d}", "summary": f"s{i}",
             "summary_inline": f"Caroline did thing {i} [D2:{i}]",
             "dia_ids": [f"D2:{i}"], "topic": name}])
    return os.path.join(d, "topics", "Caroline", "adoption.md")


def _content_lines(text):
    return [ln for ln in text.splitlines()
            if ln.strip() and not ln.startswith("#")]


# ---- 正常分节重建 ----

def test_organize_rebuilds_by_sections(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed(d, n=9)
    before = _content_lines(open(p).read())
    # 模型把 9 行分成两节（重排行序也允许）
    _fake(monkeypatch, '{"sections":[{"heading":"Early","lines":[3,1,2,4]},'
                       '{"heading":"Later","lines":[5,6,7,8,9]}]}')
    assert V8.organize_topic_sections(d, {"Caroline/adoption"}) == 1
    body = open(p).read()
    assert "## Early" in body and "## Later" in body
    # 内容行逐字保留（多重集合相等）
    assert sorted(_content_lines(body)) == sorted(before)


def test_organize_keeps_title_line(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed(d, n=9)
    body0 = open(p).read()
    with open(p, "w") as f:
        f.write("# Adoption topic\n" + body0)     # 加一级标题
    _fake(monkeypatch, '{"sections":[{"heading":"All","lines":'
                       + str(list(range(1, 10))) + '}]}')
    assert V8.organize_topic_sections(d, {"Caroline/adoption"}) == 1
    body = open(p).read()
    assert body.startswith("# Adoption topic")
    assert "## All" in body


# ---- 校验失败 → 放弃并保留原文件 ----

def test_multiset_mismatch_keeps_original(tmp_path, monkeypatch):
    # 覆盖全、无重复越界，但若代码错搬行导致文本变化，多重集合校验兜底。
    # 这里直接构造漏行的方案触发放弃（覆盖不全）。
    d = str(tmp_path)
    p = _seed(d, n=9)
    before = open(p).read()
    _fake(monkeypatch, '{"sections":[{"heading":"X","lines":[1,2,3]}]}')  # 只覆盖3行
    assert V8.organize_topic_sections(d, {"Caroline/adoption"}) == 0
    assert open(p).read() == before


def test_bad_json_keeps_original(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed(d, n=9)
    before = open(p).read()
    _fake(monkeypatch, "not json at all")
    assert V8.organize_topic_sections(d, {"Caroline/adoption"}) == 0
    assert open(p).read() == before


def test_duplicate_lineno_keeps_original(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed(d, n=9)
    before = open(p).read()
    _fake(monkeypatch, '{"sections":[{"heading":"X","lines":[1,1,2,3,4,5,6,7,8,9]}]}')
    assert V8.organize_topic_sections(d, {"Caroline/adoption"}) == 0
    assert open(p).read() == before


def test_out_of_range_lineno_keeps_original(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed(d, n=9)
    before = open(p).read()
    _fake(monkeypatch, '{"sections":[{"heading":"X","lines":'
                       '[1,2,3,4,5,6,7,8,99]}]}')
    assert V8.organize_topic_sections(d, {"Caroline/adoption"}) == 0
    assert open(p).read() == before


# ---- 触发判定 ----

def test_no_heading_file_triggers(tmp_path, monkeypatch):
    # 无 `## ` 文件：所有内容行都算未归节，≥ 阈值触发。
    d = str(tmp_path)
    p = _seed(d, n=9)                              # 9 >= min(8)
    assert V8._unsectioned_count(open(p).read()) == 9
    _fake(monkeypatch, '{"sections":[{"heading":"X","lines":'
                       + str(list(range(1, 10))) + '}]}')
    assert V8.organize_topic_sections(d, {"Caroline/adoption"}) == 1


def test_below_floor_not_triggered(tmp_path, monkeypatch):
    d = str(tmp_path)
    _seed(d, n=3)                                 # 3 < 8，不触发
    _fake(monkeypatch, '{"sections":[{"heading":"X","lines":[1,2,3]}]}')
    assert V8.organize_topic_sections(d, {"Caroline/adoption"}) == 0
    # final 强制整理仍触发
    assert V8.organize_topic_sections(d, {"Caroline/adoption"}, final=True) == 1


def test_appended_lines_are_unsectioned(tmp_path, monkeypatch):
    # 已分节文件追加新行 → 落进 `## 未整理` 节 → 被判为未归节。
    d = str(tmp_path)
    p = _seed(d, n=9)
    _fake(monkeypatch, '{"sections":[{"heading":"Old","lines":'
                       + str(list(range(1, 10))) + '}]}')
    assert V8.organize_topic_sections(d, {"Caroline/adoption"}) == 1
    # 整理后无未归节行
    assert V8._unsectioned_count(open(p).read()) == 0
    # 再追加 1 行：进 `## 未整理`
    V8.write_events(d, [{"when": "2023-06-01", "summary": "new",
                         "summary_inline": "Caroline did new thing [D3:1]",
                         "dia_ids": ["D3:1"], "topic": "Caroline/adoption"}])
    body = open(p).read()
    assert V8._UNSECTIONED_HEADING in body
    assert V8._unsectioned_count(body) == 1        # 新行被判为未归节


def test_build_hooks_sections_organize(tmp_path, monkeypatch):
    # build 主循环 session 末调 organize_topic_sections（SECTIONS=on 时），
    # 收尾再 final sweep 一次；SECTIONS=off 时不调。
    import src.adapters.run_nativemem as R
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v8")
    conv = {"session_1": [{"speaker": "C", "dia_id": "D1:1", "text": "hi"}],
            "session_1_date_time": "2023-05-07"}
    monkeypatch.setattr(V8, "distill_events",
        lambda turns, obs, dids, known_topics=None, **k: [
            {"when": "2023-05-07", "summary": "x", "dia_ids": ["D1:1"],
             "topic": "Caroline/adoption"}])
    calls = []
    monkeypatch.setattr(V8, "organize_topic_sections",
        lambda md, touched, final=False, **k: calls.append((set(touched), final)) or 0)

    monkeypatch.setenv("NATIVEMEM_V8_SECTIONS", "off")
    R.build_memory(conv, str(tmp_path / "m1"))
    assert calls == []

    monkeypatch.setenv("NATIVEMEM_V8_SECTIONS", "on")
    R.build_memory(conv, str(tmp_path / "m2"))
    assert calls == [({"Caroline/adoption"}, False), ({"Caroline/adoption"}, True)]


def test_link_dates_survive_organize(tmp_path, monkeypatch):
    # 日期链接在分节重排后照常保留（行文本逐字不动）。
    d = str(tmp_path)
    V8.write_events(d, [
        {"when": "2023-05-25", "summary": "s",
         "summary_inline": "Caroline decided on 2023-05-25 [D2:3]",
         "dia_ids": ["D2:3"], "topic": "adoption"}])
    p = os.path.join(d, "topics", "adoption.md")
    assert "[2023-05-25](timeline/2023/05/25.md)" in open(p).read()
    _fake(monkeypatch, '{"sections":[{"heading":"A","lines":[1]}]}')
    assert V8.organize_topic_sections(d, {"adoption"}, final=True) == 1
    assert "[2023-05-25](timeline/2023/05/25.md)" in open(p).read()
