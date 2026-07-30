"""v8.6 提速：去重+分节合成一次调用（tidy_topic_file）、整理并行、答题并行。
全 mock LLM，不打真实 API。"""
import json
import os
import src.v8_memory as V8


class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()


def _fake(monkeypatch, content):
    """单一固定返回（所有文件用同一方案）。"""
    monkeypatch.setattr(V8.client.chat.completions, "create",
                        lambda **k: _FakeResp(content))
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)


def _seed(d, name="Caroline/adoption", n=9):
    for i in range(n):
        V8.write_events(d, [
            {"when": f"2023-05-{7+i:02d}", "summary": f"s{i}",
             "summary_inline": f"Caroline did thing {i} [D2:{i}]",
             "dia_ids": [f"D2:{i}"], "topic": name}])
    return os.path.join(d, "topics", "Caroline", "adoption.md")


def _content_lines(text):
    return [ln for ln in text.splitlines()
            if ln.strip() and not ln.startswith("#")]


def _dia_ids(text):
    return set(V8._DIA_RE.findall(text))


# ---- 正常路径：先合并后分节都生效 ----

def test_combined_merge_then_sections(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed(d, n=9)
    before_ids = _dia_ids(open(p).read())
    # 把 [1] merge 进 [2]（2 存活），其余 8 行分两节；sections 用合并前编号，
    # 引用了被 merge 掉的 [1]（应被静默跳过），[2] 带上 D2:1 + D2:0。
    _fake(monkeypatch,
          '{"groups":[{"keep":2,"merge":[1]}],'
          '"sections":[{"heading":"Early","lines":[2,1,3,4,5]},'
          '{"heading":"Later","lines":[6,7,8,9]}]}')
    assert V8.tidy_topic_file(d, {"Caroline/adoption"}) == 1
    body = open(p).read()
    assert "## Early" in body and "## Later" in body
    # 合并生效：8 行（9 - 1 被并）
    assert len(_content_lines(body)) == 8
    # dia_id 并集不变（合并只搬引用）
    assert _dia_ids(body) == before_ids
    # keep 行（原 [2]，thing 1）吸收了被并行 [1] 的 D2:0
    keep_line = [l for l in _content_lines(body) if "thing 1" in l][0]
    assert "D2:0" in keep_line and "D2:1" in keep_line


def test_combined_no_merge_pure_sections(tmp_path, monkeypatch):
    # groups 为空 → 纯分节，行逐字不动、多重集合相等。
    d = str(tmp_path)
    p = _seed(d, n=9)
    before = sorted(_content_lines(open(p).read()))
    _fake(monkeypatch,
          '{"groups":[],"sections":[{"heading":"All","lines":'
          + str(list(range(1, 10))) + '}]}')
    assert V8.tidy_topic_file(d, {"Caroline/adoption"}) == 1
    body = open(p).read()
    assert sorted(_content_lines(body)) == before


# ---- 行号映射：漏引用被删/未引用存活行进末尾 `## 未整理` ----

def test_line_mapping_leftover_bucket(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed(d, n=9)
    before_ids = _dia_ids(open(p).read())
    # 合并 [9] 进 [8]；sections 只引用 [1..3]，其余存活行（4,5,6,7,8）无引用
    # → 全部进末尾 `## 未整理`，一条不丢。
    _fake(monkeypatch,
          '{"groups":[{"keep":8,"merge":[9]}],'
          '"sections":[{"heading":"X","lines":[1,2,3]}]}')
    assert V8.tidy_topic_file(d, {"Caroline/adoption"}) == 1
    body = open(p).read()
    assert V8._UNSECTIONED_HEADING in body
    # 合并后 8 行全部落盘（3 在 X + 5 在未整理）
    assert len(_content_lines(body)) == 8
    assert _dia_ids(body) == before_ids


def test_line_mapping_reference_to_merged_line_ignored(tmp_path, monkeypatch):
    # sections 引用被 merge 掉的行号不报错，只是跳过它。
    d = str(tmp_path)
    p = _seed(d, n=9)
    _fake(monkeypatch,
          '{"groups":[{"keep":1,"merge":[2]}],'
          # 引用了被删的 [2] 和越界的 [99]（都跳过），其余存活行进未整理
          '"sections":[{"heading":"X","lines":[1,2,99,3]}]}')
    assert V8.tidy_topic_file(d, {"Caroline/adoption"}) == 1
    body = open(p).read()
    assert len(_content_lines(body)) == 8      # 9 - 1 合并，全部保留


# ---- merge 守卫失败 → 整个文件放弃 ----

def test_merge_guard_fail_keeps_original(tmp_path, monkeypatch):
    # keep 与 merge 重叠（跨组重复）→ 合并守卫拒绝 → 整个 tidy 放弃。
    d = str(tmp_path)
    p = _seed(d, n=9)
    before = open(p).read()
    _fake(monkeypatch,
          '{"groups":[{"keep":1,"merge":[2]},{"keep":2,"merge":[3]}],'
          '"sections":[{"heading":"X","lines":[1]}]}')
    assert V8.tidy_topic_file(d, {"Caroline/adoption"}) == 0
    assert open(p).read() == before


def test_merge_out_of_range_keeps_original(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed(d, n=9)
    before = open(p).read()
    _fake(monkeypatch,
          '{"groups":[{"keep":1,"merge":[99]}],'
          '"sections":[{"heading":"X","lines":[1]}]}')
    assert V8.tidy_topic_file(d, {"Caroline/adoption"}) == 0
    assert open(p).read() == before


# ---- sections 守卫失败 → 整个文件放弃 ----

def test_sections_missing_keeps_original(tmp_path, monkeypatch):
    # sections 字段缺失 → 放弃（哪怕 groups 合法）。
    d = str(tmp_path)
    p = _seed(d, n=9)
    before = open(p).read()
    _fake(monkeypatch, '{"groups":[{"keep":1,"merge":[2]}]}')
    assert V8.tidy_topic_file(d, {"Caroline/adoption"}) == 0
    assert open(p).read() == before


def test_bad_json_keeps_original(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed(d, n=9)
    before = open(p).read()
    _fake(monkeypatch, "not json at all")
    assert V8.tidy_topic_file(d, {"Caroline/adoption"}) == 0
    assert open(p).read() == before


def test_sections_not_list_keeps_original(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed(d, n=9)
    before = open(p).read()
    _fake(monkeypatch, '{"groups":[],"sections":"oops"}')
    assert V8.tidy_topic_file(d, {"Caroline/adoption"}) == 0
    assert open(p).read() == before


# ---- 触发阈值与拆开版一致 ----

def test_below_floor_not_triggered(tmp_path, monkeypatch):
    d = str(tmp_path)
    _seed(d, n=3)                                 # 3 < 8
    _fake(monkeypatch,
          '{"groups":[],"sections":[{"heading":"X","lines":[1,2,3]}]}')
    assert V8.tidy_topic_file(d, {"Caroline/adoption"}) == 0
    # final 仍触发
    assert V8.tidy_topic_file(d, {"Caroline/adoption"}, final=True) == 1


# ---- 整理并行：2 文件 mock 并发结果 == 串行 ----

def test_concurrent_tidy_matches_serial(tmp_path, monkeypatch):
    import src.adapters.run_nativemem as R

    def build(root):
        d = str(root)
        for name in ("A", "B"):
            for i in range(9):
                V8.write_events(d, [
                    {"when": f"2023-05-{7+i:02d}", "summary": f"{name}{i}",
                     "summary_inline": f"{name} did thing {i} [D2:{i}]",
                     "dia_ids": [f"D2:{i}"], "topic": name}])
        return d

    plan = ('{"groups":[{"keep":2,"merge":[1]}],'
            '"sections":[{"heading":"H","lines":[2,3,4,5,6,7,8,9]}]}')

    ser_root = tmp_path / "ser"; par_root = tmp_path / "par"
    ser = build(ser_root); par = build(par_root)
    _fake(monkeypatch, plan)

    # 串行（并发数 1）
    monkeypatch.setenv("NATIVEMEM_V8_CONCURRENCY", "1")
    monkeypatch.setenv("NATIVEMEM_V8_TIDY_COMBINED", "on")
    R._v8_tidy(ser, {"A", "B"}, sections_on=True, article_on=False)

    # 并发（并发数 8）
    monkeypatch.setenv("NATIVEMEM_V8_CONCURRENCY", "8")
    R._v8_tidy(par, {"A", "B"}, sections_on=True, article_on=False)

    for name in ("A", "B"):
        sp = os.path.join(ser, "topics", f"{name}.md")
        pp = os.path.join(par, "topics", f"{name}.md")
        assert open(sp).read() == open(pp).read()


# ---- 答题并行：结果按题序写回 ----

def test_answer_concurrency_preserves_order(tmp_path, monkeypatch):
    import src.adapters.run_nativemem as R
    import time

    qas = [{"question": f"q{i}?", "answer": f"a{i}", "category": 1}
           for i in range(12)]

    # 乱序完成（后面的题先返回），验证 ex.map 仍按题序收集。
    def fake_collect_answer(q, md, ti, max_rounds=None):
        idx = int(q[1:-1])                         # "q7?" -> 7
        time.sleep((12 - idx) * 0.002)             # 后面的题睡得短，先完成
        return [{"text": f"m{idx}", "date": ""}], 1, f"ans{idx}"

    monkeypatch.setattr(R, "_collect_and_answer_v8", fake_collect_answer)
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v8")
    monkeypatch.setenv("NATIVEMEM_V8_SINGLE", "1")
    monkeypatch.setenv("NATIVEMEM_V8_CONCURRENCY", "8")

    data = [{"conversation": {}, "qa": qas}]
    out = str(tmp_path / "out.json")
    monkeypatch.setattr(R, "DATA_PATH", _fake_data_file(tmp_path, data))
    monkeypatch.setattr(__import__("sys"), "argv",
                        ["run_nativemem.py", "--sample", "0", "--output", out,
                         "--memory-dir", str(tmp_path)])
    R.main()

    recs = json.load(open(out))
    qrecs = [r for r in recs if r["question_id"].startswith("s0_q")]
    # 题序 0..11 严格保持
    assert [r["question_id"] for r in qrecs] == [f"s0_q{i}" for i in range(12)]
    assert [r["answer"] for r in qrecs] == [f"ans{i}" for i in range(12)]


def _fake_data_file(tmp_path, data):
    p = tmp_path / "data.json"
    p.write_text(json.dumps(data))
    return str(p)
