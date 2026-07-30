"""重复事件治理三层 + distill 结构准则的测试（层1 精确去重 / 层2 行合并 /
层3 语义话题合并 / 层4 prompt）。全 mock LLM，不打真实 API。"""
import json
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


def _timeline_lines(d):
    out = []
    tl = os.path.join(d, "timeline")
    for root, _dirs, files in os.walk(tl):
        for fn in files:
            if fn.endswith(".md"):
                with open(os.path.join(root, fn)) as f:
                    out += [l.rstrip("\n") for l in f if l.strip()]
    return out


# ---- 层1：timeline 写入路径精确去重 ----

def test_timeline_exact_dedup_hits(tmp_path):
    d = str(tmp_path)
    # 同 dia_id 集合 + 仅标点/大小写/空格差异 → 视为同一事实，只留一行
    V8.write_events(d, [{"when": "2023-05-07", "summary": "Caroline adopted a dog",
                         "dia_ids": ["D1:1"], "topic": "pets"}])
    V8.write_events(d, [{"when": "2023-05-07", "summary": "caroline   adopted a  dog.",
                         "dia_ids": ["D1:1"], "topic": "pets"}])
    tl = _timeline_lines(d)
    assert len(tl) == 1


def test_timeline_dedup_miss_on_different_ids(tmp_path):
    d = str(tmp_path)
    # 文本相同但 dia_id 集合不同 → 不是同一事实，两行都写
    V8.write_events(d, [{"when": "2023-05-07", "summary": "same text",
                         "dia_ids": ["D1:1"], "topic": "pets"}])
    V8.write_events(d, [{"when": "2023-05-07", "summary": "same text",
                         "dia_ids": ["D1:2"], "topic": "pets"}])
    assert len(_timeline_lines(d)) == 2


def test_timeline_dedup_miss_on_different_text(tmp_path):
    d = str(tmp_path)
    V8.write_events(d, [{"when": "2023-05-07", "summary": "adopted a dog",
                         "dia_ids": ["D1:1"], "topic": "pets"}])
    V8.write_events(d, [{"when": "2023-05-07", "summary": "adopted a cat",
                         "dia_ids": ["D1:1"], "topic": "pets"}])
    assert len(_timeline_lines(d)) == 2


# ---- 层2：topic 文件内同义行合并 ----

def _seed_lines(d, name, rows):
    """rows: [(when, inline_text_with_[Dx:y])]，逐条写成未分节的原始行。"""
    for when, inline in rows:
        dia = V8._DIA_RE.findall(inline)
        V8.write_events(d, [{"when": when, "summary": inline, "summary_inline": inline,
                             "dia_ids": dia, "topic": name}])
    return os.path.join(d, "topics", *name.split("/")) + ".md"


def test_merge_lines_normal(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_V8_MERGE_LINES", "on")
    # 合并与分节同阈值同频（未归节行 ≥ REWRITE_MIN 才触发）；测试降到 3 行即触发
    monkeypatch.setenv("NATIVEMEM_V8_REWRITE_MIN", "3")
    d = str(tmp_path)
    # _append_sorted 按日期排序，落盘后编号：[1]=puppy(05-07) [2]=dog(05-08) [3]=job(05-09)
    p = _seed_lines(d, "pets", [
        ("2023-05-08", "Caroline adopted a dog [D1:1]"),
        ("2023-05-07", "Caroline got a puppy [D1:2]"),
        ("2023-05-09", "Caroline started a job [D1:3]")])
    # 模型判 dog、puppy 同一事实，keep=2(dog) merge=[1](puppy)
    _fake(monkeypatch, json.dumps({"groups": [{"keep": 2, "merge": [1]}]}))
    n = V8.merge_duplicate_lines(d, {"pets"})
    assert n == 1
    body = open(p).read()
    lines = [l for l in body.splitlines() if l.strip()]
    assert len(lines) == 2                       # 3 行合成 2 行
    # keep 行拿到两个引用的并集
    kept = next(l for l in lines if "adopted a dog" in l)
    assert "D1:1" in kept and "D1:2" in kept
    # 组内最早日期上位（puppy 的 05-07 早于 keep dog 的 05-08）
    assert kept.startswith("[2023-05-07]")


def test_merge_lines_out_of_range_guard_fails(tmp_path):
    content = ["[2023-05-08] a [D1:1]", "[2023-05-07] b [D1:2]"]
    out = V8._apply_line_merges(json.dumps({"groups": [{"keep": 1, "merge": [9]}]}), content)
    assert out is None                           # 行号 9 越界 → 放弃


def test_merge_lines_dia_union_guard_fails(tmp_path, monkeypatch):
    # 直接验证并集守卫：patch _DIA_RE 让 keep 行不再采集/追加 merge 行的引用，
    # 模拟"合并丢了 dia_id"这一坏路径，守卫必须拦下、放弃整文件合并。
    content = ["[2023-05-08] a [D1:1]", "[2023-05-07] b [D1:2]"]
    real = V8._DIA_RE
    calls = {"n": 0}

    class _Once:
        def findall(self, s):
            # 第一次（before_ids 全量扫描）正常返回，之后只认 keep 行那个 id，
            # 于是追加逻辑漏掉 D1:2 → after ≠ before → 守卫触发
            calls["n"] += 1
            if calls["n"] == 1:
                return real.findall(s)
            return [x for x in real.findall(s) if x == "D1:1"]

    monkeypatch.setattr(V8, "_DIA_RE", _Once())
    out = V8._apply_line_merges(json.dumps({"groups": [{"keep": 1, "merge": [2]}]}), content)
    assert out is None                           # dia_id 并集不等 → 放弃


def test_merge_lines_overlap_guard_fails(tmp_path):
    content = ["[2023-05-08] a [D1:1]", "[2023-05-07] b [D1:2]", "[2023-05-06] c [D1:3]"]
    # 1 号既是 g1 的 keep 又是 g2 的 merge → 跨组重叠，放弃
    out = V8._apply_line_merges(
        json.dumps({"groups": [{"keep": 1, "merge": [2]}, {"keep": 3, "merge": [1]}]}),
        content)
    assert out is None


def test_merge_lines_preserves_all_dia_ids(tmp_path):
    content = ["[2023-05-08] a [D1:1]", "[2023-05-07] b [D1:2]"]
    out = V8._apply_line_merges(json.dumps({"groups": [{"keep": 1, "merge": [2]}]}), content)
    assert out is not None
    joined = "\n".join(out)
    assert set(V8._DIA_RE.findall(joined)) == {"D1:1", "D1:2"}   # 并集不丢


# ---- 层3：语义话题合并提案验证 ----

def test_validate_rejects_nonexistent_path():
    plan = V8._validate_topic_merges(
        [{"into": "pets", "from": ["ghost"]}], ["pets", "animals"])
    assert plan == []                            # ghost 不存在 → 整条丢


def test_validate_rejects_self_merge():
    plan = V8._validate_topic_merges(
        [{"into": "pets", "from": ["pets"]}], ["pets", "animals"])
    assert plan == []                            # into == from


def test_validate_rejects_cycle():
    # A←B 且 B←A：into 与 from 集合相交 → 全拒（防链/环）
    plan = V8._validate_topic_merges(
        [{"into": "pets", "from": ["animals"]}, {"into": "animals", "from": ["pets"]}],
        ["pets", "animals"])
    assert plan == []


def test_validate_accepts_clean_merge():
    plan = V8._validate_topic_merges(
        [{"into": "animals", "from": ["pets"]}], ["pets", "animals"])
    assert plan == [("animals", ["pets"])]


def test_consolidate_semantic_merge_updates_backlinks(tmp_path, monkeypatch):
    d = str(tmp_path)
    _seed_lines(d, "pets", [("2023-05-07", "Caroline adopted a dog [D1:1]")])
    _seed_lines(d, "animals", [("2023-05-08", "Caroline loves cats [D1:2]")])
    _fake(monkeypatch, json.dumps({"merges": [{"into": "animals", "from": ["pets"]}]}))
    n = V8.consolidate_topic_files(d)
    assert n == 1
    assert not os.path.exists(os.path.join(d, "topics", "pets.md"))
    merged = open(os.path.join(d, "topics", "animals.md")).read()
    assert "dog" in merged and "cats" in merged
    # timeline 反链跟着改：不再指向被删的 pets.md
    tl = "\n".join(_timeline_lines(d))
    assert "→ topics/pets.md" not in tl
    assert "→ topics/animals.md" in tl


# ---- 层4：distill prompt 补文件结构准则 ----

def test_distill_prompt_bans_synonym_new_files():
    from src.v8_memory import _V8_DISTILL_PROMPT as P
    assert "复用已有路径" in P
    assert "禁止开同义新文件" in P
    assert "adopt" in P                          # 举例的近义禁开名
