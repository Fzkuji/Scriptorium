import os
import src.v8_memory as V8


def test_article_prompt_bans_date_recompute_and_word_change():
    # 文章重写三根因之一二：禁止推算日期、相对时间原词保留；关键细节词禁止同义替换/泛化
    P = V8._V8_ARTICLE_PROMPT
    assert "照抄" in P and "禁止推算" in P
    assert "原词保留" in P
    assert "last week" in P            # 相对时间原词示例
    assert "同义替换" in P and "泛化" in P
    assert "graceful" in P             # 描述词原词保留示例


def _seed_topic(d, name="Caroline/adoption"):
    # 3 raw lines, never articled → triggers rewrite (all-raw file, >=3 lines).
    V8.write_events(d, [
        {"when": "2023-05-25", "summary": "Caroline decided to adopt",
         "summary_inline": "Caroline decided to adopt [D2:3]",
         "dia_ids": ["D2:3"], "topic": name},
        {"when": "2023-05-25", "summary": "Caroline visited an agency",
         "summary_inline": "Caroline visited an agency [D2:8]",
         "dia_ids": ["D2:8"], "topic": name},
        {"when": "2023-05-25", "summary": "Caroline chose it for its values",
         "summary_inline": "Caroline chose it for its values [D2:11]",
         "dia_ids": ["D2:11"], "topic": name},
    ])
    return os.path.join(d, "topics", "Caroline", "adoption.md")


class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()


def _fake(monkeypatch, content):
    monkeypatch.setattr(V8.client.chat.completions, "create",
                        lambda **k: _FakeResp(content))
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)


GOOD_ARTICLE = """## Caroline's adoption journey

On 2023-05-25, Caroline decided to adopt [D2:3], visited an agency [D2:8],
and chose it for its values [D2:11].
"""


def test_rewrite_replaces_file_when_dia_ids_preserved(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed_topic(d)
    _fake(monkeypatch, GOOD_ARTICLE)
    n = V8.rewrite_topic_articles(d, {"Caroline/adoption"})
    assert n == 1
    body = open(p).read()
    assert "adoption journey" in body          # 文章化生效
    assert "[D2:3]" in body and "[D2:8]" in body


def test_rewrite_rejected_when_dia_id_lost(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed_topic(d)
    before = open(p).read()
    _fake(monkeypatch, "## Article\nCaroline decided to adopt [D2:3].\n")  # 丢了 D2:8
    n = V8.rewrite_topic_articles(d, {"Caroline/adoption"})
    assert n == 0
    assert open(p).read() == before            # 拒绝写回，保持原样


def test_rewrite_rejected_when_dia_id_fabricated(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed_topic(d)
    before = open(p).read()
    _fake(monkeypatch, GOOD_ARTICLE + "Later she flew to Mars [D9:9].\n")  # 编造引用
    assert V8.rewrite_topic_articles(d, {"Caroline/adoption"}) == 0
    assert open(p).read() == before


def test_rewrite_skips_missing_and_refless_files(tmp_path, monkeypatch):
    d = str(tmp_path)
    os.makedirs(os.path.join(d, "topics"), exist_ok=True)
    with open(os.path.join(d, "topics", "empty.md"), "w") as f:
        f.write("no refs here\n")             # dia_id 集合为空 → 拒绝
    before = open(os.path.join(d, "topics", "empty.md")).read()
    _fake(monkeypatch, "rewritten\n")
    assert V8.rewrite_topic_articles(d, {"empty", "ghost/not-there"}) == 0
    assert open(os.path.join(d, "topics", "empty.md")).read() == before


def test_rewrite_model_failure_keeps_original(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed_topic(d)
    before = open(p).read()

    def boom(**k):
        raise RuntimeError("api down")
    monkeypatch.setattr(V8.client.chat.completions, "create", boom)
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)
    assert V8.rewrite_topic_articles(d, {"Caroline/adoption"}, max_retry=1) == 0
    assert open(p).read() == before


def test_build_hooks_article_rewrite(tmp_path, monkeypatch):
    # build 主循环 session 末（tidy 之后）调 rewrite_topic_articles，带 touched 集合；
    # NATIVEMEM_V8_ARTICLE=off 时跳过。
    import src.adapters.run_nativemem as R
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v8")
    conv = {"session_1": [{"speaker": "C", "dia_id": "D1:1", "text": "hi"}],
            "session_1_date_time": "2023-05-07"}
    monkeypatch.setattr(V8, "distill_events",
        lambda turns, obs, dids, known_topics=None, **k: [
            {"when": "2023-05-07", "summary": "x", "dia_ids": ["D1:1"],
             "topic": "Caroline/adoption"}])
    calls = []
    monkeypatch.setattr(V8, "rewrite_topic_articles",
        lambda md, touched, final=False, **k: calls.append((set(touched), final)) or 0)

    monkeypatch.setenv("NATIVEMEM_V8_ARTICLE", "off")
    R.build_memory(conv, str(tmp_path / "m1"))
    assert calls == []

    monkeypatch.setenv("NATIVEMEM_V8_ARTICLE", "on")
    R.build_memory(conv, str(tmp_path / "m2"))
    # per-session rewrite (touched, final=False) then one final sweep (final=True)
    assert calls == [({"Caroline/adoption"}, False), ({"Caroline/adoption"}, True)]
