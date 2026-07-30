import os
import src.v8_memory as V8
from src.v8_memory import link_dates_to_timeline, write_events


# ---- link_dates_to_timeline：纯代码日期→timeline 链接 ----

def test_bare_date_linked():
    out = link_dates_to_timeline("On 2023-05-25, Caroline decided to adopt.")
    assert "[2023-05-25](timeline/2023/05/25.md)" in out


def test_bracketed_date_linked():
    out = link_dates_to_timeline("[2023-05-25] Caroline decided to adopt [D2:3]")
    assert out.startswith("[2023-05-25](timeline/2023/05/25.md)")
    assert "[D2:3]" in out                # dia_id 引用不受影响


def test_link_points_to_day_file():
    # 天级：日期链接目标是天文件 timeline/YYYY/MM/DD.md（月日两位数字），不是月文件
    out = link_dates_to_timeline("see 2023-07-07 and 2024-01-02")
    assert "[2023-07-07](timeline/2023/07/07.md)" in out
    assert "[2024-01-02](timeline/2024/01/02.md)" in out


def test_idempotent():
    once = link_dates_to_timeline("met on 2023-05-25 and again [2024-01-02].")
    assert link_dates_to_timeline(once) == once   # 已是链接不重复包


def test_invalid_month_untouched():
    s = "code 2023-13-01 is not a date"
    assert link_dates_to_timeline(s) == s


def test_write_events_topic_line_linked_and_sorted(tmp_path):
    d = str(tmp_path)
    write_events(d, [
        {"when": "2023-05-20", "summary": "B", "dia_ids": ["D1:2"], "topic": "t"},
        {"when": "2023-05-07", "summary": "A", "dia_ids": ["D1:1"], "topic": "t"},
    ])
    body = open(os.path.join(d, "topics", "t.md")).read()
    assert "[2023-05-07](timeline/2023/05/07.md)" in body
    # 链接形态行首仍是 [日期]，_append_sorted 排序不破
    assert body.index("A") < body.index("B")


def test_rewrite_article_links_dates(tmp_path, monkeypatch):
    d = str(tmp_path)
    write_events(d, [{"when": "2023-05-25", "summary": "s",
                      "summary_inline": "Caroline decided to adopt [D2:3]",
                      "dia_ids": ["D2:3"], "topic": "adoption"}])

    class _R:
        def __init__(self):
            self.choices = [type("C", (), {"message": type("M", (), {
                "content": "## Adoption\nOn 2023-05-25, Caroline decided to adopt [D2:3].\n"})()})()]
            self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()
    monkeypatch.setattr(V8.client.chat.completions, "create", lambda **k: _R())
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)
    # single raw line is below the trigger floor; final sweep forces it.
    assert V8.rewrite_topic_articles(d, {"adoption"}, final=True) == 1
    body = open(os.path.join(d, "topics", "adoption.md")).read()
    assert "[2023-05-25](timeline/2023/05/25.md)" in body
