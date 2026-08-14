import os
from src.v8_memory import event_line, timeline_path, write_events, _sanitize_topic

def test_event_line_format():
    assert event_line("2023-05-07", "去了支持小组", ["D1:3"]) == "[2023-05-07] 去了支持小组 · [D1:3]"
    assert event_line("2023-05-07", "研究收养", ["D2:5", "D2:6"]) == "[2023-05-07] 研究收养 · [D2:5-6]"

def test_timeline_path_year_month_day(tmp_path):
    p = timeline_path(str(tmp_path), "2023-05-07")
    assert p.endswith(os.path.join("timeline", "2023", "05", "07.md"))

def test_write_events_dual_view_split_by_day(tmp_path):
    d = str(tmp_path)
    write_events(d, [
        {"when": "2023-05-20", "summary": "研究收养", "dia_ids": ["D2:5"], "topic": "Caroline-adoption"},
        {"when": "2023-05-07", "summary": "去支持小组", "dia_ids": ["D1:3"], "topic": "Caroline-support"},
    ])
    # 时间视图：不同天进不同天文件（同月同文件夹）
    p07 = os.path.join(d, "timeline", "2023", "05", "07.md")
    p20 = os.path.join(d, "timeline", "2023", "05", "20.md")
    assert "去支持小组" in open(p07).read() and "研究收养" not in open(p07).read()
    assert "研究收养" in open(p20).read() and "去支持小组" not in open(p20).read()
    # 话题视图：各自话题文件（topics/ 固定入口）
    assert "研究收养" in open(os.path.join(d, "topics", "Caroline-adoption.md")).read()
    assert "去支持小组" in open(os.path.join(d, "topics", "Caroline-support.md")).read()

def test_write_events_same_day_two_events_one_file(tmp_path):
    # 同一天两个事件进同一个天文件，且按插入排序稳定
    d = str(tmp_path)
    write_events(d, [
        {"when": "2023-05-07", "summary": "早上的事", "dia_ids": ["D1:1"], "topic": "t"},
        {"when": "2023-05-07", "summary": "晚上的事", "dia_ids": ["D1:2"], "topic": "t"},
    ])
    day = os.path.join(d, "timeline", "2023", "05", "07.md")
    text = open(day).read()
    assert "早上的事" in text and "晚上的事" in text
    assert len([l for l in text.splitlines() if l.strip()]) == 2

def test_write_events_appends_not_overwrites(tmp_path):
    d = str(tmp_path)
    write_events(d, [{"when": "2023-05-07", "summary": "A", "dia_ids": ["D1:1"], "topic": "t"}])
    write_events(d, [{"when": "2023-05-08", "summary": "B", "dia_ids": ["D1:2"], "topic": "t"}])
    t = open(os.path.join(d, "topics", "t.md")).read()
    assert "A" in t and "B" in t

def test_write_events_routes_across_months_and_years(tmp_path):
    # 三个事件横跨两个年份、三个月份 —— 每个必须落在自己的天文件，
    # 且不串到别的文件里去（回归防止 timeline_path 路由错位）。
    d = str(tmp_path)
    write_events(d, [
        {"when": "2023-05-07", "summary": "五月的事", "dia_ids": ["D1:1"], "topic": "t"},
        {"when": "2023-06-02", "summary": "六月的事", "dia_ids": ["D2:1"], "topic": "t"},
        {"when": "2024-01-15", "summary": "次年一月的事", "dia_ids": ["D3:1"], "topic": "t"},
    ])
    p_may = os.path.join(d, "timeline", "2023", "05", "07.md")
    p_jun = os.path.join(d, "timeline", "2023", "06", "02.md")
    p_jan = os.path.join(d, "timeline", "2024", "01", "15.md")
    assert os.path.exists(p_may)
    assert os.path.exists(p_jun)
    assert os.path.exists(p_jan)

    may_text = open(p_may).read()
    jun_text = open(p_jun).read()
    jan_text = open(p_jan).read()

    assert "五月的事" in may_text
    assert "六月的事" not in may_text
    assert "次年一月的事" not in may_text

    assert "六月的事" in jun_text
    assert "五月的事" not in jun_text
    assert "次年一月的事" not in jun_text

    assert "次年一月的事" in jan_text
    assert "五月的事" not in jan_text
    assert "六月的事" not in jan_text


def test_write_events_topics_side_uses_inline_refs(tmp_path):
    # 双视图分家：timeline 保持原子行（纯摘要 · refs），topics 侧写内联引用句
    d = str(tmp_path)
    write_events(d, [{
        "when": "2023-05-25",
        "summary": "Caroline decided to adopt, visited an agency",
        "summary_inline": "Caroline decided to adopt [D2:3], visited an agency [D2:8]",
        "dia_ids": ["D2:3", "D2:8"], "topic": "Caroline/adoption"}])
    tl = open(os.path.join(d, "timeline", "2023", "05", "25.md")).read()
    assert "[2023-05-25] Caroline decided to adopt, visited an agency · [D2:3,8]" in tl
    tp = open(os.path.join(d, "topics", "Caroline", "adoption.md")).read()
    # 行首日期已被代码链到 timeline，正文保持内联引用句
    assert ("[2023-05-25](timeline/2023/05/25.md) "
            "Caroline decided to adopt [D2:3], visited an agency [D2:8]") in tp
    assert "· [D2:3, D2:8]" not in tp     # topics 侧不再挂句末 refs 块


def test_write_events_without_inline_keeps_old_format(tmp_path):
    # 无 summary_inline 的事件（老调用方/测试）：topics 行体保持老格式（仅日期加链）
    d = str(tmp_path)
    write_events(d, [{"when": "2023-05-07", "summary": "x",
                      "dia_ids": ["D1:1"], "topic": "t"}])
    tp = open(os.path.join(d, "topics", "t.md")).read()
    assert "x · [D1:1]" in tp


def test_sanitize_topic_no_silent_collision():
    # `/` 现在是层级路径分隔符（不再拍成 -）：a/b 与 ab 不该碰撞
    assert _sanitize_topic("a/b") != _sanitize_topic("ab")
    assert _sanitize_topic("a/b") == "a/b"      # 保留为路径
    assert _sanitize_topic("a.b") == "a-b"      # 点等非词字符→连字符
    assert _sanitize_topic("Caroline/adoption") == "Caroline/adoption"
    # 连字符保留、空格变连字符
    assert _sanitize_topic("Caroline-adoption") == "Caroline-adoption"
    assert _sanitize_topic("Caroline adoption") == "Caroline-adoption"
