"""timeline → topics 反向链接（独立测试文件，与 commit 同 revert 粒度）。"""
import os
import re
import src.v8_memory as V8
from src.v8_memory import write_events


def test_timeline_line_has_topic_backlink(tmp_path):
    d = str(tmp_path)
    write_events(d, [{"when": "2023-05-25", "summary": "Caroline decided to adopt",
                      "dia_ids": ["D2:3"], "topic": "Caroline/adoption"}])
    tl = open(os.path.join(d, "timeline", "2023", "05", "25.md")).read()
    line = tl.strip().splitlines()[-1]
    assert line.startswith("[2023-05-25]")        # 行首日期不动，排序/解析不破
    assert line.endswith("→ topics/Caroline/adoption.md")
    assert "· [D2:3]" in line                     # dia_id 锚仍在
    # 反链不污染 dia_id 提取
    assert re.findall(r"D\d+:\d+", line) == ["D2:3"]


def test_backlink_reaches_answerer_via_event_lines(tmp_path):
    import src.adapters.run_nativemem as R
    d = str(tmp_path / "mem")
    write_events(d, [{"when": "2023-05-25", "summary": "s",
                      "dia_ids": ["D2:3"], "topic": "adoption"}])
    lines = R._v8_event_lines(d, ["D2:3"])
    assert "→ topics/adoption.md" in lines        # 命中 timeline 行可顺反链跳文章


def test_prompts_mention_backlink():
    from src.adapters.run_nativemem import _V8_RETRIEVE_PROMPT, _V8_SINGLE_PROMPT
    for p in (_V8_RETRIEVE_PROMPT, _V8_SINGLE_PROMPT):
        assert "→ topics/" in p                   # 提示模型顺反链跳完整文章
