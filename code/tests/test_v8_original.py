from src.v8_memory import build_turn_index, read_turns

def _conv():
    return {
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "hi"},
            {"speaker": "Melanie", "dia_id": "D1:2", "text": "hello there"},
            {"speaker": "Caroline", "dia_id": "D1:3", "text": "I went to a LGBTQ support group"},
        ],
        "session_2": [
            {"speaker": "Caroline", "dia_id": "D2:1", "text": "researching adoption"},
        ],
    }

def test_build_turn_index_maps_every_dia_id():
    idx = build_turn_index(_conv())
    assert set(idx.keys()) == {"D1:1", "D1:2", "D1:3", "D2:1"}
    assert idx["D1:3"]["text"] == "I went to a LGBTQ support group"
    assert idx["D1:3"]["speaker"] == "Caroline"

def test_read_turns_returns_hit_with_context():
    idx = build_turn_index(_conv())
    out = read_turns(idx, ["D1:3"], context=1)
    # D1:3 的原文 + 前一句(D1:2)作为上下文
    assert "LGBTQ support group" in out
    assert "hello there" in out  # 前一句上下文

def test_read_turns_skips_invalid_ids():
    idx = build_turn_index(_conv())
    out = read_turns(idx, ["D9:99", "D1:1"], context=0)
    assert "hi" in out          # D1:1 命中
    assert out.strip() != ""    # 无效 id 不导致崩或全空


def test_build_turn_index_attaches_session_date():
    # turn 应带上所在 session 的对话日期（标准 answerer 算相对时间的基准）
    conv = {
        "session_1": [{"speaker": "Melanie", "dia_id": "D1:1",
                       "text": "went to museum yesterday"}],
        "session_1_date_time": "1:56 pm on 6 July, 2023",
    }
    idx = build_turn_index(conv)
    assert idx["D1:1"]["date"] == "2023-07-06"


def test_read_turns_prefixes_session_date():
    # read_turns 输出每句带 (YYYY-MM-DD) 日期戳，供 answerer 换算 yesterday
    conv = {
        "session_1": [{"speaker": "Melanie", "dia_id": "D1:1",
                       "text": "went to museum yesterday"}],
        "session_1_date_time": "1:56 pm on 6 July, 2023",
    }
    idx = build_turn_index(conv)
    out = read_turns(idx, ["D1:1"], context=0)
    assert "(2023-07-06)" in out          # 说话那天的日期戳
    assert "museum yesterday" in out       # 原文相对时间也在（answerer 据此算 07-05）
