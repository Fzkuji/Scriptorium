import nativemem

def test_split_returns_text_and_dia_ids(locomo_session):
    chunks = nativemem.split_into_chunks(locomo_session, size=10)
    # 18 turns / 10 → 2 chunks
    assert len(chunks) == 2
    text0, dia0 = chunks[0]
    # 第一个 chunk 覆盖前 10 个 turn 的 dia_id
    assert dia0 == [t["dia_id"] for t in locomo_session[:10]]
    # chunk_text 含 speaker: text 格式
    assert locomo_session[0]["speaker"] + ":" in text0
    assert locomo_session[0]["text"] in text0

def test_split_skips_empty_and_nondict():
    session = [
        {"speaker": "A", "dia_id": "D1:1", "text": "hi"},
        "not a dict",
        {"speaker": "B", "dia_id": "D1:2", "text": ""},
        {"speaker": "A", "dia_id": "D1:3", "text": "bye"},
    ]
    chunks = nativemem.split_into_chunks(session, size=10)
    assert len(chunks) == 1
    text, dia = chunks[0]
    # 只保留有 text 的 dict turn
    assert dia == ["D1:1", "D1:3"]
    assert "hi" in text and "bye" in text
