import re
import nativemem

def test_process_chunk_agent_stores_and_returns_summary(tmp_path, monkeypatch):
    # 1) mock distill_chunk：返回一条已提炼 fact
    def fake_distill(chunk_text, chunk_date, max_retry=6):
        return [{"person": "Caroline", "topic": "support group",
                 "skeleton": "Caroline went to an LGBTQ support group and found it powerful.",
                 "keep_verbatim": ["LGBTQ"],
                 "verbatim": "I went to a LGBTQ support group yesterday and it was so powerful.",
                 "distilled": "Caroline went to an LGBTQ support group and found it powerful."}]
    monkeypatch.setattr(nativemem, "distill_chunk", fake_distill)

    # 2) mock 模型 agent 会话：直接落一个文件 + 返回带 <summary> 的最终文本
    calls = {"n": 0}
    class FakeResp:
        def __init__(self, content, tool_calls=None):
            self.choices = [type("C", (), {"message": type("M", (), {
                "content": content, "tool_calls": tool_calls})()})()]
            self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})()
    def fake_create(**kwargs):
        calls["n"] += 1
        # 第一次就直接给最终答复（无 tool_calls），模拟 agent 说"我存好了"
        target = tmp_path / "Caroline.md"
        target.write_text('## support group\n[2023-05-08] Caroline went to an LGBTQ support group and found it powerful.\n  > "I went to a LGBTQ support group yesterday and it was so powerful."\n  [source](D1:3)\n')
        return FakeResp("Stored. <summary>Caroline shared about an LGBTQ support group.</summary>")
    monkeypatch.setattr(nativemem.client.chat.completions, "create", fake_create)

    summary = nativemem.process_chunk_agent(
        "Caroline: I went to a LGBTQ support group yesterday and it was so powerful.\n\n",
        "2023-05-08", str(tmp_path), ["D1:3"], running_summary="", mode="oneshot")

    assert "LGBTQ support group" in summary
    assert (tmp_path / "Caroline.md").exists()
    assert "[source](D1:3)" in (tmp_path / "Caroline.md").read_text()

def test_process_chunk_agent_forwards_running_summary_into_prompt(tmp_path, monkeypatch):
    # running_summary 必须出现在发给模型的 prompt 里（跨 chunk 滚动上下文要打通）
    def fake_distill(chunk_text, chunk_date, max_retry=6):
        return [{"person": "Caroline", "topic": "support group",
                 "skeleton": "Caroline went to an LGBTQ support group.",
                 "keep_verbatim": ["LGBTQ"],
                 "verbatim": "I went to a LGBTQ support group yesterday.",
                 "distilled": "Caroline went to an LGBTQ support group."}]
    monkeypatch.setattr(nativemem, "distill_chunk", fake_distill)

    captured = {}
    class FakeResp:
        def __init__(self, content, tool_calls=None):
            self.choices = [type("C", (), {"message": type("M", (), {
                "content": content, "tool_calls": tool_calls})()})()]
            self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})()
    def fake_create(**kwargs):
        messages = kwargs["messages"]
        captured["prompt"] = messages[0]["content"]
        return FakeResp("Stored. <summary>done</summary>")
    monkeypatch.setattr(nativemem.client.chat.completions, "create", fake_create)

    nativemem.process_chunk_agent(
        "Caroline: I went to a LGBTQ support group yesterday.\n\n",
        "2023-05-08", str(tmp_path), ["D1:3"],
        running_summary="- Caroline talked about her dog.", mode="oneshot")

    assert "Caroline talked about her dog." in captured["prompt"]


def test_process_chunk_agent_empty_running_summary_renders_placeholder(tmp_path, monkeypatch):
    # running_summary="" 时 prompt 里的槽位要渲染成 （无），不能是空字符串
    def fake_distill(chunk_text, chunk_date, max_retry=6):
        return [{"person": "Caroline", "topic": "support group",
                 "skeleton": "Caroline went to an LGBTQ support group.",
                 "keep_verbatim": ["LGBTQ"],
                 "verbatim": "I went to a LGBTQ support group yesterday.",
                 "distilled": "Caroline went to an LGBTQ support group."}]
    monkeypatch.setattr(nativemem, "distill_chunk", fake_distill)

    captured = {}
    class FakeResp:
        def __init__(self, content, tool_calls=None):
            self.choices = [type("C", (), {"message": type("M", (), {
                "content": content, "tool_calls": tool_calls})()})()]
            self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})()
    def fake_create(**kwargs):
        messages = kwargs["messages"]
        captured["prompt"] = messages[0]["content"]
        return FakeResp("Stored. <summary>done</summary>")
    monkeypatch.setattr(nativemem.client.chat.completions, "create", fake_create)

    nativemem.process_chunk_agent(
        "Caroline: I went to a LGBTQ support group yesterday.\n\n",
        "2023-05-08", str(tmp_path), ["D1:3"], running_summary="", mode="oneshot")

    assert "（无）" in captured["prompt"]


def test_process_chunk_agent_empty_facts_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(nativemem, "distill_chunk", lambda *a, **k: [])
    # facts 为空时不该调模型、直接返回 ""
    def boom(**kwargs):
        raise AssertionError("should not call model on empty facts")
    monkeypatch.setattr(nativemem.client.chat.completions, "create", boom)
    out = nativemem.process_chunk_agent("x", "2023-05-08", str(tmp_path), [], mode="oneshot")
    assert out == ""
