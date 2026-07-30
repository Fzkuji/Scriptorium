import os, sys, json, importlib
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "adapters"))

def _fake_agent_final(monkeypatch, run_nativemem, final_text):
    class R:
        def __init__(self):
            self.choices = [type("C", (), {"message": type("M", (), {
                "content": final_text, "tool_calls": None})()})()]
            self.usage = type("U", (), {"prompt_tokens":1,"completion_tokens":1,"total_tokens":2})()
    monkeypatch.setattr(run_nativemem.client.chat.completions, "create", lambda **k: R())

def test_v7_retrieve_filters_source_when_no_original(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v7")
    monkeypatch.setenv("NATIVEMEM_NO_ORIGINAL", "1")
    import run_nativemem
    importlib.reload(run_nativemem)
    (tmp_path / "Caroline.md").write_text("## g\n[2023-05-08] went to support group\n  [source](D1:3)\n")
    final = ("<memories>\n[2023-05-08] went to support group\n  [source](D1:3)\n</memories>")
    _fake_agent_final(monkeypatch, run_nativemem, final)
    mems, _ = run_nativemem.collect_memories("when support group?", str(tmp_path))
    joined = " ".join(m["text"] for m in mems)
    assert "support group" in joined
    assert "D1:3" not in joined            # source 行被过滤
    assert "[source]" not in joined

def test_v7_retrieve_keeps_source_when_use_original(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v7")
    monkeypatch.delenv("NATIVEMEM_NO_ORIGINAL", raising=False)
    import run_nativemem
    importlib.reload(run_nativemem)
    (tmp_path / "Caroline.md").write_text("## g\n[2023-05-08] went to support group\n  [source](D1:3)\n")
    final = ("<memories>\n[2023-05-08] went to support group [source](D1:3)\n</memories>")
    _fake_agent_final(monkeypatch, run_nativemem, final)
    mems, _ = run_nativemem.collect_memories("when support group?", str(tmp_path))
    joined = " ".join(m["text"] for m in mems)
    assert "support group" in joined
    assert "D1:3" in joined                # with-original: source 锚必须保留，不能被误删


def test_v7_retrieve_no_original_blocks_raw_leak_via_real_bash(tmp_path, monkeypatch):
    """真实遍历测试：agent 发一个会翻 raw/ 的 grep 命令，execute_tool 不 mock，
    走真实 subprocess。no-original 口径下最终检索结果不能含 raw/ 里的独有标记串。
    这网住的是 _collect_memories_v7 漏传 hide_raw=True 导致原文经 raw/ 泄漏的 bug。"""
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v7")
    monkeypatch.setenv("NATIVEMEM_NO_ORIGINAL", "1")
    import run_nativemem
    importlib.reload(run_nativemem)

    marker = "RAWLEAK_MARKER_xyz"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "conversation_d1.txt").write_text(
        f"user: I go to a support group.\nassistant: noted, {marker} was mentioned here.\n")
    (tmp_path / "Caroline.md").write_text(
        "## group\n[2023-05-08] went to support group\n  [source](D1:3)\n")

    calls = {"n": 0}

    class ToolCall:
        def __init__(self, id_, command):
            self.id = id_
            self.function = type("F", (), {
                "name": "bash",
                "arguments": json.dumps({"command": command})})()

    class Resp:
        def __init__(self, content, tool_calls):
            self.choices = [type("C", (), {"message": type("M", (), {
                "content": content, "tool_calls": tool_calls})()})()]
            self.usage = type("U", (), {
                "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})()

    def fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # agent tries to dig into raw/ directly — this is what a real
            # model could do since it only sees a bash tool with no fence.
            tc = ToolCall("call_1", f"grep -r {marker} .")
            return Resp(None, [tc])
        # second round: agent reports back whatever the tool output contained
        last_tool_msg = messages_seen[-1]
        final = f"<memories>\n[2023-05-08] went to support group\n  {last_tool_msg}\n</memories>"
        return Resp(final, None)

    # capture the messages list passed into client.chat.completions.create so
    # we can read what the tool actually returned to the (fake) model.
    messages_seen = []
    orig_create = fake_create

    def wrapped_create(**kwargs):
        messages_seen.clear()
        messages_seen.extend(m.get("content", "") for m in kwargs.get("messages", [])
                              if isinstance(m, dict) and m.get("role") == "tool")
        return orig_create(**kwargs)

    monkeypatch.setattr(run_nativemem.client.chat.completions, "create", wrapped_create)

    mems, _ = run_nativemem.collect_memories("when support group?", str(tmp_path))
    joined = " ".join(m["text"] for m in mems)
    assert marker not in joined


def test_v7_retrieve_survives_malformed_tool_call_json(tmp_path, monkeypatch):
    """弱模型可能吐出损坏的 tool-call arguments（非法 JSON）。_collect_memories_v7
    的 json.loads 必须像其它 tool loop 一样 try/except 兜底成 {}，不能整段炸掉。"""
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v7")
    monkeypatch.delenv("NATIVEMEM_NO_ORIGINAL", raising=False)
    import run_nativemem
    importlib.reload(run_nativemem)
    (tmp_path / "Caroline.md").write_text("## g\n[2023-05-08] went to support group\n  [source](D1:3)\n")

    calls = {"n": 0}

    class ToolCall:
        def __init__(self, id_, arguments):
            self.id = id_
            self.function = type("F", (), {"name": "bash", "arguments": arguments})()

    class Resp:
        def __init__(self, content, tool_calls):
            self.choices = [type("C", (), {"message": type("M", (), {
                "content": content, "tool_calls": tool_calls})()})()]
            self.usage = type("U", (), {
                "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})()

    def fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # malformed JSON arguments — missing closing brace/value
            tc = ToolCall("call_1", '{"command": ')
            return Resp(None, [tc])
        return Resp("<memories>\n[2023-05-08] went to support group\n</memories>", None)

    monkeypatch.setattr(run_nativemem.client.chat.completions, "create", fake_create)

    mems, rounds = run_nativemem.collect_memories("when support group?", str(tmp_path))
    assert rounds >= 1
    joined = " ".join(m["text"] for m in mems)
    assert "support group" in joined
