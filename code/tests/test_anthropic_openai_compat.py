from types import SimpleNamespace

from src.anthropic_openai_compat import _messages


def test_openai_tool_roundtrip_becomes_anthropic_blocks():
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="read", arguments='{"path":"a.md"}'),
    )
    system, messages = _messages([
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ])
    assert system == "system"
    assert messages[1]["content"][0] == {
        "type": "tool_use",
        "id": "call-1",
        "name": "read",
        "input": {"path": "a.md"},
    }
    assert messages[2]["content"][0]["type"] == "tool_result"


def test_assistant_object_is_accepted():
    _, messages = _messages([
        SimpleNamespace(role="assistant", content="done", tool_calls=None)
    ])
    assert messages == [{"role": "assistant", "content": [{"type": "text", "text": "done"}]}]
