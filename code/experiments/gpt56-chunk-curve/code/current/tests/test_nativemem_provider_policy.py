from __future__ import annotations

from src import nativemem


def test_chat_create_injects_reasoning_policy(monkeypatch) -> None:
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("NATIVEMEM_REASONING_EFFORT", "none")
    monkeypatch.setattr(nativemem.client.chat.completions, "create", fake_create)
    nativemem._chat_create(model="test", messages=[])
    assert captured["reasoning_effort"] == "none"


def test_chat_create_preserves_explicit_reasoning_policy(monkeypatch) -> None:
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("NATIVEMEM_REASONING_EFFORT", "none")
    monkeypatch.setattr(nativemem.client.chat.completions, "create", fake_create)
    nativemem._chat_create(
        model="test", messages=[], reasoning_effort="minimal",
    )
    assert captured["reasoning_effort"] == "minimal"
