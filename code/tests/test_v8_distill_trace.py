from __future__ import annotations

import json

from src import v8_memory


def test_distill_trace_separates_first_pass_and_verify(
    tmp_path, monkeypatch,
) -> None:
    trace = tmp_path / "distill.jsonl"
    monkeypatch.setenv("NATIVEMEM_DISTILL_TRACE", str(trace))
    monkeypatch.setenv("NATIVEMEM_V8_VERIFY", "on")
    monkeypatch.setattr(
        v8_memory,
        "verify_event_coverage",
        lambda _turns, _events: ["Tokyo"],
    )
    responses = iter([
        '{"events":[{"when":"2026-01-01","summary":"Booked a trip",'
        '"refs":[1],"topic":"travel"}]}',
        '{"events":[{"when":"2026-01-01","summary":"The trip is to Tokyo",'
        '"refs":[1],"topic":"travel"}]}',
    ])
    monkeypatch.setattr(
        v8_memory,
        "_distill_call",
        lambda *_args, **_kwargs: next(responses),
    )

    events = v8_memory.distill_events(
        [("user", "I booked a trip to Tokyo")],
        "2026-01-01",
        ["D1:1"],
    )

    assert len(events) == 2
    record = json.loads(trace.read_text(encoding="utf-8"))
    assert record["input_dia_ids"] == ["D1:1"]
    assert len(record["first_pass_events"]) == 1
    assert record["missing_coverage_points"] == ["Tokyo"]
    assert len(record["verify_events"]) == 1
    assert len(record["post_verify_events"]) == 2
