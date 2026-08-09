"""Provider spend readings, which do not depend on prices or token counts."""

from memory.runtime import billing


def test_reads_the_dashboard_counter(monkeypatch):
    monkeypatch.setattr(
        billing, "_get_json",
        lambda url, key: {"object": "list", "total_usage": 13964.85},
    )

    reading = billing.read_spend("https://provider.example", "k")

    assert reading.value == 13964.85
    assert reading.endpoint == "/v1/dashboard/billing/usage"


def test_trailing_v1_is_not_doubled(monkeypatch):
    seen = []

    def capture(url, key):
        seen.append(url)
        return {"total_usage": 1.0}

    monkeypatch.setattr(billing, "_get_json", capture)
    billing.read_spend("https://provider.example/v1", "k")

    # The dashboard path is relative to the host, not to the /v1 suffix.
    assert seen[0] == "https://provider.example/v1/dashboard/billing/usage"


def test_nested_credits_shape_is_understood(monkeypatch):
    monkeypatch.setattr(
        billing, "_get_json",
        lambda url, key: (
            None if "dashboard" in url
            else {"data": {"total_credits": 80.0, "total_usage": 74.3}}
        ),
    )

    reading = billing.read_spend("https://provider.example", "k")

    assert reading.value == 74.3


def test_a_missing_endpoint_never_raises(monkeypatch):
    monkeypatch.setattr(billing, "_get_json", lambda url, key: None)

    # A run must not fail because billing is unavailable.
    assert billing.read_spend("https://provider.example", "k") is None
    assert billing.read_spend("", "k") is None


def test_delta_reports_the_movement():
    before = billing.SpendReading(10.0, "/v1/dashboard/billing/usage")
    after = billing.SpendReading(12.5, "/v1/dashboard/billing/usage")

    delta = billing.spend_delta(before, after)

    assert delta["delta"] == 2.5
    assert "concurrent usage" in delta["note"]


def test_delta_is_absent_when_either_reading_is():
    reading = billing.SpendReading(1.0, "/x")

    assert billing.spend_delta(None, reading) is None
    assert billing.spend_delta(reading, None) is None
