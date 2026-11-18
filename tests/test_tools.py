import socket

import pytest

from trading_agent.tools import SnapshotTools, ToolError


def test_quote(snapshot):
    q = SnapshotTools(snapshot).get_quote("SAP")
    assert q["price"] == 250.0
    assert q["day_change"] == pytest.approx(250 / 240 - 1, abs=1e-4)
    assert q["market_open"] is True
    assert SnapshotTools(snapshot).get_quote("AAPL")["market_open"] is False


def test_unknown_asset(snapshot):
    with pytest.raises(ToolError):
        SnapshotTools(snapshot).get_quote("TSLA")


def test_indicators_trend(snapshot):
    tools = SnapshotTools(snapshot)
    up = tools.get_indicators("SAP")
    down = tools.get_indicators("DAX")
    assert up["above_sma200"] is True and up["rsi14"] == 100.0
    assert down["above_sma50"] is False and down["rsi14"] == 0.0
    assert up["bars"] == 260


def test_history_window(snapshot):
    bars = SnapshotTools(snapshot).get_history("SAP", 5)
    assert len(bars) == 5
    assert bars[-1]["date"] == "2026-11-18"


def test_news_filter(snapshot):
    tools = SnapshotTools(snapshot)
    assert [n["id"] for n in tools.get_news("SAP")] == ["N1"]
    assert [n["id"] for n in tools.get_news()] == ["N1", "N2"]
    assert tools.get_news("SAP", hours=1) == []
    assert tools.news_ids() == {"N1", "N2"}


def test_overview(snapshot):
    rows = {r["asset"]: r for r in SnapshotTools(snapshot).overview()}
    assert rows["SAP"]["news_24h"] == 1
    assert rows["DAX"]["news_24h"] == 0


def test_tools_never_touch_the_network(snapshot, monkeypatch):
    """Replay-Voraussetzung: Werkzeuge lesen nur aus dem Snapshot."""

    def blocked(*a, **k):
        raise AssertionError("Netzwerkzugriff aus einem Werkzeug")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    tools = SnapshotTools(snapshot)
    tools.overview()
    for a in ("SAP", "AAPL", "DAX"):
        tools.get_quote(a)
        tools.get_indicators(a)
        tools.get_history(a, 30)
        tools.get_news(a)
