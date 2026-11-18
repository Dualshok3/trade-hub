from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from trading_agent.config import Asset, Watchlist
from trading_agent.feeds.news import NewsItem
from trading_agent.feeds.prices import Bar, Quote, StaticProvider

NOW = datetime(2026, 11, 18, 10, 30, tzinfo=UTC)  # Mittwoch, Xetra offen, New York noch zu


@pytest.fixture
def watchlist() -> Watchlist:
    return Watchlist(
        assets=(
            Asset("SAP", "SAP.DE", "SAP SE", "XETRA", "EUR", "stock", ("SAP",), "SAP Aktie"),
            Asset("AAPL", "AAPL", "Apple Inc.", "NASDAQ", "USD", "stock", ("Apple",), "Apple stock"),
            Asset("DAX", "EXS1.DE", "iShares Core DAX", "XETRA", "EUR", "etf", ("DAX",), "DAX"),
        ),
        market_news=("Börse",),
    )


def make_bars(start: float, n: int, step: float, end: date) -> list[Bar]:
    bars, price = [], start
    d = end - timedelta(days=n - 1)
    for _ in range(n):
        bars.append(Bar(d.isoformat(), price, price * 1.01, price * 0.99, price, 1000))
        price += step
        d += timedelta(days=1)
    return bars


@pytest.fixture
def provider() -> StaticProvider:
    as_of = NOW.isoformat()
    quotes = {
        "SAP.DE": Quote("SAP.DE", 250.0, 240.0, "EUR", as_of),
        "AAPL": Quote("AAPL", 230.0, 228.0, "USD", as_of),
        "EXS1.DE": Quote("EXS1.DE", 180.0, 181.0, "EUR", as_of),
        "EURUSD=X": Quote("EURUSD=X", 1.15, 1.15, "USD", as_of),
    }
    end = NOW.date()
    history = {
        "SAP.DE": [*make_bars(150.0, 260, 0.4, end), Bar("2026-11-19", 1, 1, 1, 999.0, 1)],  # Zukunft!
        "AAPL": make_bars(200.0, 260, 0.1, end),
        "EXS1.DE": make_bars(200.0, 260, -0.05, end),
    }
    return StaticProvider(quotes, history)


class FakeNews:
    name = "fake-news"

    def __init__(self, items):
        self.items = items

    def fetch(self, watchlist, now, hours):
        from trading_agent.feeds.news import deduplicate, within

        return deduplicate(within(self.items, now, hours)), []


@pytest.fixture
def news_items() -> list[NewsItem]:
    def at(h):
        return (NOW - timedelta(hours=h)).isoformat()

    return [
        NewsItem(
            "N1",
            at(2),
            "SAP hebt Cloud-Prognose an",
            "Wachstum 27 statt 25 %",
            "https://x/1",
            "Q",
            "SAP",
            ("SAP",),
        ),
        NewsItem("N2", at(5), "Apple stellt neues iPhone vor", "", "https://x/2", "Q", "Apple", ("AAPL",)),
        NewsItem("N3", at(30), "Alte DAX-Meldung", "", "https://x/3", "Q", "DAX", ("DAX",)),
        NewsItem(
            "N4", (NOW + timedelta(hours=1)).isoformat(), "Aus der Zukunft", "", "https://x/4", "Q", "", ()
        ),
    ]


@pytest.fixture
def snapshot(watchlist, provider, news_items):
    from trading_agent.snapshot import build_snapshot

    return build_snapshot(watchlist, provider, FakeNews(news_items), now=NOW)
