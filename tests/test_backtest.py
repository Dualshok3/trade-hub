"""Backtest: Snapshots aus Tageskerzen ohne Blick in die Zukunft, News erst ab dem Folgetag."""

from datetime import date, timedelta

from trading_agent.backtest import (
    BacktestError,
    DailyBarsProvider,
    HistoricalNewsProvider,
    build_day,
    decision_time,
    news_url,
    symbol_meta,
    trading_days,
)
from trading_agent.baselines import BASELINES
from trading_agent.depot import DepotRules
from trading_agent.feeds.prices import Bar
from trading_agent.runner import run_tick
from trading_agent.store import DepotStore

RULES = DepotRules(fee_eur=1.0, half_spread=0.0, min_order_pct=0.10, pending_max_ticks=3)
END = date(2026, 11, 13)  # Freitag


def bars_for(watchlist, n=260):
    """Werktags-Kerzen; der Eröffnungskurs ist immer 1 über dem Vortagesschluss, damit man sieht, woher
    ein Kurs kommt."""
    out = {}
    for k, sym in enumerate([*(a.symbol for a in watchlist.assets), watchlist.fx_symbol]):
        price, d, bars = 100.0 + 10 * k, END, []
        while len(bars) < n:
            if d.weekday() < 5:
                bars.append(d)
            d -= timedelta(days=1)
        series = []
        for i, day in enumerate(reversed(bars)):
            close = 1.1 if sym == watchlist.fx_symbol else price + i
            series.append(Bar(day.isoformat(), close + 1, close + 2, close - 1, close, 1000))
        out[sym] = series
    return out


def test_close_side_sees_only_the_day_before(watchlist):
    bars = bars_for(watchlist)
    day = date(2026, 11, 11)  # Mittwoch
    p = DailyBarsProvider(bars, symbol_meta(watchlist), day, "close")
    q = p.quotes(["SAP.DE"])["SAP.DE"]
    prev_bar = next(b for b in bars["SAP.DE"] if b.date == "2026-11-10")
    assert q.price == prev_bar.close and q.as_of.startswith("2026-11-10")
    assert all(b.date < "2026-11-11" for b in p.history(["SAP.DE"], 400)["SAP.DE"])


def test_open_side_fills_at_the_next_open(watchlist):
    bars = bars_for(watchlist)
    p = DailyBarsProvider(bars, symbol_meta(watchlist), date(2026, 11, 14), "open")  # Samstag
    q = p.quotes(["SAP.DE"])  # nach dem letzten Tag gibt es keinen Kurs: Order wartet
    assert q == {}
    p = DailyBarsProvider(bars, symbol_meta(watchlist), date(2026, 11, 11), "open")
    q = p.quotes(["AAPL"])["AAPL"]
    bar = next(b for b in bars["AAPL"] if b.date == "2026-11-11")
    assert q.price == bar.open and q.as_of == "2026-11-11T14:30:00+00:00"


def test_trading_days_are_weekdays_in_range(watchlist):
    days = trading_days(bars_for(watchlist), date(2026, 11, 2), END)
    assert days[0] == date(2026, 11, 2) and days[-1] == END and len(days) == 10


def test_day_snapshot_has_no_future_data(watchlist):
    bars = bars_for(watchlist)
    day = date(2026, 11, 11)
    snap, fill = build_day(watchlist, bars, day)
    assert snap["taken_at"] == decision_time(day).isoformat()
    assert snap["errors"] == [] and snap["open_exchanges"] == []
    for a in snap["history"].values():
        assert a[-1]["date"] == "2026-11-10"
    for asset, q in snap["quotes"].items():
        assert q["as_of"] < snap["taken_at"] < fill["quotes"][asset]["as_of"] <= fill["taken_at"]


def test_baselines_trade_through_days(watchlist, tmp_path):
    bars = bars_for(watchlist)
    store = DepotStore(tmp_path / "bt.sqlite")
    for day in trading_days(bars, date(2026, 11, 2), END):
        snap, fill = build_day(watchlist, bars, day)
        run_tick(snap, fill, store, dict(BASELINES), 100.0, RULES)
    marks = store.marks("sma-trend-100")
    assert len(marks) == 10
    assert store.load("cash-100").cash == 100.0
    store.close()


FEED = """<?xml version="1.0"?><rss><channel>
<item><title>SAP hebt Prognose an - Handelsblatt</title><link>https://x/1</link>
<pubDate>Tue, 10 Nov 2026 07:00:00 GMT</pubDate><source url="https://x">Handelsblatt</source></item>
<item><title>SAP-Aktie bricht ein - WELT</title><link>https://x/2</link>
<pubDate>Wed, 11 Nov 2026 07:00:00 GMT</pubDate><source url="https://x">WELT</source></item>
</channel></rss>"""


def test_news_count_only_from_the_next_day(watchlist, tmp_path):
    urls = []

    def fetch(url):
        urls.append(url)
        return FEED.encode()

    news = HistoricalNewsProvider(tmp_path / "cache", fetch=fetch)
    snap, _ = build_day(watchlist, bars_for(watchlist), date(2026, 11, 11), news)
    titles = {n["title"] for n in snap["news"]}
    assert titles == {"SAP hebt Prognose an"}  # die Meldung vom Entscheidungstag selbst fehlt
    assert "after%3A2026-11-09+before%3A2026-11-11" in urls[0]
    n = len(urls)
    build_day(watchlist, bars_for(watchlist), date(2026, 11, 11), news)
    assert len(urls) == n  # zweiter Lauf kommt aus dem Cache


def test_news_url_brackets_the_day():
    url = news_url("SAP Aktie", "de", date(2026, 9, 15))
    assert "after%3A2026-09-14" in url and "before%3A2026-09-16" in url and "hl=de" in url


def test_missing_news_stop_the_run(watchlist, tmp_path):
    import httpx
    import pytest

    def fetch(url):
        raise httpx.ConnectError("getaddrinfo failed")

    news = HistoricalNewsProvider(tmp_path / "cache", fetch=fetch)
    with pytest.raises(BacktestError, match="nicht abrufbar"):
        build_day(watchlist, bars_for(watchlist), date(2026, 11, 11), news)
    assert not list((tmp_path / "cache").rglob("*.xml"))  # nichts Kaputtes im Cache
