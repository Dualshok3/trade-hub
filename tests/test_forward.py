"""Forward-Score: Rechnung je Order, Fenster in Handelstagen, Takte mit Halten, gepaarter Vergleich."""

import pytest

from trading_agent.bench.forward import (
    Day,
    _window,
    depot_scores,
    eur_series,
    market_return,
    paired,
    score_fill,
)
from trading_agent.feeds.prices import Bar

DATES = ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"]


def series(**closes):
    """Je Wert eine Liste von Schlusskursen; Eröffnung = Schlusskurs des Vortags (am ersten Tag gleich)."""
    out = {}
    for asset, cs in closes.items():
        out[asset] = [
            Day(d, cs[i - 1] if i else cs[0], c) for i, (d, c) in enumerate(zip(DATES, cs, strict=False))
        ]
    return out


S = series(A=[100, 110, 120, 130, 140, 150], B=[100, 100, 100, 100, 100, 100])


def fill(side="buy", asset="A", at="2026-11-03T08:00:00+00:00", gross=50.0, exec_price=100.0, fee=1.0):
    return {
        "depot": "x-100", "order_id": "O1", "asset": asset, "side": side, "shares": gross / exec_price,
        "price_eur": exec_price, "exec_price_eur": exec_price, "gross_eur": gross, "fee_eur": fee,
        "tax_eur": 0.0, "realized_eur": 0.0, "at": at, "decided_at": "2026-11-03T06:30:00+00:00",
    }  # fmt: skip


def test_window_counts_trading_days_from_fill_day():
    first, last = _window(S["A"], "2026-11-03", 1)
    assert first.date == last.date == "2026-11-03"
    assert _window(S["A"], "2026-11-03", 4)[1].date == "2026-11-06"
    assert _window(S["A"], "2026-11-07", 1)[0].date == "2026-11-09"  # Samstag: nächster Handelstag
    assert _window(S["A"], "2026-11-03", 10) is None


def test_buy_beats_market_by_the_spread_between_assets():
    # A: Eröffnung 100 am 03.11., Schluss 130 am 05.11. → +30 %; B: 0 %; Markt = Mittel = +15 %
    assert market_return(S, "2026-11-03", 3) == pytest.approx(0.15)
    s = score_fill(fill(), 100.0, S, 3)
    assert s.asset_return == pytest.approx(0.30)
    assert s.score_eur == pytest.approx(50 * (0.30 - 0.15) - 1)
    assert s.score_pct == pytest.approx(s.score_eur / 100)


def test_sell_is_the_mirror_image():
    s = score_fill(fill("sell"), 100.0, S, 3)
    assert s.score_eur == pytest.approx(50 * (0.15 - 0.30) - 1)


def test_spread_counts_against_the_order():
    worse = score_fill(fill(exec_price=101.0), 100.0, S, 3)
    assert worse.score_eur < score_fill(fill(), 100.0, S, 3).score_eur


def test_ticks_hold_zero_and_drop_incomplete_horizons():
    rows = [
        ("s1", "2026-11-02T06:30:00+00:00", {"orders": [], "fills": []}),
        (
            "s2",
            "2026-11-03T06:30:00+00:00",
            {"orders": [{"id": "O1", "value_at_decision": 100.0}], "fills": [fill()]},
        ),
        ("s3", "2026-11-09T06:30:00+00:00", {"orders": [], "fills": []}),
    ]
    sc = depot_scores(rows, "x-100", S, 3)
    assert set(sc.by_snapshot) == {"s1", "s2"}  # s3: drei Tage ab dem 09.11. liegen nicht mehr in den Kursen
    assert sc.by_snapshot["s1"] == 0.0 and sc.by_snapshot["s2"] == pytest.approx(0.065)
    assert sc.total_eur == pytest.approx(6.5) and sc.before_fees_eur == pytest.approx(7.5)
    assert sc.hit_rate == 1.0
    other = depot_scores([(s, at, {"orders": [], "fills": []}) for s, at, _ in rows], "y-100", S, 3)
    d = paired(sc, other)
    assert d.mean == pytest.approx(0.065 / 2) and d.n == 2


def test_eur_series_converts_usd(watchlist):
    bars = {
        "SAP.DE": [Bar("2026-11-02", 10, 10, 10, 10, 1)],
        "AAPL": [Bar("2026-11-02", 11, 11, 11, 11, 1)],
        watchlist.fx_symbol: [Bar("2026-11-02", 1.1, 1.1, 1.1, 1.1, 0)],
    }
    s = eur_series(bars, watchlist)
    assert s["SAP"][0].close_eur == 10
    assert s["AAPL"][0].close_eur == pytest.approx(10.0)
