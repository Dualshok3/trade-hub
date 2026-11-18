"""Orderausführung, wartende Orders, Datenbank und Vergleichs-Depots."""

from datetime import timedelta

import pytest

from trading_agent.baselines import BASELINES, equal_weight_etfs, sma_trend
from trading_agent.depot import Depot, DepotError, DepotRules
from trading_agent.execution import execute, make_order
from trading_agent.feeds.prices import Quote, StaticProvider
from trading_agent.runner import run_tick
from trading_agent.snapshot import (
    SnapshotError,
    build_fill_quotes,
    load_fill_quotes,
    save_fill_quotes,
    save_snapshot,
)
from trading_agent.store import DepotStore

from .conftest import NOW

RULES = DepotRules(fee_eur=1.0, half_spread=0.0, min_order_pct=0.10, pending_max_ticks=3)


def fill_provider(provider, minutes=1, factor=1.0):
    t = (NOW + timedelta(minutes=minutes)).isoformat()
    quotes = {
        s: Quote(s, q.price * factor, q.previous_close, q.currency, t) for s, q in provider._quotes.items()
    }
    return StaticProvider(quotes, provider._history)


@pytest.fixture
def fill(watchlist, provider, snapshot):
    return build_fill_quotes(watchlist, fill_provider(provider), snapshot, now=NOW + timedelta(minutes=1))


@pytest.fixture
def stale_fill(watchlist, provider, snapshot):
    """Ausführungskurse ohne neuere Kurszeit (Börse zu oder Kurs verzögert)."""
    return build_fill_quotes(watchlist, provider, snapshot, now=NOW + timedelta(minutes=1))


def test_buy_executes_at_fill_price(snapshot, fill):
    d = Depot("t", 100.0, RULES)
    o = make_order(d, snapshot, "SAP", "buy", 0.5)
    out = execute(d, [o], fill)
    assert len(out.fills) == 1
    f = out.fills[0]
    assert f.at == "2026-11-18T10:31:00+00:00" and f.decided_at == "2026-11-18T10:30:00+00:00"
    assert f.gross_eur == pytest.approx(50.0)
    assert d.cash == pytest.approx(49.0)


def test_execution_uses_fill_price_not_seen_price(watchlist, provider, snapshot):
    """Der Agent sah 250 €, ausgeführt wird zum späteren Kurs (hier 10 % höher)."""
    fill = build_fill_quotes(
        watchlist, fill_provider(provider, factor=1.1), snapshot, now=NOW + timedelta(minutes=1)
    )
    d = Depot("t", 100.0, RULES)
    out = execute(d, [make_order(d, snapshot, "SAP", "buy", 0.5)], fill)
    assert out.fills[0].price_eur == pytest.approx(275.0)


def test_order_waits_for_newer_quote_and_expires(snapshot, stale_fill, fill):
    d = Depot("t", 100.0, RULES)
    o = make_order(d, snapshot, "SAP", "buy", 0.5)
    out = execute(d, [o], stale_fill)
    assert out.fills == [] and len(out.pending) == 1 and out.pending[0].ticks_waited == 1
    out2 = execute(d, out.pending, stale_fill)
    assert out2.expired == [] and out2.pending[0].ticks_waited == 2
    out3 = execute(d, out2.pending, stale_fill)
    assert out3.expired and not out3.pending  # nach 3 Takten verfallen
    assert len(execute(d, out2.pending, fill).fills) == 1  # mit neuerem Kurs wird sie ausgeführt


def test_buy_is_capped_by_cash_and_min_order(snapshot, fill):
    d = Depot("t", 100.0, RULES)
    out = execute(d, [make_order(d, snapshot, "SAP", "buy", 1.0)], fill)
    assert out.fills[0].gross_eur == pytest.approx(99.0)  # 100 € − 1 € Gebühr
    assert d.cash == pytest.approx(0.0)
    out = execute(d, [make_order(d, snapshot, "AAPL", "buy", 0.5)], fill)
    assert out.rejected and "Mindestorder" in out.rejected[0][1]


def test_sells_run_before_buys(snapshot, fill):
    d = Depot("t", 100.0, RULES)
    execute(d, [make_order(d, snapshot, "SAP", "buy", 1.0)], fill)
    orders = [make_order(d, snapshot, "AAPL", "buy", 0.5), make_order(d, snapshot, "SAP", "sell", 1.0)]
    out = execute(d, orders, fill)
    assert [f.side for f in out.fills] == ["sell", "buy"]


def test_sell_more_than_held_sells_all(snapshot, fill):
    d = Depot("t", 100.0, RULES)
    execute(d, [make_order(d, snapshot, "SAP", "buy", 0.3)], fill)
    out = execute(d, [make_order(d, snapshot, "SAP", "sell", 1.0)], fill)
    assert d.shares("SAP") == 0 and out.fills[0].side == "sell"


def test_invalid_orders(snapshot):
    d = Depot("t", 100.0, RULES)
    with pytest.raises(DepotError):
        make_order(d, snapshot, "SAP", "buy", 1.5)
    with pytest.raises(DepotError):
        make_order(d, snapshot, "TSLA", "buy", 0.5)
    with pytest.raises(DepotError):
        make_order(d, snapshot, "SAP", "short", 0.5)


def test_same_decision_in_both_depot_sizes(snapshot, fill):
    """Entscheidungen in Prozent: dieselbe Order läuft im 100-€- und im 10.000-€-Depot."""
    small, big = Depot("s", 100.0, RULES), Depot("b", 10_000.0, RULES)
    for d in (small, big):
        execute(d, [make_order(d, snapshot, "SAP", "buy", 0.5)], fill)
    assert small.fills[0].gross_eur == pytest.approx(50.0)
    assert big.fills[0].gross_eur == pytest.approx(5000.0)
    assert small.fees_paid / small.start_eur == pytest.approx(0.01)
    assert big.fees_paid / big.start_eur == pytest.approx(0.0001)


def test_fill_quotes_roundtrip_and_guard(watchlist, provider, snapshot, fill, tmp_path):
    save_snapshot(snapshot, tmp_path)
    save_fill_quotes(fill, snapshot, tmp_path)
    assert load_fill_quotes(snapshot, tmp_path) == fill
    with pytest.raises(SnapshotError):
        build_fill_quotes(watchlist, provider, snapshot, now=NOW - timedelta(minutes=1))


def test_store_rebuilds_same_state(snapshot, fill, tmp_path):
    store = DepotStore(tmp_path / "d.sqlite")
    d = store.open("t", 100.0, RULES, snapshot["taken_at"])
    out = execute(d, [make_order(d, snapshot, "SAP", "buy", 0.5)], fill)
    store.record_fills(out.fills)
    out = execute(d, [make_order(d, snapshot, "SAP", "sell", 0.2)], fill)
    store.record_fills(out.fills)
    again = store.load("t")
    assert again.cash == pytest.approx(d.cash)
    assert again.shares("SAP") == pytest.approx(d.shares("SAP"))
    assert again.taxes_paid == d.taxes_paid and again.fees_paid == d.fees_paid


def test_store_pending_roundtrip(snapshot, stale_fill, tmp_path):
    store = DepotStore(tmp_path / "d.sqlite")
    d = store.open("t", 100.0, RULES, snapshot["taken_at"])
    out = execute(d, [make_order(d, snapshot, "SAP", "buy", 0.5, "Test")], stale_fill)
    store.set_pending("t", out.pending)
    assert store.pending("t") == out.pending


def test_baselines(snapshot):
    d = Depot("t", 100.0, RULES)
    assert [x[0] for x in equal_weight_etfs(snapshot, d)] == ["DAX"]
    picks = {x[0] for x in sma_trend(snapshot, d) if x[1] == "buy"}
    assert picks == {"SAP", "AAPL"}  # Aufwärtstrend; DAX ist ETF und fällt


def test_run_tick_is_idempotent_and_marks(snapshot, fill, tmp_path):
    store = DepotStore(tmp_path / "d.sqlite")
    res = run_tick(snapshot, fill, store, BASELINES, 100.0, RULES)
    assert set(res) == {f"{n}-100" for n in BASELINES}
    assert res["eq-etfs-100"]["fills"] and res["cash-100"]["value_eur"] == 100.0
    assert run_tick(snapshot, fill, store, BASELINES, 100.0, RULES) == {}  # derselbe Takt läuft nicht doppelt
    assert len(store.marks("eq-etfs-100")) == 1
    # Ein zweites Depot mit 10.000 € läuft daneben
    res = run_tick(snapshot, fill, store, {"eq-etfs": equal_weight_etfs}, 10_000.0, RULES)
    assert res["eq-etfs-10000"]["fills"][0]["gross_eur"] == pytest.approx(9999.0)


def test_pending_is_not_duplicated(snapshot, stale_fill, tmp_path):
    store = DepotStore(tmp_path / "d.sqlite")
    strategies = {"eq-etfs": equal_weight_etfs}
    run_tick(snapshot, stale_fill, store, strategies, 100.0, RULES)
    assert len(store.pending("eq-etfs-100")) == 1
    other = dict(snapshot, hash="x" * 64)
    run_tick(other, dict(stale_fill, snapshot="x" * 64), store, strategies, 100.0, RULES)
    assert len(store.pending("eq-etfs-100")) == 1
