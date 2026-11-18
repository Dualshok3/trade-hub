"""Musterdepot und Steuern Stufe 1, mit von Hand nachgerechneten Beispielen.

Regeln in den Beispielen: 1 € pro Order, Steuersatz 26,375 %, Spread 0 (außer im Spread-Test).
"""

import pytest

from trading_agent.depot import Depot, DepotError, DepotRules, Fill, cents, load_rules

RULES = DepotRules(fee_eur=1.0, half_spread=0.0, min_order_pct=0.10, tax_rate=0.26375, allowance_eur=0.0)


def buy(depot, asset, amount, price, at="2026-11-18T10:00:00+00:00"):
    shares, exec_price, gross = depot.quote_buy(amount, price)
    f = Fill(
        depot.name, "o", asset, "buy", shares, price, exec_price, gross, depot.rules.fee_eur, 0, 0, at, at
    )
    depot.apply(f)
    return f


def sell(depot, asset, shares, price, at="2026-11-20T10:00:00+00:00"):
    pv = depot.sell_preview(asset, shares, price, int(at[:4]))
    f = Fill(
        depot.name, "o", asset, "sell", shares, price, pv["exec_price"], pv["gross"],
        depot.rules.fee_eur, pv["tax"], pv["realized"], at, at,
    )  # fmt: skip
    depot.apply(f)
    return f, pv


def test_shipped_rules():
    r = load_rules()
    assert r.fee_eur == 1.0 and r.tax_rate == 0.26375 and r.allowance_eur == 0.0


def test_cents_rounding():
    assert cents(2.11) == 2.11
    assert cents(8.571875) == 8.57
    assert cents(0.79125) == 0.79
    assert cents(0.125) == 0.13  # kaufmännisch


def test_buy_then_sell_with_gain():
    """100 € Depot. Kauf 50 € zu 20 € = 2,5 Stück, Einstand 51 € mit Gebühr.
    Verkauf zu 24 €: 60 € brutto − 1 € Gebühr − 51 € Einstand = 8 € Gewinn → 26,375 % = 2,11 € Steuer."""
    d = Depot("t", 100.0, RULES)
    buy(d, "SAP", 50, 20)
    assert d.shares("SAP") == pytest.approx(2.5)
    assert d.cash == pytest.approx(49.0)
    assert d.positions()["SAP"]["cost_eur"] == pytest.approx(51.0)
    f, pv = sell(d, "SAP", 2.5, 24)
    assert pv["realized"] == pytest.approx(8.0)
    assert f.tax_eur == 2.11
    assert d.cash == pytest.approx(49 + 60 - 1 - 2.11)  # 105,89 €
    assert d.taxes_paid == 2.11 and d.fees_paid == 2.0
    assert d.positions() == {}


def test_fifo_partial_sale():
    """Lot 1: 30 € zu 10 € = 3 Stück (Einstand 31 €). Lot 2: 30 € zu 15 € = 2 Stück (31 €).
    Verkauf 4 Stück zu 20 €: Einstand = 31 + 31/2 = 46,50 €, Erlös 80 − 1 = 79 €, Gewinn 32,50 €,
    Steuer 32,50 × 0,26375 = 8,571875 → 8,57 €. Übrig: 1 Stück aus Lot 2 mit 15,50 € Einstand."""
    d = Depot("t", 100.0, RULES)
    buy(d, "SAP", 30, 10, "2026-11-01T10:00:00+00:00")
    buy(d, "SAP", 30, 15, "2026-11-05T10:00:00+00:00")
    f, pv = sell(d, "SAP", 4, 20)
    assert pv["cost"] == pytest.approx(46.5)
    assert pv["realized"] == pytest.approx(32.5)
    assert f.tax_eur == 8.57
    assert d.shares("SAP") == pytest.approx(1.0)
    assert d.lots[0].bought_at.startswith("2026-11-05")
    assert d.lots[0].cost_eur == pytest.approx(15.5)


def test_loss_has_no_tax_and_is_recorded():
    """Kauf 50 € zu 20 € (Einstand 51 €), Verkauf zu 18 €: 45 − 1 − 51 = −7 € Verlust, keine Steuer."""
    d = Depot("t", 100.0, RULES)
    buy(d, "SAP", 50, 20)
    f, _ = sell(d, "SAP", 2.5, 18)
    assert f.tax_eur == 0.0
    assert d.realized_losses == pytest.approx(7.0)
    assert d.cash == pytest.approx(49 + 45 - 1)


def test_loss_does_not_refund_in_stage_1():
    """Stufe 1: Ein späterer Verlust erstattet keine bereits gezahlte Steuer (kommt in Stufe 2)."""
    d = Depot("t", 200.0, RULES)
    buy(d, "SAP", 50, 20)
    buy(d, "AAPL", 50, 100)
    sell(d, "SAP", 2.5, 24)  # +8 € → 2,11 € Steuer
    sell(d, "AAPL", 0.5, 80)  # −12 €
    assert d.taxes_paid == 2.11
    assert d.realized_losses == pytest.approx(12.0)


def test_allowance():
    """Freibetrag 5 €: Gewinn 8 € → 3 € steuerpflichtig → 0,79 €. Zweiter Gewinn 8 € im selben Jahr → 2,11 €.
    Im neuen Jahr gilt der Freibetrag wieder."""
    d = Depot("t", 300.0, DepotRules(fee_eur=1.0, half_spread=0.0, allowance_eur=5.0))
    for _ in range(3):
        buy(d, "SAP", 50, 20)
    assert sell(d, "SAP", 2.5, 24)[0].tax_eur == 0.79
    assert sell(d, "SAP", 2.5, 24)[0].tax_eur == 2.11
    assert sell(d, "SAP", 2.5, 24, at="2027-01-05T10:00:00+00:00")[0].tax_eur == 0.79


def test_spread():
    """0,1 % halber Spread: Kauf zu 20,02 €, Verkauf zu 19,98 €. Hin und zurück ohne Kursänderung = Verlust."""
    d = Depot("t", 100.0, DepotRules(fee_eur=1.0, half_spread=0.001))
    f = buy(d, "SAP", 50, 20)
    assert f.exec_price_eur == pytest.approx(20.02)
    assert f.gross_eur == pytest.approx(50.0)
    _, pv = sell(d, "SAP", f.shares, 20)
    assert pv["exec_price"] == pytest.approx(19.98)
    assert pv["realized"] == pytest.approx(f.shares * 19.98 - 1 - 51)
    assert pv["realized"] < -2  # 2 € Gebühren plus Spread


def test_round_trip_costs_two_percent_at_100_eur():
    d = Depot("t", 100.0, RULES)
    buy(d, "SAP", 99, 10)
    sell(d, "SAP", d.shares("SAP"), 10)
    assert d.cash == pytest.approx(98.0)  # 2 % weg, nur durch Gebühren


def test_value_and_weights():
    d = Depot("t", 100.0, RULES)
    buy(d, "SAP", 50, 20)
    assert d.value({"SAP": 22}) == pytest.approx(49 + 2.5 * 22)
    assert d.weights({"SAP": 20})["SAP"] == pytest.approx(50 / 99)
    with pytest.raises(DepotError):
        d.value({})


def test_cannot_overspend_or_oversell():
    d = Depot("t", 100.0, RULES)
    with pytest.raises(DepotError):
        buy(d, "SAP", 100, 20)  # 100 € + 1 € Gebühr > 100 € Cash
    buy(d, "SAP", 50, 20)
    with pytest.raises(DepotError):
        d.sell_preview("SAP", 3, 20, 2026)
