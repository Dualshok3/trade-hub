from datetime import UTC, datetime

import pytest

from trading_agent import indicators as ind
from trading_agent.config import DEFAULT_WATCHLIST, ConfigError, load_watchlist
from trading_agent.market import is_open, open_exchanges


def test_sma():
    assert ind.sma([1, 2, 3, 4], 2) == 3.5
    assert ind.sma([1, 2], 3) is None


def test_rsi_known_values():
    assert ind.rsi(list(range(1, 20))) == 100.0
    assert ind.rsi(list(range(20, 1, -1))) == 0.0
    assert ind.rsi([1.0] * 20) == 50.0
    # Wechselnde Bewegung gleicher Größe: RSI um 50
    assert ind.rsi([10, 11] * 15) == pytest.approx(50, abs=5)
    assert ind.rsi([1, 2, 3]) is None


def test_volatility_and_change():
    assert ind.volatility([100.0] * 30) == 0.0
    assert ind.volatility([100, 101] * 15) > 0.1
    assert ind.change([100, 110], 1) == pytest.approx(0.1)
    assert ind.change([100], 1) is None


def test_from_high():
    assert ind.from_high([90], [100]) == pytest.approx(-0.1)


def test_market_hours():
    # Mittwoch 18.11.2026
    assert is_open("XETRA", datetime(2026, 11, 18, 8, 0, tzinfo=UTC))  # 09:00 Berlin
    assert not is_open("XETRA", datetime(2026, 11, 18, 7, 59, tzinfo=UTC))
    assert not is_open("XETRA", datetime(2026, 11, 18, 16, 30, tzinfo=UTC))  # 17:30 Berlin
    assert is_open("NYSE", datetime(2026, 11, 18, 14, 30, tzinfo=UTC))  # 09:30 New York
    assert open_exchanges(datetime(2026, 11, 18, 15, 0, tzinfo=UTC)) == ["XETRA", "NYSE", "NASDAQ"]
    assert open_exchanges(datetime(2026, 11, 21, 12, 0, tzinfo=UTC)) == []  # Samstag
    with pytest.raises(ValueError):
        is_open("XETRA", datetime(2026, 11, 18, 10))


def test_shipped_watchlist_loads():
    wl = load_watchlist(DEFAULT_WATCHLIST)
    assert len(wl.assets) >= 15
    assert wl.get("SAP").symbol == "SAP.DE"
    assert {a.kind for a in wl.assets} == {"stock", "etf"}


def test_bad_watchlist(tmp_path):
    p = tmp_path / "w.toml"
    p.write_text('[[asset]]\nid="X"\nsymbol="X"\nname="X"\nexchange="LSE"\ncurrency="EUR"\nkind="stock"\n')
    with pytest.raises(ConfigError, match="Börse"):
        load_watchlist(p)


@pytest.mark.network
def test_live_yfinance_quote():
    from trading_agent.feeds.prices import YFinanceProvider

    q = YFinanceProvider().quotes(["SAP.DE"])
    assert q["SAP.DE"].price > 0
