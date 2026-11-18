"""Kursquellen hinter einer schmalen Schnittstelle, damit `yfinance` später austauschbar ist."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Protocol

log = logging.getLogger(__name__)


class PriceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Bar:
    """Eine Tageskerze. `date` ist der Handelstag (ISO-Datum der Börse)."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Quote:
    """Letzter bekannter Kurs in der Handelswährung des Werts. `as_of` ist UTC (ISO 8601)."""

    symbol: str
    price: float
    previous_close: float | None
    currency: str
    as_of: str

    def to_dict(self) -> dict:
        return asdict(self)


class PriceProvider(Protocol):
    name: str

    def quotes(self, symbols: list[str]) -> dict[str, Quote]: ...

    def history(self, symbols: list[str], days: int) -> dict[str, list[Bar]]: ...


def _clean(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN


class YFinanceProvider:
    """Kurse über `yfinance` (inoffiziell, kann ausfallen; Kurse oft 15 Minuten verzögert)."""

    name = "yfinance"

    def history(self, symbols: list[str], days: int) -> dict[str, list[Bar]]:
        import yfinance as yf

        period = f"{max(days, 5)}d"
        frame = yf.download(
            symbols,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        out: dict[str, list[Bar]] = {}
        for symbol in symbols:
            try:
                sub = frame[symbol] if len(symbols) > 1 else frame
            except KeyError:
                log.warning("keine Historie für %s", symbol)
                continue
            bars = []
            for idx, row in sub.iterrows():
                close = _clean(row.get("Close"))
                if close is None:
                    continue
                bars.append(
                    Bar(
                        date=idx.date().isoformat(),
                        open=_clean(row.get("Open")) or close,
                        high=_clean(row.get("High")) or close,
                        low=_clean(row.get("Low")) or close,
                        close=close,
                        volume=_clean(row.get("Volume")) or 0.0,
                    )
                )
            if bars:
                out[symbol] = bars
        return out

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        import yfinance as yf

        now = datetime.now(UTC).replace(microsecond=0).isoformat()
        out: dict[str, Quote] = {}
        for symbol in symbols:
            try:
                info = yf.Ticker(symbol).fast_info
                price = _clean(info["last_price"])
                prev = _clean(info["previous_close"])
                currency = info["currency"]
            except Exception as e:  # yfinance wirft sehr unterschiedliche Fehler
                log.warning("kein Kurs für %s: %s", symbol, e)
                continue
            if price is None:
                continue
            out[symbol] = Quote(symbol, price, prev, str(currency).upper(), now)
        return out


class StaticProvider:
    """Feste Kurse für Tests und Replays."""

    name = "static"

    def __init__(self, quotes: dict[str, Quote], history: dict[str, list[Bar]] | None = None):
        self._quotes = quotes
        self._history = history or {}

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        return {s: self._quotes[s] for s in symbols if s in self._quotes}

    def history(self, symbols: list[str], days: int) -> dict[str, list[Bar]]:
        return {s: self._history[s][-days:] for s in symbols if s in self._history}
