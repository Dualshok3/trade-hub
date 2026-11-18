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
    """Letzter bekannter Kurs in der Handelswährung des Werts.

    `as_of` ist der Zeitpunkt des Kurses an der Börse (UTC, ISO 8601), nicht der Abrufzeitpunkt.
    yfinance liefert je nach Börse verzögerte Kurse; `fetched_at` hält fest, wann abgerufen wurde.
    """

    symbol: str
    price: float
    previous_close: float | None
    currency: str
    as_of: str
    fetched_at: str = ""

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

    def _last_minute_bars(self, symbols: list[str]) -> dict[str, tuple[float, str]]:
        """Letzte Minutenkerze je Symbol: (Schlusskurs, Zeitpunkt der Kerze an der Börse in UTC)."""
        import yfinance as yf

        frame = yf.download(
            symbols,
            period="5d",
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            prepost=False,
            progress=False,
            threads=True,
        )
        out = {}
        for symbol in symbols:
            try:
                closes = (frame[symbol] if len(symbols) > 1 else frame)["Close"].dropna()
            except KeyError:
                continue
            if closes.empty:
                continue
            ts = closes.index[-1]
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            out[symbol] = (float(closes.iloc[-1]), ts.to_pydatetime().replace(microsecond=0).isoformat())
        return out

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        import yfinance as yf

        fetched = datetime.now(UTC).replace(microsecond=0).isoformat()
        try:
            minute = self._last_minute_bars(symbols)
        except Exception as e:  # yfinance wirft sehr unterschiedliche Fehler
            log.warning("Minutenkurse fehlgeschlagen: %s", e)
            minute = {}
        out: dict[str, Quote] = {}
        for symbol in symbols:
            try:
                info = yf.Ticker(symbol).fast_info
                prev = _clean(info["previous_close"])
                currency = str(info["currency"]).upper()
                fallback = _clean(info["last_price"])
            except Exception as e:
                log.warning("kein Kurs für %s: %s", symbol, e)
                continue
            if symbol in minute:
                price, as_of = minute[symbol]
            elif fallback is not None:
                # ohne Minutenkerze ist der Kurszeitpunkt unbekannt: leer lassen,
                # statt den Abrufzeitpunkt als Kurszeit auszugeben
                price, as_of = fallback, ""
            else:
                continue
            out[symbol] = Quote(symbol, price, prev, currency, as_of, fetched)
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
