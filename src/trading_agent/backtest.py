"""Backtest: die Agenten rückwirkend über vergangene Handelstage laufen lassen.

Für jeden Handelstag T entsteht ein normaler Snapshot, so wie ihn ein Live-Takt vor Börsenstart gebaut hätte:

- **Entscheidung** am Tag T um 06:30 UTC, vor der Xetra-Eröffnung. Sichtbar sind nur Tageskerzen bis zum
  Vortag (Kurs = letzter Schlusskurs) und News, die Google News auf den Vortag datiert.
- **Ausführung** zum Eröffnungskurs des Tages T (bzw. des nächsten Handelstags des Werts), also immer zu einem
  Kurs, den das Modell nicht kannte.

Historische Google-News-Treffer tragen nur ein Datum (Zeit 07:00 GMT). Deshalb zählt eine Meldung erst ab dem
Folgetag: Eine Meldung vom Tag T könnte nach der Entscheidung erschienen sein. Danach läuft alles über
dieselben Funktionen wie live (`build_snapshot`, `run_tick`), mit eigener Datenbank und eigenem
Snapshot-Ordner.

Look-ahead durch das Modell selbst: Ein Modell kennt aus dem Training, was vor seinem Wissensstichtag
passiert ist. Aussagekräftig ist ein Backtest deshalb nur für die Zeit **nach** dem Stichtag aller
verglichenen Modelle.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from trading_agent.config import Watchlist
from trading_agent.feeds.news import NewsItem, RssNewsProvider, deduplicate, parse_feed, within
from trading_agent.feeds.prices import Bar, Quote

log = logging.getLogger(__name__)

DEFAULT_ROOT = Path("data") / "backtest"
DECIDE_AT = "06:30:00"  # UTC, vor der Xetra-Eröffnung
# Vereinfachte Sitzungszeiten in UTC (Winterzeit). Sie bestimmen nur die Kurszeit `as_of`, die Kurse selbst
# kommen aus den Tageskerzen.
CLOSE_UTC = {"XETRA": "16:30:00", "NYSE": "21:00:00", "NASDAQ": "21:00:00"}
OPEN_UTC = {"XETRA": "08:00:00", "NYSE": "14:30:00", "NASDAQ": "14:30:00"}


class BacktestError(RuntimeError):
    pass


def decision_time(day: date) -> datetime:
    return datetime.fromisoformat(f"{day.isoformat()}T{DECIDE_AT}+00:00")


class DailyBarsProvider:
    """PriceProvider über heruntergeladene Tageskerzen, eingefroren auf einen Handelstag.

    `side="close"`: was vor Börsenstart am Tag `day` bekannt war (letzter Schlusskurs vor `day`).
    `side="open"`: der erste Eröffnungskurs ab `day` (Ausführungskurs).
    """

    name = "yfinance-daily"

    def __init__(self, bars: dict[str, list[Bar]], meta: dict[str, tuple[str, str]], day: date, side: str):
        if side not in ("close", "open"):
            raise ValueError(side)
        self.bars, self.meta, self.day, self.side = bars, meta, day.isoformat(), side

    def _before(self, symbol: str) -> list[Bar]:
        return [b for b in self.bars.get(symbol, []) if b.date < self.day]

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        out = {}
        for s in symbols:
            currency, exchange = self.meta[s]
            prior = self._before(s)
            if self.side == "close":
                if not prior:
                    continue
                b = prior[-1]
                prev = prior[-2].close if len(prior) > 1 else None
                out[s] = Quote(s, b.close, prev, currency, f"{b.date}T{CLOSE_UTC[exchange]}+00:00")
            else:
                later = [b for b in self.bars.get(s, []) if b.date >= self.day]
                if not later:
                    continue
                b = later[0]
                prev = prior[-1].close if prior else None
                out[s] = Quote(s, b.open, prev, currency, f"{b.date}T{OPEN_UTC[exchange]}+00:00")
        return out

    def history(self, symbols: list[str], days: int) -> dict[str, list[Bar]]:
        return {s: self._before(s)[-days:] for s in symbols if self._before(s)}

    def latest_as_of(self, symbols: list[str]) -> datetime | None:
        times = [datetime.fromisoformat(q.as_of) for q in self.quotes(symbols).values()]
        return max(times) if times else None


def symbol_meta(watchlist: Watchlist) -> dict[str, tuple[str, str]]:
    meta = {a.symbol: (a.currency, a.exchange) for a in watchlist.assets}
    meta[watchlist.fx_symbol] = ("USD", "XETRA")
    return meta


def download_bars(
    watchlist: Watchlist, start: date, end: date, history_days: int = 420
) -> dict[str, list[Bar]]:
    """Tageskerzen aller Werte und des Wechselkurses von `start - history_days` bis einschließlich `end`."""
    import yfinance as yf

    from trading_agent.feeds.prices import _clean

    symbols = [*(a.symbol for a in watchlist.assets), watchlist.fx_symbol]
    frame = yf.download(
        symbols,
        start=(start - timedelta(days=history_days)).isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    out: dict[str, list[Bar]] = {}
    for s in symbols:
        try:
            sub = frame[s]
        except KeyError:
            log.warning("keine Historie für %s", s)
            continue
        bars = []
        for idx, row in sub.iterrows():
            close = _clean(row.get("Close"))
            if close is None:
                continue
            bars.append(
                Bar(
                    idx.date().isoformat(),
                    _clean(row.get("Open")) or close,
                    _clean(row.get("High")) or close,
                    _clean(row.get("Low")) or close,
                    close,
                    _clean(row.get("Volume")) or 0.0,
                )
            )
        out[s] = bars
    return out


def trading_days(bars: dict[str, list[Bar]], start: date, end: date) -> list[date]:
    """Tage zwischen `start` und `end`, an denen mindestens eine Börse der Watchlist gehandelt hat."""
    days = {b.date for bs in bars.values() for b in bs if start.isoformat() <= b.date <= end.isoformat()}
    return [date.fromisoformat(d) for d in sorted(days) if date.fromisoformat(d).weekday() < 5]


def news_url(query: str, lang: str, day: date) -> str:
    """Google-News-RSS für Meldungen, die auf `day` datiert sind (after/before umschließen den Tag)."""
    from urllib.parse import quote_plus

    hl, gl, ceid = {"de": ("de", "DE", "DE:de"), "en": ("en-US", "US", "US:en")}[lang]
    window = f"after:{(day - timedelta(days=1)).isoformat()} before:{(day + timedelta(days=1)).isoformat()}"
    return (
        f"https://news.google.com/rss/search?q={quote_plus(query + ' ' + window)}&hl={hl}&gl={gl}&ceid={ceid}"
    )


class HistoricalNewsProvider(RssNewsProvider):
    """Google News für vergangene Tage. Rohantworten werden zwischengespeichert, damit jeder Tag nur einmal
    abgefragt wird."""

    name = "google-news-rss-history"

    def __init__(
        self,
        cache: Path,
        timeout: float = 15.0,
        pause: float = 1.0,
        fetch=None,
        retry_waits=(30.0, 120.0, 300.0),
    ):
        super().__init__(timeout)
        self.cache, self.pause, self.retry_waits = cache, pause, retry_waits
        self._fetch = fetch  # zum Testen: fetch(url) -> bytes

    def _get(self, client, url: str, day: date) -> bytes:
        path = self.cache / day.isoformat() / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".xml")
        if path.exists():
            return path.read_bytes()
        raw = self._fetch(url) if self._fetch is not None else self._download(client, url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return raw

    def _download(self, client, url: str) -> bytes:
        """Mit Pausen wiederholen: Google antwortet bei vielen Anfragen mit 429, dazu kommen Netzaussetzer."""
        import httpx

        for wait in (*self.retry_waits, None):
            try:
                resp = client.get(url)
                resp.raise_for_status()
                time.sleep(self.pause)  # Google nicht mit Anfragen überhäufen
                return resp.content
            except httpx.HTTPError as e:
                if wait is None:
                    raise
                log.warning("News-Abruf fehlgeschlagen (%s), neuer Versuch in %.0f s", e, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def fetch(
        self, watchlist: Watchlist, now: datetime, hours: float = 24
    ) -> tuple[list[NewsItem], list[str]]:
        import httpx

        from trading_agent.feeds.news import USER_AGENT, alias_patterns

        day = (now - timedelta(days=1)).date()  # nur Meldungen vom Vortag, siehe Moduldoku
        cutoff = datetime.combine(now.date(), datetime.min.time(), UTC)
        patterns = alias_patterns(watchlist)
        items: list[NewsItem] = []
        with httpx.Client(
            timeout=self.timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True
        ) as c:
            for query, lang in self.queries(watchlist):
                try:
                    raw = self._get(c, news_url(query, lang, day), day)
                except httpx.HTTPError as e:
                    # Ein Tag mit Lücken würde den Vergleich verfälschen: abbrechen, der Lauf macht später
                    # an dieser Stelle weiter.
                    raise BacktestError(f"{day}: News für {query!r} nicht abrufbar ({e})") from e
                items.extend(
                    i for i in parse_feed(raw, query, patterns)
                    if i.published and datetime.fromisoformat(i.published) < cutoff
                )  # fmt: skip
        return deduplicate(within(items, now, hours)), []


def build_day(watchlist: Watchlist, bars: dict, day: date, news_provider=None) -> tuple[dict, dict]:
    """Snapshot (vor Börsenstart am Tag `day`) und Ausführungskurse (Eröffnung) für einen Backtest-Tag."""
    from trading_agent.snapshot import build_fill_quotes, build_snapshot

    meta = symbol_meta(watchlist)
    seen = DailyBarsProvider(bars, meta, day, "close")
    snap = build_snapshot(watchlist, seen, news_provider, now=decision_time(day), news_hours=24)
    fills = DailyBarsProvider(bars, meta, day, "open")
    fill_at = fills.latest_as_of(list(meta)) or decision_time(day)
    fill = build_fill_quotes(watchlist, fills, snap, now=max(fill_at, decision_time(day)))
    return snap, fill
