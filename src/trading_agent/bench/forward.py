"""Forward-Score (PLAN.md 5.2): Was hat eine Entscheidung nach h Handelstagen gebracht, verglichen mit
Nichtstun?

Je ausgeführter Order, in Euro und in % des Depotwerts bei der Entscheidung, nach Gebühren, ohne Steuern:

- Kauf für x €:     x × (Rendite des Werts − Rendite des Vergleichsmarkts) − Gebühr
- Verkauf für x €:  x × (Rendite des Vergleichsmarkts − Rendite des Werts) − Gebühr
- Halten:           0 (Nulllinie)

Die Rendite des Werts startet beim tatsächlichen Ausführungskurs (inklusive Spread) und endet beim Schlusskurs
des h-ten Handelstags ab dem Ausführungstag. Der Vergleichsmarkt ist das gleichgewichtete Watchlist-Depot über
dasselbe Fenster (Eröffnung des Ausführungstags bis Schlusskurs). Gezählt wird nur, ob der gewählte
Wert besser lief als die Alternativen, nicht ob der Gesamtmarkt gestiegen ist.

Je Takt wird die Summe über seine Orders gebildet (Halten = 0). Konfidenzintervalle kommen aus dem
Block-Bootstrap über Handelswochen (harness-it-all), Vergleiche zweier Agenten gepaart über dieselben
Snapshots.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime

from harness_it_all.stats import Estimate, block_bootstrap, paired_block_bootstrap

from trading_agent.config import Watchlist
from trading_agent.feeds.prices import Bar

HORIZONS = (1, 5, 20)
MAIN_HORIZON = 5


@dataclass(frozen=True)
class Day:
    date: str
    open_eur: float
    close_eur: float


def eur_series(bars: dict[str, list[Bar]], watchlist: Watchlist) -> dict[str, list[Day]]:
    """Tageskerzen je Wert in Euro (USD-Werte mit dem EURUSD-Schlusskurs desselben oder des letzten Tages)."""
    fx = bars.get(watchlist.fx_symbol, [])
    fx_dates = [b.date for b in fx]

    def rate(day: str) -> float | None:
        i = bisect_right(fx_dates, day) - 1
        return fx[i].close if i >= 0 else None

    out: dict[str, list[Day]] = {}
    for a in watchlist.assets:
        days = []
        for b in bars.get(a.symbol, []):
            r = 1.0 if a.currency == "EUR" else rate(b.date)
            if r:
                k = 1.0 if a.currency == "EUR" else 1 / r
                days.append(Day(b.date, b.open * k, b.close * k))
        out[a.id] = days
    return out


def _window(days: list[Day], start: str, h: int) -> tuple[Day, Day] | None:
    """Ausführungstag (erster Handelstag ab `start`) und h-ter Handelstag ab dort."""
    dates = [d.date for d in days]
    i = bisect_right(dates, start) - 1
    if i < 0 or dates[i] != start:
        i += 1  # Ausführung an einem Tag ohne Kerze: nächster Handelstag
    j = i + h - 1
    if i >= len(days) or j >= len(days):
        return None
    return days[i], days[j]


def market_return(series: dict[str, list[Day]], start: str, h: int) -> float | None:
    rets = []
    for days in series.values():
        w = _window(days, start, h)
        if w:
            rets.append(w[1].close_eur / w[0].open_eur - 1)
    return sum(rets) / len(rets) if rets else None


@dataclass(frozen=True)
class OrderScore:
    depot: str
    snapshot_at: str
    asset: str
    side: str
    horizon: int
    amount_eur: float
    asset_return: float
    market_return: float
    score_eur: float
    score_pct: float  # in % des Depotwerts bei der Entscheidung (0.01 = 1 %)


def score_fill(
    fill: dict, value_at_decision: float, series: dict[str, list[Day]], h: int
) -> OrderScore | None:
    start = fill["at"][:10]
    days = series.get(fill["asset"])
    w = _window(days or [], start, h)
    m = market_return(series, start, h)
    if w is None or m is None:
        return None  # Horizont reicht über die vorhandenen Kurse hinaus
    a = w[1].close_eur / fill["exec_price_eur"] - 1
    x = fill["gross_eur"]
    edge = a - m if fill["side"] == "buy" else m - a
    score = x * edge - fill["fee_eur"]
    return OrderScore(
        fill["depot"], fill["decided_at"], fill["asset"], fill["side"], h, x, a, m, score,
        score / value_at_decision,
    )  # fmt: skip


@dataclass(frozen=True)
class TickScores:
    """Forward-Score je Takt eines Depots: Schlüssel ist der Snapshot-Hash."""

    depot: str
    horizon: int
    by_snapshot: dict[str, float]  # Summe der Orders in % des Depots
    week_of: dict[str, str]
    orders: list[OrderScore]

    def estimate(self) -> Estimate:
        keys = sorted(self.by_snapshot)
        return block_bootstrap([self.by_snapshot[k] for k in keys], [self.week_of[k] for k in keys])

    @property
    def total_eur(self) -> float:
        return sum(o.score_eur for o in self.orders)

    @property
    def before_fees_eur(self) -> float:
        return sum(
            o.amount_eur * (o.asset_return - o.market_return) * (1 if o.side == "buy" else -1)
            for o in self.orders
        )

    @property
    def hit_rate(self) -> float | None:
        return sum(o.score_eur > 0 for o in self.orders) / len(self.orders) if self.orders else None


def _week(at: str) -> str:
    y, w, _ = datetime.fromisoformat(at).isocalendar()
    return f"{y}-W{w:02d}"


def depot_scores(
    rows: list[tuple[str, str, dict]], depot: str, series: dict[str, list[Day]], h: int
) -> TickScores:
    """`rows`: Entscheidungen des Depots als (Snapshot-Hash, Zeitpunkt, Payload aus der Datenbank)."""
    value_of: dict[str, float] = {}
    fills: list[dict] = []
    for _, _, payload in rows:
        for o in payload.get("orders", []):
            value_of[o["id"]] = o["value_at_decision"]
        fills.extend(payload.get("fills", []))
    snapshot_of = {at: snap for snap, at, _ in rows}
    calendar = sorted({d.date for days in series.values() for d in days})

    by_snapshot: dict[str, float] = {}
    week_of: dict[str, str] = {}
    unscorable: set[str] = set()
    for snap, at, _ in rows:
        # Auch Halten zählt nur, wenn der Horizont ab diesem Takt vollständig in den Kursen liegt
        i = bisect_right(calendar, at[:10])
        if i + h - 1 >= len(calendar):
            unscorable.add(snap)
            continue
        by_snapshot[snap] = 0.0
        week_of[snap] = _week(at)
    orders = []
    for f in fills:
        snap = snapshot_of.get(f["decided_at"])
        if snap is None or snap in unscorable:
            continue
        s = score_fill(f, value_of.get(f["order_id"], 0.0) or 1.0, series, h)
        if s is None:
            by_snapshot.pop(snap, None)
            unscorable.add(snap)
            continue
        orders.append(s)
        by_snapshot[snap] += s.score_pct
    orders = [o for o in orders if snapshot_of[o.snapshot_at] in by_snapshot]
    return TickScores(depot, h, by_snapshot, week_of, orders)


def load_rows(store, depot: str) -> list[tuple[str, str, dict]]:
    rows = store.db.execute(
        "SELECT snapshot, at, payload FROM decisions WHERE depot = ? ORDER BY at", (depot,)
    ).fetchall()
    return [(s, at, json.loads(p)) for s, at, p in rows]


def paired(a: TickScores, b: TickScores) -> Estimate:
    """Differenz A − B je Takt auf denselben Snapshots, Blöcke = Handelswochen."""
    return paired_block_bootstrap(a.by_snapshot, b.by_snapshot, a.week_of | b.week_of)
