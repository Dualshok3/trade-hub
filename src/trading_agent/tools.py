"""Werkzeuge für den Agenten. Sie lesen ausschließlich aus einem eingefrorenen Snapshot, nie aus dem Netz.

Das ist die Voraussetzung dafür, dass Live-Takte später exakt wieder abgespielt werden können.
Die Gemma-Engine kann keine nativen Tool-Aufrufe; der Graph ruft diese Funktionen deshalb selbst auf
und gibt die Ergebnisse als Kontext weiter. Gerechnet wird nur hier, nie im Modell.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from trading_agent import indicators as ind


class ToolError(ValueError):
    pass


def _r(x, nd=4):
    return None if x is None else round(x, nd)


class SnapshotTools:
    def __init__(self, snapshot: dict):
        self.s = snapshot
        self.assets = {a["id"]: a for a in snapshot["watchlist"]}
        self.now = datetime.fromisoformat(snapshot["taken_at"])

    def _asset(self, asset_id: str) -> dict:
        if asset_id not in self.assets:
            raise ToolError(f"{asset_id!r} ist nicht in der Watchlist")
        return self.assets[asset_id]

    def get_quote(self, asset_id: str) -> dict:
        asset = self._asset(asset_id)
        q = self.s["quotes"].get(asset_id)
        if q is None:
            return {"asset": asset_id, "error": "kein Kurs in diesem Snapshot"}
        day = None
        if q.get("previous_close"):
            day = q["price"] / q["previous_close"] - 1
        return {
            "asset": asset_id,
            "name": asset["name"],
            "price": q["price"],
            "currency": q["currency"],
            "price_eur": q["price_eur"],
            "day_change": _r(day),
            "as_of": q["as_of"],
            "market_open": asset["exchange"] in self.s["open_exchanges"],
        }

    def get_history(self, asset_id: str, days: int = 20) -> list[dict]:
        self._asset(asset_id)
        if days < 1:
            raise ToolError("days muss mindestens 1 sein")
        return self.s["history"].get(asset_id, [])[-days:]

    def get_indicators(self, asset_id: str) -> dict:
        self._asset(asset_id)
        bars = self.s["history"].get(asset_id, [])
        closes = [b["close"] for b in bars]
        highs = [b["high"] for b in bars]
        last = closes[-1] if closes else None
        sma = {n: ind.sma(closes, n) for n in (20, 50, 200)}
        return {
            "asset": asset_id,
            "last_close": last,
            "sma20": _r(sma[20]),
            "sma50": _r(sma[50]),
            "sma200": _r(sma[200]),
            "above_sma50": None if sma[50] is None or last is None else last > sma[50],
            "above_sma200": None if sma[200] is None or last is None else last > sma[200],
            "rsi14": _r(ind.rsi(closes), 1),
            "volatility20": _r(ind.volatility(closes)),
            "change_5d": _r(ind.change(closes, 5)),
            "change_20d": _r(ind.change(closes, 20)),
            "from_52w_high": _r(ind.from_high(closes, highs)),
            "bars": len(closes),
        }

    def get_news(self, asset_id: str | None = None, hours: float = 24, limit: int = 10) -> list[dict]:
        """Meldungen zu einem Wert (oder alle mit `None`) aus den letzten `hours` Stunden vor dem Snapshot."""
        if asset_id is not None:
            self._asset(asset_id)
        start = self.now - timedelta(hours=hours)
        out = []
        for n in self.s["news"]:
            if not n["published"] or not (start <= datetime.fromisoformat(n["published"]) <= self.now):
                continue
            if asset_id is not None and asset_id not in n["assets"]:
                continue
            out.append({k: n[k] for k in ("id", "published", "title", "snippet", "source", "assets")})
        return out[:limit]

    def news_ids(self) -> set[str]:
        """Alle gültigen News-IDs dieses Snapshots (zum Prüfen zitierter Quellen)."""
        return {n["id"] for n in self.s["news"]}

    def overview(self) -> list[dict]:
        """Eine Zeile pro Wert: Kurs, Tagesänderung, Trend, Anzahl News. Grundlage des Watchlist-Checks."""
        rows = []
        for asset_id in self.assets:
            q = self.get_quote(asset_id)
            i = self.get_indicators(asset_id)
            rows.append(
                {
                    "asset": asset_id,
                    "price_eur": q.get("price_eur"),
                    "day_change": q.get("day_change"),
                    "change_5d": i["change_5d"],
                    "rsi14": i["rsi14"],
                    "above_sma50": i["above_sma50"],
                    "news_24h": len(self.get_news(asset_id, 24, limit=1000)),
                    "market_open": q.get("market_open"),
                }
            )
        return rows
