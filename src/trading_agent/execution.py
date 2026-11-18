"""Orders ausführen: in Prozent des Depots, zum eingefrorenen Ausführungskurs nach der Entscheidung.

Eine Order wird nur ausgeführt, wenn der Ausführungskurs **neuer** ist als der Kurs, den der Agent bei der
Entscheidung gesehen hat. Sonst (Börse zu, Kurs verzögert) wartet sie auf den nächsten Takt und verfällt nach
`pending_max_ticks` Takten. Live und Replay rufen dieselbe Funktion mit denselben eingefrorenen Kursen auf.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from datetime import datetime

from trading_agent.depot import EPS, Depot, DepotError, Fill


@dataclass(frozen=True)
class Order:
    """`pct`: Anteil des Depotwerts zum Entscheidungszeitpunkt.

    Ein Verkauf über mehr als den Bestand verkauft alles."""

    depot: str
    asset: str
    side: str  # buy | sell
    pct: float
    decided_at: str  # Zeitpunkt des Entscheidungs-Snapshots
    seen_as_of: str  # Kurszeit, die der Agent gesehen hat
    value_at_decision: float  # Depotwert bei der Entscheidung (für pct → Euro)
    ticks_waited: int = 0
    reason: str = ""

    @property
    def id(self) -> str:
        key = f"{self.depot}|{self.asset}|{self.side}|{self.pct}|{self.decided_at}"
        return "O" + hashlib.sha1(key.encode()).hexdigest()[:10]

    def to_dict(self) -> dict:
        return asdict(self) | {"id": self.id}


@dataclass(frozen=True)
class Outcome:
    fills: list[Fill]
    pending: list[Order]
    rejected: list[tuple[Order, str]]
    expired: list[Order]


def make_order(depot: Depot, snapshot: dict, asset: str, side: str, pct: float, reason: str = "") -> Order:
    """Order aus einer Entscheidung auf einem Snapshot bauen."""
    if side not in ("buy", "sell"):
        raise DepotError(f"unbekannte Seite {side!r}")
    if not 0 < pct <= 1:
        raise DepotError(f"Anteil {pct} muss zwischen 0 und 1 liegen")
    q = snapshot["quotes"].get(asset)
    if q is None:
        raise DepotError(f"{asset}: kein Kurs im Snapshot")
    prices = {a: v["price_eur"] for a, v in snapshot["quotes"].items()}
    return Order(
        depot=depot.name,
        asset=asset,
        side=side,
        pct=pct,
        decided_at=snapshot["taken_at"],
        seen_as_of=q["as_of"],
        value_at_decision=depot.value(prices),
        reason=reason,
    )


def _newer(fill_as_of: str, seen_as_of: str) -> bool:
    if not fill_as_of:
        return False
    if not seen_as_of:
        return True
    return datetime.fromisoformat(fill_as_of) > datetime.fromisoformat(seen_as_of)


def execute(depot: Depot, orders: list[Order], fill_quotes: dict) -> Outcome:
    """Orders der Reihe nach ausführen (Verkäufe zuerst, damit Cash für Käufe frei wird)."""
    rules = depot.rules
    fills, pending, rejected, expired = [], [], [], []
    ordered = sorted(orders, key=lambda o: o.side != "sell")
    for order in ordered:
        q = fill_quotes["quotes"].get(order.asset)
        if q is None or q.get("price_eur") is None or not _newer(q.get("as_of", ""), order.seen_as_of):
            if order.ticks_waited + 1 >= rules.pending_max_ticks:
                expired.append(order)
            else:
                pending.append(replace(order, ticks_waited=order.ticks_waited + 1))
            continue
        price = q["price_eur"]
        amount = order.pct * order.value_at_decision
        min_order = rules.min_order_pct * order.value_at_decision
        year = int(q["as_of"][:4])
        if order.side == "buy":
            amount = min(amount, depot.cash - rules.fee_eur)
            if amount < min_order - EPS:
                rejected.append(
                    (order, f"Kauf {amount:.2f} € unter Mindestorder {min_order:.2f} € oder Cash fehlt")
                )
                continue
            shares, exec_price, gross = depot.quote_buy(amount, price)
            fill = Fill(
                depot.name, order.id, order.asset, "buy", shares, price, exec_price, gross,
                rules.fee_eur, 0.0, 0.0, q["as_of"], order.decided_at,
            )  # fmt: skip
        else:
            held = depot.shares(order.asset)
            if held <= EPS:
                rejected.append((order, f"{order.asset} nicht im Depot"))
                continue
            shares = min(amount / price, held)
            if shares * price < min_order - EPS and shares < held - EPS:
                rejected.append(
                    (order, f"Verkauf {shares * price:.2f} € unter Mindestorder {min_order:.2f} €")
                )
                continue
            if held - shares < EPS * 1e3 or (held - shares) * price < 0.01:
                shares = held  # Reste unter einem Cent mitverkaufen
            pv = depot.sell_preview(order.asset, shares, price, year)
            fill = Fill(
                depot.name, order.id, order.asset, "sell", shares, price, pv["exec_price"], pv["gross"],
                rules.fee_eur, pv["tax"], pv["realized"], q["as_of"], order.decided_at,
            )  # fmt: skip
        depot.apply(fill)
        fills.append(fill)
    return Outcome(fills, pending, rejected, expired)
