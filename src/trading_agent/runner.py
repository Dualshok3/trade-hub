"""Ein Takt: Entscheidungen aller Depots auf einem Snapshot, Ausführung zu eingefrorenen Ausführungskursen.

Live und Replay nutzen dieselbe Funktion. Der Unterschied ist nur, woher Snapshot und Ausführungskurse kommen
(frisch abgerufen oder von der Platte geladen).
"""

from __future__ import annotations

from trading_agent.depot import DepotError, DepotRules
from trading_agent.execution import execute, make_order
from trading_agent.store import DepotStore


def _prices(snapshot: dict, fill_quotes: dict) -> dict[str, float]:
    """Bewertungskurse: Ausführungskurs, sonst Snapshot-Kurs."""
    prices = {a: q["price_eur"] for a, q in snapshot["quotes"].items() if q.get("price_eur") is not None}
    prices.update(
        {a: q["price_eur"] for a, q in fill_quotes["quotes"].items() if q.get("price_eur") is not None}
    )
    return prices


def run_tick(
    snapshot: dict,
    fill_quotes: dict,
    store: DepotStore,
    strategies: dict,
    start_eur: float,
    rules: DepotRules,
) -> dict[str, dict]:
    """Alle Strategien auf einem Snapshot laufen lassen. Ergebnis je Depot: Orders, Fills, Ablehnungen …"""
    if fill_quotes["snapshot"] != snapshot["hash"]:
        raise ValueError("Ausführungskurse gehören zu einem anderen Snapshot")
    results = {}
    existing = set(store.names())
    for strategy_name, strategy in strategies.items():
        name = f"{strategy_name}-{int(start_eur)}"
        if name not in existing:
            store.open(name, start_eur, rules, snapshot["taken_at"])
        if store.done(name, snapshot["hash"]):
            continue  # dieser Takt ist für dieses Depot schon gelaufen
        depot = store.load(name)
        orders, rejected = store.pending(name), []
        waiting = {(o.asset, o.side) for o in orders}
        for asset, side, pct, reason in strategy(snapshot, depot):
            if (asset, side) in waiting:
                continue  # gleiche Order wartet schon auf einen neueren Kurs
            waiting.add((asset, side))
            try:
                orders.append(make_order(depot, snapshot, asset, side, pct, reason))
            except DepotError as e:
                rejected.append({"asset": asset, "side": side, "pct": pct, "reason": str(e)})
        outcome = execute(depot, orders, fill_quotes)
        store.record_fills(outcome.fills)
        store.set_pending(name, outcome.pending)
        prices = _prices(snapshot, fill_quotes)
        value = depot.value(prices)
        store.mark(name, fill_quotes["taken_at"], snapshot["hash"], value, depot.cash)
        result = {
            "orders": [o.to_dict() for o in orders],
            "fills": [f.to_dict() for f in outcome.fills],
            "pending": [o.id for o in outcome.pending],
            "expired": [o.id for o in outcome.expired],
            "rejected": rejected + [{"order": o.id, "reason": r} for o, r in outcome.rejected],
            "value_eur": round(value, 4),
            "summary": depot.summary(prices),
        }
        store.record_decision(name, snapshot["hash"], snapshot["taken_at"], result)
        results[name] = result
    return results
