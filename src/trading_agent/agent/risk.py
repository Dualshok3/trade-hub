"""Risk-Check: harte Regeln im Code, kein LLM.

Jeder Verstoß wird als verständlicher Satz zurückgegeben. Der Graph gibt ihn ans Modell zurück und zählt ihn.
Alle Grenzen gelten in Prozent des Depotwerts, damit 100 € und 10.000 € unter denselben Bedingungen handeln.
"""

from __future__ import annotations

from trading_agent.agent.config import RiskConfig
from trading_agent.agent.schema import INDICATORS
from trading_agent.depot import EPS, Depot


def valid_source(source: str, news_ids: set[str]) -> bool:
    """Eine Quelle ist eine News-ID aus dem Snapshot oder `indicator:<name>`."""
    if source in news_ids:
        return True
    kind, _, name = source.partition(":")
    return kind == "indicator" and name in INDICATORS


def check(
    orders: list[dict],
    depot: Depot,
    prices: dict[str, float],
    watchlist: set[str],
    news_ids: set[str],
    limits: RiskConfig,
) -> list[str]:
    rules = depot.rules
    value = depot.value(prices)
    weights = depot.weights(prices)
    min_pct = round(rules.min_order_pct * 100)
    max_pct = round(limits.max_weight * 100)
    out: list[str] = []

    if len(orders) > limits.max_orders:
        out.append(f"{len(orders)} Orders, erlaubt sind höchstens {limits.max_orders} pro Takt.")
    seen: set[str] = set()
    cash = depot.cash
    for i, o in enumerate(orders, start=1):
        asset, side, pct = o["asset"], o["side"], o["pct"]
        tag = f"Order {i} ({side} {asset} {pct} %)"
        if asset not in watchlist:
            out.append(f"{tag}: {asset} ist nicht in der Watchlist.")
            continue
        if prices.get(asset) is None:
            out.append(f"{tag}: kein Kurs für {asset} in diesem Snapshot.")
            continue
        if asset in seen:
            out.append(f"{tag}: {asset} kommt mehrfach vor, höchstens eine Order pro Wert.")
        seen.add(asset)
        if pct < min_pct:
            out.append(f"{tag}: Mindestorder sind {min_pct} % des Depots.")
        bad = [s for s in o.get("sources", []) if not valid_source(s, news_ids)]
        if bad:
            out.append(
                f"{tag}: unbekannte Quellen {bad}. "
                "Erlaubt sind News-IDs aus dem Kontext und indicator:<name>."
            )
        amount = pct / 100 * value
        if side == "buy":
            if pct > max_pct:
                out.append(f"{tag}: höchstens {max_pct} % des Depots pro Order und Wert.")
            elif weights.get(asset, 0.0) + pct / 100 > limits.max_weight + EPS:
                have = round(weights.get(asset, 0.0) * 100)
                out.append(f"{tag}: {asset} hätte danach über {max_pct} % Gewicht (jetzt {have} %).")
            if not any(valid_source(s, news_ids) for s in o.get("sources", [])):
                out.append(f"{tag}: Jeder Kauf braucht mindestens eine gültige Quelle.")
            cash -= amount * (1 + rules.half_spread) + rules.fee_eur
        else:
            held = depot.shares(asset) * prices[asset]
            if held <= EPS:
                out.append(f"{tag}: {asset} ist nicht im Depot.")
                continue
            if pct > 100:
                out.append(f"{tag}: mehr als 100 % geht nicht.")
            cash += min(amount, held) * (1 - rules.half_spread) - rules.fee_eur
    if cash < -EPS:
        out.append(f"Nicht genug Cash: Nach allen Orders und Gebühren fehlen {-cash:.2f} €.")
    return out
