"""Vergleichs-Depots ohne LLM. Sie handeln nach festen Regeln und zahlen dieselben Gebühren und Steuern.

Schlagen die LLM-Agenten diese einfachen Regeln nicht, ist das ein ehrliches Ergebnis.
Jede Strategie liefert Entscheidungen als (Wert, Seite, Anteil am Depotwert, Begründung).
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable

from trading_agent.depot import Depot
from trading_agent.tools import SnapshotTools

Decision = tuple[str, str, float, str]
Strategy = Callable[[dict, Depot], list[Decision]]


def cash_only(snapshot: dict, depot: Depot) -> list[Decision]:
    return []


def buy_and_hold_etf(snapshot: dict, depot: Depot, etf: str = "MSCIW") -> list[Decision]:
    if depot.shares(etf) > 0 or depot.fills or etf not in snapshot["quotes"]:
        return []
    return [(etf, "buy", 1.0, f"Einmal alles in {etf}, dann halten")]


def equal_weight_etfs(snapshot: dict, depot: Depot) -> list[Decision]:
    if depot.fills:
        return []
    etfs = [a["id"] for a in snapshot["watchlist"] if a["kind"] == "etf" and a["id"] in snapshot["quotes"]]
    return [(e, "buy", 1 / len(etfs), "Gleichgewichtet in alle ETFs, dann halten") for e in etfs]


def sma_trend(snapshot: dict, depot: Depot, top: int = 3) -> list[Decision]:
    """Aktien im Aufwärtstrend (Kurs > SMA50 und SMA20 > SMA50), die 3 mit dem stärksten 20-Tage-Momentum."""
    tools = SnapshotTools(snapshot)
    candidates = []
    for a in snapshot["watchlist"]:
        if a["kind"] != "stock" or a["id"] not in snapshot["quotes"]:
            continue
        i = tools.get_indicators(a["id"])
        if None in (i["sma20"], i["sma50"], i["change_20d"]):
            continue
        if i["last_close"] > i["sma50"] and i["sma20"] > i["sma50"]:
            candidates.append((i["change_20d"], a["id"]))
    target = {asset for _, asset in sorted(candidates, reverse=True)[:top]}
    held = set(depot.positions())
    out: list[Decision] = [(a, "sell", 1.0, "Trend gebrochen") for a in sorted(held - target)]
    out += [(a, "buy", 1 / top, "Neu im Trend") for a in sorted(target - held)]
    return out


def random_trader(snapshot: dict, depot: Depot) -> list[Decision]:
    """Zufällige Orders, aber reproduzierbar (Seed aus Depotname und Snapshot)."""
    seed = int(hashlib.sha1(f"{depot.name}|{snapshot['hash']}".encode()).hexdigest()[:12], 16)
    rng = random.Random(seed)
    assets = sorted(snapshot["quotes"])
    roll = rng.random()
    if roll < 0.15:
        return [(rng.choice(assets), "buy", 0.25, "Zufall")]
    held = sorted(depot.positions())
    if roll < 0.25 and held:
        return [(rng.choice(held), "sell", 1.0, "Zufall")]
    return []


BASELINES: dict[str, Strategy] = {
    "cash": cash_only,
    "hold-msciw": buy_and_hold_etf,
    "eq-etfs": equal_weight_etfs,
    "sma-trend": sma_trend,
    "random": random_trader,
}
