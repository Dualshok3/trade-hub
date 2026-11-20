"""Prompts des Agenten. Jede inhaltliche Änderung erhöht `PROMPT_VERSION`, damit Läufe vergleichbar
bleiben."""

from __future__ import annotations

import json

from trading_agent.agent.config import AgentConfig
from trading_agent.depot import Depot
from trading_agent.tools import SnapshotTools

PROMPT_VERSION = "decide-v1"

SYSTEM = """Du verwaltest ein simuliertes Aktiendepot mit Spielgeld. Ziel: über die nächsten Tage besser
abschneiden als der Gesamtmarkt, nach Gebühren. Nichts tun ist eine gültige und oft gute Entscheidung.

Regeln (werden im Code geprüft, Verstöße werden abgelehnt):
- Jede Order in ganzen Prozent des aktuellen Depotwerts (Feld pct). Mindestens {min_pct} %.
- Käufe: höchstens {max_pct} % pro Order, und kein Wert darf danach mehr als {max_pct} % des Depots ausmachen.
- Verkäufe: pct ist der Anteil des Depotwerts, der verkauft werden soll. Mehr als der Bestand verkauft alles.
- Höchstens {max_orders} Orders pro Takt, höchstens eine Order pro Wert, Cash darf nicht negativ werden.
- Jeder Kauf braucht mindestens eine Quelle: eine News-ID aus dem Kontext (z. B. "{example_id}") oder
  "indicator:<name>" mit name aus {indicators}. Erfinde keine IDs.
- Kosten: {fee:.2f} € Gebühr pro Order, {spread:.1f} % halber Spread. Auf Gewinne beim Verkauf fallen
  {tax:.3f} % Abgeltungsteuer an.
- Ausgeführt wird erst zu einem späteren Kurs. Geschlossene Börsen: Die Order wartet.

Antworte nur mit JSON nach dem vorgegebenen Schema:
- assessment: 1–3 Sätze zur Lage
- orders: Liste der Orders, leer heißt halten
- note: kurze Notiz für dich selbst im nächsten Takt (Erwartung, worauf du achtest)"""


def system_prompt(depot: Depot, cfg: AgentConfig, example_id: str = "N1a2b3c4d5e") -> str:
    from trading_agent.agent.schema import INDICATORS

    r = depot.rules
    return SYSTEM.format(
        min_pct=round(r.min_order_pct * 100),
        max_pct=round(cfg.risk.max_weight * 100),
        max_orders=cfg.risk.max_orders,
        example_id=example_id,
        indicators=", ".join(INDICATORS),
        fee=r.fee_eur,
        spread=r.half_spread * 100,
        tax=r.tax_rate * 100,
    )


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:+.1f} %"


def portfolio_block(depot: Depot, prices: dict[str, float]) -> str:
    s = depot.summary(prices)
    lines = [
        f"Depotwert {s['value_eur']:.2f} € (Start {depot.start_eur:.2f} €, {_pct(s['return'])}), "
        f"Cash {s['cash_eur']:.2f} €, Gebühren bisher {s['fees_eur']:.2f} €, "
        f"Steuern bisher {s['taxes_eur']:.2f} €"
    ]
    for a, p in s["positions"].items():
        w = p["value_eur"] / s["value_eur"] if s["value_eur"] else 0
        pl = p["value_eur"] / p["cost_eur"] - 1 if p["cost_eur"] else None
        lines.append(f"- {a}: {p['value_eur']:.2f} € ({w * 100:.0f} % des Depots), seit Kauf {_pct(pl)}")
    if not s["positions"]:
        lines.append("- keine Positionen, nur Cash")
    return "\n".join(lines)


def overview_block(rows: list[dict]) -> str:
    out = ["Wert | Kurs € | Tag | 5 Tage | RSI | über SMA50 | News 24 h | Börse offen"]
    for r in rows:
        price = "–" if r["price_eur"] is None else f"{r['price_eur']:.2f}"
        out.append(
            f"{r['asset']} | {price} | {_pct(r['day_change'])} | {_pct(r['change_5d'])} | "
            f"{r['rsi14'] if r['rsi14'] is not None else '–'} | "
            f"{'ja' if r['above_sma50'] else 'nein'} | {r['news_24h']} | "
            f"{'ja' if r['market_open'] else 'nein'}"
        )
    return "\n".join(out)


def _num(x: float | None) -> str:
    return "–" if x is None else f"{x:.2f}"


def _news_lines(items: list[dict]) -> list[str]:
    return [f"  [{n['id']}] {n['published'][:16]} {n['source']}: {n['title']}" for n in items]


def focus_block(tools: SnapshotTools, asset: str, news: list[dict]) -> str:
    q, i = tools.get_quote(asset), tools.get_indicators(asset)
    keys = ("rsi14", "sma20", "sma50", "sma200", "volatility20", "change_20d", "from_52w_high")
    ind = {k: i[k] for k in keys}
    lines = [
        f"### {asset} ({q.get('name', '')})",
        f"Kurs {_num(q.get('price'))} {q.get('currency')} ({_num(q.get('price_eur'))} €), "
        f"Tag {_pct(q.get('day_change'))}, "
        f"Kurs von {q.get('as_of')}, Börse {'offen' if q.get('market_open') else 'zu'}",
        "Indikatoren: " + json.dumps(ind, ensure_ascii=False),
        "News:" if news else "News: keine relevanten",
        *_news_lines(news),
    ]
    return "\n".join(lines)


def user_prompt(
    snapshot: dict,
    portfolio: str,
    overview: str,
    focus: list[str],
    market_news: list[dict],
    journal: list[str],
) -> str:
    parts = [
        f"Zeitpunkt: {snapshot['taken_at']} (UTC). "
        f"Offene Börsen: {', '.join(snapshot['open_exchanges']) or 'keine'}.",
        "## Depot",
        portfolio,
        "## Watchlist",
        overview,
        "## Marktnachrichten",
        "\n".join(_news_lines(market_news)) or "keine",
        "## Werte im Detail",
        "\n\n".join(focus) or "keine",
    ]
    if journal:
        parts += ["## Deine Notizen aus früheren Takten", "\n".join(f"- {n}" for n in journal)]
    parts.append("Entscheide jetzt.")
    return "\n\n".join(parts)


def rejection_prompt(violations: list[str], left: int) -> str:
    return (
        "Der Risk-Check hat deine Entscheidung abgelehnt:\n"
        + "\n".join(f"- {v}" for v in violations)
        + f"\n\nKorrigiere sie. Noch {left} Versuch(e), danach wird automatisch gehalten."
    )
