"""Kommandozeile: `trading-agent snapshot`, `trading-agent show`, `trading-agent list`."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from trading_agent.config import DEFAULT_WATCHLIST, load_watchlist
from trading_agent.snapshot import DEFAULT_DIR, build_snapshot, list_snapshots, load_snapshot, save_snapshot
from trading_agent.tools import SnapshotTools

app = typer.Typer(
    no_args_is_help=True, help="Trading-Agent: Musterdepot mit Spielgeld, lokales Modell gegen Opus."
)
console = Console()


def _pct(x):
    return "–" if x is None else f"{x * 100:+.2f} %"


def _print_overview(snap: dict) -> None:
    tools = SnapshotTools(snap)
    t = Table(title=f"Snapshot {snap['taken_at']} · {snap['hash'][:12]}")
    for col in ("Wert", "Kurs €", "Tag", "5 Tage", "RSI", "> SMA50", "News 24 h", "offen"):
        t.add_column(col, justify="right" if col not in ("Wert",) else "left")
    for r in tools.overview():
        t.add_row(
            r["asset"],
            "–" if r["price_eur"] is None else f"{r['price_eur']:.2f}",
            _pct(r["day_change"]),
            _pct(r["change_5d"]),
            "–" if r["rsi14"] is None else f"{r['rsi14']:.0f}",
            {True: "ja", False: "nein", None: "–"}[r["above_sma50"]],
            str(r["news_24h"]),
            "ja" if r["market_open"] else "",
        )
    console.print(t)
    console.print(
        f"News gesamt: {len(snap['news'])} · offene Börsen: {', '.join(snap['open_exchanges']) or 'keine'}"
    )
    if snap["errors"]:
        console.print(f"[yellow]{len(snap['errors'])} Hinweise:[/yellow]")
        for e in snap["errors"][:20]:
            console.print(f"  · {e}")


@app.command()
def snapshot(
    watchlist: Path = typer.Option(DEFAULT_WATCHLIST, help="Watchlist (TOML)"),
    out: Path = typer.Option(DEFAULT_DIR, help="Zielordner"),
    news_hours: float = typer.Option(24, help="News der letzten N Stunden"),
    no_news: bool = typer.Option(False, help="Ohne News (nur Kurse)"),
) -> None:
    """Live-Kurse und News holen, als Snapshot einfrieren und speichern."""
    from trading_agent.feeds.news import RssNewsProvider
    from trading_agent.feeds.prices import YFinanceProvider

    logging.basicConfig(level=logging.WARNING)
    wl = load_watchlist(watchlist)
    snap = build_snapshot(
        wl, YFinanceProvider(), None if no_news else RssNewsProvider(), news_hours=news_hours
    )
    path = save_snapshot(snap, out)
    _print_overview(snap)
    console.print(f"Gespeichert: {path}")


@app.command()
def show(path: Path) -> None:
    """Einen gespeicherten Snapshot prüfen (Hash) und anzeigen."""
    _print_overview(load_snapshot(path))


@app.command(name="list")
def list_cmd(root: Path = typer.Option(DEFAULT_DIR, help="Snapshot-Ordner")) -> None:
    """Alle gespeicherten Snapshots auflisten."""
    for p in list_snapshots(root):
        console.print(str(p))


if __name__ == "__main__":
    app()
