"""Kommandozeile: snapshot, show, list, news, decide, tick, backtest, depots."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
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


@app.command()
def news(
    path: Path,
    asset: str = typer.Option(None, help="Nur dieser Wert (ID aus der Watchlist)"),
    limit: int = typer.Option(5, help="Meldungen pro Wert"),
    hours: float = typer.Option(24, help="Zeitfenster in Stunden"),
) -> None:
    """Die sortierten Top-News eines Snapshots anzeigen, so wie der Agent sie bekommt."""
    tools = SnapshotTools(load_snapshot(path))
    st = tools.ranker.stats()
    console.print(
        f"News: {st['total']} gesamt · {st['noise']} als Rauschen aussortiert · "
        f"{st['unassigned']} ohne Wert-Bezug"
    )
    ids = [asset] if asset else list(tools.assets)
    for asset_id in ids:
        top = tools.get_top_news(asset_id, hours, limit)
        console.print(f"\n[bold]{asset_id}[/bold] ({len(top)})")
        for n in top:
            extra = f" +{n['sources_count'] - 1} Quellen" if n["sources_count"] > 1 else ""
            console.print(_news_line(n) + extra)
    if not asset:
        console.print("\n[bold]Markt[/bold]")
        for n in tools.get_market_news(hours, limit):
            console.print(_news_line(n))


def _news_line(n: dict) -> str:
    return f"  {n['score']:.2f}  {n['published'][11:16]}  {n['source'][:18]:<18} {n['title']}"


@app.command(name="list")
def list_cmd(root: Path = typer.Option(DEFAULT_DIR, help="Snapshot-Ordner")) -> None:
    """Alle gespeicherten Snapshots auflisten."""
    for p in list_snapshots(root):
        console.print(str(p))


def _fill_ready(snap: dict, fill: dict) -> bool:
    """Ausführungskurse sind brauchbar, wenn jeder Wert an einer offenen Börse einen neueren Kurs hat."""
    from datetime import datetime

    open_ex = set(snap["open_exchanges"])
    for a in snap["watchlist"]:
        if a["exchange"] not in open_ex:
            continue
        seen = snap["quotes"].get(a["id"], {}).get("as_of")
        now = fill["quotes"].get(a["id"], {}).get("as_of")
        if seen and (not now or datetime.fromisoformat(now) <= datetime.fromisoformat(seen)):
            return False
    return True


@app.command()
def tick(
    watchlist: Path = typer.Option(DEFAULT_WATCHLIST, help="Watchlist (TOML)"),
    out: Path = typer.Option(DEFAULT_DIR, help="Snapshot-Ordner"),
    db: Path = typer.Option(Path("data") / "depots.sqlite", help="Depot-Datenbank"),
    start_eur: float = typer.Option(None, help="Startkapital (Standard: live_eur aus configs/depot.toml)"),
    wait: int = typer.Option(1200, help="Höchstens so viele Sekunden auf neuere Ausführungskurse warten"),
    no_news: bool = typer.Option(False, help="Ohne News"),
    agents: str = typer.Option("", help="LLM-Agenten aus configs/agent.toml, z. B. gemma,opus"),
) -> None:
    """Ein Live-Takt: Snapshot einfrieren, Depots entscheiden lassen, zu neueren Kursen ausführen."""
    import time
    import tomllib

    from trading_agent.baselines import BASELINES
    from trading_agent.depot import DEFAULT_RULES, load_rules
    from trading_agent.feeds.news import RssNewsProvider
    from trading_agent.feeds.prices import YFinanceProvider
    from trading_agent.runner import run_tick
    from trading_agent.snapshot import build_fill_quotes, save_fill_quotes
    from trading_agent.store import DepotStore

    logging.basicConfig(level=logging.WARNING)
    rules = load_rules()
    if start_eur is None:
        start_eur = tomllib.loads(DEFAULT_RULES.read_text(encoding="utf-8"))["sizes"]["live_eur"]
    wl = load_watchlist(watchlist)
    prices = YFinanceProvider()
    snap = build_snapshot(wl, prices, None if no_news else RssNewsProvider())
    save_snapshot(snap, out)
    store = DepotStore(db)
    strategies = dict(BASELINES) | _agents(agents, store)
    # Die Agenten entscheiden auf dem Snapshot; die Ausführung folgt erst zu neueren Kursen (run_tick)
    deadline = time.monotonic() + wait
    fill = build_fill_quotes(wl, prices, snap)
    # Xetra-Kurse sind bei yfinance rund 15 Minuten verzögert: so lange neu abfragen, bis jeder Wert an einer
    # offenen Börse einen neueren Kurs hat als im Snapshot, höchstens `wait` Sekunden. Erst danach warten
    # Orders auf den nächsten Takt.
    while not _fill_ready(snap, fill) and time.monotonic() < deadline:
        time.sleep(60)
        fill = build_fill_quotes(wl, prices, snap)
    save_fill_quotes(fill, snap, out)
    results = run_tick(snap, fill, store, strategies, start_eur, rules)
    t = Table(title=f"Takt {snap['taken_at']} · Ausführung {fill['taken_at']}")
    for col in ("Depot", "Wert €", "Rendite", "Orders", "ausgeführt", "wartend", "abgelehnt", "Gebühren €"):
        t.add_column(col, justify="left" if col == "Depot" else "right")
    for name, r in results.items():
        s = r["summary"]
        t.add_row(
            name,
            f"{s['value_eur']:.2f}",
            _pct(s["return"]),
            str(len(r["orders"])),
            str(len(r["fills"])),
            str(len(r["pending"])),
            str(len(r["rejected"])),
            f"{s['fees_eur']:.2f}",
        )
    console.print(t)
    for name, r in results.items():
        if "agent" in r:
            _print_trace(name, r["agent"])
    store.close()


def _agents(names: str, store=None, records: Path | None = None) -> dict:
    """LLM-Agenten als Strategien bauen. Jeder Aufruf wird in data/records/decisions.jsonl aufgezeichnet."""
    if not names.strip():
        return {}
    from harness_it_all.recorder import Recorder

    from trading_agent.agent import LLMAgent, load_agent_config
    from trading_agent.agent.models import build_model, load_profile

    cfg = load_agent_config()
    profile = load_profile(cfg)
    recorder = Recorder(records or Path("data") / "records" / "decisions.jsonl")
    out = {}
    for name in (n.strip() for n in names.split(",") if n.strip()):
        journal = (lambda depot, s=store: s.notes(depot, cfg.context.journal_notes)) if store else None
        out[name] = LLMAgent(build_model(name, cfg), profile, cfg, recorder, journal)
    return out


def _print_trace(name: str, tr: dict) -> None:
    console.print()
    console.print(f"[bold]{name}[/bold] · {tr['model']} · {tr['outcome']}")
    if tr["assessment"]:
        console.print(f"  Lage: {tr['assessment']}")
    for o in tr["orders"]:
        src = escape("[" + ", ".join(o.get("sources", [])) + "]")
        console.print(f"  {o['side']} {o['asset']} {o['pct']} % · {o.get('reason', '')} · {src}")
    for rj in tr["rejections"]:
        console.print(f"  [yellow]abgelehnt ({rj['attempt']}):[/yellow] " + " | ".join(rj["violations"]))
    if tr.get("model_error"):
        console.print(f"  [red]Fehler:[/red] {tr['model_error']}")
    for c in tr["calls"]:
        console.print(
            f"  Aufruf {c['attempt']}: {c.get('input_tokens')} -> {c.get('output_tokens')} Tokens, "
            f"{c.get('latency_s') or 0:.1f} s" + (f", {c['cost_usd']:.4f} $" if c.get("cost_usd") else "")
        )
    if tr["note"]:
        console.print(f"  Notiz: {tr['note']}")


@app.command()
def decide(
    path: Path,
    agent: str = typer.Option("gemma", help="Agent aus configs/agent.toml"),
    start_eur: float = typer.Option(100.0, help="Startkapital des Probedepots"),
) -> None:
    """Probeentscheidung auf einem gespeicherten Snapshot mit leerem Depot. Nichts wird ausgeführt."""
    from trading_agent.depot import Depot, load_rules

    snap = load_snapshot(path)
    strategy = _agents(agent)[agent]
    depot = Depot(f"probe-{agent}", start_eur, load_rules())
    strategy(snap, depot)
    _print_trace(agent, strategy.trace)


@app.command()
def backtest(
    start: str = typer.Option("2026-07-01", help="Erster Handelstag (nach den Wissensstichtagen)"),
    end: str = typer.Option("", help="Letzter Handelstag (Standard: gestern)"),
    agents: str = typer.Option("", help="LLM-Agenten aus configs/agent.toml, z. B. gemma,deepseek"),
    sizes: str = typer.Option("100", help="Startkapital, mehrere mit Komma, z. B. 100,10000"),
    name: str = typer.Option("default", help="Name des Laufs (eigene Datenbank unter data/backtest/)"),
    root: Path = typer.Option(Path("data") / "backtest", help="Ordner für Snapshots, News-Cache und Depots"),
    no_news: bool = typer.Option(False, help="Ohne News"),
    watchlist: Path = typer.Option(DEFAULT_WATCHLIST, help="Watchlist (TOML)"),
) -> None:
    """Rückwirkend handeln: für jeden Handelstag ein Snapshot vor Börsenstart, Ausführung zur Eröffnung.

    Bricht der Lauf ab, macht derselbe Aufruf dort weiter (fertige Tage werden übersprungen).
    """
    from datetime import date, timedelta

    from trading_agent.backtest import (
        BacktestError,
        HistoricalNewsProvider,
        build_day,
        download_bars,
        trading_days,
    )
    from trading_agent.baselines import BASELINES
    from trading_agent.depot import load_rules
    from trading_agent.runner import run_tick
    from trading_agent.snapshot import save_fill_quotes
    from trading_agent.store import DepotStore

    logging.basicConfig(level=logging.WARNING)
    first = date.fromisoformat(start)
    last = date.fromisoformat(end) if end else date.today() - timedelta(days=1)
    wl = load_watchlist(watchlist)
    rules = load_rules()
    console.print(f"Lade Tageskerzen {first} bis {last} …")
    bars = download_bars(wl, first, last)
    days = trading_days(bars, first, last)
    news = None if no_news else HistoricalNewsProvider(root / "news-cache")
    snaps = root / "snapshots"
    store = DepotStore(root / f"{name}.sqlite")
    strategies = dict(BASELINES) | _agents(agents, store, root / f"{name}.records.jsonl")
    amounts = [float(x) for x in sizes.split(",") if x.strip()]
    console.print(f"{len(days)} Handelstage, Depots: {', '.join(strategies)} · {sizes} €")
    for i, day in enumerate(days, 1):
        try:
            snap, fill = build_day(wl, bars, day, news)
        except BacktestError as e:
            console.print(f"[red]Abbruch:[/red] {e}. Derselbe Aufruf macht später hier weiter.")
            store.close()
            raise typer.Exit(3) from e
        save_snapshot(snap, snaps)
        save_fill_quotes(fill, snap, snaps)
        line = []
        for amount in amounts:
            for dname, r in run_tick(snap, fill, store, strategies, amount, rules).items():
                if "agent" in r:
                    tr = r["agent"]
                    acts = ", ".join(f"{o['side']} {o['asset']} {o['pct']}%" for o in tr["orders"])
                    line.append(f"{dname}: {tr['outcome']}" + (f" ({acts})" if acts else ""))
        console.print(f"[dim]{i:3d}/{len(days)}[/dim] {day} · {len(snap['news'])} News · " + " · ".join(line))
    _backtest_table(store, f"Backtest {first} bis {last} · {name}")
    store.close()


def _backtest_table(store, title: str) -> None:
    t = Table(title=title)
    for col in ("Depot", "Wert €", "Rendite", "max. Verlust", "Trades", "Gebühren €", "Steuern €"):
        t.add_column(col, justify="left" if col == "Depot" else "right")
    rows = []
    for name in store.names():
        depot = store.load(name)
        marks = [v for _, v in store.marks(name)]
        last = marks[-1] if marks else depot.start_eur
        peak, dd = depot.start_eur, 0.0
        for v in marks:
            peak = max(peak, v)
            dd = min(dd, v / peak - 1)
        trades = len(depot.fills)
        rows.append((last / depot.start_eur - 1, name, last, dd, trades, depot.fees_paid, depot.taxes_paid))
    for ret, name, last, dd, trades, fees, taxes in sorted(rows, reverse=True):
        t.add_row(name, f"{last:.2f}", _pct(ret), _pct(dd), str(trades), f"{fees:.2f}", f"{taxes:.2f}")
    console.print(t)


@app.command()
def depots(db: Path = typer.Option(Path("data") / "depots.sqlite", help="Depot-Datenbank")) -> None:
    """Stand aller Depots (aus dem Ereignisprotokoll neu berechnet)."""
    from trading_agent.store import DepotStore

    store = DepotStore(db)
    for name in store.names():
        depot = store.load(name)
        marks = store.marks(name)
        last = marks[-1][1] if marks else depot.start_eur
        pos = ", ".join(f"{a} {p['shares']:.4f}" for a, p in depot.positions().items()) or "nur Cash"
        console.print(
            f"[bold]{name}[/bold]  {last:.2f} € ({_pct(last / depot.start_eur - 1)}) · "
            f"Cash {depot.cash:.2f} € · "
            f"Gebühren {depot.fees_paid:.2f} € · Steuern {depot.taxes_paid:.2f} € · {pos}"
        )
    store.close()


if __name__ == "__main__":
    app()
