"""Snapshots: der eingefrorene Marktzustand eines Takts.

Beide Agenten entscheiden auf exakt demselben Snapshot, und jeder Snapshot lässt sich später wieder abspielen.
Ein Snapshot ist JSON (gzip). Sein Hash (SHA-256 über die kanonische Form) steht im Dateinamen
und in der Datei.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from trading_agent.config import Watchlist
from trading_agent.feeds.prices import PriceProvider
from trading_agent.market import open_exchanges

SCHEMA = 1
HISTORY_DAYS = 400  # reicht für SMA 200 und das 52-Wochen-Hoch
DEFAULT_DIR = Path("data") / "snapshots"


class SnapshotError(RuntimeError):
    pass


def canonical(body: dict) -> bytes:
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def digest(body: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in body.items() if k != "hash"})).hexdigest()


def build_snapshot(
    watchlist: Watchlist,
    prices: PriceProvider,
    news_provider=None,
    now: datetime | None = None,
    news_hours: float = 24,
    history_days: int = HISTORY_DAYS,
) -> dict:
    """Holt Kurse, Historie, Wechselkurs und News und friert sie zum Zeitpunkt `now` ein."""
    now = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    errors: list[str] = []
    symbols = [a.symbol for a in watchlist.assets]

    raw_quotes = prices.quotes([*symbols, watchlist.fx_symbol])
    fx = raw_quotes.get(watchlist.fx_symbol)
    eurusd = fx.price if fx else None
    if eurusd is None and any(a.currency == "USD" for a in watchlist.assets):
        errors.append(f"kein Wechselkurs {watchlist.fx_symbol}: USD-Werte ohne Euro-Kurs")

    raw_history = prices.history(symbols, history_days)
    today = now.date().isoformat()

    quotes, history = {}, {}
    for asset in watchlist.assets:
        q = raw_quotes.get(asset.symbol)
        if q is None:
            errors.append(f"{asset.id}: kein Kurs")
        elif q.currency != asset.currency:
            errors.append(f"{asset.id}: Kurs in {q.currency}, erwartet {asset.currency}")
        elif q.as_of and datetime.fromisoformat(q.as_of) > now:
            errors.append(f"{asset.id}: Kurszeit {q.as_of} liegt nach dem Snapshot")  # Look-ahead-Schutz
        else:
            rate = 1.0 if asset.currency == "EUR" else (1 / eurusd if eurusd else None)
            delay = None
            if q.as_of:
                delay = round((now - datetime.fromisoformat(q.as_of)).total_seconds() / 60, 1)
            else:
                errors.append(f"{asset.id}: Kurszeit unbekannt")
            quotes[asset.id] = {
                **q.to_dict(),
                "price_eur": round(q.price * rate, 6) if rate else None,
                "age_minutes": delay,  # Alter des Kurses zum Snapshot-Zeitpunkt (verzögert oder Börse zu)
            }
        bars = [b.to_dict() for b in raw_history.get(asset.symbol, []) if b.date <= today]
        if not bars:
            errors.append(f"{asset.id}: keine Historie")
        history[asset.id] = bars

    news, news_source = [], None
    if news_provider is not None:
        items, news_errors = news_provider.fetch(watchlist, now, news_hours)
        news = [n.to_dict() for n in items]
        errors.extend(news_errors)
        news_source = news_provider.name

    body = {
        "schema": SCHEMA,
        "taken_at": now.isoformat(),
        "open_exchanges": open_exchanges(now),
        "watchlist": [
            asdict(a) | {"aliases": list(a.aliases), "exclude": list(a.exclude)} for a in watchlist.assets
        ],
        "fx": {"EURUSD": eurusd, "symbol": watchlist.fx_symbol},
        "quotes": quotes,
        "history": history,
        "news": news,
        "sources": {"prices": prices.name, "news": news_source},
        "errors": errors,
    }
    body["hash"] = digest(body)
    return body


def snapshot_path(snapshot: dict, root: Path = DEFAULT_DIR) -> Path:
    t = datetime.fromisoformat(snapshot["taken_at"])
    return root / t.strftime("%Y-%m-%d") / f"{t.strftime('%H%M%SZ')}-{snapshot['hash'][:12]}.json.gz"


def save_snapshot(snapshot: dict, root: Path = DEFAULT_DIR) -> Path:
    if snapshot.get("hash") != digest(snapshot):
        raise SnapshotError("Hash passt nicht zum Inhalt, Snapshot wurde nach dem Bau verändert")
    path = snapshot_path(snapshot, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path  # gleicher Inhalt, gleicher Name: nichts zu tun
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as f:
        f.write(canonical(snapshot))
    tmp.replace(path)
    return path


def load_snapshot(path: Path | str) -> dict:
    with gzip.open(path, "rb") as f:
        snapshot = json.loads(f.read())
    if snapshot.get("schema") != SCHEMA:
        raise SnapshotError(f"{path}: unbekanntes Schema {snapshot.get('schema')}")
    if snapshot.get("hash") != digest(snapshot):
        raise SnapshotError(f"{path}: Hash stimmt nicht, Datei ist beschädigt oder verändert")
    return snapshot


def list_snapshots(root: Path = DEFAULT_DIR) -> list[Path]:
    return sorted(root.glob("*/*.json.gz"))
