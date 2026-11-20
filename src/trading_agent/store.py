"""Depots in SQLite: ein Ereignisprotokoll pro Depot, daraus wird der Zustand jederzeit neu aufgebaut.

Tabellen:
- `events`: Eröffnung (Startkapital und Regeln) und jede Ausführung, nur anhängen, nie ändern
- `pending`: Orders, die auf einen neueren Kurs warten
- `marks`: Depotwert je Takt (für Equity-Kurven)
- `decisions`: Ergebnis jedes Takts je Depot (Orders, Ablehnungen, erzwungenes Halten …)
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from trading_agent.depot import Depot, DepotRules, Fill
from trading_agent.execution import Order

DEFAULT_DB = Path("data") / "depots.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    depot TEXT NOT NULL,
    kind TEXT NOT NULL,
    at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_depot ON events (depot, id);
CREATE TABLE IF NOT EXISTS pending (
    depot TEXT NOT NULL,
    order_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (depot, order_id)
);
CREATE TABLE IF NOT EXISTS marks (
    depot TEXT NOT NULL,
    at TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    value_eur REAL NOT NULL,
    cash_eur REAL NOT NULL,
    PRIMARY KEY (depot, at)
);
CREATE TABLE IF NOT EXISTS decisions (
    depot TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    at TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (depot, snapshot)
);
"""


class StoreError(RuntimeError):
    pass


class DepotStore:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    # ------------------------------------------------------------------ Depots

    def names(self) -> list[str]:
        rows = self.db.execute("SELECT DISTINCT depot FROM events WHERE kind = 'open' ORDER BY depot")
        return [r[0] for r in rows]

    def open(self, name: str, start_eur: float, rules: DepotRules, at: str) -> Depot:
        if name in self.names():
            raise StoreError(f"Depot {name!r} gibt es schon")
        payload = {"start_eur": start_eur, "rules": asdict(rules)}
        with self.db:
            self.db.execute(
                "INSERT INTO events (depot, kind, at, payload) VALUES (?, 'open', ?, ?)",
                (name, at, json.dumps(payload)),
            )
        return Depot(name, start_eur, rules)

    def load(self, name: str) -> Depot:
        rows = list(self.db.execute("SELECT kind, payload FROM events WHERE depot = ? ORDER BY id", (name,)))
        if not rows or rows[0][0] != "open":
            raise StoreError(f"Depot {name!r} nicht gefunden")
        opened = json.loads(rows[0][1])
        depot = Depot(name, opened["start_eur"], DepotRules(**opened["rules"]))
        for kind, payload in rows[1:]:
            if kind == "fill":
                depot.apply(Fill(**json.loads(payload)))
        return depot

    def record_fills(self, fills: list[Fill]) -> None:
        with self.db:
            self.db.executemany(
                "INSERT INTO events (depot, kind, at, payload) VALUES (?, 'fill', ?, ?)",
                [(f.depot, f.at, json.dumps(f.to_dict())) for f in fills],
            )

    # ------------------------------------------------------------- Wartende

    def pending(self, name: str) -> list[Order]:
        rows = self.db.execute("SELECT payload FROM pending WHERE depot = ? ORDER BY rowid", (name,))
        out = []
        for (payload,) in rows:
            d = json.loads(payload)
            d.pop("id", None)
            out.append(Order(**d))
        return out

    def set_pending(self, name: str, orders: list[Order]) -> None:
        with self.db:
            self.db.execute("DELETE FROM pending WHERE depot = ?", (name,))
            self.db.executemany(
                "INSERT INTO pending (depot, order_id, payload) VALUES (?, ?, ?)",
                [(name, o.id, json.dumps(o.to_dict())) for o in orders],
            )

    # ----------------------------------------------------------- Verlauf

    def mark(self, name: str, at: str, snapshot: str, value: float, cash: float) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO marks (depot, at, snapshot, value_eur, cash_eur) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, at, snapshot, value, cash),
            )

    def marks(self, name: str) -> list[tuple[str, float]]:
        rows = self.db.execute("SELECT at, value_eur FROM marks WHERE depot = ? ORDER BY at", (name,))
        return list(rows)

    def record_decision(self, name: str, snapshot: str, at: str, payload: dict) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO decisions (depot, snapshot, at, payload) VALUES (?, ?, ?, ?)",
                (name, snapshot, at, json.dumps(payload, ensure_ascii=False)),
            )

    def done(self, name: str, snapshot: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM decisions WHERE depot = ? AND snapshot = ?", (name, snapshot)
        ).fetchone()
        return row is not None

    def notes(self, name: str, n: int = 3) -> list[str]:
        """Die letzten `n` Notizen, die ein Agent in diesem Depot für sich selbst hinterlassen hat."""
        rows = self.db.execute(
            "SELECT at, payload FROM decisions WHERE depot = ? ORDER BY at DESC LIMIT ?", (name, 50)
        ).fetchall()
        out = []
        for at, payload in rows:
            note = (json.loads(payload).get("agent") or {}).get("note")
            if note:
                out.append(f"{at[:16]}: {note}")
            if len(out) == n:
                break
        return list(reversed(out))
