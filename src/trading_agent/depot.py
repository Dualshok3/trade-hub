"""Das Musterdepot: Cash, Kauf-Lots, Gebühren und Steuern (Stufe 1).

Ein Depot ist eine Folge von Ereignissen (Eröffnung, Ausführungen). Der Zustand entsteht, indem man die
Ereignisse der Reihe nach anwendet. So lässt sich jedes Depot aus seiner Datenbank exakt wiederherstellen,
und ein Replay erzeugt dieselben Ereignisse wie der Live-Lauf.

Steuern Stufe 1: 26,375 % auf realisierte Gewinne, FIFO, Gebühren mindern den Gewinn. Verluste werden nur
festgehalten (für Stufe 2: Verlusttöpfe und Erstattung), sie erzeugen noch keine Erstattung.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_RULES = Path(__file__).resolve().parents[2] / "configs" / "depot.toml"
EPS = 1e-9


class DepotError(ValueError):
    pass


@dataclass(frozen=True)
class DepotRules:
    fee_eur: float = 1.0
    half_spread: float = 0.001
    min_order_pct: float = 0.10
    pending_max_ticks: int = 3
    tax_rate: float = 0.26375
    allowance_eur: float = 0.0


def load_rules(path: Path | str = DEFAULT_RULES) -> DepotRules:
    d = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return DepotRules(
        fee_eur=d["fees"]["per_order_eur"],
        half_spread=d["fees"]["half_spread"],
        min_order_pct=d["orders"]["min_order_pct"],
        pending_max_ticks=d["orders"]["pending_max_ticks"],
        tax_rate=d["tax"]["rate"],
        allowance_eur=d["tax"]["allowance_eur"],
    )


def cents(x: float) -> float:
    """Kaufmännisch auf Cent runden (Steuern und Gebühren werden in Cent abgerechnet)."""
    return float(int(abs(x) * 100 + 0.5 + EPS)) / 100 * (1 if x >= 0 else -1)


@dataclass
class Lot:
    asset: str
    shares: float
    cost_eur: float  # Einstand inkl. anteiliger Kaufgebühr und Spread
    bought_at: str


@dataclass(frozen=True)
class Fill:
    """Eine ausgeführte Order. Enthält alle Ergebnisse, damit das Anwenden deterministisch ist."""

    depot: str
    order_id: str
    asset: str
    side: str  # buy | sell
    shares: float
    price_eur: float  # Kurs vor Spread
    exec_price_eur: float  # Kurs inkl. Spread
    gross_eur: float  # shares × exec_price
    fee_eur: float
    tax_eur: float
    realized_eur: float  # Gewinn/Verlust nach Gebühren (nur Verkauf)
    at: str  # Kurszeit der Ausführung
    decided_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Depot:
    name: str
    start_eur: float
    rules: DepotRules = field(default_factory=DepotRules)
    cash: float = 0.0
    lots: list[Lot] = field(default_factory=list)
    fees_paid: float = 0.0
    taxes_paid: float = 0.0
    realized_gains: float = 0.0
    realized_losses: float = 0.0  # für Stufe 2 (Verlusttöpfe)
    allowance_used: dict[int, float] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)

    def __post_init__(self):
        if not self.fills and self.cash == 0.0:
            self.cash = self.start_eur

    # ------------------------------------------------------------------ Bestand

    def shares(self, asset: str) -> float:
        return sum(lot.shares for lot in self.lots if lot.asset == asset)

    def positions(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for lot in self.lots:
            p = out.setdefault(lot.asset, {"shares": 0.0, "cost_eur": 0.0})
            p["shares"] += lot.shares
            p["cost_eur"] += lot.cost_eur
        return out

    def value(self, prices_eur: dict[str, float]) -> float:
        """Depotwert = Cash + Positionen zu den gegebenen Kursen (ohne Spread, ohne Steuer)."""
        total = self.cash
        for asset, p in self.positions().items():
            if asset not in prices_eur or prices_eur[asset] is None:
                raise DepotError(f"kein Kurs für {asset} zur Bewertung")
            total += p["shares"] * prices_eur[asset]
        return total

    def weights(self, prices_eur: dict[str, float]) -> dict[str, float]:
        v = self.value(prices_eur)
        return {a: p["shares"] * prices_eur[a] / v for a, p in self.positions().items()} if v > 0 else {}

    # --------------------------------------------------------------- Rechnen

    def quote_buy(self, amount_eur: float, price_eur: float) -> tuple[float, float, float]:
        """(Stück, Ausführungskurs, Brutto) für einen Kauf über `amount_eur` (ohne Gebühr)."""
        exec_price = price_eur * (1 + self.rules.half_spread)
        shares = amount_eur / exec_price
        return shares, exec_price, shares * exec_price

    def tax_on(self, gain: float, year: int) -> tuple[float, float]:
        """(Steuer, verbrauchter Freibetrag) für einen realisierten Gewinn."""
        if gain <= 0:
            return 0.0, 0.0
        left = max(self.rules.allowance_eur - self.allowance_used.get(year, 0.0), 0.0)
        use = min(left, gain)
        return cents((gain - use) * self.rules.tax_rate), use

    def sell_preview(self, asset: str, shares: float, price_eur: float, year: int) -> dict:
        """Was ein Verkauf ergäbe: Erlös, Gewinn nach FIFO, Steuer. Ändert nichts."""
        held = self.shares(asset)
        if shares > held + EPS:
            raise DepotError(f"{asset}: nur {held:.6f} Stück im Depot")
        exec_price = price_eur * (1 - self.rules.half_spread)
        gross = shares * exec_price
        cost, rest = 0.0, shares
        for lot in self.lots:
            if lot.asset != asset or rest <= EPS:
                continue
            take = min(lot.shares, rest)
            cost += lot.cost_eur * take / lot.shares
            rest -= take
        realized = gross - self.rules.fee_eur - cost
        tax, allowance = self.tax_on(realized, year)
        return {
            "exec_price": exec_price,
            "gross": gross,
            "cost": cost,
            "realized": realized,
            "tax": tax,
            "allowance": allowance,
            "net": gross - self.rules.fee_eur - tax,
        }

    # -------------------------------------------------------------- Anwenden

    def apply(self, fill: Fill) -> None:
        """Eine Ausführung auf das Depot anwenden (auch beim Wiederherstellen aus der Datenbank)."""
        year = int(fill.at[:4])
        if fill.side == "buy":
            total = fill.gross_eur + fill.fee_eur
            if total > self.cash + 1e-6:
                raise DepotError(f"{self.name}: Kauf über {total:.2f} € bei {self.cash:.2f} € Cash")
            self.cash -= total
            self.lots.append(Lot(fill.asset, fill.shares, total, fill.at))
        elif fill.side == "sell":
            rest = fill.shares
            for lot in self.lots:
                if lot.asset != fill.asset or rest <= EPS:
                    continue
                take = min(lot.shares, rest)
                part = lot.cost_eur * take / lot.shares
                lot.shares -= take
                lot.cost_eur -= part
                rest -= take
            if rest > 1e-6:
                raise DepotError(f"{self.name}: Verkauf von mehr {fill.asset} als vorhanden")
            self.lots = [lot for lot in self.lots if lot.shares > EPS]
            self.cash += fill.gross_eur - fill.fee_eur - fill.tax_eur
            if fill.realized_eur >= 0:
                self.realized_gains += fill.realized_eur
                _, used = self.tax_on(fill.realized_eur, year)
                self.allowance_used[year] = self.allowance_used.get(year, 0.0) + used
            else:
                self.realized_losses += -fill.realized_eur
            self.taxes_paid += fill.tax_eur
        else:
            raise DepotError(f"unbekannte Seite {fill.side!r}")
        self.fees_paid += fill.fee_eur
        self.fills.append(fill)

    def summary(self, prices_eur: dict[str, float]) -> dict:
        v = self.value(prices_eur)
        return {
            "depot": self.name,
            "value_eur": round(v, 4),
            "cash_eur": round(self.cash, 4),
            "return": round(v / self.start_eur - 1, 6),
            "fees_eur": round(self.fees_paid, 2),
            "taxes_eur": round(self.taxes_paid, 2),
            "realized_gains_eur": round(self.realized_gains, 4),
            "realized_losses_eur": round(self.realized_losses, 4),
            "positions": {
                a: {
                    "shares": round(p["shares"], 6),
                    "value_eur": round(p["shares"] * prices_eur[a], 4),
                    "cost_eur": round(p["cost_eur"], 4),
                }
                for a, p in self.positions().items()
            },
            "orders": len(self.fills),
        }
