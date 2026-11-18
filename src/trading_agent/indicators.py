"""Technische Kennzahlen aus Tagesschlusskursen (reines Python, gut testbar)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence


def sma(closes: Sequence[float], n: int) -> float | None:
    if n <= 0 or len(closes) < n:
        return None
    return sum(closes[-n:]) / n


def rsi(closes: Sequence[float], n: int = 14) -> float | None:
    """RSI nach Wilder. None bei zu wenig Daten."""
    if len(closes) < n + 1:
        return None
    deltas = [b - a for a, b in itertools.pairwise(closes)]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:n]) / n
    avg_loss = sum(losses[:n]) / n
    for g, loss in zip(gains[n:], losses[n:], strict=True):
        avg_gain = (avg_gain * (n - 1) + g) / n
        avg_loss = (avg_loss * (n - 1) + loss) / n
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def volatility(closes: Sequence[float], n: int = 20) -> float | None:
    """Annualisierte Volatilität der täglichen Log-Renditen über `n` Tage (252 Handelstage)."""
    if len(closes) < n + 1:
        return None
    window = closes[-(n + 1) :]
    rets = [math.log(b / a) for a, b in itertools.pairwise(window) if a > 0 and b > 0]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252)


def change(closes: Sequence[float], days: int) -> float | None:
    """Relative Änderung über `days` Handelstage (0.05 = +5 %)."""
    if len(closes) < days + 1 or closes[-days - 1] == 0:
        return None
    return closes[-1] / closes[-days - 1] - 1


def from_high(closes: Sequence[float], highs: Sequence[float], days: int = 252) -> float | None:
    """Abstand des letzten Schlusskurses zum Hoch der letzten `days` Tage (-0.1 = 10 % darunter)."""
    if not closes or not highs:
        return None
    top = max(highs[-days:])
    return closes[-1] / top - 1 if top else None
