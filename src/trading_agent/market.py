"""Handelszeiten der Börsen (vereinfacht: Mo–Fr, ohne Feiertage)."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

SESSIONS = {
    "XETRA": (ZoneInfo("Europe/Berlin"), time(9, 0), time(17, 30)),
    "NYSE": (ZoneInfo("America/New_York"), time(9, 30), time(16, 0)),
    "NASDAQ": (ZoneInfo("America/New_York"), time(9, 30), time(16, 0)),
}


def is_open(exchange: str, at: datetime) -> bool:
    """Ob `exchange` zum Zeitpunkt `at` (mit Zeitzone) handelt. Feiertage kennt das MVP noch nicht."""
    if at.tzinfo is None:
        raise ValueError("Zeitpunkt braucht eine Zeitzone")
    tz, start, end = SESSIONS[exchange]
    local = at.astimezone(tz)
    return local.weekday() < 5 and start <= local.time() < end


def open_exchanges(at: datetime) -> list[str]:
    return [ex for ex in SESSIONS if is_open(ex, at)]
