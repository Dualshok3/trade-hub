"""Das Antwortformat des Modells.

Bewusst nur der kleinste gemeinsame Nenner beider Backends: geschlossene Objekte, Arrays, String-Enums,
Integer ohne Grenzen. Grenzen (10–40 %, höchstens 3 Orders) prüft der Risk-Check im Code, nicht das Schema.
So bekommt das Modell eine verständliche Ablehnung statt eines stumm abgeschnittenen Werts.
"""

from __future__ import annotations

import json
import re

INDICATORS = (
    "day_change",
    "change_5d",
    "change_20d",
    "rsi14",
    "sma20",
    "sma50",
    "sma200",
    "volatility20",
    "from_52w_high",
)


def decision_schema(asset_ids: list[str]) -> dict:
    order = {
        "type": "object",
        "additionalProperties": False,
        "required": ["asset", "side", "pct", "reason", "sources"],
        "properties": {
            "asset": {"type": "string", "enum": list(asset_ids)},
            "side": {"type": "string", "enum": ["buy", "sell"]},
            "pct": {"type": "integer"},
            "reason": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["assessment", "orders", "note"],
        "properties": {
            "assessment": {"type": "string"},
            "orders": {"type": "array", "items": order},
            "note": {"type": "string"},
        },
    }


class ParseError(ValueError):
    pass


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_decision(text: str) -> dict:
    """Antwort des Modells lesen und die Form prüfen. Wirft `ParseError` mit einer Begründung fürs Modell."""
    t = _FENCE.sub("", text.strip())
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < start:
        raise ParseError("Die Antwort enthält kein JSON-Objekt.")
    try:
        d = json.loads(t[start : end + 1])
    except json.JSONDecodeError as e:
        raise ParseError(f"Die Antwort ist kein gültiges JSON ({e.msg}).") from None
    if not isinstance(d, dict) or not isinstance(d.get("orders"), list):
        raise ParseError("Das Feld 'orders' fehlt oder ist keine Liste.")
    for i, o in enumerate(d["orders"]):
        if not isinstance(o, dict):
            raise ParseError(f"Order {i + 1} ist kein Objekt.")
        missing = [k for k in ("asset", "side", "pct") if k not in o]
        if missing:
            raise ParseError(f"Order {i + 1}: es fehlen {', '.join(missing)}.")
        if not isinstance(o["pct"], int) or isinstance(o["pct"], bool):
            raise ParseError(f"Order {i + 1}: 'pct' muss eine ganze Zahl sein.")
        o.setdefault("reason", "")
        o.setdefault("sources", [])
    for key in ("assessment", "note"):
        # c0014 setzt vor Freitext manchmal ": " (Rest des Thinking-Kanals); das gehört nicht zur Antwort
        d[key] = str(d.get(key, "")).lstrip(": ").strip()
    return d
