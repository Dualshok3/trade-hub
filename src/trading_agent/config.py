"""Watchlist und Markt-Konfiguration aus `configs/watchlist.toml`."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

EXCHANGES = {"XETRA", "NYSE", "NASDAQ"}
KINDS = {"stock", "etf"}

DEFAULT_WATCHLIST = Path(__file__).resolve().parents[2] / "configs" / "watchlist.toml"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Asset:
    id: str
    symbol: str
    name: str
    exchange: str
    currency: str
    kind: str
    aliases: tuple[str, ...]
    news: str
    exclude: tuple[str, ...] = ()  # Namen, die nicht dieser Wert sind (z. B. Siemens Energy)


@dataclass(frozen=True)
class Watchlist:
    assets: tuple[Asset, ...]
    market_news: tuple[str, ...] = ()
    fx_symbol: str = "EURUSD=X"
    _by_id: dict[str, Asset] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        self._by_id.update({a.id: a for a in self.assets})

    def get(self, asset_id: str) -> Asset:
        try:
            return self._by_id[asset_id]
        except KeyError:
            raise ConfigError(f"unbekannter Wert {asset_id!r}") from None

    @property
    def ids(self) -> list[str]:
        return [a.id for a in self.assets]


def load_watchlist(path: Path | str = DEFAULT_WATCHLIST) -> Watchlist:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    assets = []
    for raw in data.get("asset", []):
        try:
            asset = Asset(
                id=raw["id"],
                symbol=raw["symbol"],
                name=raw["name"],
                exchange=raw["exchange"],
                currency=raw["currency"],
                kind=raw["kind"],
                aliases=tuple(raw.get("aliases", ())),
                news=raw.get("news", raw["name"]),
                exclude=tuple(raw.get("exclude", ())),
            )
        except KeyError as e:
            raise ConfigError(f"Watchlist-Eintrag ohne Feld {e}: {raw}") from None
        if asset.exchange not in EXCHANGES:
            raise ConfigError(f"{asset.id}: unbekannte Börse {asset.exchange!r}")
        if asset.kind not in KINDS:
            raise ConfigError(f"{asset.id}: unbekannte Art {asset.kind!r}")
        if asset.currency not in {"EUR", "USD"}:
            raise ConfigError(f"{asset.id}: Währung {asset.currency!r} wird nicht unterstützt")
        assets.append(asset)
    ids = [a.id for a in assets]
    if len(ids) != len(set(ids)):
        raise ConfigError("doppelte IDs in der Watchlist")
    if not assets:
        raise ConfigError("Watchlist ist leer")
    market = data.get("market", {})
    return Watchlist(
        assets=tuple(assets),
        market_news=tuple(market.get("news", ())),
        fx_symbol=market.get("fx_symbol", "EURUSD=X"),
    )
