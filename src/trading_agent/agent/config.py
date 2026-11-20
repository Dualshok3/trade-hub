"""Einstellungen des Agenten aus `configs/agent.toml`."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AGENT = ROOT / "configs" / "agent.toml"
DEFAULT_PROFILES = ROOT / "configs" / "profiles.toml"


@dataclass(frozen=True)
class ScreenConfig:
    day_move: float = 0.02
    move_5d: float = 0.05
    news_hours: float = 6
    min_news_score: float = 1.0
    idle_cash_share: float = 0.5
    max_focus: int = 6


@dataclass(frozen=True)
class ContextConfig:
    news_per_asset: int = 3
    market_news: int = 5
    journal_notes: int = 3


@dataclass(frozen=True)
class RiskConfig:
    max_weight: float = 0.40
    max_orders: int = 3
    max_rejections: int = 2


@dataclass(frozen=True)
class AgentConfig:
    profile: str = "minimal"
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    models: dict[str, dict] = field(default_factory=dict)
    version: str = "default"  # Hash der Konfigurationsdatei


def load_agent_config(path: Path | str = DEFAULT_AGENT) -> AgentConfig:
    raw = Path(path).read_bytes()
    d = tomllib.loads(raw.decode("utf-8"))
    return AgentConfig(
        profile=d.get("profile", "minimal"),
        screen=ScreenConfig(**d.get("screen", {})),
        context=ContextConfig(**d.get("context", {})),
        risk=RiskConfig(**d.get("risk", {})),
        models=d.get("models", {}),
        version=hashlib.sha256(raw).hexdigest()[:12],
    )
