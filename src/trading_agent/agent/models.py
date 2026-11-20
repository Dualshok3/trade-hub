"""Modelle aus `configs/agent.toml` bauen. Die Adapter selbst kommen aus harness-it-all."""

from __future__ import annotations

from harness_it_all.models.base import ChatModel
from harness_it_all.profiles import Profile, load_profiles

from trading_agent.agent.config import DEFAULT_PROFILES, AgentConfig


def build_model(name: str, cfg: AgentConfig) -> ChatModel:
    if name not in cfg.models:
        raise ValueError(f"Modell {name!r} fehlt in configs/agent.toml (vorhanden: {', '.join(cfg.models)})")
    m = cfg.models[name]
    if m["backend"] == "openai":
        from harness_it_all.models.openai_compat import OpenAICompatModel

        return OpenAICompatModel(m["model"], base_url=m.get("base_url", "http://127.0.0.1:8000/v1"))
    if m["backend"] == "deepseek":
        from harness_it_all.models.openai_compat import OpenAICompatModel

        return OpenAICompatModel(
            m["model"],
            base_url=m.get("base_url", "https://api.deepseek.com/v1"),
            profile_key="deepseek",
            api_key_env=m.get("api_key_env", "DEEPSEEK_API_KEY"),
            schema_mode="json_object",
        )
    if m["backend"] == "claude_code":
        from harness_it_all.models.claude_code import ClaudeCodeModel

        return ClaudeCodeModel(m["model"])
    if m["backend"] == "anthropic":
        from harness_it_all.models.anthropic_model import AnthropicModel

        return AnthropicModel(m["model"])
    raise ValueError(f"unbekanntes Backend {m['backend']!r}")


def load_profile(cfg: AgentConfig, path=DEFAULT_PROFILES) -> Profile:
    return load_profiles(path)[cfg.profile]
