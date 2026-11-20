"""Der Agent als Strategie für `run_tick`: (Snapshot, Depot) -> Entscheidungen.

Jeder Modellaufruf läuft über den Recorder von harness-it-all. Damit ist jeder Aufruf mit Eingabe,
Profil-Hash, Modellversion, Tokens, Latenz und Kosten gespeichert und lässt sich später mit einem anderen
Modell wiederholen.
Was der Agent im Takt getan hat, steht danach in `trace` und wird mit der Entscheidung des Depots gespeichert.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from harness_it_all.models.base import ChatModel, ModelError
from harness_it_all.profiles import Profile
from harness_it_all.recorder import Recorder

from trading_agent.agent.config import AgentConfig
from trading_agent.agent.graph import build_graph
from trading_agent.agent.prompts import PROMPT_VERSION
from trading_agent.depot import Depot

Decision = tuple[str, str, float, str]
CALL_FIELDS = ("model_served", "finish_reason", "input_tokens", "output_tokens", "latency_s", "cost_usd")


def iso_week(ts: str) -> str:
    y, w, _ = datetime.fromisoformat(ts).isocalendar()
    return f"{y}-W{w:02d}"


class LLMAgent:
    def __init__(
        self,
        model: ChatModel,
        profile: Profile,
        cfg: AgentConfig,
        recorder: Recorder | None = None,
        journal: Callable[[str], list[str]] | None = None,
    ):
        self.model, self.profile, self.cfg = model, profile, cfg
        self.recorder = recorder
        self.journal = journal or (lambda depot_name: [])
        self.trace: dict | None = None

    def _caller(self, snapshot: dict, depot: Depot):
        def call(messages: list[dict], schema: dict, attempt: int) -> tuple[str | None, dict]:
            case = f"{depot.name}:{snapshot['hash']}:{attempt}"
            tags = {
                "snapshot": snapshot["hash"],
                "depot": depot.name,
                "week": iso_week(snapshot["taken_at"]),
                "attempt": attempt,
                "agent_config": self.cfg.version,
            }
            if self.recorder is not None:
                rec = self.recorder.run(
                    self.model, self.profile, case, messages, schema, PROMPT_VERSION, tags
                )
                if rec.error is not None or rec.completion is None:
                    return None, {"record": rec.id, "error": rec.error}
                c = rec.completion
                info = {"record": rec.id}
            else:
                try:
                    c = self.model.complete(messages, self.profile, schema).to_dict()
                except ModelError as e:
                    return None, {"error": str(e)}
                info = {}
            info |= {k: c.get(k) for k in CALL_FIELDS}
            info["fallback_used"] = c.get("fallback_used", False)
            if c.get("fallback_used"):
                # Ein anderes Modell hat geantwortet: nicht als Entscheidung dieses Modells werten
                return None, info | {"error": f"Fallback auf {c.get('model_served')}"}
            return c["text"], info

        return call

    def __call__(self, snapshot: dict, depot: Depot) -> list[Decision]:
        graph = build_graph(self.cfg, self._caller(snapshot, depot))
        state = graph.invoke({"snapshot": snapshot, "depot": depot, "journal": self.journal(depot.name)})
        decision = state.get("decision") or {}
        self.trace = {
            "model": self.model.model,
            "backend": self.model.backend,
            "profile": self.profile.name,
            "profile_hash": self.profile.hash,
            "prompt_version": PROMPT_VERSION,
            "agent_config": self.cfg.version,
            "outcome": state["outcome"],
            "screen": state.get("screen_reasons", {}),
            "focus": state.get("focus", []),
            "news": state.get("news_context"),
            "assessment": decision.get("assessment", ""),
            "note": decision.get("note", ""),
            "orders": state.get("orders", []),
            "rejections": state.get("rejections", []),
            "schema_errors": state.get("schema_errors", 0),
            "model_error": state.get("model_error"),
            "calls": state.get("calls", []),
        }
        return [(o["asset"], o["side"], o["pct"] / 100, o.get("reason", "")) for o in state.get("orders", [])]
