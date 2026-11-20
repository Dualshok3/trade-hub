"""Der Entscheidungs-Graph (LangGraph).

    screen ──(nichts los)──────────────────────────────► finish (halten, ohne LLM)
       │
       ▼
    gather ─► decide ─► risk ──(ok)──────────────────► finish
                 ▲        │
                 └─(abgelehnt, Versuche übrig)┘
                          └─(zu oft abgelehnt)──────► finish (erzwungenes Halten)

Nur `decide` ruft das Modell. Alle anderen Knoten sind deterministisches Python und lesen nur aus dem
Snapshot.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from trading_agent.agent import prompts
from trading_agent.agent.config import AgentConfig
from trading_agent.agent.risk import check
from trading_agent.agent.schema import ParseError, decision_schema, parse_decision
from trading_agent.depot import Depot
from trading_agent.tools import SnapshotTools

# (messages, schema, attempt) -> (Antworttext, Infos zum Aufruf)
CallModel = Callable[[list[dict], dict, int], tuple[str | None, dict]]


class State(TypedDict, total=False):
    snapshot: dict
    depot: Depot
    journal: list[str]
    focus: list[str]
    screen_reasons: dict[str, list[str]]
    news_context: dict
    messages: list[dict]
    decision: dict | None
    violations: list[str]
    rejections: Annotated[list[dict], operator.add]
    calls: Annotated[list[dict], operator.add]
    schema_errors: int
    orders: list[dict]
    model_error: str | None
    outcome: str  # quiet | hold | trade | forced_hold | error


def prices_of(snapshot: dict) -> dict[str, float]:
    return {a: q["price_eur"] for a, q in snapshot["quotes"].items() if q.get("price_eur") is not None}


def screen(snapshot: dict, depot: Depot, tools: SnapshotTools, cfg: AgentConfig) -> dict[str, list[str]]:
    """Watchlist-Check ohne LLM: welche Werte sich bewegt haben oder relevante News haben."""
    sc = cfg.screen
    prices = prices_of(snapshot)
    reasons: dict[str, list[str]] = {}
    for row in tools.overview():
        a, why = row["asset"], []
        if row["day_change"] is not None and abs(row["day_change"]) >= sc.day_move:
            why.append(f"Tag {row['day_change'] * 100:+.1f} %")
        if row["change_5d"] is not None and abs(row["change_5d"]) >= sc.move_5d:
            why.append(f"5 Tage {row['change_5d'] * 100:+.1f} %")
        top = tools.get_top_news(a, sc.news_hours, 1)
        if top and top[0]["score"] >= sc.min_news_score:
            why.append("neue relevante News")
        if why:
            reasons[a] = why
    value = depot.value(prices)
    if value > 0 and depot.cash / value > sc.idle_cash_share:
        reasons.setdefault("_depot", []).append(f"{depot.cash / value * 100:.0f} % Cash")
    return reasons


def build_graph(cfg: AgentConfig, call_model: CallModel):
    def node_screen(state: State) -> dict:
        snap, depot = state["snapshot"], state["depot"]
        tools = SnapshotTools(snap)
        reasons = screen(snap, depot, tools, cfg)
        held = [a for a in depot.positions() if a in tools.assets]
        movers = sorted(
            (a for a in reasons if a != "_depot"),
            key=lambda a: -len(reasons[a]),
        )
        focus = list(dict.fromkeys(held + movers))[: cfg.screen.max_focus]
        if not reasons:
            focus = []
        return {"focus": focus, "screen_reasons": reasons}

    def after_screen(state: State) -> str:
        return "gather" if state["screen_reasons"] else "finish"

    def node_gather(state: State) -> dict:
        snap, depot = state["snapshot"], state["depot"]
        tools = SnapshotTools(snap)
        ctx = tools.news_context(state["focus"], 24, cfg.context.news_per_asset)
        market = tools.get_market_news(24, cfg.context.market_news)
        ctx["market"] = market
        ctx["shown_ids"]["_market"] = [n["id"] for n in market]
        prices = prices_of(snap)
        user = prompts.user_prompt(
            snap,
            prompts.portfolio_block(depot, prices),
            prompts.overview_block(tools.overview()),
            [prompts.focus_block(tools, a, ctx["per_asset"][a]) for a in state["focus"]],
            market,
            state.get("journal", [])[-cfg.context.journal_notes :],
        )
        example = next((n["id"] for n in market), None) or next(iter(tools.news_ids()), "N1a2b3c4d5e")
        messages = [
            {"role": "system", "content": prompts.system_prompt(depot, cfg, example)},
            {"role": "user", "content": user},
        ]
        return {
            "messages": messages,
            "news_context": {k: ctx[k] for k in ("news_rank_version", "snapshot", "shown_ids")},
        }

    def node_decide(state: State) -> dict:
        snap = state["snapshot"]
        schema = decision_schema([a["id"] for a in snap["watchlist"]])
        attempt = len(state.get("rejections", [])) + 1
        text, info = call_model(state["messages"], schema, attempt)
        info = {"attempt": attempt, **info}
        if text is None:
            err = info.get("error") or "Modell nicht erreichbar"
            return {"decision": None, "violations": [err], "model_error": err, "calls": [info]}
        try:
            decision = parse_decision(text)
        except ParseError as e:
            return {
                "decision": None,
                "violations": [str(e)],
                "schema_errors": state.get("schema_errors", 0) + 1,
                "calls": [info],
                "messages": [*state["messages"], {"role": "assistant", "content": text}],
            }
        return {
            "decision": decision,
            "violations": [],
            "calls": [info],
            "messages": [*state["messages"], {"role": "assistant", "content": text}],
        }

    def node_risk(state: State) -> dict:
        if state.get("model_error"):
            return {}  # kein Fehler des Modells, sondern der Verbindung: nicht als Ablehnung zählen
        if state.get("decision") is None:
            violations = state["violations"]
        else:
            snap, depot = state["snapshot"], state["depot"]
            tools = SnapshotTools(snap)
            violations = check(
                state["decision"]["orders"],
                depot,
                prices_of(snap),
                set(tools.assets),
                tools.news_ids(),
                cfg.risk,
            )
        if not violations:
            return {"violations": []}
        rejected = {"attempt": len(state.get("rejections", [])) + 1, "violations": violations}
        left = cfg.risk.max_rejections - rejected["attempt"]
        update: dict[str, Any] = {"violations": violations, "rejections": [rejected]}
        if left > 0:
            update["messages"] = [
                *state["messages"],
                {"role": "user", "content": prompts.rejection_prompt(violations, left)},
            ]
        return update

    def after_risk(state: State) -> str:
        if not state["violations"] or state.get("model_error"):
            return "finish"
        if len(state["rejections"]) >= cfg.risk.max_rejections:
            return "finish"
        return "decide"

    def node_finish(state: State) -> dict:
        if not state.get("screen_reasons"):
            return {"outcome": "quiet", "orders": []}
        if state.get("model_error"):
            return {"outcome": "error", "orders": []}
        if state.get("violations"):
            return {"outcome": "forced_hold", "orders": []}
        orders = state["decision"]["orders"]
        return {"outcome": "trade" if orders else "hold", "orders": orders}

    g = StateGraph(State)
    g.add_node("screen", node_screen)
    g.add_node("gather", node_gather)
    g.add_node("decide", node_decide)
    g.add_node("risk", node_risk)
    g.add_node("finish", node_finish)
    g.add_edge(START, "screen")
    g.add_conditional_edges("screen", after_screen, ["gather", "finish"])
    g.add_edge("gather", "decide")
    g.add_edge("decide", "risk")
    g.add_conditional_edges("risk", after_risk, ["decide", "finish"])
    g.add_edge("finish", END)
    return g.compile()
