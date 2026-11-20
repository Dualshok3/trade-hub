"""Der LLM-Agent mit Fake-Modell: Graph, Risk-Check, Ablehnungen, Journal, Aufzeichnung und Runner."""

import json
from datetime import timedelta

import pytest
from harness_it_all.models.base import ModelError
from harness_it_all.models.fake import FakeModel
from harness_it_all.recorder import Recorder, read_records

from trading_agent.agent import LLMAgent
from trading_agent.agent.config import DEFAULT_AGENT, AgentConfig, RiskConfig, load_agent_config
from trading_agent.agent.models import load_profile
from trading_agent.agent.risk import check
from trading_agent.agent.schema import ParseError, decision_schema, parse_decision
from trading_agent.depot import Depot, DepotRules
from trading_agent.execution import make_order
from trading_agent.runner import run_tick
from trading_agent.snapshot import build_fill_quotes
from trading_agent.store import DepotStore
from trading_agent.tools import SnapshotTools

from .conftest import NOW
from .test_execution import fill_provider

RULES = DepotRules(fee_eur=1.0, half_spread=0.0, min_order_pct=0.10, pending_max_ticks=3)
CFG = load_agent_config()
PROFILE = load_profile(CFG)


def answer(*orders, note="weiter beobachten"):
    return json.dumps({"assessment": "Lage", "orders": list(orders), "note": note})


def order(asset="SAP", side="buy", pct=20, sources=("N1",), reason="Prognose angehoben"):
    return {"asset": asset, "side": side, "pct": pct, "reason": reason, "sources": list(sources)}


def scripted(*texts):
    """Fake-Modell, das nacheinander die gegebenen Antworten liefert."""
    it = iter(texts)
    return FakeModel("fake-gemma", answer=lambda messages, schema: next(it))


def depot(start=100.0):
    return Depot("gemma-100", start, RULES)


def prices(snapshot):
    return {a: q["price_eur"] for a, q in snapshot["quotes"].items()}


# ------------------------------------------------------------------ Schema und Antwort lesen


def test_schema_is_closed_and_uses_only_common_features(snapshot):
    s = decision_schema(["SAP", "AAPL"])
    assert s["additionalProperties"] is False
    item = s["properties"]["orders"]["items"]
    assert item["additionalProperties"] is False
    assert item["properties"]["asset"]["enum"] == ["SAP", "AAPL"]
    text = json.dumps(s)
    for unsupported in ("minimum", "maximum", "maxItems", "pattern"):
        assert unsupported not in text


def test_parse_decision_accepts_fences_and_stray_text():
    d = parse_decision("```json\n" + answer(order()) + "\n```")
    assert d["orders"][0]["asset"] == "SAP"
    d = parse_decision("thought\n" + answer())
    assert d["orders"] == []


@pytest.mark.parametrize(
    "text",
    [
        "kein json",
        '{"orders": 5}',
        '{"orders": [{"asset": "SAP"}]}',
        '{"orders": [{"asset": "SAP", "side": "buy", "pct": 0.2}]}',
    ],
)
def test_parse_decision_rejects_bad_answers(text):
    with pytest.raises(ParseError):
        parse_decision(text)


# ------------------------------------------------------------------ Risk-Check


def risk(orders, snapshot, d=None, limits=None):
    tools = SnapshotTools(snapshot)
    return check(
        orders, d or depot(), prices(snapshot), set(tools.assets), tools.news_ids(), limits or CFG.risk
    )


def test_risk_accepts_valid_orders(snapshot):
    assert risk([order(), order("DAX", pct=10, sources=["indicator:sma50"])], snapshot) == []


@pytest.mark.parametrize(
    ("orders", "expected"),
    [
        ([order(pct=5)], "Mindestorder"),
        ([order(pct=50)], "höchstens 40 %"),
        ([order(sources=[])], "gültige Quelle"),
        ([order(sources=["N999"])], "unbekannte Quellen"),
        ([order(sources=["indicator:mondphase"])], "unbekannte Quellen"),
        ([order(side="sell")], "nicht im Depot"),
        ([order(), order(pct=10)], "mehrfach"),
        ([order(asset="XYZ")], "nicht in der Watchlist"),
        ([order(), order("AAPL"), order("DAX"), order("SAP", "sell", 10)], "höchstens 3"),
        ([order(pct=40), order("AAPL", pct=40), order("DAX", pct=30)], "Nicht genug Cash"),
    ],
)
def test_risk_rejects(orders, expected, snapshot):
    violations = risk(orders, snapshot)
    assert any(expected in v for v in violations), violations


def test_risk_counts_existing_weight(snapshot, watchlist, provider):
    d = depot()
    fill = build_fill_quotes(watchlist, fill_provider(provider), snapshot, now=NOW + timedelta(minutes=1))
    from trading_agent.execution import execute

    out = execute(d, [make_order(d, snapshot, "SAP", "buy", 0.30)], fill)  # bucht die Fills selbst
    assert len(out.fills) == 1 and d.shares("SAP") > 0
    assert any("Gewicht" in v for v in risk([order(pct=20)], snapshot, d))
    assert risk([order("SAP", "sell", 10, sources=[])], snapshot, d) == []


def test_risk_rules_scale_with_depot_size(snapshot):
    """Dieselbe Entscheidung ist bei 100 € und 10.000 € gleich gültig oder ungültig."""
    for o in ([order(pct=20)], [order(pct=5)], [order(pct=50)]):
        assert bool(risk(o, snapshot, depot(100))) == bool(risk(o, snapshot, depot(10_000)))


# ------------------------------------------------------------------ Graph


def test_trade_on_first_try(snapshot):
    model = scripted(answer(order()))
    agent = LLMAgent(model, PROFILE, CFG)
    decisions = agent(snapshot, depot())
    assert decisions == [("SAP", "buy", 0.2, "Prognose angehoben")]
    tr = agent.trace
    assert tr["outcome"] == "trade"
    assert tr["rejections"] == [] and len(tr["calls"]) == 1
    assert "SAP" in tr["focus"]
    assert tr["news"]["news_rank_version"] and "SAP" in tr["news"]["shown_ids"]
    system, user = model.calls[0][0]["content"], model.calls[0][1]["content"]
    assert "Mindestens 10 %" in system and "## Depot" in user and "[N1]" in user


def test_rejection_is_fed_back_and_corrected(snapshot):
    model = scripted(answer(order(pct=80)), answer(order(pct=30)))
    agent = LLMAgent(model, PROFILE, CFG)
    assert agent(snapshot, depot()) == [("SAP", "buy", 0.3, "Prognose angehoben")]
    assert agent.trace["outcome"] == "trade"
    assert len(agent.trace["rejections"]) == 1
    second = model.calls[1]
    assert second[-2]["role"] == "assistant" and "abgelehnt" in second[-1]["content"]


def test_two_rejections_force_hold(snapshot):
    model = scripted(answer(order(pct=80)), answer(order(pct=90)), answer(order()))
    agent = LLMAgent(model, PROFILE, CFG)
    assert agent(snapshot, depot()) == []
    assert agent.trace["outcome"] == "forced_hold"
    assert len(agent.trace["rejections"]) == CFG.risk.max_rejections == 2
    assert len(model.calls) == 2  # kein dritter Versuch


def test_invalid_json_counts_as_schema_error(snapshot):
    agent = LLMAgent(scripted("Ich würde SAP kaufen.", answer()), PROFILE, CFG)
    assert agent(snapshot, depot()) == []
    assert agent.trace["outcome"] == "hold"
    assert agent.trace["schema_errors"] == 1


def test_quiet_market_skips_the_model(snapshot):
    calm = AgentConfig(
        screen=CFG.screen.__class__(day_move=1.0, move_5d=1.0, min_news_score=99, idle_cash_share=1.1),
        risk=CFG.risk,
    )
    model = scripted()
    agent = LLMAgent(model, PROFILE, calm)
    assert agent(snapshot, depot()) == []
    assert agent.trace["outcome"] == "quiet" and model.calls == []


def test_model_error_is_not_a_rejection(snapshot):
    class Broken(FakeModel):
        def complete(self, messages, profile, schema=None):
            raise ModelError("Verbindung abgelehnt")

    agent = LLMAgent(Broken("kaputt"), PROFILE, CFG)
    assert agent(snapshot, depot()) == []
    assert agent.trace["outcome"] == "error"
    assert agent.trace["rejections"] == []
    assert "Verbindung" in agent.trace["model_error"]


def test_journal_notes_reach_the_prompt(snapshot):
    model = scripted(answer())
    agent = LLMAgent(model, PROFILE, CFG, journal=lambda name: ["SAP nach Zahlen beobachten"])
    agent(snapshot, depot())
    assert "SAP nach Zahlen beobachten" in model.calls[0][1]["content"]


def test_calls_are_recorded_for_replay(snapshot, tmp_path):
    rec = Recorder(tmp_path / "decisions.jsonl")
    agent = LLMAgent(scripted(answer(order(pct=80)), answer()), PROFILE, CFG, rec)
    agent(snapshot, depot())
    records = list(read_records(tmp_path / "decisions.jsonl"))
    assert len(records) == 2
    assert records[0].tags["snapshot"] == snapshot["hash"] and records[0].tags["week"] == "2026-W47"
    assert records[1].tags["attempt"] == 2
    assert records[0].schema and records[0].messages  # Eingabe gespeichert → Replay möglich
    assert [c["record"] for c in agent.trace["calls"]] == [r.id for r in records]


def test_both_models_get_identical_prompts(snapshot):
    a, b = scripted(answer()), scripted(answer())
    LLMAgent(a, PROFILE, CFG)(snapshot, depot())
    LLMAgent(b, PROFILE, CFG)(snapshot, depot())
    assert a.calls == b.calls


# ------------------------------------------------------------------ mit Runner und Datenbank


def test_agent_in_run_tick_stores_trace_and_journal(snapshot, watchlist, provider, tmp_path):
    store = DepotStore(tmp_path / "d.sqlite")
    fill = build_fill_quotes(watchlist, fill_provider(provider), snapshot, now=NOW + timedelta(minutes=1))
    agent = LLMAgent(scripted(answer(order(), note="SAP halten bis Zahlen")), PROFILE, CFG)
    results = run_tick(snapshot, fill, store, {"gemma": agent}, 100.0, RULES)
    r = results["gemma-100"]
    assert len(r["fills"]) == 1 and r["agent"]["outcome"] == "trade"
    assert store.notes("gemma-100") == [f"{snapshot['taken_at'][:16]}: SAP halten bis Zahlen"]
    store.close()


def test_agent_config_has_version_and_models():
    cfg = load_agent_config(DEFAULT_AGENT)
    assert len(cfg.version) == 12
    assert {"gemma", "opus"} <= set(cfg.models)
    assert isinstance(cfg.risk, RiskConfig)
