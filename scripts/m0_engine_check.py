"""Meilenstein 0: Prüft, was die crit-Engine für den Trading-Agenten kann.

Nur Standardbibliothek. Aufruf:
    python scripts/m0_engine_check.py [--base http://127.0.0.1:8000] [--repeats 5] [--out runs/m0]

Prüft:
  1. /health und /v1/models (Capabilities, Limits, Kontext)
  2. Freitext: News auf Deutsch zusammenfassen
  3. Strukturierte Ausgabe per response_format json_schema (falls chat-constraints-v1)
  4. Native Tool-Aufrufe inkl. Tool-Ergebnis zurückgeben (falls chat-tools-v1)
  5. Prompt-Modus ohne native Tools: Entscheidung als JSON im Freitext
  6. Parallele Requests (für Self-Consistency)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

NEWS = [
    {
        "id": "N1",
        "ticker": "SAP",
        "title": "SAP hebt Cloud-Prognose an, Aktie steigt vorbörslich um 4 %",
        "summary": "Der Softwarekonzern erwartet für das Gesamtjahr ein Cloud-Wachstum "
        "von 27 statt 25 Prozent.",
    },
    {
        "id": "N2",
        "ticker": "BAS",
        "title": "BASF senkt Ausblick wegen schwacher Chemienachfrage",
        "summary": "Das Ebitda soll am unteren Ende der Spanne liegen, "
        "Analysten erwarten weitere Sparmaßnahmen.",
    },
    {
        "id": "N3",
        "ticker": "SIE",
        "title": "Siemens erhält Großauftrag für Bahntechnik in Indien",
        "summary": "Das Volumen liegt bei rund 3 Milliarden Euro, Auslieferung ab 2028.",
    },
]
PORTFOLIO = {"cash_eur": 100.0, "positions": [], "fee_per_order_eur": 1.0, "fractional_shares": True}
QUOTES = {"SAP": 251.40, "BAS": 44.12, "SIE": 212.75}
TICKERS = list(QUOTES)

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
        "ticker": {"type": "string", "enum": [*TICKERS, "NONE"]},
        "amount_eur": {"type": "integer", "minimum": 0, "maximum": 100},
        "reason": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["action", "ticker", "amount_eur", "reason", "sources"],
    "additionalProperties": False,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Aktueller Kurs einer Aktie in Euro.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "enum": TICKERS}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": "Cash und Positionen des Musterdepots.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM = (
    "Du bist ein vorsichtiger Trading-Agent für ein Musterdepot mit Spielgeld. "
    "Du triffst Entscheidungen nur auf Basis der gegebenen Daten und zitierst News über ihre ID."
)


def http(method, url, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null"), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw), time.perf_counter() - t0
        except json.JSONDecodeError:
            return e.code, {"raw": raw}, time.perf_counter() - t0


def chat(base, model, messages, **extra):
    body = {"model": model, "messages": messages, "max_tokens": extra.pop("max_tokens", 600), **extra}
    return http("POST", f"{base}/v1/chat/completions", body)


def check_decision(d):
    errors = []
    if not isinstance(d, dict):
        return ["kein Objekt"]
    for k in DECISION_SCHEMA["required"]:
        if k not in d:
            errors.append(f"fehlt: {k}")
    if d.get("action") not in ("buy", "sell", "hold"):
        errors.append("action ungültig")
    if d.get("ticker") not in [*TICKERS, "NONE"]:
        errors.append("ticker ungültig")
    amt = d.get("amount_eur")
    if not isinstance(amt, (int, float)) or not 0 <= amt <= 100:
        errors.append("amount_eur ungültig")
    srcs = d.get("sources") or []
    bad = [s for s in srcs if s not in {n["id"] for n in NEWS}]
    if bad:
        errors.append(f"erfundene Quellen: {bad}")
    return errors


def extract_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def usage_of(resp):
    u = (resp or {}).get("usage") or {}
    return {k: u.get(k) for k in ("prompt_tokens", "completion_tokens")}


def news_block():
    return "\n".join(f"[{n['id']}] {n['ticker']}: {n['title']}. {n['summary']}" for n in NEWS)


def case_summary(base, model):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": "Fasse diese Meldungen in drei kurzen deutschen Stichpunkten zusammen "
            "und nenne je die ID:\n" + news_block(),
        },
    ]
    st, r, dt = chat(base, model, msgs, temperature=0.3)
    text = (r or {}).get("choices", [{}])[0].get("message", {}).get("content") if st == 200 else None
    ok = bool(text) and all(n["id"] in text for n in NEWS)
    return {
        "ok": ok,
        "status": st,
        "seconds": round(dt, 2),
        "usage": usage_of(r),
        "text": text,
        "error": None if st == 200 else r,
    }


def decision_prompt():
    return (
        f"Depot: {json.dumps(PORTFOLIO, ensure_ascii=False)}\nKurse (EUR): {json.dumps(QUOTES)}\n"
        f"News:\n{news_block()}\n\nTriff genau eine Entscheidung für den nächsten Takt."
    )


def case_json_schema(base, model):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": decision_prompt()}]
    st, r, dt = chat(
        base,
        model,
        msgs,
        temperature=0.7,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "decision", "schema": DECISION_SCHEMA, "strict": True},
        },
    )
    if st != 200:
        return {"ok": False, "status": st, "seconds": round(dt, 2), "error": r}
    text = r["choices"][0]["message"].get("content")
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        d = None
    errs = check_decision(d) if d is not None else ["kein gültiges JSON"]
    return {
        "ok": not errs,
        "status": st,
        "seconds": round(dt, 2),
        "usage": usage_of(r),
        "decision": d,
        "errors": errs,
    }


def run_tool(name, args):
    if name == "get_quote":
        t = args.get("ticker")
        return {"ticker": t, "price_eur": QUOTES.get(t)} if t in QUOTES else {"error": "unbekannter Ticker"}
    if name == "get_portfolio":
        return PORTFOLIO
    return {"error": f"unbekanntes Tool {name}"}


def case_tools(base, model):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": "Wie viel kostet eine SAP-Aktie gerade und wie viel Cash habe ich? "
            "Nutze die Tools und antworte dann in einem Satz.",
        },
    ]
    calls_made, t_total = [], 0.0
    for _ in range(4):
        st, r, dt = chat(base, model, msgs, tools=TOOLS, tool_choice="auto", temperature=0.2)
        t_total += dt
        if st != 200:
            return {"ok": False, "status": st, "seconds": round(t_total, 2), "error": r, "calls": calls_made}
        msg = r["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        if not tcs:
            text = msg.get("content") or ""
            ok = bool(calls_made) and "251" in text and "100" in text
            return {"ok": ok, "status": st, "seconds": round(t_total, 2), "calls": calls_made, "text": text}
        msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
        for tc in tcs:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
                valid = True
            except json.JSONDecodeError:
                args, valid = {}, False
            calls_made.append({"name": fn, "args": args, "valid_json": valid})
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(run_tool(fn, args), ensure_ascii=False),
                }
            )
    return {"ok": False, "seconds": round(t_total, 2), "calls": calls_made, "error": "zu viele Runden"}


def case_prompt_mode(base, model):
    schema_hint = json.dumps(
        {k: v.get("enum", v.get("type")) for k, v in DECISION_SCHEMA["properties"].items()},
        ensure_ascii=False,
    )
    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": decision_prompt() + f"\n\nAntworte NUR mit einem JSON-Objekt dieser Form: "
            f"{schema_hint}",
        },
    ]
    st, r, dt = chat(base, model, msgs, temperature=0.7)
    if st != 200:
        return {"ok": False, "status": st, "seconds": round(dt, 2), "error": r}
    text = r["choices"][0]["message"].get("content")
    d = extract_json(text)
    errs = check_decision(d) if d is not None else ["kein JSON gefunden"]
    return {
        "ok": not errs,
        "status": st,
        "seconds": round(dt, 2),
        "usage": usage_of(r),
        "decision": d,
        "errors": errs,
    }


def case_parallel(base, model, n=4):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": decision_prompt() + "\nAntworte in zwei Sätzen."},
    ]
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(n) as ex:
        res = list(ex.map(lambda _: chat(base, model, msgs, temperature=0.8, max_tokens=150), range(n)))
    wall = time.perf_counter() - t0
    single = [round(dt, 2) for _, _, dt in res]
    return {
        "ok": all(st == 200 for st, _, _ in res),
        "n": n,
        "wall_seconds": round(wall, 2),
        "per_request_seconds": single,
        "statuses": [st for st, _, _ in res],
        "errors": [r for st, r, _ in res if st != 200][:1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="runs/m0")
    a = ap.parse_args()

    report = {"base": a.base, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _, health, _ = http("GET", f"{a.base}/health")
    report["health"] = (
        {k: health.get(k) for k in ("status", "ready", "loaded", "engine", "max_concurrent_requests")}
        if isinstance(health, dict)
        else health
    )
    _st, models, _ = http("GET", f"{a.base}/v1/models")
    data = (models or {}).get("data", [{}])
    card = next((m for m in data if m.get("default")), data[0])
    model = card.get("id")
    caps = card.get("capabilities") or []
    report["model"] = {
        k: card.get(k)
        for k in (
            "id",
            "context_length",
            "capabilities",
            "supported_sampling_parameters",
            "limits",
            "max_concurrent_requests",
        )
    }
    print(f"Modell: {model}\nCapabilities: {caps}")

    cases = {"summary": case_summary, "prompt_mode": case_prompt_mode}
    if "chat-constraints-v1" in caps:
        cases["json_schema"] = case_json_schema
    if "chat-tools-v1" in caps or "chat-tool-template-v1" in caps:
        cases["tools"] = case_tools
    report["cases"] = {}
    for name, fn in cases.items():
        runs = []
        for i in range(a.repeats):
            res = fn(a.base, model)
            runs.append(res)
            print(f"  {name} #{i + 1}: {'OK ' if res['ok'] else 'FEHLER'} ({res.get('seconds')} s)")
        report["cases"][name] = {"pass": sum(r["ok"] for r in runs), "of": len(runs), "runs": runs}
    report["cases"]["parallel"] = case_parallel(a.base, model)
    print(f"  parallel: {report['cases']['parallel']['wall_seconds']} s für 4 Requests")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"m0-{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nBericht: {path}")


if __name__ == "__main__":
    main()
