# Trading-Agent

[![CI](https://github.com/Dualshok3/Trading-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Dualshok3/Trading-Agent/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![Lizenz: MIT](https://img.shields.io/badge/lizenz-MIT-green)](LICENSE)

> Ein lokales Modell auf der eigenen GPU gegen Claude Opus 5.5: Zwei identische Agenten verfolgen echte Märkte und
> News und handeln in je einem Musterdepot mit **100 € Spielgeld**, mit realistischen Gebühren und deutscher
> Abgeltungsteuer. Ein Benchmark zeigt, wie nah das lokale Modell herankommt.

**Status:** im Aufbau. Fertig sind Meilenstein 0 (lokale Engine geprüft) und Meilenstein 1 (Kurse, News und
Snapshots). Den ganzen Plan beschreibt [PLAN.md](PLAN.md).

## Idee

| | Lokal | Referenz |
|---|---|---|
| Modell | Gemma 4 26B-A4B (IQ4_XS) | Claude Opus 5.5 |
| Inferenz | eigene [crit](https://github.com/crit-labs/crit)-Engine auf einer RTX 5060 Ti 16 GB, ca. 100 tok/s | Anthropic-API |
| Kosten pro Entscheidung | nur Strom | API-Preis |

Beide Agenten bekommen in jedem Takt **exakt denselben eingefrorenen Marktzustand** (Snapshot). So ist der
Vergleich fair, und jeder Takt lässt sich später wieder abspielen, etwa um neue Prompts oder Techniken auf echten
vergangenen Marktsituationen zu testen.

## Was schon geht

```text
> uv run trading-agent snapshot

            Snapshot 2026-11-18T09:19:44+00:00 · 2ce20b33117e
┌───────┬────────┬─────────┬─────────┬─────┬─────────┬───────────┬───────┐
│ Wert  │ Kurs € │     Tag │  5 Tage │ RSI │ > SMA50 │ News 24 h │ offen │
├───────┼────────┼─────────┼─────────┼─────┼─────────┼───────────┼───────┤
│ SAP   │ 202.25 │ +0.47 % │ -4.12 % │  45 │      ja │        31 │    ja │
│ NVDA  │ 141.78 │ -4.59 % │ -7.95 % │  22 │    nein │       116 │       │
│ …     │        │         │         │     │         │           │       │
└───────┴────────┴─────────┴─────────┴─────┴─────────┴───────────┴───────┘
News gesamt: 705 · offene Börsen: XETRA
Gespeichert: data\snapshots\2026-11-18\091944Z-2ce20b33117e.json.gz
```

- **Kurse** für eine feste Watchlist mit 15 Werten (DAX-Werte, US-Tech, drei ETFs) über `yfinance`. US-Werte werden
  zum aktuellen Wechselkurs in Euro umgerechnet.
- **News** aus Google-News-RSS, gesäubert, dedupliziert und den Werten zugeordnet. Gespeichert werden nur Titel,
  Anriss, Link und Quelle.
- **Snapshots** als gzip-JSON mit SHA-256-Hash. Ein veränderter Snapshot wird beim Laden erkannt und abgelehnt.
- **Schutz vor Look-ahead:** Kerzen und News, die nach dem Snapshot-Zeitpunkt liegen, werden verworfen.
- **Werkzeuge für den Agenten** (`SnapshotTools`): Kurs, Historie, Indikatoren (SMA 20/50/200, RSI, Volatilität,
  Abstand zum 52-Wochen-Hoch) und News. Sie lesen nur aus dem Snapshot, ein Test stellt sicher, dass sie nie ins Netz
  gehen.

## Quickstart

Voraussetzung: [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:Dualshok3/Trading-Agent.git && cd Trading-Agent
uv sync
uv run trading-agent snapshot            # Live-Kurse + News einfrieren
uv run trading-agent list                # gespeicherte Snapshots
uv run trading-agent show <datei>        # Hash prüfen und anzeigen
uv run pytest -q                         # Tests ohne Netz
uv run pytest -q -m network              # Live-Test gegen yfinance
```

## Wie es weitergeht

Musterdepot mit Bruchstücken, Gebühren und Steuern pro Trade (FIFO, Verlusttöpfe) → LangGraph-Agent → Anbindung
beider Modelle → Dashboard → Replay-Benchmark. Details und Messergebnisse stehen in [PLAN.md](PLAN.md).

## Hinweis

Dieses Projekt ist ein Benchmark für LLM-Agenten. Es handelt ausschließlich mit Spielgeld in einem simulierten
Depot. Es ist keine Anlageberatung, und die Ergebnisse sagen nichts darüber aus, ob man mit einer Strategie echtes
Geld verdienen kann. Die Steuerberechnung ist eine vereinfachte Nachbildung, keine Steuerberatung.

Kurse kommen über [`yfinance`](https://github.com/ranaroussi/yfinance), eine inoffizielle Schnittstelle zu Yahoo
Finance. Sie ist für private Projekte üblich, aber nicht von Yahoo freigegeben, und Kurse sind je nach Börse
verzögert (Xetra rund 15 Minuten). Jeder Snapshot speichert deshalb den Zeitpunkt des Kurses an der Börse. Für eine
kommerzielle Nutzung wäre ein Kursanbieter mit offizieller API nötig.
