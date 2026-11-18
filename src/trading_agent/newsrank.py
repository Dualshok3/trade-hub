"""News zuordnen, Rauschen aussortieren, ähnliche Meldungen bündeln und nach Relevanz sortieren.

Alles ist deterministisch und läuft beim Lesen eines Snapshots. Ein Snapshot speichert die News
unverändert, damit spätere, bessere Regeln auch auf alte Snapshots anwendbar sind (Replay).
Die Regeln stehen in `configs/news_rank.toml`.

Relevanz einer Meldung für einen Wert:

    score = Quelle × Ereignis × Thema × Aktualität × Abdeckung

- Quelle: Gewicht der Nachrichtenquelle (seriöse Wirtschaftspresse hoch, automatisch erzeugte Seiten niedrig)
- Ereignis: Schlagwörter für kursrelevante Ereignisse (Prognose, Übernahme, Kursziel, earnings, …)
- Thema: Steht der Wert im Titel, und zwar vorne? Oder nur im Anriss?
- Aktualität: Halbwertszeit in Stunden
- Abdeckung: Wie viele Quellen berichten dieselbe Geschichte?
"""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DEFAULT_RULES = Path(__file__).resolve().parents[2] / "configs" / "news_rank.toml"

# „DAX 40-Titel Henkel“, „S&P 500-Wert Avery Dennison“: Der Index ist hier nur Etikett, nicht das Thema
INDEX_LABEL = re.compile(
    r"(?<!\w)(?:DAX|MDAX|SDAX|TecDAX|Dow Jones|NASDAQ(?:[ -]?100)?|S&P[ -]?500"
    r"|Euro ?STOXX ?50|STOXX Europe 600)"
    r"(?: ?40)?-(?:Titel|Wert|Papier|Aktie|Mitglied|Unternehmen)\b",
    re.IGNORECASE,
)

_STOPWORDS_TEXT = (
    "der die das und oder mit von für auf ist im in am an zu den dem des ein eine einer "
    "nach bei als aus wie the a an and or of for on in at to is are as by with from after "
    "over its it this that be will has have aktie aktien stock stocks shares news"
)
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())

# „JPMorgan upgrades Millicom“, „JP Morgan gives a Buy rating“: Die Bank ist hier Analyst, nicht das Thema
ANALYST_ROLE = re.compile(
    r"\s*(?:'s)?\s*(?:initiates|upgrades|downgrades|reiterates|raises|lowers|cuts|gives|assigns|rates|maintains"
    r"|resumes|starts|sets|stuft|bestätigt|vergibt|kappt|sieht)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Rules:
    noise: tuple[re.Pattern, ...]
    penalty: tuple[re.Pattern, ...]
    penalty_factor: float
    events: re.Pattern
    event_factor: float
    sources: dict[str, float]
    default_source: float
    half_life_hours: float
    cluster_jaccard: float
    macro: re.Pattern


def _rx(patterns: list[str]) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


def load_rules(path: Path | str = DEFAULT_RULES) -> Rules:
    d = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    sources = {}
    for weight_name, names in d["sources"].items():
        weight = d["source_weights"][weight_name]
        for name in names:
            sources[name.lower()] = weight
    return Rules(
        noise=_rx(d["noise"]["patterns"]),
        penalty=_rx(d["penalty"]["patterns"]),
        penalty_factor=d["penalty"]["factor"],
        events=re.compile(r"(?<!\w)(?:" + "|".join(d["events"]["words"]) + r")(?!\w)", re.IGNORECASE),
        event_factor=d["events"]["factor"],
        sources=sources,
        default_source=d["source_weights"]["default"],
        half_life_hours=d["scoring"]["half_life_hours"],
        cluster_jaccard=d["scoring"]["cluster_jaccard"],
        macro=re.compile(r"(?<!\w)(?:" + "|".join(d["macro"]["words"]) + r")(?!\w)", re.IGNORECASE),
    )


# ---------------------------------------------------------------------------------------------- Zuordnung


def _alias_regex(alias: str) -> re.Pattern:
    # Kurze Kürzel in Großbuchstaben (SAP, BASF, DAX) nur genau so geschrieben;
    # sonst trifft „Sap“ oder „Dax“ alles Mögliche
    flags = 0 if (alias.isupper() and len(alias) <= 5) else re.IGNORECASE
    return re.compile(rf"(?<![\w&]){re.escape(alias)}(?![\w&])", flags)


@dataclass
class AssetMatcher:
    """Ordnet Text Werten der Watchlist zu.

    `assets`: Dicts oder Objekte mit id, aliases und optional exclude."""

    assets: list
    _rules: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        for a in self.assets:
            get = a.get if isinstance(a, dict) else lambda k, d=None, a=a: getattr(a, k, d)
            aliases = sorted(get("aliases", ()) or (), key=len, reverse=True)
            excludes = [_alias_regex(x) for x in (get("exclude", ()) or ())]
            self._rules.append((get("id"), [_alias_regex(x) for x in aliases], excludes))

    def positions(self, text: str) -> dict[str, int]:
        """Wert → Position des ersten Treffers im Text (Zeichen), nur Werte mit Treffer.

        Nicht gezählt werden: Index-Etiketten („DAX 40-Titel …“), ausgeschlossene Namen („Siemens Energy“) und
        Treffer, bei denen der Wert als Analyst auftritt („JPMorgan upgrades …“)."""
        text = INDEX_LABEL.sub(lambda m: " " * len(m.group(0)), text)
        out = {}
        for asset_id, aliases, excludes in self._rules:
            masked = text
            for ex in excludes:
                masked = ex.sub(lambda m: " " * len(m.group(0)), masked)
            hits = [
                m.start()
                for p in aliases
                for m in p.finditer(masked)
                if not ANALYST_ROLE.match(masked, m.end())
            ]
            if hits:
                out[asset_id] = min(hits)
        return out

    def match(self, text: str) -> tuple[str, ...]:
        return tuple(self.positions(text))


# ---------------------------------------------------------------------------------------------- Bewertung


def is_noise(title: str, rules: Rules) -> bool:
    return any(p.search(title) for p in rules.noise)


def source_weight(source: str, rules: Rules) -> float:
    return rules.sources.get((source or "").lower(), rules.default_source)


def tokens(title: str) -> frozenset[str]:
    words = re.findall(r"[a-zäöüß0-9]+", INDEX_LABEL.sub(" ", title.lower()))
    return frozenset(w for w in words if len(w) >= 3 and w not in STOPWORDS)


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


@dataclass
class Scored:
    item: dict
    score: float
    parts: dict
    tokens: frozenset
    also: list = field(default_factory=list)  # weitere Quellen derselben Geschichte

    def to_dict(self) -> dict:
        out = {k: self.item[k] for k in ("id", "published", "title", "snippet", "source")}
        out["score"] = round(self.score, 3)
        out["also_reported_by"] = sorted({s for s in self.also if s and s != self.item["source"]})
        out["sources_count"] = 1 + len(self.also)
        return out


def score_item(item: dict, asset_pos: int | None, in_title: bool, now: datetime, rules: Rules) -> Scored:
    title = item["title"]
    age_h = max((now - datetime.fromisoformat(item["published"])).total_seconds() / 3600, 0.0)
    parts = {
        "source": source_weight(item["source"], rules),
        "event": rules.event_factor if rules.events.search(f"{title} {item['snippet']}") else 1.0,
        "penalty": rules.penalty_factor if any(p.search(title) for p in rules.penalty) else 1.0,
        "recency": 0.5 ** (age_h / rules.half_life_hours),
    }
    if asset_pos is None:
        parts["subject"] = 1.0
    elif not in_title:
        parts["subject"] = 0.5
    else:
        parts["subject"] = 1.3 if asset_pos <= 25 else 1.0
    score = math.prod(parts.values())
    return Scored(item, score, parts, tokens(title))


def cluster(scored: list[Scored], threshold: float) -> list[Scored]:
    """Ähnliche Titel zu einer Geschichte bündeln.

    Vertreter ist die bestbewertete Meldung, die Abdeckung durch mehrere Quellen erhöht ihren Score."""
    reps: list[Scored] = []
    for s in sorted(scored, key=lambda s: s.score, reverse=True):
        rep = next((r for r in reps if jaccard(r.tokens, s.tokens) >= threshold), None)
        if rep is None:
            reps.append(s)
        else:
            rep.also.append(s.item["source"])
    for r in reps:
        sources = len({r.item["source"], *r.also})
        r.parts["coverage"] = 1 + 0.25 * math.log2(sources)
        r.score *= r.parts["coverage"]
    return sorted(reps, key=lambda s: s.score, reverse=True)


class NewsRanker:
    def __init__(self, snapshot: dict, rules: Rules | None = None):
        self.rules = rules or load_rules()
        self.now = datetime.fromisoformat(snapshot["taken_at"])
        self.matcher = AssetMatcher(snapshot["watchlist"])
        self.items = [n for n in snapshot["news"] if n["published"]]
        self.noise = [n for n in self.items if is_noise(n["title"], self.rules)]
        noise_ids = {n["id"] for n in self.noise}
        self.kept = [n for n in self.items if n["id"] not in noise_ids]
        # Zuordnung neu berechnen (bessere Regeln gelten so auch für alte Snapshots)
        self.title_pos = {n["id"]: self.matcher.positions(n["title"]) for n in self.kept}
        self.snippet_hit = {n["id"]: set(self.matcher.match(n["snippet"])) for n in self.kept}

    def assets_of(self, item: dict) -> set[str]:
        return set(self.title_pos[item["id"]]) | self.snippet_hit[item["id"]]

    def _window(self, hours: float) -> list[dict]:
        return [
            n
            for n in self.kept
            if 0 <= (self.now - datetime.fromisoformat(n["published"])).total_seconds() <= hours * 3600
        ]

    def for_asset(self, asset_id: str, hours: float = 24, limit: int = 5) -> list[dict]:
        scored = []
        for n in self._window(hours):
            pos = self.title_pos[n["id"]].get(asset_id)
            if pos is None and asset_id not in self.snippet_hit[n["id"]]:
                continue
            scored.append(score_item(n, pos, pos is not None, self.now, self.rules))
        return [s.to_dict() for s in cluster(scored, self.rules.cluster_jaccard)[:limit]]

    def market(self, hours: float = 24, limit: int = 5) -> list[dict]:
        """Allgemeine Markt- und Makro-Meldungen (Zinsen, Inflation, Indizes)."""
        scored = []
        for n in self._window(hours):
            text = f"{n['title']} {n['snippet']}"
            if not self.rules.macro.search(text):
                continue
            scored.append(score_item(n, None, False, self.now, self.rules))
        return [s.to_dict() for s in cluster(scored, self.rules.cluster_jaccard)[:limit]]

    def stats(self) -> dict:
        return {
            "total": len(self.items),
            "noise": len(self.noise),
            "kept": len(self.kept),
            "unassigned": sum(1 for n in self.kept if not self.assets_of(n)),
        }
