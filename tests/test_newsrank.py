"""News-Sortierung: Zuordnung, Rauschen, Bündelung, Relevanz. Titel sind nachgebaute Beispiele echter Muster."""

from datetime import UTC, datetime, timedelta

import pytest

from trading_agent.newsrank import AssetMatcher, NewsRanker, is_noise, load_rules

RULES = load_rules()
NOW = datetime(2026, 11, 18, 10, 0, tzinfo=UTC)

WATCHLIST = [
    {"id": "SAP", "aliases": ["SAP"], "exclude": []},
    {"id": "SIE", "aliases": ["Siemens"], "exclude": ["Siemens Energy", "Siemens Healthineers"]},
    {"id": "JPM", "aliases": ["JPMorgan", "JP Morgan"], "exclude": []},
    {"id": "DAX", "aliases": ["DAX"], "exclude": []},
    {"id": "DTE", "aliases": ["Deutsche Telekom", "Telekom"], "exclude": []},
]


@pytest.fixture
def matcher():
    return AssetMatcher(WATCHLIST)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SAP hebt Prognose an", ("SAP",)),
        ("Sapporo meldet Rekordabsatz", ()),
        ("Siemens Energy verliert am Nachmittag", ()),
        ("Siemens und Siemens Energy im Fokus", ("SIE",)),
        ("Siemens Healthineers: Neuer Scanner", ()),
        ("Deutsche Telekom-Aktie: Sie steigt weiter", ("DTE",)),  # „Sie“ ist nicht Siemens (SIE)
        ("DAX 40-Titel Henkel-Aktie: So viel Gewinn", ()),  # Index nur als Etikett
        ("DAX schließt im Plus", ("DAX",)),
        ("MDAX und TecDAX fallen", ()),
        ("JPMorgan upgrades Millicom stock rating", ()),  # JPMorgan als Analyst
        ("Microsoft Corporation : JP Morgan gives a Buy rating", ()),
        ("JPMorgan names new CFO", ("JPM",)),
        ("JPMorgan’s CFO plans exit", ("JPM",)),
        ("JPMorgan upgrades Millicom, JPMorgan names new CFO", ("JPM",)),  # ein echter Treffer reicht
    ],
)
def test_matching(matcher, text, expected):
    assert matcher.match(text) == expected


@pytest.mark.parametrize(
    "title",
    [
        "DAX 40-Wert Mercedes-Benz Group-Aktie: So viel Verlust hätte ein Investment eingebracht",
        "S&P 500-Wert Avery Dennison-Aktie: So viel hätten Anleger verdient",
        "Große Auswahl - Open End Turbos auf DAX, Nasdaq-100 und NVIDIA",
        "Hannover Rück Aktie News: Hannover Rück legt am Vormittag zu",
        "iShares MSCI World ETF verteuert sich",
        "Amazon.com, Inc. $AMZN Stock: Aprio Wealth Management LLC Grows Holdings",
        "Apple Inc. $AAPL Stock Acquired by Sunesis Advisors LLC",
        "Key facts: Citadel, State Street Transfers; JPMorgan Adds 1,328 Shares",
        "Option traders moderately bearish in Pfizer with shares down 1.23%",
        "A live webcast will carry Target Hospitality's J.P. Morgan presentation today.",
    ],
)
def test_noise(title):
    assert is_noise(title, RULES)


@pytest.mark.parametrize(
    "title",
    [
        "SAP hebt Cloud-Prognose an",
        "Siemens erhöht die Prognose",
        "Nvidia beats forecasts and rules out Huang’s return as CEO",
        'JEFFERIES: BASF "sell"',
        "Allianz-Aktie: Barclays Capital vergibt Bewertung mit Underweight",
        "Dax erleidet massive Verluste am Nachmittag",  # echte Marktmeldung trotz Tageszeit
        "Fed's Musalem: US likely needs several rate hikes by spring",
    ],
)
def test_not_noise(title):
    assert not is_noise(title, RULES)


def _news(i, title, source, hours_ago, snippet=""):
    return {
        "id": i,
        "published": (NOW - timedelta(hours=hours_ago)).isoformat(),
        "title": title,
        "snippet": snippet,
        "source": source,
        "link": "",
        "query": "",
        "assets": [],
    }


def _snapshot(news):
    return {"taken_at": NOW.isoformat(), "watchlist": WATCHLIST, "news": news}


def test_event_from_good_source_beats_clickbait():
    r = NewsRanker(
        _snapshot(
            [
                _news("N1", "Prediction: SAP Will Soar in 2027?", "The Motley Fool", 1),
                _news("N2", "SAP hebt Prognose für Cloud-Umsatz an", "Reuters", 3),
                _news("N3", "SAP-Aktie: Was Anleger jetzt wissen sollten", "Börse Express", 1),
            ]
        )
    )
    assert [n["id"] for n in r.for_asset("SAP")] == ["N2", "N3", "N1"]


def test_title_beats_snippet_only():
    r = NewsRanker(
        _snapshot(
            [
                _news("N1", "Softwarewerte schwach", "Reuters", 1, snippet="Auch SAP gab nach."),
                _news("N2", "SAP-Aktie unter Druck", "Reuters", 1),
            ]
        )
    )
    top = r.for_asset("SAP")
    assert [n["id"] for n in top] == ["N2", "N1"]


def test_recency_decay():
    r = NewsRanker(
        _snapshot(
            [
                _news("OLD", "SAP-Aktie: Analyst hebt Kursziel", "Reuters", 20),
                _news("NEW", "SAP-Aktie: Bank senkt Kursziel", "Reuters", 1),
            ]
        )
    )
    assert r.for_asset("SAP")[0]["id"] == "NEW"


def test_clustering_counts_sources():
    r = NewsRanker(
        _snapshot(
            [
                _news("N1", "Siemens erhöht Prognose für das Geschäftsjahr", "Reuters", 2),
                _news("N2", "Siemens erhöht die Prognose für das Geschäftsjahr", "Handelsblatt", 2),
                _news("N3", "Siemens erhöht Prognose – Aktie legt zu", "boerse.de", 1),
                _news("N4", "Siemens erhält Großauftrag in Indien", "Reuters", 2),
            ]
        )
    )
    top = r.for_asset("SIE")
    assert len(top) == 2
    assert top[0]["sources_count"] == 3
    assert set(top[0]["also_reported_by"]) | {top[0]["source"]} == {"Reuters", "Handelsblatt", "boerse.de"}


def test_window_and_noise_stats():
    r = NewsRanker(
        _snapshot(
            [
                _news("N1", "SAP-Aktie fällt", "Reuters", 1),
                _news("N2", "SAP-Aktie: So viel hätte eine Investition gebracht", "Finanzen.net", 1),
                _news("N3", "SAP-Aktie steigt", "Reuters", 30),  # außerhalb 24 h
                _news("N4", "Aus der Zukunft: SAP", "Reuters", -2),
                _news("N5", "Ölpreis steigt", "Reuters", 1),
            ]
        )
    )
    assert [n["id"] for n in r.for_asset("SAP")] == ["N1"]
    assert r.stats() == {"total": 5, "noise": 1, "kept": 4, "unassigned": 1}
    assert [n["id"] for n in r.market()] == ["N5"]


def test_old_snapshots_get_new_matching(snapshot):
    """Zuordnung wird beim Lesen neu berechnet, gespeicherte assets-Felder spielen keine Rolle."""
    for n in snapshot["news"]:
        n["assets"] = []
    from trading_agent.tools import SnapshotTools

    assert [n["id"] for n in SnapshotTools(snapshot).get_top_news("SAP")] == ["N1"]
