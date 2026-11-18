from datetime import UTC, datetime, timedelta

from trading_agent.feeds.news import (
    NewsItem,
    alias_patterns,
    clean_text,
    deduplicate,
    google_news_url,
    match_assets,
    parse_feed,
    within,
)

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>t</title>
<item>
  <title>SAP hebt Prognose an - Handelsblatt</title>
  <link>https://news.example/a</link>
  <pubDate>Wed, 18 Nov 2026 08:00:00 GMT</pubDate>
  <description>&lt;a href="x"&gt;SAP hebt Prognose an&lt;/a&gt;&amp;nbsp;&amp;nbsp;&lt;font&gt;Handelsblatt&lt;/font&gt;</description>
  <source url="https://handelsblatt.com">Handelsblatt</source>
</item>
<item>
  <title>Apple and Microsoft lead Wall Street higher - Reuters</title>
  <link>https://news.example/b</link>
  <pubDate>Wed, 18 Nov 2026 07:00:00 +0000</pubDate>
  <description>Tech stocks rallied as Apple gained 2%.</description>
  <source url="https://reuters.com">Reuters</source>
</item>
<item>
  <title>Sapporo beer sales rise - Nikkei</title>
  <link>https://news.example/c</link>
  <pubDate>Wed, 18 Nov 2026 06:00:00 GMT</pubDate>
  <description>Nothing about software.</description>
  <source url="https://nikkei.com">Nikkei</source>
</item>
</channel></rss>"""


def test_parse_feed(watchlist):
    items = parse_feed(RSS, "SAP Aktie", alias_patterns(watchlist))
    assert [i.title for i in items] == [
        "SAP hebt Prognose an",
        "Apple and Microsoft lead Wall Street higher",
        "Sapporo beer sales rise",
    ]
    sap, apple, sapporo = items
    assert sap.source == "Handelsblatt"
    assert sap.snippet == ""  # Anriss wiederholte nur Titel und Quelle
    assert sap.published == "2026-11-18T08:00:00+00:00"
    assert sap.assets == ("SAP",)
    assert apple.assets == ("AAPL",)
    assert sapporo.assets == ()  # „Sapporo“ ist kein Treffer für „SAP“
    assert sap.id.startswith("N") and len(sap.id) == 11


def test_alias_word_boundaries(watchlist):
    pats = alias_patterns(watchlist)
    assert match_assets("DAX schließt im Plus", pats) == ("DAX",)
    assert match_assets("MDAX und TecDAX fallen", pats) == ()
    assert match_assets("apple-aktie steigt", pats) == ("AAPL",)


def test_clean_text():
    assert clean_text("<b>Hallo</b>&nbsp;&amp; <i>Welt</i>") == "Hallo & Welt"


def _item(i, title, hours_ago, assets, now):
    return NewsItem(i, (now - timedelta(hours=hours_ago)).isoformat(), title, "", "l", "s", "q", assets)


def test_deduplicate_merges_assets_and_keeps_earliest():
    now = datetime(2026, 11, 18, 12, tzinfo=UTC)
    a = _item("N1", "SAP und Siemens legen zu", 1, ("SAP",), now)
    b = _item("N2", "SAP und Siemens legen zu!", 3, ("SIE",), now)
    c = _item("N3", "Etwas anderes", 2, (), now)
    out = deduplicate([a, b, c])
    assert len(out) == 2
    merged = next(n for n in out if n.title.startswith("SAP"))
    assert merged.id == "N2" and set(merged.assets) == {"SAP", "SIE"}
    assert out[0].published >= out[1].published


def test_within_drops_future_and_old():
    now = datetime(2026, 11, 18, 12, tzinfo=UTC)
    items = [_item("A", "a", 1, (), now), _item("B", "b", 30, (), now), _item("C", "c", -1, (), now)]
    assert [i.id for i in within(items, now, 24)] == ["A"]


def test_google_news_url():
    url = google_news_url("S&P 500", "en")
    assert "q=S%26P+500+when:2d" in url and "hl=en-US" in url
