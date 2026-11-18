"""News aus RSS-Feeds: holen, säubern, deduplizieren und Werten der Watchlist zuordnen.

Gespeichert werden nur Titel, ein kurzer Anriss, Link, Quelle und Zeit, keine Volltexte.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

from trading_agent.config import Watchlist

log = logging.getLogger(__name__)

SNIPPET_CHARS = 300
USER_AGENT = "trading-agent/0.1 (+https://github.com/Dualshok3/Trading-Agent)"


@dataclass(frozen=True)
class NewsItem:
    id: str
    published: str  # UTC, ISO 8601
    title: str
    snippet: str
    link: str
    source: str
    query: str
    assets: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["assets"] = list(self.assets)
        return d


def google_news_url(query: str, lang: str = "de") -> str:
    region = {"de": ("de", "DE", "DE:de"), "en": ("en-US", "US", "US:en")}[lang]
    hl, gl, ceid = region
    return f"https://news.google.com/rss/search?q={quote_plus(query)}+when:2d&hl={hl}&gl={gl}&ceid={ceid}"


_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    text = html.unescape(_TAG.sub(" ", text or ""))
    return _SPACE.sub(" ", text).strip()


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + " …"


def _published(entry) -> datetime | None:
    for key in ("published", "updated"):
        raw = entry.get(key)
        if raw:
            try:
                dt = parsedate_to_datetime(raw)
            except (TypeError, ValueError):
                continue
            return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    return None


def _split_source(title: str, entry) -> tuple[str, str]:
    """Google News hängt die Quelle als „Titel - Quelle“ an; die Quelle steht auch in entry.source."""
    source = ""
    src = entry.get("source")
    if isinstance(src, dict):
        source = src.get("title", "") or ""
    if source and title.endswith(f" - {source}"):
        title = title[: -len(source) - 3]
    return title.strip(), source


def item_id(link: str, title: str) -> str:
    return "N" + hashlib.sha1(f"{link}|{title}".encode()).hexdigest()[:10]


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9äöüß]+", " ", title.lower()).strip()


def alias_patterns(watchlist: Watchlist) -> dict[str, re.Pattern]:
    pats = {}
    for asset in watchlist.assets:
        names = sorted({*asset.aliases, asset.id}, key=len, reverse=True)
        alt = "|".join(re.escape(n) for n in names)
        pats[asset.id] = re.compile(rf"(?<![\w&])(?:{alt})(?![\w&])", re.IGNORECASE)
    return pats


def match_assets(text: str, patterns: dict[str, re.Pattern]) -> tuple[str, ...]:
    return tuple(asset_id for asset_id, pat in patterns.items() if pat.search(text))


def parse_feed(raw: bytes | str, query: str, patterns: dict[str, re.Pattern]) -> list[NewsItem]:
    import feedparser

    feed = feedparser.parse(raw)
    items = []
    for entry in feed.entries:
        title, source = _split_source(clean_text(entry.get("title")), entry)
        if not title:
            continue
        snippet = clean_text(entry.get("summary"))
        # Google News wiederholt im Anriss oft nur den Titel samt Quelle
        if snippet.startswith(title):
            snippet = snippet[len(title) :].strip(" -–")
            if snippet == source:
                snippet = ""
        published = _published(entry)
        link = entry.get("link", "")
        items.append(
            NewsItem(
                id=item_id(link, title),
                published=published.replace(microsecond=0).isoformat() if published else "",
                title=title,
                snippet=_snippet(snippet),
                link=link,
                source=source,
                query=query,
                assets=match_assets(f"{title} {snippet}", patterns),
            )
        )
    return items


def deduplicate(items: Iterable[NewsItem]) -> list[NewsItem]:
    """Gleiche Meldung aus mehreren Feeds nur einmal behalten, die Wert-Zuordnungen werden vereinigt."""
    by_title: dict[str, NewsItem] = {}
    for item in items:
        key = _normalize_title(item.title)
        prev = by_title.get(key)
        if prev is None:
            by_title[key] = item
            continue
        merged = tuple(dict.fromkeys(prev.assets + item.assets))
        keep = prev if (prev.published or "~") <= (item.published or "~") else item
        by_title[key] = NewsItem(**{**asdict(keep), "assets": merged})
    return sorted(by_title.values(), key=lambda n: n.published, reverse=True)


def within(items: Iterable[NewsItem], now: datetime, hours: float) -> list[NewsItem]:
    """Nur Meldungen aus den letzten `hours` Stunden und keine aus der Zukunft (Look-ahead-Schutz)."""
    start = now - timedelta(hours=hours)
    out = []
    for item in items:
        if not item.published:
            continue
        t = datetime.fromisoformat(item.published)
        if start <= t <= now:
            out.append(item)
    return out


class RssNewsProvider:
    """Holt Google-News-RSS für jeden Wert der Watchlist und die allgemeinen Markt-Suchen."""

    name = "google-news-rss"

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def queries(self, watchlist: Watchlist) -> list[tuple[str, str]]:
        q = []
        for asset in watchlist.assets:
            lang = "de" if asset.currency == "EUR" else "en"
            q.append((asset.news, lang))
        for text in watchlist.market_news:
            lang = "en" if any(w in text for w in ("Wall Street", "Fed", "stocks")) else "de"
            q.append((text, lang))
        return q

    def fetch(
        self, watchlist: Watchlist, now: datetime, hours: float = 24
    ) -> tuple[list[NewsItem], list[str]]:
        import httpx

        patterns = alias_patterns(watchlist)
        items: list[NewsItem] = []
        errors: list[str] = []
        with httpx.Client(
            timeout=self.timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True
        ) as c:
            for query, lang in self.queries(watchlist):
                try:
                    resp = c.get(google_news_url(query, lang))
                    resp.raise_for_status()
                except httpx.HTTPError as e:
                    errors.append(f"{query}: {e}")
                    log.warning("News-Feed %r fehlgeschlagen: %s", query, e)
                    continue
                items.extend(parse_feed(resp.content, query, patterns))
        return deduplicate(within(items, now, hours)), errors
