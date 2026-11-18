import gzip
import json

import pytest

from trading_agent.snapshot import SnapshotError, digest, load_snapshot, save_snapshot, snapshot_path

from .conftest import NOW


def test_snapshot_contents(snapshot):
    assert snapshot["taken_at"] == "2026-11-18T10:30:00+00:00"
    assert snapshot["open_exchanges"] == ["XETRA"]
    assert set(snapshot["quotes"]) == {"SAP", "AAPL", "DAX"}
    assert snapshot["errors"] == []


def test_usd_converted_to_eur(snapshot):
    assert snapshot["quotes"]["AAPL"]["price_eur"] == pytest.approx(230 / 1.15)
    assert snapshot["quotes"]["SAP"]["price_eur"] == 250.0


def test_no_future_bars(snapshot):
    """Look-ahead-Schutz: Kerzen nach dem Snapshot-Tag werden verworfen."""
    assert all(b["date"] <= "2026-11-18" for b in snapshot["history"]["SAP"])
    assert snapshot["history"]["SAP"][-1]["close"] != 999.0


def test_no_future_or_stale_news(snapshot):
    ids = [n["id"] for n in snapshot["news"]]
    assert ids == ["N1", "N2"]  # N3 älter als 24 h, N4 aus der Zukunft


def test_hash_is_deterministic(snapshot):
    assert snapshot["hash"] == digest(snapshot)
    assert len(snapshot["hash"]) == 64


def test_roundtrip(snapshot, tmp_path):
    path = save_snapshot(snapshot, tmp_path)
    assert path == snapshot_path(snapshot, tmp_path)
    assert snapshot["hash"][:12] in path.name
    assert load_snapshot(path) == snapshot
    assert save_snapshot(snapshot, tmp_path) == path  # idempotent


def test_tampered_file_is_rejected(snapshot, tmp_path):
    path = save_snapshot(snapshot, tmp_path)
    data = json.loads(gzip.decompress(path.read_bytes()))
    data["quotes"]["SAP"]["price"] = 1.0
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    with pytest.raises(SnapshotError, match="Hash"):
        load_snapshot(path)


def test_modified_snapshot_cannot_be_saved(snapshot, tmp_path):
    snapshot["news"] = []
    with pytest.raises(SnapshotError):
        save_snapshot(snapshot, tmp_path)


def test_missing_quote_and_fx_are_reported(watchlist, provider):
    from trading_agent.feeds.prices import StaticProvider
    from trading_agent.snapshot import build_snapshot

    quotes = {k: v for k, v in provider._quotes.items() if k not in ("EURUSD=X", "EXS1.DE")}
    snap = build_snapshot(watchlist, StaticProvider(quotes, provider._history), now=NOW)
    assert any("Wechselkurs" in e for e in snap["errors"])
    assert any(e.startswith("DAX: kein Kurs") for e in snap["errors"])
    assert snap["quotes"]["AAPL"]["price_eur"] is None


def test_currency_mismatch_is_reported(watchlist, provider):
    from trading_agent.feeds.prices import Quote, StaticProvider
    from trading_agent.snapshot import build_snapshot

    quotes = dict(provider._quotes)
    quotes["SAP.DE"] = Quote("SAP.DE", 270.0, 260.0, "USD", "x")
    snap = build_snapshot(watchlist, StaticProvider(quotes, provider._history), now=NOW)
    assert "SAP" not in snap["quotes"]
    assert any("SAP: Kurs in USD" in e for e in snap["errors"])


def test_quote_time_and_age(snapshot):
    q = snapshot["quotes"]["SAP"]
    assert q["as_of"] == "2026-11-18T10:30:00+00:00"
    assert q["age_minutes"] == 0.0


def test_quote_from_future_is_rejected(watchlist, provider):
    from trading_agent.feeds.prices import Quote, StaticProvider
    from trading_agent.snapshot import build_snapshot

    from .conftest import NOW

    quotes = dict(provider._quotes)
    quotes["SAP.DE"] = Quote("SAP.DE", 251.0, 240.0, "EUR", "2026-11-18T10:45:00+00:00")
    snap = build_snapshot(watchlist, StaticProvider(quotes, provider._history), now=NOW)
    assert "SAP" not in snap["quotes"]
    assert any("liegt nach dem Snapshot" in e for e in snap["errors"])


def test_unknown_quote_time_is_reported(watchlist, provider):
    from trading_agent.feeds.prices import Quote, StaticProvider
    from trading_agent.snapshot import build_snapshot

    from .conftest import NOW

    quotes = dict(provider._quotes)
    quotes["SAP.DE"] = Quote("SAP.DE", 251.0, 240.0, "EUR", "")
    snap = build_snapshot(watchlist, StaticProvider(quotes, provider._history), now=NOW)
    assert snap["quotes"]["SAP"]["age_minutes"] is None
    assert "SAP: Kurszeit unbekannt" in snap["errors"]
