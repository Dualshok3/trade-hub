# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-11-20

### Added
- LangGraph agent: screen, gather, decide, risk check, finish. Decisions as JSON with a closed schema that both the
  crit engine and Claude accept.
- Risk check in code with feedback to the model; forced hold after two rejections; connection errors counted
  separately.
- Every model call recorded through harness-it-all (replayable), with snapshot, portfolio, week, attempt and
  config hash as tags.
- Journal: the agent's note from the last tick goes into the next prompt.
- Models: Gemma 4 on the local crit engine, Claude Opus 5.5 through the Claude Code CLI, DeepSeek as an extra
  reference.
- Commands `decide` (probe decision on a saved snapshot) and `tick --agents`.

### Changed
- README in English.

## [0.2.0] - 2026-11-18

### Added
- Paper portfolio as an event-sourced SQLite store: orders as % of the portfolio, fractional shares, €1 fee,
  0.1 % half spread, fills only at a newer frozen quote, pending orders that expire after three ticks.
- German flat-rate tax on every sale (26.375 %, FIFO, fees reduce the gain, allowance configurable).
- Five rule-based baseline portfolios: `cash`, `hold-msciw`, `eq-etfs`, `sma-trend`, `random`.
- Commands `tick` and `depots`.

## [0.1.0] - 2026-11-18

### Added
- Quotes for a 15-asset watchlist through yfinance, converted to euros, with the exchange time of each quote.
- Google News RSS: cleaned, deduplicated, assigned to assets, ranked by source, event, subject, recency and
  coverage.
- Snapshots as gzip JSON with a SHA-256 hash and look-ahead guards; read-only tools over a snapshot.
- Commands `snapshot`, `show`, `list` and `news`.

[0.3.0]: https://github.com/Dualshok3/trade-hub/releases/tag/v0.3.0
[0.2.0]: https://github.com/Dualshok3/trade-hub/releases/tag/v0.2.0
[0.1.0]: https://github.com/Dualshok3/trade-hub/releases/tag/v0.1.0
