# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backfill YES bid/ask market quote candles into ``market_candles``."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from bob.db import (
    MARKET_HOUR_MINUTES,
    MarketQuoteBar,
    expected_market_minute_ends,
    iter_markets_needing_quotes,
    market_candle_inventory,
    normalize_market_quote_hour,
    store_market_candles,
)
from bob.kalshi import KalshiClient, KalshiParseError


@dataclass(frozen=True, slots=True)
class MarketCandleBackfillCounts:
    fetched: int
    written: int
    skipped: int
    empty: int
    errors: int


def run_backfill_market_candles(
    connection: sqlite3.Connection,
    client: KalshiClient,
    start: datetime,
    end: datetime,
    *,
    force: bool = False,
    cutoff_refresh_every: int = 500,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> MarketCandleBackfillCounts:
    """Fetch and persist 60 quote slots per market ticker in range.

    Progress is ticker/request based. Each successful fetch (including empty /
    dual-404) writes exactly 60 rows so restarts skip completed tickers.
    Parse errors leave the ticker incomplete for retry. 429 exhaustion raises.
    """
    inventory = market_candle_inventory(connection, start, end, force=force)
    targets = iter_markets_needing_quotes(connection, start, end, force=force)
    skipped = inventory.markets - len(targets) if not force else 0
    fetched = 0
    written = 0
    empty = 0
    errors = 0
    total = len(targets)
    cutoff = client.market_settled_cutoff()
    for index, (_event, ticker, close_ts) in enumerate(targets, start=1):
        if cutoff_refresh_every > 0 and index % cutoff_refresh_every == 0:
            cutoff = client.market_settled_cutoff()
        ends = expected_market_minute_ends(close_ts)
        start_ts = ends[0]
        end_ts = ends[-1]
        try:
            bars = client.fetch_market_quote_candles(
                ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                close_ts=close_ts,
                cutoff=cutoff,
            )
        except KalshiParseError:
            errors += 1
            if on_progress is not None:
                on_progress(index, total, ticker)
            continue
        fetched += 1
        hour = normalize_market_quote_hour(ticker, close_ts, bars)
        if len(hour) != MARKET_HOUR_MINUTES:
            raise AssertionError("normalize_market_quote_hour must return 60 bars")
        if all(
            bar.yes_bid_close is None and bar.yes_ask_close is None for bar in hour
        ):
            empty += 1
        written += store_market_candles(connection, hour)
        if on_progress is not None:
            on_progress(index, total, ticker)
    return MarketCandleBackfillCounts(
        fetched=fetched,
        written=written,
        skipped=skipped,
        empty=empty,
        errors=errors,
    )


def sparse_quote_rate(bars: tuple[MarketQuoteBar, ...]) -> float:
    """Fraction of slots with both bid and ask null."""
    if not bars:
        return 0.0
    sparse = sum(
        1
        for bar in bars
        if bar.yes_bid_close is None and bar.yes_ask_close is None
    )
    return sparse / len(bars)
