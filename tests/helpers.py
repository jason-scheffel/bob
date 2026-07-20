# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timezone
from decimal import Decimal

from bob.db import (
    MarketQuoteBar,
    MinuteBar,
    store_btc_candles,
    store_market_candles,
    utc_hour_starts,
)
from bob.kalshi import STATUS_COMPLETE, Bracket, Event, SettledEvent
from bob.research.common import checkpoint_end_ts

RESEARCH_CLOSE = datetime(2099, 6, 1, 12, 0, tzinfo=timezone.utc)


def research_settled(*, expiration: str, winner: str) -> SettledEvent:
    """Three closed $100-ish brackets (A/B/C) for strategy unit tests."""
    event_ticker = "KXBTC-99JUN0108"
    return SettledEvent(
        event=Event(
            event_ticker=event_ticker,
            close_ts=RESEARCH_CLOSE,
            status=STATUS_COMPLETE,
            expiration_value=Decimal(expiration),
        ),
        brackets=(
            Bracket(
                ticker=f"{event_ticker}-A",
                event_ticker=event_ticker,
                floor_strike=Decimal("100"),
                cap_strike=Decimal("199.99"),
                won=winner == "a",
            ),
            Bracket(
                ticker=f"{event_ticker}-B",
                event_ticker=event_ticker,
                floor_strike=Decimal("200"),
                cap_strike=Decimal("299.99"),
                won=winner == "b",
            ),
            Bracket(
                ticker=f"{event_ticker}-C",
                event_ticker=event_ticker,
                floor_strike=Decimal("300"),
                cap_strike=Decimal("399.99"),
                won=winner == "c",
            ),
        ),
    )


def research_flat_bars(minutes: range, price: str) -> list[MinuteBar]:
    return [
        MinuteBar(
            end_ts=checkpoint_end_ts(RESEARCH_CLOSE, minute),
            open=price,
            high=price,
            low=price,
            close=price,
        )
        for minute in minutes
    ]


def seed_research_yes_quote(
    connection,
    *,
    ticker: str,
    minute: int,
    bid: str = "0.40",
    ask: str = "0.45",
    close_ts: datetime | None = None,
) -> None:
    """Seed one YES bid/ask close so quote-sim CLI tests are not excluded."""
    when = RESEARCH_CLOSE if close_ts is None else close_ts
    store_market_candles(
        connection,
        [
            MarketQuoteBar(
                ticker=ticker,
                end_ts=checkpoint_end_ts(when, minute),
                yes_bid_close=bid,
                yes_ask_close=ask,
            )
        ],
    )


def seed_complete_candle_hours(
    connection,
    start: datetime,
    end: datetime,
) -> int:
    """Write full 60-minute coverage for every UTC hour overlapping ``[start, end)``."""
    bars: list[MinuteBar] = []
    for hour_start in utc_hour_starts(start, end):
        start_ts = int(hour_start.astimezone(timezone.utc).timestamp())
        for offset in range(1, 61):
            bars.append(
                MinuteBar(
                    end_ts=start_ts + offset * 60,
                    open="1",
                    high="1",
                    low="1",
                    close="1",
                )
            )
    return store_btc_candles(connection, bars)


def cf_empty_response() -> dict:
    return {
        "_fixture": "SYNTHETIC — empty CF history (not real)",
        "data": {
            "serverTime": "2099-01-01T00:00:00.000Z",
            "payload": [],
        },
    }
