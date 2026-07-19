# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared helpers for named research strategies."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from bob.browse import BracketRow, EventRow
from bob.kalshi import STATUS_COMPLETE


def finite_decimal(value: str) -> Decimal | None:
    try:
        amount = Decimal(value)
    except InvalidOperation:
        return None
    if not amount.is_finite():
        return None
    return amount


def price_in_bracket(
    price: Decimal,
    floor_strike: str | None,
    cap_strike: str | None,
) -> bool:
    """Inclusive floor/cap; both-null never matches.

    Malformed or non-finite strikes do not match (never raise).
    """
    if not price.is_finite():
        return False
    if floor_strike is None and cap_strike is None:
        return False
    if floor_strike is None:
        cap = finite_decimal(cap_strike)
        return cap is not None and price <= cap
    if cap_strike is None:
        floor = finite_decimal(floor_strike)
        return floor is not None and price >= floor
    floor = finite_decimal(floor_strike)
    cap = finite_decimal(cap_strike)
    if floor is None or cap is None:
        return False
    return floor <= price <= cap


def brackets_containing(
    price: Decimal,
    brackets: Sequence[BracketRow],
) -> tuple[BracketRow, ...]:
    return tuple(
        bracket
        for bracket in brackets
        if price_in_bracket(price, bracket.floor_strike, bracket.cap_strike)
    )


def bracket_containing(
    price: Decimal,
    brackets: Sequence[BracketRow],
) -> BracketRow | None:
    """Return the sole containing bracket, or None if not exactly one."""
    matches = brackets_containing(price, brackets)
    if len(matches) != 1:
        return None
    return matches[0]


def checkpoint_end_ts(close_ts: datetime | int, minute: int) -> int:
    """Unix ``end_ts`` of the 1m bar at minute ``M`` into the close hour."""
    if not 1 <= minute <= 59:
        raise ValueError(f"minute must be in 1..59, got {minute}")
    if isinstance(close_ts, datetime):
        if close_ts.tzinfo is None:
            raise ValueError("close_ts must be timezone-aware")
        close_unix = int(close_ts.timestamp())
    else:
        close_unix = int(close_ts)
    return close_unix - (60 - minute) * 60


def load_candle_close(
    connection: sqlite3.Connection,
    end_ts: int,
) -> str | None:
    row = connection.execute(
        "SELECT close FROM btc_candles WHERE end_ts = ?",
        (end_ts,),
    ).fetchone()
    if row is None:
        return None
    return row[0]


def load_all_complete_events(
    connection: sqlite3.Connection,
) -> tuple[EventRow, ...]:
    rows = connection.execute(
        """
        SELECT event_ticker, close_ts, expiration_value
        FROM events
        WHERE status = ?
          AND expiration_value IS NOT NULL
        ORDER BY close_ts, event_ticker
        """,
        (STATUS_COMPLETE,),
    ).fetchall()
    return tuple(
        EventRow(
            event_ticker=event_ticker,
            close_ts=datetime.fromtimestamp(int(close_ts), tz=timezone.utc),
            expiration_value=expiration_value,
        )
        for event_ticker, close_ts, expiration_value in rows
    )
