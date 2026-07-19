# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared helpers for named research strategies."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Literal

from bob.browse import BracketRow, EventRow
from bob.kalshi import STATUS_COMPLETE


@dataclass(frozen=True, slots=True)
class CandleOHLC:
    end_ts: int
    open: str
    high: str
    low: str
    close: str


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
    bar = load_candle_ohlc(connection, end_ts)
    if bar is None:
        return None
    return bar.close


def load_candle_ohlc(
    connection: sqlite3.Connection,
    end_ts: int,
) -> CandleOHLC | None:
    row = connection.execute(
        """
        SELECT end_ts, open, high, low, close
        FROM btc_candles
        WHERE end_ts = ?
        """,
        (end_ts,),
    ).fetchone()
    if row is None:
        return None
    end_ts_v, open_v, high_v, low_v, close_v = row
    return CandleOHLC(
        end_ts=int(end_ts_v),
        open=open_v,
        high=high_v,
        low=low_v,
        close=close_v,
    )


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


def load_minute_closes(
    connection: sqlite3.Connection,
    close_ts: datetime,
    minutes: Sequence[int],
) -> list[Decimal] | None:
    """Load closes for ``minutes``; None if any bar is missing/invalid."""
    closes: list[Decimal] = []
    for minute in minutes:
        raw = load_candle_close(connection, checkpoint_end_ts(close_ts, minute))
        if raw is None:
            return None
        value = finite_decimal(raw)
        if value is None:
            return None
        closes.append(value)
    return closes


def ols_slope(values: Sequence[Decimal]) -> Decimal | None:
    """Slope of values vs index 0..n-1; None if fewer than 2 points."""
    n = len(values)
    if n < 2:
        return None
    ys = [float(value) for value in values]
    mean_x = (n - 1) / 2.0
    mean_y = sum(ys) / n
    numerator = sum((index - mean_x) * (y - mean_y) for index, y in enumerate(ys))
    denominator = sum((index - mean_x) ** 2 for index in range(n))
    if denominator == 0:
        return Decimal("0")
    return Decimal(str(numerator / denominator))


def sample_stdev(values: Sequence[Decimal]) -> Decimal | None:
    """Sample standard deviation; None if fewer than 2 points."""
    n = len(values)
    if n < 2:
        return None
    xs = [float(value) for value in values]
    mean = sum(xs) / n
    variance = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return Decimal(str(math.sqrt(variance)))


def _bracket_sort_key(bracket: BracketRow) -> tuple[int, Decimal]:
    if bracket.floor_strike is None and bracket.cap_strike is None:
        return (2, Decimal(0))
    if bracket.floor_strike is None:
        cap = finite_decimal(bracket.cap_strike)
        return (0, cap if cap is not None else Decimal(0))
    floor = finite_decimal(bracket.floor_strike)
    return (1, floor if floor is not None else Decimal(0))


def ordered_brackets(brackets: Sequence[BracketRow]) -> tuple[BracketRow, ...]:
    return tuple(sorted(brackets, key=_bracket_sort_key))


def neighbor_bracket(
    brackets: Sequence[BracketRow],
    selected: BracketRow,
    *,
    direction: Literal["up", "down"],
) -> BracketRow | None:
    ordered = ordered_brackets(brackets)
    index = next(
        (
            position
            for position, bracket in enumerate(ordered)
            if bracket.ticker == selected.ticker
        ),
        None,
    )
    if index is None:
        return None
    neighbor = index + 1 if direction == "up" else index - 1
    if 0 <= neighbor < len(ordered):
        return ordered[neighbor]
    return None


def closed_bracket_width(
    floor_strike: str | None,
    cap_strike: str | None,
) -> Decimal | None:
    if floor_strike is None or cap_strike is None:
        return None
    floor = finite_decimal(floor_strike)
    cap = finite_decimal(cap_strike)
    if floor is None or cap is None:
        return None
    width = cap - floor
    if width <= 0:
        return None
    return width


def distance_to_nearest_edge(
    price: Decimal,
    floor_strike: str | None,
    cap_strike: str | None,
) -> Decimal | None:
    width = closed_bracket_width(floor_strike, cap_strike)
    if width is None:
        return None
    floor = Decimal(floor_strike)  # validated by closed_bracket_width
    cap = Decimal(cap_strike)
    if not (floor <= price <= cap):
        return None
    return min(price - floor, cap - price)
