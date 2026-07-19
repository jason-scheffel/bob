# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class EventRow:
    event_ticker: str
    close_ts: datetime
    expiration_value: str


@dataclass(frozen=True, slots=True)
class BracketRow:
    ticker: str
    floor_strike: str | None
    cap_strike: str | None
    won: bool


def format_close_et(close_ts: datetime) -> str:
    """Human close time in America/New_York (KXBTC hour labels)."""
    local = close_ts.astimezone(_ET)
    hour = local.strftime("%I:%M %p").lstrip("0")
    return f"{local.strftime('%b')} {local.day}, {local.year} {hour} ET"


def format_btc(value: str | Decimal) -> str:
    amount = value if isinstance(value, Decimal) else Decimal(value)
    quantized = format(amount, "f")
    if "." in quantized:
        whole, frac = quantized.split(".", 1)
        frac = frac.rstrip("0")
        body = f"{int(whole):,}" + (f".{frac}" if frac else "")
    else:
        body = f"{int(quantized):,}"
    return f"${body}"


def format_bracket_range(
    floor_strike: str | None,
    cap_strike: str | None,
) -> str:
    if floor_strike is None and cap_strike is None:
        return "—"
    if floor_strike is None:
        return f"below {format_btc(cap_strike)}"
    if cap_strike is None:
        return f"{format_btc(floor_strike)} and above"
    return f"{format_btc(floor_strike)} – {format_btc(cap_strike)}"


def format_event_label(event: EventRow) -> str:
    return f"{format_close_et(event.close_ts)} · BTC {format_btc(event.expiration_value)}"


def winning_bracket(brackets: tuple[BracketRow, ...]) -> BracketRow | None:
    winners = [bracket for bracket in brackets if bracket.won]
    if len(winners) != 1:
        return None
    return winners[0]


def _strike_sort_key(value: str | None) -> tuple[int, Decimal]:
    if value is None:
        return (0, Decimal(0))
    try:
        return (1, Decimal(value))
    except InvalidOperation:
        return (2, Decimal(0))


def load_events(
    connection: sqlite3.Connection,
    start: datetime,
    end: datetime,
) -> tuple[EventRow, ...]:
    """Return events with ``close_ts`` in half-open ``[start, end)``."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    start_unix = int(start.timestamp())
    end_unix = int(end.timestamp())
    rows = connection.execute(
        """
        SELECT event_ticker, close_ts, expiration_value
        FROM events
        WHERE close_ts >= ? AND close_ts < ?
        ORDER BY close_ts, event_ticker
        """,
        (start_unix, end_unix),
    ).fetchall()
    return tuple(
        EventRow(
            event_ticker=event_ticker,
            close_ts=datetime.fromtimestamp(int(close_ts), tz=timezone.utc),
            expiration_value=expiration_value,
        )
        for event_ticker, close_ts, expiration_value in rows
    )


def load_brackets(
    connection: sqlite3.Connection,
    event_ticker: str,
) -> tuple[BracketRow, ...]:
    rows = connection.execute(
        """
        SELECT ticker, floor_strike, cap_strike, won
        FROM brackets
        WHERE event_ticker = ?
        """,
        (event_ticker,),
    ).fetchall()
    brackets = [
        BracketRow(
            ticker=ticker,
            floor_strike=floor_strike,
            cap_strike=cap_strike,
            won=bool(won),
        )
        for ticker, floor_strike, cap_strike, won in rows
    ]
    brackets.sort(key=lambda item: _strike_sort_key(item.floor_strike))
    return tuple(brackets)
