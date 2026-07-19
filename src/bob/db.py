# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bob.kalshi import (
    STATUS_COMPLETE,
    STATUS_MISSING_EXPIRATION,
    STATUS_NO_MARKETS,
    SettledEvent,
)

SCHEMA_VERSION = 3
CANDLE_LAG_S = 15 * 60

# Stronger observations may replace weaker ones; never the reverse.
_STATUS_RANK = {
    STATUS_NO_MARKETS: 0,
    STATUS_MISSING_EXPIRATION: 1,
    STATUS_COMPLETE: 2,
}

_BTC_CANDLES_DDL = """
CREATE TABLE IF NOT EXISTS btc_candles (
    end_ts INTEGER PRIMARY KEY CHECK (end_ts % 60 = 0),
    open TEXT NOT NULL,
    high TEXT NOT NULL,
    low TEXT NOT NULL,
    close TEXT NOT NULL
);
"""

_SCHEMA = f"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS events (
    event_ticker TEXT PRIMARY KEY,
    close_ts INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            '{STATUS_COMPLETE}',
            '{STATUS_MISSING_EXPIRATION}',
            '{STATUS_NO_MARKETS}'
        )
    ),
    expiration_value TEXT,
    CHECK (
        (status = '{STATUS_COMPLETE}' AND expiration_value IS NOT NULL)
        OR
        (status != '{STATUS_COMPLETE}' AND expiration_value IS NULL)
    )
);

CREATE TABLE IF NOT EXISTS brackets (
    ticker TEXT PRIMARY KEY,
    event_ticker TEXT NOT NULL
        REFERENCES events(event_ticker) ON DELETE CASCADE,
    floor_strike TEXT,
    cap_strike TEXT,
    won INTEGER NOT NULL CHECK (won IN (0, 1))
);

CREATE INDEX IF NOT EXISTS brackets_event_idx ON brackets(event_ticker);
CREATE INDEX IF NOT EXISTS events_close_idx ON events(close_ts);

{_BTC_CANDLES_DDL}
"""


@dataclass(frozen=True, slots=True)
class StoreCounts:
    events: int
    brackets: int
    candles: int = 0


@dataclass(frozen=True, slots=True)
class MinuteBar:
    """One UTC minute bar covering ``[end_ts - 60, end_ts)``."""

    end_ts: int
    open: str
    high: str
    low: str
    close: str


def connect(path: Path | str) -> sqlite3.Connection:
    if path == ":memory:":
        connection = sqlite3.connect(":memory:")
    else:
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema version {version} is newer than "
            f"supported {SCHEMA_VERSION}"
        )
    if version == SCHEMA_VERSION:
        return
    if version == 0:
        connection.executescript(_SCHEMA)
        connection.execute("PRAGMA user_version = 3")
        connection.commit()
        return
    if version == 1:
        _migrate_v1_to_v2(connection)
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == 2:
        _migrate_v2_to_v3(connection)


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    previous_isolation = connection.isolation_level
    connection.isolation_level = None
    connection.execute("PRAGMA foreign_keys = OFF")
    try:
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                f"""
                CREATE TABLE events_v2 (
                    event_ticker TEXT PRIMARY KEY,
                    close_ts INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN (
                            '{STATUS_COMPLETE}',
                            '{STATUS_MISSING_EXPIRATION}',
                            '{STATUS_NO_MARKETS}'
                        )
                    ),
                    expiration_value TEXT,
                    CHECK (
                        (status = '{STATUS_COMPLETE}'
                            AND expiration_value IS NOT NULL)
                        OR
                        (status != '{STATUS_COMPLETE}'
                            AND expiration_value IS NULL)
                    )
                )
                """
            )
            connection.execute(
                f"""
                INSERT INTO events_v2 (
                    event_ticker, close_ts, status, expiration_value
                )
                SELECT
                    event_ticker,
                    close_ts,
                    '{STATUS_COMPLETE}',
                    expiration_value
                FROM events
                """
            )
            connection.execute(
                """
                CREATE TABLE brackets_v2 (
                    ticker TEXT PRIMARY KEY,
                    event_ticker TEXT NOT NULL
                        REFERENCES events_v2(event_ticker) ON DELETE CASCADE,
                    floor_strike TEXT,
                    cap_strike TEXT,
                    won INTEGER NOT NULL CHECK (won IN (0, 1))
                )
                """
            )
            connection.execute(
                """
                INSERT INTO brackets_v2 (
                    ticker, event_ticker, floor_strike, cap_strike, won
                )
                SELECT ticker, event_ticker, floor_strike, cap_strike, won
                FROM brackets
                """
            )
            connection.execute("DROP TABLE brackets")
            connection.execute("DROP TABLE events")
            connection.execute("ALTER TABLE events_v2 RENAME TO events")
            connection.execute("ALTER TABLE brackets_v2 RENAME TO brackets")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS brackets_event_idx "
                "ON brackets(event_ticker)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS events_close_idx ON events(close_ts)"
            )
            bad = connection.execute("PRAGMA foreign_key_check").fetchall()
            if bad:
                raise RuntimeError(
                    f"foreign key check failed after migrate: {bad}"
                )
            connection.execute("PRAGMA user_version = 2")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.isolation_level = previous_isolation


def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
    previous_isolation = connection.isolation_level
    connection.isolation_level = None
    try:
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS btc_candles (
                    end_ts INTEGER PRIMARY KEY CHECK (end_ts % 60 = 0),
                    open TEXT NOT NULL,
                    high TEXT NOT NULL,
                    low TEXT NOT NULL,
                    close TEXT NOT NULL
                )
                """
            )
            connection.execute("PRAGMA user_version = 3")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    finally:
        connection.isolation_level = previous_isolation


def candle_in_event_window(end_ts: int, close_ts: int) -> bool:
    """True when bar ``end_ts`` belongs to the hour ending at ``close_ts``.

    Predicate: ``end_ts > close_ts - 3600 AND end_ts <= close_ts``.
    """
    return close_ts - 3600 < end_ts <= close_ts


def hour_is_provisional(
    hour_end: datetime,
    *,
    now: datetime | None = None,
    lag_s: int = CANDLE_LAG_S,
) -> bool:
    """Hours whose end is newer than ``now - lag`` are not final yet."""
    if hour_end.tzinfo is None:
        raise ValueError(f"hour_end must be timezone-aware: {hour_end!r}")
    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError(f"now must be timezone-aware: {current!r}")
    return current.timestamp() < hour_end.timestamp() + lag_s


def expected_minute_ends(
    hour_start: datetime,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[int]:
    """Minute ``end_ts`` values needed for ``hour_start``, clipped to ``[start, end)``.

    When ``start``/``end`` are omitted, all 60 ends for the UTC hour are returned.
    """
    if hour_start.tzinfo is None:
        raise ValueError(f"hour_start must be timezone-aware: {hour_start!r}")
    hour_ts = int(hour_start.astimezone(timezone.utc).timestamp())
    hour_end_ts = hour_ts + 3600
    lo = hour_ts if start is None else int(start.astimezone(timezone.utc).timestamp())
    hi = (
        hour_end_ts
        if end is None
        else int(end.astimezone(timezone.utc).timestamp())
    )
    lo = max(lo, hour_ts)
    hi = min(hi, hour_end_ts)
    if lo >= hi:
        return []
    return [ts for ts in range(hour_ts + 60, hour_end_ts + 1, 60) if lo < ts <= hi]


def hour_has_complete_minutes(
    connection: sqlite3.Connection,
    hour_start: datetime,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> bool:
    """True when every expected minute bar for the (clipped) hour exists."""
    expected = expected_minute_ends(hour_start, start=start, end=end)
    if not expected:
        return True
    placeholders = ",".join("?" * len(expected))
    count = connection.execute(
        f"""
        SELECT COUNT(*) FROM btc_candles
        WHERE end_ts IN ({placeholders})
        """,
        expected,
    ).fetchone()[0]
    return count == len(expected)


def utc_hour_starts(start: datetime, end: datetime) -> list[datetime]:
    """UTC hour starts that overlap half-open ``[start, end)``."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    if start_utc >= end_utc:
        return []
    cursor = start_utc.replace(minute=0, second=0, microsecond=0)
    hours: list[datetime] = []
    while cursor < end_utc:
        hours.append(cursor)
        cursor += timedelta(hours=1)
    return hours


def hours_needing_candles(
    connection: sqlite3.Connection,
    start: datetime,
    end: datetime,
    *,
    force: bool = False,
    now: datetime | None = None,
) -> list[datetime]:
    """UTC hour starts to (re)fetch for candle coverage of ``[start, end)``."""
    current = now if now is not None else datetime.now(timezone.utc)
    needed: list[datetime] = []
    for hour_start in utc_hour_starts(start, end):
        hour_end = hour_start + timedelta(hours=1)
        if force or hour_is_provisional(hour_end, now=current):
            needed.append(hour_start)
            continue
        if not hour_has_complete_minutes(
            connection, hour_start, start=start, end=end
        ):
            needed.append(hour_start)
    return needed


def store_btc_candles(
    connection: sqlite3.Connection,
    bars: Iterable[MinuteBar],
) -> int:
    """Upsert minute bars by ``end_ts``. Returns number of rows written."""
    rows = tuple(bars)
    if not rows:
        return 0
    try:
        connection.executemany(
            """
            INSERT INTO btc_candles (end_ts, open, high, low, close)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(end_ts) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close
            """,
            [
                (bar.end_ts, bar.open, bar.high, bar.low, bar.close)
                for bar in rows
            ],
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return len(rows)


def existing_event_tickers(
    connection: sqlite3.Connection,
    tickers: Iterable[str],
) -> set[str]:
    """Return the subset of ``tickers`` already present in ``events``."""
    wanted = tuple(dict.fromkeys(tickers))
    if not wanted:
        return set()
    placeholders = ",".join("?" * len(wanted))
    rows = connection.execute(
        f"SELECT event_ticker FROM events WHERE event_ticker IN ({placeholders})",
        wanted,
    ).fetchall()
    return {row[0] for row in rows}


def event_tickers_in_close_range(
    connection: sqlite3.Connection,
    start: datetime,
    end: datetime,
) -> set[str]:
    """Return event tickers with ``close_ts`` in half-open ``[start, end)``."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    rows = connection.execute(
        """
        SELECT event_ticker FROM events
        WHERE close_ts >= ? AND close_ts < ?
        """,
        (int(start.timestamp()), int(end.timestamp())),
    ).fetchall()
    return {row[0] for row in rows}


def store_settled_events(
    connection: sqlite3.Connection,
    events: Iterable[SettledEvent],
) -> StoreCounts:
    settled = tuple(events)
    written_events = 0
    written_brackets = 0
    try:
        for item in settled:
            if _upsert_event(connection, item):
                written_events += 1
                written_brackets += len(item.brackets)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return StoreCounts(events=written_events, brackets=written_brackets)


def _upsert_event(connection: sqlite3.Connection, item: SettledEvent) -> bool:
    """Write ``item`` unless a stronger status already exists. Return wrote?"""
    event = item.event
    if event.close_ts.tzinfo is None:
        raise ValueError(
            f"close_ts must be timezone-aware: {event.close_ts!r}"
        )
    if event.status not in _STATUS_RANK:
        raise ValueError(f"unknown event status: {event.status!r}")
    existing = connection.execute(
        "SELECT status FROM events WHERE event_ticker = ?",
        (event.event_ticker,),
    ).fetchone()
    if (
        existing is not None
        and _STATUS_RANK[event.status] < _STATUS_RANK[existing[0]]
    ):
        return False
    if event.status == STATUS_COMPLETE:
        if event.expiration_value is None:
            raise ValueError(
                f"complete event requires expiration_value: "
                f"{event.event_ticker!r}"
            )
        expiration_value: str | None = format(event.expiration_value, "f")
    else:
        if event.expiration_value is not None:
            raise ValueError(
                f"{event.status} event must not have expiration_value: "
                f"{event.event_ticker!r}"
            )
        expiration_value = None
    connection.execute(
        """
        INSERT INTO events (event_ticker, close_ts, status, expiration_value)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(event_ticker) DO UPDATE SET
            close_ts = excluded.close_ts,
            status = excluded.status,
            expiration_value = excluded.expiration_value
        """,
        (
            event.event_ticker,
            int(event.close_ts.timestamp()),
            event.status,
            expiration_value,
        ),
    )
    connection.execute(
        "DELETE FROM brackets WHERE event_ticker = ?",
        (event.event_ticker,),
    )
    connection.executemany(
        """
        INSERT INTO brackets (
            ticker, event_ticker, floor_strike, cap_strike, won
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                bracket.ticker,
                bracket.event_ticker,
                None
                if bracket.floor_strike is None
                else format(bracket.floor_strike, "f"),
                None
                if bracket.cap_strike is None
                else format(bracket.cap_strike, "f"),
                1 if bracket.won else 0,
            )
            for bracket in item.brackets
        ],
    )
    return True
