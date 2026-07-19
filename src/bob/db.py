# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bob.kalshi import (
    STATUS_COMPLETE,
    STATUS_MISSING_EXPIRATION,
    STATUS_NO_MARKETS,
    SettledEvent,
)

SCHEMA_VERSION = 2

# Stronger observations may replace weaker ones; never the reverse.
_STATUS_RANK = {
    STATUS_NO_MARKETS: 0,
    STATUS_MISSING_EXPIRATION: 1,
    STATUS_COMPLETE: 2,
}

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
"""


@dataclass(frozen=True, slots=True)
class StoreCounts:
    events: int
    brackets: int


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
    if version == 1:
        _migrate_v1_to_v2(connection)
        return
    connection.executescript(_SCHEMA)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.commit()


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
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.isolation_level = previous_isolation


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
