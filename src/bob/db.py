# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from bob.kalshi import SettledEvent

SCHEMA_VERSION = 1

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS events (
    event_ticker TEXT PRIMARY KEY,
    close_ts INTEGER NOT NULL,
    expiration_value TEXT NOT NULL
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
    connection.executescript(_SCHEMA)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.commit()


def store_settled_events(
    connection: sqlite3.Connection,
    events: Iterable[SettledEvent],
) -> StoreCounts:
    settled = tuple(events)
    try:
        for item in settled:
            _upsert_event(connection, item)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return StoreCounts(
        events=len(settled),
        brackets=sum(len(item.brackets) for item in settled),
    )


def _upsert_event(connection: sqlite3.Connection, item: SettledEvent) -> None:
    event = item.event
    if event.close_ts.tzinfo is None:
        raise ValueError(
            f"close_ts must be timezone-aware: {event.close_ts!r}"
        )
    connection.execute(
        """
        INSERT INTO events (event_ticker, close_ts, expiration_value)
        VALUES (?, ?, ?)
        ON CONFLICT(event_ticker) DO UPDATE SET
            close_ts = excluded.close_ts,
            expiration_value = excluded.expiration_value
        """,
        (
            event.event_ticker,
            int(event.close_ts.timestamp()),
            format(event.expiration_value, "f"),
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
