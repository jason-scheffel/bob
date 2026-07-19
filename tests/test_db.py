# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from bob.db import (
    connect,
    event_tickers_in_close_range,
    existing_event_tickers,
    initialize_schema,
    store_settled_events,
)
from bob.kalshi import (
    STATUS_COMPLETE,
    STATUS_MISSING_EXPIRATION,
    STATUS_NO_MARKETS,
    Bracket,
    Event,
    SettledEvent,
    no_markets_event,
)

CLOSE = datetime(2099, 4, 1, 0, 0, tzinfo=timezone.utc)


def _settled(
    *,
    event_ticker: str = "KXBTC-99APR0100",
    expiration: str = "420.69",
    brackets: tuple[Bracket, ...] | None = None,
) -> SettledEvent:
    if brackets is None:
        brackets = (
            Bracket(
                ticker=f"{event_ticker}-T100",
                event_ticker=event_ticker,
                floor_strike=None,
                cap_strike=Decimal("100"),
                won=False,
            ),
            Bracket(
                ticker=f"{event_ticker}-B420",
                event_ticker=event_ticker,
                floor_strike=Decimal("400"),
                cap_strike=Decimal("499.99"),
                won=True,
            ),
            Bracket(
                ticker=f"{event_ticker}-T999999",
                event_ticker=event_ticker,
                floor_strike=Decimal("999999"),
                cap_strike=None,
                won=False,
            ),
        )
    return SettledEvent(
        event=Event(
            event_ticker=event_ticker,
            close_ts=CLOSE,
            status=STATUS_COMPLETE,
            expiration_value=Decimal(expiration),
        ),
        brackets=brackets,
    )


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_store_rejects_naive_close_ts(db) -> None:
    naive = SettledEvent(
        event=Event(
            event_ticker="KXBTC-99APR0100",
            close_ts=datetime(2099, 4, 1, 0, 0),
            status=STATUS_COMPLETE,
            expiration_value=Decimal("420.69"),
        ),
        brackets=(
            Bracket(
                ticker="KXBTC-99APR0100-B420",
                event_ticker="KXBTC-99APR0100",
                floor_strike=Decimal("400"),
                cap_strike=Decimal("499.99"),
                won=True,
            ),
        ),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        store_settled_events(db, [naive])


def test_store_round_trip(db) -> None:
    counts = store_settled_events(db, [_settled()])
    assert counts.events == 1
    assert counts.brackets == 3
    row = db.execute(
        "SELECT event_ticker, close_ts, status, expiration_value FROM events"
    ).fetchone()
    assert row == (
        "KXBTC-99APR0100",
        int(CLOSE.timestamp()),
        STATUS_COMPLETE,
        "420.69",
    )
    brackets = db.execute(
        "SELECT ticker, floor_strike, cap_strike, won FROM brackets ORDER BY ticker"
    ).fetchall()
    assert brackets == [
        ("KXBTC-99APR0100-B420", "400", "499.99", 1),
        ("KXBTC-99APR0100-T100", None, "100", 0),
        ("KXBTC-99APR0100-T999999", "999999", None, 0),
    ]


def test_existing_event_tickers(db) -> None:
    store_settled_events(db, [_settled()])
    assert existing_event_tickers(db, []) == set()
    assert existing_event_tickers(
        db, ["KXBTC-99APR0100", "KXBTC-MISSING", "KXBTC-99APR0100"]
    ) == {"KXBTC-99APR0100"}
    assert existing_event_tickers(db, ["KXBTC-MISSING"]) == set()


def test_event_tickers_in_close_range(db) -> None:
    store_settled_events(db, [_settled()])
    assert event_tickers_in_close_range(
        db, CLOSE, datetime(2099, 4, 2, tzinfo=timezone.utc)
    ) == {"KXBTC-99APR0100"}
    assert (
        event_tickers_in_close_range(
            db,
            datetime(2099, 4, 2, tzinfo=timezone.utc),
            datetime(2099, 4, 3, tzinfo=timezone.utc),
        )
        == set()
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        event_tickers_in_close_range(
            db, datetime(2099, 4, 1), datetime(2099, 4, 2, tzinfo=timezone.utc)
        )


def test_store_rerun_idempotent(db) -> None:
    store_settled_events(db, [_settled()])
    store_settled_events(db, [_settled()])
    assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM brackets").fetchone()[0] == 3


def test_store_updates_parent_and_replaces_brackets(db) -> None:
    store_settled_events(db, [_settled()])
    updated = _settled(
        expiration="421.00",
        brackets=(
            Bracket(
                ticker="KXBTC-99APR0100-B421",
                event_ticker="KXBTC-99APR0100",
                floor_strike=Decimal("421"),
                cap_strike=Decimal("421.99"),
                won=True,
            ),
        ),
    )
    store_settled_events(db, [updated])
    assert db.execute(
        "SELECT expiration_value FROM events WHERE event_ticker = ?",
        ("KXBTC-99APR0100",),
    ).fetchone() == ("421.00",)
    tickers = {
        row[0]
        for row in db.execute("SELECT ticker FROM brackets").fetchall()
    }
    assert tickers == {"KXBTC-99APR0100-B421"}


def test_store_missing_expiration_and_no_markets(db) -> None:
    missing = SettledEvent(
        event=Event(
            event_ticker="KXBTC-99APR0101",
            close_ts=datetime(2099, 4, 1, 1, tzinfo=timezone.utc),
            status=STATUS_MISSING_EXPIRATION,
            expiration_value=None,
        ),
        brackets=(
            Bracket(
                ticker="KXBTC-99APR0101-B420",
                event_ticker="KXBTC-99APR0101",
                floor_strike=Decimal("400"),
                cap_strike=Decimal("499.99"),
                won=True,
            ),
        ),
    )
    empty = no_markets_event(
        "KXBTC-99APR0102",
        datetime(2099, 4, 1, 2, tzinfo=timezone.utc),
    )
    counts = store_settled_events(db, [missing, empty])
    assert counts.events == 2
    assert counts.brackets == 1
    rows = {
        row[0]: (row[1], row[2])
        for row in db.execute(
            "SELECT event_ticker, status, expiration_value FROM events"
        )
    }
    assert rows["KXBTC-99APR0101"] == (STATUS_MISSING_EXPIRATION, None)
    assert rows["KXBTC-99APR0102"] == (STATUS_NO_MARKETS, None)
    assert db.execute(
        "SELECT COUNT(*) FROM brackets WHERE event_ticker = ?",
        ("KXBTC-99APR0101",),
    ).fetchone() == (1,)
    assert db.execute(
        "SELECT COUNT(*) FROM brackets WHERE event_ticker = ?",
        ("KXBTC-99APR0102",),
    ).fetchone() == (0,)


def _seed_v1_schema(connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE events (
            event_ticker TEXT PRIMARY KEY,
            close_ts INTEGER NOT NULL,
            expiration_value TEXT NOT NULL
        );
        CREATE TABLE brackets (
            ticker TEXT PRIMARY KEY,
            event_ticker TEXT NOT NULL
                REFERENCES events(event_ticker) ON DELETE CASCADE,
            floor_strike TEXT,
            cap_strike TEXT,
            won INTEGER NOT NULL CHECK (won IN (0, 1))
        );
        """
    )
    connection.execute(
        """
        INSERT INTO events (event_ticker, close_ts, expiration_value)
        VALUES (?, ?, ?)
        """,
        ("KXBTC-99APR0100", int(CLOSE.timestamp()), "420.69"),
    )
    connection.execute(
        """
        INSERT INTO brackets (
            ticker, event_ticker, floor_strike, cap_strike, won
        ) VALUES (?, ?, ?, ?, ?)
        """,
        ("KXBTC-99APR0100-B420", "KXBTC-99APR0100", "400", "499.99", 1),
    )
    connection.execute("PRAGMA user_version = 1")
    connection.commit()


def test_migrate_v1_to_v2() -> None:
    connection = connect(":memory:")
    _seed_v1_schema(connection)

    initialize_schema(connection)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    assert connection.execute(
        "SELECT status, expiration_value FROM events WHERE event_ticker = ?",
        ("KXBTC-99APR0100",),
    ).fetchone() == (STATUS_COMPLETE, "420.69")
    assert connection.execute("SELECT COUNT(*) FROM brackets").fetchone()[0] == 1
    connection.close()


def test_migrate_v1_to_v2_rolls_back_on_fk_failure() -> None:
    connection = connect(":memory:")
    _seed_v1_schema(connection)
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute(
        """
        INSERT INTO brackets (
            ticker, event_ticker, floor_strike, cap_strike, won
        ) VALUES ('orphan', 'MISSING', NULL, NULL, 0)
        """
    )
    connection.execute("PRAGMA foreign_keys = ON")
    connection.commit()

    with pytest.raises(RuntimeError, match="foreign key check failed"):
        initialize_schema(connection)

    assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    cols = {
        row[1] for row in connection.execute("PRAGMA table_info(events)")
    }
    assert cols == {"event_ticker", "close_ts", "expiration_value"}
    assert connection.execute(
        "SELECT expiration_value FROM events WHERE event_ticker = ?",
        ("KXBTC-99APR0100",),
    ).fetchone() == ("420.69",)
    connection.close()


def test_store_does_not_downgrade_complete(db) -> None:
    store_settled_events(db, [_settled()])
    weaker = no_markets_event("KXBTC-99APR0100", CLOSE)
    counts = store_settled_events(db, [weaker])
    assert counts.events == 0
    assert counts.brackets == 0
    assert db.execute(
        "SELECT status, expiration_value FROM events WHERE event_ticker = ?",
        ("KXBTC-99APR0100",),
    ).fetchone() == (STATUS_COMPLETE, "420.69")
    assert db.execute("SELECT COUNT(*) FROM brackets").fetchone()[0] == 3


def test_store_upgrades_no_markets_to_complete(db) -> None:
    store_settled_events(db, [no_markets_event("KXBTC-99APR0100", CLOSE)])
    counts = store_settled_events(db, [_settled()])
    assert counts.events == 1
    assert counts.brackets == 3
    assert db.execute(
        "SELECT status, expiration_value FROM events WHERE event_ticker = ?",
        ("KXBTC-99APR0100",),
    ).fetchone() == (STATUS_COMPLETE, "420.69")


def test_store_upgrades_no_markets_to_missing_expiration(db) -> None:
    store_settled_events(db, [no_markets_event("KXBTC-99APR0100", CLOSE)])
    missing = SettledEvent(
        event=Event(
            event_ticker="KXBTC-99APR0100",
            close_ts=CLOSE,
            status=STATUS_MISSING_EXPIRATION,
            expiration_value=None,
        ),
        brackets=(
            Bracket(
                ticker="KXBTC-99APR0100-B420",
                event_ticker="KXBTC-99APR0100",
                floor_strike=Decimal("400"),
                cap_strike=Decimal("499.99"),
                won=True,
            ),
        ),
    )
    counts = store_settled_events(db, [missing])
    assert counts.events == 1
    assert counts.brackets == 1
    assert db.execute(
        "SELECT status, expiration_value FROM events WHERE event_ticker = ?",
        ("KXBTC-99APR0100",),
    ).fetchone() == (STATUS_MISSING_EXPIRATION, None)


def test_store_rolls_back_on_failure(db) -> None:
    store_settled_events(db, [_settled()])
    bad = _settled(
        brackets=(
            Bracket(
                ticker="KXBTC-99APR0100-B420",
                event_ticker="KXBTC-99APR0100",
                floor_strike=Decimal("400"),
                cap_strike=Decimal("499.99"),
                won=True,
            ),
            Bracket(
                ticker="KXBTC-99APR0100-B420",
                event_ticker="KXBTC-99APR0100",
                floor_strike=Decimal("401"),
                cap_strike=Decimal("401.99"),
                won=False,
            ),
        )
    )
    with pytest.raises(Exception):
        store_settled_events(db, [bad])
    assert db.execute(
        "SELECT expiration_value FROM events WHERE event_ticker = ?",
        ("KXBTC-99APR0100",),
    ).fetchone() == ("420.69",)
    assert db.execute("SELECT COUNT(*) FROM brackets").fetchone()[0] == 3


def test_foreign_keys_and_cascade(db) -> None:
    store_settled_events(db, [_settled()])
    with pytest.raises(Exception):
        db.execute(
            """
            INSERT INTO brackets (ticker, event_ticker, won)
            VALUES ('orphan', 'MISSING', 0)
            """
        )
        db.commit()
    db.rollback()
    db.execute("DELETE FROM events WHERE event_ticker = ?", ("KXBTC-99APR0100",))
    db.commit()
    assert db.execute("SELECT COUNT(*) FROM brackets").fetchone()[0] == 0
