# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from bob.db import connect, initialize_schema, store_settled_events
from bob.kalshi import Bracket, Event, SettledEvent

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
        "SELECT event_ticker, close_ts, expiration_value FROM events"
    ).fetchone()
    assert row == ("KXBTC-99APR0100", int(CLOSE.timestamp()), "420.69")
    brackets = db.execute(
        "SELECT ticker, floor_strike, cap_strike, won FROM brackets ORDER BY ticker"
    ).fetchall()
    assert brackets == [
        ("KXBTC-99APR0100-B420", "400", "499.99", 1),
        ("KXBTC-99APR0100-T100", None, "100", 0),
        ("KXBTC-99APR0100-T999999", "999999", None, 0),
    ]


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
