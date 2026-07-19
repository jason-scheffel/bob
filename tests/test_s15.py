# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bob.cli import app
from bob.db import (
    MinuteBar,
    connect,
    initialize_schema,
    store_btc_candles,
    store_settled_events,
)
from bob.kalshi import STATUS_COMPLETE, Bracket, Event, SettledEvent
from bob.research.common import checkpoint_end_ts
from bob.research.s15 import DEFAULT_DWELL, DEFAULT_MOVE, STRATEGY, evaluate
from helpers import RESEARCH_CLOSE, research_settled

runner = CliRunner()
EVENT = "KXBTC-99JUN0108"


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def _impulse_settled(*, expiration: str, winner_b: bool) -> SettledEvent:
    return SettledEvent(
        event=Event(
            event_ticker=EVENT,
            close_ts=RESEARCH_CLOSE,
            status=STATUS_COMPLETE,
            expiration_value=Decimal(expiration),
        ),
        brackets=(
            Bracket(
                ticker=f"{EVENT}-A",
                event_ticker=EVENT,
                floor_strike=Decimal("100"),
                cap_strike=Decimal("199.99"),
                won=not winner_b,
            ),
            Bracket(
                ticker=f"{EVENT}-B",
                event_ticker=EVENT,
                floor_strike=Decimal("300"),
                cap_strike=Decimal("399.99"),
                won=winner_b,
            ),
        ),
    )


def _seed_impulse_dwell(
    connection,
    close_ts,
    *,
    open_px: str,
    close_px: str,
    dwell_from: int,
) -> None:
    """Open at open_px; closes jump to close_px from dwell_from onward."""
    bars = []
    for minute in range(1, 60):
        close = open_px if minute < dwell_from else close_px
        bars.append(
            MinuteBar(
                end_ts=checkpoint_end_ts(close_ts, minute),
                open=open_px if minute == 1 else close,
                high=close,
                low=close,
                close=close,
            )
        )
    store_btc_candles(connection, bars)


def test_s15_trades_after_impulse_and_dwell(db) -> None:
    store_settled_events(db, [_impulse_settled(expiration="350", winner_b=True)])
    _seed_impulse_dwell(
        db, RESEARCH_CLOSE, open_px="100", close_px="350", dwell_from=42
    )

    report = evaluate(
        db,
        minutes=(45,),
        side="yes",
        move=DEFAULT_MOVE,
        dwell=DEFAULT_DWELL,
    )

    assert report.strategy == STRATEGY == "s15"
    assert report.move == DEFAULT_MOVE
    assert report.dwell == DEFAULT_DWELL
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1


def test_s15_abstains_small_move(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    _seed_impulse_dwell(
        db, RESEARCH_CLOSE, open_px="150", close_px="160", dwell_from=42
    )

    report = evaluate(db, minutes=(45,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("small_move", 0) == 1


def test_s15_abstains_in_transit(db) -> None:
    store_settled_events(db, [_impulse_settled(expiration="350", winner_b=True)])
    _seed_impulse_dwell(
        db, RESEARCH_CLOSE, open_px="100", close_px="350", dwell_from=45
    )

    report = evaluate(db, minutes=(45,), side="yes", dwell=4)

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("in_transit", 0) == 1


def test_research_s15_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(
        connection, [_impulse_settled(expiration="350", winner_b=True)]
    )
    _seed_impulse_dwell(
        connection,
        RESEARCH_CLOSE,
        open_px="100",
        close_px="350",
        dwell_from=42,
    )
    connection.close()

    result = runner.invoke(
        app,
        [
            "research",
            "s15",
            "--db",
            str(db_path),
            "--minutes",
            "45",
            "--move",
            "250",
            "--dwell",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "s15" in result.output
    assert "move=250" in result.output
    assert "dwell=4" in result.output
