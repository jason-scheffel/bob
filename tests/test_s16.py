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
from bob.research.s16 import DEFAULT_MAX_MOVE, STRATEGY, evaluate
from helpers import RESEARCH_CLOSE, research_flat_bars, research_settled

runner = CliRunner()
EVENT = "KXBTC-99JUN0108"


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_s16_trades_when_confirmed_and_calm(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    store_btc_candles(db, research_flat_bars(range(1, 56), "150"))

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.strategy == STRATEGY == "s16"
    assert report.max_move == DEFAULT_MAX_MOVE
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1


def test_s16_abstains_unconfirmed(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(1, 2), "150")
    bars.extend(research_flat_bars(range(50, 51), "250"))
    bars.extend(research_flat_bars(range(55, 56), "150"))
    store_btc_candles(db, bars)

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("unconfirmed_bracket", 0) == 1


def test_s16_abstains_large_move(db) -> None:
    store_settled_events(
        db,
        [
            SettledEvent(
                event=Event(
                    event_ticker=EVENT,
                    close_ts=RESEARCH_CLOSE,
                    status=STATUS_COMPLETE,
                    expiration_value=Decimal("350"),
                ),
                brackets=(
                    Bracket(
                        ticker=f"{EVENT}-A",
                        event_ticker=EVENT,
                        floor_strike=Decimal("100"),
                        cap_strike=Decimal("199.99"),
                        won=False,
                    ),
                    Bracket(
                        ticker=f"{EVENT}-B",
                        event_ticker=EVENT,
                        floor_strike=Decimal("300"),
                        cap_strike=Decimal("399.99"),
                        won=True,
                    ),
                ),
            )
        ],
    )
    bars = [
        MinuteBar(
            end_ts=checkpoint_end_ts(RESEARCH_CLOSE, 1),
            open="100",
            high="100",
            low="100",
            close="100",
        ),
        MinuteBar(
            end_ts=checkpoint_end_ts(RESEARCH_CLOSE, 50),
            open="350",
            high="350",
            low="350",
            close="350",
        ),
        MinuteBar(
            end_ts=checkpoint_end_ts(RESEARCH_CLOSE, 55),
            open="350",
            high="350",
            low="350",
            close="350",
        ),
    ]
    store_btc_candles(db, bars)

    report = evaluate(db, minutes=(55,), side="yes", max_move=Decimal("250"))

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("large_move", 0) == 1


def test_research_s16_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [research_settled(expiration="150", winner="a")])
    store_btc_candles(connection, research_flat_bars(range(1, 56), "150"))
    connection.close()

    result = runner.invoke(
        app,
        [
            "research",
            "s16",
            "--db",
            str(db_path),
            "--minutes",
            "55",
            "--max-move",
            "250",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "s16" in result.output
    assert "max_move=250" in result.output
