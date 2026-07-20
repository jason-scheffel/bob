# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bob.cli import app
from bob.db import MinuteBar, connect, initialize_schema, store_btc_candles, store_settled_events
from bob.kalshi import STATUS_COMPLETE, Bracket, Event, SettledEvent
from bob.research.common import checkpoint_end_ts
from bob.research.s20 import STRATEGY, evaluate, trailing_minutes

runner = CliRunner()


def _settled(
    close: datetime,
    *,
    expiration: str,
    winner: str,
    label: str,
) -> SettledEvent:
    event_ticker = f"KXBTC-99JUN01{label}"
    return SettledEvent(
        event=Event(
            event_ticker=event_ticker,
            close_ts=close,
            status=STATUS_COMPLETE,
            expiration_value=Decimal(expiration),
        ),
        brackets=(
            Bracket(
                ticker=f"{event_ticker}-A",
                event_ticker=event_ticker,
                floor_strike=Decimal("100"),
                cap_strike=Decimal("199.99"),
                won=winner == "a",
            ),
            Bracket(
                ticker=f"{event_ticker}-B",
                event_ticker=event_ticker,
                floor_strike=Decimal("200"),
                cap_strike=Decimal("299.99"),
                won=winner == "b",
            ),
            Bracket(
                ticker=f"{event_ticker}-C",
                event_ticker=event_ticker,
                floor_strike=Decimal("300"),
                cap_strike=Decimal("399.99"),
                won=winner == "c",
            ),
        ),
    )


def _hour_bars(close: datetime, price: str) -> list[MinuteBar]:
    return [
        MinuteBar(
            end_ts=checkpoint_end_ts(close, minute)
            if minute < 60
            else int(close.timestamp()),
            open=price,
            high=price,
            low=price,
            close=price,
        )
        for minute in range(1, 61)
    ]


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_trailing_minutes_matched_remaining() -> None:
    assert tuple(trailing_minutes(55)) == (51, 52, 53, 54, 55)
    assert tuple(trailing_minutes(50)) == tuple(range(41, 51))


def test_s20_first_event_abstains_without_analogs(db) -> None:
    close = datetime(2099, 6, 1, 10, 0, tzinfo=timezone.utc)
    store_settled_events(db, [_settled(close, expiration="150", winner="a", label="10")])
    store_btc_candles(db, _hour_bars(close, "150"))

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.strategy == STRATEGY == "s20"
    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions == {"insufficient_analogs": 1}


def test_s20_trades_after_building_analog_library(db) -> None:
    base = datetime(2099, 6, 1, 10, 0, tzinfo=timezone.utc)
    events = []
    bars: list[MinuteBar] = []
    for offset in range(3):
        close = base + timedelta(hours=offset)
        events.append(
            _settled(
                close,
                expiration="150",
                winner="a",
                label=f"{10 + offset:02d}",
            )
        )
        bars.extend(_hour_bars(close, "150"))
    store_settled_events(db, events)
    store_btc_candles(db, bars)

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.minutes[0].eligible == 2
    assert report.minutes[0].wins == 2
    assert report.minutes[0].abstentions.get("insufficient_analogs", 0) == 1


def test_research_s20_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    base = datetime(2099, 6, 1, 10, 0, tzinfo=timezone.utc)
    events = []
    bars: list[MinuteBar] = []
    for offset in range(3):
        close = base + timedelta(hours=offset)
        events.append(
            _settled(
                close,
                expiration="150",
                winner="a",
                label=f"{10 + offset:02d}",
            )
        )
        bars.extend(_hour_bars(close, "150"))
    store_settled_events(connection, events)
    store_btc_candles(connection, bars)
    connection.close()

    result = runner.invoke(
        app,
        ["research", "s20", "--db", str(db_path), "--minutes", "55"],
    )

    assert result.exit_code == 0, result.output
    assert "s20" in result.output
    assert "path-analog" in result.output
