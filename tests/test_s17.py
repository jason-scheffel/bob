# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bob.cli import app
from bob.db import connect, initialize_schema, store_btc_candles, store_settled_events
from bob.research.s17 import DEFAULT_MAX_MOVE, DEFAULT_MIN_OCCUPANCY, STRATEGY, evaluate
from helpers import research_flat_bars, research_settled

runner = CliRunner()


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_s17_trades_when_sticky_current_and_calm(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    store_btc_candles(db, research_flat_bars(range(1, 56), "150"))

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.strategy == STRATEGY == "s17"
    assert report.min_occupancy == DEFAULT_MIN_OCCUPANCY
    assert report.max_move == DEFAULT_MAX_MOVE
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1


def test_s17_abstains_when_mode_not_current(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    # Most closes in 200s bracket; current at 150
    bars = research_flat_bars(range(1, 40), "250")
    bars.extend(research_flat_bars(range(40, 56), "150"))
    store_btc_candles(db, bars)

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("mode_not_current", 0) == 1


def test_s17_abstains_low_occupancy(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    # Alternate brackets so unique mode exists but share < 0.60
    bars = []
    for minute in range(1, 56):
        price = "150" if minute % 2 == 0 else "250"
        bars.extend(research_flat_bars(range(minute, minute + 1), price))
    store_btc_candles(db, bars)

    report = evaluate(db, minutes=(55,), side="yes", min_occupancy=Decimal("0.60"))

    assert report.minutes[0].eligible == 0
    assert (
        report.minutes[0].abstentions.get("low_occupancy", 0)
        + report.minutes[0].abstentions.get("no_unique_mode", 0)
        + report.minutes[0].abstentions.get("mode_not_current", 0)
        >= 1
    )


def test_research_s17_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
            "s17",
            "--db",
            str(db_path),
            "--minutes",
            "55",
            "--min-occupancy",
            "0.60",
            "--max-move",
            "250",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "s17" in result.output
    assert "min_occupancy=0.60" in result.output
