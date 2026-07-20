# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

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
from bob.research.common import checkpoint_end_ts
from bob.research.s9 import STRATEGY, evaluate, horizon_minutes
from helpers import RESEARCH_CLOSE, research_settled, seed_research_yes_quote

runner = CliRunner()


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def _seed_horizon(connection, *, extreme_high: str = "160") -> None:
    bars = [
        MinuteBar(
            end_ts=checkpoint_end_ts(RESEARCH_CLOSE, minute),
            open="150",
            high=extreme_high if minute == 53 else "160",
            low="140",
            close="150",
        )
        for minute in horizon_minutes(55)
    ]
    store_btc_candles(connection, bars)


def test_horizon_minutes_matches_remaining_time() -> None:
    assert horizon_minutes(45) == range(31, 46)
    assert horizon_minutes(55) == range(51, 56)
    with pytest.raises(ValueError, match="30..59"):
        horizon_minutes(29)


def test_s9_takes_trade_when_excursion_fits(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    _seed_horizon(db)

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.strategy == STRATEGY == "s9"
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1
    assert report.minutes[0].abstained == 0


def test_s9_abstains_when_excursion_exceeds_edge_room(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    _seed_horizon(db, extreme_high="210")

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions == {"thin_excursion_buffer": 1}


def test_research_s9_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [research_settled(expiration="150", winner="a")])
    _seed_horizon(connection)
    seed_research_yes_quote(connection, ticker="KXBTC-99JUN0108-A", minute=55)
    connection.close()

    result = runner.invoke(
        app,
        ["research", "s9", "--db", str(db_path), "--minutes", "55"],
    )

    assert result.exit_code == 0, result.output
    assert "s9" in result.output
    assert "100.0%" in result.output
    assert "abstained" in result.output
