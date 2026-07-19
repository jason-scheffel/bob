# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pytest
from typer.testing import CliRunner

from bob.cli import app
from bob.db import connect, initialize_schema, store_btc_candles, store_settled_events
from bob.research.s18 import STRATEGY, evaluate
from helpers import research_flat_bars, research_settled

runner = CliRunner()


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_s18_trades_on_confirmed_edgeward_move(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(1, 51), "150")
    bars.extend(research_flat_bars(range(51, 56), "125"))
    store_btc_candles(db, bars)

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.strategy == STRATEGY == "s18"
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1


def test_s18_abstains_when_anchor_is_in_another_bracket(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(1, 51), "250")
    bars.extend(research_flat_bars(range(51, 56), "150"))
    store_btc_candles(db, bars)

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions == {"unconfirmed_bracket": 1}


def test_s18_abstains_when_move_is_not_edgeward(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(1, 51), "125")
    bars.extend(research_flat_bars(range(51, 56), "150"))
    store_btc_candles(db, bars)

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions == {"not_edgeward": 1}


def test_research_s18_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(1, 51), "150")
    bars.extend(research_flat_bars(range(51, 56), "125"))
    store_btc_candles(connection, bars)
    connection.close()

    result = runner.invoke(
        app,
        ["research", "s18", "--db", str(db_path), "--minutes", "55"],
    )

    assert result.exit_code == 0, result.output
    assert "s18" in result.output
    assert "matched-horizon edgeward" in result.output
