# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pytest
from typer.testing import CliRunner

from bob.cli import app
from bob.db import connect, initialize_schema, store_btc_candles, store_settled_events
from bob.research.s19 import STRATEGY, evaluate, path_minutes
from helpers import research_flat_bars, research_settled

runner = CliRunner()


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_path_minutes_matched_horizon() -> None:
    assert path_minutes(55) == (50, 51, 52, 53, 54, 55)
    assert path_minutes(50) == (40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50)


def test_s19_trades_on_excursion_and_reclaim(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(1, 51), "150")
    bars.extend(research_flat_bars(range(51, 53), "250"))
    bars.extend(research_flat_bars(range(53, 56), "150"))
    store_btc_candles(db, bars)

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.strategy == STRATEGY == "s19"
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1


def test_s19_abstains_without_intermediate_excursion(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    store_btc_candles(db, research_flat_bars(range(1, 56), "150"))

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions == {"no_excursion": 1}


def test_s19_abstains_when_anchor_is_in_another_bracket(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(1, 51), "250")
    bars.extend(research_flat_bars(range(51, 56), "150"))
    store_btc_candles(db, bars)

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions == {"unconfirmed_bracket": 1}


def test_research_s19_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(1, 51), "150")
    bars.extend(research_flat_bars(range(51, 53), "250"))
    bars.extend(research_flat_bars(range(53, 56), "150"))
    store_btc_candles(connection, bars)
    connection.close()

    result = runner.invoke(
        app,
        ["research", "s19", "--db", str(db_path), "--minutes", "55"],
    )

    assert result.exit_code == 0, result.output
    assert "s19" in result.output
    assert "excursion-reclaim" in result.output
