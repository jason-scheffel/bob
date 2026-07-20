# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bob.browse import load_brackets
from bob.cli import app
from bob.db import connect, initialize_schema, store_btc_candles, store_settled_events
from bob.research.common import brackets_containing
from bob.research.s23 import (
    STRATEGY,
    contiguous_bracket_age,
    evaluate,
    sojourn_drawdown,
)
from helpers import research_flat_bars, research_settled

runner = CliRunner()


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_contiguous_age_and_sojourn_drawdown(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    brackets = load_brackets(db, "KXBTC-99JUN0108")
    selected = brackets_containing(Decimal("150"), brackets)[0]
    closes = [Decimal("160")] * 10 + [Decimal("145")] * 5
    assert contiguous_bracket_age(closes, brackets, selected) == 15
    assert sojourn_drawdown(
        closes,
        age=15,
        floor=Decimal("100"),
        cap=Decimal("199.99"),
    ) == (Decimal("160") - Decimal("145")) / (Decimal("199.99") - Decimal("100"))


def test_s23_trades_when_aged_unfinished_pullback(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(1, 31), "160")
    bars.extend(research_flat_bars(range(31, 46), "145"))
    store_btc_candles(db, bars)

    report = evaluate(db, minutes=(45,), side="yes")

    assert report.strategy == STRATEGY == "s23"
    assert report.age_min == 15
    assert report.dd_min == Decimal("0.10")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1


def test_s23_abstains_young_age(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(1, 36), "250")
    bars.extend(research_flat_bars(range(36, 46), "145"))
    store_btc_candles(db, bars)

    report = evaluate(db, minutes=(45,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("young_age", 0) == 1


def test_s23_abstains_finished_pullback(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    store_btc_candles(db, research_flat_bars(range(1, 46), "150"))

    report = evaluate(db, minutes=(45,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("finished_pullback", 0) == 1


def test_research_s23_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(1, 31), "160")
    bars.extend(research_flat_bars(range(31, 46), "145"))
    store_btc_candles(connection, bars)
    connection.close()

    result = runner.invoke(
        app,
        ["research", "s23", "--db", str(db_path), "--minutes", "45"],
    )

    assert result.exit_code == 0, result.output
    assert "s23" in result.output
    assert "aged unfinished sojourn-drawdown" in result.output
